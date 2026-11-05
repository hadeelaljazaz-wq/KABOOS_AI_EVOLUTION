"""Expert base: the shared pipeline, kept separate from any one school.

A concrete expert implements only ``_compute`` (pure arithmetic on the
snapshot's closed bars). The base turns that into the project's existing
contracts (``ExpertReport`` + optional ``TradePlan``), attaches an Arabic
narrative (deterministic by rule, or via the real LLM when available and
validated so it cannot change numbers), records the expert's own memory, and
emits a structured screen bundle for a future per-expert UI.

Independence is structural: an expert receives only the shared snapshot, the
shared cycle id, and ITS OWN memory. It never sees another expert's report.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from gold_data.contracts import (
    Direction,
    ExpertReport,
    MarketState,
    PlanStatus,
    PriceLevel,
    Recommendation,
    TradePlan,
    Zone as CZone,
)
from gold_data.snapshot import MarketSnapshot

from .narrative import narrate


@dataclass
class MethodSettings:
    """Base for an expert's tunable settings. ``proven`` stays False: these are
    INITIAL method settings whose effectiveness has NOT been established."""
    proven: bool = False


@dataclass
class PlanProposal:
    """A concrete, computed trade idea (all prices from calculations/zones)."""
    direction: Direction
    entry_low: float
    entry_high: float
    stop: float
    targets: list[float]
    activation_condition: str
    price_source: str
    expiry: Optional[str] = None


def plan_feasible(direction: Direction, entry_low: float, entry_high: float,
                  stop: float, targets: list[float]) -> bool:
    """Pre-check a proposed plan's geometry so we never emit an invalid plan.

    Mirrors TradePlan's ordering rules; if False, the expert should WAIT rather
    than publish a structurally broken plan.
    """
    if entry_low > entry_high or not targets:
        return False
    if any(x <= 0 for x in [entry_low, entry_high, stop, *targets]):
        return False
    if direction == Direction.LONG:
        return stop < entry_low and all(t > entry_high for t in targets) and targets == sorted(targets)
    if direction == Direction.SHORT:
        return stop > entry_high and all(t < entry_low for t in targets) and targets == sorted(targets, reverse=True)
    return False


@dataclass
class ComputeOutput:
    direction: Direction
    market_state: MarketState
    evidence: list[str]                      # computational references (candles/indicators)
    zones: list[CZone] = field(default_factory=list)
    levels: list[PriceLevel] = field(default_factory=list)
    analysis_points: list[str] = field(default_factory=list)  # Arabic facts for the narrator
    proposal: Optional[PlanProposal] = None
    wait_reason: Optional[str] = None
    missing: list[str] = field(default_factory=list)
    screen: dict[str, Any] = field(default_factory=dict)
    allowed_numbers: list[float] = field(default_factory=list)  # numbers the narrative may cite


@dataclass
class ExpertResult:
    expert_id: str
    specialty: str
    method_version: str
    cycle_id: str
    snapshot_id: str
    timeframes_used: list[str]
    report: ExpertReport
    plan: Optional[TradePlan]
    analysis_ar: str
    run_mode: str                 # 'deterministic' | 'llm' | 'deterministic (llm requested, no key)'
    success: bool
    abstained: bool
    abstain_reason: Optional[str]
    missing: list[str]
    screen: dict[str, Any]

    def summary_ar(self) -> str:
        r = self.report
        rec = r.recommendation.value if r.recommendation else "—"
        head = (f"[{self.specialty}] {self.expert_id} (منهج {self.method_version}) — "
                f"الاتجاه: {r.direction.value} | حالة السوق: {r.market_state.value} | التوصية: {rec}")
        plan = ""
        if self.plan and self.plan.entry is not None:
            p = self.plan
            plan = (f"\n   خطة {p.direction.value} [{p.status.value}]: "
                    f"دخول {p.entry.low}-{p.entry.high} | وقف {p.stop_loss.value} | "
                    f"أهداف {[t.value for t in p.targets]} | تفعيل: {p.activation_condition}")
        elif r.recommendation == Recommendation.WAIT:
            plan = f"\n   انتظار: {r.wait_reason}"
        miss = f"\n   ناقص: {', '.join(self.missing)}" if self.missing else ""
        return f"{head}\n   {self.analysis_ar}{plan}{miss}\n   (وضع التشغيل: {self.run_mode})"


class BaseExpert:
    expert_id: str = "expert_base"
    specialty: str = "base"
    method_version: str = "0.0"
    context_tfs: tuple[str, ...] = ("H4", "H1")
    trigger_tfs: tuple[str, ...] = ("M15", "M5")

    def __init__(self, settings: Optional[MethodSettings] = None):
        self.settings = settings or MethodSettings()

    # subclasses implement this (pure arithmetic only)
    def _compute(self, snapshot: MarketSnapshot, frames: dict[str, pd.DataFrame],
                 reference_time: Optional[str]) -> ComputeOutput:
        raise NotImplementedError

    def _pick(self, snapshot: MarketSnapshot, prefs: tuple[str, ...]) -> Optional[str]:
        for tf in prefs:
            d = snapshot.timeframes.get(tf)
            if d is not None and d.bar_count > 0:
                return tf
        return None

    def analyze(self, snapshot: MarketSnapshot, cycle_id: str,
                reference_time: Optional[str] = None, memory=None,
                llm_backend=None, run_mode: str = "deterministic") -> ExpertResult:
        symbol = snapshot.symbol_canonical
        context_tf = self._pick(snapshot, self.context_tfs)
        trigger_tf = self._pick(snapshot, self.trigger_tfs)
        used = [tf for tf in (context_tf, trigger_tf) if tf]
        frames = {tf: snapshot.timeframes[tf].frame for tf in used}

        # recall this expert's own prior memory BEFORE analysing (never others')
        recalled = []
        if memory is not None:
            try:
                recalled = [dict(r) for r in memory.recall(self.expert_id,
                                                            category=f"analysis:{symbol}", limit=5)]
            except Exception:
                recalled = []

        # abstain cleanly if it has no usable timeframe at all
        if not frames:
            return self._abstain(snapshot, cycle_id, used,
                                 "لا يوجد أي فريم صالح في اللقطة لهذا الخبير", run_mode)

        comp = self._compute(snapshot, frames, reference_time)
        comp.screen.setdefault("recalled_memory_count", len(recalled))

        plan = None
        recommendation = Recommendation.WAIT
        if comp.proposal is not None:
            pr = comp.proposal
            plan = TradePlan(
                owner=self.expert_id, owner_kind="expert", cycle_id=cycle_id,
                snapshot_id=snapshot.snapshot_id, direction=pr.direction,
                entry=CZone("demand" if pr.direction == Direction.LONG else "supply",
                            pr.entry_low, pr.entry_high, f"{self.expert_id}:{pr.price_source}"),
                stop_loss=PriceLevel(pr.stop, f"{self.expert_id}:{pr.price_source}", "stop"),
                targets=[PriceLevel(t, f"{self.expert_id}:{pr.price_source}", "target") for t in pr.targets],
                activation_condition=pr.activation_condition, expiry=pr.expiry,
                status=PlanStatus.PENDING)
            recommendation = Recommendation.BUY if pr.direction == Direction.LONG else Recommendation.SELL

        wait_reason = None
        if recommendation == Recommendation.WAIT:
            wait_reason = comp.wait_reason or "لا تتوفر شروط منهج هذا الخبير لإصدار خطة الآن"

        report = ExpertReport(
            expert_id=self.expert_id, cycle_id=cycle_id, snapshot_id=snapshot.snapshot_id,
            specialty=self.specialty, method_version=self.method_version,
            direction=comp.direction, market_state=comp.market_state,
            timeframes_used=used, evidence=comp.evidence, zones=comp.zones, levels=comp.levels,
            recommendation=recommendation, wait_reason=wait_reason,
            trade_plan_id=plan.plan_id if plan else None)

        analysis_ar, used_mode = narrate(self, comp, report, run_mode, llm_backend)

        # record the expert's own memory (analysis != confirmed knowledge)
        if memory is not None:
            try:
                memory.remember(
                    self.expert_id, category=f"analysis:{symbol}",
                    content=json.dumps({"cycle_id": cycle_id, "snapshot_id": snapshot.snapshot_id,
                                        "direction": comp.direction.value,
                                        "market_state": comp.market_state.value,
                                        "recommendation": recommendation.value}),
                    evidence={"data_source": snapshot.instrument_kind.value,
                              "source": f"snapshot:{snapshot.snapshot_id}",
                              "metrics": {"timeframes": used}})
            except Exception:
                pass

        screen = self._build_screen(snapshot, used, comp, report, plan, analysis_ar, used_mode)
        return ExpertResult(
            expert_id=self.expert_id, specialty=self.specialty, method_version=self.method_version,
            cycle_id=cycle_id, snapshot_id=snapshot.snapshot_id, timeframes_used=used,
            report=report, plan=plan, analysis_ar=analysis_ar, run_mode=used_mode,
            success=True, abstained=False, abstain_reason=None,
            missing=comp.missing, screen=screen)

    def _abstain(self, snapshot, cycle_id, used, reason, run_mode) -> ExpertResult:
        report = ExpertReport(
            expert_id=self.expert_id, cycle_id=cycle_id, snapshot_id=snapshot.snapshot_id,
            specialty=self.specialty, method_version=self.method_version,
            direction=Direction.NEUTRAL, market_state=MarketState.UNCLEAR,
            timeframes_used=used, evidence=[], recommendation=Recommendation.WAIT,
            wait_reason=reason)
        return ExpertResult(
            expert_id=self.expert_id, specialty=self.specialty, method_version=self.method_version,
            cycle_id=cycle_id, snapshot_id=snapshot.snapshot_id, timeframes_used=used,
            report=report, plan=None, analysis_ar=reason, run_mode=run_mode,
            success=False, abstained=True, abstain_reason=reason, missing=["usable_timeframe"],
            screen={"abstained": True, "reason": reason})

    def _build_screen(self, snapshot, used, comp: ComputeOutput, report, plan, analysis_ar, mode) -> dict:
        screen = {
            "expert_id": self.expert_id, "specialty": self.specialty,
            "method_version": self.method_version, "cycle_id": report.cycle_id,
            "snapshot_id": snapshot.snapshot_id, "symbol": snapshot.symbol_canonical,
            "instrument_kind": snapshot.instrument_kind.value,
            "timezone": snapshot.timezone or "UNKNOWN", "tz_known": snapshot.tz_known,
            "analysis_reference_time": snapshot.analysis_reference_time,
            "timeframes_used": used,
            "zones": [vars(z) | {"kind": z.kind.value} for z in comp.zones],
            "levels": [vars(l) for l in comp.levels],
            "direction": report.direction.value, "market_state": report.market_state.value,
            "recommendation": report.recommendation.value if report.recommendation else None,
            "plan_status": plan.status.value if plan else None,
            "report_text_ar": analysis_ar, "run_mode": mode,
            "warnings": list(snapshot.warnings)[:20], "missing": comp.missing,
        }
        screen.update(comp.screen)
        return screen
