"""Expert C — supply/demand & liquidity (a small, declared SMC subset).

Method: detect base-before-impulse supply/demand zones and liquidity sweeps of
prior confirmed swings (definitions in ``indicators.py``). If price is reacting
at a fresh/tested demand zone after a sell-side sweep (mirror for supply), it
proposes a pending plan using the zone's own boundaries; otherwise it waits.
Zones are reported with their status: new / tested / consumed / invalidated.
This is NOT a claim to implement all of SMC — only this subset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from gold_data.contracts import Direction, MarketState, PriceLevel
from gold_data.contracts import Zone as CZone
from gold_data.snapshot import MarketSnapshot

from . import indicators as ind
from .base import BaseExpert, ComputeOutput, MethodSettings, PlanProposal, plan_feasible


@dataclass
class SupplyDemandSettings(MethodSettings):
    impulse_atr_mult: float = 1.5
    atr_period: int = 14
    lookback: int = 300
    stop_buffer_frac: float = 0.1     # buffer beyond the zone, as fraction of zone height


class SupplyDemandExpert(BaseExpert):
    expert_id = "expert_supply_demand"
    specialty = "العرض والطلب والسيولة"
    method_version = "sd-0.1"
    context_tfs = ("H1", "H4")
    trigger_tfs = ("M15", "M5")

    def __init__(self, settings: Optional[SupplyDemandSettings] = None):
        super().__init__(settings or SupplyDemandSettings())

    def _compute(self, snapshot: MarketSnapshot, frames, reference_time) -> ComputeOutput:
        s: SupplyDemandSettings = self.settings  # type: ignore
        ctx_tf = next((tf for tf in self.context_tfs if tf in frames), None)
        ctx = frames[ctx_tf]
        last_close = float(ctx["close"].iloc[-1])

        izones = ind.supply_demand_zones(ctx, s.impulse_atr_mult, s.atr_period, s.lookback)
        swings = ind.swing_points(ctx, 2, 2)
        sweeps = ind.liquidity_sweeps(ctx, swings)

        zones: list[CZone] = []
        for z in izones[-20:]:
            zones.append(CZone(kind=z.kind, low=round(z.low, 2), high=round(z.high, 2),
                               source=f"{self.expert_id}:{ctx_tf}",
                               invalidation=round(z.invalidation, 2),
                               note=f"formed={z.formed_at};confirmed={z.confirmed_at};status={z.status}"))

        demand = [z for z in izones if z.kind == "demand" and z.status in ("new", "tested")]
        supply = [z for z in izones if z.kind == "supply" and z.status in ("new", "tested")]
        # nearest actionable zones around price
        below = sorted([z for z in demand if z.high <= last_close + (z.high - z.low)],
                       key=lambda z: abs(last_close - z.high))
        above = sorted([z for z in supply if z.low >= last_close - (z.high - z.low)],
                       key=lambda z: abs(z.low - last_close))
        recent_sweep = sweeps[-1] if sweeps else None

        evidence = [f"فريم {ctx_tf}: مناطق مكتشفة={len(izones)} (طلب نشط={len(demand)}, عرض نشط={len(supply)})",
                    f"سحوبات سيولة={len(sweeps)}"]
        points = [f"عدد مناطق العرض/الطلب النشطة: طلب {len(demand)} / عرض {len(supply)}"]
        if recent_sweep:
            points.append(f"آخر سحب سيولة: {recent_sweep.kind} عند {recent_sweep.swept_level} ({recent_sweep.time})")
            evidence.append(f"آخر سحب: {recent_sweep.kind}@{recent_sweep.swept_level}")

        direction = Direction.NEUTRAL
        market_state = MarketState.RANGE
        proposal = None
        wait_reason = None
        missing = []
        if not izones:
            wait_reason = "لم تُكتشف مناطق عرض/طلب مؤكدة ضمن النطاق المتاح"
            market_state = MarketState.UNCLEAR
        else:
            in_demand = next((z for z in demand if z.low <= last_close <= z.high + (z.high - z.low)), None)
            in_supply = next((z for z in supply if z.low - (z.high - z.low) <= last_close <= z.high), None)
            if in_demand and (recent_sweep is None or recent_sweep.kind == "sell_side"):
                direction = Direction.LONG; market_state = MarketState.TREND_UP
                height = max(in_demand.high - in_demand.low, 1e-6)
                entry_low, entry_high = in_demand.low, in_demand.high
                stop = in_demand.invalidation - s.stop_buffer_frac * height
                tgt = above[0].low if above else entry_high + 2 * height
                targets = [round(tgt, 2)]
                points.append("السعر يتفاعل مع منطقة طلب نشطة" + (" بعد سحب سيولة بيعية" if recent_sweep else ""))
                if plan_feasible(Direction.LONG, entry_low, entry_high, stop, targets):
                    proposal = PlanProposal(Direction.LONG, round(entry_low, 2), round(entry_high, 2),
                                            round(stop, 2), targets,
                                            f"ارتداد مؤكد من منطقة الطلب {round(entry_low,2)}-{round(entry_high,2)}",
                                            f"{ctx_tf}-demand-zone")
                else:
                    wait_reason = "هندسة خطة الطلب غير صالحة (هدف غير متوفر فوق المنطقة)"
            elif in_supply and (recent_sweep is None or recent_sweep.kind == "buy_side"):
                direction = Direction.SHORT; market_state = MarketState.TREND_DOWN
                height = max(in_supply.high - in_supply.low, 1e-6)
                entry_low, entry_high = in_supply.low, in_supply.high
                stop = in_supply.invalidation + s.stop_buffer_frac * height
                tgt = below[0].high if below else entry_low - 2 * height
                targets = [round(tgt, 2)]
                points.append("السعر يتفاعل مع منطقة عرض نشطة" + (" بعد سحب سيولة شرائية" if recent_sweep else ""))
                if plan_feasible(Direction.SHORT, entry_low, entry_high, stop, targets):
                    proposal = PlanProposal(Direction.SHORT, round(entry_low, 2), round(entry_high, 2),
                                            round(stop, 2), targets,
                                            f"ارتداد مؤكد من منطقة العرض {round(entry_low,2)}-{round(entry_high,2)}",
                                            f"{ctx_tf}-supply-zone")
                else:
                    wait_reason = "هندسة خطة العرض غير صالحة (هدف غير متوفر تحت المنطقة)"
            else:
                wait_reason = "السعر ليس عند منطقة عرض/طلب نشطة الآن"

        screen = {"zones": [vars(z) for z in izones[-20:]],
                  "sweeps": [vars(x) for x in sweeps[-20:]],
                  "last_close": last_close, "context_tf": ctx_tf}
        allowed = [last_close] + [z.low for z in izones] + [z.high for z in izones] + \
                  [x.swept_level for x in sweeps]
        return ComputeOutput(direction=direction, market_state=market_state, evidence=evidence,
                             zones=zones, analysis_points=points, proposal=proposal,
                             wait_reason=wait_reason, missing=missing, screen=screen,
                             allowed_numbers=[round(x, 2) for x in allowed])
