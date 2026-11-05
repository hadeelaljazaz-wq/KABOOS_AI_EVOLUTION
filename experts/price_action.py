"""Expert A — price structure (market structure / BOS).

Method: read CONFIRMED swing highs/lows (fractal, with right-side confirmation
so a pivot is never claimed before its confirmation bars exist), classify the
trend from the higher-timeframe structure, and look for a pullback to a
structural level on the trigger timeframe. Prices come only from swing levels
and an ATR buffer (ATR is a declared shared helper, used here just for the
stop distance).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from gold_data.contracts import Direction, MarketState, PriceLevel
from gold_data.snapshot import MarketSnapshot

from . import indicators as ind
from .base import BaseExpert, ComputeOutput, MethodSettings, PlanProposal, plan_feasible


@dataclass
class PriceStructureSettings(MethodSettings):
    swing_left: int = 2
    swing_right: int = 2
    atr_period: int = 14
    stop_atr_mult: float = 1.0
    pullback_atr: float = 1.5     # how close to the level price must be to call it a pullback


class PriceStructureExpert(BaseExpert):
    expert_id = "expert_price_structure"
    specialty = "بنية السعر"
    method_version = "pa-0.1"
    context_tfs = ("H4", "H1")
    trigger_tfs = ("M15", "M5")

    def __init__(self, settings: Optional[PriceStructureSettings] = None):
        super().__init__(settings or PriceStructureSettings())

    def _compute(self, snapshot: MarketSnapshot, frames, reference_time) -> ComputeOutput:
        s: PriceStructureSettings = self.settings  # type: ignore
        ctx_tf = next((tf for tf in self.context_tfs if tf in frames), None)
        trg_tf = next((tf for tf in self.trigger_tfs if tf in frames), None)
        ctx = frames[ctx_tf]
        trg = frames.get(trg_tf, ctx)
        missing = []
        if "H4" not in frames and "D1" not in snapshot.timeframes:
            missing.append("سياق فريم أعلى (H4/D1 غير متاح؛ استُخدم المتاح)")

        swings = ind.swing_points(ctx, s.swing_left, s.swing_right)
        struct = ind.classify_structure(swings)
        atr_ctx = ind.last_value(ind.atr(ctx, s.atr_period))
        last_close = float(ctx["close"].iloc[-1])

        evidence = [f"فريم السياق {ctx_tf}: عدد القمم/القيعان المؤكدة = {len(swings)}"]
        levels: list[PriceLevel] = []
        points = []
        if struct["last_high"]:
            levels.append(PriceLevel(struct["last_high"]["price"], f"{self.expert_id}:{ctx_tf}", "resistance"))
            points.append(f"آخر قمة مؤكدة عند {struct['last_high']['price']} ({struct['last_high']['time']})")
            evidence.append(f"آخر قمة مؤكدة: {struct['last_high']['price']} @ {struct['last_high']['time']}")
        if struct["last_low"]:
            levels.append(PriceLevel(struct["last_low"]["price"], f"{self.expert_id}:{ctx_tf}", "support"))
            points.append(f"آخر قاع مؤكد عند {struct['last_low']['price']} ({struct['last_low']['time']})")
            evidence.append(f"آخر قاع مؤكد: {struct['last_low']['price']} @ {struct['last_low']['time']}")

        trend = struct["trend"]
        direction = {"up": Direction.LONG, "down": Direction.SHORT}.get(trend, Direction.NEUTRAL)
        market_state = {"up": MarketState.TREND_UP, "down": MarketState.TREND_DOWN}.get(trend, MarketState.RANGE)
        points.append(f"هيكل {ctx_tf}: {'تصاعدي' if trend=='up' else 'تنازلي' if trend=='down' else 'عرضي'}")

        allowed = [last_close] + [l.value for l in levels] + ([atr_ctx] if atr_ctx else [])
        screen = {"swings": [vars(x) for x in swings[-12:]], "structure": struct,
                  "atr_context": atr_ctx, "last_close": last_close, "context_tf": ctx_tf, "trigger_tf": trg_tf}

        proposal = None
        wait_reason = None
        if trend == "range" or not (struct["last_high"] and struct["last_low"]):
            wait_reason = "لا يوجد هيكل اتجاهي مؤكد (سوق عرضي أو قمم/قيعان غير كافية)"
        elif atr_ctx is None:
            wait_reason = "ATR غير متاح بعد (فترة تهيئة غير مكتملة)"
            missing.append("ATR(context)")
        else:
            if trend == "up":
                support = struct["last_low"]["price"]; resist = struct["last_high"]["price"]
                near = support <= last_close <= support + s.pullback_atr * atr_ctx
                if not near:
                    wait_reason = "الاتجاه صاعد لكن السعر لم يرتد إلى منطقة الدعم البنيوية بعد"
                else:
                    entry_low, entry_high = support, min(last_close, support + 0.5 * s.pullback_atr * atr_ctx)
                    stop = support - s.stop_atr_mult * atr_ctx
                    risk = entry_high - stop
                    targets = [resist, resist + (resist - support)]
                    if plan_feasible(Direction.LONG, entry_low, entry_high, stop, targets):
                        proposal = PlanProposal(Direction.LONG, round(entry_low, 2), round(entry_high, 2),
                                                round(stop, 2), [round(t, 2) for t in targets],
                                                f"إغلاق {trg_tf} فوق {round(entry_high,2)} يؤكد الارتداد من الدعم",
                                                f"{ctx_tf}-structure+ATR")
                    else:
                        wait_reason = "هندسة الخطة غير صالحة (ترتيب الدخول/الوقف/الأهداف)"
            else:  # down
                support = struct["last_low"]["price"]; resist = struct["last_high"]["price"]
                near = resist - s.pullback_atr * atr_ctx <= last_close <= resist
                if not near:
                    wait_reason = "الاتجاه هابط لكن السعر لم يرتد إلى منطقة المقاومة البنيوية بعد"
                else:
                    entry_high, entry_low = resist, max(last_close, resist - 0.5 * s.pullback_atr * atr_ctx)
                    stop = resist + s.stop_atr_mult * atr_ctx
                    targets = [support, support - (resist - support)]
                    if plan_feasible(Direction.SHORT, entry_low, entry_high, stop, targets):
                        proposal = PlanProposal(Direction.SHORT, round(entry_low, 2), round(entry_high, 2),
                                                round(stop, 2), [round(t, 2) for t in targets],
                                                f"إغلاق {trg_tf} تحت {round(entry_low,2)} يؤكد الارتداد من المقاومة",
                                                f"{ctx_tf}-structure+ATR")
                    else:
                        wait_reason = "هندسة الخطة غير صالحة (ترتيب الدخول/الوقف/الأهداف)"

        return ComputeOutput(direction=direction, market_state=market_state, evidence=evidence,
                             levels=levels, analysis_points=points, proposal=proposal,
                             wait_reason=wait_reason, missing=missing, screen=screen,
                             allowed_numbers=[round(x, 2) for x in allowed if x is not None])
