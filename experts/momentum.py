"""Expert B — momentum & trend.

Method: EMA(fast/slow), MACD and RSI on the context timeframe. Direction from
EMA alignment + MACD sign; strength from the histogram and RSI; divergence from
the last two confirmed swings vs RSI. An indicator that cannot be computed yet
(warm-up) is reported MISSING — never replaced by a neutral number. Plan
prices are built from the last close and an ATR-scaled risk (declared helper).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from gold_data.contracts import Direction, MarketState
from gold_data.snapshot import MarketSnapshot

from . import indicators as ind
from .base import BaseExpert, ComputeOutput, MethodSettings, PlanProposal, plan_feasible


@dataclass
class MomentumSettings(MethodSettings):
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    ema_fast: int = 50
    ema_slow: int = 200
    atr_period: int = 14
    stop_atr_mult: float = 1.5
    rr: tuple[float, ...] = (1.0, 2.0)
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0


class MomentumTrendExpert(BaseExpert):
    expert_id = "expert_momentum_trend"
    specialty = "الزخم والاتجاه"
    method_version = "mom-0.1"
    context_tfs = ("H1", "H4")
    trigger_tfs = ("M15", "M5")

    def __init__(self, settings: Optional[MomentumSettings] = None):
        super().__init__(settings or MomentumSettings())

    def _compute(self, snapshot: MarketSnapshot, frames, reference_time) -> ComputeOutput:
        s: MomentumSettings = self.settings  # type: ignore
        ctx_tf = next((tf for tf in self.context_tfs if tf in frames), None)
        trg_tf = next((tf for tf in self.trigger_tfs if tf in frames), None)
        ctx = frames[ctx_tf]
        close = ctx["close"]
        missing = []

        rsi_v = ind.last_value(ind.rsi(close, s.rsi_period))
        macd_line, signal_line, hist = ind.macd(close, s.macd_fast, s.macd_slow, s.macd_signal)
        macd_v, sig_v, hist_v = ind.last_value(macd_line), ind.last_value(signal_line), ind.last_value(hist)
        ema_f = ind.last_value(ind.ema(close, s.ema_fast))
        ema_s = ind.last_value(ind.ema(close, s.ema_slow))
        atr_v = ind.last_value(ind.atr(ctx, s.atr_period))
        last_close = float(close.iloc[-1])

        for name, val in [("RSI", rsi_v), ("MACD", macd_v), ("EMA_fast", ema_f),
                          ("EMA_slow", ema_s), ("ATR", atr_v)]:
            if val is None:
                missing.append(f"{name} (فترة تهيئة غير مكتملة على {ctx_tf})")

        evidence, points = [f"فريم {ctx_tf}"], []
        if rsi_v is not None:
            evidence.append(f"RSI={round(rsi_v,2)}"); points.append(f"RSI = {round(rsi_v,2)}")
        if macd_v is not None and sig_v is not None:
            evidence.append(f"MACD={round(macd_v,4)} / إشارة={round(sig_v,4)} / هيستوغرام={round(hist_v,4)}")
            points.append(f"MACD {'فوق' if macd_v>sig_v else 'تحت'} خط الإشارة")
        if ema_f is not None and ema_s is not None:
            evidence.append(f"EMA{s.ema_fast}={round(ema_f,2)} / EMA{s.ema_slow}={round(ema_s,2)}")
            points.append(f"المتوسط السريع {'فوق' if ema_f>ema_s else 'تحت'} البطيء")

        # direction from EMA alignment (if available) + MACD sign
        direction = Direction.NEUTRAL
        aligned = None
        if ema_f is not None and ema_s is not None:
            aligned = Direction.LONG if ema_f > ema_s else Direction.SHORT
        macd_dir = None
        if macd_v is not None and sig_v is not None:
            macd_dir = Direction.LONG if macd_v > sig_v else Direction.SHORT
        if aligned and macd_dir and aligned == macd_dir:
            direction = aligned
            market_state = MarketState.TREND_UP if direction == Direction.LONG else MarketState.TREND_DOWN
        elif aligned and macd_dir and aligned != macd_dir:
            direction = aligned
            market_state = MarketState.VOLATILE
            points.append("تعارض بين اتجاه المتوسطات وإشارة الزخم (تذبذب)")
        elif macd_dir:
            direction = macd_dir
            market_state = MarketState.TREND_UP if direction == Direction.LONG else MarketState.TREND_DOWN
        else:
            market_state = MarketState.UNCLEAR

        # divergence from last two confirmed swings vs RSI at those times
        swings = ind.swing_points(ctx, 2, 2)
        rsi_series = ind.rsi(close, s.rsi_period)
        diverg = _divergence(ctx, swings, rsi_series)
        if diverg:
            points.append(diverg)
            evidence.append(f"دايفرجنس: {diverg}")

        allowed = [x for x in [last_close, rsi_v, macd_v, sig_v, hist_v, ema_f, ema_s, atr_v] if x is not None]
        screen = {"rsi": rsi_v, "macd": macd_v, "signal": sig_v, "hist": hist_v,
                  "ema_fast": ema_f, "ema_slow": ema_s, "atr": atr_v, "last_close": last_close,
                  "context_tf": ctx_tf, "divergence": diverg}

        proposal = None
        wait_reason = None
        if direction == Direction.NEUTRAL or market_state in (MarketState.UNCLEAR, MarketState.VOLATILE):
            wait_reason = "الزخم غير حاسم أو متعارض؛ لا إشارة اتجاهية نظيفة"
        elif atr_v is None or rsi_v is None:
            wait_reason = "مؤشرات أساسية غير متاحة بعد (تهيئة)"
        elif direction == Direction.LONG and rsi_v >= s.rsi_overbought:
            wait_reason = f"الاتجاه صاعد لكن RSI في منطقة تشبع شرائي ({round(rsi_v,1)})"
        elif direction == Direction.SHORT and rsi_v <= s.rsi_oversold:
            wait_reason = f"الاتجاه هابط لكن RSI في منطقة تشبع بيعي ({round(rsi_v,1)})"
        else:
            if direction == Direction.LONG:
                entry_low, entry_high = last_close - 0.25 * atr_v, last_close
                stop = entry_low - s.stop_atr_mult * atr_v
                risk = entry_high - stop
                targets = [entry_high + rr * risk for rr in s.rr]
                ok = plan_feasible(Direction.LONG, entry_low, entry_high, stop, targets)
                act = f"استمرار الزخم الصاعد مع إغلاق فوق {round(entry_high,2)}"
            else:
                entry_high, entry_low = last_close + 0.25 * atr_v, last_close
                stop = entry_high + s.stop_atr_mult * atr_v
                risk = stop - entry_low
                targets = [entry_low - rr * risk for rr in s.rr]
                ok = plan_feasible(Direction.SHORT, entry_low, entry_high, stop, targets)
                act = f"استمرار الزخم الهابط مع إغلاق تحت {round(entry_low,2)}"
            if ok:
                proposal = PlanProposal(direction, round(entry_low, 2), round(entry_high, 2),
                                        round(stop, 2), [round(t, 2) for t in targets], act,
                                        f"{ctx_tf}-momentum+ATR")
            else:
                wait_reason = "هندسة الخطة غير صالحة بعد حساب المخاطرة"

        return ComputeOutput(direction=direction, market_state=market_state, evidence=evidence,
                             analysis_points=points, proposal=proposal, wait_reason=wait_reason,
                             missing=missing, screen=screen,
                             allowed_numbers=[round(x, 2) for x in allowed])


def _divergence(ctx, swings, rsi_series) -> Optional[str]:
    highs = [s for s in swings if s.kind == "high"]
    lows = [s for s in swings if s.kind == "low"]
    tmap = {str(t): i for i, t in enumerate(ctx.index)}

    def rsi_at(ts):
        i = tmap.get(ts)
        if i is None:
            return None
        v = rsi_series.iloc[i]
        return None if v != v else float(v)  # NaN check

    if len(highs) >= 2:
        p1, p2 = highs[-2], highs[-1]
        r1, r2 = rsi_at(p1.time), rsi_at(p2.time)
        if r1 is not None and r2 is not None and p2.price > p1.price and r2 < r1:
            return "دايفرجنس هابط (قمة أعلى في السعر وأدنى في RSI)"
    if len(lows) >= 2:
        p1, p2 = lows[-2], lows[-1]
        r1, r2 = rsi_at(p1.time), rsi_at(p2.time)
        if r1 is not None and r2 is not None and p2.price < p1.price and r2 > r1:
            return "دايفرجنس صاعد (قاع أدنى في السعر وأعلى في RSI)"
    return None
