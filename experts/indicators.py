"""Pure, documented, look-ahead-safe calculations for the gold expert desk.

Everything here is deterministic arithmetic on a single timeframe's CLOSED
bars (OHLCV DataFrame indexed by bar-open time). No LLM, no fabricated values:
a value that cannot be computed yet (warm-up) stays NaN / is reported missing,
never replaced by a "neutral" number.

Look-ahead safety: a swing pivot, a zone, or a sweep is only ever reported as
CONFIRMED when the bars that confirm it already exist in the given frame. The
snapshot layer has already removed any bar not closed at the analysis
reference time, so "exists in the frame" == "known at the reference time".
A pivot in the final ``right`` bars is therefore NOT yet confirmed and is
excluded — this is the rule that stops a top/bottom being claimed before its
confirmation bars have printed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# -- moving averages / oscillators -----------------------------------------


def sma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window).mean()


def ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI. First ``period`` values are NaN (warm-up, not fabricated)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    out[(avg_loss == 0) & (avg_gain > 0)] = 100.0
    return out


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Return (macd_line, signal_line, histogram); warm-up stays NaN."""
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return macd_line, signal_line, macd_line - signal_line


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (Wilder). A SHARED helper (declared, not hidden).

    Used by any expert only to size a distance (e.g. a stop buffer); it does
    not merge the experts' schools — each decides whether/how to use it.
    """
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(),
                    (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def last_value(series: pd.Series) -> Optional[float]:
    """Last non-NaN value as float, or None if unavailable (warm-up/empty)."""
    s = series.dropna()
    if s.empty:
        return None
    return float(s.iloc[-1])


# -- swing structure --------------------------------------------------------


@dataclass
class Swing:
    time: str
    price: float
    kind: str            # 'high' | 'low'
    confirmed_at: str    # time of the bar that completed confirmation


def swing_points(df: pd.DataFrame, left: int = 2, right: int = 2) -> list[Swing]:
    """Confirmed fractal swing highs/lows.

    A bar i is a swing HIGH if its high is strictly greater than the ``left``
    highs before it and >= the ``right`` highs after it; symmetric for lows.
    Only pivots with all ``right`` confirmation bars present in the frame are
    returned, so the last ``right`` bars never host a confirmed swing.
    """
    highs = df["high"].to_numpy(dtype="float64")
    lows = df["low"].to_numpy(dtype="float64")
    times = [str(t) for t in df.index]
    n = len(df)
    out: list[Swing] = []
    for i in range(left, n - right):
        hw = highs[i - left:i + right + 1]
        lw = lows[i - left:i + right + 1]
        if np.isnan(hw).any() or np.isnan(lw).any():
            continue
        center = left
        if highs[i] == hw.max() and (hw[:center] < highs[i]).all() and (hw[center + 1:] <= highs[i]).all():
            out.append(Swing(times[i], float(highs[i]), "high", times[i + right]))
        if lows[i] == lw.min() and (lw[:center] > lows[i]).all() and (lw[center + 1:] >= lows[i]).all():
            out.append(Swing(times[i], float(lows[i]), "low", times[i + right]))
    out.sort(key=lambda s: s.time)
    return out


def classify_structure(swings: list[Swing]) -> dict:
    """Trend from the last confirmed highs and lows (HH/HL up, LH/LL down)."""
    highs = [s for s in swings if s.kind == "high"]
    lows = [s for s in swings if s.kind == "low"]
    res = {"trend": "range", "last_high": None, "last_low": None, "bos": None}
    if highs:
        res["last_high"] = {"time": highs[-1].time, "price": highs[-1].price}
    if lows:
        res["last_low"] = {"time": lows[-1].time, "price": lows[-1].price}
    if len(highs) >= 2 and len(lows) >= 2:
        hh = highs[-1].price > highs[-2].price
        hl = lows[-1].price > lows[-2].price
        lh = highs[-1].price < highs[-2].price
        ll = lows[-1].price < lows[-2].price
        if hh and hl:
            res["trend"] = "up"
        elif lh and ll:
            res["trend"] = "down"
        else:
            res["trend"] = "range"
    return res


# -- supply / demand zones (small, documented scope) -----------------------


@dataclass
class Zone:
    kind: str            # 'demand' | 'supply'
    low: float
    high: float
    formed_at: str       # base-candle time
    confirmed_at: str    # impulse-close time
    status: str          # 'new' | 'tested' | 'consumed' | 'invalidated'
    invalidation: float  # close beyond this invalidates the zone


def supply_demand_zones(df: pd.DataFrame, impulse_atr_mult: float = 1.5,
                        atr_period: int = 14, lookback: int = 300) -> list[Zone]:
    """Base-before-impulse zones (a deliberately small, declared SMC subset).

    Demand: a bar (the 'base') immediately followed by an up bar whose range
    exceeds ``impulse_atr_mult`` * ATR and which closes above the base high.
    The zone is [base low, base high]; invalidation is a close below base low.
    Supply is the mirror image. Status is evolved only with bars that already
    exist in the frame:
      new        - not revisited since formation
      tested     - later price re-entered the zone but did not close through it
      consumed   - later price traded through and closed beyond the far side
      invalidated- a later close broke the invalidation level
    """
    a = atr(df, atr_period)
    o = df["open"].to_numpy("float64"); h = df["high"].to_numpy("float64")
    low = df["low"].to_numpy("float64"); c = df["close"].to_numpy("float64")
    av = a.to_numpy("float64")
    times = [str(t) for t in df.index]
    n = len(df)
    start = max(1, n - lookback)
    zones: list[Zone] = []
    for i in range(start, n - 1):
        imp = i + 1
        if np.isnan(av[i]) or av[i] <= 0:
            continue
        rng = h[imp] - low[imp]
        if rng < impulse_atr_mult * av[i]:
            continue
        if c[imp] > o[imp] and c[imp] > h[i]:          # up impulse -> demand base at i
            z = Zone("demand", float(low[i]), float(h[i]), times[i], times[imp], "new", float(low[i]))
        elif c[imp] < o[imp] and c[imp] < low[i]:      # down impulse -> supply base at i
            z = Zone("supply", float(low[i]), float(h[i]), times[i], times[imp], "new", float(h[i]))
        else:
            continue
        _evolve_zone_status(z, df.iloc[imp + 1:])
        zones.append(z)
    return zones


def _evolve_zone_status(z: Zone, after: pd.DataFrame) -> None:
    if after.empty:
        return
    h = after["high"].to_numpy("float64"); low = after["low"].to_numpy("float64")
    c = after["close"].to_numpy("float64")
    for k in range(len(after)):
        if z.kind == "demand":
            if c[k] < z.invalidation:
                z.status = "invalidated"; return
            if c[k] > z.high and low[k] <= z.high:     # traded up away after touching
                z.status = "consumed" if z.status != "invalidated" else z.status
            elif low[k] <= z.high and z.status == "new":
                z.status = "tested"
        else:  # supply
            if c[k] > z.invalidation:
                z.status = "invalidated"; return
            if c[k] < z.low and h[k] >= z.low:
                z.status = "consumed" if z.status != "invalidated" else z.status
            elif h[k] >= z.low and z.status == "new":
                z.status = "tested"


# -- liquidity sweeps -------------------------------------------------------


@dataclass
class Sweep:
    time: str
    kind: str            # 'buy_side' (above a high) | 'sell_side' (below a low)
    swept_level: float
    reference_swing_time: str


def liquidity_sweeps(df: pd.DataFrame, swings: list[Swing]) -> list[Sweep]:
    """A bar that pierces a prior confirmed swing then closes back inside it.

    buy_side: bar high > a prior swing high but bar close < that high (buy-side
    liquidity above the high was swept). sell_side is the mirror. Only prior
    CONFIRMED swings are used, so no future information is involved.
    """
    out: list[Sweep] = []
    highs = [s for s in swings if s.kind == "high"]
    lows = [s for s in swings if s.kind == "low"]
    idx_time = {str(t): p for p, t in enumerate(df.index)}
    H = df["high"].to_numpy("float64"); L = df["low"].to_numpy("float64"); C = df["close"].to_numpy("float64")
    times = [str(t) for t in df.index]
    for s in highs:
        sp = idx_time.get(s.confirmed_at)
        if sp is None:
            continue
        for k in range(sp + 1, len(df)):
            if H[k] > s.price and C[k] < s.price:
                out.append(Sweep(times[k], "buy_side", s.price, s.time)); break
    for s in lows:
        sp = idx_time.get(s.confirmed_at)
        if sp is None:
            continue
        for k in range(sp + 1, len(df)):
            if L[k] < s.price and C[k] > s.price:
                out.append(Sweep(times[k], "sell_side", s.price, s.time)); break
    out.sort(key=lambda x: x.time)
    return out
