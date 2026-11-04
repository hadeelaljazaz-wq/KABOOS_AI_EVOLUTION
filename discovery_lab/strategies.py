"""First testable strategies for the Discovery Lab.

Every signal here is computed with .shift(1): the position held *during*
bar t is decided only from information available at the close of bar
t-1. That shift is the one line standing between this and look-ahead
bias — removing it would let a signal "see" the same bar's own close.
"""

from __future__ import annotations

import pandas as pd


def sma_crossover_signal(close: pd.Series, fast: int = 20, slow: int = 50) -> pd.Series:
    """+1 long / -1 short / 0 flat, decided from yesterday's moving averages."""
    if fast >= slow:
        raise ValueError("fast window must be shorter than slow window")
    sma_fast = close.rolling(fast).mean()
    sma_slow = close.rolling(slow).mean()
    raw_signal = (sma_fast > sma_slow).astype(int) - (sma_fast < sma_slow).astype(int)
    return raw_signal.shift(1).fillna(0)


def buy_and_hold_signal(close: pd.Series) -> pd.Series:
    """Benchmark: always long, from the first bar once data exists."""
    return pd.Series(1, index=close.index).shift(1).fillna(0)
