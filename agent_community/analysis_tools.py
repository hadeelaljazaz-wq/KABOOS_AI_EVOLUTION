"""Real analysis any agent can request (spec section 3: "request data analysis").

These compute genuine indicators from the real cached OHLCV in
``discovery_lab`` — no fabricated numbers. Every summary carries the honest
provenance string (e.g. gold = GC=F COMEX *futures*, daily EOD bars, NOT XAUUSD
spot) so nothing downstream can pass the series off as live spot.

The indicators deliberately span several schools (trend, momentum, volatility,
return) so an agent is never limited to one lens.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from discovery_lab.data import fetch_ohlcv, market_source


def rsi(close: pd.Series, period: int = 14) -> float:
    """Classic Wilder-style RSI on the most recent bar (0..100)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi_series = 100.0 - 100.0 / (1.0 + rs)
    val = rsi_series.iloc[-1]
    if pd.isna(val):
        return 50.0
    return float(val)


def sma_trend(close: pd.Series, fast: int = 20, slow: int = 50) -> int:
    """+1 if fast SMA above slow SMA, -1 if below, 0 if undefined/equal."""
    if len(close) < slow:
        return 0
    f = close.rolling(fast).mean().iloc[-1]
    s = close.rolling(slow).mean().iloc[-1]
    if pd.isna(f) or pd.isna(s) or f == s:
        return 0
    return 1 if f > s else -1


def annualised_vol(close: pd.Series, window: int = 20) -> float:
    rets = close.pct_change().dropna()
    if len(rets) < 2:
        return 0.0
    return float(rets.tail(window).std(ddof=0) * np.sqrt(252))


def window_return(close: pd.Series, window: int = 20) -> float:
    if len(close) <= window:
        return 0.0
    return float(close.iloc[-1] / close.iloc[-1 - window] - 1.0)


@dataclass
class MarketSnapshot:
    """A self-describing bundle of real indicators for one market/window."""

    market: str
    provenance: str
    is_spot: bool
    last_bar: str
    n_bars: int
    window: int
    indicators: dict[str, Any] = field(default_factory=dict)

    def as_evidence(self) -> dict[str, Any]:
        """A well-formed evidence payload for collective_memory.validate_evidence."""
        return {
            "data_source": self.provenance,
            "sample_size": int(self.n_bars),
            "metrics": dict(self.indicators),
        }


def summarize_market(market: str = "XAUUSD", window: int = 20, period: str = "2y") -> MarketSnapshot:
    """Compute a multi-school indicator snapshot from real cached data.

    ``is_spot`` is derived from the provenance note, so a caller can never
    accidentally treat the GC=F futures proxy as live XAUUSD spot.
    """
    df = fetch_ohlcv(market, period=period)
    close = df["Close"].astype(float)
    provenance = market_source(market)
    is_spot = ("spot" in provenance.lower()) and ("not spot" not in provenance.lower())
    indicators = {
        "trend": sma_trend(close),                 # trend_following
        "rsi": round(rsi(close), 2),               # momentum
        "ret_window": round(window_return(close, window), 5),  # return over window
        "vol_annual": round(annualised_vol(close, window), 5),  # volatility
        "last_close": round(float(close.iloc[-1]), 2),
    }
    return MarketSnapshot(
        market=market,
        provenance=provenance,
        is_spot=is_spot,
        last_bar=str(df.index[-1]),
        n_bars=int(len(df)),
        window=window,
        indicators=indicators,
    )
