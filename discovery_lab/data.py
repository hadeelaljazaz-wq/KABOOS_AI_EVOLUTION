"""Real OHLCV data access for the Strategy Discovery Lab.

Starts with gold (XAUUSD) per spec section 7. Uses yfinance (keyless,
free) and caches every fetch to disk so re-runs don't hit the network
and so results are reproducible from a fixed snapshot.

No synthetic/fabricated prices are ever generated here — a failed fetch
with no cache raises, it never falls back to made-up data.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parents[1] / "data_cache"

# yfinance symbol per supported market (spec section 7). Gold first; the
# rest are wired for when the lab is extended to those markets.
SYMBOLS = {
    "XAUUSD": "GC=F",   # COMEX gold futures (continuous) — gold spot proxy
    "XAGUSD": "SI=F",
    "BTCUSD": "BTC-USD",
    "US100": "^NDX",
    "US30": "^DJI",
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "USOIL": "CL=F",
}


def _cache_path(market: str, interval: str) -> Path:
    return CACHE_DIR / f"{market}_{interval}.csv"


def fetch_ohlcv(
    market: str,
    period: str = "2y",
    interval: str = "1d",
    use_cache: bool = True,
    refresh: bool = False,
) -> pd.DataFrame:
    """Real daily OHLCV for one of the markets in SYMBOLS.

    Returns columns: Open, High, Low, Close, Volume, indexed by date.
    Raises ValueError for an unknown market, and RuntimeError if there is
    no cache and the network fetch fails or returns nothing (never
    silently substitutes fake data).
    """
    if market not in SYMBOLS:
        raise ValueError(f"unknown market '{market}', known: {sorted(SYMBOLS)}")

    cache_file = _cache_path(market, interval)
    if use_cache and not refresh and cache_file.exists():
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        return df

    try:
        import yfinance as yf

        raw = yf.download(SYMBOLS[market], period=period, interval=interval,
                           progress=False, auto_adjust=False)
    except Exception as exc:  # network / yfinance failure
        if cache_file.exists():
            return pd.read_csv(cache_file, index_col=0, parse_dates=True)
        raise RuntimeError(f"could not fetch real data for {market} and no cache exists") from exc

    if raw is None or raw.empty:
        if cache_file.exists():
            return pd.read_csv(cache_file, index_col=0, parse_dates=True)
        raise RuntimeError(f"yfinance returned no data for {market}")

    # yfinance returns a MultiIndex column (Field, Ticker) for a single symbol too
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index.name = "Date"

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_file)
    meta_file = cache_file.with_suffix(".meta.txt")
    meta_file.write_text(
        f"source=yfinance symbol={SYMBOLS[market]} fetched_at={datetime.datetime.now(datetime.timezone.utc).isoformat()}"
        f" rows={len(df)} period={period} interval={interval}\n"
    )
    return df
