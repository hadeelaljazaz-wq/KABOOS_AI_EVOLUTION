"""Real OHLCV data access for the Strategy Discovery Lab.

Starts with gold per spec section 7. Uses yfinance (keyless, free) and
caches every fetch to disk so re-runs don't hit the network and so
results are reproducible from a fixed snapshot.

IMPORTANT — what this data actually is (spec Phase 2 requires honesty here):
The gold series is sourced from ``GC=F``, the COMEX gold *futures*
continuous contract, as *daily end-of-day bars*. It is NOT XAUUSD spot and
it is NOT a real-time/intraday feed. Futures carry roll and basis effects
that make them a close-but-imperfect proxy for spot XAUUSD. Every fetch
records its real yfinance symbol and fetch timestamp in a ``.meta.txt``
sidecar; callers that need to know provenance should read it rather than
assume "live spot". ``market_source(market)`` returns this provenance in
code.

No synthetic/fabricated prices are ever generated here — a failed fetch
with no cache raises, it never falls back to made-up data.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parents[1] / "data_cache"

# yfinance symbol per supported market (spec section 7). Each entry records
# the real instrument and bar granularity, so nothing is silently passed off
# as spot or as real-time. Gold first; the rest are wired for when the lab
# is extended to those markets.
SYMBOLS = {
    "XAUUSD": "GC=F",    # COMEX gold FUTURES continuous, daily bars (NOT spot XAUUSD)
    "XAGUSD": "SI=F",    # COMEX silver futures continuous, daily bars
    "BTCUSD": "BTC-USD",  # spot crypto, daily bars
    "US100": "^NDX",      # Nasdaq-100 index level, daily bars
    "US30": "^DJI",       # Dow Jones index level, daily bars
    "EURUSD": "EURUSD=X",  # FX, daily bars
    "GBPUSD": "GBPUSD=X",  # FX, daily bars
    "USDJPY": "USDJPY=X",  # FX, daily bars
    "USOIL": "CL=F",      # WTI crude futures continuous, daily bars
}

# Human-readable provenance, so no layer above this one has to guess whether
# a series is spot, futures, or an index level.
SOURCE_NOTES = {
    "XAUUSD": "COMEX gold futures (GC=F), daily EOD bars — proxy for XAUUSD spot, not spot itself",
    "XAGUSD": "COMEX silver futures (SI=F), daily EOD bars — proxy for XAGUSD spot",
    "BTCUSD": "BTC-USD spot, daily bars",
    "US100": "Nasdaq-100 index level (^NDX), daily bars",
    "US30": "Dow Jones index level (^DJI), daily bars",
    "EURUSD": "EUR/USD FX, daily bars",
    "GBPUSD": "GBP/USD FX, daily bars",
    "USDJPY": "USD/JPY FX, daily bars",
    "USOIL": "WTI crude futures (CL=F), daily bars",
}


def market_source(market: str) -> str:
    """Honest provenance string for a market key (raises for unknown keys)."""
    if market not in SOURCE_NOTES:
        raise ValueError(f"unknown market '{market}', known: {sorted(SYMBOLS)}")
    return SOURCE_NOTES[market]


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
    last_bar = str(df.index[-1]) if len(df) else "n/a"
    meta_file.write_text(
        f"source=yfinance symbol={SYMBOLS[market]} "
        f"provenance=\"{SOURCE_NOTES[market]}\" "
        f"fetched_at={datetime.datetime.now(datetime.timezone.utc).isoformat()} "
        f"last_bar={last_bar} rows={len(df)} period={period} interval={interval}\n"
    )
    return df
