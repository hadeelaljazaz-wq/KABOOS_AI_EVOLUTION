"""KABOOS gold data foundation.

A unified, honest data layer for XAUUSD analysis:

  - ``symbols`` / ``timeframe``: declared normalisation + timeframe resolution.
  - ``mt4_csv``: import MT4/MT5 OHLCV CSV exports (encoding/delimiter/header
    sniffed, timezone never assumed, nothing silently repaired).
  - ``quality``: validation producing a report of what was accepted/rejected
    and why.
  - ``snapshot``: a reproducible multi-timeframe ``MarketSnapshot`` with
    validity-for-use, save/load, and a look-ahead cutoff.
  - ``contracts``: validated output contracts (ExpertReport, TradePlan,
    DiscussionMessage, Outcome) the experts and the office will populate.

This layer deliberately builds NO experts, discussion engine, UI, scheduling,
or LLM calls — only the data foundation and the contracts.
"""

from .symbols import InstrumentKind, NormalizedSymbol, normalize_symbol, same_instrument
from .timeframe import (
    STANDARD_SNAPSHOT_TIMEFRAMES,
    Timeframe,
    infer_timeframe,
    parse_timeframe,
    resolve_timeframe,
)
from .mt4_csv import ImportedSeries, UnsupportedFormatError, read_mt4_csv
from .quality import QualityReport, run_quality_checks
from .snapshot import (
    ConsumerRequirements,
    IntegrityError,
    MarketSnapshot,
    TimeframeData,
    ValidityResult,
    load_snapshot,
)
from .timeutil import TimezoneError, validate_timezone
from .contracts import (
    Direction,
    DiscussionMessage,
    ExpertReport,
    MessageType,
    Outcome,
    PlanStatus,
    PriceLevel,
    TradePlan,
    Zone,
)

__all__ = [
    # symbols / timeframe
    "InstrumentKind", "NormalizedSymbol", "normalize_symbol", "same_instrument",
    "Timeframe", "parse_timeframe", "resolve_timeframe", "infer_timeframe",
    "STANDARD_SNAPSHOT_TIMEFRAMES",
    # import
    "read_mt4_csv", "ImportedSeries", "UnsupportedFormatError",
    # quality
    "run_quality_checks", "QualityReport",
    # snapshot
    "MarketSnapshot", "TimeframeData", "load_snapshot", "ConsumerRequirements",
    "ValidityResult", "IntegrityError",
    # time
    "validate_timezone", "TimezoneError",
    # contracts
    "ExpertReport", "TradePlan", "DiscussionMessage", "Outcome",
    "Direction", "PlanStatus", "MessageType", "Zone", "PriceLevel",
]
