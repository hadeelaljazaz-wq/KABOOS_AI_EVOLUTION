"""Timeframes for the gold data layer.

The analytical ``MarketSnapshot`` is built around five standard timeframes
(D1, H4, H1, M15, M5); M1 and M30 are also recognised because real MT4/MT5
exports commonly use them. Each timeframe has a fixed bar duration in minutes.

A timeframe must come from an EXPLICIT input or trusted file info, then be
verified against the data's actual median bar spacing. The filename is only a
weak hint and is never authoritative on its own (spec section 2).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd


class Timeframe(str, Enum):
    M1 = "M1"
    M5 = "M5"
    M15 = "M15"
    M30 = "M30"
    H1 = "H1"
    H4 = "H4"
    D1 = "D1"


# Canonical bar duration in minutes.
MINUTES: dict[Timeframe, int] = {
    Timeframe.M1: 1,
    Timeframe.M5: 5,
    Timeframe.M15: 15,
    Timeframe.M30: 30,
    Timeframe.H1: 60,
    Timeframe.H4: 240,
    Timeframe.D1: 1440,
}

# The five timeframes the analytical snapshot is organised around.
STANDARD_SNAPSHOT_TIMEFRAMES: tuple[Timeframe, ...] = (
    Timeframe.D1, Timeframe.H4, Timeframe.H1, Timeframe.M15, Timeframe.M5,
)

# MT4/MT5 period-in-minutes tags sometimes embedded in filenames.
_PERIOD_TAG = {1: Timeframe.M1, 5: Timeframe.M5, 15: Timeframe.M15,
               30: Timeframe.M30, 60: Timeframe.H1, 240: Timeframe.H4, 1440: Timeframe.D1}


def parse_timeframe(value: str) -> Timeframe:
    """Parse 'H1', 'h1', '60', 'M15', '15', '1h', '4h', '1d' ... to Timeframe."""
    if isinstance(value, Timeframe):
        return value
    s = str(value).strip().upper()
    if not s:
        raise ValueError("empty timeframe")
    try:
        return Timeframe(s)
    except ValueError:
        pass
    if s.isdigit():  # minutes tag
        tf = _PERIOD_TAG.get(int(s))
        if tf:
            return tf
    aliases = {"1H": Timeframe.H1, "4H": Timeframe.H4, "1D": Timeframe.D1, "DAILY": Timeframe.D1,
               "1M": Timeframe.M1, "5M": Timeframe.M5, "15M": Timeframe.M15, "30M": Timeframe.M30}
    if s in aliases:
        return aliases[s]
    raise ValueError(f"unknown timeframe {value!r}; known: {[t.value for t in Timeframe]}")


def timeframe_from_filename(name: str) -> Optional[Timeframe]:
    """Weak hint only: pull a trailing timeframe tag out of a filename.

    e.g. 'XAUUSD.smlM15.csv' -> M15, 'EURUSD60.csv' -> H1. Returns None if no
    confident match. Callers must still verify against the data.
    """
    stem = name.rsplit("/", 1)[-1]
    stem = stem.rsplit(".csv", 1)[0].rsplit(".txt", 1)[0].upper()
    # longest explicit token first
    for tag in ("M1", "M5", "M15", "M30", "H1", "H4", "D1"):
        if stem.endswith(tag):
            return Timeframe(tag)
    # trailing digits = period in minutes
    trailing = ""
    for ch in reversed(stem):
        if ch.isdigit():
            trailing = ch + trailing
        else:
            break
    if trailing and int(trailing) in _PERIOD_TAG:
        return _PERIOD_TAG[int(trailing)]
    return None


@dataclass(frozen=True)
class TimeframeInference:
    inferred: Optional[Timeframe]
    median_spacing_min: float
    consistent_fraction: float  # fraction of consecutive gaps equal to the modal bar size

    def is_confident(self) -> bool:
        return self.inferred is not None and self.consistent_fraction >= 0.5


def infer_timeframe(timestamps: pd.Series) -> TimeframeInference:
    """Infer the timeframe from the data's own bar spacing (robust to gaps).

    Uses the MODAL positive gap (most common spacing) rather than the mean, so
    weekend/holiday gaps don't distort it, then snaps to the nearest known
    timeframe when the modal gap matches one exactly.
    """
    ts = pd.to_datetime(pd.Series(timestamps)).sort_values().drop_duplicates()
    if len(ts) < 3:
        return TimeframeInference(None, float("nan"), 0.0)
    deltas = ts.diff().dropna().dt.total_seconds() / 60.0
    deltas = deltas[deltas > 0]
    if deltas.empty:
        return TimeframeInference(None, float("nan"), 0.0)
    modal = float(deltas.round().mode().iloc[0])
    consistent = float((deltas.round() == modal).mean())
    inferred = next((tf for tf, m in MINUTES.items() if m == modal), None)
    return TimeframeInference(inferred, float(deltas.median()), consistent)


@dataclass(frozen=True)
class TimeframeResolution:
    timeframe: Timeframe
    source: str                     # 'explicit', 'filename+verified', 'inferred'
    inference: TimeframeInference
    consistent: bool                # explicit/hint agrees with the data
    note: str = ""


def resolve_timeframe(
    timestamps: pd.Series,
    explicit: Optional[str] = None,
    filename: Optional[str] = None,
) -> TimeframeResolution:
    """Decide the timeframe from explicit input or trusted hints, VERIFIED against data.

    Priority: explicit > filename hint (only if it agrees with the data) >
    inference. An explicit timeframe that disagrees with the data is still
    used (the caller asked for it) but flagged ``consistent=False`` with a note,
    so a mislabelled export is surfaced rather than hidden.
    """
    inf = infer_timeframe(timestamps)
    if explicit is not None:
        tf = parse_timeframe(explicit)
        consistent = inf.inferred == tf
        note = "" if consistent else (
            f"explicit timeframe {tf.value} disagrees with data spacing "
            f"(modal≈{inf.median_spacing_min:.1f}min, inferred="
            f"{inf.inferred.value if inf.inferred else 'none'})"
        )
        return TimeframeResolution(tf, "explicit", inf, consistent, note)

    if filename:
        hint = timeframe_from_filename(filename)
        if hint is not None and inf.inferred == hint:
            return TimeframeResolution(hint, "filename+verified", inf, True,
                                       "filename tag confirmed by data spacing")
        if hint is not None and inf.inferred is not None:
            # filename and data disagree -> do NOT trust filename; use inference
            return TimeframeResolution(
                inf.inferred, "inferred", inf, True,
                f"filename hint {hint.value} contradicted by data; used inferred "
                f"{inf.inferred.value}",
            )

    if inf.inferred is not None:
        return TimeframeResolution(inf.inferred, "inferred", inf, inf.is_confident(),
                                   "no explicit timeframe; inferred from data spacing")
    raise ValueError(
        "could not determine timeframe: pass --timeframe explicitly "
        f"(data median spacing ≈ {inf.median_spacing_min:.2f} min did not match a known TF)"
    )
