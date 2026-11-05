"""Quality checks for imported OHLCV candles (spec section 2, "فحوص الجودة").

The checker NEVER silently repairs values or drops rows. It returns a
:class:`QualityReport` that enumerates, with reasons:

  - rows rejected (bad timestamp, non-finite / non-positive price, OHLC
    inconsistency, negative volume, conflicting duplicate, or a bar later than
    the analysis reference time);
  - identical duplicates collapsed (same timestamp AND same values);
  - whether the input was out of chronological order (it is sorted for the
    cleaned output, and that reordering is reported — not hidden);
  - time gaps, with weekend/market-closed gaps classified separately so a
    closed market is not treated as an error;
  - staleness vs. an EXPLICIT reference time and a configurable tolerance.

The cleaned, analysis-ready frame is returned on ``report.accepted_frame``; it
contains exactly the accepted rows and nothing was changed in them.
"""

from __future__ import annotations

import datetime as _dt
import math
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from .mt4_csv import ImportedSeries
from .timeframe import MINUTES, Timeframe

_MAX_SAMPLES = 50  # cap on per-reason example rows kept in the report


def _coerce_ref_time(ref: Any) -> Optional[pd.Timestamp]:
    if ref is None:
        return None
    ts = pd.Timestamp(ref)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)  # compare in the same naive clock as MT4 bars
    return ts


@dataclass
class QualityReport:
    symbol: str
    timeframe: str
    total_rows: int
    accepted_rows: int
    rejected_counts: dict[str, int] = field(default_factory=dict)
    rejected_samples: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    identical_duplicates_collapsed: int = 0
    conflicting_duplicate_groups: int = 0
    was_out_of_order: bool = False
    gap_count: int = 0
    unexpected_gap_count: int = 0
    largest_gaps: list[dict[str, Any]] = field(default_factory=list)
    staleness: dict[str, Any] = field(default_factory=dict)
    first_bar: Optional[str] = None
    last_bar: Optional[str] = None
    # the cleaned, accepted-only frame (nothing modified in the surviving rows)
    accepted_frame: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)

    @property
    def rejected_total(self) -> int:
        return sum(self.rejected_counts.values())

    @property
    def is_empty_after_checks(self) -> bool:
        return self.accepted_rows == 0

    def summary(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "accepted_frame"}
        return d


def _record_reject(counts, samples, mask, frame, reason):
    n = int(mask.sum())
    if n == 0:
        return
    counts[reason] = counts.get(reason, 0) + n
    for ts in frame.index[mask][:_MAX_SAMPLES]:
        if len(samples) < _MAX_SAMPLES * 4:
            samples.append({"time": str(ts), "reason": reason})


def run_quality_checks(
    imported: ImportedSeries,
    reference_time: Any = None,
    stale_tolerance: Optional[_dt.timedelta] = None,
    cutoff_time: Any = None,
) -> QualityReport:
    """Validate an :class:`ImportedSeries`; produce a full accept/reject report.

    ``reference_time`` drives the staleness check (and is the default cutoff).
    ``cutoff_time`` (defaults to ``reference_time``) rejects any bar stamped
    AFTER it — the look-ahead guard for an analytical snapshot. Timestamps are
    naive MT4/MT5 clock values; the reference time is compared in that same
    naive clock, and the assumption is noted when the timezone is unknown.
    """
    f = imported.frame
    tf = imported.timeframe.timeframe
    counts: dict[str, int] = {}
    samples: list[dict[str, Any]] = []
    warnings: list[str] = list(imported.notes)

    ref = _coerce_ref_time(reference_time)
    cutoff = _coerce_ref_time(cutoff_time) if cutoff_time is not None else ref
    if (ref is not None or cutoff is not None) and not imported.tz_known:
        warnings.append(
            "reference/cutoff time compared against naive MT4 timestamps while "
            "timezone is UNKNOWN; staleness and look-ahead results assume the "
            "same clock and are provisional"
        )

    # Work on a copy with a positional reset so masks align with original order.
    df = f.copy()
    o, h, l, c, v = (df["open"], df["high"], df["low"], df["close"], df["volume"])

    bad_time = df.index.isna()
    # finite check only where price present
    def _nonfinite(s):
        return s.apply(lambda x: isinstance(x, float) and (math.isinf(x))) | s.isna()
    nonfinite = _nonfinite(o) | _nonfinite(h) | _nonfinite(l) | _nonfinite(c)
    valid_prices = ~(bad_time | nonfinite)

    nonpos = pd.Series(False, index=df.index)
    nonpos[valid_prices] = (o[valid_prices] <= 0) | (h[valid_prices] <= 0) | \
                           (l[valid_prices] <= 0) | (c[valid_prices] <= 0)

    ohlc_bad = pd.Series(False, index=df.index)
    chk = valid_prices & ~nonpos
    hi_ok = h[chk] >= o[chk]
    hi_ok &= h[chk] >= c[chk]
    hi_ok &= h[chk] >= l[chk]
    lo_ok = l[chk] <= o[chk]
    lo_ok &= l[chk] <= c[chk]
    lo_ok &= l[chk] <= h[chk]
    ohlc_bad[chk] = ~(hi_ok & lo_ok)

    neg_vol = pd.Series(False, index=df.index)
    has_vol = v.notna()
    neg_vol[has_vol] = v[has_vol] < 0

    after_cutoff = pd.Series(False, index=df.index)
    if cutoff is not None:
        valid_t = ~bad_time
        after_cutoff[valid_t] = df.index[valid_t] > cutoff

    # Priority order so each row is rejected for one primary reason.
    rejected = pd.Series(False, index=df.index)
    for mask, reason in [
        (bad_time, "bad_timestamp"),
        (nonfinite & ~rejected, "nonfinite_or_missing_price"),
        (nonpos & ~rejected, "nonpositive_price"),
        (ohlc_bad & ~rejected, "ohlc_inconsistent"),
        (neg_vol & ~rejected, "negative_volume"),
        (after_cutoff & ~rejected, "after_reference_time"),
    ]:
        m = mask & ~rejected
        _record_reject(counts, samples, m, df, reason)
        rejected |= m

    surviving = df[~rejected].copy()

    # Ordering: report (don't hide) that the input wasn't sorted.
    was_out_of_order = not surviving.index.is_monotonic_increasing
    if was_out_of_order:
        warnings.append("input rows were not in chronological order; sorted for the cleaned output")
    surviving = surviving.sort_index(kind="stable")

    # Duplicates: identical (collapse, report) vs conflicting (reject all, report).
    identical_collapsed = 0
    conflicting_groups = 0
    if surviving.index.has_duplicates:
        dup_times = surviving.index[surviving.index.duplicated(keep=False)].unique()
        keep_mask = pd.Series(True, index=range(len(surviving)))
        cols = ["open", "high", "low", "close", "volume"]
        vals = surviving[cols].reset_index(drop=True)
        idx_arr = surviving.index
        drop_positions: list[int] = []
        for t in dup_times:
            positions = [i for i, tt in enumerate(idx_arr) if tt == t]
            block = vals.iloc[positions]
            if block.drop_duplicates().shape[0] == 1:
                # identical duplicates -> keep first, drop the rest
                identical_collapsed += len(positions) - 1
                drop_positions.extend(positions[1:])
            else:
                # conflicting -> ambiguous, reject the whole group
                conflicting_groups += 1
                counts["conflicting_duplicate"] = counts.get("conflicting_duplicate", 0) + len(positions)
                for i in positions:
                    if len(samples) < _MAX_SAMPLES * 4:
                        samples.append({"time": str(t), "reason": "conflicting_duplicate"})
                drop_positions.extend(positions)
        if drop_positions:
            keep = [i for i in range(len(surviving)) if i not in set(drop_positions)]
            surviving = surviving.iloc[keep]
        if identical_collapsed:
            warnings.append(f"{identical_collapsed} identical duplicate bar(s) collapsed to one each")

    # Gaps (weekend/market-closed classified separately, not treated as errors).
    gap_count = 0
    unexpected = 0
    largest: list[dict[str, Any]] = []
    if len(surviving) >= 2:
        step = MINUTES[tf]
        # A gap is classified, never treated as an error:
        #   weekend_or_closed : the gap spans a Saturday/Sunday (market closed)
        #   session_break     : a small intraday gap (<= 4 bars or <= 180 min),
        #                       typically a daily broker rollover / session close
        #   unexpected        : a larger non-weekend gap that may be missing data
        # Only 'unexpected' is counted as something to worry about.
        minor_threshold = max(180.0, 4 * step)
        deltas = surviving.index.to_series().diff().dropna().dt.total_seconds() / 60.0
        gap_mask = deltas > 1.5 * step
        gap_count = int(gap_mask.sum())
        gap_points = deltas[gap_mask]
        for end_time, minutes in gap_points.items():
            pos = surviving.index.get_loc(end_time)
            start_time = surviving.index[pos - 1]
            if _is_weekend_gap(start_time, end_time):
                category = "weekend_or_closed"
            elif minutes <= minor_threshold:
                category = "session_break"
            else:
                category = "unexpected"
                unexpected += 1
            largest.append({
                "from": str(start_time), "to": str(end_time),
                "minutes": round(float(minutes), 1),
                "missing_bars": int(round(minutes / step)) - 1,
                "category": category,
            })
        largest.sort(key=lambda g: g["minutes"], reverse=True)
        largest = largest[:10]
        if unexpected:
            warnings.append(f"{unexpected} unexpected (non-weekend, non-session-break) "
                            "time gap(s) detected — review for missing data")

    # Staleness vs explicit reference time.
    staleness: dict[str, Any] = {"status": "unknown"}
    last_bar = str(surviving.index[-1]) if len(surviving) else None
    first_bar = str(surviving.index[0]) if len(surviving) else None
    if ref is not None and len(surviving):
        age = ref - surviving.index[-1]
        age_min = age.total_seconds() / 60.0
        tol_min = stale_tolerance.total_seconds() / 60.0 if stale_tolerance else None
        staleness = {
            "status": "evaluated",
            "reference_time": str(ref),
            "last_bar": last_bar,
            "age_minutes": round(age_min, 2),
            "tolerance_minutes": tol_min,
            "is_stale": (age_min > tol_min) if tol_min is not None else None,
            "tz_known": imported.tz_known,
        }
        if tol_min is not None and age_min > tol_min:
            warnings.append(f"data is STALE: last bar {last_bar} is {age_min:.0f} min "
                            f"before reference {ref} (tolerance {tol_min:.0f} min)")

    return QualityReport(
        symbol=imported.symbol.canonical,
        timeframe=tf.value,
        total_rows=int(len(df)),
        accepted_rows=int(len(surviving)),
        rejected_counts=counts,
        rejected_samples=samples[:_MAX_SAMPLES * 4],
        warnings=warnings,
        identical_duplicates_collapsed=identical_collapsed,
        conflicting_duplicate_groups=conflicting_groups,
        was_out_of_order=was_out_of_order,
        gap_count=gap_count,
        unexpected_gap_count=unexpected,
        largest_gaps=largest,
        staleness=staleness,
        first_bar=first_bar,
        last_bar=last_bar,
        accepted_frame=surviving,
    )


def _is_weekend_gap(start: pd.Timestamp, end: pd.Timestamp) -> bool:
    """Heuristic: does the gap span a weekend / likely market close?

    True if any calendar day strictly between the two bars (inclusive of the
    start/end days) is a Saturday or Sunday, or the gap runs Friday->Monday.
    Does not claim holidays; those simply show as non-weekend gaps for review.
    """
    if pd.isna(start) or pd.isna(end):
        return False
    day = start.normalize()
    last = end.normalize()
    while day <= last:
        if day.weekday() >= 5:  # Sat/Sun
            return True
        day += pd.Timedelta(days=1)
    return False
