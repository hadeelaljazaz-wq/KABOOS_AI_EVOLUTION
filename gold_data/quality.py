"""Quality checks for imported OHLCV candles (spec section 2 + review items).

The checker NEVER silently repairs values or drops rows. It returns a
:class:`QualityReport` that enumerates, with reasons and SOURCE ROW NUMBERS:

  - rows rejected (bad timestamp, non-finite/missing or non-positive price,
    OHLC inconsistency, negative volume, conflicting duplicate, or a bar whose
    OPEN is after the reference time);
  - the forming/incomplete last bar at the reference time, set ASIDE from the
    analysis-safe set (its final OHLC is not known until it closes) — this is
    the look-ahead guard (review item 2);
  - identical duplicates collapsed (same timestamp AND same OHLCV AND same
    extras) vs. conflicting duplicates (same timestamp, any difference incl.
    extras) which are rejected;
  - out-of-order input (sorted for output, and reported, not hidden);
  - time gaps, classified conservatively (weekend/closed vs. UNRESOLVED) with
    the number of possibly-missing bars always visible — no gap is declared a
    "session break" from its duration alone (review item 5);
  - staleness measured from the last CLOSED bar's CLOSE time vs. an EXPLICIT
    reference, flagged provisional when the timezone is unknown (items 1, 2).

``accepted_frame`` holds exactly the analysis-safe (closed) rows, unmodified;
``accepted_extras`` holds the preserved extra_* columns for the same rows.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from .mt4_csv import ImportedSeries
from .timeframe import MINUTES, Timeframe
from .timeutil import comparison_index, coerce_reference

_MAX_SAMPLES = 50
_UNSET = object()


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
    unresolved_gap_count: int = 0
    largest_gaps: list[dict[str, Any]] = field(default_factory=list)
    staleness: dict[str, Any] = field(default_factory=dict)
    forming_bar: Optional[dict[str, Any]] = None
    forming_excluded_count: int = 0
    first_bar: Optional[str] = None
    last_bar: Optional[str] = None
    accepted_frame: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    accepted_extras: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)

    @property
    def rejected_total(self) -> int:
        return sum(self.rejected_counts.values())

    @property
    def is_empty_after_checks(self) -> bool:
        return self.accepted_rows == 0

    def summary(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()
                if k not in ("accepted_frame", "accepted_extras")}


def run_quality_checks(
    imported: ImportedSeries,
    reference_time: Any = None,
    stale_tolerance: Optional[_dt.timedelta] = None,
    cutoff_time: Any = _UNSET,
    session_break_minutes: Optional[set[int]] = None,
) -> QualityReport:
    """Validate an :class:`ImportedSeries`; produce a full accept/reject report.

    ``reference_time`` drives staleness and (by default) the look-ahead cutoff.
    A bar is analysis-safe only if it has CLOSED by the reference time
    (open + timeframe <= reference); the still-forming bar is set aside, and
    bars opening after the reference are rejected. ``cutoff_time`` overrides the
    cutoff; pass ``None`` to disable it. Comparisons use UTC when the timezone
    is known, else the naive broker clock (then flagged provisional).
    """
    data_tz = imported.timezone
    tz_known = imported.tz_known
    tf: Timeframe = imported.timeframe.timeframe
    step = MINUTES[tf]
    tf_delta = pd.Timedelta(minutes=step)

    frame = imported.frame
    extras = imported.extras
    n = len(frame)
    src_rows = imported.source_rows or list(range(1, n + 1))

    ref = coerce_reference(reference_time, data_tz, tz_known)
    cutoff = ref if cutoff_time is _UNSET else coerce_reference(cutoff_time, data_tz, tz_known)

    warnings: list[str] = list(imported.notes)
    if (ref is not None or cutoff is not None) and not tz_known:
        warnings.append(
            "reference/cutoff compared against naive timestamps while timezone is "
            "UNKNOWN; staleness/closed/look-ahead results are PROVISIONAL, not a "
            "confirmed temporal verification"
        )

    idx = frame.index
    cmp_idx = comparison_index(idx, data_tz, tz_known)

    o = pd.to_numeric(frame["open"], errors="coerce").to_numpy(dtype="float64")
    h = pd.to_numeric(frame["high"], errors="coerce").to_numpy(dtype="float64")
    low = pd.to_numeric(frame["low"], errors="coerce").to_numpy(dtype="float64")
    c = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype="float64")
    v = pd.to_numeric(frame["volume"], errors="coerce").to_numpy(dtype="float64")

    bad_time = np.asarray(idx.isna())
    with np.errstate(invalid="ignore"):
        nonfinite = ~(np.isfinite(o) & np.isfinite(h) & np.isfinite(low) & np.isfinite(c))
        nonpos = (o <= 0) | (h <= 0) | (low <= 0) | (c <= 0)
        ohlc_bad = (h < o) | (h < c) | (h < low) | (low > o) | (low > c) | (low > h)
        neg_vol = v < 0  # NaN -> False

    # look-ahead split (future rejected, forming set aside)
    future = np.zeros(n, dtype=bool)
    forming = np.zeros(n, dtype=bool)
    if cutoff is not None:
        valid_t = ~bad_time
        open_cmp = cmp_idx
        gt = np.asarray(open_cmp > cutoff)
        close_cmp = open_cmp + tf_delta
        le_open = np.asarray(open_cmp <= cutoff)
        gt_close = np.asarray(close_cmp > cutoff)
        future = valid_t & gt
        forming = valid_t & le_open & gt_close

    rejected = np.zeros(n, dtype=bool)
    reason = np.array([""] * n, dtype=object)

    def mark(mask: np.ndarray, label: str):
        m = mask & ~rejected
        reason[m] = label
        rejected[m] = True

    mark(bad_time, "bad_timestamp")
    mark(nonfinite, "nonfinite_or_missing_price")
    mark(nonpos, "nonpositive_price")
    mark(ohlc_bad, "ohlc_inconsistent")
    mark(neg_vol, "negative_volume")
    mark(future, "after_reference_time")

    forming_mask = forming & ~rejected  # set aside, not an error
    accepted_mask = ~rejected & ~forming_mask

    counts: dict[str, int] = {}
    samples: list[dict[str, Any]] = []
    for label in ("bad_timestamp", "nonfinite_or_missing_price", "nonpositive_price",
                  "ohlc_inconsistent", "negative_volume", "after_reference_time"):
        positions = np.nonzero(reason == label)[0]
        if len(positions):
            counts[label] = int(len(positions))
            for pos in positions[:_MAX_SAMPLES]:
                samples.append({"source_row": int(src_rows[pos]),
                                "time": str(idx[pos]), "reason": label})

    forming_bar = None
    forming_excluded = int(forming_mask.sum())
    if forming_excluded:
        last_forming = np.nonzero(forming_mask)[0][-1]
        forming_bar = {
            "source_row": int(src_rows[last_forming]),
            "open_time": str(idx[last_forming]),
            "closes_at": str((cmp_idx[last_forming] + tf_delta)),
            "note": "forming/incomplete at reference; excluded from analysis-safe set",
        }
        warnings.append(f"{forming_excluded} bar(s) still forming at the reference time "
                        "excluded from the analysis-safe set (kept only descriptively)")

    surviving = frame.iloc[accepted_mask].copy()
    surv_extras = extras.iloc[accepted_mask].copy() if len(extras.columns) else \
        pd.DataFrame(index=surviving.index)
    surv_src = [int(src_rows[i]) for i in np.nonzero(accepted_mask)[0]]

    was_out_of_order = not surviving.index.is_monotonic_increasing
    if was_out_of_order:
        warnings.append("input rows were not in chronological order; sorted for the cleaned output")
    order = np.argsort(surviving.index.values, kind="stable")
    surviving = surviving.iloc[order]
    surv_extras = surv_extras.iloc[order]
    surv_src = [surv_src[i] for i in order]

    # duplicates: identical requires OHLCV AND extras equal; else conflicting
    identical_collapsed = 0
    conflicting_groups = 0
    if surviving.index.has_duplicates:
        combined = surviving.copy()
        for col in surv_extras.columns:
            combined[f"__x_{col}"] = surv_extras[col].to_numpy()
        idx_arr = surviving.index
        dup_times = idx_arr[idx_arr.duplicated(keep=False)].unique()
        drop_positions: list[int] = []
        vals = combined.reset_index(drop=True)
        for t in dup_times:
            positions = [i for i, tt in enumerate(idx_arr) if tt == t]
            block = vals.iloc[positions]
            if block.drop_duplicates().shape[0] == 1:
                identical_collapsed += len(positions) - 1
                drop_positions.extend(positions[1:])
            else:
                conflicting_groups += 1
                counts["conflicting_duplicate"] = counts.get("conflicting_duplicate", 0) + len(positions)
                for i in positions:
                    samples.append({"source_row": surv_src[i], "time": str(t),
                                    "reason": "conflicting_duplicate"})
                drop_positions.extend(positions)
        if drop_positions:
            keep = [i for i in range(len(surviving)) if i not in set(drop_positions)]
            surviving = surviving.iloc[keep]
            surv_extras = surv_extras.iloc[keep]
            surv_src = [surv_src[i] for i in keep]
        if identical_collapsed:
            warnings.append(f"{identical_collapsed} identical duplicate bar(s) collapsed to one each")
        if conflicting_groups:
            warnings.append(f"{conflicting_groups} conflicting duplicate timestamp group(s) rejected "
                            "(same time, different OHLCV or extras)")

    # gaps (conservative classification; missing bars always visible)
    gap_count = 0
    unresolved = 0
    largest: list[dict[str, Any]] = []
    if len(surviving) >= 2:
        weekend_bar_cap = int(round(2.75 * 1440 / step))  # a plain weekend closure, generously
        deltas = surviving.index.to_series().diff().dropna().dt.total_seconds() / 60.0
        gap_points = deltas[deltas > 1.5 * step]
        gap_count = int(len(gap_points))
        for end_time, minutes in gap_points.items():
            pos = surviving.index.get_loc(end_time)
            start_time = surviving.index[pos - 1]
            missing_bars = int(round(minutes / step)) - 1
            weekend = _is_weekend_gap(start_time, end_time)
            if session_break_minutes and round(minutes) in session_break_minutes and not weekend:
                category = "session_break_configured"
            elif weekend and missing_bars <= weekend_bar_cap:
                category = "weekend_or_closed"
            else:
                category = "unresolved"
                unresolved += 1
            largest.append({"from": str(start_time), "to": str(end_time),
                            "minutes": round(float(minutes), 1),
                            "missing_bars": missing_bars, "category": category})
        largest.sort(key=lambda g: g["minutes"], reverse=True)
        largest = largest[:10]
        if unresolved:
            missing_total = sum(g["missing_bars"] for g in largest if g["category"] == "unresolved")
            warnings.append(
                f"{unresolved} UNRESOLVED gap(s) (not a plain weekend; cannot be proven a "
                "session break without a configured session calendar) — review for missing "
                f"data; possible missing bars in the largest listed: {missing_total}")

    # staleness: measured from the last CLOSED bar's CLOSE time
    staleness: dict[str, Any] = {"status": "unknown", "measured_from": "bar_close", "unit": "minutes"}
    last_bar = str(surviving.index[-1]) if len(surviving) else None
    first_bar = str(surviving.index[0]) if len(surviving) else None
    if ref is not None and len(surviving):
        last_cmp = comparison_index(surviving.index[[-1]], data_tz, tz_known)[0]
        last_close = last_cmp + tf_delta
        age_min = (ref - last_close).total_seconds() / 60.0
        tol_min = stale_tolerance.total_seconds() / 60.0 if stale_tolerance else None
        staleness.update({
            "status": "evaluated" if tz_known else "provisional",
            "reference_time": str(ref), "last_closed_bar_open": str(surviving.index[-1]),
            "last_closed_bar_close": str(last_close), "age_minutes": round(age_min, 2),
            "tolerance_minutes": tol_min,
            "is_stale": (age_min > tol_min) if tol_min is not None else None,
            "tz_known": tz_known,
        })
        if tol_min is not None and age_min > tol_min:
            warnings.append(f"data is STALE: last closed bar closes {last_close} is "
                            f"{age_min:.0f} min before reference {ref} (tolerance {tol_min:.0f} min)"
                            + ("" if tz_known else " [PROVISIONAL: timezone unknown]"))

    surviving.index.name = "time"
    return QualityReport(
        symbol=imported.symbol.canonical, timeframe=tf.value,
        total_rows=int(n), accepted_rows=int(len(surviving)),
        rejected_counts=counts, rejected_samples=samples[:_MAX_SAMPLES * 4],
        warnings=warnings, identical_duplicates_collapsed=identical_collapsed,
        conflicting_duplicate_groups=conflicting_groups, was_out_of_order=was_out_of_order,
        gap_count=gap_count, unresolved_gap_count=unresolved, largest_gaps=largest,
        staleness=staleness, forming_bar=forming_bar, forming_excluded_count=forming_excluded,
        first_bar=first_bar, last_bar=last_bar,
        accepted_frame=surviving, accepted_extras=surv_extras,
    )


def _is_weekend_gap(start: pd.Timestamp, end: pd.Timestamp) -> bool:
    """True if the gap spans a Saturday/Sunday (likely weekend market close).

    A weekend span does NOT by itself make a long gap 'weekend-only': the caller
    still compares the number of missing bars against a plain-weekend cap, so a
    gap that also drops extra trading days is classed UNRESOLVED, not hidden.
    """
    if pd.isna(start) or pd.isna(end):
        return False
    day = start.normalize()
    last = end.normalize()
    while day <= last:
        if day.weekday() >= 5:
            return True
        day += pd.Timedelta(days=1)
    return False
