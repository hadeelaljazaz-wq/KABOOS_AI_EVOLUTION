"""The unified multi-timeframe ``MarketSnapshot`` (spec section 3).

A snapshot gathers the AVAILABLE timeframes (organised around D1, H4, H1, M15,
M5) for ONE instrument at one analysis moment. It never invents missing
timeframes and never fabricates synthetic candles to fill them. Re-aggregating
a higher timeframe from a lower one needs declared rules and is intentionally
out of scope here.

It carries everything a consumer needs to judge whether it may be used:

  - a reproducible ``snapshot_id`` fingerprint and full source provenance;
  - available vs. missing timeframes, and each timeframe's time span;
  - timezone and whether it is known;
  - creation time and the analysis reference time;
  - per-timeframe closed-candle status (known / provisional / unknown);
  - the quality report and warnings;
  - validity-for-use judged against EXPLICIT consumer requirements — an
    under-qualified snapshot is never silently treated as valid.

A look-ahead guard rejects any bar later than the analysis reference time, so
an analytical snapshot can never contain data from after the moment it claims
to describe. Snapshots save/load as a directory (meta.json + per-timeframe
CSV) preserving identity and source metadata.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from .mt4_csv import ImportedSeries
from .quality import QualityReport, run_quality_checks
from .symbols import InstrumentKind
from .timeframe import MINUTES, STANDARD_SNAPSHOT_TIMEFRAMES, Timeframe

SCHEMA_VERSION = "gold_snapshot/1"


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _closed_status(last_open: Optional[pd.Timestamp], tf: Timeframe,
                   ref: Optional[pd.Timestamp], tz_known: bool) -> dict[str, Any]:
    """Is the last bar closed at the reference time?

    MT4/MT5 stamp a bar with its OPEN time, so the bar covers
    [open, open+duration). It is closed once ref >= open+duration. Without a
    reference time the status is 'unknown'; with an unknown timezone it is
    'provisional' (the naive-clock comparison may be off by the broker offset).
    """
    if last_open is None or pd.isna(last_open):
        return {"status": "unknown", "reason": "no bars"}
    if ref is None:
        return {"status": "unknown", "reason": "no analysis reference time provided"}
    close_time = last_open + pd.Timedelta(minutes=MINUTES[tf])
    is_closed = ref >= close_time
    return {
        "status": "provisional" if not tz_known else "known",
        "last_bar_open": str(last_open),
        "last_bar_closes_at": str(close_time),
        "reference_time": str(ref),
        "is_closed": bool(is_closed),
    }


@dataclass
class TimeframeData:
    timeframe: str
    source: dict[str, Any]              # ImportedSeries.source_fingerprint()
    quality: dict[str, Any]             # QualityReport.summary()
    first_bar: Optional[str]
    last_bar: Optional[str]
    bar_count: int
    closed_last_bar: dict[str, Any]
    frame: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)


@dataclass
class ConsumerRequirements:
    """What a given consumer needs before it will trust a snapshot."""
    required_timeframes: tuple[str, ...] = ()
    min_bars_per_tf: int = 0
    require_tz_known: bool = False
    require_closed_last_bar: bool = False
    max_staleness: Optional[_dt.timedelta] = None


@dataclass
class ValidityResult:
    valid: bool
    reasons: list[str]
    requirements: dict[str, Any]


class MarketSnapshot:
    def __init__(
        self,
        symbol_canonical: str,
        instrument_kind: InstrumentKind,
        symbol_originals: list[str],
        timezone: Optional[str],
        tz_known: bool,
        timeframes: dict[str, TimeframeData],
        analysis_reference_time: Optional[str],
        created_at: Optional[str] = None,
        warnings: Optional[list[str]] = None,
        requested_timeframes: tuple[str, ...] = tuple(t.value for t in STANDARD_SNAPSHOT_TIMEFRAMES),
        snapshot_id: Optional[str] = None,
    ):
        self.schema_version = SCHEMA_VERSION
        self.symbol_canonical = symbol_canonical
        self.instrument_kind = instrument_kind
        self.symbol_originals = symbol_originals
        self.timezone = timezone
        self.tz_known = tz_known
        self.timeframes = timeframes
        self.analysis_reference_time = analysis_reference_time
        self.created_at = created_at or _now_iso()
        self.warnings = warnings or []
        self.requested_timeframes = tuple(requested_timeframes)
        self.snapshot_id = snapshot_id or self.fingerprint()

    # -- identity --------------------------------------------------------

    def fingerprint(self) -> str:
        """Reproducible id from instrument + per-TF spans/row-counts/file hashes."""
        basis = {
            "schema": SCHEMA_VERSION,
            "symbol": self.symbol_canonical,
            "kind": self.instrument_kind.value,
            "tz": self.timezone or "UNKNOWN",
            "ref": self.analysis_reference_time or "NONE",
            "timeframes": {
                tf: {
                    "first": d.first_bar, "last": d.last_bar, "bars": d.bar_count,
                    "sha": d.source.get("file_sha256"),
                }
                for tf, d in sorted(self.timeframes.items())
            },
        }
        blob = json.dumps(basis, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    # -- availability ----------------------------------------------------

    @property
    def available_timeframes(self) -> list[str]:
        return sorted(self.timeframes, key=lambda t: MINUTES[Timeframe(t)])

    @property
    def missing_timeframes(self) -> list[str]:
        return [tf for tf in self.requested_timeframes if tf not in self.timeframes]

    # -- validity (explicit; never silently "valid") --------------------

    def validity(self, requirements: Optional[ConsumerRequirements] = None) -> ValidityResult:
        req = requirements or ConsumerRequirements()
        reasons: list[str] = []

        for tf in req.required_timeframes:
            if tf not in self.timeframes:
                reasons.append(f"required timeframe {tf} is missing")
            elif self.timeframes[tf].bar_count < req.min_bars_per_tf:
                reasons.append(
                    f"{tf} has {self.timeframes[tf].bar_count} bars (< required "
                    f"{req.min_bars_per_tf})"
                )
        if req.min_bars_per_tf and not req.required_timeframes:
            for tf, d in self.timeframes.items():
                if d.bar_count < req.min_bars_per_tf:
                    reasons.append(f"{tf} has {d.bar_count} bars (< required {req.min_bars_per_tf})")

        if req.require_tz_known and not self.tz_known:
            reasons.append("timezone is UNKNOWN but consumer requires a known timezone")

        if req.require_closed_last_bar:
            for tf in (req.required_timeframes or tuple(self.timeframes)):
                d = self.timeframes.get(tf)
                if d is None:
                    continue
                st = d.closed_last_bar
                if st.get("status") == "unknown" or st.get("is_closed") is not True:
                    reasons.append(f"{tf} last bar is not confirmed closed ({st.get('status')})")

        if req.max_staleness is not None:
            tol_min = req.max_staleness.total_seconds() / 60.0
            for tf in (req.required_timeframes or tuple(self.timeframes)):
                d = self.timeframes.get(tf)
                if d is None:
                    continue
                stale = d.quality.get("staleness", {})
                if stale.get("status") != "evaluated":
                    reasons.append(f"{tf} staleness could not be evaluated (no reference time)")
                elif stale.get("age_minutes", 0) > tol_min:
                    reasons.append(f"{tf} is stale ({stale.get('age_minutes')}min > {tol_min}min)")

        # An empty-requirements check still reports that validity is unqualified.
        if requirements is None:
            reasons.append("no consumer requirements supplied — validity is UNQUALIFIED; "
                           "declare ConsumerRequirements to get a real yes/no")
            return ValidityResult(valid=False, reasons=reasons, requirements={})

        return ValidityResult(valid=not reasons, reasons=reasons, requirements=asdict(req))

    # -- build -----------------------------------------------------------

    @classmethod
    def build(
        cls,
        imports: list[ImportedSeries],
        analysis_reference_time: Any = None,
        reference_time: Any = None,
        stale_tolerance: Optional[_dt.timedelta] = None,
        requested_timeframes: tuple[str, ...] = tuple(t.value for t in STANDARD_SNAPSHOT_TIMEFRAMES),
    ) -> "MarketSnapshot":
        """Build a snapshot from one ImportedSeries per timeframe.

        All imports must be the SAME instrument (canonical symbol + kind);
        mixing spot and futures, or two different instruments, is refused.
        A look-ahead cutoff at ``analysis_reference_time`` rejects later bars.
        Staleness uses ``reference_time`` (defaults to the analysis time).
        """
        if not imports:
            raise ValueError("no imports supplied to build a snapshot")

        canon = {imp.symbol.canonical for imp in imports}
        kinds = {imp.symbol.kind for imp in imports}
        if len(canon) != 1 or len(kinds) != 1:
            raise ValueError(
                f"refusing to mix instruments in one snapshot: symbols={sorted(canon)}, "
                f"kinds={sorted(k.value for k in kinds)} — spot and futures (and "
                "different symbols) must never be combined"
            )
        symbol_canonical = next(iter(canon))
        kind = next(iter(kinds))

        timezones = {imp.timezone for imp in imports}
        if len(timezones) > 1:
            raise ValueError(f"inconsistent timezones across files: {timezones}")
        timezone = next(iter(timezones))
        tz_known = all(imp.tz_known for imp in imports)

        ref_for_stale = reference_time if reference_time is not None else analysis_reference_time
        ref_closed = pd.Timestamp(analysis_reference_time).tz_localize(None) \
            if analysis_reference_time is not None and pd.Timestamp(analysis_reference_time).tzinfo \
            else (pd.Timestamp(analysis_reference_time) if analysis_reference_time is not None else None)

        warnings: list[str] = []
        originals = sorted({imp.symbol.original for imp in imports})
        if len({imp.symbol.broker_suffix for imp in imports}) > 1:
            warnings.append(f"multiple broker symbol variants combined: {originals} "
                            "(same canonical instrument, different broker suffixes)")

        tfmap: dict[str, TimeframeData] = {}
        for imp in imports:
            tf = imp.timeframe.timeframe
            if tf.value in tfmap:
                raise ValueError(f"two files resolved to the same timeframe {tf.value}; "
                                 "supply one file per timeframe")
            report: QualityReport = run_quality_checks(
                imp, reference_time=ref_for_stale, stale_tolerance=stale_tolerance,
                cutoff_time=analysis_reference_time,
            )
            closed = _closed_status(
                report.accepted_frame.index[-1] if report.accepted_rows else None,
                tf, ref_closed, tz_known,
            )
            tfmap[tf.value] = TimeframeData(
                timeframe=tf.value,
                source=imp.source_fingerprint(),
                quality=report.summary(),
                first_bar=report.first_bar,
                last_bar=report.last_bar,
                bar_count=report.accepted_rows,
                closed_last_bar=closed,
                frame=report.accepted_frame,
            )
            for w in report.warnings:
                warnings.append(f"[{tf.value}] {w}")

        return cls(
            symbol_canonical=symbol_canonical,
            instrument_kind=kind,
            symbol_originals=originals,
            timezone=timezone,
            tz_known=tz_known,
            timeframes=tfmap,
            analysis_reference_time=str(analysis_reference_time) if analysis_reference_time is not None else None,
            warnings=warnings,
            requested_timeframes=requested_timeframes,
        )

    # -- serialisation ---------------------------------------------------

    def to_meta(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "snapshot_id": self.snapshot_id,
            "symbol_canonical": self.symbol_canonical,
            "instrument_kind": self.instrument_kind.value,
            "symbol_originals": self.symbol_originals,
            "timezone": self.timezone or "UNKNOWN",
            "tz_known": self.tz_known,
            "created_at": self.created_at,
            "analysis_reference_time": self.analysis_reference_time,
            "requested_timeframes": list(self.requested_timeframes),
            "available_timeframes": self.available_timeframes,
            "missing_timeframes": self.missing_timeframes,
            "warnings": self.warnings,
            "timeframes": {
                tf: {
                    "timeframe": d.timeframe,
                    "source": d.source,
                    "quality": d.quality,
                    "first_bar": d.first_bar,
                    "last_bar": d.last_bar,
                    "bar_count": d.bar_count,
                    "closed_last_bar": d.closed_last_bar,
                    "bars_file": f"{tf}.csv",
                }
                for tf, d in self.timeframes.items()
            },
        }

    def save(self, directory: str | Path) -> Path:
        """Save as a directory: meta.json + one CSV of bars per timeframe."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        meta = self.to_meta()
        for tf, data in self.timeframes.items():
            out = data.frame.copy()
            out.index.name = "time"
            out.to_csv(d / f"{tf}.csv")
        (d / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
        return d


def load_snapshot(directory: str | Path) -> MarketSnapshot:
    """Reconstruct a snapshot saved by :meth:`MarketSnapshot.save`.

    The stored ``snapshot_id`` is preserved exactly, so identity survives the
    round-trip even though the live bars are reloaded from the CSV sidecars.
    """
    d = Path(directory)
    meta = json.loads((d / "meta.json").read_text())
    tfmap: dict[str, TimeframeData] = {}
    for tf, m in meta["timeframes"].items():
        frame = pd.read_csv(d / m["bars_file"], index_col="time", parse_dates=["time"])
        tfmap[tf] = TimeframeData(
            timeframe=m["timeframe"], source=m["source"], quality=m["quality"],
            first_bar=m["first_bar"], last_bar=m["last_bar"], bar_count=m["bar_count"],
            closed_last_bar=m["closed_last_bar"], frame=frame,
        )
    tz = meta["timezone"]
    snap = MarketSnapshot(
        symbol_canonical=meta["symbol_canonical"],
        instrument_kind=InstrumentKind(meta["instrument_kind"]),
        symbol_originals=meta["symbol_originals"],
        timezone=None if tz == "UNKNOWN" else tz,
        tz_known=meta["tz_known"],
        timeframes=tfmap,
        analysis_reference_time=meta["analysis_reference_time"],
        created_at=meta["created_at"],
        warnings=meta["warnings"],
        requested_timeframes=tuple(meta["requested_timeframes"]),
        snapshot_id=meta["snapshot_id"],
    )
    return snap
