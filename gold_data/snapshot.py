"""The unified multi-timeframe ``MarketSnapshot`` (spec section 3 + review items).

A snapshot gathers the AVAILABLE timeframes for ONE instrument at one analysis
moment. It never invents missing timeframes, never fabricates synthetic
candles, and never silently mixes instruments or data sources.

Identity & integrity (review item 8): the ``snapshot_id`` is a fingerprint of
the ACTUAL bar CONTENT (plus instrument/timezone/reference), not of the source
input file. Saving writes each timeframe's bars atomically and records a
content hash; loading re-verifies those hashes, the schema, columns, counts
and the recomputed id, and refuses a tampered, missing or incompatible file.

Validity (review items 2, 3): validity is judged against EXPLICIT, validated
``ConsumerRequirements``. A missing timeframe, a timeframe whose declared value
conflicts with the data, a present-but-empty timeframe, an unconfirmed closed
bar (provisional / unknown), or staleness that cannot be confirmed all make a
snapshot NOT valid for that consumer.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from .mt4_csv import ImportedSeries
from .quality import QualityReport, run_quality_checks
from .symbols import InstrumentKind
from .timeframe import MINUTES, STANDARD_SNAPSHOT_TIMEFRAMES, Timeframe
from .timeutil import comparison_index, coerce_reference

SCHEMA_VERSION = "gold_snapshot/2"
_OHLCV = ["open", "high", "low", "close", "volume"]
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_]+\.csv$")


class IntegrityError(ValueError):
    """Raised when a loaded snapshot fails content/schema verification."""


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _normalize_bars(frame: pd.DataFrame, extras: pd.DataFrame) -> pd.DataFrame:
    """Fixed, round-trip-stable representation of one timeframe's bars.

    - index: naive DatetimeIndex named 'time'
    - open/high/low/close/volume: float64 (missing -> NaN)
    - extra_* columns: kept as RAW TEXT (object strings) so an uninterpreted
      value such as '007' keeps its leading zeros and 'NA' stays the text 'NA'
      rather than being turned into a number or a missing value.

    Applying this to an already-normalised frame is idempotent, so
    build -> save -> load -> save reproduces identical bytes.
    """
    out = pd.DataFrame(index=pd.DatetimeIndex(frame.index))
    out.index.name = "time"
    for col in _OHLCV:
        if col in frame.columns:
            out[col] = pd.to_numeric(frame[col], errors="coerce").astype("float64")
        else:
            out[col] = pd.Series([float("nan")] * len(frame), index=out.index, dtype="float64")
    for col in extras.columns:
        out[col] = [("" if (v is None or (isinstance(v, float) and pd.isna(v))) else str(v))
                    for v in extras[col].to_numpy()]
    return out


def _canonical_csv_bytes(frame: pd.DataFrame, extras: pd.DataFrame) -> bytes:
    """Deterministic byte serialisation of a timeframe's bars (OHLCV + extras)."""
    return _normalize_bars(frame, extras).to_csv().encode("utf-8")


def _parse_canonical(raw: bytes) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inverse of the canonical serialisation: strings stay strings, prices float.

    Returns (frame[OHLCV], extras) in the SAME normalised form, so recomputing
    the content hash from the result reproduces the stored hash exactly.
    """
    import io
    df = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False, na_filter=False)
    if "time" not in df.columns:
        raise IntegrityError("bars file has no 'time' column")
    index = pd.DatetimeIndex(pd.to_datetime(df["time"]) if len(df) else [], name="time")
    frame = pd.DataFrame(index=index)
    for col in _OHLCV:
        if col in df.columns:
            frame[col] = pd.to_numeric(df[col], errors="coerce").astype("float64").to_numpy()
    extra_cols = [c for c in df.columns if c != "time" and c not in _OHLCV]
    extras = pd.DataFrame(index=index)
    for col in extra_cols:
        extras[col] = df[col].to_numpy()  # raw text preserved
    return frame, extras


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _closed_status(last_open: Optional[pd.Timestamp], tf: Timeframe,
                   ref: Optional[pd.Timestamp], tz_known: bool) -> dict[str, Any]:
    """Closed status of the last accepted (already analysis-safe) bar.

    'known' only when the timezone is known; otherwise 'provisional'; 'unknown'
    when there is no reference time. A consumer requiring a confirmed close must
    reject anything other than 'known' + is_closed True.
    """
    if last_open is None or pd.isna(last_open):
        return {"status": "unknown", "reason": "no bars"}
    if ref is None:
        return {"status": "unknown", "reason": "no analysis reference time provided"}
    close_time = last_open + pd.Timedelta(minutes=MINUTES[tf])
    return {
        "status": "known" if tz_known else "provisional",
        "last_bar_open": str(last_open),
        "last_bar_closes_at": str(close_time),
        "reference_time": str(ref),
        "is_closed": bool(ref >= close_time),
    }


@dataclass
class TimeframeData:
    timeframe: str
    timeframe_consistent: bool
    timeframe_source: str
    source: dict[str, Any]
    quality: dict[str, Any]
    first_bar: Optional[str]
    last_bar: Optional[str]
    bar_count: int
    closed_last_bar: dict[str, Any]
    content_sha256: str
    forming_bar: Optional[dict[str, Any]] = None
    frame: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    extras: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)


@dataclass
class ConsumerRequirements:
    """What a consumer needs before it will trust a snapshot. Values validated."""
    required_timeframes: tuple[str, ...] = ()
    min_bars_per_tf: int = 0
    require_tz_known: bool = False
    require_closed_last_bar: bool = False
    max_staleness: Optional[_dt.timedelta] = None

    def __post_init__(self):
        valid = {t.value for t in Timeframe}
        bad = [tf for tf in self.required_timeframes if tf not in valid]
        if bad:
            raise ValueError(f"unknown timeframe(s) in requirements: {bad}; valid: {sorted(valid)}")
        if not isinstance(self.min_bars_per_tf, int) or isinstance(self.min_bars_per_tf, bool):
            raise ValueError("min_bars_per_tf must be an int")
        if self.min_bars_per_tf < 0:
            raise ValueError("min_bars_per_tf must be >= 0")
        if self.max_staleness is not None:
            if not isinstance(self.max_staleness, _dt.timedelta):
                raise ValueError("max_staleness must be a timedelta")
            secs = self.max_staleness.total_seconds()
            if secs < 0 or not np.isfinite(secs):
                raise ValueError("max_staleness must be a non-negative, finite duration")


@dataclass
class ValidityResult:
    valid: bool
    reasons: list[str]
    requirements: dict[str, Any]


class MarketSnapshot:
    def __init__(self, symbol_canonical, instrument_kind, symbol_originals, source_ids,
                 timezone, tz_known, timeframes, analysis_reference_time, created_at=None,
                 warnings=None, requested_timeframes=tuple(t.value for t in STANDARD_SNAPSHOT_TIMEFRAMES),
                 snapshot_id=None):
        self.schema_version = SCHEMA_VERSION
        self.symbol_canonical = symbol_canonical
        self.instrument_kind = instrument_kind
        self.symbol_originals = symbol_originals
        self.source_ids = source_ids
        self.timezone = timezone
        self.tz_known = tz_known
        self.timeframes: dict[str, TimeframeData] = timeframes
        self.analysis_reference_time = analysis_reference_time
        self.created_at = created_at or _now_iso()
        self.warnings = warnings or []
        self.requested_timeframes = tuple(requested_timeframes)
        self.snapshot_id = snapshot_id or self.compute_id()

    # -- identity / integrity -------------------------------------------

    def _id_basis(self) -> dict[str, Any]:
        # Everything that defines WHICH data this snapshot is: instrument,
        # declared source(s), timezone + whether it is known, the analysis
        # reference time, and each timeframe's content + span + count. Changing
        # any of these must change the id (review item 6).
        return {
            "schema": SCHEMA_VERSION,
            "symbol": self.symbol_canonical,
            "kind": self.instrument_kind.value,
            "source_ids": sorted(self.source_ids),
            "tz": self.timezone or "UNKNOWN",
            "tz_known": bool(self.tz_known),
            "ref": self.analysis_reference_time or "NONE",
            "timeframes": {
                tf: {"first": d.first_bar, "last": d.last_bar, "bars": d.bar_count,
                     "content": d.content_sha256}
                for tf, d in sorted(self.timeframes.items())
            },
        }

    def compute_id(self) -> str:
        blob = json.dumps(self._id_basis(), sort_keys=True, separators=(",", ":"))
        return _sha(blob.encode())[:16]

    def current_content_hashes(self) -> dict[str, str]:
        """Recompute content hashes from the CURRENT in-memory bars."""
        return {tf: _sha(_canonical_csv_bytes(d.frame, d.extras))
                for tf, d in self.timeframes.items()}

    def verify_integrity(self) -> None:
        """Raise IntegrityError if in-memory content drifted from the identity.

        Detects a frame mutated in place after the snapshot was built/loaded:
        the recomputed content hash (and thus id) no longer matches.
        """
        current = self.current_content_hashes()
        for tf, d in self.timeframes.items():
            if current[tf] != d.content_sha256:
                raise IntegrityError(
                    f"timeframe {tf} content changed since the snapshot id was fixed "
                    "(in-memory mutation); build a new snapshot instead")
        if self.compute_id() != self.snapshot_id:
            raise IntegrityError("snapshot id no longer matches its content")

    # -- availability ----------------------------------------------------

    @property
    def available_timeframes(self) -> list[str]:
        return sorted(self.timeframes, key=lambda t: MINUTES[Timeframe(t)])

    @property
    def missing_timeframes(self) -> list[str]:
        return [tf for tf in self.requested_timeframes if tf not in self.timeframes]

    # -- validity --------------------------------------------------------

    def validity(self, requirements: Optional[ConsumerRequirements] = None) -> ValidityResult:
        if requirements is None:
            return ValidityResult(
                valid=False,
                reasons=["no consumer requirements supplied — validity is UNQUALIFIED; "
                         "declare ConsumerRequirements for a real yes/no"],
                requirements={})
        req = requirements
        reasons: list[str] = []

        # (review item 1) integrity is a binding part of the usage gate: a
        # snapshot whose in-memory content drifted from its fixed id is not
        # valid, no matter what the requirements say.
        try:
            self.verify_integrity()
        except IntegrityError as exc:
            return ValidityResult(valid=False, reasons=[f"integrity: {exc}"],
                                  requirements=asdict(req))

        # (review item 4) a snapshot with no analysis-usable data anywhere is
        # never valid — a known timezone is not evidence of valid empty data.
        if not self.timeframes or all(d.bar_count == 0 for d in self.timeframes.values()):
            reasons.append("no accepted rows in any timeframe (empty/descriptive snapshot)")

        checked = req.required_timeframes or tuple(self.timeframes)
        for tf in req.required_timeframes:
            if tf not in self.timeframes:
                reasons.append(f"required timeframe {tf} is missing")

        # empty/conflict checks apply to EVERY timeframe the consumer will use
        # (named explicitly, or all available when none are named).
        for tf in checked:
            d = self.timeframes.get(tf)
            if d is None:
                continue
            if d.bar_count == 0:
                reasons.append(f"{tf} has no accepted rows (present but empty)")
            if not d.timeframe_consistent:
                reasons.append(f"{tf} declared timeframe conflicts with the data "
                               "(proven inconsistency) — resolve before use")
            if req.min_bars_per_tf and 0 < d.bar_count < req.min_bars_per_tf:
                reasons.append(f"{tf} has {d.bar_count} bars (< required {req.min_bars_per_tf}) "
                               "— cannot verify on this little data")

        if req.require_tz_known and not self.tz_known:
            reasons.append("timezone is UNKNOWN but consumer requires a known timezone")

        if req.require_closed_last_bar:
            for tf in checked:
                d = self.timeframes.get(tf)
                if d is None:
                    continue
                st = d.closed_last_bar
                if st.get("status") != "known":
                    reasons.append(f"{tf} closed-bar status is '{st.get('status')}', not a "
                                   "confirmed close (timezone must be known)")
                elif st.get("is_closed") is not True:
                    reasons.append(f"{tf} last bar is not closed at the reference time")

        if req.max_staleness is not None:
            if not self.tz_known:
                reasons.append("staleness cannot be CONFIRMED while timezone is UNKNOWN")
            tol_min = req.max_staleness.total_seconds() / 60.0
            for tf in checked:
                d = self.timeframes.get(tf)
                if d is None:
                    continue
                stale = d.quality.get("staleness", {})
                if stale.get("status") not in ("evaluated", "provisional"):
                    reasons.append(f"{tf} staleness could not be evaluated (no reference time)")
                elif stale.get("age_minutes", 0) > tol_min:
                    reasons.append(f"{tf} is stale ({stale.get('age_minutes')}min > {tol_min}min)")

        return ValidityResult(valid=not reasons, reasons=reasons, requirements=asdict(req))

    # -- build -----------------------------------------------------------

    @classmethod
    def build(cls, imports, analysis_reference_time=None, reference_time=None,
              stale_tolerance=None,
              requested_timeframes=tuple(t.value for t in STANDARD_SNAPSHOT_TIMEFRAMES),
              session_break_minutes=None, allow_mixed_sources=False):
        if not imports:
            raise ValueError("no imports supplied to build a snapshot")
        canon = {im.symbol.canonical for im in imports}
        kinds = {im.symbol.kind for im in imports}
        if len(canon) != 1 or len(kinds) != 1:
            raise ValueError(
                f"refusing to mix instruments in one snapshot: symbols={sorted(canon)}, "
                f"kinds={sorted(k.value for k in kinds)} — spot and futures (and different "
                "symbols) must never be combined")
        symbol_canonical = next(iter(canon))
        kind = next(iter(kinds))

        timezones = {im.timezone for im in imports}
        if len(timezones) > 1:
            raise ValueError(f"inconsistent timezones across files: {timezones}")
        timezone = next(iter(timezones))
        tz_known = all(im.tz_known for im in imports)

        source_ids = sorted({im.source_id for im in imports})
        if len(source_ids) > 1 and not allow_mixed_sources:
            raise ValueError(
                f"refusing to mix data sources {source_ids} in one analytical snapshot; "
                "pass allow_mixed_sources=True to do so explicitly (provenance is preserved)")

        warnings: list[str] = []
        if len(source_ids) > 1:
            warnings.append(f"MIXED SOURCES combined explicitly: {source_ids}")
        originals = sorted({im.symbol.original for im in imports})
        if len({im.symbol.broker_suffix for im in imports}) > 1:
            warnings.append(f"multiple broker symbol variants combined: {originals}")

        ref_for_stale = reference_time if reference_time is not None else analysis_reference_time
        ref_closed = coerce_reference(analysis_reference_time, timezone, tz_known)

        tfmap: dict[str, TimeframeData] = {}
        for im in imports:
            tf = im.timeframe.timeframe
            if tf.value in tfmap:
                raise ValueError(f"two files resolved to the same timeframe {tf.value}; "
                                 "supply one file per timeframe")
            rep: QualityReport = run_quality_checks(
                im, reference_time=ref_for_stale, stale_tolerance=stale_tolerance,
                cutoff_time=analysis_reference_time, session_break_minutes=session_break_minutes)
            last_open = rep.accepted_frame.index[-1] if rep.accepted_rows else None
            last_cmp = comparison_index(rep.accepted_frame.index[[-1]], timezone, tz_known)[0] \
                if rep.accepted_rows else None
            closed = _closed_status(last_cmp, tf, ref_closed, tz_known)
            if last_open is not None:
                closed["last_bar_open"] = str(last_open)
            # store the normalised, round-trip-stable representation so a built
            # snapshot and a loaded one are byte-identical in content
            norm = _normalize_bars(rep.accepted_frame, rep.accepted_extras)
            norm_frame = norm[[c for c in _OHLCV if c in norm.columns]].copy()
            norm_extras = norm[[c for c in norm.columns if c not in _OHLCV]].copy()
            content_sha = _sha(norm.to_csv().encode("utf-8"))
            tfmap[tf.value] = TimeframeData(
                timeframe=tf.value, timeframe_consistent=im.timeframe.consistent,
                timeframe_source=im.timeframe.source, source=im.source_fingerprint(),
                quality=rep.summary(), first_bar=rep.first_bar, last_bar=rep.last_bar,
                bar_count=rep.accepted_rows, closed_last_bar=closed, content_sha256=content_sha,
                forming_bar=rep.forming_bar, frame=norm_frame, extras=norm_extras)
            warnings.extend(f"[{tf.value}] {w}" for w in rep.warnings)

        return cls(symbol_canonical=symbol_canonical, instrument_kind=kind,
                   symbol_originals=originals, source_ids=source_ids, timezone=timezone,
                   tz_known=tz_known, timeframes=tfmap,
                   analysis_reference_time=str(analysis_reference_time) if analysis_reference_time is not None else None,
                   warnings=warnings, requested_timeframes=requested_timeframes)

    # -- serialisation ---------------------------------------------------

    def to_meta(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version, "snapshot_id": self.snapshot_id,
            "symbol_canonical": self.symbol_canonical, "instrument_kind": self.instrument_kind.value,
            "symbol_originals": self.symbol_originals, "source_ids": self.source_ids,
            "timezone": self.timezone or "UNKNOWN", "tz_known": self.tz_known,
            "created_at": self.created_at, "analysis_reference_time": self.analysis_reference_time,
            "requested_timeframes": list(self.requested_timeframes),
            "available_timeframes": self.available_timeframes,
            "missing_timeframes": self.missing_timeframes, "warnings": self.warnings,
            "timeframes": {
                tf: {"timeframe": d.timeframe, "timeframe_consistent": d.timeframe_consistent,
                     "timeframe_source": d.timeframe_source, "source": d.source,
                     "quality": d.quality, "first_bar": d.first_bar, "last_bar": d.last_bar,
                     "bar_count": d.bar_count, "closed_last_bar": d.closed_last_bar,
                     "content_sha256": d.content_sha256, "forming_bar": d.forming_bar,
                     "bars_file": f"{tf}.csv", "extra_columns": list(d.extras.columns)}
                for tf, d in self.timeframes.items()},
        }

    def save(self, directory: str | Path) -> Path:
        """Atomically save: per-timeframe CSV (written via temp+rename) then meta.json.

        The content hash stored in meta equals the sha256 of each written CSV, so
        a later edit to the file is detected on load.
        """
        self.verify_integrity()
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        for tf, data in self.timeframes.items():
            payload = _canonical_csv_bytes(data.frame, data.extras)
            tmp = d / f"{tf}.csv.tmp"
            tmp.write_bytes(payload)
            os.replace(tmp, d / f"{tf}.csv")
        meta_tmp = d / "meta.json.tmp"
        meta_tmp.write_text(json.dumps(self.to_meta(), indent=2, default=str))
        os.replace(meta_tmp, d / "meta.json")
        return d


def _split_ohlcv_extras(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = df[[c for c in _OHLCV if c in df.columns]].copy()
    extra_cols = [c for c in df.columns if c not in _OHLCV]
    extras = df[extra_cols].copy() if extra_cols else pd.DataFrame(index=df.index)
    return frame, extras


def load_snapshot(directory: str | Path) -> MarketSnapshot:
    """Load a saved snapshot, verifying integrity and schema.

    Rejects (IntegrityError) a tampered or missing bars file, a mismatched
    schema/columns/row-count, or a recomputed id that disagrees with the stored
    one. Rejects any ``bars_file`` that would escape the snapshot directory.
    """
    d = Path(directory)
    meta_path = d / "meta.json"
    if not meta_path.exists():
        raise IntegrityError(f"no meta.json in {d} (incomplete or not a snapshot directory)")
    meta = json.loads(meta_path.read_text())
    if meta.get("schema_version") != SCHEMA_VERSION:
        raise IntegrityError(f"unsupported schema_version {meta.get('schema_version')!r}")

    tz = meta["timezone"]
    # cross-check tz_known vs timezone (review item 6): they must agree
    if meta["tz_known"] != (tz != "UNKNOWN"):
        raise IntegrityError(f"tz_known={meta['tz_known']} disagrees with timezone={tz!r}")

    tfmap: dict[str, TimeframeData] = {}
    for tf, m in meta["timeframes"].items():
        bars_file = m["bars_file"]
        if not _SAFE_NAME.match(bars_file) or bars_file != f"{tf}.csv":
            raise IntegrityError(f"unsafe or unexpected bars_file {bars_file!r} for {tf}")
        fp = d / bars_file
        if not fp.exists():
            raise IntegrityError(f"missing bars file {bars_file} for timeframe {tf}")
        raw = fp.read_bytes()
        if _sha(raw) != m["content_sha256"]:
            raise IntegrityError(f"content hash mismatch for {tf} ({bars_file} was modified)")
        # parse into the normalised form; strings stay strings, prices float
        frame, extras = _parse_canonical(raw)
        if _sha(_canonical_csv_bytes(frame, extras)) != m["content_sha256"]:
            raise IntegrityError(f"{tf} serialisation drift on load (content not round-trip stable)")
        missing_cols = [c for c in ["open", "high", "low", "close"] if c not in frame.columns]
        if missing_cols:
            raise IntegrityError(f"{tf} bars missing columns {missing_cols}")
        # RE-DERIVE identity-bearing facts from the actual content; never trust
        # the metadata copies on their own (review item 6).
        if len(frame) != m["bar_count"]:
            raise IntegrityError(f"{tf} row count {len(frame)} != stored {m['bar_count']}")
        derived_first = str(frame.index[0]) if len(frame) else None
        derived_last = str(frame.index[-1]) if len(frame) else None
        if (m["first_bar"] or None) != derived_first:
            raise IntegrityError(f"{tf} first bar {derived_first} != stored {m['first_bar']}")
        if (m["last_bar"] or None) != derived_last:
            raise IntegrityError(f"{tf} last bar {derived_last} != stored {m['last_bar']}")
        # per-timeframe source must match the snapshot's instrument
        src = m.get("source", {})
        if src.get("symbol_canonical") not in (None, meta["symbol_canonical"]):
            raise IntegrityError(f"{tf} source symbol {src.get('symbol_canonical')} != "
                                 f"snapshot symbol {meta['symbol_canonical']}")
        if src.get("instrument_kind") not in (None, meta["instrument_kind"]):
            raise IntegrityError(f"{tf} source kind {src.get('instrument_kind')} != "
                                 f"snapshot kind {meta['instrument_kind']}")
        tfmap[tf] = TimeframeData(
            timeframe=m["timeframe"], timeframe_consistent=m["timeframe_consistent"],
            timeframe_source=m["timeframe_source"], source=m["source"], quality=m["quality"],
            first_bar=m["first_bar"], last_bar=m["last_bar"], bar_count=m["bar_count"],
            closed_last_bar=m["closed_last_bar"], content_sha256=m["content_sha256"],
            forming_bar=m.get("forming_bar"), frame=frame, extras=extras)

    snap = MarketSnapshot(
        symbol_canonical=meta["symbol_canonical"],
        instrument_kind=InstrumentKind(meta["instrument_kind"]),
        symbol_originals=meta["symbol_originals"], source_ids=meta.get("source_ids", ["unknown"]),
        timezone=None if tz == "UNKNOWN" else tz, tz_known=meta["tz_known"], timeframes=tfmap,
        analysis_reference_time=meta["analysis_reference_time"], created_at=meta["created_at"],
        warnings=meta["warnings"], requested_timeframes=tuple(meta["requested_timeframes"]),
        snapshot_id=meta["snapshot_id"])
    if snap.compute_id() != meta["snapshot_id"]:
        raise IntegrityError("recomputed snapshot id does not match stored id (metadata tampered)")
    return snap
