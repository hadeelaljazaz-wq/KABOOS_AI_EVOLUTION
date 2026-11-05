"""Importer for MT4 / MT5 exported OHLCV candles from CSV.

Supported, DOCUMENTED input shapes (verified against real broker exports):

  1. MT4 standard bar export, header or no header, comma- or tab-separated::

         Date, Time, Open, High, Low, Close, Volume
         2024.01.02,00:00,2045.12,2047.80,2044.00,2046.50,1234

  2. MT5-style export with a combined ``date time`` field and extra trailing
     columns (tick volume + a broker-specific column whose meaning is not
     declared in the file)::

         2025.06.06 20:40,3322.55000,3322.96500,3321.78000,3322.33000,578,0

Encodings handled: UTF-16 (LE/BE, with BOM — the real files are UTF-16-LE),
UTF-8 (with/without BOM), latin-1 fallback. Delimiter (``,`` / tab / ``;``)
and the presence of a header row are sniffed and reported, not assumed.

What the importer DOES NOT do:
  - It never guesses the timezone (MT4/MT5 server time is broker-specific). The
    timezone is UNKNOWN unless passed in explicitly.
  - It never silently repairs or drops rows. Unparseable numeric cells become
    NaN / NaT and are left for the quality layer to report and reject.
  - It never relabels one instrument as another; the symbol is normalised by
    the declared rule in ``symbols.py`` and the broker original is preserved.
  - It never calls the last close "the current price".

Unknown shapes are rejected with ``UnsupportedFormatError`` and a helpful
message that shows the offending line and lists the supported shapes.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from .symbols import NormalizedSymbol, normalize_symbol
from .timeframe import TimeframeResolution, resolve_timeframe
from .timeutil import validate_timezone

OHLC = ("open", "high", "low", "close")

_DATE_RE = re.compile(r"^\d{4}[.\-/]\d{1,2}[.\-/]\d{1,2}$")
_TIME_RE = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?$")
_DATETIME_RE = re.compile(r"^\d{4}[.\-/]\d{1,2}[.\-/]\d{1,2}[ T]\d{1,2}:\d{2}(:\d{2})?$")

_DT_FORMATS = (
    "%Y.%m.%d %H:%M", "%Y.%m.%d %H:%M:%S", "%Y.%m.%d",
    "%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d",
    "%Y/%m/%d %H:%M", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d",
)

# Recognised header names -> canonical role. A header cell is normalised by
# lowercasing and stripping <> and surrounding whitespace before lookup.
_COL_ALIASES = {
    "date": "date", "time": "time",
    "datetime": "datetime", "date time": "datetime", "timestamp": "datetime",
    "open": "open", "o": "open",
    "high": "high", "h": "high",
    "low": "low", "l": "low",
    "close": "close", "c": "close",
    "tickvol": "volume", "tick_volume": "volume", "tickvolume": "volume",
    "volume": "volume", "vol": "volume",
    "spread": "spread", "real_volume": "real_volume", "realvolume": "real_volume",
}
_REQUIRED_ROLES = ("open", "high", "low", "close")
_HEADER_TOKEN_SET = set(_COL_ALIASES)


class UnsupportedFormatError(ValueError):
    """Raised when a file does not match any documented MT4/MT5 CSV shape."""


@dataclass
class ImportedSeries:
    frame: pd.DataFrame                 # index = naive Timestamp 'time'; cols open/high/low/close/volume
    extras: pd.DataFrame                # same index; preserved trailing columns, meaning undeclared
    symbol: NormalizedSymbol
    timeframe: TimeframeResolution
    timezone: Optional[str]             # None => UNKNOWN (not assumed)
    tz_known: bool
    source_path: str
    source_name: str
    file_sha256: str
    encoding: str
    delimiter: str
    had_header: bool
    header_tokens: list[str]
    imported_at: str
    raw_row_count: int
    source_id: str = "unknown"          # declared broker/feed/server id; never inferred from the symbol
    source_rows: list[int] = field(default_factory=list)  # source line number per parsed row (for audit)
    original_columns: list[str] = field(default_factory=list)  # original header names, if any
    notes: list[str] = field(default_factory=list)

    @property
    def canonical_symbol(self) -> str:
        return self.symbol.canonical

    def source_fingerprint(self) -> dict[str, object]:
        """Everything needed to identify this import's origin (no local path)."""
        return {
            "source_name": self.source_name,
            "source_id": self.source_id,
            "file_sha256": self.file_sha256,
            "symbol_original": self.symbol.original,
            "symbol_canonical": self.symbol.canonical,
            "instrument_kind": self.symbol.kind.value,
            "timeframe": self.timeframe.timeframe.value,
            "timeframe_source": self.timeframe.source,
            "timeframe_consistent": self.timeframe.consistent,
            "encoding": self.encoding,
            "delimiter": "tab" if self.delimiter == "\t" else self.delimiter,
            "had_header": self.had_header,
            "original_columns": self.original_columns,
            "extra_columns": list(self.extras.columns),
            "timezone": self.timezone or "UNKNOWN",
            "tz_known": self.tz_known,
            "imported_at": self.imported_at,
            "raw_row_count": self.raw_row_count,
        }


# -- sniffing ---------------------------------------------------------------


def _detect_encoding(raw: bytes) -> str:
    if raw[:2] == b"\xff\xfe" or raw[:2] == b"\xfe\xff":
        return "utf-16"
    if raw[:3] == b"\xef\xbb\xbf":
        return "utf-8-sig"
    try:
        raw[:4096].decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        return "latin-1"


def _detect_delimiter(lines: list[str]) -> str:
    sample = [ln for ln in lines[:30] if ln.strip()][:20]
    best, best_count = ",", -1
    for delim in (",", "\t", ";"):
        counts = [ln.count(delim) for ln in sample]
        if not counts:
            continue
        modal = max(set(counts), key=counts.count)
        if modal > best_count:
            best, best_count = delim, modal
    if best_count <= 0:
        raise UnsupportedFormatError(
            "could not find a comma, tab or semicolon delimiter in the file"
        )
    return best


def _norm_header_cell(cell: str) -> str:
    return cell.strip().strip("<>").strip().lower()


def _is_header_row(fields: list[str]) -> bool:
    """A row is a header only if enough cells are RECOGNISED column names.

    This deliberately does NOT treat "first field is not a date" as a header:
    a corrupt first data row must be rejected, not silently swallowed as a
    header (spec review item 4).
    """
    recognised = sum(1 for f in fields if _norm_header_cell(f) in _HEADER_TOKEN_SET)
    return recognised >= 3


def _map_header(fields: list[str]) -> dict[str, int]:
    """Map a header row to {role: column_index}, rejecting duplicates/missing.

    Supports columns in any order via their names. A duplicate recognised role
    or a missing required price column raises UnsupportedFormatError.
    """
    roles: dict[str, int] = {}
    extras_seen: list[str] = []
    for i, cell in enumerate(fields):
        role = _COL_ALIASES.get(_norm_header_cell(cell))
        if role is None:
            continue
        # 'volume' may be claimed by tickvol AND volume/vol; keep the first,
        # treat the later one as an extra rather than a hard duplicate error.
        if role in roles:
            if role == "volume":
                extras_seen.append(cell)
                continue
            raise UnsupportedFormatError(
                f"duplicate column for '{role}' in header {fields!r}"
            )
        roles[role] = i
    missing = [r for r in _REQUIRED_ROLES if r not in roles]
    if missing:
        raise UnsupportedFormatError(
            f"header {fields!r} is missing required column(s): {missing}. "
            "Supported names: date/time (or a combined datetime), open, high, low, "
            "close, and optionally volume."
        )
    if "datetime" not in roles and "date" not in roles:
        raise UnsupportedFormatError(
            f"header {fields!r} has no date/datetime column"
        )
    return roles


def _layout(fields: list[str]) -> tuple[str, int]:
    """Return (datetime_mode, price_start_index) for a HEADERLESS data row, or raise."""
    f0 = fields[0].strip()
    if _DATETIME_RE.match(f0):
        return "combined", 1
    if _DATE_RE.match(f0) and len(fields) > 1 and _TIME_RE.match(fields[1].strip()):
        return "split", 2
    if _DATE_RE.match(f0):
        return "date_only", 1
    raise UnsupportedFormatError(
        "first field is not a recognised date/datetime: "
        f"{f0!r}. With no header, only these shapes are supported: "
        "'YYYY.MM.DD', 'YYYY.MM.DD HH:MM[:SS]', or a 'Date','Time' pair. "
        "(If this row was meant to be a header, its column names were not "
        "recognised.)"
    )


# -- parsing ----------------------------------------------------------------


def _parse_datetimes(s: pd.Series) -> pd.Series:
    """Parse MT4/MT5 datetime strings; unparseable -> NaT (never dropped)."""
    sample = next((v for v in s if isinstance(v, str) and v.strip()), None)
    if sample is not None:
        for fmt in _DT_FORMATS:
            try:
                _dt.datetime.strptime(sample.strip(), fmt)
                return pd.to_datetime(s.str.strip(), format=fmt, errors="coerce")
            except ValueError:
                continue
    return pd.to_datetime(s, errors="coerce")  # flexible fallback; bad -> NaT


def read_mt4_csv(
    path: str | Path,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    timezone: Optional[str] = None,
    source_id: Optional[str] = None,
) -> ImportedSeries:
    """Read one MT4/MT5 CSV export into an :class:`ImportedSeries`.

    ``symbol`` defaults to the stem of the filename (broker symbol kept, then
    normalised). ``timeframe`` may be given explicitly; otherwise it is
    resolved from a filename hint and/or the data's own spacing and verified.
    ``timezone`` is VALIDATED (a bogus name raises) and recorded; omit it to
    mark the timezone UNKNOWN. ``source_id`` is a declared broker/feed/server
    identity; it is never inferred from the symbol suffix and defaults to
    'unknown'.
    """
    timezone = validate_timezone(timezone)  # bogus tz -> raises, never tz_known
    p = Path(path)
    raw = p.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    encoding = _detect_encoding(raw)
    text = raw.decode(encoding, errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    text = text.lstrip("﻿")
    all_lines = text.split("\n")
    # keep source line numbers (1-based) for audit; skip blank lines
    numbered = [(i + 1, ln) for i, ln in enumerate(all_lines) if ln.strip()]
    if not numbered:
        raise UnsupportedFormatError(f"file {p.name} is empty")

    nonempty = [ln for _, ln in numbered]
    delim = _detect_delimiter(nonempty)
    first_fields = nonempty[0].split(delim)
    notes: list[str] = []

    header_map: Optional[dict[str, int]] = None
    had_header = _is_header_row(first_fields)
    header_tokens: list[str] = []
    if had_header:
        header_tokens = [f.strip() for f in first_fields]
        header_map = _map_header(first_fields)   # raises on duplicate/missing
        data_numbered = numbered[1:]
    else:
        data_numbered = numbered
    if not data_numbered:
        raise UnsupportedFormatError(f"file {p.name} has a header but no data rows")

    src_rows = [n for n, _ in data_numbered]
    data_lines = [ln for _, ln in data_numbered]

    buf = io.StringIO("\n".join(data_lines))
    df = pd.read_csv(buf, header=None, sep=delim, dtype=str,
                     skip_blank_lines=False, engine="c")
    ncols = df.shape[1]

    if header_map is not None:
        # locate columns BY NAME (supports any column order, no positional swap)
        def col(role):
            return df.iloc[:, header_map[role]]
        if "datetime" in header_map:
            dt_str = col("datetime")
        elif "time" in header_map:
            dt_str = col("date").str.strip() + " " + col("time").str.strip()
        else:
            dt_str = col("date")
        o = pd.to_numeric(col("open"), errors="coerce")
        h = pd.to_numeric(col("high"), errors="coerce")
        low = pd.to_numeric(col("low"), errors="coerce")
        c = pd.to_numeric(col("close"), errors="coerce")
        used = set(header_map.values())
        if "volume" in header_map:
            volume = pd.to_numeric(col("volume"), errors="coerce")
            notes.append("volume column is broker tick volume (not exchange/contract volume)")
        else:
            volume = pd.Series([pd.NA] * len(df), index=df.index)
            notes.append("no volume column present in source")
        extra_idx = [i for i in range(ncols) if i not in used]
        extra_names = [header_tokens[i] if i < len(header_tokens) and header_tokens[i]
                       else f"extra_{i}" for i in extra_idx]
    else:
        dt_mode, price_start = _layout(data_lines[0].split(delim))
        if ncols < price_start + 4:
            raise UnsupportedFormatError(
                f"file {p.name}: found {ncols} columns but need a datetime plus at "
                f"least 4 price columns (O/H/L/C). First data row: {data_lines[0]!r}. "
                "Supported shapes: 'Date,Time,O,H,L,C,V' or 'datetime,O,H,L,C,V[,...]'."
            )
        if dt_mode == "split":
            dt_str = df.iloc[:, 0].str.strip() + " " + df.iloc[:, 1].str.strip()
        else:
            dt_str = df.iloc[:, 0]
        o = pd.to_numeric(df.iloc[:, price_start + 0], errors="coerce")
        h = pd.to_numeric(df.iloc[:, price_start + 1], errors="coerce")
        low = pd.to_numeric(df.iloc[:, price_start + 2], errors="coerce")
        c = pd.to_numeric(df.iloc[:, price_start + 3], errors="coerce")
        vol_idx = price_start + 4
        if vol_idx < ncols:
            volume = pd.to_numeric(df.iloc[:, vol_idx], errors="coerce")
            notes.append("volume column is broker tick volume (not exchange/contract volume)")
            extra_idx = list(range(vol_idx + 1, ncols))
        else:
            volume = pd.Series([pd.NA] * len(df), index=df.index)
            notes.append("no volume column present in source")
            extra_idx = []
        extra_names = [f"extra_{j}" for j in range(1, len(extra_idx) + 1)]

    times = _parse_datetimes(dt_str)
    frame = pd.DataFrame({"open": o.to_numpy(), "high": h.to_numpy(),
                          "low": low.to_numpy(), "close": c.to_numpy(),
                          "volume": volume.to_numpy()})
    frame.index = pd.DatetimeIndex(pd.to_datetime(times).to_numpy(), name="time")

    if extra_idx:
        extras = df.iloc[:, extra_idx].copy()
        extras.columns = extra_names
        extras.index = frame.index
        notes.append(
            f"{len(extra_idx)} extra column(s) preserved {extra_names} "
            "(broker-specific; meaning not declared — possibly real volume or spread)"
        )
    else:
        extras = pd.DataFrame(index=frame.index)

    sym = normalize_symbol(symbol if symbol is not None else _symbol_from_name(p.name))
    tfres = resolve_timeframe(frame.index.to_series(), explicit=timeframe, filename=p.name)
    if not tfres.consistent and tfres.note:
        notes.append(tfres.note)
    if timezone is None:
        notes.append("timezone UNKNOWN (MT4/MT5 server time not declared); "
                     "operations needing confirmed time sync must be blocked")
    if source_id is None:
        notes.append("source_id UNKNOWN (broker/feed/server not declared; "
                     "not inferred from the symbol suffix)")

    return ImportedSeries(
        frame=frame,
        extras=extras,
        symbol=sym,
        timeframe=tfres,
        timezone=timezone,
        tz_known=timezone is not None,
        source_path=str(p.resolve()),
        source_name=p.name,
        file_sha256=sha,
        encoding=encoding,
        delimiter=delim,
        had_header=had_header,
        header_tokens=header_tokens,
        imported_at=_dt.datetime.now(_dt.timezone.utc).isoformat(),
        raw_row_count=int(len(frame)),
        source_id=source_id or "unknown",
        source_rows=src_rows,
        original_columns=header_tokens,
        notes=notes,
    )


def _symbol_from_name(name: str) -> str:
    """Pull the broker symbol out of a filename like 'XAUUSD.smlM15.csv'.

    Strips a trailing timeframe tag so the symbol itself is left intact
    (including the broker suffix, e.g. 'XAUUSD.sml').
    """
    stem = name.rsplit(".csv", 1)[0].rsplit(".txt", 1)[0]
    # also drop an upload-style hash prefix 'abcd1234-'
    stem = re.sub(r"^[0-9a-f]{6,}-", "", stem)
    for tag in ("M1", "M5", "M15", "M30", "H1", "H4", "D1"):
        if stem.endswith(tag):
            return stem[: -len(tag)]
    stem = re.sub(r"\d+$", "", stem)  # trailing period-in-minutes tag
    return stem or name
