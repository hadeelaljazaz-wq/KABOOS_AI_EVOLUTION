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

OHLC = ("open", "high", "low", "close")

_DATE_RE = re.compile(r"^\d{4}[.\-/]\d{1,2}[.\-/]\d{1,2}$")
_TIME_RE = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?$")
_DATETIME_RE = re.compile(r"^\d{4}[.\-/]\d{1,2}[.\-/]\d{1,2}[ T]\d{1,2}:\d{2}(:\d{2})?$")

_DT_FORMATS = (
    "%Y.%m.%d %H:%M", "%Y.%m.%d %H:%M:%S", "%Y.%m.%d",
    "%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d",
    "%Y/%m/%d %H:%M", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d",
)

_HEADER_TOKENS = {"date", "time", "open", "high", "low", "close", "volume", "vol",
                  "tickvol", "<date>", "<time>", "<open>", "<high>", "<low>",
                  "<close>", "<tickvol>", "<vol>", "<spread>"}


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
    notes: list[str] = field(default_factory=list)

    @property
    def canonical_symbol(self) -> str:
        return self.symbol.canonical

    def source_fingerprint(self) -> dict[str, object]:
        """Everything needed to identify this import's origin (no price data)."""
        return {
            "source_name": self.source_name,
            "file_sha256": self.file_sha256,
            "symbol_original": self.symbol.original,
            "symbol_canonical": self.symbol.canonical,
            "instrument_kind": self.symbol.kind.value,
            "timeframe": self.timeframe.timeframe.value,
            "timeframe_source": self.timeframe.source,
            "encoding": self.encoding,
            "delimiter": "tab" if self.delimiter == "\t" else self.delimiter,
            "had_header": self.had_header,
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


def _looks_like_header(fields: list[str]) -> bool:
    joined = [f.strip().lower() for f in fields]
    if any(tok in _HEADER_TOKENS for tok in joined):
        return True
    # a data row starts with a date/datetime; a header does not
    return not (_DATE_RE.match(fields[0].strip()) or _DATETIME_RE.match(fields[0].strip()))


def _layout(fields: list[str]) -> tuple[str, int]:
    """Return (datetime_mode, price_start_index) for a DATA row, or raise."""
    f0 = fields[0].strip()
    if _DATETIME_RE.match(f0):
        return "combined", 1
    if _DATE_RE.match(f0) and len(fields) > 1 and _TIME_RE.match(fields[1].strip()):
        return "split", 2
    if _DATE_RE.match(f0):
        return "date_only", 1
    raise UnsupportedFormatError(
        "first field is not a recognised date/datetime: "
        f"{f0!r}. Supported: 'YYYY.MM.DD', 'YYYY.MM.DD HH:MM[:SS]', or "
        "a 'Date','Time' pair."
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
) -> ImportedSeries:
    """Read one MT4/MT5 CSV export into an :class:`ImportedSeries`.

    ``symbol`` defaults to the stem of the filename (broker symbol kept, then
    normalised). ``timeframe`` may be given explicitly; otherwise it is
    resolved from a filename hint and/or the data's own spacing and verified.
    ``timezone`` is recorded as-is; omit it to mark the timezone UNKNOWN.
    """
    p = Path(path)
    raw = p.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    encoding = _detect_encoding(raw)
    text = raw.decode(encoding, errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    text = text.lstrip("﻿")
    lines = [ln for ln in text.split("\n")]
    nonempty = [ln for ln in lines if ln.strip()]
    if not nonempty:
        raise UnsupportedFormatError(f"file {p.name} is empty")

    delim = _detect_delimiter(nonempty)
    first_fields = nonempty[0].split(delim)
    had_header = _looks_like_header(first_fields)
    header_tokens = [f.strip() for f in first_fields] if had_header else []
    data_lines = nonempty[1:] if had_header else nonempty
    if not data_lines:
        raise UnsupportedFormatError(f"file {p.name} has a header but no data rows")

    dt_mode, price_start = _layout(data_lines[0].split(delim))

    # Heavy parse via pandas on the data portion only.
    buf = io.StringIO("\n".join(data_lines))
    df = pd.read_csv(buf, header=None, sep=delim, dtype=str,
                     skip_blank_lines=True, engine="c")
    df = df.dropna(how="all")
    ncols = df.shape[1]
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
    times = _parse_datetimes(dt_str)

    o = pd.to_numeric(df.iloc[:, price_start + 0], errors="coerce")
    h = pd.to_numeric(df.iloc[:, price_start + 1], errors="coerce")
    low = pd.to_numeric(df.iloc[:, price_start + 2], errors="coerce")
    c = pd.to_numeric(df.iloc[:, price_start + 3], errors="coerce")
    vol_idx = price_start + 4
    notes: list[str] = []
    if vol_idx < ncols:
        volume = pd.to_numeric(df.iloc[:, vol_idx], errors="coerce")
        notes.append("volume column is broker tick volume (not exchange/contract volume)")
    else:
        volume = pd.Series([pd.NA] * len(df), index=df.index)
        notes.append("no volume column present in source")

    frame = pd.DataFrame({"open": o, "high": h, "low": low, "close": c, "volume": volume})
    frame.index = pd.DatetimeIndex(times, name="time")

    # Preserve any trailing columns verbatim; we do NOT guess their meaning.
    extra_cols = list(range(vol_idx + 1, ncols))
    if extra_cols:
        extras = df.iloc[:, extra_cols].copy()
        extras.columns = [f"extra_{i}" for i in range(1, len(extra_cols) + 1)]
        extras.index = frame.index
        notes.append(
            f"{len(extra_cols)} trailing column(s) preserved as extra_* "
            "(broker-specific; meaning not declared in file — possibly real "
            "volume or spread)"
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
