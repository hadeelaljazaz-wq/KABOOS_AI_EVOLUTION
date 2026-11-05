"""Timezone/instant handling for the gold data layer (spec review item 1).

MT4/MT5 bar timestamps are NAIVE wall-clock values in the broker's server
timezone. This module centralises the one correct way to reason about them:

  - ``validate_timezone`` rejects a bogus timezone string, so a value like
    ``"not/a_timezone"`` can never be accepted (and thus can never make
    ``tz_known`` True).
  - When the data timezone is KNOWN, bar times are localised to it and
    converted to UTC; a reference time is converted to the SAME instant in UTC,
    so "the same moment written in two zones" compares equal.
  - A NAIVE reference time, when the data timezone is known, is interpreted in
    the DATA's timezone — a single, documented policy, never a guess of UTC or
    the machine's local zone.
  - When the data timezone is UNKNOWN, a timezone-AWARE reference is REFUSED
    (we cannot know the broker offset), and naive comparisons are allowed only
    as explicitly provisional — never a confirmed temporal verification.
  - DST-ambiguous or DST-nonexistent local times are turned into NaT by an
    explicit policy and then surface through the quality layer, rather than
    being silently shifted.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

try:  # Python 3.9+
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore
    ZoneInfoNotFoundError = Exception  # type: ignore


class TimezoneError(ValueError):
    """Raised for an invalid timezone or an unsafe reference/timezone combo."""


def validate_timezone(tz: Optional[str]) -> Optional[str]:
    """Return ``tz`` if it is a real IANA timezone, else raise TimezoneError.

    ``None`` passes through (meaning: timezone deliberately left UNKNOWN).
    """
    if tz is None:
        return None
    if not isinstance(tz, str) or not tz.strip():
        raise TimezoneError("timezone must be a non-empty IANA name or None")
    if ZoneInfo is None:  # pragma: no cover
        raise TimezoneError("zoneinfo unavailable; cannot validate timezone")
    try:
        ZoneInfo(tz)
    except (ZoneInfoNotFoundError, ValueError, OSError) as exc:
        raise TimezoneError(f"invalid timezone {tz!r}: {exc}") from exc
    return tz


def index_to_utc(index: pd.DatetimeIndex, data_tz: str) -> pd.DatetimeIndex:
    """Localise naive broker-clock bar times to ``data_tz`` then convert to UTC.

    DST policy: ambiguous or nonexistent local times become NaT (explicit,
    not silently shifted) so the quality layer can report them.
    """
    validate_timezone(data_tz)
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        return idx.tz_convert("UTC")
    localized = idx.tz_localize(data_tz, ambiguous="NaT", nonexistent="NaT")
    return localized.tz_convert("UTC")


def comparison_index(index: pd.DatetimeIndex, data_tz: Optional[str],
                     tz_known: bool) -> pd.DatetimeIndex:
    """Bar times in the clock used for comparisons: UTC-aware if the timezone is
    known, else the naive broker clock (comparisons then only provisional)."""
    if tz_known and data_tz:
        return index_to_utc(index, data_tz)
    return pd.DatetimeIndex(index)


def coerce_reference(ref, data_tz: Optional[str], tz_known: bool) -> Optional[pd.Timestamp]:
    """Map a reference time into the comparison clock (see module docstring)."""
    if ref is None:
        return None
    ts = pd.Timestamp(ref)
    if tz_known and data_tz:
        if ts.tz is None:
            # documented policy: a naive reference is read in the DATA's timezone;
            # a nonexistent/ambiguous local reference is a hard, explicit error
            # (we never silently pick one of the two possible instants).
            try:
                ts = ts.tz_localize(data_tz, ambiguous="raise", nonexistent="raise")
            except Exception as exc:
                raise TimezoneError(
                    f"reference time {ref!r} does not exist or is ambiguous in "
                    f"{data_tz} (DST transition): {exc}") from exc
        return ts.tz_convert("UTC")
    # data timezone unknown
    if ts.tz is not None:
        raise TimezoneError(
            "a timezone-aware reference cannot be compared against data whose "
            "timezone is UNKNOWN; pass the data timezone explicitly, or give a "
            "naive reference expressed in the data's own clock"
        )
    return ts  # naive; any result using this is provisional
