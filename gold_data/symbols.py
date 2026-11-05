"""Symbol normalisation for the gold data layer.

Brokers decorate the base symbol with their own suffixes: ``XAUUSDm``,
``XAUUSD.sml``, ``XAUUSD.r``, ``XAUUSD-ECN``. We normalise to a canonical base
by a single DECLARED rule and never throw the original away.

DECLARED RULE
-------------
Uppercase the raw symbol, strip a leading ``$``/``#`` decoration, then find the
longest known canonical base in ``CANONICAL_BASES`` that the uppercased raw
symbol STARTS WITH. That base is the canonical symbol; everything after it is a
broker-specific suffix, preserved verbatim as ``broker_suffix``. If no known
base is a prefix, the symbol is NOT normalised: ``canonical`` is left equal to
the (uppercased) raw symbol and ``known`` is False, so a caller can refuse it
rather than silently mislabel it.

INSTRUMENT IDENTITY
-------------------
``XAUUSD`` (OTC spot/CFD gold vs USD) and ``GC=F`` (COMEX gold *futures*) are
DIFFERENT instruments. This module never maps one onto the other, and
``same_instrument`` / the snapshot layer use ``InstrumentKind`` to keep them
apart. Substituting futures for spot (or vice versa) is a bug, not a fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class InstrumentKind(str, Enum):
    SPOT = "spot"              # OTC spot / CFD (broker feed), e.g. XAUUSD
    FUTURES = "futures"        # exchange futures, e.g. GC=F
    INDEX = "index"            # index level
    CRYPTO = "crypto"          # spot crypto
    FX = "fx"                  # FX pair
    UNKNOWN = "unknown"


# Known canonical bases, longest-first matching handled in code. Each maps to
# the instrument kind of the *spot/cash* reading of that ticker as a broker
# would export it. Futures tickers (with '=F') are separate entries.
CANONICAL_BASES: dict[str, InstrumentKind] = {
    "XAUUSD": InstrumentKind.SPOT,
    "XAGUSD": InstrumentKind.SPOT,
    "XAUEUR": InstrumentKind.SPOT,
    "XPTUSD": InstrumentKind.SPOT,
    "XPDUSD": InstrumentKind.SPOT,
    "EURUSD": InstrumentKind.FX,
    "GBPUSD": InstrumentKind.FX,
    "USDJPY": InstrumentKind.FX,
    "BTCUSD": InstrumentKind.CRYPTO,
    "ETHUSD": InstrumentKind.CRYPTO,
    "USOIL": InstrumentKind.SPOT,
    # Exchange futures kept explicitly distinct from their spot cousins:
    "GC=F": InstrumentKind.FUTURES,
    "SI=F": InstrumentKind.FUTURES,
    "CL=F": InstrumentKind.FUTURES,
}


@dataclass(frozen=True)
class NormalizedSymbol:
    canonical: str
    original: str
    broker_suffix: str
    kind: InstrumentKind
    known: bool

    @property
    def is_spot(self) -> bool:
        return self.kind == InstrumentKind.SPOT

    @property
    def is_futures(self) -> bool:
        return self.kind == InstrumentKind.FUTURES

    def provenance(self) -> str:
        if not self.known:
            return f"UNKNOWN instrument '{self.original}' (not normalised; treat with care)"
        base = f"{self.canonical} ({self.kind.value})"
        if self.is_spot:
            base += " — OTC spot/CFD from a broker feed, NOT exchange futures"
        if self.is_futures:
            base += " — exchange futures, NOT spot"
        if self.broker_suffix:
            base += f"; broker symbol '{self.original}' (suffix '{self.broker_suffix}')"
        return base


def normalize_symbol(raw: str) -> NormalizedSymbol:
    """Apply the declared normalisation rule. Never discards the original."""
    if raw is None or not str(raw).strip():
        raise ValueError("symbol must be a non-empty string")
    original = str(raw).strip()
    up = original.upper().lstrip("$#").strip()

    # longest known base that is a prefix of the uppercased symbol
    match = ""
    for base in CANONICAL_BASES:
        if up.startswith(base) and len(base) > len(match):
            match = base
    if match:
        suffix = original[len(match):] if original.upper().startswith(match) else up[len(match):]
        return NormalizedSymbol(
            canonical=match,
            original=original,
            broker_suffix=suffix,
            kind=CANONICAL_BASES[match],
            known=True,
        )
    # not recognised — keep as-is, flag unknown (caller decides whether to use)
    return NormalizedSymbol(
        canonical=up, original=original, broker_suffix="",
        kind=InstrumentKind.UNKNOWN, known=False,
    )


def same_instrument(a: NormalizedSymbol, b: NormalizedSymbol) -> bool:
    """True only if canonical base AND instrument kind match.

    XAUUSD(spot) and GC=F(futures) are never the same instrument.
    """
    return a.canonical == b.canonical and a.kind == b.kind
