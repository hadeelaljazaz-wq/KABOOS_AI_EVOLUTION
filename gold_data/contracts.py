"""Validated output contracts for the experts and the office (spec section 4).

These are DATA CONTRACTS only — no analysis, no discussion engine, no outcome
monitor is implemented here. They exist so that, later, each expert's report
and trade plan (and the office's combined plan) can be tracked SEPARATELY from
one another, with their results linked back by id.

Design rules honoured here:
  - no default win rate and no fixed default confidence are ever injected;
    ``uncertainty`` and outcome ``result`` are optional and left unset unless a
    real value is supplied;
  - a plan marked actionable (pending/active) must carry its core execution
    fields, and entry/stop/target ordering is validated against the plan's
    direction when the plan is complete;
  - "wait, no plan" (an expert recommending WAIT with no plan) is distinct from
    "a pending plan awaiting an activation condition";
  - every price level records its source; a revised expert report is a NEW
    report linked to the original, never an overwrite.
"""

from __future__ import annotations

import copy
import datetime as _dt
import math
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


def _check_price(value: Any, what: str) -> float:
    """Return value as float, or raise: must be a finite, positive, non-bool number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{what} must be a numeric price, got {value!r}")
    fv = float(value)
    if not math.isfinite(fv):
        raise ValueError(f"{what} must be finite (not NaN/Infinity), got {value!r}")
    if fv <= 0:
        raise ValueError(f"{what} must be positive, got {fv}")
    return fv


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _short_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _bad_nested(field_name: str, value: Any):
    raise ValueError(f"{field_name} entries must be the proper contract object or a dict "
                     f"to construct one, got {type(value).__name__}: {value!r}")


def _validate_iso(value: Any, what: str) -> str:
    try:
        _dt.datetime.fromisoformat(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{what} must be an ISO-8601 datetime, got {value!r}") from exc
    return str(value)


class Direction(str, Enum):
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


class MarketState(str, Enum):
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    RANGE = "range"
    VOLATILE = "volatile"
    UNCLEAR = "unclear"


class Recommendation(str, Enum):
    BUY = "buy"
    SELL = "sell"
    WAIT = "wait"


class ZoneKind(str, Enum):
    SUPPORT = "support"
    RESISTANCE = "resistance"
    SUPPLY = "supply"
    DEMAND = "demand"
    ORDER_BLOCK = "order_block"
    FVG = "fair_value_gap"
    PIVOT = "pivot"
    OTHER = "other"


class PlanStatus(str, Enum):
    CANDIDATE = "candidate"     # draft; may be incomplete
    PENDING = "pending"         # complete, awaiting an activation condition
    ACTIVE = "active"           # triggered / live
    CLOSED = "closed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    REJECTED = "rejected"


class MessageType(str, Enum):
    CLAIM = "claim"
    OBJECTION = "objection"
    QUESTION = "question"
    EVIDENCE = "evidence"
    PROPOSAL = "proposal"
    UPDATE = "update"
    CONCLUSION = "conclusion"


_ACTIONABLE = {PlanStatus.PENDING, PlanStatus.ACTIVE}


@dataclass
class PriceLevel:
    value: float
    source: str                        # which report/expert/method produced it
    kind: str = "level"                # 'entry'|'stop'|'target'|'zone_edge'|...
    label: Optional[str] = None

    def __post_init__(self):
        self.value = _check_price(self.value, "price level value")
        if not self.source or not str(self.source).strip():
            raise ValueError("every price level must record a non-empty source")


@dataclass
class Zone:
    kind: ZoneKind
    low: float
    high: float
    source: str
    invalidation: Optional[float] = None
    note: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.kind, str):
            self.kind = ZoneKind(self.kind)
        self.low = _check_price(self.low, "zone low")
        self.high = _check_price(self.high, "zone high")
        if self.invalidation is not None:
            self.invalidation = _check_price(self.invalidation, "zone invalidation")
        if self.low > self.high:
            raise ValueError(f"zone low ({self.low}) must be <= high ({self.high})")
        if not self.source or not str(self.source).strip():
            raise ValueError("every zone must record a non-empty source")


@dataclass
class ExpertReport:
    expert_id: str
    cycle_id: str
    snapshot_id: str
    specialty: str
    method_version: str
    direction: Direction
    market_state: MarketState
    timeframes_used: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    zones: list[Zone] = field(default_factory=list)
    levels: list[PriceLevel] = field(default_factory=list)
    recommendation: Optional[Recommendation] = None
    wait_reason: Optional[str] = None
    trade_plan_id: Optional[str] = None       # the expert's own independent plan, if any
    uncertainty: Optional[float] = None        # NO default; set only if real
    issued_at: str = field(default_factory=_now_iso)
    report_id: str = field(default_factory=lambda: _short_id("report"))
    revision: int = 1
    revises_report_id: Optional[str] = None    # link, so a revised report never overwrites

    def __post_init__(self):
        if isinstance(self.direction, str):
            self.direction = Direction(self.direction)
        if isinstance(self.market_state, str):
            self.market_state = MarketState(self.market_state)
        if self.recommendation is not None and isinstance(self.recommendation, str):
            self.recommendation = Recommendation(self.recommendation)
        # coerce + type-check nested values through their validated constructors
        self.zones = [z if isinstance(z, Zone) else Zone(**z) if isinstance(z, dict)
                      else _bad_nested("zones", z) for z in self.zones]
        self.levels = [lv if isinstance(lv, PriceLevel) else PriceLevel(**lv)
                       if isinstance(lv, dict) else _bad_nested("levels", lv)
                       for lv in self.levels]
        if not self.expert_id or not self.snapshot_id or not self.cycle_id:
            raise ValueError("expert_id, cycle_id and snapshot_id are required")
        if self.uncertainty is not None and not (0.0 <= float(self.uncertainty) <= 1.0):
            raise ValueError(f"uncertainty must be in [0,1] if set, got {self.uncertainty}")
        if self.revision < 1:
            raise ValueError("revision must be >= 1")
        if self.revises_report_id and self.revision < 2:
            raise ValueError("a report that revises another must have revision >= 2")
        # WAIT (or no recommendation) must explain why; a directional call must
        # agree with the stated direction.
        if self.recommendation in (None, Recommendation.WAIT):
            if not (self.wait_reason and self.wait_reason.strip()):
                raise ValueError("a WAIT / no-call report must give a wait_reason")
        elif self.recommendation == Recommendation.BUY and self.direction != Direction.LONG:
            raise ValueError("BUY recommendation requires direction=long")
        elif self.recommendation == Recommendation.SELL and self.direction != Direction.SHORT:
            raise ValueError("SELL recommendation requires direction=short")

    def revise(self, **changes: Any) -> "ExpertReport":
        """Return a NEW report derived from this one (original untouched)."""
        # DEEP copy of mutable/nested values so the revised report shares no
        # object with the original (editing one never touches the other).
        base = {
            "expert_id": self.expert_id, "cycle_id": self.cycle_id,
            "snapshot_id": self.snapshot_id, "specialty": self.specialty,
            "method_version": self.method_version, "direction": self.direction,
            "market_state": self.market_state,
            "timeframes_used": copy.deepcopy(self.timeframes_used),
            "evidence": copy.deepcopy(self.evidence), "zones": copy.deepcopy(self.zones),
            "levels": copy.deepcopy(self.levels), "recommendation": self.recommendation,
            "wait_reason": self.wait_reason, "trade_plan_id": self.trade_plan_id,
            "uncertainty": self.uncertainty,
        }
        base.update(copy.deepcopy(changes))
        base["revision"] = self.revision + 1
        base["revises_report_id"] = self.report_id
        return ExpertReport(**base)


@dataclass
class TradePlan:
    owner: str                         # an expert_id, or "office"
    owner_kind: str                    # 'expert' | 'office'
    cycle_id: str
    snapshot_id: str
    direction: Direction
    supporting_report_ids: list[str] = field(default_factory=list)
    entry: Optional[Zone] = None
    activation_condition: Optional[str] = None
    stop_loss: Optional[PriceLevel] = None
    targets: list[PriceLevel] = field(default_factory=list)
    cancel_conditions: list[str] = field(default_factory=list)
    expiry: Optional[str] = None
    status: PlanStatus = PlanStatus.CANDIDATE
    version: int = 1
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    plan_id: str = field(default_factory=lambda: _short_id("plan"))

    def __post_init__(self):
        if isinstance(self.direction, str):
            self.direction = Direction(self.direction)
        if isinstance(self.status, str):
            self.status = PlanStatus(self.status)
        # coerce + type-check nested price objects through validated constructors
        if isinstance(self.entry, dict):
            self.entry = Zone(**self.entry)
        elif self.entry is not None and not isinstance(self.entry, Zone):
            _bad_nested("entry", self.entry)
        if isinstance(self.stop_loss, dict):
            self.stop_loss = PriceLevel(**self.stop_loss)
        elif self.stop_loss is not None and not isinstance(self.stop_loss, PriceLevel):
            _bad_nested("stop_loss", self.stop_loss)
        self.targets = [t if isinstance(t, PriceLevel) else PriceLevel(**t)
                        if isinstance(t, dict) else _bad_nested("targets", t)
                        for t in self.targets]
        # Isolate nested price objects from any external reference the caller
        # may still hold and mutate after construction (defensive copy).
        self.entry = copy.deepcopy(self.entry)
        self.stop_loss = copy.deepcopy(self.stop_loss)
        self.targets = copy.deepcopy(self.targets)
        if self.owner_kind not in ("expert", "office"):
            raise ValueError("owner_kind must be 'expert' or 'office'")
        if not self.owner or not self.cycle_id or not self.snapshot_id:
            raise ValueError("owner, cycle_id and snapshot_id are required")
        if self.version < 1:
            raise ValueError("version must be >= 1")
        if self.expiry is not None:
            self.expiry = _validate_iso(self.expiry, "expiry")
        self.validate()

    def updated(self, **changes: Any) -> "TradePlan":
        """Return a NEW, re-validated plan with ``changes`` applied (version+1).

        Direct attribute mutation is NOT re-validated; go through this method so
        a modified plan cannot be used as valid without passing the checks again.
        """
        fields = {
            "owner": self.owner, "owner_kind": self.owner_kind, "cycle_id": self.cycle_id,
            "snapshot_id": self.snapshot_id, "direction": self.direction,
            "supporting_report_ids": copy.deepcopy(self.supporting_report_ids),
            "entry": copy.deepcopy(self.entry), "activation_condition": self.activation_condition,
            "stop_loss": copy.deepcopy(self.stop_loss), "targets": copy.deepcopy(self.targets),
            "cancel_conditions": copy.deepcopy(self.cancel_conditions), "expiry": self.expiry,
            "status": self.status, "plan_id": self.plan_id,
        }
        fields.update(copy.deepcopy(changes))
        fields["version"] = self.version + 1
        return TradePlan(**fields)

    @property
    def is_actionable(self) -> bool:
        """STATUS only — the plan is in a pending/active status.

        This is NOT a validity claim. A plan can carry an actionable status and
        still be invalid if a nested value was mutated after construction; use
        ``is_usable``/``validate`` for the real check before acting on a plan.
        """
        return self.status in _ACTIONABLE

    @property
    def is_usable(self) -> bool:
        """True iff the plan is in an actionable status AND re-validates now."""
        if self.status not in _ACTIONABLE:
            return False
        try:
            self.validate()
            return True
        except ValueError:
            return False

    def _revalidate_nested(self) -> None:
        """Re-run each nested price object's OWN invariants against its CURRENT
        values, so a mutation after construction (e.g. stop_loss.value = -1) is
        caught rather than trusted because the object once validated."""
        for obj in [self.entry, self.stop_loss, *self.targets]:
            if obj is not None:
                obj.__post_init__()  # re-checks finite/positive/order/source now

    def _has_core_execution_fields(self) -> bool:
        return (self.direction in (Direction.LONG, Direction.SHORT)
                and self.entry is not None
                and self.stop_loss is not None
                and len(self.targets) >= 1)

    def validate(self) -> None:
        # Re-check the CURRENT values of nested price objects first: a prior
        # successful construction does not prove they are still valid.
        self._revalidate_nested()
        # A pending plan is "awaiting activation" and MUST say on what; that is
        # what distinguishes it from a mere draft or from "wait, no plan".
        if self.status == PlanStatus.PENDING and not (
            self.activation_condition and self.activation_condition.strip()
        ):
            raise ValueError(
                "a PENDING plan must carry an activation_condition (otherwise it is a "
                "CANDIDATE draft, not a plan awaiting activation)"
            )

        if self.is_actionable:
            if not self._has_core_execution_fields():
                raise ValueError(
                    f"plan marked {self.status.value} but missing core execution fields "
                    "(direction long/short, entry zone, stop_loss, >=1 target)"
                )
            self._check_level_ordering()
        else:
            # Even a draft, if it has a full set of levels, must not be internally
            # contradictory.
            if self._has_core_execution_fields():
                self._check_level_ordering()

    def _check_level_ordering(self) -> None:
        entry_low, entry_high = self.entry.low, self.entry.high
        stop = self.stop_loss.value
        tvals = [t.value for t in self.targets]
        if self.direction == Direction.LONG:
            if not (stop < entry_low):
                raise ValueError(f"long plan: stop ({stop}) must be below entry low ({entry_low})")
            if not all(t > entry_high for t in tvals):
                raise ValueError("long plan: every target must be above the entry high")
            if tvals != sorted(tvals):
                raise ValueError("long plan: targets must be in ascending order")
        elif self.direction == Direction.SHORT:
            if not (stop > entry_high):
                raise ValueError(f"short plan: stop ({stop}) must be above entry high ({entry_high})")
            if not all(t < entry_low for t in tvals):
                raise ValueError("short plan: every target must be below the entry low")
            if tvals != sorted(tvals, reverse=True):
                raise ValueError("short plan: targets must be in descending order")
        else:
            raise ValueError("a neutral plan cannot be actionable/complete")


@dataclass
class DiscussionMessage:
    """Contract only — the discussion ENGINE is not built here (spec section 4)."""
    cycle_id: str
    sender: str
    message_type: MessageType
    claim: Optional[str] = None
    related_report_id: Optional[str] = None
    evidence_refs: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=_now_iso)
    message_id: str = field(default_factory=lambda: _short_id("msg"))

    def __post_init__(self):
        if isinstance(self.message_type, str):
            self.message_type = MessageType(self.message_type)
        if not self.sender or not self.cycle_id:
            raise ValueError("sender and cycle_id are required")


@dataclass
class Outcome:
    """Contract only — the outcome MONITOR is not built here (spec section 4).

    Links a result back to a specific plan + version + owner. No default win
    rate or confidence is stored; ``result`` stays None until a real outcome is
    recorded.
    """
    plan_id: str
    plan_version: int
    owner: str
    events: list[dict[str, Any]] = field(default_factory=list)  # {time,type,price,...}
    costs: Optional[dict[str, Any]] = None
    result: Optional[dict[str, Any]] = None                     # e.g. {"r_multiple":..} — no default
    ambiguity_status: str = "unknown"                           # 'clear'|'ambiguous'|'unknown'
    created_at: str = field(default_factory=_now_iso)
    outcome_id: str = field(default_factory=lambda: _short_id("outcome"))

    def __post_init__(self):
        if not self.plan_id or not self.owner:
            raise ValueError("plan_id and owner are required")
        if self.plan_version < 1:
            raise ValueError("plan_version must be >= 1")
        if self.ambiguity_status not in ("clear", "ambiguous", "unknown"):
            raise ValueError("ambiguity_status must be 'clear', 'ambiguous' or 'unknown'")
