"""Collective Intelligence Memory for KABOOS AI EVOLUTION.

SQLite-backed so the whole system's experience survives a restart (per spec
section 6). Four kinds of state live here:

  - agent_memory:         each agent's own long-term notes
  - discussions/messages: real inter-agent debate (claims, objections,
                           evidence, questions) tied to a topic, resumable
                           after a restart (each discussion carries a goal
                           and an open/closed status)
  - knowledge:            the collective memory. A row starts as a
                           "hypothesis" and can only move to "confirmed" or
                           "rejected" through a validated state transition
                           with real, well-formed evidence attached — an
                           unproven conclusion can never silently become a
                           confirmed fact (spec section 6, last line).

Concurrency: one SQLite file is shared by every agent. The connection runs
in WAL mode with a busy timeout, and every write is serialised through a
re-entrant lock and wrapped in a transaction, so several agents acting at
once cannot corrupt state or half-apply a status transition.
"""

from __future__ import annotations

import datetime
import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT NOT NULL,
    category TEXT NOT NULL,
    content TEXT NOT NULL,
    confidence REAL,
    evidence TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS discussions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT NOT NULL,
    goal TEXT,
    status TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'closed')),
    started_by TEXT NOT NULL,
    created_at TEXT NOT NULL,
    closed_at TEXT
);

CREATE TABLE IF NOT EXISTS discussion_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    discussion_id INTEGER NOT NULL REFERENCES discussions(id),
    agent_id TEXT NOT NULL,
    message_type TEXT NOT NULL CHECK (message_type IN
        ('claim', 'objection', 'question', 'evidence', 'proposal', 'update', 'conclusion')),
    message TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS knowledge (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT NOT NULL,
    statement TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('hypothesis', 'confirmed', 'rejected')),
    created_by TEXT NOT NULL,
    evidence TEXT,
    validated_by TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""

# Message types an agent may post into a discussion.
MESSAGE_TYPES = frozenset(
    {"claim", "objection", "question", "evidence", "proposal", "update", "conclusion"}
)

# A valid evidence payload must reference at least one of these backing
# keys. This checks that evidence is *well-formed and points at something*
# (a backtest, a dataset, a statistic, a citation) — it does NOT and cannot
# verify that the evidence is true. Truth is established by the experiment
# the evidence points to, not by this guard.
_EVIDENCE_BACKING_KEYS = frozenset(
    {
        "backtest_id",
        "experiment_id",
        "source",
        "data_source",
        "sample_size",
        "p_value",
        "citation",
        "url",
        "metrics",
    }
)


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def validate_evidence(evidence: Any) -> None:
    """Raise ValueError unless ``evidence`` is a well-formed backing payload.

    Requirements:
      - a non-empty dict
      - at least one recognised backing key with a non-empty value
      - if ``sample_size`` is given, an int >= 2
      - if ``p_value`` is given, a float in [0, 1]

    This does not prove the evidence is correct; it rejects empty or
    hand-wavy payloads (e.g. ``{"note": "looks good"}``) that carry no
    pointer to a reproducible experiment, dataset, or statistic.
    """
    if not isinstance(evidence, dict) or not evidence:
        raise ValueError("evidence must be a non-empty dict")
    present = _EVIDENCE_BACKING_KEYS & set(evidence)
    if not present:
        raise ValueError(
            "evidence must reference at least one backing key "
            f"({', '.join(sorted(_EVIDENCE_BACKING_KEYS))}); got keys {sorted(evidence)}"
        )
    if not any(evidence.get(k) not in (None, "", [], {}) for k in present):
        raise ValueError("every backing key in evidence is empty")
    if "sample_size" in evidence:
        ss = evidence["sample_size"]
        if not isinstance(ss, int) or isinstance(ss, bool) or ss < 2:
            raise ValueError(f"sample_size must be an int >= 2, got {ss!r}")
    if "p_value" in evidence:
        pv = evidence["p_value"]
        if not isinstance(pv, (int, float)) or isinstance(pv, bool) or not (0.0 <= pv <= 1.0):
            raise ValueError(f"p_value must be a number in [0, 1], got {pv!r}")


@dataclass
class KnowledgeItem:
    id: int
    topic: str
    statement: str
    status: str
    created_by: str
    evidence: Optional[str]
    validated_by: Optional[str]
    created_at: str
    updated_at: str


class CollectiveMemory:
    """One SQLite file shared by every agent in the KABOOS research community."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            self.db_path, check_same_thread=False, isolation_level=None, timeout=30.0
        )
        self._conn.row_factory = sqlite3.Row
        # Concurrency / integrity pragmas. WAL lets readers and a writer
        # coexist; busy_timeout rides out a lock instead of erroring; FK
        # enforcement makes the discussion_id reference real.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        # Serialises writes and multi-statement transactions across agent threads.
        self._lock = threading.RLock()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ---- per-agent long-term memory ----------------------------------

    def remember(
        self,
        agent_id: str,
        category: str,
        content: str,
        confidence: Optional[float] = None,
        evidence: Optional[dict[str, Any]] = None,
    ) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO agent_memory (agent_id, category, content, confidence, evidence, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (agent_id, category, content, confidence,
                 json.dumps(evidence) if evidence else None, _now()),
            )
            return cur.lastrowid

    def recall(self, agent_id: str, category: Optional[str] = None, limit: int = 50) -> list[sqlite3.Row]:
        if category:
            q = ("SELECT * FROM agent_memory WHERE agent_id=? AND category=?"
                 " ORDER BY id DESC LIMIT ?")
            params: tuple[Any, ...] = (agent_id, category, limit)
        else:
            q = "SELECT * FROM agent_memory WHERE agent_id=? ORDER BY id DESC LIMIT ?"
            params = (agent_id, limit)
        with self._lock:
            return self._conn.execute(q, params).fetchall()

    # ---- real collaboration: discussions -----------------------------

    def start_discussion(self, topic: str, started_by: str, goal: Optional[str] = None) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO discussions (topic, goal, status, started_by, created_at)"
                " VALUES (?, ?, 'open', ?, ?)",
                (topic, goal, started_by, _now()),
            )
            return cur.lastrowid

    def get_discussion_meta(self, discussion_id: int) -> Optional[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM discussions WHERE id=?", (discussion_id,)
            ).fetchone()

    def post_message(self, discussion_id: int, agent_id: str, message_type: str, message: str) -> int:
        if message_type not in MESSAGE_TYPES:
            raise ValueError(
                f"unknown message_type {message_type!r}; valid: {sorted(MESSAGE_TYPES)}"
            )
        with self._lock:
            meta = self._conn.execute(
                "SELECT status FROM discussions WHERE id=?", (discussion_id,)
            ).fetchone()
            if meta is None:
                raise ValueError(f"no discussion with id={discussion_id}")
            if meta["status"] == "closed":
                raise ValueError(f"discussion id={discussion_id} is closed; reopen it to post")
            cur = self._conn.execute(
                "INSERT INTO discussion_messages (discussion_id, agent_id, message_type, message, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (discussion_id, agent_id, message_type, message, _now()),
            )
            return cur.lastrowid

    def get_discussion(self, discussion_id: int) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM discussion_messages WHERE discussion_id=? ORDER BY id ASC",
                (discussion_id,),
            ).fetchall()

    def close_discussion(self, discussion_id: int) -> None:
        with self._lock:
            meta = self._conn.execute(
                "SELECT status FROM discussions WHERE id=?", (discussion_id,)
            ).fetchone()
            if meta is None:
                raise ValueError(f"no discussion with id={discussion_id}")
            self._conn.execute(
                "UPDATE discussions SET status='closed', closed_at=? WHERE id=?",
                (_now(), discussion_id),
            )

    def reopen_discussion(self, discussion_id: int) -> None:
        with self._lock:
            meta = self._conn.execute(
                "SELECT status FROM discussions WHERE id=?", (discussion_id,)
            ).fetchone()
            if meta is None:
                raise ValueError(f"no discussion with id={discussion_id}")
            self._conn.execute(
                "UPDATE discussions SET status='open', closed_at=NULL WHERE id=?",
                (discussion_id,),
            )

    def open_discussions(self) -> list[sqlite3.Row]:
        """Resumable discussions still awaiting a conclusion (survives restart)."""
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM discussions WHERE status='open' ORDER BY id ASC"
            ).fetchall()

    # ---- collective knowledge: hypothesis -> confirmed/rejected -------

    def propose_hypothesis(self, topic: str, statement: str, created_by: str) -> int:
        now = _now()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO knowledge (topic, statement, status, created_by, evidence,"
                " validated_by, created_at, updated_at) VALUES (?, ?, 'hypothesis', ?, NULL, NULL, ?, ?)",
                (topic, statement, created_by, now, now),
            )
            return cur.lastrowid

    def _require_pending_hypothesis(self, knowledge_id: int) -> sqlite3.Row:
        row = self._conn.execute(
            "SELECT status FROM knowledge WHERE id=?", (knowledge_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"no knowledge row with id={knowledge_id}")
        if row["status"] != "hypothesis":
            raise ValueError(
                f"knowledge id={knowledge_id} is already '{row['status']}'; only a pending "
                "hypothesis can be confirmed or rejected (no re-deciding a settled item)"
            )
        return row

    def confirm_hypothesis(self, knowledge_id: int, evidence: dict[str, Any], validated_by: str) -> None:
        """Promote a pending hypothesis to a confirmed fact.

        Requires well-formed backing evidence (see ``validate_evidence``) and
        a pending-hypothesis source state. Raises ValueError otherwise. The
        read of current status and the write happen under one lock, so two
        agents cannot both win a confirm/reject race on the same row.
        """
        validate_evidence(evidence)
        with self._lock:
            self._require_pending_hypothesis(knowledge_id)
            self._conn.execute(
                "UPDATE knowledge SET status='confirmed', evidence=?, validated_by=?, updated_at=? WHERE id=?",
                (json.dumps(evidence), validated_by, _now(), knowledge_id),
            )

    def reject_hypothesis(self, knowledge_id: int, reason: str, validated_by: str) -> None:
        """Reject a pending hypothesis. Only a pending hypothesis may be rejected.

        A non-empty reason is required, and an already-confirmed or
        already-rejected row cannot be flipped (same transition guard as
        ``confirm_hypothesis``).
        """
        if not reason or not reason.strip():
            raise ValueError("a non-empty reason is required to reject a hypothesis")
        with self._lock:
            self._require_pending_hypothesis(knowledge_id)
            self._conn.execute(
                "UPDATE knowledge SET status='rejected', evidence=?, validated_by=?, updated_at=? WHERE id=?",
                (json.dumps({"reason": reason}), validated_by, _now(), knowledge_id),
            )

    def query_knowledge(self, topic: Optional[str] = None, status: Optional[str] = None) -> list[KnowledgeItem]:
        q = "SELECT * FROM knowledge WHERE 1=1"
        params: list[Any] = []
        if topic:
            q += " AND topic=?"
            params.append(topic)
        if status:
            q += " AND status=?"
            params.append(status)
        q += " ORDER BY id DESC"
        with self._lock:
            rows = self._conn.execute(q, params).fetchall()
        return [KnowledgeItem(**{k: row[k] for k in row.keys()}) for row in rows]
