"""Collective Intelligence Memory for KABOOS AI EVOLUTION.

SQLite-backed so the whole system's experience survives a restart (per spec
section 6). Three kinds of state live here:

  - agent_memory:         each agent's own long-term notes
  - discussions/messages: real inter-agent debate (claims, objections,
                           evidence, questions) tied to a topic
  - knowledge:            the collective memory. A row starts as a
                           "hypothesis" and can only move to "confirmed" or
                           "rejected" when evidence is attached — an
                           unproven conclusion can never silently become a
                           confirmed fact (spec section 6, last line).
"""

from __future__ import annotations

import datetime
import json
import sqlite3
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
    started_by TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS discussion_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    discussion_id INTEGER NOT NULL REFERENCES discussions(id),
    agent_id TEXT NOT NULL,
    message_type TEXT NOT NULL CHECK (message_type IN
        ('claim', 'objection', 'question', 'evidence', 'proposal', 'update')),
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


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


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
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
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
        cur = self._conn.execute(
            "INSERT INTO agent_memory (agent_id, category, content, confidence, evidence, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (agent_id, category, content, confidence,
             json.dumps(evidence) if evidence else None, _now()),
        )
        self._conn.commit()
        return cur.lastrowid

    def recall(self, agent_id: str, category: Optional[str] = None, limit: int = 50) -> list[sqlite3.Row]:
        if category:
            q = ("SELECT * FROM agent_memory WHERE agent_id=? AND category=?"
                 " ORDER BY id DESC LIMIT ?")
            params = (agent_id, category, limit)
        else:
            q = "SELECT * FROM agent_memory WHERE agent_id=? ORDER BY id DESC LIMIT ?"
            params = (agent_id, limit)
        return self._conn.execute(q, params).fetchall()

    # ---- real collaboration: discussions -----------------------------

    def start_discussion(self, topic: str, started_by: str) -> int:
        cur = self._conn.execute(
            "INSERT INTO discussions (topic, started_by, created_at) VALUES (?, ?, ?)",
            (topic, started_by, _now()),
        )
        self._conn.commit()
        return cur.lastrowid

    def post_message(self, discussion_id: int, agent_id: str, message_type: str, message: str) -> int:
        cur = self._conn.execute(
            "INSERT INTO discussion_messages (discussion_id, agent_id, message_type, message, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (discussion_id, agent_id, message_type, message, _now()),
        )
        self._conn.commit()
        return cur.lastrowid

    def get_discussion(self, discussion_id: int) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM discussion_messages WHERE discussion_id=? ORDER BY id ASC",
            (discussion_id,),
        ).fetchall()

    # ---- collective knowledge: hypothesis -> confirmed/rejected -------

    def propose_hypothesis(self, topic: str, statement: str, created_by: str) -> int:
        now = _now()
        cur = self._conn.execute(
            "INSERT INTO knowledge (topic, statement, status, created_by, evidence,"
            " validated_by, created_at, updated_at) VALUES (?, ?, 'hypothesis', ?, NULL, NULL, ?, ?)",
            (topic, statement, created_by, now, now),
        )
        self._conn.commit()
        return cur.lastrowid

    def confirm_hypothesis(self, knowledge_id: int, evidence: dict[str, Any], validated_by: str) -> None:
        """Promote a hypothesis to a confirmed fact. Requires real evidence.

        Raises ValueError if ``evidence`` is empty or the row isn't a pending
        hypothesis — a conclusion is never allowed to become "confirmed"
        without something to point at.
        """
        if not evidence:
            raise ValueError("cannot confirm a hypothesis without evidence")
        row = self._conn.execute("SELECT status FROM knowledge WHERE id=?", (knowledge_id,)).fetchone()
        if row is None:
            raise ValueError(f"no knowledge row with id={knowledge_id}")
        if row["status"] != "hypothesis":
            raise ValueError(f"knowledge id={knowledge_id} is already '{row['status']}', not a pending hypothesis")
        self._conn.execute(
            "UPDATE knowledge SET status='confirmed', evidence=?, validated_by=?, updated_at=? WHERE id=?",
            (json.dumps(evidence), validated_by, _now(), knowledge_id),
        )
        self._conn.commit()

    def reject_hypothesis(self, knowledge_id: int, reason: str, validated_by: str) -> None:
        row = self._conn.execute("SELECT status FROM knowledge WHERE id=?", (knowledge_id,)).fetchone()
        if row is None:
            raise ValueError(f"no knowledge row with id={knowledge_id}")
        self._conn.execute(
            "UPDATE knowledge SET status='rejected', evidence=?, validated_by=?, updated_at=? WHERE id=?",
            (json.dumps({"reason": reason}), validated_by, _now(), knowledge_id),
        )
        self._conn.commit()

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
        rows = self._conn.execute(q, params).fetchall()
        return [KnowledgeItem(**{k: row[k] for k in row.keys()}) for row in rows]
