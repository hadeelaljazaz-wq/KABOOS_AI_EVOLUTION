"""Durable, structured storage for expert-desk cycles (reuses SQLite).

One SQLite file holds cycles, each expert's ORIGINAL report, its own plan
(separate per expert), its screen bundle, run mode, and success/abstain state.
Re-saving the same cycle is idempotent: the original per-(cycle, expert) rows
are kept and never silently duplicated. This complements — does not replace —
the collective_memory store, which holds each expert's long-term memory.
"""

from __future__ import annotations

import datetime as _dt
import enum
import json
import sqlite3
import threading
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Optional

from gold_data.contracts import ExpertReport, TradePlan

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cycles (
    cycle_id TEXT PRIMARY KEY,
    snapshot_id TEXT NOT NULL,
    symbol TEXT,
    reference_time TEXT,
    run_mode TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reports (
    cycle_id TEXT NOT NULL,
    expert_id TEXT NOT NULL,
    report_json TEXT NOT NULL,
    plan_json TEXT,
    screen_json TEXT,
    analysis_ar TEXT,
    run_mode TEXT,
    success INTEGER,
    abstained INTEGER,
    abstain_reason TEXT,
    specialty TEXT,
    method_version TEXT,
    timeframes_used TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY (cycle_id, expert_id)
);
"""


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


class ExpertDeskStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False,
                                     isolation_level=None, timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.executescript(_SCHEMA)
        self._lock = threading.RLock()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def cycle_exists(self, cycle_id: str) -> bool:
        with self._lock:
            return self._conn.execute("SELECT 1 FROM cycles WHERE cycle_id=?",
                                      (cycle_id,)).fetchone() is not None

    def save_cycle(self, cycle_id: str, snapshot_id: str, symbol: str,
                   reference_time: Optional[str], run_mode: str, results: list) -> bool:
        """Persist a cycle and its experts. Idempotent: if the cycle already
        exists, nothing is overwritten and no duplicate reports are created.
        Returns True if this call inserted a new cycle, False if it already existed."""
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                existed = self._conn.execute("SELECT 1 FROM cycles WHERE cycle_id=?",
                                             (cycle_id,)).fetchone() is not None
                if existed:
                    self._conn.execute("ROLLBACK")
                    return False
                self._conn.execute(
                    "INSERT INTO cycles (cycle_id, snapshot_id, symbol, reference_time, run_mode, created_at)"
                    " VALUES (?,?,?,?,?,?)",
                    (cycle_id, snapshot_id, symbol, reference_time, run_mode, _now()))
                for r in results:
                    self._conn.execute(
                        "INSERT OR IGNORE INTO reports (cycle_id, expert_id, report_json, plan_json,"
                        " screen_json, analysis_ar, run_mode, success, abstained, abstain_reason,"
                        " specialty, method_version, timeframes_used, created_at)"
                        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (cycle_id, r.expert_id,
                         json.dumps(_to_jsonable(r.report)),
                         json.dumps(_to_jsonable(r.plan)) if r.plan else None,
                         json.dumps(_to_jsonable(r.screen)),
                         r.analysis_ar, r.run_mode, int(r.success), int(r.abstained),
                         r.abstain_reason, r.specialty, r.method_version,
                         json.dumps(r.timeframes_used), _now()))
                self._conn.execute("COMMIT")
                return True
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise

    def load_cycle(self, cycle_id: str) -> Optional[dict]:
        with self._lock:
            crow = self._conn.execute("SELECT * FROM cycles WHERE cycle_id=?",
                                      (cycle_id,)).fetchone()
            if crow is None:
                return None
            rows = self._conn.execute("SELECT * FROM reports WHERE cycle_id=? ORDER BY expert_id",
                                      (cycle_id,)).fetchall()
        experts = []
        for row in rows:
            report = ExpertReport(**json.loads(row["report_json"]))
            plan = TradePlan(**json.loads(row["plan_json"])) if row["plan_json"] else None
            experts.append({
                "expert_id": row["expert_id"], "specialty": row["specialty"],
                "method_version": row["method_version"],
                "timeframes_used": json.loads(row["timeframes_used"]),
                "report": report, "plan": plan, "analysis_ar": row["analysis_ar"],
                "run_mode": row["run_mode"], "success": bool(row["success"]),
                "abstained": bool(row["abstained"]), "abstain_reason": row["abstain_reason"],
                "screen": json.loads(row["screen_json"]),
            })
        return {"cycle_id": crow["cycle_id"], "snapshot_id": crow["snapshot_id"],
                "symbol": crow["symbol"], "reference_time": crow["reference_time"],
                "run_mode": crow["run_mode"], "created_at": crow["created_at"], "experts": experts}

    def list_cycles(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM cycles ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]
