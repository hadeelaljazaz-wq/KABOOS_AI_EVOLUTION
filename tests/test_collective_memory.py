import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import threading

import pytest

from collective_memory import CollectiveMemory, validate_evidence


@pytest.fixture()
def mem(tmp_path):
    m = CollectiveMemory(tmp_path / "test_memory.db")
    yield m
    m.close()


def test_agent_memory_roundtrip(mem):
    mem.remember("analyst_technical", "pattern", "Gold broke above 200-day MA with rising volume", confidence=0.7)
    rows = mem.recall("analyst_technical", category="pattern")
    assert len(rows) == 1
    assert "200-day MA" in rows[0]["content"]


def test_discussion_is_real_back_and_forth(mem):
    disc_id = mem.start_discussion("Is the gold rally driven by real yields or by USD weakness?", "researcher_bull")
    mem.post_message(disc_id, "researcher_bull", "claim", "USD weakness explains most of the move")
    mem.post_message(disc_id, "researcher_bear", "objection", "Real yields fell more than DXY over the same window")
    mem.post_message(disc_id, "researcher_bull", "evidence", "10y real yield -40bps vs DXY -1.2% over 20 sessions")
    msgs = mem.get_discussion(disc_id)
    assert [m["message_type"] for m in msgs] == ["claim", "objection", "evidence"]


def test_hypothesis_cannot_be_confirmed_without_evidence(mem):
    hid = mem.propose_hypothesis("xauusd", "Gold tends to rally in the week before a Fed cut", "quant_researcher")
    with pytest.raises(ValueError):
        mem.confirm_hypothesis(hid, evidence={}, validated_by="risk_manager")
    # still a hypothesis, not silently promoted
    [item] = mem.query_knowledge(topic="xauusd")
    assert item.status == "hypothesis"


def test_hypothesis_confirmed_with_real_evidence(mem):
    hid = mem.propose_hypothesis("xauusd", "Gold tends to rally in the week before a Fed cut", "quant_researcher")
    mem.confirm_hypothesis(
        hid,
        evidence={"backtest_id": "bt_0012", "sample_size": 11, "avg_return": 0.014, "p_value": 0.03},
        validated_by="risk_manager",
    )
    [item] = mem.query_knowledge(topic="xauusd", status="confirmed")
    assert item.validated_by == "risk_manager"


def test_persists_across_reopen(tmp_path):
    db = tmp_path / "persist.db"
    m1 = CollectiveMemory(db)
    m1.remember("agent_a", "lesson", "Don't trust thin-holiday liquidity breakouts")
    m1.close()

    m2 = CollectiveMemory(db)
    rows = m2.recall("agent_a")
    m2.close()
    assert len(rows) == 1
    assert "holiday" in rows[0]["content"]


# --- regressions for the five reported bugs ---------------------------


def test_bug1_evidence_must_be_well_formed(mem):
    hid = mem.propose_hypothesis("xauusd", "stmt", "agent_a")
    # a non-empty but hand-wavy dict with no backing key is rejected
    with pytest.raises(ValueError):
        mem.confirm_hypothesis(hid, evidence={"note": "looks good to me"}, validated_by="agent_b")
    # a backing key present but empty is rejected
    with pytest.raises(ValueError):
        mem.confirm_hypothesis(hid, evidence={"backtest_id": ""}, validated_by="agent_b")
    # implausible statistics are rejected
    with pytest.raises(ValueError):
        mem.confirm_hypothesis(hid, evidence={"p_value": 1.5}, validated_by="agent_b")
    with pytest.raises(ValueError):
        mem.confirm_hypothesis(hid, evidence={"sample_size": 1}, validated_by="agent_b")
    # still a pending hypothesis after all those rejections
    assert mem.query_knowledge(topic="xauusd")[0].status == "hypothesis"


def test_bug1_validate_evidence_standalone():
    with pytest.raises(ValueError):
        validate_evidence({})
    with pytest.raises(ValueError):
        validate_evidence("not a dict")
    validate_evidence({"backtest_id": "bt_1", "sample_size": 20, "p_value": 0.02})  # ok


def test_bug2_foreign_keys_enforced(mem):
    # foreign_keys pragma must be ON, so a dangling discussion_id can't be inserted
    val = mem._conn.execute("PRAGMA foreign_keys").fetchone()[0]
    assert val == 1


def test_bug3_post_message_rejects_missing_discussion(mem):
    with pytest.raises(ValueError):
        mem.post_message(999999, "agent_a", "claim", "to a non-existent discussion")


def test_bug3_post_message_rejects_bad_type(mem):
    did = mem.start_discussion("topic", "agent_a")
    with pytest.raises(ValueError):
        mem.post_message(did, "agent_a", "rant", "not a valid message type")


def test_bug4_cannot_reject_a_confirmed_hypothesis(mem):
    hid = mem.propose_hypothesis("xauusd", "stmt", "agent_a")
    mem.confirm_hypothesis(hid, evidence={"backtest_id": "bt_9", "sample_size": 30}, validated_by="agent_b")
    with pytest.raises(ValueError):
        mem.reject_hypothesis(hid, reason="changed my mind", validated_by="agent_c")
    with pytest.raises(ValueError):
        mem.confirm_hypothesis(hid, evidence={"backtest_id": "bt_10"}, validated_by="agent_c")
    assert mem.query_knowledge(topic="xauusd")[0].status == "confirmed"


def test_bug4_reject_requires_reason(mem):
    hid = mem.propose_hypothesis("t", "s", "a")
    with pytest.raises(ValueError):
        mem.reject_hypothesis(hid, reason="   ", validated_by="b")


def test_bug5_concurrent_writes_do_not_lose_rows(tmp_path):
    db = tmp_path / "concurrent.db"
    mem = CollectiveMemory(db)
    n_threads, per_thread = 8, 50

    def worker(tid):
        for i in range(per_thread):
            mem.remember(f"agent_{tid}", "note", f"row {i}")

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    total = sum(len(mem.recall(f"agent_{t}", limit=1000)) for t in range(n_threads))
    mem.close()
    assert total == n_threads * per_thread  # no lost/corrupted writes under contention


def test_bug5_concurrent_confirm_has_single_winner(tmp_path):
    # Many threads race to confirm the same hypothesis; exactly one may win,
    # the rest must raise (no double-transition, no corruption).
    db = tmp_path / "race.db"
    mem = CollectiveMemory(db)
    hid = mem.propose_hypothesis("t", "s", "a")
    wins, losses = [], []

    def racer():
        try:
            mem.confirm_hypothesis(hid, evidence={"backtest_id": "bt", "sample_size": 10}, validated_by="x")
            wins.append(1)
        except ValueError:
            losses.append(1)

    threads = [threading.Thread(target=racer) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    mem.close()
    assert len(wins) == 1
    assert len(losses) == 9


def test_bug5_confirm_single_winner_across_separate_connections(tmp_path):
    # Stronger than the shared-connection race: each thread opens its OWN
    # CollectiveMemory (its own sqlite connection) on the same file, so the
    # in-process RLock gives no protection. Only the BEGIN IMMEDIATE
    # transaction + busy_timeout can keep this to a single winner. This is the
    # multi-process concurrency guarantee the spec asks for.
    db = tmp_path / "multiconn.db"
    seed = CollectiveMemory(db)
    hid = seed.propose_hypothesis("t", "s", "a")
    seed.close()

    wins, losses = [], []

    def racer():
        conn = CollectiveMemory(db)
        try:
            conn.confirm_hypothesis(
                hid, evidence={"backtest_id": "bt", "sample_size": 10}, validated_by="x"
            )
            wins.append(1)
        except ValueError:
            losses.append(1)
        finally:
            conn.close()

    threads = [threading.Thread(target=racer) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(wins) == 1
    assert len(losses) == 9
    # and the persisted state is exactly one confirmed row
    check = CollectiveMemory(db)
    items = check.query_knowledge(topic="t")
    check.close()
    assert len(items) == 1 and items[0].status == "confirmed"


def test_write_txn_rolls_back_on_error(tmp_path):
    # A failed transition must leave no partial state behind.
    db = tmp_path / "rollback.db"
    mem = CollectiveMemory(db)
    hid = mem.propose_hypothesis("t", "s", "a")
    # Rejecting with an empty reason raises BEFORE the transaction opens;
    # a wrong-state transition raises INSIDE it and must roll back cleanly,
    # leaving the row still a pending hypothesis and the connection usable.
    mem.confirm_hypothesis(hid, evidence={"backtest_id": "x", "sample_size": 5}, validated_by="v")
    with pytest.raises(ValueError):
        mem.reject_hypothesis(hid, reason="too late", validated_by="v")
    # connection still works after the rollback
    hid2 = mem.propose_hypothesis("t2", "s2", "a")
    mem.reject_hypothesis(hid2, reason="valid rejection", validated_by="v")
    assert mem.query_knowledge(topic="t")[0].status == "confirmed"
    assert mem.query_knowledge(topic="t2")[0].status == "rejected"
    mem.close()


def test_discussion_resumable_after_restart(tmp_path):
    db = tmp_path / "resume.db"
    m1 = CollectiveMemory(db)
    did = m1.start_discussion("Gold direction into Fed", "agent_a", goal="agree a bias with evidence")
    m1.post_message(did, "agent_a", "claim", "bullish on softening real yields")
    m1.close()

    m2 = CollectiveMemory(db)
    still_open = m2.open_discussions()
    assert any(d["id"] == did for d in still_open)
    m2.post_message(did, "agent_b", "objection", "DXY firmed overnight")
    msgs = m2.get_discussion(did)
    assert [x["message_type"] for x in msgs] == ["claim", "objection"]
    m2.close_discussion(did)
    assert all(d["id"] != did for d in m2.open_discussions())
    m2.close()
