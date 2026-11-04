import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from collective_memory import CollectiveMemory


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
