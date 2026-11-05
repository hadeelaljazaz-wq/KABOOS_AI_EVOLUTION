import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import pytest

from agent_community import (
    ANALYSIS_SCHOOLS,
    AgentCommunity,
    AgentIdentity,
    MockLLM,
    default_identities,
    identity_by_id,
)
from agent_community.llm import get_llm


@pytest.fixture()
def comm(tmp_path):
    c = AgentCommunity(tmp_path / "community.db", llm_provider="mock")
    yield c
    c.close()


# --- identity (spec section 2) ---------------------------------------


def test_ten_stable_independent_identities():
    idents = default_identities()
    assert len(idents) == 10
    ids = [i.agent_id for i in idents]
    assert len(set(ids)) == 10  # all distinct, stable keys
    # stable across re-import / lookup
    assert identity_by_id("agent_trend").name == "Terra"


def test_every_agent_can_access_every_school(comm):
    # spec: don't confine an agent to one school; a lean is a start, not a cage
    for agent in comm.all_agents():
        assert set(agent.identity.accessible_schools) == set(ANALYSIS_SCHOOLS)
        assert agent.identity.lean in ANALYSIS_SCHOOLS


def test_leans_are_diverse():
    leans = {i.lean for i in default_identities()}
    assert len(leans) >= 8  # genuinely varied research interests


# --- analysis is real + honest (sections 3,4,6) ----------------------


def test_analysis_is_structured_and_cites_real_evidence(comm):
    res = comm.agent("agent_trend").analyze("XAUUSD")
    assert res.bias in {"bullish", "bearish", "neutral"}
    assert 0.0 <= res.confidence <= 1.0
    # evidence is a well-formed, real-data payload
    assert res.evidence["sample_size"] >= 200
    assert "futures" in res.evidence["data_source"].lower()
    assert set(res.evidence["metrics"]) >= {"trend", "rsi", "ret_window", "vol_annual"}


def test_gold_is_not_mislabelled_as_spot(comm):
    res = comm.agent("agent_macro").analyze("XAUUSD")
    # GC=F futures proxy must never be reported as live XAUUSD spot
    assert res.is_spot is False
    assert "not spot" in res.provenance.lower() or "proxy" in res.provenance.lower()


def test_mock_backend_is_flagged(comm):
    assert comm.used_mock_llm is True
    res = comm.agent("agent_quant").analyze("XAUUSD")
    assert res.used_mock_llm is True


# --- communication is real + durable (section 3) ---------------------


def test_collaboration_posts_real_durable_messages(comm):
    t = comm.collaborate_on("XAUUSD", "agent_trend", "agent_macro")
    types = [m["message_type"] for m in t.messages]
    assert types[0] == "claim"
    assert types[-1] == "update"
    # review posted something evidence-bearing (evidence/objection/question)
    assert any(ty in {"evidence", "objection", "question"} for ty in types)
    # messages are persisted, not just returned
    reloaded = comm.memory.get_discussion(t.discussion_id)
    assert len(reloaded) == len(t.messages) >= 3


def test_request_help_calls_another_agent(comm):
    did = comm.memory.start_discussion("need a second look at gold", "agent_trend")
    out = comm.request_help("agent_trend", "agent_vol", did, "Is vol regime a concern?")
    msgs = comm.memory.get_discussion(did)
    assert [m["message_type"] for m in msgs] == ["question", "evidence"]
    assert out["responder"] == "agent_vol"


# --- loop / API-spend guards (section 3) -----------------------------


def test_debate_is_bounded_and_never_infinite(comm):
    did = comm.memory.start_discussion("bounded debate", "agent_trend")
    out = comm.run_bounded_debate(did, ["agent_trend", "agent_macro", "agent_vol"],
                                  market="XAUUSD", max_rounds=3)
    assert out["stopped"] in {"converged", "max_rounds", "message_cap"}
    assert out["rounds"] <= 3
    from agent_community.community import MAX_MESSAGES_PER_DISCUSSION
    assert len(comm.memory.get_discussion(did)) <= MAX_MESSAGES_PER_DISCUSSION


# --- evidence-driven revision, not voting (section 3) ----------------


def test_mock_reconsider_rules_are_evidence_driven():
    # Prove the revision dynamic directly (deterministic), independent of market
    # data: a contradiction lowers confidence; support raises it; mixed softens.
    llm = MockLLM()

    def recon(bias, conf, stance):
        prompt = ("x <<<MOCK_INPUT " + json.dumps({
            "kind": "reconsider", "original_bias": bias,
            "original_confidence": conf, "review_stance": stance,
        }) + " MOCK_INPUT>>>")
        return llm.complete("sys", prompt).json()

    contra = recon("bullish", 0.8, "contradicts")
    assert contra["revised_confidence"] < 0.8 and contra["changed"] is True

    supp = recon("bullish", 0.5, "supports")
    assert supp["revised_confidence"] > 0.5 and supp["revised_bias"] == "bullish"

    mixed = recon("bearish", 0.6, "mixed")
    assert mixed["revised_confidence"] < 0.6

    # a strong contradiction can push an uncertain call to neutral
    flip = recon("bullish", 0.5, "contradicts")
    assert flip["revised_bias"] == "neutral"


# --- scalability to 10+ independent agents (section 5) ---------------


def test_architecture_scales_beyond_ten_agents(tmp_path):
    # The roster is just data: build 12 distinct identities and confirm each
    # is an independent mind with its own memory, all over the same store.
    idents = tuple(
        AgentIdentity(f"agent_x{i}", f"X{i}", "statistical_quant",
                      f"test agent {i}")
        for i in range(12)
    )
    comm = AgentCommunity(tmp_path / "scale.db", identities=idents, llm_provider="mock")
    assert len(comm.ids()) == 12
    for aid in comm.ids():
        res = comm.agent(aid).analyze("XAUUSD")
        assert res.agent_id == aid
    # each agent's memory is separate and recallable by its own id
    rows = comm.memory.recall("agent_x3", category="analysis:XAUUSD")
    assert len(rows) == 1
    comm.close()


def test_get_llm_returns_mock_without_keys(monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY", "KABOOS_LLM_PROVIDER"):
        monkeypatch.delenv(k, raising=False)
    llm = get_llm()
    assert llm.is_mock is True
    # and it must refuse to pretend when the caller demands a real backend
    with pytest.raises(RuntimeError):
        get_llm(allow_mock=False)
