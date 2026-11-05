"""The spec section 5 experiment, as an automated regression test.

Uses the deterministic MockLLM, so it proves the collaboration *architecture*
(memory wiring, discussion flow, persistence across restart, revision trail) —
NOT real collaborative intelligence. That distinction is asserted explicitly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_community import AgentCommunity

AUTHOR = "agent_trend"
REVIEWER = "agent_macro"
MARKET = "XAUUSD"


def test_full_two_agent_cycle_persists_and_resumes(tmp_path):
    db = tmp_path / "cycle.db"

    # ---- session 1: analyse -> review -> reconsider -> record ----------
    comm = AgentCommunity(db, llm_provider="mock")
    assert comm.used_mock_llm is True  # honest: this run is NOT real AI

    t = comm.collaborate_on(MARKET, AUTHOR, REVIEWER, goal="evidence-backed gold bias")
    disc_id = t.discussion_id
    assert t.author_analysis.bias in {"bullish", "bearish", "neutral"}
    # the reviewer genuinely acted on the author's conclusion
    assert t.review.target_agent_id == AUTHOR
    # author re-evaluated (confidence or bias recorded as a revision)
    assert t.reconsideration.agent_id == AUTHOR
    original_conf = t.author_analysis.confidence
    comm.close()

    # ---- RESTART: fresh objects, same DB file --------------------------
    comm2 = AgentCommunity(db, llm_provider="mock")

    # the discussion is resumable (survived the restart)
    assert any(d["id"] == disc_id for d in comm2.resumable_discussions())
    # and still holds the full prior exchange
    msgs = comm2.memory.get_discussion(disc_id)
    assert [m["message_type"] for m in msgs][:1] == ["claim"]
    assert len(msgs) >= 3

    # both agents' identities + memories came back by id
    author = comm2.agent(AUTHOR)
    ctx = author.recall_context(MARKET, f"{MARKET}_direction")
    assert len(ctx.own_past) >= 1                 # own prior conclusion recalled
    assert len(ctx.related_discussions) >= 1      # the open discussion is related

    # ---- a NEW task after restart that uses the recalled context -------
    prior_bias = author._last_bias(MARKET)
    assert prior_bias is not None                 # it remembers last time
    new_res = author.analyze(MARKET, window=10, topic=f"{MARKET}_direction")
    # changed_from_prior is computed against the recalled prior conclusion
    assert new_res.changed_from_prior == (prior_bias != new_res.bias)

    # the revision trail from session 1 is durable and readable after restart
    revs = comm2.memory.recall(AUTHOR, category=f"revision:{MARKET}", limit=10)
    assert len(revs) >= 1
    comm2.close()


def test_identity_and_memory_survive_restart(tmp_path):
    db = tmp_path / "survive.db"
    c1 = AgentCommunity(db, llm_provider="mock")
    c1.agent(AUTHOR).analyze(MARKET)
    ids_before = c1.ids()
    c1.close()

    c2 = AgentCommunity(db, llm_provider="mock")
    assert c2.ids() == ids_before                 # same stable identities
    rows = c2.agent(AUTHOR).memory.recall(AUTHOR, category=f"analysis:{MARKET}")
    assert len(rows) == 1                          # memory persisted on disk
    c2.close()


def test_shared_knowledge_is_visible_to_other_agents(tmp_path):
    db = tmp_path / "share.db"
    comm = AgentCommunity(db, llm_provider="mock")
    hid = comm.agent(AUTHOR).share_hypothesis(f"{MARKET}_direction",
                                              "Gold trends down while real yields rise")
    # another agent recalls it as community knowledge before its own analysis
    ctx = comm.agent(REVIEWER).recall_context(MARKET, f"{MARKET}_direction")
    assert any("real yields" in k["statement"] for k in ctx.collective_knowledge)
    assert ctx.collective_knowledge[0]["status"] == "hypothesis"  # not a confirmed fact
    comm.close()
