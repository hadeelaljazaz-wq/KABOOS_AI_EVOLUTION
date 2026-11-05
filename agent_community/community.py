"""KABOOS AGENT COMMUNITY — a persistent community of research agents.

A thin coordination layer over ``collective_memory`` (durable state) and the
``ResearchAgent`` minds. It does NOT replace or rewrite TradingAgents; the
TradingAgents graph can be plugged in as one more analysis source through
``tradingagents_adapter`` when its vendored package and an LLM key are present.

What this provides (spec sections 2-4):
  - a fixed roster of independent agents, each with its own id/identity/memory;
  - any agent can start a discussion, call another for help, request analysis,
    object with evidence, propose a hypothesis, or share a finding;
  - every exchange is stored, so a discussion is resumable after a restart;
  - hard limits on debate length so there are no infinite loops or runaway API
    spend (spec section 3).

Deliberately NOT a voting system: conclusions move through evidence and
re-evaluation, not by counting votes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from collective_memory import CollectiveMemory

from .agent import AnalysisResult, ReconsiderResult, ResearchAgent, ReviewResult
from .identity import AgentIdentity, default_identities
from .llm import BaseLLM, get_llm

# Hard safety limits (spec section 3: no infinite loops, no wasted API spend).
MAX_DEBATE_ROUNDS = 6
MAX_MESSAGES_PER_DISCUSSION = 40


@dataclass
class CollaborationTranscript:
    discussion_id: int
    market: str
    topic: str
    author_analysis: AnalysisResult
    review: ReviewResult
    reconsideration: ReconsiderResult
    used_mock_llm: bool
    messages: list[dict[str, Any]] = field(default_factory=list)


class AgentCommunity:
    def __init__(
        self,
        db_path: str | Path,
        identities: Optional[tuple[AgentIdentity, ...]] = None,
        llm: Optional[BaseLLM] = None,
        llm_provider: Optional[str] = None,
        allow_mock: bool = True,
    ):
        self.memory = CollectiveMemory(db_path)
        self.llm = llm or get_llm(prefer=llm_provider, allow_mock=allow_mock)
        idents = identities or default_identities()
        self._agents: dict[str, ResearchAgent] = {
            ident.agent_id: ResearchAgent(ident, self.memory, self.llm) for ident in idents
        }

    # -- roster ----------------------------------------------------------

    def agent(self, agent_id: str) -> ResearchAgent:
        return self._agents[agent_id]

    def ids(self) -> list[str]:
        return list(self._agents)

    def all_agents(self) -> list[ResearchAgent]:
        return list(self._agents.values())

    @property
    def used_mock_llm(self) -> bool:
        return self.llm.is_mock

    # -- section 3: one full collaboration cycle (bounded) --------------

    def collaborate_on(
        self,
        market: str,
        author_id: str,
        reviewer_id: str,
        topic: Optional[str] = None,
        goal: Optional[str] = None,
        author_window: int = 20,
    ) -> CollaborationTranscript:
        """Author analyses -> reviewer challenges with evidence -> author
        re-evaluates. One structured, evidence-driven round (not a vote)."""
        topic = topic or f"{market}_direction"
        author = self.agent(author_id)
        reviewer = self.agent(reviewer_id)

        analysis = author.analyze(market=market, window=author_window, topic=topic)
        disc_id = self.memory.start_discussion(
            topic=f"{topic} :: {market}", started_by=author_id,
            goal=goal or f"Reach an evidence-backed bias on {market}",
        )
        self.memory.post_message(
            disc_id, author_id, "claim",
            f"{analysis.bias} (conf {analysis.confidence}): {analysis.rationale}",
        )
        review = reviewer.review(analysis, disc_id)
        reconsideration = author.reconsider(analysis, review, disc_id)

        return CollaborationTranscript(
            discussion_id=disc_id,
            market=market,
            topic=topic,
            author_analysis=analysis,
            review=review,
            reconsideration=reconsideration,
            used_mock_llm=self.used_mock_llm,
            messages=[dict(m) for m in self.memory.get_discussion(disc_id)],
        )

    # -- section 3: an agent calls another for help (bounded) -----------

    def request_help(self, from_id: str, to_id: str, discussion_id: int, question: str,
                     market: str = "XAUUSD") -> dict[str, Any]:
        """``from_id`` asks ``to_id`` for an independent read inside a discussion."""
        self.memory.post_message(discussion_id, from_id, "question", question)
        helper = self.agent(to_id)
        snap_analysis = helper.analyze(market=market, topic=f"help:{market}")
        answer = (
            f"{snap_analysis.bias} (conf {snap_analysis.confidence}); "
            f"{snap_analysis.rationale}"
        )
        mid = self.memory.post_message(discussion_id, to_id, "evidence", answer)
        return {"responder": to_id, "message_id": mid, "analysis": snap_analysis}

    # -- section 3: a bounded free debate (the loop guard) --------------

    def run_bounded_debate(self, discussion_id: int, participant_ids: list[str],
                           market: str, max_rounds: int = MAX_DEBATE_ROUNDS) -> dict[str, Any]:
        """Let participants post in turn for a *capped* number of rounds.

        Stops early if the discussion hits the message cap or everyone agrees.
        This is the explicit guard against infinite loops / runaway spend.
        """
        rounds = 0
        last_biases: dict[str, str] = {}
        while rounds < max_rounds:
            if len(self.memory.get_discussion(discussion_id)) >= MAX_MESSAGES_PER_DISCUSSION:
                return {"stopped": "message_cap", "rounds": rounds}
            for pid in participant_ids:
                res = self.agent(pid).analyze(market=market, topic=f"debate:{market}")
                self.memory.post_message(
                    discussion_id, pid, "claim",
                    f"{res.bias} (conf {res.confidence})",
                )
                last_biases[pid] = res.bias
            rounds += 1
            if len(set(last_biases.values())) == 1:  # converged — stop, don't spin
                return {"stopped": "converged", "rounds": rounds, "bias": next(iter(set(last_biases.values())))}
        return {"stopped": "max_rounds", "rounds": rounds, "biases": last_biases}

    # -- section 4 / restart: what can be resumed -----------------------

    def resumable_discussions(self) -> list[dict[str, Any]]:
        return [dict(d) for d in self.memory.open_discussions()]

    def close(self) -> None:
        self.memory.close()
