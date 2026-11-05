"""The ResearchAgent: a persistent mind wired to collective memory.

This is the layer the spec's sections 2-4 ask for, built *on top of* the
existing ``collective_memory`` store and ``discovery_lab`` data — nothing in
those is rewritten.

Lifecycle of one analysis (spec section 4):

  before -> ``recall_context``: pull this agent's own relevant past
            experience, the community's confirmed/ hypothesised knowledge on
            the topic, and any related open discussion.
  think  -> ``analyze``: compute real indicators, hand them (plus the recalled
            context) to the LLM, parse a structured conclusion.
  after  -> persist the conclusion with its evidence and provenance, note
            whether it is only a hypothesis, and record how it differs from
            this agent's previous conclusion (the revision trail).

Agents talk to each other only through ``collective_memory`` discussions, so
every exchange is durable and resumable after a restart.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from collective_memory import CollectiveMemory

from .analysis_tools import MarketSnapshot, summarize_market
from .identity import AgentIdentity
from .llm import BaseLLM


@dataclass
class AnalysisResult:
    agent_id: str
    market: str
    window: int
    bias: str
    confidence: float
    rationale: str
    hypothesis: str
    provenance: str
    is_spot: bool
    evidence: dict[str, Any]
    used_mock_llm: bool
    prior_bias: Optional[str] = None
    changed_from_prior: bool = False
    memory_id: Optional[int] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self))


@dataclass
class ReviewResult:
    reviewer_id: str
    target_agent_id: str
    stance: str  # supports | contradicts | mixed
    confidence: float
    critique: str
    counter_evidence: dict[str, Any]
    used_mock_llm: bool
    message_id: Optional[int] = None


@dataclass
class ReconsiderResult:
    agent_id: str
    original_bias: str
    original_confidence: float
    revised_bias: str
    revised_confidence: float
    changed: bool
    reasoning: str
    used_mock_llm: bool
    message_id: Optional[int] = None


@dataclass
class RecalledContext:
    own_past: list[dict[str, Any]] = field(default_factory=list)
    collective_knowledge: list[dict[str, Any]] = field(default_factory=list)
    related_discussions: list[dict[str, Any]] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.own_past or self.collective_knowledge or self.related_discussions)


class ResearchAgent:
    def __init__(self, identity: AgentIdentity, memory: CollectiveMemory, llm: BaseLLM):
        self.identity = identity
        self.memory = memory
        self.llm = llm

    @property
    def agent_id(self) -> str:
        return self.identity.agent_id

    # -- section 4: recall relevant context BEFORE thinking --------------

    def recall_context(self, market: str, topic: str, limit: int = 5) -> RecalledContext:
        own = self.memory.recall(self.agent_id, category=f"analysis:{market}", limit=limit)
        own_past = [
            {"content": r["content"], "confidence": r["confidence"], "at": r["created_at"]}
            for r in own
        ]
        knowledge = self.memory.query_knowledge(topic=topic)
        collective = [
            {"statement": k.statement, "status": k.status, "by": k.created_by}
            for k in knowledge
        ]
        related = [
            {"id": d["id"], "topic": d["topic"], "goal": d["goal"]}
            for d in self.memory.open_discussions()
            if topic in (d["topic"] or "") or market in (d["topic"] or "")
        ]
        return RecalledContext(own_past, collective, related)

    # -- section 4: think, using the recalled context -------------------

    def analyze(self, market: str = "XAUUSD", window: int = 20, topic: Optional[str] = None) -> AnalysisResult:
        topic = topic or f"{market}_direction"
        context = self.recall_context(market, topic)
        snap = summarize_market(market, window=window)

        user = self._analyze_prompt(snap, context, topic)
        resp = self.llm.complete(self.identity.system_prompt(), user)
        out = resp.json()

        prior_bias = self._last_bias(market)
        changed = prior_bias is not None and prior_bias != out["bias"]

        result = AnalysisResult(
            agent_id=self.agent_id,
            market=market,
            window=window,
            bias=out["bias"],
            confidence=float(out["confidence"]),
            rationale=out["rationale"],
            hypothesis=out["hypothesis"],
            provenance=snap.provenance,
            is_spot=snap.is_spot,
            evidence=snap.as_evidence(),
            used_mock_llm=resp.is_mock,
            prior_bias=prior_bias,
            changed_from_prior=changed,
        )

        # persist the conclusion with its evidence + provenance (section 4)
        mem_id = self.memory.remember(
            self.agent_id,
            category=f"analysis:{market}",
            content=json.dumps({"bias": result.bias, "confidence": result.confidence,
                                "rationale": result.rationale, "topic": topic}),
            confidence=result.confidence,
            evidence=result.evidence,
        )
        result.memory_id = mem_id

        # revision trail: how did this conclusion change vs. last time?
        if changed:
            self.memory.remember(
                self.agent_id,
                category=f"revision:{market}",
                content=json.dumps({"from": prior_bias, "to": result.bias,
                                    "trigger": "fresh_analysis", "topic": topic}),
                confidence=result.confidence,
            )
        return result

    # -- section 3: peer review (another agent challenges a conclusion) --

    def review(self, target: AnalysisResult, discussion_id: int, window: Optional[int] = None) -> ReviewResult:
        # Independent read: deliberately a DIFFERENT lookback so the reviewer
        # is not just recomputing the same number.
        rev_window = window or max(5, target.window // 2)
        snap = summarize_market(target.market, window=rev_window)
        user = self._review_prompt(target, snap)
        resp = self.llm.complete(self.identity.system_prompt(), user)
        out = resp.json()

        stance = out["stance"]
        msg_type = {"supports": "evidence", "contradicts": "objection", "mixed": "question"}.get(
            stance, "update"
        )
        msg = json.dumps({"reviewer": self.agent_id, "stance": stance,
                          "critique": out["critique"], "evidence": out.get("counter_evidence", {})})
        mid = self.memory.post_message(discussion_id, self.agent_id, msg_type, msg)

        # reviewer records what it found, with its own evidence
        self.memory.remember(
            self.agent_id,
            category=f"review:{target.market}",
            content=json.dumps({"target": target.agent_id, "stance": stance,
                                "critique": out["critique"]}),
            confidence=float(out["confidence"]),
            evidence=snap.as_evidence(),
        )
        return ReviewResult(
            reviewer_id=self.agent_id,
            target_agent_id=target.agent_id,
            stance=stance,
            confidence=float(out["confidence"]),
            critique=out["critique"],
            counter_evidence=out.get("counter_evidence", {}),
            used_mock_llm=resp.is_mock,
            message_id=mid,
        )

    # -- section 3+4: receive the review and re-evaluate ----------------

    def reconsider(self, original: AnalysisResult, review: ReviewResult, discussion_id: int) -> ReconsiderResult:
        user = self._reconsider_prompt(original, review)
        resp = self.llm.complete(self.identity.system_prompt(), user)
        out = resp.json()

        res = ReconsiderResult(
            agent_id=self.agent_id,
            original_bias=original.bias,
            original_confidence=original.confidence,
            revised_bias=out["revised_bias"],
            revised_confidence=float(out["revised_confidence"]),
            changed=bool(out["changed"]),
            reasoning=out["reasoning"],
            used_mock_llm=resp.is_mock,
        )
        # tell the discussion how the conclusion moved (durable, resumable)
        msg = json.dumps({"agent": self.agent_id, "revised_bias": res.revised_bias,
                          "revised_confidence": res.revised_confidence,
                          "changed": res.changed, "reasoning": res.reasoning})
        res.message_id = self.memory.post_message(discussion_id, self.agent_id, "update", msg)

        # revision trail entry (section 4: keep a record of how it changed)
        self.memory.remember(
            self.agent_id,
            category=f"revision:{original.market}",
            content=json.dumps({"from": original.bias, "to": res.revised_bias,
                                "from_conf": original.confidence, "to_conf": res.revised_confidence,
                                "trigger": f"peer_review:{review.reviewer_id}:{review.stance}"}),
            confidence=res.revised_confidence,
        )
        return res

    # -- section 4: share a useful finding as collective knowledge ------

    def share_hypothesis(self, topic: str, statement: str) -> int:
        return self.memory.propose_hypothesis(topic, statement, self.agent_id)

    # -- helpers ---------------------------------------------------------

    def _last_bias(self, market: str) -> Optional[str]:
        rows = self.memory.recall(self.agent_id, category=f"analysis:{market}", limit=1)
        if not rows:
            return None
        try:
            return json.loads(rows[0]["content"]).get("bias")
        except (json.JSONDecodeError, TypeError):
            return None

    @staticmethod
    def _context_block(context: RecalledContext) -> str:
        if context.is_empty():
            return "No prior context found (first time analysing this)."
        parts = []
        if context.own_past:
            parts.append("Your own recent conclusions: " + json.dumps(context.own_past))
        if context.collective_knowledge:
            parts.append("Community knowledge: " + json.dumps(context.collective_knowledge))
        if context.related_discussions:
            parts.append("Open related discussions: " + json.dumps(context.related_discussions))
        return "\n".join(parts)

    def _analyze_prompt(self, snap: MarketSnapshot, context: RecalledContext, topic: str) -> str:
        mock_input = {
            "kind": "analyze",
            "focus": self.identity.lean,
            "window": f"{snap.window}-bar",
            "indicators": snap.indicators,
            "provenance": snap.provenance,
        }
        return (
            f"Analyse {snap.market}. Data provenance (be honest, this is NOT necessarily "
            f"live spot): {snap.provenance}. Last bar: {snap.last_bar} over {snap.n_bars} bars.\n"
            f"Real indicators (multiple schools): {json.dumps(snap.indicators)}\n"
            f"{self._context_block(context)}\n"
            f"Topic: {topic}. Decide a directional bias.\n"
            'Reply with JSON: {"bias": "bullish|bearish|neutral", "confidence": 0..1, '
            '"rationale": str, "key_drivers": [str], "hypothesis": str}.\n'
            f"<<<MOCK_INPUT {json.dumps(mock_input)} MOCK_INPUT>>>"
        )

    def _review_prompt(self, target: AnalysisResult, snap: MarketSnapshot) -> str:
        mock_input = {
            "kind": "review",
            "focus": self.identity.lean,
            "original_bias": target.bias,
            "window": f"{snap.window}-bar",
            "indicators": snap.indicators,
        }
        return (
            f"A peer agent ({target.agent_id}) concluded a '{target.bias}' bias "
            f"(confidence {target.confidence}) on {target.market}: {target.rationale}\n"
            f"Independently check it on a {snap.window}-bar lookback. Your real indicators: "
            f"{json.dumps(snap.indicators)} (provenance: {snap.provenance}).\n"
            "Do your findings support or contradict the peer? Bring evidence.\n"
            'Reply with JSON: {"stance": "supports|contradicts|mixed", "confidence": 0..1, '
            '"critique": str, "counter_evidence": object}.\n'
            f"<<<MOCK_INPUT {json.dumps(mock_input)} MOCK_INPUT>>>"
        )

    def _reconsider_prompt(self, original: AnalysisResult, review: ReviewResult) -> str:
        mock_input = {
            "kind": "reconsider",
            "original_bias": original.bias,
            "original_confidence": original.confidence,
            "review_stance": review.stance,
        }
        return (
            f"You concluded '{original.bias}' (confidence {original.confidence}) on "
            f"{original.market}. A peer reviewer ({review.reviewer_id}) responded "
            f"'{review.stance}': {review.critique} with evidence {json.dumps(review.counter_evidence)}.\n"
            "Re-evaluate honestly. You may keep, soften, or change your conclusion; do not "
            "dig in just to save face, and do not flip without reason.\n"
            'Reply with JSON: {"revised_bias": str, "revised_confidence": 0..1, '
            '"changed": bool, "reasoning": str}.\n'
            f"<<<MOCK_INPUT {json.dumps(mock_input)} MOCK_INPUT>>>"
        )
