"""LLM access for the KABOOS agent community.

Three concerns, kept strictly separate and honest:

  1. ``BaseLLM`` — the interface every agent talks to.
  2. ``AnthropicLLM`` / ``OpenAILLM`` — real model calls, used only when the
     matching SDK is installed AND an API key is present in the environment.
     These make the agents *actually think* with a real model.
  3. ``MockLLM`` — a deterministic, rule-based stand-in used when no key is
     available. It is explicitly flagged ``is_mock=True`` on every response.
     It exists so the *architecture* (memory wiring, discussion flow,
     persistence, revision trails) can be exercised and tested without a key.

     IMPORTANT (spec section 5, last line): the mock is NOT a demonstration of
     real collaborative intelligence. It applies transparent numeric rules to
     the indicators handed to it. Any report must say plainly when the mock
     was used instead of a real model. ``get_llm()`` logs which backend it
     chose, and never prints the key.

The agent builds a prompt that carries both (a) natural-language instructions
for a real model and (b) a machine-readable ``<<<MOCK_INPUT ... MOCK_INPUT>>>``
JSON block. Real models ignore the block and follow the instructions; the mock
reads the block and applies its rules. Both return the SAME JSON shape, so the
agent's parsing path is identical either way.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("kaboos.llm")

_MOCK_BLOCK_RE = re.compile(r"<<<MOCK_INPUT\s*(.*?)\s*MOCK_INPUT>>>", re.DOTALL)


@dataclass
class LLMResponse:
    """What every backend returns. ``is_mock`` is never silently dropped."""

    text: str
    is_mock: bool
    model: str
    raw: Any = None

    def json(self) -> dict[str, Any]:
        """Best-effort parse of a JSON object out of ``text``."""
        return _extract_json_object(self.text)


def _extract_json_object(text: str) -> dict[str, Any]:
    """Pull the first balanced ``{...}`` object out of a model reply."""
    start = text.find("{")
    if start == -1:
        raise ValueError(f"no JSON object in LLM reply: {text!r}")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError(f"unbalanced JSON in LLM reply: {text!r}")


class BaseLLM:
    """Minimal chat interface. One system prompt + one user message."""

    is_mock: bool = False
    model: str = "base"

    def complete(self, system: str, user: str, *, max_tokens: int = 1024) -> LLMResponse:
        raise NotImplementedError


# --------------------------------------------------------------------------
# Real backends (used only with a real key + installed SDK)
# --------------------------------------------------------------------------


class AnthropicLLM(BaseLLM):
    is_mock = False

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        import anthropic  # raises ImportError if not installed

        self.model = model or os.environ.get("KABOOS_LLM_MODEL", "claude-sonnet-5-5")
        self._client = anthropic.Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])

    def complete(self, system: str, user: str, *, max_tokens: int = 1024) -> LLMResponse:
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(block.text for block in msg.content if getattr(block, "type", None) == "text")
        return LLMResponse(text=text, is_mock=False, model=self.model, raw=msg)


class OpenAILLM(BaseLLM):
    is_mock = False

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        import openai  # raises ImportError if not installed

        self.model = model or os.environ.get("KABOOS_LLM_MODEL", "gpt-4o")
        self._client = openai.OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])

    def complete(self, system: str, user: str, *, max_tokens: int = 1024) -> LLMResponse:
        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return LLMResponse(
            text=resp.choices[0].message.content or "", is_mock=False, model=self.model, raw=resp
        )


# --------------------------------------------------------------------------
# Deterministic mock (no key required) — transparent, rule-based, flagged
# --------------------------------------------------------------------------


@dataclass
class MockLLM(BaseLLM):
    """Deterministic rule-based stand-in. Never a proof of real reasoning.

    It reads the ``<<<MOCK_INPUT ... MOCK_INPUT>>>`` JSON block the agent
    embeds and returns structured JSON by fixed numeric rules. Given the same
    input it always returns the same output, which is exactly what makes the
    collaboration *architecture* testable — and exactly why it must not be
    reported as intelligence.
    """

    is_mock: bool = True
    model: str = "mock-rules-v1"
    calls: list[dict[str, Any]] = field(default_factory=list)

    def complete(self, system: str, user: str, *, max_tokens: int = 1024) -> LLMResponse:
        m = _MOCK_BLOCK_RE.search(user)
        payload: dict[str, Any] = json.loads(m.group(1)) if m else {}
        self.calls.append(payload)
        kind = payload.get("kind", "analyze")
        if kind == "analyze":
            out = self._analyze(payload)
        elif kind == "review":
            out = self._review(payload)
        elif kind == "reconsider":
            out = self._reconsider(payload)
        else:
            out = {"error": f"unknown mock kind {kind!r}"}
        return LLMResponse(text=json.dumps(out), is_mock=True, model=self.model)

    # --- rules ----------------------------------------------------------

    @staticmethod
    def _bias_from_indicators(ind: dict[str, Any]) -> tuple[str, float]:
        trend = float(ind.get("trend", 0.0))
        rsi = float(ind.get("rsi", 50.0))
        ret = float(ind.get("ret_window", 0.0))
        vol = float(ind.get("vol_annual", 0.0)) or 1e-9
        # trend sign decides direction; RSI extremes temper it; confidence
        # scales with how large the recent move is relative to volatility.
        if trend > 0 and rsi < 75:
            bias = "bullish"
        elif trend < 0 and rsi > 25:
            bias = "bearish"
        else:
            bias = "neutral"
        strength = min(abs(ret) / vol, 1.0)
        confidence = round(0.4 + 0.5 * strength, 3)
        if bias == "neutral":
            confidence = round(min(confidence, 0.45), 3)
        return bias, confidence

    def _analyze(self, p: dict[str, Any]) -> dict[str, Any]:
        ind = p.get("indicators", {})
        bias, confidence = self._bias_from_indicators(ind)
        focus = p.get("focus", "general")
        return {
            "bias": bias,
            "confidence": confidence,
            "rationale": (
                f"[mock/{focus}] trend={ind.get('trend')}, RSI={ind.get('rsi')}, "
                f"{p.get('window', 'window')} return={ind.get('ret_window')}, "
                f"ann.vol={ind.get('vol_annual')} -> {bias}"
            ),
            "key_drivers": [k for k in ("trend", "rsi", "ret_window") if k in ind],
            "hypothesis": (
                f"Gold ({p.get('provenance', 'source unknown')}) is biased {bias} "
                f"on a {p.get('window', '')} horizon"
            ),
        }

    def _review(self, p: dict[str, Any]) -> dict[str, Any]:
        original_bias = p.get("original_bias", "neutral")
        my_bias, my_conf = self._bias_from_indicators(p.get("indicators", {}))
        if my_bias == original_bias:
            stance = "supports"
        elif "neutral" in (my_bias, original_bias):
            stance = "mixed"
        else:
            stance = "contradicts"
        return {
            "stance": stance,
            "confidence": my_conf,
            "critique": (
                f"[mock] reviewing a {original_bias} call on a different lookback "
                f"({p.get('window')}): my read is {my_bias} -> {stance}"
            ),
            "counter_evidence": p.get("indicators", {}),
        }

    def _reconsider(self, p: dict[str, Any]) -> dict[str, Any]:
        bias = p.get("original_bias", "neutral")
        conf = float(p.get("original_confidence", 0.5))
        stance = p.get("review_stance", "supports")
        if stance == "contradicts":
            new_conf = round(max(conf - 0.25, 0.1), 3)
            new_bias = "neutral" if new_conf < 0.4 else bias
        elif stance == "mixed":
            new_conf = round(max(conf - 0.1, 0.1), 3)
            new_bias = bias
        else:  # supports
            new_conf = round(min(conf + 0.1, 0.95), 3)
            new_bias = bias
        changed = (new_bias != bias) or (abs(new_conf - conf) > 1e-9)
        return {
            "revised_bias": new_bias,
            "revised_confidence": new_conf,
            "changed": changed,
            "reasoning": (
                f"[mock] peer review was '{stance}'; moved confidence "
                f"{conf}->{new_conf}, bias {bias}->{new_bias}"
            ),
        }


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------


def get_llm(prefer: Optional[str] = None, *, allow_mock: bool = True) -> BaseLLM:
    """Return the best available backend without ever printing the key.

    Order: explicit ``prefer`` ('anthropic'|'openai'|'mock'), else whichever
    real provider has both its SDK and key, else the mock. Logs the choice.
    Set ``allow_mock=False`` to force a hard failure when no real key exists
    (useful for a run that must prove real intelligence).
    """
    prefer = (prefer or os.environ.get("KABOOS_LLM_PROVIDER", "")).lower() or None

    def _try_anthropic() -> Optional[BaseLLM]:
        if os.environ.get("ANTHROPIC_API_KEY"):
            try:
                llm = AnthropicLLM()
                logger.info("LLM backend: Anthropic model=%s (real)", llm.model)
                return llm
            except Exception as exc:  # pragma: no cover - needs real env
                logger.warning("Anthropic unavailable (%s); falling back", type(exc).__name__)
        return None

    def _try_openai() -> Optional[BaseLLM]:
        if os.environ.get("OPENAI_API_KEY"):
            try:
                llm = OpenAILLM()
                logger.info("LLM backend: OpenAI model=%s (real)", llm.model)
                return llm
            except Exception as exc:  # pragma: no cover - needs real env
                logger.warning("OpenAI unavailable (%s); falling back", type(exc).__name__)
        return None

    if prefer == "mock":
        logger.info("LLM backend: MockLLM (forced) — NOT real intelligence")
        return MockLLM()
    if prefer == "anthropic":
        llm = _try_anthropic()
        if llm:
            return llm
    if prefer == "openai":
        llm = _try_openai()
        if llm:
            return llm
    if prefer is None:
        for factory in (_try_anthropic, _try_openai):
            llm = factory()
            if llm:
                return llm

    if not allow_mock:
        raise RuntimeError(
            "no real LLM backend available (set ANTHROPIC_API_KEY or OPENAI_API_KEY) "
            "and allow_mock=False"
        )
    logger.warning(
        "LLM backend: MockLLM — no API key found. Architecture will run, but this "
        "is a deterministic rule-based stand-in, NOT a demonstration of real AI "
        "collaboration."
    )
    return MockLLM()
