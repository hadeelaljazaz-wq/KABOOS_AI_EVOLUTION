"""Adapter to reuse the vendored TradingAgents graph WITHOUT modifying it.

Spec section 6: keep ``vendor_tradingagents`` untouched and integrate through
an adapter. This wraps the TradingAgents ``TradingAgentsGraph`` as one more
analysis source a KABOOS agent can consult, and maps its verdict into the same
shape the community already uses.

It is import-guarded on purpose. In this container the vendored package and an
LLM key may both be absent; in that case ``is_available()`` is False and the
rest of the community runs unaffected. Nothing here fabricates a TradingAgents
result when the real graph cannot run — it raises instead.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional


def is_available() -> tuple[bool, str]:
    """(usable, reason). Usable only if the package imports AND a key exists."""
    try:
        import tradingagents.graph.trading_graph  # noqa: F401
    except Exception as exc:  # not installed / not on path
        return False, f"vendor_tradingagents not importable: {type(exc).__name__}: {exc}"
    has_key = any(os.environ.get(k) for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"))
    if not has_key:
        return False, "no LLM API key set (OPENAI/ANTHROPIC/GOOGLE) — TradingAgents cannot think"
    return True, "ready"


@dataclass
class TradingAgentsVerdict:
    decision: str            # the graph's final signal (e.g. BUY/HOLD/SELL)
    raw_state: Any
    source: str = "vendor_tradingagents"


class TradingAgentsAdapter:
    """Lazily constructs the vendored graph; never mutates vendored code."""

    def __init__(self, config: Optional[dict[str, Any]] = None):
        ok, reason = is_available()
        if not ok:
            raise RuntimeError(f"TradingAgents adapter unavailable: {reason}")
        from tradingagents.default_config import DEFAULT_CONFIG
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        cfg = dict(DEFAULT_CONFIG)
        if config:
            cfg.update(config)
        self._graph = TradingAgentsGraph(config=cfg)

    def analyze(self, symbol: str, date: str) -> TradingAgentsVerdict:
        """Run one TradingAgents analysis cycle for a symbol/date.

        ``symbol`` here should be a ticker the vendored data vendors understand
        (e.g. 'GLD' or 'GC=F'), NOT the spot label 'XAUUSD' — the caller is
        responsible for passing an instrument the real data layer can fetch.
        """
        state, decision = self._graph.propagate(symbol, date)
        return TradingAgentsVerdict(decision=str(decision), raw_state=state)
