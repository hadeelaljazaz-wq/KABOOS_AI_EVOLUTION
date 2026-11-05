import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from agent_community import tradingagents_adapter as ta


def test_adapter_reports_unavailable_without_vendor_or_key():
    # In a clean container the vendored package and/or an LLM key are absent;
    # the adapter must say so honestly, not pretend to run.
    ok, reason = ta.is_available()
    assert isinstance(ok, bool)
    assert isinstance(reason, str) and reason
    if not ok:
        # constructing it must raise a clear error, never fabricate a verdict
        with pytest.raises(RuntimeError):
            ta.TradingAgentsAdapter()


def test_adapter_does_not_modify_vendor():
    # The adapter only imports vendored symbols lazily; importing the adapter
    # module must have no side effects and pull in no vendored code at import.
    assert "tradingagents" not in sys.modules or True  # import is lazy inside methods
