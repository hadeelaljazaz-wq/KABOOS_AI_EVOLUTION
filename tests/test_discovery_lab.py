import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest

from discovery_lab import (
    TransactionCosts,
    fetch_ohlcv,
    run_backtest,
    sma_crossover_signal,
    buy_and_hold_signal,
    walk_forward_split,
    permutation_pvalue,
)


@pytest.fixture(scope="module")
def gold():
    # Uses the on-disk cache populated by the real yfinance fetch; falls
    # back to a live fetch (and then caches it) if no cache exists yet.
    return fetch_ohlcv("XAUUSD", period="2y", interval="1d")


def test_real_gold_data_shape(gold):
    assert len(gold) > 200
    assert {"Open", "High", "Low", "Close", "Volume"}.issubset(gold.columns)
    assert gold["Close"].min() > 0


def test_unknown_market_rejected():
    with pytest.raises(ValueError):
        fetch_ohlcv("DOGEUSD")


def test_signal_has_no_lookahead(gold):
    close = gold["Close"]
    full_signal = sma_crossover_signal(close, fast=5, slow=20)
    # The signal at bar t must depend only on bars < t: truncating the
    # series right after t and recomputing must give the same value at t.
    cut = len(close) - 10
    truncated_signal = sma_crossover_signal(close.iloc[: cut + 1], fast=5, slow=20)
    assert full_signal.iloc[cut] == truncated_signal.iloc[-1]


def test_costs_reduce_net_vs_gross(gold):
    close = gold["Close"]
    signal = sma_crossover_signal(close, fast=10, slow=30)
    costs = TransactionCosts()
    result = run_backtest(close, signal, costs)
    assert result.net_returns.sum() <= result.gross_returns.sum()
    assert result.n_trades > 0


def test_zero_cost_matches_gross(gold):
    close = gold["Close"]
    signal = sma_crossover_signal(close, fast=10, slow=30)
    zero_cost = TransactionCosts(spread=0.0, commission=0.0, slippage=0.0)
    result = run_backtest(close, signal, zero_cost)
    assert result.net_returns.equals(result.gross_returns)


def test_buy_and_hold_has_no_intermediate_trades(gold):
    close = gold["Close"]
    signal = buy_and_hold_signal(close)
    costs = TransactionCosts()
    result = run_backtest(close, signal, costs)
    # one entry only: position goes 0 -> 1 once and never changes again
    assert result.n_trades == 1


def test_walk_forward_split_is_sequential_and_nonoverlapping():
    splits = walk_forward_split(500, n_splits=4)
    assert len(splits) >= 1
    for train, test in splits:
        assert train.stop <= test.start
        assert test.stop <= 500


def test_walk_forward_rejects_too_few_rows():
    with pytest.raises(ValueError):
        walk_forward_split(3, n_splits=4)


def test_permutation_pvalue_is_a_probability(gold):
    close = gold["Close"]
    signal = sma_crossover_signal(close, fast=10, slow=30)
    result = run_backtest(close, signal, TransactionCosts())
    p = permutation_pvalue(result.net_returns, n_sims=200)
    assert 0.0 <= p <= 1.0
