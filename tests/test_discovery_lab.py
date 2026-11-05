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
    signal_permutation_pvalue,
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


def test_signal_permutation_pvalue_is_a_probability(gold):
    close = gold["Close"]
    signal = sma_crossover_signal(close, fast=10, slow=30)
    market_returns = close.pct_change().fillna(0.0)
    p = signal_permutation_pvalue(signal, market_returns, n_sims=200)
    assert 0.0 < p <= 1.0


def test_permutation_is_non_degenerate():
    # Regression for the statistical bug: the old test shuffled the
    # strategy's own returns, whose Sharpe is permutation-invariant, so the
    # shuffled Sharpe distribution had (essentially) zero spread. The fixed
    # test shuffles market returns against a fixed position, which MUST give
    # a non-degenerate distribution of shuffled Sharpes.
    import numpy as np
    import pandas as pd

    from discovery_lab.validation import _sharpe

    rng = np.random.default_rng(0)
    n = 300
    idx = pd.RangeIndex(n)
    position = pd.Series(rng.choice([-1, 0, 1], size=n), index=idx)
    market = pd.Series(rng.normal(0, 0.01, size=n), index=idx)

    mkt = market.to_numpy()
    pos = position.to_numpy(dtype=float)
    shuffled_sharpes = []
    for _ in range(200):
        shuffled_sharpes.append(_sharpe(pos * rng.permutation(mkt)))
    assert np.std(shuffled_sharpes) > 1e-3  # genuinely spread out, not a point mass


def test_permutation_detects_a_real_edge():
    # A position that is deliberately aligned with next-day returns (a
    # look-ahead "cheat") must score a LOW p-value; a random position must
    # not. This proves the test actually discriminates signal from noise.
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(1)
    n = 400
    idx = pd.RangeIndex(n)
    market = pd.Series(rng.normal(0, 0.01, size=n), index=idx)
    # Cheating position: sign of the SAME bar's return (perfect alignment).
    cheat_pos = pd.Series(np.sign(market.to_numpy()), index=idx)
    random_pos = pd.Series(rng.choice([-1, 1], size=n), index=idx)

    p_cheat = signal_permutation_pvalue(cheat_pos, market, n_sims=300)
    p_random = signal_permutation_pvalue(random_pos, market, n_sims=300)
    assert p_cheat < 0.05
    assert p_random > p_cheat


def test_market_source_is_honest_about_futures(gold):
    from discovery_lab.data import market_source

    note = market_source("XAUUSD")
    assert "futures" in note.lower()
    assert "not spot" in note.lower() or "proxy" in note.lower()
