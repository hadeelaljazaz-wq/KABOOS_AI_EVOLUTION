"""Guards against the failure modes named in spec section 5.

This is a first, honest slice — not a full research-grade validation
suite. It gives two real checks:

  - walk_forward_split: sequential, non-overlapping train/test windows,
    so a strategy's parameters are never fit on the same bars its
    performance is judged on (out-of-sample testing, no shuffling of
    time order -> no look-ahead).

  - signal_permutation_pvalue: a real Monte Carlo significance test for
    whether a strategy's *positioning* has any relationship to the
    returns that follow it.

WHY THE NAIVE VERSION WAS WRONG (and is now removed): shuffling a
strategy's own realised daily returns and recomputing the Sharpe ratio
is meaningless. The Sharpe ratio is mean/std, and both the mean and the
standard deviation of a set of numbers are invariant under permutation —
so every shuffle reproduces the observed Sharpe exactly, and the only
reason the old p-value wasn't exactly 1.0 was floating-point summation
noise. That is a misleading significance estimate, not a test.

The correct permutation test keeps the strategy's position series FIXED
and shuffles the *market* returns, breaking the time alignment between
each day's position and the return that actually followed it. Under that
null ("the positions carry no information about next-step returns") the
mean strategy return genuinely changes from shuffle to shuffle, so the
Sharpe distribution is non-degenerate and the p-value is informative.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def walk_forward_split(n_rows: int, n_splits: int = 4) -> list[tuple[slice, slice]]:
    """Sequential (train, test) index-slice pairs, each test block after its train block.

    Example with n_splits=4: train grows each fold, test is the next
    contiguous block — never data from the future of its own train set.
    """
    if n_splits < 2:
        raise ValueError("need at least 2 splits to have both a train and a test block")
    fold_size = n_rows // (n_splits + 1)
    if fold_size == 0:
        raise ValueError("not enough rows for the requested number of splits")
    splits = []
    for i in range(1, n_splits + 1):
        train_end = fold_size * i
        test_end = min(fold_size * (i + 1), n_rows)
        if train_end >= test_end:
            continue
        splits.append((slice(0, train_end), slice(train_end, test_end)))
    return splits


def _sharpe(returns: np.ndarray) -> float:
    std = returns.std(ddof=0)
    if std == 0:
        return 0.0
    return float(returns.mean() / std)


def signal_permutation_pvalue(
    position: pd.Series,
    market_returns: pd.Series,
    n_sims: int = 1000,
    seed: int = 7,
) -> float:
    """P(a strategy that positions blind to the data matches/beats this Sharpe).

    ``position`` is the (already shifted, look-ahead-free) position held on
    each bar; ``market_returns`` is the underlying instrument's per-bar
    return on the same index. The observed statistic is the Sharpe of
    ``position * market_returns``. Each simulation shuffles the market
    returns, keeps the positions fixed, and recomputes the Sharpe — the
    null being that the positions have no predictive relationship to the
    returns that follow. A low p-value means that relationship is unlikely
    to be chance on this sample.

    This tests the signal's *timing edge*. It is not a test for
    overfitting across many candidate strategies (that needs a
    multiple-testing correction) — don't read it as one.
    """
    pos = position.to_numpy(dtype=float)
    mkt = market_returns.reindex(position.index).fillna(0.0).to_numpy(dtype=float)
    if len(pos) != len(mkt):
        raise ValueError("position and market_returns must align on the same index")
    if np.count_nonzero(pos) < 10:
        raise ValueError("need at least 10 bars with a non-zero position for a permutation test")

    observed = _sharpe(pos * mkt)
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_sims):
        shuffled_mkt = rng.permutation(mkt)
        if _sharpe(pos * shuffled_mkt) >= observed:
            hits += 1
    # +1 smoothing (observed counts as one realisation under the null) avoids
    # reporting an impossible p-value of exactly 0.
    return (hits + 1) / (n_sims + 1)
