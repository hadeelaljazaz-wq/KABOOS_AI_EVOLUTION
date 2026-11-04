"""Guards against the failure modes named in spec section 5.

This is a first, honest slice — not a full research-grade validation
suite. It gives two real checks:

  - walk_forward_split: sequential, non-overlapping train/test windows,
    so a strategy's parameters are never fit on the same bars its
    performance is judged on (out-of-sample testing, no shuffling of
    time order -> no look-ahead).
  - permutation_pvalue: a real Monte Carlo significance test — shuffles
    the strategy's own daily returns many times and asks how often
    random luck alone would produce a Sharpe ratio this high. It checks
    for "this could be noise", not for overfitting on indicator choice;
    say so plainly rather than overclaiming what it covers.
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


def permutation_pvalue(daily_returns: pd.Series, n_sims: int = 1000, seed: int = 7) -> float:
    """P(a random shuffle of these same daily returns matches or beats the observed Sharpe).

    Shuffling preserves the marginal distribution of returns but destroys
    any real time-dependent edge, so a low p-value means the ordering/edge
    is unlikely to be pure chance given this sample.
    """
    returns = daily_returns.dropna().to_numpy()
    if len(returns) < 10:
        raise ValueError("need at least 10 return observations for a permutation test")
    observed_sharpe = _sharpe(returns)
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_sims):
        shuffled = rng.permutation(returns)
        if _sharpe(shuffled) >= observed_sharpe:
            hits += 1
    return hits / n_sims


def _sharpe(returns: np.ndarray) -> float:
    std = returns.std(ddof=0)
    if std == 0:
        return 0.0
    return returns.mean() / std
