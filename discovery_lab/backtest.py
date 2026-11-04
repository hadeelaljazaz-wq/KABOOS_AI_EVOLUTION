"""Backtest engine: turns a position series into real, cost-aware performance.

Every call here is after-the-fact bookkeeping on a signal already shifted
by the caller (see strategies.py) — this module doesn't know or care how
the signal was produced, so it can't itself introduce look-ahead bias.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .costs import TransactionCosts

TRADING_DAYS_PER_YEAR = 252


@dataclass
class BacktestResult:
    gross_returns: pd.Series
    net_returns: pd.Series
    equity_curve: pd.Series
    n_trades: int
    cagr: float
    annual_vol: float
    sharpe: float
    max_drawdown: float
    win_rate: float


def run_backtest(close: pd.Series, position: pd.Series, costs: TransactionCosts) -> BacktestResult:
    close = close.loc[position.index]
    bar_return = close.pct_change().fillna(0.0)
    gross = position * bar_return
    net = costs.apply(position, gross)

    equity = (1.0 + net).cumprod()
    n_days = max(len(net), 1)
    cagr = equity.iloc[-1] ** (TRADING_DAYS_PER_YEAR / n_days) - 1.0 if equity.iloc[-1] > 0 else -1.0
    annual_vol = net.std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = (net.mean() * TRADING_DAYS_PER_YEAR) / annual_vol if annual_vol > 0 else 0.0
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_dd = drawdown.min()
    traded_days = net[position.diff().fillna(position).abs() > 0]
    win_rate = (traded_days > 0).mean() if len(traded_days) else 0.0
    n_trades = int((position.diff().fillna(position) != 0).sum())

    return BacktestResult(
        gross_returns=gross,
        net_returns=net,
        equity_curve=equity,
        n_trades=n_trades,
        cagr=float(cagr),
        annual_vol=float(annual_vol),
        sharpe=float(sharpe),
        max_drawdown=float(max_dd),
        win_rate=float(win_rate),
    )
