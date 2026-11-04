"""Real trading-cost assumptions (spec section 5: never backtest cost-free).

Costs are charged only on days the position actually changes (turnover),
which is how spread/commission/slippage really hit a strategy.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TransactionCosts:
    """All figures in fractional return terms (0.0005 = 5 bps) per side."""

    spread: float = 0.0004     # typical retail XAUUSD spread, round-turn-ish
    commission: float = 0.00005
    slippage: float = 0.0002

    @property
    def per_trade(self) -> float:
        return self.spread + self.commission + self.slippage

    def apply(self, position: pd.Series, gross_returns: pd.Series) -> pd.Series:
        """Deduct cost on every bar where ``position`` changed from the prior bar."""
        turnover = position.diff().abs().fillna(position.abs())
        cost = turnover * self.per_trade
        return gross_returns - cost
