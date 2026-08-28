"""Deliberately dumb strategies.

These exist to test the machinery, not to make money. A backtester that
cannot price buy-and-hold correctly cannot price anything, so
:class:`AlwaysLongStrategy` is the reference case the Phase 2 accounting
proof is built on.
"""

from __future__ import annotations

from typing import Sequence

from backtest.engine import MarketView
from execution.paper_exchange import DecisionContext, Order
from strategy.base import BaseStrategy


class FlatStrategy(BaseStrategy):
    """Never trades. Equity must stay exactly flat at the starting capital."""

    name = "flat"

    def on_bar(self, view: MarketView) -> Sequence[Order]:
        return []


class AlwaysLongStrategy(BaseStrategy):
    """Buy a fixed notional on the first bar and hold it forever.

    No rebalancing, so the resulting P&L is a closed-form function of the
    entry and exit prices, which is exactly what makes it a usable check on
    the exchange's arithmetic.
    """

    name = "always_long"

    def __init__(self, coins: Sequence[str] | None = None, notional_per_coin: float = 10_000.0):
        self.coins = list(coins) if coins else None
        self.notional_per_coin = notional_per_coin
        self._entered: set[str] = set()

    def on_bar(self, view: MarketView) -> Sequence[Order]:
        coins = self.coins or view.coins
        orders: list[Order] = []
        for coin in coins:
            if coin in self._entered:
                continue
            size = view.notional_to_size(coin, self.notional_per_coin)
            if size <= 0:
                continue
            context = DecisionContext(
                reason=f"always-long baseline: open {self.notional_per_coin:,.0f} USD",
                regime="n/a",
                risk_checks=["baseline strategy, no risk engine in the loop"],
                target_notional=self.notional_per_coin,
            )
            orders.extend(self.orders_to_reach(view, coin, size, context))
            self._entered.add(coin)
        return orders


class AlwaysLongLeveredStrategy(AlwaysLongStrategy):
    """Buy-and-hold sized as a multiple of starting equity.

    Used to demonstrate liquidation: at 10x, a ~9% adverse move ends the
    account, which is the whole point of testing the aggressive profile.
    """

    name = "always_long_levered"

    def __init__(self, coins: Sequence[str] | None = None, leverage: float = 10.0):
        super().__init__(coins, notional_per_coin=0.0)
        self.leverage = leverage

    def on_bar(self, view: MarketView) -> Sequence[Order]:
        if not self._entered:
            coins = self.coins or view.coins
            self.notional_per_coin = view.equity() * self.leverage / max(1, len(coins))
        return super().on_bar(view)
