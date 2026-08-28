"""Strategy interface and shared position-targeting helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from backtest.engine import MarketView
from execution.paper_exchange import DecisionContext, Order
from execution.simulator import Side

#: Positions smaller than this in base units are treated as flat.
DUST = 1e-9


class BaseStrategy(ABC):
    """A strategy turns a point-in-time view into orders. It never fills them."""

    name: str = "base"

    @abstractmethod
    def on_bar(self, view: MarketView) -> Sequence[Order]:
        ...

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def orders_to_reach(
        view: MarketView,
        coin: str,
        target_size: float,
        context: DecisionContext,
        *,
        min_trade_notional: float = 10.0,
    ) -> list[Order]:
        """Emit the single order that moves ``coin`` to ``target_size``.

        Returns nothing when the adjustment is smaller than
        ``min_trade_notional``, so a strategy cannot bleed fees churning
        rounding errors.
        """
        current = view.position_size(coin)
        delta = target_size - current
        price = view.price(coin)
        if abs(delta) < DUST or abs(delta) * price < min_trade_notional:
            return []
        side = Side.BUY if delta > 0 else Side.SELL
        reduce_only = abs(target_size) < abs(current) and (
            target_size == 0.0 or (target_size > 0) == (current > 0)
        )
        return [
            Order(
                coin=coin,
                side=side,
                size=abs(delta),
                reduce_only=reduce_only,
                context=context,
            )
        ]
