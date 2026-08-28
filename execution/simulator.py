"""Fill simulation: what a paper order actually costs.

Separated from the exchange's bookkeeping so the cost model can be tested
in isolation and swapped without touching account accounting.

Every model here is **deterministic and pessimistic**. No randomness, so a
backtest is reproducible; adverse-by-construction, so an optimistic bug
shows up as a worse result rather than a better one. Concretely:

* the spread is always crossed in the direction that hurts,
* market impact grows with the square root of participation,
* latency drift is charged as an adverse move, never a favourable one,
* an order may not eat more than a configured share of a bar's volume.

The one thing this cannot model from bar data is queue position on resting
limit orders. Limit fills therefore use a conservative rule documented on
:meth:`FillSimulator.simulate` and should be treated as optimistic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from config.settings import SETTINGS, ExecutionConfig


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"

    @property
    def sign(self) -> int:
        return 1 if self is Side.BUY else -1

    @property
    def opposite(self) -> "Side":
        return Side.SELL if self is Side.BUY else Side.BUY


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


@dataclass(frozen=True)
class MarketSnapshot:
    """Everything the exchange is allowed to know about one coin at one bar.

    A snapshot describes a bar that has already CLOSED. Orders generated
    from it are executed against the *following* bar, so nothing here can
    leak into a decision that precedes it.
    """

    ts_ms: int
    coin: str
    open: float
    high: float
    low: float
    close: float
    volume: float                     # base units traded in the bar
    interval_ms: int
    funding_rate: float = 0.0         # hourly rate; positive = longs pay
    mark_px: float | None = None      # falls back to close
    half_spread: float | None = None  # fraction of price, from the book if known
    max_asset_leverage: float | None = None

    @property
    def mark(self) -> float:
        return self.close if self.mark_px is None else self.mark_px

    @property
    def notional_volume(self) -> float:
        return self.volume * self.close

    @property
    def range_fraction(self) -> float:
        """Bar high-low range as a fraction of its close."""
        if self.close <= 0:
            return 0.0
        return max(0.0, (self.high - self.low) / self.close)


@dataclass(frozen=True)
class FillResult:
    """Outcome of one order against one bar."""

    filled_size: float          # base units actually filled (>= 0)
    requested_size: float
    price: float                # average execution price including costs
    reference_price: float      # price before any cost was applied
    fee: float                  # USD
    is_maker: bool
    rejected_reason: str | None = None

    @property
    def notional(self) -> float:
        return self.filled_size * self.price

    @property
    def slippage_cost(self) -> float:
        """USD lost to spread + impact + latency, excluding fees."""
        return abs(self.price - self.reference_price) * self.filled_size

    @property
    def partial(self) -> bool:
        return 0 < self.filled_size < self.requested_size

    @property
    def filled(self) -> bool:
        return self.filled_size > 0


class FillSimulator:
    def __init__(self, config: ExecutionConfig | None = None) -> None:
        self.config = config or SETTINGS.execution

    # -- cost components ---------------------------------------------------

    def half_spread_fraction(self, snapshot: MarketSnapshot) -> float:
        if snapshot.half_spread is not None:
            return max(0.0, snapshot.half_spread)
        return self.config.default_half_spread

    def impact_fraction(self, size: float, snapshot: MarketSnapshot) -> float:
        """Square-root impact in participation terms.

        ``impact = k * sqrt(order_notional / bar_notional)``. A bar with no
        volume gets the worst case rather than a divide-by-zero, because a
        market with no prints is exactly where an order would hurt most.
        """
        bar_notional = snapshot.notional_volume
        order_notional = abs(size) * snapshot.close
        if bar_notional <= 0:
            return self.config.impact_coefficient
        participation = order_notional / bar_notional
        return self.config.impact_coefficient * math.sqrt(participation)

    def latency_fraction(self, snapshot: MarketSnapshot) -> float:
        """Adverse drift over the decision-to-fill delay.

        The bar's own range is the best available estimate of how fast the
        price moves; we charge the share of it that elapses during the
        latency window, always against us.
        """
        if snapshot.interval_ms <= 0:
            return 0.0
        share = min(1.0, self.config.latency_ms / snapshot.interval_ms)
        return 0.5 * snapshot.range_fraction * share

    def max_fillable(self, snapshot: MarketSnapshot) -> float:
        """Base units this bar can absorb from a single order."""
        return max(0.0, snapshot.volume * self.config.max_bar_volume_share)

    # -- the fill ----------------------------------------------------------

    def simulate(
        self,
        *,
        side: Side,
        size: float,
        snapshot: MarketSnapshot,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
    ) -> FillResult:
        """Execute ``size`` base units against ``snapshot``.

        Market orders reference the bar's OPEN -- the first price available
        after the decision was made -- never its close, which would be
        look-ahead.

        Limit orders fill only if the bar traded through the limit, at the
        limit price, paying the maker fee. This ignores queue position and
        so is the most optimistic assumption in the whole system.
        """
        size = abs(size)
        if size <= 0:
            return FillResult(0.0, 0.0, snapshot.open, snapshot.open, 0.0, False, "zero size")

        capacity = self.max_fillable(snapshot)
        if capacity <= 0:
            return FillResult(
                0.0, size, snapshot.open, snapshot.open, 0.0, False, "no volume in bar"
            )
        filled = min(size, capacity)

        if order_type is OrderType.LIMIT:
            if limit_price is None:
                raise ValueError("limit orders require a limit_price")
            crossed = (
                snapshot.low <= limit_price if side is Side.BUY else snapshot.high >= limit_price
            )
            if not crossed:
                return FillResult(
                    0.0, size, limit_price, limit_price, 0.0, True, "limit not reached"
                )
            fee = filled * limit_price * self.config.maker_fee
            return FillResult(filled, size, limit_price, limit_price, fee, True)

        reference = snapshot.open
        cost_fraction = (
            self.half_spread_fraction(snapshot)
            + self.impact_fraction(filled, snapshot)
            + self.latency_fraction(snapshot)
        )
        price = reference * (1.0 + side.sign * cost_fraction)
        fee = filled * price * self.config.taker_fee
        return FillResult(filled, size, price, reference, fee, False)

    def simulate_liquidation(self, *, side: Side, size: float, snapshot: MarketSnapshot) -> FillResult:
        """Force-close at the mark plus a liquidation penalty.

        Liquidations ignore the per-bar volume cap: the exchange closes the
        position whether or not the book can absorb it politely, which is
        precisely why the penalty exists.
        """
        size = abs(size)
        reference = snapshot.mark
        penalty = self.config.liquidation_penalty + self.half_spread_fraction(snapshot)
        price = reference * (1.0 + side.sign * penalty)
        fee = size * price * self.config.taker_fee
        return FillResult(size, size, price, reference, fee, False)
