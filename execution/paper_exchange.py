"""The paper exchange: positions, margin, funding, liquidation, audit trail.

Simulated cross-margin perpetuals, modelled on Hyperliquid's rules:

* one USD collateral pool backs every position (cross margin),
* account value = cash + unrealised P&L,
* each position requires maintenance margin of ``notional / (2 * max
  leverage)`` -- Hyperliquid's "half the initial margin at max leverage",
* when account value falls below total maintenance margin the WHOLE account
  is liquidated, not just the offending position,
* funding is exchanged hourly on the position's notional.

This module holds no keys, speaks to no network, and cannot place a real
order. It is the only place account state is mutated.

Every fill carries the ``DecisionContext`` that produced it -- features,
model confidence, regime, and the risk check -- so any trade in the ledger
can be explained after the fact.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from config.settings import SETTINGS, ExecutionConfig
from execution.simulator import (
    FillResult,
    FillSimulator,
    MarketSnapshot,
    OrderType,
    Side,
)

log = logging.getLogger(__name__)

HOUR_MS = 3_600_000


@dataclass
class DecisionContext:
    """Why a trade happened. Attached to every order and every fill."""

    reason: str = ""
    model_probability: float | None = None
    model_confidence: float | None = None
    regime: str | None = None
    features: dict[str, float] = field(default_factory=dict)
    risk_checks: list[str] = field(default_factory=list)
    target_notional: float | None = None

    def summary(self) -> str:
        bits = [self.reason or "unspecified"]
        if self.model_probability is not None:
            bits.append(f"p={self.model_probability:.3f}")
        if self.regime:
            bits.append(f"regime={self.regime}")
        if self.features:
            top = sorted(self.features.items(), key=lambda kv: -abs(kv[1]))[:4]
            bits.append("features=" + ", ".join(f"{k}{v:+.3f}" for k, v in top))
        if self.risk_checks:
            bits.append("risk=" + "; ".join(self.risk_checks))
        return " | ".join(bits)


@dataclass
class Order:
    coin: str
    side: Side
    size: float                       # base units, always positive
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    reduce_only: bool = False
    context: DecisionContext = field(default_factory=DecisionContext)

    def __post_init__(self) -> None:
        if self.size < 0:
            raise ValueError("order size must be positive; use side to express direction")


@dataclass
class Fill:
    ts_ms: int
    coin: str
    side: Side
    size: float
    price: float
    fee: float
    slippage_cost: float
    realized_pnl: float
    is_maker: bool
    is_liquidation: bool
    context: DecisionContext

    @property
    def notional(self) -> float:
        return self.size * self.price


@dataclass
class Position:
    coin: str
    size: float = 0.0            # signed base units; negative = short
    entry_price: float = 0.0
    opened_ts_ms: int | None = None
    fees_paid: float = 0.0
    funding_paid: float = 0.0    # positive = we paid out
    realized_pnl: float = 0.0
    max_abs_size: float = 0.0
    #: The decision that opened this position, kept so a closed trade
    #: can explain both ends of the round trip.
    open_context: DecisionContext = field(default_factory=DecisionContext)

    @property
    def is_flat(self) -> bool:
        return abs(self.size) < 1e-12

    @property
    def direction(self) -> str:
        if self.is_flat:
            return "flat"
        return "long" if self.size > 0 else "short"

    def notional(self, price: float) -> float:
        return abs(self.size) * price

    def unrealized_pnl(self, price: float) -> float:
        return self.size * (price - self.entry_price)


@dataclass
class ClosedTrade:
    """A completed round trip, emitted when a position returns to flat."""

    coin: str
    direction: str
    size: float                  # base units at peak
    entry_price: float
    exit_price: float
    opened_ts_ms: int
    closed_ts_ms: int
    gross_pnl: float
    fees: float
    funding: float
    net_pnl: float
    liquidated: bool
    open_context: DecisionContext
    close_context: DecisionContext

    @property
    def holding_ms(self) -> int:
        return self.closed_ts_ms - self.opened_ts_ms

    @property
    def won(self) -> bool:
        return self.net_pnl > 0


class InsufficientMargin(RuntimeError):
    pass


class PaperExchange:
    """Simulated cross-margin perp account. Virtual money only."""

    def __init__(
        self,
        starting_capital: float | None = None,
        *,
        config: ExecutionConfig | None = None,
        simulator: FillSimulator | None = None,
        max_account_leverage: float | None = None,
    ) -> None:
        self.config = config or SETTINGS.execution
        self.simulator = simulator or FillSimulator(self.config)
        self.starting_capital = (
            SETTINGS.risk.starting_capital if starting_capital is None else starting_capital
        )
        #: Ceiling the *exchange* enforces on gross notional. The risk engine
        #: applies its own, stricter, limit before an order ever gets here.
        self.max_account_leverage = max_account_leverage

        self.cash: float = self.starting_capital
        self.positions: dict[str, Position] = {}
        self.fills: list[Fill] = []
        self.closed_trades: list[ClosedTrade] = []
        self.rejections: list[tuple[int, str, str]] = []

        self.total_fees: float = 0.0
        self.total_funding: float = 0.0
        self.total_slippage: float = 0.0
        self.liquidation_count: int = 0
        self.bankrupt: bool = False
        self._last_funding_hour: int | None = None

    # -- account state -----------------------------------------------------

    def position(self, coin: str) -> Position:
        return self.positions.setdefault(coin, Position(coin=coin))

    def open_positions(self) -> list[Position]:
        return [p for p in self.positions.values() if not p.is_flat]

    def unrealized_pnl(self, marks: Mapping[str, float]) -> float:
        return sum(
            p.unrealized_pnl(marks[p.coin])
            for p in self.open_positions()
            if p.coin in marks
        )

    def equity(self, marks: Mapping[str, float]) -> float:
        """Account value = collateral + unrealised P&L."""
        return self.cash + self.unrealized_pnl(marks)

    def gross_notional(self, marks: Mapping[str, float]) -> float:
        return sum(p.notional(marks[p.coin]) for p in self.open_positions() if p.coin in marks)

    def leverage(self, marks: Mapping[str, float]) -> float:
        equity = self.equity(marks)
        return 0.0 if equity <= 0 else self.gross_notional(marks) / equity

    def maintenance_margin(
        self, marks: Mapping[str, float], leverages: Mapping[str, float] | None = None
    ) -> float:
        total = 0.0
        for position in self.open_positions():
            if position.coin not in marks:
                continue
            max_lev = (leverages or {}).get(position.coin)
            fraction = self.config.maintenance_margin_fraction(max_lev)
            total += position.notional(marks[position.coin]) * fraction
        return total

    # -- order handling ----------------------------------------------------

    def submit(self, order: Order, snapshot: MarketSnapshot, *, ts_ms: int | None = None) -> Fill | None:
        """Execute an order against a bar. Returns the Fill, or None if unfilled."""
        ts_ms = snapshot.ts_ms if ts_ms is None else ts_ms
        position = self.position(order.coin)

        size = order.size
        if order.reduce_only:
            # Never let a reduce-only order flip the position.
            closing_capacity = abs(position.size)
            if closing_capacity <= 0:
                return self._reject(ts_ms, order, "reduce-only with no position")
            wrong_way = (order.side is Side.BUY) == (position.size > 0)
            if wrong_way:
                return self._reject(ts_ms, order, "reduce-only would increase position")
            size = min(size, closing_capacity)

        result = self.simulator.simulate(
            side=order.side,
            size=size,
            snapshot=snapshot,
            order_type=order.order_type,
            limit_price=order.limit_price,
        )
        if not result.filled:
            return self._reject(ts_ms, order, result.rejected_reason or "unfilled")

        if not order.reduce_only and not self._margin_allows(order, result, snapshot):
            return self._reject(ts_ms, order, "insufficient margin at exchange leverage cap")

        return self._book(ts_ms, order, result, snapshot)

    def _margin_allows(self, order: Order, result: FillResult, snapshot: MarketSnapshot) -> bool:
        """Reject orders that would exceed the exchange's leverage ceiling."""
        cap = self.max_account_leverage or snapshot.max_asset_leverage
        if cap is None:
            cap = self.config.default_max_asset_leverage
        position = self.position(order.coin)
        prospective = abs(position.size + order.side.sign * result.filled_size) * result.price
        others = sum(
            p.notional(snapshot.mark)
            for p in self.open_positions()
            if p.coin != order.coin
        )
        equity = self.cash + self.unrealized_pnl({order.coin: snapshot.mark})
        if equity <= 0:
            return False
        return (prospective + others) <= equity * cap + 1e-9

    def _reject(self, ts_ms: int, order: Order, why: str) -> None:
        self.rejections.append((ts_ms, order.coin, why))
        log.debug("order rejected (%s %s %s): %s", order.coin, order.side.value, order.size, why)
        return None

    def _book(
        self,
        ts_ms: int,
        order: Order,
        result: FillResult,
        snapshot: MarketSnapshot,
        *,
        is_liquidation: bool = False,
    ) -> Fill:
        position = self.position(order.coin)
        signed = order.side.sign * result.filled_size
        realized = self._apply_to_position(position, signed, result.price, ts_ms, order.context)

        self.cash += realized - result.fee
        position.fees_paid += result.fee
        position.realized_pnl += realized
        self.total_fees += result.fee
        self.total_slippage += result.slippage_cost

        fill = Fill(
            ts_ms=ts_ms,
            coin=order.coin,
            side=order.side,
            size=result.filled_size,
            price=result.price,
            fee=result.fee,
            slippage_cost=result.slippage_cost,
            realized_pnl=realized,
            is_maker=result.is_maker,
            is_liquidation=is_liquidation,
            context=order.context,
        )
        self.fills.append(fill)
        log.debug(
            "fill %s %s %.6f @ %.2f fee=%.2f pnl=%.2f | %s",
            order.coin, order.side.value, result.filled_size, result.price,
            result.fee, realized, order.context.summary(),
        )
        return fill

    def _apply_to_position(
        self,
        position: Position,
        signed_size: float,
        price: float,
        ts_ms: int,
        context: DecisionContext,
    ) -> float:
        """Net a fill into a position and return the realised P&L."""
        if position.is_flat:
            position.size = signed_size
            position.entry_price = price
            position.opened_ts_ms = ts_ms
            position.max_abs_size = abs(signed_size)
            position.fees_paid = 0.0
            position.funding_paid = 0.0
            position.realized_pnl = 0.0
            position.open_context = context
            return 0.0

        same_direction = (position.size > 0) == (signed_size > 0)
        if same_direction:
            total = abs(position.size) + abs(signed_size)
            position.entry_price = (
                position.entry_price * abs(position.size) + price * abs(signed_size)
            ) / total
            position.size += signed_size
            position.max_abs_size = max(position.max_abs_size, abs(position.size))
            return 0.0

        closing = min(abs(signed_size), abs(position.size))
        direction = 1.0 if position.size > 0 else -1.0
        realized = closing * (price - position.entry_price) * direction
        remainder = abs(signed_size) - closing
        entry_before = position.entry_price
        opened_before = position.opened_ts_ms
        was_direction = position.direction
        peak_size = position.max_abs_size
        fees_before = position.fees_paid
        funding_before = position.funding_paid

        position.size += signed_size
        if abs(position.size) < 1e-12:
            position.size = 0.0

        if position.is_flat or remainder > 0:
            self._emit_closed_trade(
                position,
                direction=was_direction,
                size=peak_size,
                entry_price=entry_before,
                exit_price=price,
                opened_ts_ms=opened_before or ts_ms,
                closed_ts_ms=ts_ms,
                gross_pnl=realized,
                fees=fees_before,
                funding=funding_before,
                open_context=position.open_context,
                close_context=context,
            )

        if remainder > 0:
            # Position flipped: the remainder opens a fresh trade.
            position.entry_price = price
            position.opened_ts_ms = ts_ms
            position.max_abs_size = remainder
            position.fees_paid = 0.0
            position.funding_paid = 0.0
            position.realized_pnl = 0.0
            position.open_context = context
        elif position.is_flat:
            position.entry_price = 0.0
            position.opened_ts_ms = None
            position.max_abs_size = 0.0

        return realized

    def _emit_closed_trade(self, position: Position, **kwargs: Any) -> None:
        liquidated = kwargs.pop("liquidated", False)
        gross = kwargs["gross_pnl"]
        fees = kwargs["fees"]
        funding = kwargs["funding"]
        trade = ClosedTrade(
            coin=position.coin,
            net_pnl=gross - fees - funding,
            liquidated=liquidated,
            **kwargs,
        )
        self.closed_trades.append(trade)

    # -- funding -----------------------------------------------------------

    def apply_funding(self, ts_ms: int, snapshots: Mapping[str, MarketSnapshot]) -> float:
        """Exchange funding once per hour boundary. Returns USD paid (>0 = we paid)."""
        hour = ts_ms // HOUR_MS
        if self._last_funding_hour is None:
            self._last_funding_hour = hour
            return 0.0
        if hour <= self._last_funding_hour:
            return 0.0
        self._last_funding_hour = hour

        paid = 0.0
        for position in self.open_positions():
            snapshot = snapshots.get(position.coin)
            if snapshot is None:
                continue
            # Longs pay when the rate is positive; shorts receive.
            payment = position.size * snapshot.mark * snapshot.funding_rate
            self.cash -= payment
            position.funding_paid += payment
            paid += payment
        self.total_funding += paid
        if paid:
            log.debug("funding settled at %s: %.4f USD", ts_ms, paid)
        return paid

    # -- liquidation -------------------------------------------------------

    def check_liquidation(
        self,
        ts_ms: int,
        snapshots: Mapping[str, MarketSnapshot],
        *,
        leverages: Mapping[str, float] | None = None,
        use_intrabar_worst_case: bool = True,
    ) -> list[Fill]:
        """Liquidate the account if maintenance margin is breached.

        With ``use_intrabar_worst_case`` the check uses each bar's adverse
        extreme (low for longs, high for shorts) rather than its close. A
        10x position that touched -9% intrabar and recovered by the close
        was liquidated in reality, and pretending otherwise is exactly the
        kind of optimism that makes a leveraged backtest look survivable
        when it was not.
        """
        if not self.open_positions():
            return []

        # Each snapshot knows its asset's exchange max leverage, which sets
        # the maintenance-margin fraction. An explicit mapping wins, but
        # falling back to the global default when the snapshot already knows
        # the answer would quietly double the margin requirement.
        effective_leverages: dict[str, float] = {}
        stress_marks: dict[str, float] = {}
        for position in self.open_positions():
            snapshot = snapshots.get(position.coin)
            if snapshot is None:
                continue
            explicit = (leverages or {}).get(position.coin)
            resolved = explicit if explicit is not None else snapshot.max_asset_leverage
            if resolved is not None:
                effective_leverages[position.coin] = resolved
            if use_intrabar_worst_case:
                stress_marks[position.coin] = (
                    snapshot.low if position.size > 0 else snapshot.high
                )
            else:
                stress_marks[position.coin] = snapshot.mark

        equity = self.equity(stress_marks)
        required = self.maintenance_margin(stress_marks, effective_leverages)
        if equity >= required:
            return []

        log.warning(
            "LIQUIDATION at %s: stressed equity %.2f < maintenance margin %.2f",
            ts_ms, equity, required,
        )
        self.liquidation_count += 1
        return self._liquidate_all(ts_ms, snapshots)

    def _liquidate_all(self, ts_ms: int, snapshots: Mapping[str, MarketSnapshot]) -> list[Fill]:
        fills = []
        for position in list(self.open_positions()):
            snapshot = snapshots.get(position.coin)
            if snapshot is None:
                continue
            side = Side.SELL if position.size > 0 else Side.BUY
            result = self.simulator.simulate_liquidation(
                side=side, size=abs(position.size), snapshot=snapshot
            )
            context = DecisionContext(
                reason="forced liquidation: maintenance margin breached",
                risk_checks=["exchange liquidation"],
            )
            order = Order(position.coin, side, abs(position.size), reduce_only=True, context=context)
            trades_before = len(self.closed_trades)
            fill = self._book(ts_ms, order, result, snapshot, is_liquidation=True)
            for trade in self.closed_trades[trades_before:]:
                trade.liquidated = True
            fills.append(fill)

        if self.cash < 0:
            # Hyperliquid's insurance fund absorbs the shortfall; the trader
            # is simply wiped out. Record it rather than carrying a negative.
            log.error("account bankrupt: cash %.2f clamped to zero", self.cash)
            self.bankrupt = True
            self.cash = 0.0
        return fills

    # -- reporting ---------------------------------------------------------

    def snapshot_state(self, marks: Mapping[str, float]) -> dict[str, Any]:
        return {
            "cash": self.cash,
            "equity": self.equity(marks),
            "unrealized_pnl": self.unrealized_pnl(marks),
            "gross_notional": self.gross_notional(marks),
            "leverage": self.leverage(marks),
            "open_positions": len(self.open_positions()),
            "total_fees": self.total_fees,
            "total_funding": self.total_funding,
            "total_slippage": self.total_slippage,
            "liquidations": self.liquidation_count,
            "bankrupt": self.bankrupt,
        }
