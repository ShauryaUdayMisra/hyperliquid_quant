"""Account accounting: the numbers must balance before anything else matters."""

from __future__ import annotations

import pytest

from conftest import BASE_MS, frictionless_config, realistic_config
from execution.paper_exchange import (
    DecisionContext,
    Order,
    PaperExchange,
    Position,
)
from execution.simulator import FillSimulator, MarketSnapshot, Side

HOUR = 3_600_000


def snap(price: float, *, ts_ms: int = BASE_MS, funding: float = 0.0, high=None, low=None,
         volume: float = 1e9, max_lev: float | None = 40.0) -> MarketSnapshot:
    return MarketSnapshot(
        ts_ms=ts_ms,
        coin="BTC",
        open=price,
        high=price if high is None else high,
        low=price if low is None else low,
        close=price,
        volume=volume,
        interval_ms=HOUR,
        funding_rate=funding,
        max_asset_leverage=max_lev,
    )


def exchange(config=None, capital: float = 100_000.0) -> PaperExchange:
    config = config or frictionless_config()
    return PaperExchange(capital, config=config, simulator=FillSimulator(config))


def buy(ex: PaperExchange, size: float, s: MarketSnapshot, **kw):
    return ex.submit(Order("BTC", Side.BUY, size, context=DecisionContext(reason="test"), **kw), s)


def sell(ex: PaperExchange, size: float, s: MarketSnapshot, **kw):
    return ex.submit(Order("BTC", Side.SELL, size, context=DecisionContext(reason="test"), **kw), s)


# --------------------------------------------------------------------------
# The core identity
# --------------------------------------------------------------------------

def test_new_account_equity_equals_capital() -> None:
    ex = exchange()
    assert ex.equity({}) == 100_000.0
    assert ex.cash == 100_000.0
    assert ex.open_positions() == []


def test_equity_identity_holds_after_every_operation() -> None:
    """equity == cash + unrealised, always."""
    ex = exchange(realistic_config())
    for price, action in [(100, "buy"), (110, "buy"), (105, "sell"), (95, "sell"), (99, "buy")]:
        s = snap(price)
        (buy if action == "buy" else sell)(ex, 5.0, s)
        marks = {"BTC": price}
        assert ex.equity(marks) == pytest.approx(ex.cash + ex.unrealized_pnl(marks), abs=1e-9)


def test_opening_a_position_does_not_move_cash_when_free() -> None:
    """Perps post margin, they do not spend collateral. Only costs move cash."""
    ex = exchange()
    buy(ex, 10.0, snap(100.0))
    assert ex.cash == 100_000.0
    assert ex.equity({"BTC": 100.0}) == 100_000.0


def test_long_pnl_is_exactly_size_times_price_change() -> None:
    ex = exchange()
    buy(ex, 10.0, snap(100.0))
    assert ex.equity({"BTC": 130.0}) == pytest.approx(100_000.0 + 10 * 30)
    assert ex.equity({"BTC": 70.0}) == pytest.approx(100_000.0 - 10 * 30)


def test_short_pnl_has_the_opposite_sign() -> None:
    ex = exchange()
    sell(ex, 10.0, snap(100.0))
    assert ex.position("BTC").size == -10.0
    assert ex.equity({"BTC": 90.0}) == pytest.approx(100_000.0 + 100)
    assert ex.equity({"BTC": 110.0}) == pytest.approx(100_000.0 - 100)


def test_round_trip_realises_the_paper_profit_into_cash() -> None:
    ex = exchange()
    buy(ex, 10.0, snap(100.0))
    sell(ex, 10.0, snap(120.0))
    assert ex.position("BTC").is_flat
    assert ex.cash == pytest.approx(100_200.0)
    assert ex.equity({"BTC": 500.0}) == pytest.approx(100_200.0)


# --------------------------------------------------------------------------
# Position netting
# --------------------------------------------------------------------------

def test_adding_to_a_long_averages_the_entry() -> None:
    ex = exchange()
    buy(ex, 10.0, snap(100.0))
    buy(ex, 10.0, snap(120.0))
    position = ex.position("BTC")
    assert position.size == 20.0
    assert position.entry_price == pytest.approx(110.0)


def test_partially_closing_realises_only_the_closed_share() -> None:
    ex = exchange()
    buy(ex, 10.0, snap(100.0))
    sell(ex, 4.0, snap(150.0))
    assert ex.position("BTC").size == pytest.approx(6.0)
    assert ex.position("BTC").entry_price == pytest.approx(100.0)
    assert ex.cash == pytest.approx(100_000.0 + 4 * 50)


def test_flipping_closes_the_old_trade_and_opens_a_new_one() -> None:
    ex = exchange()
    buy(ex, 10.0, snap(100.0))
    sell(ex, 25.0, snap(120.0))
    position = ex.position("BTC")
    assert position.size == pytest.approx(-15.0)
    assert position.entry_price == pytest.approx(120.0)
    assert ex.cash == pytest.approx(100_000.0 + 10 * 20)
    assert len(ex.closed_trades) == 1
    assert ex.closed_trades[0].direction == "long"


def test_closed_trade_records_both_ends_of_the_decision() -> None:
    ex = exchange()
    ex.submit(Order("BTC", Side.BUY, 5.0, context=DecisionContext(reason="entry signal")), snap(100.0))
    ex.submit(Order("BTC", Side.SELL, 5.0, context=DecisionContext(reason="exit signal")), snap(110.0))
    trade = ex.closed_trades[0]
    assert trade.open_context.reason == "entry signal"
    assert trade.close_context.reason == "exit signal"
    assert trade.net_pnl == pytest.approx(50.0)
    assert trade.won


def test_closed_trade_nets_fees_and_funding_out_of_pnl() -> None:
    config = realistic_config(default_half_spread=0.0, impact_coefficient=0.0, latency_ms=0)
    ex = exchange(config)
    buy(ex, 10.0, snap(100.0))
    ex.apply_funding(BASE_MS, {"BTC": snap(100.0)})
    ex.apply_funding(BASE_MS + HOUR, {"BTC": snap(100.0, funding=0.0001)})
    sell(ex, 10.0, snap(110.0))
    trade = ex.closed_trades[0]
    assert trade.gross_pnl == pytest.approx(100.0)
    assert trade.fees > 0
    assert trade.funding == pytest.approx(10 * 100 * 0.0001)
    assert trade.net_pnl == pytest.approx(trade.gross_pnl - trade.fees - trade.funding)


# --------------------------------------------------------------------------
# Fees
# --------------------------------------------------------------------------

def test_fees_come_straight_out_of_cash() -> None:
    config = realistic_config(default_half_spread=0.0, impact_coefficient=0.0, latency_ms=0)
    ex = exchange(config)
    buy(ex, 10.0, snap(100.0))
    expected = 10 * 100 * config.taker_fee
    assert ex.cash == pytest.approx(100_000.0 - expected)
    assert ex.total_fees == pytest.approx(expected)


def test_a_round_trip_pays_two_fees() -> None:
    config = realistic_config(default_half_spread=0.0, impact_coefficient=0.0, latency_ms=0)
    ex = exchange(config)
    buy(ex, 10.0, snap(100.0))
    sell(ex, 10.0, snap(100.0))
    assert ex.total_fees == pytest.approx(2 * 10 * 100 * config.taker_fee)
    assert ex.cash == pytest.approx(100_000.0 - ex.total_fees)


def test_slippage_is_tracked_separately_from_fees() -> None:
    ex = exchange(realistic_config())
    buy(ex, 10.0, snap(100.0, high=110.0, low=90.0))
    assert ex.total_slippage > 0
    assert ex.total_fees > 0


# --------------------------------------------------------------------------
# Funding
# --------------------------------------------------------------------------

def test_longs_pay_funding_when_the_rate_is_positive() -> None:
    ex = exchange()
    buy(ex, 10.0, snap(100.0))
    ex.apply_funding(BASE_MS, {"BTC": snap(100.0, funding=0.0001)})
    paid = ex.apply_funding(BASE_MS + HOUR, {"BTC": snap(100.0, funding=0.0001)})
    assert paid == pytest.approx(10 * 100 * 0.0001)
    assert ex.cash == pytest.approx(100_000.0 - paid)


def test_shorts_receive_funding_when_the_rate_is_positive() -> None:
    ex = exchange()
    sell(ex, 10.0, snap(100.0))
    ex.apply_funding(BASE_MS, {"BTC": snap(100.0, funding=0.0001)})
    paid = ex.apply_funding(BASE_MS + HOUR, {"BTC": snap(100.0, funding=0.0001)})
    assert paid == pytest.approx(-10 * 100 * 0.0001)
    assert ex.cash > 100_000.0


def test_funding_settles_once_per_hour_not_once_per_call() -> None:
    ex = exchange()
    buy(ex, 10.0, snap(100.0))
    ex.apply_funding(BASE_MS, {"BTC": snap(100.0, funding=0.0001)})
    charges = [
        ex.apply_funding(BASE_MS + minutes * 60_000, {"BTC": snap(100.0, funding=0.0001)})
        for minutes in (10, 20, 30, 60, 70, 120)
    ]
    # Only the crossings into hour+1 and hour+2 cost anything.
    assert sum(1 for c in charges if c != 0) == 2


def test_a_flat_account_pays_no_funding() -> None:
    ex = exchange()
    ex.apply_funding(BASE_MS, {"BTC": snap(100.0, funding=0.01)})
    assert ex.apply_funding(BASE_MS + HOUR, {"BTC": snap(100.0, funding=0.01)}) == 0.0
    assert ex.cash == 100_000.0


# --------------------------------------------------------------------------
# Liquidation
# --------------------------------------------------------------------------

def levered_long(leverage: float, price: float = 100.0, capital: float = 100_000.0):
    """Open a `leverage`x long and return the exchange."""
    config = frictionless_config()
    ex = PaperExchange(capital, config=config, simulator=FillSimulator(config),
                       max_account_leverage=leverage)
    size = capital * leverage / price
    buy(ex, size, snap(price))
    return ex


def test_a_10x_long_survives_a_small_adverse_move() -> None:
    ex = levered_long(10.0)
    fills = ex.check_liquidation(BASE_MS + HOUR, {"BTC": snap(96.0, low=96.0)})
    assert fills == []
    assert not ex.position("BTC").is_flat


def test_a_10x_long_is_liquidated_by_a_ten_percent_move() -> None:
    """Hyperliquid maintains 1/(2*40) = 1.25% on BTC, so 10x dies near -8.9%."""
    ex = levered_long(10.0)
    fills = ex.check_liquidation(BASE_MS + HOUR, {"BTC": snap(90.0, low=90.0)})
    assert len(fills) == 1
    assert fills[0].is_liquidation
    assert ex.position("BTC").is_flat
    assert ex.liquidation_count == 1


def test_the_10x_liquidation_boundary_sits_just_under_nine_percent() -> None:
    survived = [p for p in range(100, 85, -1) if not levered_long(10.0).check_liquidation(
        BASE_MS + HOUR, {"BTC": snap(float(p), low=float(p))})]
    worst_survivable = min(survived)
    assert 91 <= worst_survivable <= 92, worst_survivable


def test_a_2x_long_survives_what_kills_a_10x_long() -> None:
    ex = levered_long(2.0)
    assert ex.check_liquidation(BASE_MS + HOUR, {"BTC": snap(90.0, low=90.0)}) == []


def test_an_intrabar_wick_liquidates_even_if_the_close_recovers() -> None:
    """A 10x position that touched -12% was gone before the bar closed."""
    ex = levered_long(10.0)
    wick = snap(100.0, low=88.0, high=101.0)
    assert ex.check_liquidation(BASE_MS + HOUR, {"BTC": wick}) != []


def test_ignoring_intrabar_extremes_would_have_missed_it() -> None:
    ex = levered_long(10.0)
    wick = snap(100.0, low=88.0, high=101.0)
    assert ex.check_liquidation(BASE_MS + HOUR, {"BTC": wick}, use_intrabar_worst_case=False) == []


def test_a_short_is_liquidated_by_an_upward_wick() -> None:
    config = frictionless_config()
    ex = PaperExchange(100_000.0, config=config, simulator=FillSimulator(config),
                       max_account_leverage=10.0)
    sell(ex, 10_000.0, snap(100.0))
    assert ex.check_liquidation(BASE_MS + HOUR, {"BTC": snap(105.0, high=112.0)}) != []


def test_liquidation_wipes_the_account_and_flags_bankruptcy() -> None:
    ex = levered_long(10.0)
    ex.check_liquidation(BASE_MS + HOUR, {"BTC": snap(80.0, low=80.0)})
    assert ex.cash == 0.0
    assert ex.bankrupt
    assert ex.equity({"BTC": 80.0}) == 0.0


def test_a_liquidated_trade_is_labelled_as_such() -> None:
    ex = levered_long(10.0)
    ex.check_liquidation(BASE_MS + HOUR, {"BTC": snap(90.0, low=90.0)})
    assert ex.closed_trades[-1].liquidated
    assert "liquidation" in ex.closed_trades[-1].close_context.reason


def test_liquidation_is_not_disabled_by_the_aggressive_profile() -> None:
    """Loosening risk limits must not switch the exchange's backstop off."""
    from config.settings import resolve_risk_profile

    risk = resolve_risk_profile("aggressive")
    assert risk.max_leverage == 10.0
    ex = levered_long(risk.max_leverage)
    assert ex.check_liquidation(BASE_MS + HOUR, {"BTC": snap(89.0, low=89.0)}) != []


# --------------------------------------------------------------------------
# Order guards
# --------------------------------------------------------------------------

def test_the_exchange_rejects_orders_beyond_its_leverage_ceiling() -> None:
    config = frictionless_config()
    ex = PaperExchange(100_000.0, config=config, simulator=FillSimulator(config),
                       max_account_leverage=10.0)
    assert buy(ex, 20_000.0, snap(100.0)) is None  # would be 20x
    assert ex.position("BTC").is_flat
    assert ex.rejections and "margin" in ex.rejections[-1][2]


def test_reduce_only_cannot_open_or_flip_a_position() -> None:
    ex = exchange()
    assert sell(ex, 5.0, snap(100.0), reduce_only=True) is None
    buy(ex, 10.0, snap(100.0))
    assert buy(ex, 5.0, snap(100.0), reduce_only=True) is None
    fill = sell(ex, 50.0, snap(100.0), reduce_only=True)
    assert fill is not None
    assert fill.size == pytest.approx(10.0)
    assert ex.position("BTC").is_flat


def test_negative_order_size_is_a_programming_error() -> None:
    with pytest.raises(ValueError, match="size must be positive"):
        Order("BTC", Side.BUY, -1.0)


def test_rejections_are_recorded_with_a_reason() -> None:
    ex = exchange(realistic_config())
    ex.submit(Order("BTC", Side.BUY, 1.0), snap(100.0, volume=0.0))
    assert ex.rejections[-1][1] == "BTC"
    assert "volume" in ex.rejections[-1][2]


def test_two_exchanges_do_not_share_state() -> None:
    a, b = exchange(), exchange()
    buy(a, 10.0, snap(100.0))
    assert b.position("BTC").is_flat
    assert b.cash == 100_000.0
