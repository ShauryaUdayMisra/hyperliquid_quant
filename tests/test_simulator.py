"""The cost model: pessimistic, deterministic, and in the right direction."""

from __future__ import annotations

import math

import pytest

from conftest import BASE_MS, frictionless_config, realistic_config
from execution.simulator import FillSimulator, MarketSnapshot, OrderType, Side


def snapshot(
    *, open_=100.0, high=None, low=None, close=None, volume=1_000.0, interval_ms=3_600_000, **kw
) -> MarketSnapshot:
    close = open_ if close is None else close
    return MarketSnapshot(
        ts_ms=BASE_MS,
        coin="BTC",
        open=open_,
        high=open_ if high is None else high,
        low=open_ if low is None else low,
        close=close,
        volume=volume,
        interval_ms=interval_ms,
        **kw,
    )


def test_frictionless_market_order_fills_at_the_open() -> None:
    sim = FillSimulator(frictionless_config())
    result = sim.simulate(side=Side.BUY, size=1.0, snapshot=snapshot())
    assert result.price == 100.0
    assert result.fee == 0.0
    assert result.filled_size == 1.0
    assert result.slippage_cost == 0.0


def test_market_orders_reference_the_open_not_the_close() -> None:
    """Filling at the close of the bar you decided on is look-ahead."""
    sim = FillSimulator(frictionless_config())
    result = sim.simulate(side=Side.BUY, size=1.0, snapshot=snapshot(open_=100.0, close=150.0))
    assert result.price == 100.0


@pytest.mark.parametrize("side,expected", [(Side.BUY, 1), (Side.SELL, -1)])
def test_costs_always_push_the_price_against_us(side: Side, expected: int) -> None:
    sim = FillSimulator(realistic_config())
    result = sim.simulate(side=side, size=1.0, snapshot=snapshot(high=101, low=99))
    assert math.copysign(1, result.price - 100.0) == expected
    assert result.slippage_cost > 0


def test_half_spread_is_charged_exactly_once() -> None:
    config = realistic_config(impact_coefficient=0.0, latency_ms=0, default_half_spread=0.001)
    sim = FillSimulator(config)
    result = sim.simulate(side=Side.BUY, size=1.0, snapshot=snapshot())
    assert result.price == pytest.approx(100.0 * 1.001)


def test_book_spread_overrides_the_default_when_known() -> None:
    config = realistic_config(impact_coefficient=0.0, latency_ms=0, default_half_spread=0.001)
    sim = FillSimulator(config)
    result = sim.simulate(side=Side.BUY, size=1.0, snapshot=snapshot(half_spread=0.005))
    assert result.price == pytest.approx(100.0 * 1.005)


def test_impact_grows_with_the_square_root_of_participation() -> None:
    config = realistic_config(default_half_spread=0.0, latency_ms=0, max_bar_volume_share=1.0)
    sim = FillSimulator(config)
    snap = snapshot(volume=10_000.0)
    small = sim.impact_fraction(100.0, snap)
    quadrupled = sim.impact_fraction(400.0, snap)
    assert quadrupled == pytest.approx(2 * small)


def test_a_bar_with_no_volume_gets_the_worst_case_not_a_free_fill() -> None:
    config = realistic_config()
    sim = FillSimulator(config)
    assert sim.impact_fraction(1.0, snapshot(volume=0.0)) == config.impact_coefficient


def test_latency_charges_a_share_of_the_bar_range() -> None:
    config = realistic_config(
        default_half_spread=0.0, impact_coefficient=0.0, latency_ms=360_000
    )
    sim = FillSimulator(config)
    # 10% bar range, 360s latency on a 3600s bar => 0.5 * 0.10 * 0.1 = 0.5%.
    snap = snapshot(high=105.0, low=95.0)
    assert sim.latency_fraction(snap) == pytest.approx(0.005)


def test_latency_cost_is_zero_on_a_flat_bar() -> None:
    sim = FillSimulator(realistic_config())
    assert sim.latency_fraction(snapshot()) == 0.0


def test_order_larger_than_the_volume_cap_fills_partially() -> None:
    config = realistic_config(max_bar_volume_share=0.10)
    sim = FillSimulator(config)
    result = sim.simulate(side=Side.BUY, size=500.0, snapshot=snapshot(volume=1_000.0))
    assert result.filled_size == pytest.approx(100.0)
    assert result.partial
    assert result.requested_size == 500.0


def test_order_inside_the_cap_fills_completely() -> None:
    sim = FillSimulator(realistic_config(max_bar_volume_share=0.10))
    result = sim.simulate(side=Side.BUY, size=50.0, snapshot=snapshot(volume=1_000.0))
    assert result.filled_size == 50.0
    assert not result.partial


def test_no_volume_means_no_fill_at_all() -> None:
    sim = FillSimulator(realistic_config())
    result = sim.simulate(side=Side.BUY, size=1.0, snapshot=snapshot(volume=0.0))
    assert not result.filled
    assert result.rejected_reason == "no volume in bar"


def test_taker_fee_is_charged_on_notional() -> None:
    config = realistic_config(default_half_spread=0.0, impact_coefficient=0.0, latency_ms=0)
    sim = FillSimulator(config)
    result = sim.simulate(side=Side.BUY, size=2.0, snapshot=snapshot())
    assert result.fee == pytest.approx(2.0 * 100.0 * config.taker_fee)
    assert not result.is_maker


def test_limit_order_fills_at_the_limit_and_pays_the_maker_fee() -> None:
    config = realistic_config()
    sim = FillSimulator(config)
    result = sim.simulate(
        side=Side.BUY,
        size=1.0,
        snapshot=snapshot(high=101.0, low=98.0),
        order_type=OrderType.LIMIT,
        limit_price=99.0,
    )
    assert result.price == 99.0
    assert result.is_maker
    assert result.fee == pytest.approx(99.0 * config.maker_fee)


def test_limit_order_that_never_traded_does_not_fill() -> None:
    sim = FillSimulator(realistic_config())
    result = sim.simulate(
        side=Side.BUY,
        size=1.0,
        snapshot=snapshot(high=101.0, low=100.0),
        order_type=OrderType.LIMIT,
        limit_price=95.0,
    )
    assert not result.filled
    assert result.rejected_reason == "limit not reached"


def test_limit_without_a_price_is_a_programming_error() -> None:
    sim = FillSimulator(realistic_config())
    with pytest.raises(ValueError, match="limit_price"):
        sim.simulate(side=Side.BUY, size=1.0, snapshot=snapshot(), order_type=OrderType.LIMIT)


def test_liquidation_ignores_the_volume_cap_but_pays_a_penalty() -> None:
    config = realistic_config(max_bar_volume_share=0.01, liquidation_penalty=0.01)
    sim = FillSimulator(config)
    result = sim.simulate_liquidation(side=Side.SELL, size=10_000.0, snapshot=snapshot(volume=1.0))
    assert result.filled_size == 10_000.0
    assert result.price < 100.0 * (1 - 0.01) + 1e-9


def test_the_model_is_deterministic() -> None:
    sim = FillSimulator(realistic_config())
    snap = snapshot(high=103.0, low=97.0, volume=500.0)
    prices = {sim.simulate(side=Side.BUY, size=3.0, snapshot=snap).price for _ in range(20)}
    assert len(prices) == 1


def test_zero_size_is_rejected_not_silently_filled() -> None:
    sim = FillSimulator(realistic_config())
    assert not sim.simulate(side=Side.BUY, size=0.0, snapshot=snapshot()).filled
