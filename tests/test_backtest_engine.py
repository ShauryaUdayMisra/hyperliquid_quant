"""Engine ordering, look-ahead protection, and the buy-and-hold proof."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import BASE_MS, bars_from_prices, frictionless_config, realistic_config
from backtest.engine import BacktestConfig, BacktestEngine, MarketView
from execution.paper_exchange import DecisionContext, Order, PaperExchange
from execution.simulator import FillSimulator, Side
from strategy.base import BaseStrategy
from strategy.baselines import AlwaysLongStrategy, FlatStrategy

HOUR = 3_600_000


def engine_for(bars, config=None, *, exec_config=None, capital=100_000.0, **cfg):
    exec_config = exec_config or frictionless_config()
    exchange = PaperExchange(capital, config=exec_config, simulator=FillSimulator(exec_config))
    return BacktestEngine(
        bars,
        config or BacktestConfig(interval="1h", **cfg),
        exchange=exchange,
    )


# --------------------------------------------------------------------------
# Data alignment
# --------------------------------------------------------------------------

def test_coins_are_aligned_on_shared_timestamps() -> None:
    btc = bars_from_prices([100, 101, 102, 103], coin="BTC")
    eth = bars_from_prices([10, 11, 12], coin="ETH", start_ms=BASE_MS + HOUR)
    eng = engine_for({"BTC": btc, "ETH": eth})
    assert len(eng.bars["BTC"]) == len(eng.bars["ETH"]) == 3
    assert eng.bars["BTC"]["ts_ms"].tolist() == eng.bars["ETH"]["ts_ms"].tolist()


def test_disjoint_histories_are_an_error_not_a_silent_fill() -> None:
    btc = bars_from_prices([100, 101], coin="BTC")
    eth = bars_from_prices([10, 11], coin="ETH", start_ms=BASE_MS + 100 * HOUR)
    with pytest.raises(ValueError, match="share no common timestamps"):
        engine_for({"BTC": btc, "ETH": eth})


def test_missing_columns_are_rejected() -> None:
    bad = bars_from_prices([100, 101]).drop(columns=["volume"])
    with pytest.raises(ValueError, match="missing columns"):
        engine_for({"BTC": bad})


# --------------------------------------------------------------------------
# Look-ahead protection
# --------------------------------------------------------------------------

class Spy(BaseStrategy):
    """Records what it was shown on each bar."""

    name = "spy"

    def __init__(self) -> None:
        self.seen: list[tuple[int, int, float]] = []

    def on_bar(self, view: MarketView):
        history = view.history("BTC")
        self.seen.append((view.bar_index, len(history), float(history["close"].iloc[-1])))
        assert int(history["ts_ms"].iloc[-1]) == view.ts_ms
        return []


def test_a_strategy_only_ever_sees_bars_up_to_the_current_one() -> None:
    prices = [100, 110, 120, 130, 140]
    eng = engine_for({"BTC": bars_from_prices(prices)})
    spy = Spy()
    eng.run(spy)
    # The last bar is never offered: there would be no bar left to trade on.
    assert [s[0] for s in spy.seen] == [0, 1, 2, 3]
    assert [s[1] for s in spy.seen] == [1, 2, 3, 4]
    assert [s[2] for s in spy.seen] == [100.0, 110.0, 120.0, 130.0]


def test_history_respects_the_lookback_window() -> None:
    class Window(BaseStrategy):
        name = "window"

        def __init__(self):
            self.lengths = []

        def on_bar(self, view):
            self.lengths.append(len(view.history("BTC", lookback=3)))
            return []

    eng = engine_for({"BTC": bars_from_prices([100] * 6)})
    strategy = Window()
    eng.run(strategy)
    assert strategy.lengths == [1, 2, 3, 3, 3]


def test_orders_fill_on_the_next_bar_open_never_the_signal_bar() -> None:
    class BuyOnce(BaseStrategy):
        name = "buy_once"

        def __init__(self):
            self.done = False

        def on_bar(self, view):
            if self.done:
                return []
            self.done = True
            return [Order("BTC", Side.BUY, 1.0, context=DecisionContext(reason="once"))]

    # A price spike on bar 1 must be what we pay, not bar 0's 100.
    eng = engine_for({"BTC": bars_from_prices([100, 200, 300])})
    result = eng.run(BuyOnce())
    assert len(result.fills) == 1
    assert result.fills[0].price == pytest.approx(200.0)
    assert result.fills[0].ts_ms == BASE_MS + HOUR


# --------------------------------------------------------------------------
# The dumb-strategy accounting proof
# --------------------------------------------------------------------------

def test_doing_nothing_changes_nothing() -> None:
    eng = engine_for({"BTC": bars_from_prices([100, 150, 80, 120])})
    result = eng.run(FlatStrategy())
    assert (result.equity_curve["equity"] == 100_000.0).all()
    assert result.trades == []
    assert result.fills == []
    assert result.reconcile()["balanced"]


def test_buy_and_hold_pnl_is_exactly_size_times_price_change() -> None:
    """The reference case. If this is wrong, nothing else can be trusted."""
    prices = [100, 100, 110, 120, 130]
    eng = engine_for({"BTC": bars_from_prices(prices)})
    result = eng.run(AlwaysLongStrategy(["BTC"], notional_per_coin=10_000.0))

    size = 10_000.0 / prices[0]        # sized off bar 0's close
    entry = prices[1]                  # filled at bar 1's open
    exit_price = prices[-1]
    expected = 100_000.0 + size * (exit_price - entry)

    assert result.final_equity == pytest.approx(expected)
    assert result.reconcile()["balanced"]
    assert result.exchange.total_fees == 0.0


def test_buy_and_hold_loses_exactly_as_much_when_price_falls() -> None:
    prices = [100, 100, 90, 80, 70]
    eng = engine_for({"BTC": bars_from_prices(prices)})
    result = eng.run(AlwaysLongStrategy(["BTC"], notional_per_coin=10_000.0))
    assert result.final_equity == pytest.approx(100_000.0 + 100 * (70 - 100))


def test_buy_and_hold_across_three_markets_adds_up() -> None:
    bars = {
        "BTC": bars_from_prices([100, 100, 110], coin="BTC"),
        "ETH": bars_from_prices([50, 50, 40], coin="ETH"),
        "SOL": bars_from_prices([10, 10, 10], coin="SOL"),
    }
    eng = engine_for(bars)
    result = eng.run(AlwaysLongStrategy(notional_per_coin=10_000.0))
    expected = 100_000.0 + 100 * 10 + 200 * (-10) + 0
    assert result.final_equity == pytest.approx(expected)
    assert result.reconcile()["balanced"]


def test_the_books_balance_with_full_costs_switched_on() -> None:
    prices = [100 + 10 * np.sin(i / 5) for i in range(200)]
    eng = engine_for(
        {"BTC": bars_from_prices(prices, range_fraction=0.02, funding_rate=0.0001, volume=5_000.0)},
        exec_config=realistic_config(),
    )
    result = eng.run(AlwaysLongStrategy(["BTC"], notional_per_coin=10_000.0))
    checks = result.reconcile()
    assert checks["balanced"], checks
    assert checks["fees"] > 0
    assert checks["funding"] != 0
    assert checks["slippage"] > 0


def test_costs_make_buy_and_hold_strictly_worse_than_the_frictionless_case() -> None:
    prices = [100, 100, 110, 120]
    clean = engine_for({"BTC": bars_from_prices(prices, funding_rate=0.0001)}).run(
        AlwaysLongStrategy(["BTC"], notional_per_coin=10_000.0)
    )
    costly = engine_for(
        {"BTC": bars_from_prices(prices, range_fraction=0.02, funding_rate=0.0001, volume=1_000.0)},
        exec_config=realistic_config(),
    ).run(AlwaysLongStrategy(["BTC"], notional_per_coin=10_000.0))
    assert costly.final_equity < clean.final_equity


def test_funding_alone_bleeds_a_held_position() -> None:
    prices = [100.0] * 48
    eng = engine_for({"BTC": bars_from_prices(prices, funding_rate=0.0001)})
    result = eng.run(AlwaysLongStrategy(["BTC"], notional_per_coin=10_000.0))
    # 100 units held from bar 1; funding settles on each hourly bar after that.
    assert result.final_equity < 100_000.0
    assert result.exchange.total_funding > 0
    assert result.reconcile()["balanced"]


# --------------------------------------------------------------------------
# Liquidation inside a run
# --------------------------------------------------------------------------

def test_a_levered_long_is_liquidated_on_a_grinding_decline() -> None:
    from strategy.baselines import AlwaysLongLeveredStrategy

    prices = [100, 100] + [100 - i for i in range(1, 30)]
    eng = engine_for({"BTC": bars_from_prices(prices)}, asset_max_leverage={"BTC": 40.0})
    result = eng.run(AlwaysLongLeveredStrategy(["BTC"], leverage=10.0))

    assert result.exchange.liquidation_count == 1
    assert result.exchange.position("BTC").is_flat
    # Liquidation triggers at the maintenance-margin boundary, roughly -9%
    # on a 10x long, so ~90% of the account is gone.
    assert -0.95 < result.total_return < -0.85
    # The remaining balance is a residual, not a debt: that buffer is the
    # entire reason exchanges liquidate before equity reaches zero.
    assert 0 < result.final_equity < 15_000
    assert not result.exchange.bankrupt


def test_a_gap_through_the_liquidation_price_wipes_the_account_and_halts() -> None:
    """No chance to liquidate on the way down: equity goes negative."""
    from strategy.baselines import AlwaysLongLeveredStrategy

    prices = [100, 100, 80, 80, 80]
    eng = engine_for({"BTC": bars_from_prices(prices)}, asset_max_leverage={"BTC": 40.0})
    result = eng.run(AlwaysLongLeveredStrategy(["BTC"], leverage=10.0))

    assert result.exchange.liquidation_count == 1
    assert result.exchange.bankrupt
    assert result.final_equity == 0.0
    assert result.halted_reason is not None
    # The run stops rather than trading on from a zeroed account.
    assert result.bars_processed < len(prices)


def test_the_same_path_survives_at_two_times_leverage() -> None:
    from strategy.baselines import AlwaysLongLeveredStrategy

    prices = [100, 100] + [100 - i for i in range(1, 30)]
    eng = engine_for({"BTC": bars_from_prices(prices)}, asset_max_leverage={"BTC": 40.0})
    result = eng.run(AlwaysLongLeveredStrategy(["BTC"], leverage=2.0))
    assert result.exchange.liquidation_count == 0
    assert not result.exchange.bankrupt


# --------------------------------------------------------------------------
# Bookkeeping details
# --------------------------------------------------------------------------

def test_equity_curve_carries_marks_positions_and_costs() -> None:
    eng = engine_for({"BTC": bars_from_prices([100, 100, 110])})
    result = eng.run(AlwaysLongStrategy(["BTC"], notional_per_coin=10_000.0))
    curve = result.equity_curve
    for column in ("ts_ms", "ts", "equity", "cash", "unrealized_pnl", "mark_BTC", "pos_BTC"):
        assert column in curve.columns
    assert curve["pos_BTC"].iloc[0] == 0.0
    assert curve["pos_BTC"].iloc[-1] == pytest.approx(100.0)


def test_warmup_bars_suppress_early_trading() -> None:
    eng = engine_for({"BTC": bars_from_prices([100] * 10)}, warmup_bars=5)
    result = eng.run(AlwaysLongStrategy(["BTC"], notional_per_coin=10_000.0))
    assert result.fills[0].ts_ms == BASE_MS + 6 * HOUR


def test_every_fill_carries_its_decision_context() -> None:
    eng = engine_for({"BTC": bars_from_prices([100, 100, 110])})
    result = eng.run(AlwaysLongStrategy(["BTC"], notional_per_coin=10_000.0))
    context = result.fills[0].context
    assert "always-long" in context.reason
    assert context.target_notional == 10_000.0
    assert context.risk_checks
