"""Analytics: correct annualisation, honest NaNs, exact drawdowns."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import BASE_MS
from backtest.metrics import (
    compute_metrics,
    max_drawdown,
    metrics_by_regime,
    periods_per_year,
    sharpe_ratio,
    sortino_ratio,
)
from execution.paper_exchange import ClosedTrade, DecisionContext

HOUR = 3_600_000
DAY = 86_400_000


def curve(equities, *, interval_ms: int = HOUR) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_ms": [BASE_MS + i * interval_ms for i in range(len(equities))],
            "equity": [float(e) for e in equities],
            "cash": [float(e) for e in equities],
            "unrealized_pnl": [0.0] * len(equities),
        }
    )


def trade(pnl: float, *, coin: str = "BTC", liquidated: bool = False) -> ClosedTrade:
    context = DecisionContext(reason="test")
    return ClosedTrade(
        coin=coin,
        direction="long",
        size=1.0,
        entry_price=100.0,
        exit_price=100.0 + pnl,
        opened_ts_ms=BASE_MS,
        closed_ts_ms=BASE_MS + HOUR,
        gross_pnl=pnl,
        fees=0.0,
        funding=0.0,
        net_pnl=pnl,
        liquidated=liquidated,
        open_context=context,
        close_context=context,
    )


# -- annualisation ---------------------------------------------------------

def test_periods_per_year_matches_the_bar_size() -> None:
    assert periods_per_year(DAY) == pytest.approx(365.25)
    assert periods_per_year(HOUR) == pytest.approx(365.25 * 24)
    assert periods_per_year(60_000) == pytest.approx(365.25 * 24 * 60)


def test_sharpe_scales_with_the_square_root_of_frequency() -> None:
    returns = np.array([0.01, -0.005, 0.02, 0.0, -0.01, 0.015])
    hourly = sharpe_ratio(returns, periods_per_year(HOUR))
    daily = sharpe_ratio(returns, periods_per_year(DAY))
    assert hourly / daily == pytest.approx(np.sqrt(24))


def test_a_constant_return_series_has_undefined_sharpe_not_infinite() -> None:
    """Float noise in an otherwise constant series must not become a Sharpe."""
    assert np.isnan(sharpe_ratio(np.array([0.01] * 10), 365.25))
    assert np.isnan(sharpe_ratio(np.zeros(10), 365.25))


def test_sortino_ignores_upside_volatility() -> None:
    steady = np.array([0.01, 0.01, -0.01, 0.01, 0.01])
    spiky = np.array([0.05, 0.05, -0.01, 0.05, 0.05])
    assert sortino_ratio(spiky, 365.25) > sortino_ratio(steady, 365.25)


def test_sortino_is_undefined_when_nothing_ever_lost() -> None:
    assert np.isnan(sortino_ratio(np.array([0.01, 0.02, 0.03]), 365.25))


# -- drawdown --------------------------------------------------------------

def test_max_drawdown_is_measured_peak_to_trough() -> None:
    depth, duration = max_drawdown([100, 120, 90, 95, 130])
    assert depth == pytest.approx(0.25)
    # Underwater at 90 and 95; the 130 makes a new peak.
    assert duration == 2


def test_a_monotonic_curve_has_no_drawdown() -> None:
    depth, duration = max_drawdown([100, 110, 120])
    assert depth == 0.0
    assert duration == 0


def test_a_wipeout_is_a_hundred_percent_drawdown() -> None:
    depth, _ = max_drawdown([100_000, 50_000, 0])
    assert depth == pytest.approx(1.0)


# -- aggregate metrics -----------------------------------------------------

def test_flat_equity_produces_zero_return_and_no_ratios() -> None:
    metrics = compute_metrics(
        curve([100_000] * 100), [], [], interval_ms=HOUR, starting_equity=100_000
    )
    assert metrics.total_return == 0.0
    assert metrics.max_drawdown == 0.0
    assert np.isnan(metrics.sharpe)
    assert metrics.trades == 0


def test_total_return_and_pnl_agree() -> None:
    metrics = compute_metrics(
        curve([100_000, 105_000, 110_000]), [], [], interval_ms=HOUR, starting_equity=100_000
    )
    assert metrics.total_pnl == pytest.approx(10_000)
    assert metrics.total_return == pytest.approx(0.10)


def test_cagr_compounds_over_the_measured_span() -> None:
    equities = [100_000 * (1.001 ** i) for i in range(366)]
    metrics = compute_metrics(
        curve(equities, interval_ms=DAY), [], [], interval_ms=DAY, starting_equity=100_000
    )
    assert metrics.cagr == pytest.approx(0.44, abs=0.02)


def test_a_wiped_out_account_reports_minus_one_hundred_percent_cagr() -> None:
    metrics = compute_metrics(
        curve([100_000, 50_000, 0]), [], [], interval_ms=DAY,
        starting_equity=100_000, bankrupt=True, liquidations=1,
    )
    assert metrics.cagr == -1.0
    assert metrics.total_return == -1.0
    assert metrics.bankrupt
    assert "WIPED OUT" in metrics.describe()


def test_trade_statistics_are_computed_from_closed_trades() -> None:
    trades = [trade(100), trade(-50), trade(200), trade(-25)]
    metrics = compute_metrics(
        curve([100_000, 100_225]), trades, [], interval_ms=HOUR, starting_equity=100_000
    )
    assert metrics.trades == 4
    assert metrics.win_rate == pytest.approx(0.5)
    assert metrics.profit_factor == pytest.approx(300 / 75)
    assert metrics.average_trade == pytest.approx(56.25)
    assert metrics.largest_win == 200
    assert metrics.largest_loss == -50


def test_profit_factor_is_undefined_when_nothing_lost() -> None:
    metrics = compute_metrics(
        curve([100_000, 100_300]), [trade(100), trade(200)], [],
        interval_ms=HOUR, starting_equity=100_000,
    )
    assert np.isnan(metrics.profit_factor)


def test_exposure_counts_bars_with_a_position_open() -> None:
    frame = curve([100_000] * 10)
    frame["pos_BTC"] = [0.0] * 5 + [1.0] * 5
    metrics = compute_metrics(frame, [], [], interval_ms=HOUR, starting_equity=100_000)
    assert metrics.exposure == pytest.approx(0.5)


def test_describe_is_readable_and_flags_missing_values() -> None:
    text = compute_metrics(
        curve([100_000] * 10), [], [], interval_ms=HOUR, starting_equity=100_000
    ).describe()
    assert "Sharpe" in text and "n/a" in text
    assert "max drawdown" in text


# -- regimes ---------------------------------------------------------------

def test_regime_breakdown_splits_the_curve_by_label() -> None:
    frame = curve([100, 110, 121, 121, 110, 99])
    regimes = pd.Series(["trend"] * 3 + ["chop"] * 3)
    table = metrics_by_regime(frame, regimes, interval_ms=DAY)
    assert set(table["regime"]) == {"trend", "chop"}
    assert table["bars"].sum() == 6
    assert table["share"].sum() == pytest.approx(1.0)
    trend = table.loc[table["regime"] == "trend"].iloc[0]
    chop = table.loc[table["regime"] == "chop"].iloc[0]
    assert trend["total_return"] > 0 > chop["total_return"]


def test_regime_breakdown_survives_missing_labels() -> None:
    frame = curve([100, 101, 102, 103])
    regimes = pd.Series(["trend", None, None, "chop"])
    table = metrics_by_regime(frame, regimes, interval_ms=DAY)
    assert table["bars"].sum() == 4


def test_empty_input_gives_an_empty_table_not_a_crash() -> None:
    assert metrics_by_regime(pd.DataFrame(), pd.Series(dtype=str), interval_ms=HOUR).empty
