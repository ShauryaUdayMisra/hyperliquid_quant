"""Phase 3: features must be point-in-time correct, or nothing downstream counts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import BASE_MS, synthetic_bars, synthetic_book, synthetic_universe
from config.settings import INTERVAL_MS
from features import cross_asset, funding as fund, momentum, orderbook, regime, volatility, volume
from features.base import rolling_percentile, rolling_z
from features.pipeline import (
    FeatureConfig,
    assert_point_in_time,
    build_for_coin,
    build_universe,
    coverage,
    feature_columns,
)

HOUR = INTERVAL_MS["1h"]


@pytest.fixture(scope="module")
def universe():
    return synthetic_universe(900)


@pytest.fixture(scope="module")
def matrix(universe):
    return build_universe(universe)["BTC"]


# ==========================================================================
# THE headline test
# ==========================================================================

def test_no_feature_depends_on_a_future_bar(universe) -> None:
    """Recompute on truncated history; every value at t must be unchanged.

    If any feature reads forward, hiding the future changes its value here.
    """
    report = assert_point_in_time(universe, coin="BTC", checkpoints=4)
    leaking = sorted({c for row in report["mismatches"] for c in row})
    assert not leaking, f"look-ahead detected in: {leaking}"
    assert report["features_checked"].min() > 50


def test_the_causality_check_actually_catches_a_leak(universe) -> None:
    """A guard that cannot fail proves nothing, so plant a leak and detect it."""
    import features.momentum as momentum_module

    original = momentum_module.compute

    def leaky(bars, *args, **kwargs):
        out = original(bars, *args, **kwargs)
        # The classic mistake: tomorrow's return, available today.
        out["mom_tomorrow"] = bars["close"].shift(-1) / bars["close"] - 1.0
        return out

    momentum_module.compute = leaky
    try:
        report = assert_point_in_time(universe, coin="BTC", checkpoints=2)
        leaking = sorted({c for row in report["mismatches"] for c in row})
        assert "mom_tomorrow" in leaking
    finally:
        momentum_module.compute = original


def test_no_feature_column_uses_a_negative_shift() -> None:
    """Static backstop: shift(-n) outside label construction is a leak."""
    import pathlib

    offenders = []
    for path in pathlib.Path("features").glob("*.py"):
        text = path.read_text()
        for lineno, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if ".shift(-" in stripped:
                offenders.append(f"{path}:{lineno}")
    assert not offenders, f"backward-looking shift in feature code: {offenders}"


# ==========================================================================
# Momentum
# ==========================================================================

def test_return_features_match_a_hand_computation() -> None:
    bars = synthetic_bars(300, funding_rate=None)
    out = momentum.compute(bars)
    close = bars["close"]
    expected = close.iloc[100] / close.iloc[100 - 24] - 1.0
    assert out["mom_ret_24"].iloc[100] == pytest.approx(expected)


def test_rsi_stays_inside_its_bounds() -> None:
    out = momentum.compute(synthetic_bars(400, funding_rate=None))
    rsi = out["mom_rsi_14"].dropna()
    assert len(rsi) > 100
    assert rsi.between(0.0, 1.0).all()


def test_rsi_saturates_high_on_an_unbroken_rally() -> None:
    rising = pd.DataFrame({
        "ts_ms": [BASE_MS + i * HOUR for i in range(80)],
        "close": np.linspace(100, 200, 80),
    })
    assert momentum.rsi(rising["close"], 14).iloc[-1] > 99


def test_efficiency_ratio_is_one_for_a_straight_line() -> None:
    straight = pd.Series(np.linspace(100, 200, 100))
    assert momentum.efficiency_ratio(straight, 24).iloc[-1] == pytest.approx(1.0)


def test_efficiency_ratio_collapses_when_price_oscillates() -> None:
    zigzag = pd.Series([100 + (5 if i % 2 else -5) for i in range(100)])
    assert momentum.efficiency_ratio(zigzag, 24).iloc[-1] < 0.2


def test_percent_of_range_is_bounded() -> None:
    out = momentum.compute(synthetic_bars(900, funding_rate=None))
    values = out["mom_pct_of_range_168"].dropna()
    assert values.between(0.0, 1.0).all()


# ==========================================================================
# Volatility
# ==========================================================================

def test_realized_vol_matches_the_annualised_definition() -> None:
    bars = synthetic_bars(300, funding_rate=None)
    result = volatility.realized_vol(bars["close"], 24)
    returns = np.log(bars["close"] / bars["close"].shift(1))
    expected = returns.iloc[100 - 23 : 101].std(ddof=1) * np.sqrt(365.25 * 24)
    assert result.iloc[100] == pytest.approx(expected)


def test_a_calmer_series_reports_lower_volatility() -> None:
    calm = synthetic_bars(400, vol=0.002, seed=1, funding_rate=None)
    wild = synthetic_bars(400, vol=0.05, seed=1, funding_rate=None)
    assert (
        volatility.compute(calm)["vol_rv_24"].dropna().mean()
        < volatility.compute(wild)["vol_rv_24"].dropna().mean()
    )


def test_garman_klass_needs_no_future_bar_but_uses_the_whole_current_one() -> None:
    bars = synthetic_bars(200, funding_rate=None)
    full = volatility.garman_klass_vol(bars, 24)
    truncated = volatility.garman_klass_vol(bars.iloc[:150], 24)
    assert full.iloc[149] == pytest.approx(truncated.iloc[149])


def test_vol_percentile_is_a_trailing_rank() -> None:
    out = volatility.compute(synthetic_bars(900, funding_rate=None))
    ranks = out["vol_vol_percentile_168"].dropna()
    assert ranks.between(0.0, 1.0).all()


def test_atr_is_expressed_as_a_fraction_of_price() -> None:
    out = volatility.compute(synthetic_bars(300, funding_rate=None))
    atr = out["vol_atr_14"].dropna()
    assert (atr > 0).all() and (atr < 1).all()


# ==========================================================================
# Volume
# ==========================================================================

def test_volume_zscore_is_standardised_against_the_past_only() -> None:
    bars = synthetic_bars(400, funding_rate=None)
    out = volume.compute(bars)
    manual = rolling_z(bars["volume"], 24)
    assert out["vol_flow_volume_z_24"].iloc[200] == pytest.approx(manual.iloc[200])


def test_close_location_is_bounded_and_signed() -> None:
    out = volume.compute(synthetic_bars(300, funding_rate=None))
    location = out["vol_flow_close_location"].dropna()
    assert location.between(-1.0, 1.0).all()


def test_a_volume_spike_shows_up_as_a_high_zscore() -> None:
    bars = synthetic_bars(300, funding_rate=None)
    bars.loc[250, "volume"] = bars["volume"].iloc[:250].mean() * 50
    out = volume.compute(bars)
    assert out["vol_flow_volume_z_24"].iloc[250] > 3


# ==========================================================================
# Funding -- the merge is the leak risk
# ==========================================================================

def test_funding_merge_never_pulls_a_rate_from_the_future() -> None:
    """A rate published at bar t+1 must be invisible at bar t."""
    bars = pd.DataFrame({
        "ts_ms": [BASE_MS + i * HOUR for i in range(5)],
        "close": [100.0] * 5,
        "high": [100.0] * 5, "low": [100.0] * 5, "open": [100.0] * 5,
        "volume": [1.0] * 5,
    })
    funding = pd.DataFrame({
        "ts_ms": [BASE_MS, BASE_MS + 3 * HOUR],
        "coin": "BTC",
        "funding_rate": [0.0001, 0.0009],
        "premium": [0.0, 0.0],
    })
    merged = fund.attach_funding_to_bars(bars, funding)
    # Bars 0-2 only know the first rate; the 0.0009 arrives at bar 3.
    assert merged["funding_rate"].tolist() == [0.0001, 0.0001, 0.0001, 0.0009, 0.0009]


def test_funding_before_the_first_print_is_nan_not_zero() -> None:
    bars = pd.DataFrame({
        "ts_ms": [BASE_MS + i * HOUR for i in range(3)],
        "close": [100.0] * 3, "high": [100.0] * 3,
        "low": [100.0] * 3, "open": [100.0] * 3, "volume": [1.0] * 3,
    })
    funding = pd.DataFrame({
        "ts_ms": [BASE_MS + 2 * HOUR], "coin": "BTC",
        "funding_rate": [0.0005], "premium": [0.0],
    })
    merged = fund.attach_funding_to_bars(bars, funding)
    assert merged["funding_rate"].isna().iloc[:2].all()
    assert merged["funding_rate"].iloc[2] == 0.0005


def test_cumulative_funding_is_the_carry_actually_paid() -> None:
    bars = synthetic_bars(200)
    bars["funding_rate"] = 0.0001
    out = fund.compute(bars)
    assert out["fund_cum_24"].iloc[100] == pytest.approx(0.0001 * 24)


def test_funding_sign_persistence_saturates_when_one_sided() -> None:
    bars = synthetic_bars(200)
    bars["funding_rate"] = 0.0002
    out = fund.compute(bars)
    assert out["fund_sign_persistence_72"].iloc[150] == pytest.approx(1.0)


def test_missing_funding_column_yields_an_empty_block() -> None:
    bars = synthetic_bars(100, funding_rate=None)
    assert fund.compute(bars).empty


# ==========================================================================
# Order book
# ==========================================================================

def test_book_imbalance_is_positive_when_bids_dominate() -> None:
    bars = synthetic_bars(50)
    book = synthetic_book(bars.iloc[:50])
    metrics = orderbook.snapshot_metrics(book)
    assert (metrics["imbalance_5"] > 0).mean() > 0.9


def test_spread_is_positive_and_small() -> None:
    metrics = orderbook.snapshot_metrics(synthetic_book(synthetic_bars(20)))
    assert (metrics["spread_bps"] > 0).all()
    assert metrics["spread_bps"].median() < 100


def test_snapshots_land_in_the_bar_that_contains_them() -> None:
    bars = synthetic_bars(10)
    book = synthetic_book(bars)
    metrics = orderbook.snapshot_metrics(book)
    aligned = orderbook.aggregate_to_bars(metrics, bars["ts_ms"], HOUR)
    assert len(aligned) == len(bars)
    assert (aligned["snapshot_count"] == 6).all()


def test_absent_order_book_data_leaves_the_block_empty_not_zeroed() -> None:
    """No collector ran, so these features are unknown -- not neutral."""
    bars = synthetic_bars(50)
    assert orderbook.compute(bars, None, HOUR).empty
    assert orderbook.compute(bars, pd.DataFrame(), HOUR).empty


def test_order_book_features_reach_the_matrix_when_data_exists() -> None:
    bars = synthetic_bars(200)
    book = synthetic_book(bars)
    result = build_for_coin("BTC", bars, book_snapshots=book)
    ob_columns = [c for c in result.columns if c.startswith("ob_")]
    assert ob_columns
    assert result["ob_imbalance_5_mean"].notna().mean() > 0.9


def test_malformed_book_frame_is_rejected() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        orderbook.snapshot_metrics(pd.DataFrame({"ts_ms": [1], "coin": ["BTC"]}))


# ==========================================================================
# Cross-asset
# ==========================================================================

def test_correlation_with_the_benchmark_is_bounded(universe) -> None:
    out = cross_asset.compute("ETH", universe)
    correlations = out["cross_corr_btc_168"].dropna()
    assert correlations.between(-1.0, 1.0).all()


def test_a_coin_is_perfectly_correlated_with_a_copy_of_itself() -> None:
    bars = synthetic_bars(400, coin="BTC")
    twin = bars.copy()
    twin["coin"] = "ETH"
    out = cross_asset.compute("ETH", {"BTC": bars, "ETH": twin})
    assert out["cross_corr_btc_168"].dropna().iloc[-1] == pytest.approx(1.0, abs=1e-6)
    assert out["cross_beta_btc_168"].dropna().iloc[-1] == pytest.approx(1.0, abs=1e-6)


def test_relative_strength_is_the_return_difference(universe) -> None:
    out = cross_asset.compute("SOL", universe)
    sol, btc = universe["SOL"]["close"], universe["BTC"]["close"]
    expected = (sol.iloc[300] / sol.iloc[276] - 1) - (btc.iloc[300] / btc.iloc[276] - 1)
    assert out["cross_rel_strength_24"].iloc[300] == pytest.approx(expected)


def test_misaligned_coins_are_rejected_rather_than_silently_correlated() -> None:
    a = synthetic_bars(200, coin="BTC")
    b = synthetic_bars(150, coin="ETH")
    with pytest.raises(ValueError, match="not aligned"):
        cross_asset.compute("ETH", {"BTC": a, "ETH": b})


# ==========================================================================
# Regime
# ==========================================================================

def test_regime_labels_are_from_the_known_set(matrix) -> None:
    assert set(matrix["regime"]) <= set(regime.REGIMES)


def test_a_steady_climb_is_labelled_trending_up() -> None:
    n = 900
    bars = pd.DataFrame({
        "ts_ms": [BASE_MS + i * HOUR for i in range(n)],
        "close": np.linspace(100, 300, n),
    })
    bars["open"] = bars["close"]
    bars["high"] = bars["close"] * 1.001
    bars["low"] = bars["close"] * 0.999
    bars["volume"] = 100.0
    labels = regime.classify(bars)
    assert labels["regime"].iloc[-1] == "trending_up"


def test_a_choppy_market_is_labelled_ranging() -> None:
    n = 900
    close = 100 + np.array([3 if i % 2 else -3 for i in range(n)], dtype=float)
    bars = pd.DataFrame({
        "ts_ms": [BASE_MS + i * HOUR for i in range(n)],
        "close": close, "open": close,
        "high": close + 0.5, "low": close - 0.5, "volume": 100.0,
    })
    assert regime.classify(bars)["regime"].iloc[-1] in {"ranging", "high_volatility"}


def test_regime_is_unknown_before_enough_history_exists() -> None:
    labels = regime.classify(synthetic_bars(50, funding_rate=None))
    assert (labels["regime"] == "unknown").all()


def test_regime_description_is_plain_english(matrix) -> None:
    labelled = matrix[matrix["regime"] != "unknown"]
    text = regime.describe(labelled.iloc[-1])
    assert any(word in text for word in ("trending", "range-bound", "volatile"))
    assert "efficiency" in text


# ==========================================================================
# Pipeline
# ==========================================================================

def test_matrix_carries_metadata_and_a_useful_number_of_features(matrix) -> None:
    for column in ("ts_ms", "coin", "close", "regime"):
        assert column in matrix.columns
    assert len(feature_columns(matrix)) > 50


def test_metadata_is_excluded_from_model_features(matrix) -> None:
    columns = feature_columns(matrix)
    for excluded in ("ts_ms", "close", "coin", "regime"):
        assert excluded not in columns


def test_no_infinities_survive_into_the_matrix(matrix) -> None:
    numeric = matrix[feature_columns(matrix)]
    assert not np.isinf(numeric.to_numpy(dtype="float64")).any()


def test_most_features_are_populated_after_warmup(matrix) -> None:
    report = coverage(matrix)
    assert (report["non_null"] > 0.9).mean() > 0.8, report.head(15).to_string()


def test_every_coin_gets_its_own_matrix(universe) -> None:
    matrices = build_universe(universe)
    assert set(matrices) == set(universe)
    for coin, frame in matrices.items():
        assert (frame["coin"] == coin).all()
        assert len(frame) == len(universe[coin])


def test_duplicate_timestamps_are_rejected() -> None:
    bars = synthetic_bars(100)
    doubled = pd.concat([bars, bars.iloc[[50]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate timestamps"):
        build_for_coin("BTC", doubled)


def test_pipeline_works_without_cross_asset_data() -> None:
    bars = synthetic_bars(300)
    result = build_for_coin("BTC", bars, config=FeatureConfig(include_cross_asset=False))
    assert not [c for c in result.columns if c.startswith("cross_")]
    assert len(feature_columns(result)) > 40


# ==========================================================================
# Uniform schema across coins (regression: the benchmark could never trade)
# ==========================================================================

def test_every_coin_matrix_has_identical_columns(universe) -> None:
    """The benchmark lacks cross_*_btc_* naturally; alignment fills them NaN.

    Without this, inference reads BTC's matrix, finds 8 model features
    missing, and silently refuses to ever produce a BTC signal.
    """
    matrices = build_universe(universe)
    schemas = {coin: list(m.columns) for coin, m in matrices.items()}
    reference = schemas["BTC"]
    for coin, columns in schemas.items():
        assert columns == reference, f"{coin} has a different column set"


def test_the_benchmarks_self_referential_features_are_nan_not_absent(universe) -> None:
    matrices = build_universe(universe)
    btc = matrices["BTC"]
    assert "cross_corr_btc_168" in btc.columns
    assert btc["cross_corr_btc_168"].isna().all()
    # Other coins still carry real values there.
    assert matrices["ETH"]["cross_corr_btc_168"].notna().any()


def test_alignment_matches_what_training_stacked(universe) -> None:
    """Inference must see the same schema training saw."""
    from models.dataset import assemble
    from models.labels import LabelConfig

    matrices = build_universe(universe)
    stacked = assemble(matrices, LabelConfig())
    for coin, frame in matrices.items():
        assert set(feature_columns(frame)) == set(feature_columns(stacked))
