"""Phase 4: labels, chronological splits, and a model that cannot cheat."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import synthetic_universe
from features.pipeline import EXCLUDED_COLUMNS, build_universe, feature_columns
from models.backend import ModelParams, make_backend
from models.dataset import (
    HoldoutViolation,
    SplitConfig,
    assemble,
    development_and_holdout,
    prepare_xy,
    usable_rows,
    walk_forward_folds,
)
from models.labels import (
    LABEL_COLUMNS,
    LabelConfig,
    class_balance,
    forward_return,
    labelled_rows,
    make_labels,
)
from models.predict import SignalGenerator
from models.train import (
    LabelLeakError,
    TrainedModel,
    classification_metrics,
    evaluate_holdout,
    plausibility_warnings,
    render_report,
    train_walk_forward,
)


@pytest.fixture(scope="module")
def dataset():
    return assemble(build_universe(synthetic_universe(2500)), LabelConfig())


@pytest.fixture(scope="module")
def trained(dataset):
    model, holdout = train_walk_forward(
        dataset, split_config=SplitConfig(n_folds=3), params=ModelParams(n_estimators=120)
    )
    return model, holdout


# ==========================================================================
# Labels
# ==========================================================================

def test_forward_return_looks_exactly_horizon_bars_ahead() -> None:
    close = pd.Series([100.0, 110.0, 121.0, 133.1])
    result = forward_return(close, 2)
    assert result.iloc[0] == pytest.approx(0.21)
    assert pd.isna(result.iloc[2]) and pd.isna(result.iloc[3])


def test_label_fires_only_above_the_threshold() -> None:
    matrix = pd.DataFrame({
        "ts_ms": [0, 1, 2, 3],
        "close": [100.0, 100.2, 101.0, 100.0],
        "coin": "BTC",
    })
    labelled = make_labels(matrix, LabelConfig(horizon_bars=1, threshold=0.005))
    # +0.2% misses the 0.5% bar; +0.8% clears it.
    assert labelled["label"].tolist()[:2] == [0, 1]


def test_the_last_horizon_rows_have_no_known_label() -> None:
    matrix = pd.DataFrame({"ts_ms": range(20), "close": np.linspace(100, 120, 20), "coin": "BTC"})
    labelled = make_labels(matrix, LabelConfig(horizon_bars=4))
    assert (~labelled["label_known"]).sum() == 4
    assert (labelled.loc[~labelled["label_known"], "label"] == -1).all()
    assert len(labelled_rows(labelled)) == 16


def test_a_down_label_is_the_mirror_of_an_up_label() -> None:
    matrix = pd.DataFrame({"ts_ms": range(5), "close": [100, 90, 100, 90, 100.0], "coin": "BTC"})
    up = make_labels(matrix, LabelConfig(horizon_bars=1, threshold=0.05, direction="up"))
    down = make_labels(matrix, LabelConfig(horizon_bars=1, threshold=0.05, direction="down"))
    known = up["label_known"]
    assert (up.loc[known, "label"] != down.loc[known, "label"]).all()


@pytest.mark.parametrize("kwargs", [{"horizon_bars": 0}, {"threshold": -0.1}, {"direction": "sideways"}])
def test_nonsense_label_config_is_rejected(kwargs) -> None:
    with pytest.raises(ValueError):
        LabelConfig(**kwargs)


def test_class_balance_reports_a_usable_positive_rate(dataset) -> None:
    balance = class_balance(dataset)
    assert balance["rows"] > 1000
    assert 0.05 < balance["positive_rate"] < 0.95


# ==========================================================================
# THE leak that actually happened
# ==========================================================================

def test_label_columns_never_become_features(dataset) -> None:
    """Regression: `label` was once a feature and scored a perfect 1.000 AUC."""
    columns = feature_columns(dataset)
    assert not set(columns) & set(LABEL_COLUMNS)
    for name in LABEL_COLUMNS:
        assert name in EXCLUDED_COLUMNS


def test_the_trainer_refuses_a_leaked_label_even_if_selection_slips(dataset, monkeypatch) -> None:
    """The guard must be independent of the column filter it backs up."""
    import models.train as train_module

    monkeypatch.setattr(
        train_module, "feature_columns", lambda df: ["label", "mom_ret_4"]
    )
    with pytest.raises(LabelLeakError, match="trained on its own answer"):
        train_walk_forward(dataset)


def test_close_price_is_not_a_feature(dataset) -> None:
    """Raw price is non-stationary and encodes the date. Returns, not levels."""
    assert "close" not in feature_columns(dataset)


# ==========================================================================
# Splits
# ==========================================================================

def test_development_ends_before_the_holdout_begins(dataset) -> None:
    development, holdout = development_and_holdout(usable_rows(dataset), SplitConfig())
    assert development["ts_ms"].max() < holdout.span[0]


def test_an_embargo_separates_development_from_the_holdout(dataset) -> None:
    """A training label reaching into the test period would contaminate it."""
    config = SplitConfig(embargo_bars=24)
    development, holdout = development_and_holdout(usable_rows(dataset), config, horizon_bars=4)
    gap_ms = holdout.span[0] - int(development["ts_ms"].max())
    assert gap_ms >= (24 + 4) * 3_600_000


def test_the_holdout_cannot_be_read_before_the_model_is_locked(dataset) -> None:
    _, holdout = development_and_holdout(usable_rows(dataset), SplitConfig())
    with pytest.raises(HoldoutViolation, match="before the model was locked"):
        holdout.release()
    holdout.lock_model()
    assert len(holdout.release()) == holdout.rows


def test_walk_forward_validation_always_follows_training(dataset) -> None:
    development, _ = development_and_holdout(usable_rows(dataset), SplitConfig(n_folds=3))
    folds = list(walk_forward_folds(development, SplitConfig(n_folds=3), horizon_bars=4))
    assert len(folds) >= 2
    for train, validate, info in folds:
        assert train["ts_ms"].max() < validate["ts_ms"].min()
        assert info["val_start_ts"] - info["train_end_ts"] >= 4 * 3_600_000


def test_the_training_window_expands_across_folds(dataset) -> None:
    development, _ = development_and_holdout(usable_rows(dataset), SplitConfig(n_folds=3))
    sizes = [len(t) for t, _, _ in walk_forward_folds(development, SplitConfig(n_folds=3))]
    assert sizes == sorted(sizes)


def test_all_coins_are_cut_at_the_same_instant(dataset) -> None:
    development, holdout = development_and_holdout(usable_rows(dataset), SplitConfig())
    released = holdout.release.__self__._data
    per_coin_start = released.groupby("coin")["ts_ms"].min()
    assert per_coin_start.nunique() == 1


def test_prepare_xy_keeps_nans_rather_than_inventing_values(dataset) -> None:
    clean = usable_rows(dataset)
    X, y = prepare_xy(clean, feature_columns(dataset))
    assert X.isna().to_numpy().any()
    assert set(np.unique(y)) <= {0, 1}


# ==========================================================================
# Training on noise
# ==========================================================================

def test_a_random_walk_yields_no_edge(trained) -> None:
    """Synthetic data has no signal. Anything much above 0.5 is a leak."""
    model, _ = trained
    assert 0.42 < model.mean_val_auc < 0.60, (
        f"AUC {model.mean_val_auc:.3f} on pure noise indicates look-ahead"
    )


def test_noise_training_raises_the_no_edge_warning(trained) -> None:
    model, _ = trained
    warnings = plausibility_warnings(model)
    assert any("barely above chance" in w or "no better than" in w for w in warnings)


def test_every_fold_validates_on_out_of_sample_rows(trained) -> None:
    model, _ = trained
    assert len(model.fold_results) >= 2
    for result in model.fold_results:
        assert result.info["val_rows"] > 0
        assert result.info["train_end_ts"] < result.info["val_start_ts"]


def test_the_model_records_what_it_was_asked(trained) -> None:
    model, _ = trained
    assert model.label_config.name.startswith("P(return_")
    assert model.backend_name in {"lightgbm", "sklearn_hist"}
    assert len(model.features) > 50


def test_an_implausibly_high_auc_is_flagged(trained) -> None:
    """The warning that caught the real leak must still fire."""
    model, _ = trained
    model.fold_results[0].val_metrics["auc"] = 0.99
    for result in model.fold_results[1:]:
        result.val_metrics["auc"] = 0.99
    assert any("implausibly high" in w for w in plausibility_warnings(model))


def test_holdout_is_scored_only_after_locking(trained) -> None:
    model, holdout = trained
    assert not holdout.was_released
    metrics = evaluate_holdout(model, holdout)
    assert holdout.was_released
    assert metrics["rows"] > 0
    assert model.holdout_metrics is not None


def test_report_is_readable_and_states_the_warnings(trained) -> None:
    model, _ = trained
    text = render_report(model)
    assert "Walk-forward folds" in text
    assert "mean validation AUC" in text
    assert "WARNINGS" in text or "No plausibility warnings" in text


# ==========================================================================
# Metrics
# ==========================================================================

def test_a_perfect_predictor_scores_auc_one() -> None:
    y = np.array([0, 0, 1, 1])
    assert classification_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))["auc"] == 1.0


def test_a_useless_predictor_has_no_log_loss_lift() -> None:
    y = np.array([0, 1] * 50)
    metrics = classification_metrics(y, np.full(100, 0.5))
    assert metrics["log_loss_lift"] == pytest.approx(0.0, abs=1e-9)


def test_auc_is_undefined_when_only_one_class_is_present() -> None:
    metrics = classification_metrics(np.zeros(10), np.linspace(0, 1, 10))
    assert np.isnan(metrics["auc"])


# ==========================================================================
# Signals
# ==========================================================================

def test_signal_carries_probability_confidence_and_drivers(trained, dataset) -> None:
    model, _ = trained
    generator = SignalGenerator(model)
    generator.calibrate_feature_scales(usable_rows(dataset))
    matrix = usable_rows(dataset)
    matrix = matrix[matrix["coin"] == "BTC"]

    signal = generator.latest(matrix, coin="BTC")
    assert 0.0 <= signal.probability <= 1.0
    assert 0.0 <= signal.confidence <= 1.0
    assert signal.direction in {"long", "short", "flat"}
    assert signal.top_features
    assert signal.explanation_method in {"shap", "importance_weighted"}
    assert "P(up)" in signal.describe()


def test_confidence_is_measured_against_the_base_rate() -> None:
    from models.predict import Signal

    # With a 20% base rate, 0.35 is a strong lean, not a weak one.
    strong = Signal("BTC", 0, probability=0.35, base_rate=0.20, direction="long")
    neutral = Signal("BTC", 0, probability=0.20, base_rate=0.20, direction="flat")
    assert strong.confidence > 0.15
    assert neutral.confidence == pytest.approx(0.0)
    assert strong.edge > 0


def test_without_a_down_model_a_low_probability_means_flat_not_short(trained, dataset) -> None:
    """Low P(up) also covers 'goes nowhere'. Shorting it is an assumption."""
    model, _ = trained
    generator = SignalGenerator(model, long_threshold=0.99)
    matrix = usable_rows(dataset).iloc[-5:]
    assert {s.direction for s in generator.generate(matrix)} == {"flat"}


def test_a_missing_feature_column_is_an_error_not_a_guess(trained, dataset) -> None:
    model, _ = trained
    matrix = usable_rows(dataset).drop(columns=model.features[:3])
    with pytest.raises(ValueError, match="missing"):
        SignalGenerator(model).generate(matrix.iloc[-2:])


# ==========================================================================
# Persistence
# ==========================================================================

def test_a_saved_model_predicts_identically_after_reload(trained, dataset, tmp_path) -> None:
    model, _ = trained
    path = model.save(tmp_path / "model.pkl")
    reloaded = TrainedModel.load(path)

    sample = usable_rows(dataset).iloc[-50:]
    X = sample[model.features].astype("float64")
    assert np.allclose(model.predict_proba(X), reloaded.predict_proba(X))
    assert reloaded.features == model.features
    assert reloaded.backend_name == model.backend_name
