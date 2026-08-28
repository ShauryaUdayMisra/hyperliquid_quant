"""Walk-forward training and honest evaluation.

The output of this module is a *probability*, never a BUY or a SELL. The
strategy layer decides what to do with it, and the risk engine can veto
that decision. Keeping the model's job to "estimate P(move)" is what makes
the rest of the system auditable.

Evaluation is deliberately unflattering:

* the headline number is validation AUC across walk-forward folds, not
  training AUC,
* log loss is compared against the base rate, so a model that has learned
  nothing shows up as no better than always guessing the average,
* calibration is reported, because a probability that is not calibrated is
  not a probability,
* :func:`plausibility_warnings` flags results that are too good, since on
  financial data an AUC of 0.7 means a bug far more often than an edge.
"""

from __future__ import annotations

import logging
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.settings import SETTINGS
from features.pipeline import feature_columns
from models.backend import Backend, ModelParams, make_backend
from models.dataset import (
    LockedHoldout,
    SplitConfig,
    development_and_holdout,
    prepare_xy,
    usable_rows,
    walk_forward_folds,
)
from models.labels import LABEL_COLUMNS, LabelConfig, class_balance

log = logging.getLogger(__name__)

#: Above this validation AUC on financial data, suspect a leak before
#: celebrating. Published cross-sectional equity models rarely clear 0.55.
SUSPICIOUS_AUC = 0.65


class LabelLeakError(RuntimeError):
    """Raised when a future-derived column reaches the feature matrix."""


def classification_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        log_loss,
        roc_auc_score,
    )

    y_true = np.asarray(y_true)
    base_rate = float(y_true.mean()) if len(y_true) else float("nan")

    metrics: dict[str, float] = {
        "rows": int(len(y_true)),
        "base_rate": base_rate,
        "mean_prediction": float(np.mean(probabilities)) if len(probabilities) else float("nan"),
    }
    # AUC is undefined when only one class is present -- report it as such
    # rather than letting sklearn raise mid-run.
    if len(np.unique(y_true)) < 2:
        metrics.update(auc=float("nan"), average_precision=float("nan"),
                       log_loss=float("nan"), brier=float("nan"), log_loss_lift=float("nan"))
        return metrics

    metrics["auc"] = float(roc_auc_score(y_true, probabilities))
    metrics["average_precision"] = float(average_precision_score(y_true, probabilities))
    metrics["log_loss"] = float(log_loss(y_true, np.clip(probabilities, 1e-6, 1 - 1e-6)))
    metrics["brier"] = float(brier_score_loss(y_true, probabilities))

    # What a model that only knows the base rate would score. Anything at or
    # below zero lift means the features contributed nothing.
    baseline = np.full_like(probabilities, base_rate, dtype=float)
    baseline_loss = float(log_loss(y_true, np.clip(baseline, 1e-6, 1 - 1e-6)))
    metrics["baseline_log_loss"] = baseline_loss
    metrics["log_loss_lift"] = baseline_loss - metrics["log_loss"]
    return metrics


def calibration_table(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Predicted probability versus realised frequency, by decile."""
    frame = pd.DataFrame({"p": probabilities, "y": y_true})
    try:
        frame["bucket"] = pd.qcut(frame["p"], bins, duplicates="drop")
    except ValueError:
        return pd.DataFrame(columns=["bucket", "count", "mean_predicted", "actual_rate"])
    grouped = frame.groupby("bucket", observed=True)
    return pd.DataFrame(
        {
            "count": grouped.size(),
            "mean_predicted": grouped["p"].mean(),
            "actual_rate": grouped["y"].mean(),
        }
    ).reset_index()


@dataclass
class FoldResult:
    info: dict[str, Any]
    train_metrics: dict[str, float]
    val_metrics: dict[str, float]

    @property
    def overfit_gap(self) -> float:
        """Train AUC minus validation AUC. A wide gap means memorisation."""
        train_auc = self.train_metrics.get("auc", float("nan"))
        val_auc = self.val_metrics.get("auc", float("nan"))
        return train_auc - val_auc


@dataclass
class TrainedModel:
    """The saved artefact. Everything needed to reproduce and audit a prediction."""

    backend: Backend
    features: list[str]
    label_config: LabelConfig
    params: ModelParams
    backend_name: str
    trained_at_ms: int
    train_span: tuple[int, int]
    fold_results: list[FoldResult] = field(default_factory=list)
    feature_importance: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    calibration: pd.DataFrame = field(default_factory=pd.DataFrame)
    holdout_metrics: dict[str, float] | None = None
    class_balance: dict[str, float] = field(default_factory=dict)
    #: Training-set mean and standard deviation per feature. Used only to
    #: explain a prediction when the backend cannot produce SHAP values.
    #: Computed from training data and frozen with the model, so live
    #: explanation never standardises against data from the future.
    feature_means: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    feature_stds: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    @property
    def mean_val_auc(self) -> float:
        values = [f.val_metrics.get("auc", np.nan) for f in self.fold_results]
        return float(np.nanmean(values)) if values else float("nan")

    @property
    def mean_log_loss_lift(self) -> float:
        values = [f.val_metrics.get("log_loss_lift", np.nan) for f in self.fold_results]
        return float(np.nanmean(values)) if values else float("nan")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.backend.predict_proba(X[self.features])

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump(self, handle)
        log.info("model saved to %s", path)
        return path

    @staticmethod
    def load(path: Path | str) -> "TrainedModel":
        with open(Path(path), "rb") as handle:
            return pickle.load(handle)


def plausibility_warnings(model: TrainedModel) -> list[str]:
    """Reasons to distrust a result. An empty list is the good outcome."""
    warnings: list[str] = []
    auc = model.mean_val_auc

    if np.isnan(auc):
        warnings.append("Validation AUC is undefined - a fold likely had one class only.")
    elif auc >= SUSPICIOUS_AUC:
        warnings.append(
            f"Validation AUC {auc:.3f} is implausibly high for hourly crypto returns. "
            "Treat this as a look-ahead leak until proven otherwise: re-run the "
            "point-in-time check and confirm the label horizon is embargoed."
        )
    elif auc <= 0.52:
        warnings.append(
            f"Validation AUC {auc:.3f} is barely above chance. The model has found "
            "little or no edge; do not deploy it expecting one."
        )

    if model.mean_log_loss_lift <= 0:
        warnings.append(
            "Log loss is no better than always predicting the base rate - the "
            "features add nothing over a constant."
        )

    gaps = [f.overfit_gap for f in model.fold_results if np.isfinite(f.overfit_gap)]
    if gaps and np.mean(gaps) > 0.15:
        warnings.append(
            f"Train AUC exceeds validation AUC by {np.mean(gaps):.3f} on average - "
            "the model is memorising. Reduce depth or increase min_samples_leaf."
        )

    balance = model.class_balance.get("positive_rate", float("nan"))
    if np.isfinite(balance) and (balance < 0.02 or balance > 0.98):
        warnings.append(
            f"Positive class rate is {balance:.1%}; the label threshold is "
            "mis-specified for this horizon and the model will just predict the "
            "majority class."
        )

    aucs = [f.val_metrics.get("auc", np.nan) for f in model.fold_results]
    finite = [a for a in aucs if np.isfinite(a)]
    if len(finite) >= 3 and np.std(finite) > 0.08:
        warnings.append(
            f"Fold AUCs vary widely (sd {np.std(finite):.3f}); the edge is not "
            "stable across time and may be an artefact of one period."
        )
    return warnings


def train_walk_forward(
    dataset: pd.DataFrame,
    *,
    label_config: LabelConfig | None = None,
    split_config: SplitConfig | None = None,
    params: ModelParams | None = None,
    prefer_backend: str | None = None,
) -> tuple[TrainedModel, LockedHoldout]:
    """Fit across expanding walk-forward folds, then refit on all development data."""
    label_config = label_config or LabelConfig()
    split_config = split_config or SplitConfig()
    params = params or ModelParams()

    features = feature_columns(dataset)
    if not features:
        raise ValueError("no numeric feature columns found")

    # Second, independent check. feature_columns already filters these, but a
    # leaked label is catastrophic and silent -- it produces a perfect score
    # that looks like success -- so it is worth refusing twice.
    leaked = sorted(set(features) & set(LABEL_COLUMNS))
    if leaked:
        raise LabelLeakError(
            f"label-derived columns reached the feature list: {leaked}. "
            "The model would be trained on its own answer."
        )

    clean = usable_rows(dataset, features)
    if clean.empty:
        raise ValueError("no usable labelled rows after filtering")

    development, holdout = development_and_holdout(
        clean, split_config, horizon_bars=label_config.horizon_bars
    )

    fold_results: list[FoldResult] = []
    for train_frame, val_frame, info in walk_forward_folds(
        development, split_config, horizon_bars=label_config.horizon_bars
    ):
        X_train, y_train = prepare_xy(train_frame, features)
        X_val, y_val = prepare_xy(val_frame, features)

        backend = make_backend(params, prefer=prefer_backend)
        backend.fit(X_train, y_train, X_val, y_val)

        fold_results.append(
            FoldResult(
                info=info,
                train_metrics=classification_metrics(y_train, backend.predict_proba(X_train)),
                val_metrics=classification_metrics(y_val, backend.predict_proba(X_val)),
            )
        )
        log.info(
            "fold %d: train=%d val=%d val_auc=%.4f",
            info["fold"], info["train_rows"], info["val_rows"],
            fold_results[-1].val_metrics.get("auc", float("nan")),
        )

    if not fold_results:
        raise ValueError("no walk-forward fold produced enough data to train on")

    # Final model: refit on the whole development period, holding back the
    # last fold as the early-stopping set. The holdout is never touched.
    cut = int(len(development) * 0.85)
    train_frame, val_frame = development.iloc[:cut], development.iloc[cut:]
    X_train, y_train = prepare_xy(train_frame, features)
    X_val, y_val = prepare_xy(val_frame, features)

    backend = make_backend(params, prefer=prefer_backend)
    backend.fit(X_train, y_train, X_val, y_val)

    model = TrainedModel(
        backend=backend,
        features=features,
        label_config=label_config,
        params=params,
        backend_name=backend.name,
        trained_at_ms=int(time.time() * 1000),
        train_span=(int(development["ts_ms"].min()), int(development["ts_ms"].max())),
        fold_results=fold_results,
        feature_importance=backend.feature_importance(),
        calibration=calibration_table(y_val, backend.predict_proba(X_val)),
        class_balance=class_balance(clean),
        feature_means=X_train.mean(),
        feature_stds=X_train.std(ddof=1).replace(0.0, np.nan),
    )
    return model, holdout


def evaluate_holdout(model: TrainedModel, holdout: LockedHoldout) -> dict[str, float]:
    """Score the locked test set. Call this ONCE, after the model is final.

    Anything tuned after seeing these numbers makes them in-sample.
    """
    holdout.lock_model()
    frame = holdout.release()
    X, y = prepare_xy(frame, model.features)
    metrics = classification_metrics(y, model.predict_proba(X))
    model.holdout_metrics = metrics
    return metrics


def render_report(model: TrainedModel) -> str:
    lines = [
        "MODEL TRAINING REPORT",
        f"  question          : {model.label_config.name}",
        f"  backend           : {model.backend_name}",
        f"  features          : {len(model.features)}",
        f"  training span     : "
        f"{pd.Timestamp(model.train_span[0], unit='ms', tz='UTC'):%Y-%m-%d} -> "
        f"{pd.Timestamp(model.train_span[1], unit='ms', tz='UTC'):%Y-%m-%d}",
        f"  labelled rows     : {model.class_balance.get('rows', 0):,}"
        f"  (positive rate {model.class_balance.get('positive_rate', float('nan')):.1%})",
        "",
        "  Walk-forward folds (validation is always AFTER training, with an embargo):",
        f"    {'fold':>4} {'train':>9} {'val':>8} {'val AUC':>9} {'train AUC':>10} {'lift':>9}",
    ]
    for result in model.fold_results:
        lines.append(
            f"    {result.info['fold']:>4} {result.info['train_rows']:>9,} "
            f"{result.info['val_rows']:>8,} "
            f"{result.val_metrics.get('auc', float('nan')):>9.4f} "
            f"{result.train_metrics.get('auc', float('nan')):>10.4f} "
            f"{result.val_metrics.get('log_loss_lift', float('nan')):>9.5f}"
        )
    lines += [
        "",
        f"  mean validation AUC : {model.mean_val_auc:.4f}  (0.50 = coin flip)",
        f"  mean log-loss lift  : {model.mean_log_loss_lift:+.5f}  (vs always predicting the base rate)",
    ]
    if model.holdout_metrics:
        lines += [
            "",
            "  LOCKED HOLDOUT (touched once, after the model was frozen):",
            f"    rows {model.holdout_metrics['rows']:,}  "
            f"AUC {model.holdout_metrics.get('auc', float('nan')):.4f}  "
            f"lift {model.holdout_metrics.get('log_loss_lift', float('nan')):+.5f}",
        ]

    top = model.feature_importance.head(12)
    if len(top) and top.sum() > 0:
        lines += ["", "  Top features by importance:"]
        lines += [f"    {name:<38}{value:>8.3%}" for name, value in top.items()]

    warnings = plausibility_warnings(model)
    lines.append("")
    if warnings:
        lines.append("  WARNINGS - do not deploy until these are understood:")
        lines += [f"    - {w}" for w in warnings]
    else:
        lines.append("  No plausibility warnings raised.")
    return "\n".join(lines)
