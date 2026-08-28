"""Gradient-boosting backend selection.

The brief calls for LightGBM, and that is what production uses -- the
Docker image installs ``libgomp1`` so it loads cleanly on Linux. On macOS
LightGBM needs Homebrew's ``libomp``, which is not always present, so this
module falls back to scikit-learn's ``HistGradientBoostingClassifier``.

The two are the same family of algorithm (histogram-binned gradient-boosted
trees) and produce comparable models, but they are **not identical**. Every
artefact records which backend trained it, and results from different
backends must not be compared as if they were the same experiment.

The one real capability difference is explainability:

* LightGBM computes exact per-prediction SHAP contributions.
* The fallback has no equivalent, so per-decision attribution is an
  approximation, clearly labelled as such wherever it is surfaced.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def lightgbm_available() -> bool:
    """True only if LightGBM both imports and can load its native library."""
    try:
        import lightgbm  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - OSError on a missing libomp
        log.debug("LightGBM unavailable: %s", exc)
        return False
    return True


class Backend(Protocol):
    name: str
    supports_shap: bool

    def fit(self, X: pd.DataFrame, y: np.ndarray, X_val, y_val) -> "Backend": ...
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...
    def feature_importance(self) -> pd.Series: ...
    def contributions(self, X: pd.DataFrame) -> pd.DataFrame | None: ...


@dataclass
class ModelParams:
    """Deliberately conservative. Financial data has a terrible signal-to-noise
    ratio, and a deep tree on 100 features will memorise it perfectly."""

    n_estimators: int = 400
    learning_rate: float = 0.03
    max_depth: int = 4
    num_leaves: int = 15
    min_samples_leaf: int = 200
    subsample: float = 0.8
    colsample: float = 0.7
    l2_regularization: float = 1.0
    early_stopping_rounds: int = 50
    random_state: int = 42
    extra: dict[str, Any] = field(default_factory=dict)


class LightGBMBackend:
    name = "lightgbm"
    supports_shap = True

    def __init__(self, params: ModelParams) -> None:
        self.params = params
        self.model = None
        self._columns: list[str] = []

    def fit(self, X, y, X_val=None, y_val=None):
        import lightgbm as lgb

        self._columns = list(X.columns)
        self.model = lgb.LGBMClassifier(
            n_estimators=self.params.n_estimators,
            learning_rate=self.params.learning_rate,
            max_depth=self.params.max_depth,
            num_leaves=self.params.num_leaves,
            min_child_samples=self.params.min_samples_leaf,
            subsample=self.params.subsample,
            subsample_freq=1,
            colsample_bytree=self.params.colsample,
            reg_lambda=self.params.l2_regularization,
            random_state=self.params.random_state,
            objective="binary",
            verbosity=-1,
            **self.params.extra,
        )
        callbacks = []
        eval_set = None
        if X_val is not None and len(X_val) > 0:
            eval_set = [(X_val, y_val)]
            callbacks.append(
                lgb.early_stopping(self.params.early_stopping_rounds, verbose=False)
            )
        self.model.fit(X, y, eval_set=eval_set, callbacks=callbacks)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X[self._columns])[:, 1]

    def feature_importance(self):
        booster = self.model.booster_
        gains = booster.feature_importance(importance_type="gain")
        total = gains.sum()
        values = gains / total if total > 0 else gains
        return pd.Series(values, index=self._columns).sort_values(ascending=False)

    def contributions(self, X):
        """Exact SHAP values: one contribution per feature per prediction."""
        raw = self.model.booster_.predict(X[self._columns], pred_contrib=True)
        # Final column is the base value (expected log-odds).
        return pd.DataFrame(raw[:, :-1], columns=self._columns, index=X.index)


class SklearnHistBackend:
    name = "sklearn_hist"
    supports_shap = False

    def __init__(self, params: ModelParams) -> None:
        self.params = params
        self.model = None
        self._columns: list[str] = []
        self._importance: pd.Series | None = None

    def fit(self, X, y, X_val=None, y_val=None):
        from sklearn.ensemble import HistGradientBoostingClassifier

        self._columns = list(X.columns)
        use_early_stopping = X_val is not None and len(X_val) > 0
        self.model = HistGradientBoostingClassifier(
            max_iter=self.params.n_estimators,
            learning_rate=self.params.learning_rate,
            max_depth=self.params.max_depth,
            max_leaf_nodes=self.params.num_leaves,
            min_samples_leaf=self.params.min_samples_leaf,
            l2_regularization=self.params.l2_regularization,
            early_stopping=use_early_stopping,
            n_iter_no_change=self.params.early_stopping_rounds,
            validation_fraction=0.15 if use_early_stopping else None,
            random_state=self.params.random_state,
        )
        self.model.fit(X, y)

        # No native gain importance, so use permutation importance on the
        # validation fold. Slower, but it measures what actually matters:
        # how much the score degrades when a feature is scrambled.
        if use_early_stopping and len(X_val) > 50:
            self._importance = self._permutation_importance(X_val, y_val)
        return self

    def _permutation_importance(self, X_val, y_val) -> pd.Series:
        from sklearn.inspection import permutation_importance

        sample = min(len(X_val), 4000)
        result = permutation_importance(
            self.model,
            X_val.iloc[-sample:],
            y_val[-sample:],
            n_repeats=3,
            random_state=self.params.random_state,
            scoring="roc_auc",
        )
        values = np.clip(result.importances_mean, 0, None)
        total = values.sum()
        if total > 0:
            values = values / total
        return pd.Series(values, index=self._columns).sort_values(ascending=False)

    def predict_proba(self, X):
        return self.model.predict_proba(X[self._columns])[:, 1]

    def feature_importance(self):
        if self._importance is not None:
            return self._importance
        return pd.Series(0.0, index=self._columns)

    def contributions(self, X):
        """No exact attribution available for this backend."""
        return None


def make_backend(params: ModelParams | None = None, *, prefer: str | None = None) -> Backend:
    params = params or ModelParams()
    if prefer == "sklearn_hist" or (prefer is None and not lightgbm_available()):
        if prefer is None:
            log.warning(
                "LightGBM could not be loaded (macOS needs `brew install libomp`); "
                "falling back to scikit-learn HistGradientBoosting. Production "
                "Docker images use LightGBM."
            )
        return SklearnHistBackend(params)
    return LightGBMBackend(params)
