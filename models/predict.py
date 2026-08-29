"""Inference: turn a feature row into an explained probability.

The model answers one question -- P(price rises more than the threshold
over the horizon). It does not say "buy". Converting a probability into a
position is the strategy's job (Phase 5), and the risk engine can still
veto whatever the strategy decides.

Shorting deserves a note. A low P(up) is not the same as a high P(down):
it also covers "goes nowhere". Inferring a short from a low up-probability
is an assumption, so :class:`SignalGenerator` supports a separately
trained down-model and will only short when one is supplied.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from models.train import TrainedModel

log = logging.getLogger(__name__)


@dataclass
class FeatureContribution:
    name: str
    value: float          # the feature's own value at this bar
    contribution: float   # signed push toward (positive) or away from (negative) "up"
    method: str           # "shap" or "importance_weighted"

    def describe(self) -> str:
        arrow = "+" if self.contribution >= 0 else "-"
        return f"{self.name} {arrow} ({self.value:+.3f})"


@dataclass
class Signal:
    """One model opinion about one market at one bar, with its reasoning."""

    coin: str
    ts_ms: int
    probability: float
    base_rate: float
    direction: str                  # "long" | "short" | "flat"
    regime: str = "unknown"
    top_features: list[FeatureContribution] = field(default_factory=list)
    down_probability: float | None = None
    #: The down-model's own unconditional rate. Kept beside the up-model's
    #: because the two questions have different base rates, and confidence
    #: is measured against whichever one the signal is actually expressing.
    down_base_rate: float | None = None
    model_backend: str = ""
    label_question: str = ""
    #: The short side's question, so a short can be reported by the model
    #: that actually authorised it.
    down_label_question: str = ""
    explanation_method: str = "none"

    @property
    def confidence(self) -> float:
        """How far the model has moved from its unconditional guess, in [0, 1].

        Measured against the base rate rather than 0.5: if only 20% of bars
        are positive, predicting 0.35 is a strong opinion, not a weak one.

        A short is scored on the *down* model against the *down* base rate.
        The portfolio ranks candidates by this number to allocate a limited
        number of slots, so scoring a short by how far P(up) sits from the
        up-model's base rate would rank longs and shorts on two different
        scales and quietly favour one side.
        """
        probability, base = self.probability, self.base_rate
        if self.direction == "short" and self.down_probability is not None:
            probability = self.down_probability
            base = self.base_rate if self.down_base_rate is None else self.down_base_rate
        denominator = max(base, 1.0 - base)
        if denominator <= 0:
            return 0.0
        return float(min(1.0, abs(probability - base) / denominator))

    @property
    def edge(self) -> float:
        """Signed distance from the base rate. Positive leans up."""
        return float(self.probability - self.base_rate)

    def describe(self) -> str:
        bits = [
            f"{self.coin}: P(up)={self.probability:.3f} vs base {self.base_rate:.3f}",
            f"confidence {self.confidence:.2f}",
            f"leans {self.direction}",
            f"regime {self.regime}",
        ]
        if self.top_features:
            drivers = ", ".join(f.describe() for f in self.top_features[:4])
            bits.append(f"drivers: {drivers}")
        return " | ".join(bits)


class SignalGenerator:
    """Wraps a trained model (or a long/short pair) for live use."""

    def __init__(
        self,
        model: TrainedModel,
        *,
        down_model: TrainedModel | None = None,
        long_threshold: float = 0.55,
        short_threshold: float = 0.55,
        top_n_features: int = 6,
    ) -> None:
        self.model = model
        self.down_model = down_model
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.top_n_features = top_n_features
        self._base_rate = float(model.class_balance.get("positive_rate", 0.5))
        self._down_base_rate = (
            float(down_model.class_balance.get("positive_rate", 0.5))
            if down_model is not None else None
        )
        # Prefer the scales frozen with the model. Falling back to
        # calibrate_feature_scales() supports artefacts trained before these
        # were stored, and keeps the explicit override available for tests.
        stored_means = getattr(model, "feature_means", None)
        stored_stds = getattr(model, "feature_stds", None)
        has_stored = stored_means is not None and len(stored_means) > 0
        self._feature_means: pd.Series | None = stored_means if has_stored else None
        self._feature_stds: pd.Series | None = stored_stds if has_stored else None

    def calibrate_feature_scales(self, reference: pd.DataFrame) -> None:
        """Record training-set feature scales, for the fallback explanation.

        Only used when the backend cannot produce SHAP values. Computed from
        training data, never from live data, so it introduces no leak.
        """
        columns = [c for c in self.model.features if c in reference.columns]
        self._feature_means = reference[columns].mean()
        self._feature_stds = reference[columns].std(ddof=1).replace(0.0, np.nan)

    # -- explanation -------------------------------------------------------

    def explain(self, X: pd.DataFrame) -> tuple[list[list[FeatureContribution]], str]:
        contributions = self.model.backend.contributions(X)

        if contributions is not None:
            method = "shap"
            frames = contributions
        else:
            # No exact attribution. Approximate: how unusual is this feature
            # right now, weighted by how much the model relies on it. This is
            # a heuristic, labelled as such everywhere it appears.
            method = "importance_weighted"
            importance = self.model.feature_importance
            if self._feature_means is None or importance.empty:
                return [[] for _ in range(len(X))], "none"
            columns = [c for c in X.columns if c in self._feature_means.index]
            z = (X[columns] - self._feature_means[columns]) / self._feature_stds[columns]
            frames = z.mul(importance.reindex(columns).fillna(0.0), axis=1)

        results = []
        for position in range(len(X)):
            row = frames.iloc[position]
            ranked = row.reindex(row.abs().sort_values(ascending=False).index)
            top = ranked.head(self.top_n_features)
            results.append(
                [
                    FeatureContribution(
                        name=name,
                        value=float(X.iloc[position].get(name, np.nan)),
                        contribution=float(value),
                        method=method,
                    )
                    for name, value in top.items()
                    if np.isfinite(value)
                ]
            )
        return results, method

    # -- prediction --------------------------------------------------------

    def generate(self, matrix: pd.DataFrame, *, coin: str | None = None) -> list[Signal]:
        """Signals for every row of a feature matrix."""
        missing = [c for c in self.model.features if c not in matrix.columns]
        if missing:
            raise ValueError(f"feature matrix is missing {len(missing)} columns, e.g. {missing[:5]}")

        X = matrix[self.model.features].astype("float64")
        probabilities = self.model.predict_proba(X)
        down_probabilities = (
            self.down_model.predict_proba(matrix[self.down_model.features].astype("float64"))
            if self.down_model is not None
            else None
        )
        explanations, method = self.explain(X)

        signals = []
        for i in range(len(matrix)):
            row = matrix.iloc[i]
            p_up = float(probabilities[i])
            p_down = float(down_probabilities[i]) if down_probabilities is not None else None
            signals.append(
                Signal(
                    coin=coin or str(row.get("coin", "?")),
                    ts_ms=int(row["ts_ms"]),
                    probability=p_up,
                    base_rate=self._base_rate,
                    direction=self._direction(p_up, p_down),
                    regime=str(row.get("regime", "unknown")),
                    top_features=explanations[i],
                    down_probability=p_down,
                    down_base_rate=self._down_base_rate,
                    model_backend=self.model.backend_name,
                    label_question=self.model.label_config.name,
                    down_label_question=(
                        self.down_model.label_config.name
                        if self.down_model is not None else ""
                    ),
                    explanation_method=method,
                )
            )
        return signals

    def latest(self, matrix: pd.DataFrame, *, coin: str | None = None) -> Signal:
        """Signal for the most recent bar only -- what the live loop needs."""
        return self.generate(matrix.iloc[[-1]], coin=coin)[0]

    def _direction(self, p_up: float, p_down: float | None) -> str:
        if p_up >= self.long_threshold:
            return "long"
        if p_down is not None and p_down >= self.short_threshold:
            return "short"
        # Without a down-model, a low P(up) means "no long", not "short".
        return "flat"
