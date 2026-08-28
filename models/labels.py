"""Label construction.

**This is the only module in the project permitted to look forward.** A
supervised label is by definition a fact about the future; the discipline
is confining that to one file, making the horizon explicit, and deleting
the rows where the future is not yet known.

Two rules enforced here:

1. The final ``horizon`` bars have no complete future and are dropped. Left
   in, they would be labelled from a partial window and bias the model
   toward whatever the tail of the data happened to do.
2. The threshold must clear trading costs. Labelling "up" as any positive
   return teaches the model to chase moves too small to profit from once
   fees, spread and funding are paid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Columns derived from the future. Feeding any of these to a model is a
#: total leak -- it trains on the answer and scores a perfect AUC. The
#: feature selector and the trainer both exclude this set explicitly.
LABEL_COLUMNS = ("label", "forward_return", "label_known")


@dataclass(frozen=True)
class LabelConfig:
    """Defines the question the model is asked.

    The default asks: *will the price be more than 0.30% higher four bars
    from now?* On 1h bars that is a 4-hour horizon, and 30bp comfortably
    exceeds a round trip at Hyperliquid's base tier (~9bp of fees plus
    spread and impact).
    """

    horizon_bars: int = 4
    threshold: float = 0.003
    #: "up" asks P(return > +threshold); "down" asks P(return < -threshold).
    direction: str = "up"

    def __post_init__(self) -> None:
        if self.horizon_bars < 1:
            raise ValueError("horizon_bars must be at least 1")
        if self.threshold < 0:
            raise ValueError("threshold must be non-negative")
        if self.direction not in {"up", "down"}:
            raise ValueError("direction must be 'up' or 'down'")

    @property
    def name(self) -> str:
        sign = ">" if self.direction == "up" else "<"
        signed = self.threshold if self.direction == "up" else -self.threshold
        return f"P(return_{self.horizon_bars}bar {sign} {signed:+.2%})"


def forward_return(close: pd.Series, horizon: int) -> pd.Series:
    """Return from bar *t*'s close to bar *t+horizon*'s close.

    Uses ``shift(-horizon)``, which is legitimate ONLY because this value is
    a training target, never a feature. The last ``horizon`` entries are NaN
    by construction and must be dropped, not filled.
    """
    return close.shift(-horizon) / close - 1.0


def make_labels(matrix: pd.DataFrame, config: LabelConfig | None = None) -> pd.DataFrame:
    """Attach ``label``, ``forward_return`` and ``label_known`` to a matrix."""
    config = config or LabelConfig()
    if "close" not in matrix.columns:
        raise ValueError("feature matrix must carry a 'close' column to label")

    out = matrix.copy()
    future = forward_return(out["close"], config.horizon_bars)
    out["forward_return"] = future

    if config.direction == "up":
        out["label"] = (future > config.threshold).astype("int8")
    else:
        out["label"] = (future < -config.threshold).astype("int8")

    # Rows whose future is not fully observed. Kept but flagged, so the
    # caller decides explicitly rather than a silent dropna doing it.
    out["label_known"] = future.notna()
    out.loc[~out["label_known"], "label"] = -1
    return out


def labelled_rows(matrix: pd.DataFrame) -> pd.DataFrame:
    """Only the rows whose label is fully determined."""
    if "label_known" not in matrix.columns:
        raise ValueError("call make_labels first")
    return matrix.loc[matrix["label_known"]].copy()


def class_balance(matrix: pd.DataFrame) -> dict[str, float]:
    """Positive rate and count, for sanity-checking the threshold.

    A positive rate near 0 or 1 means the threshold is mis-specified for the
    horizon: the model will learn to predict the majority class and report
    a flattering accuracy that means nothing.
    """
    known = labelled_rows(matrix)
    if known.empty:
        return {"rows": 0, "positive_rate": float("nan"), "positives": 0}
    return {
        "rows": int(len(known)),
        "positives": int(known["label"].sum()),
        "positive_rate": float(known["label"].mean()),
        "mean_forward_return": float(known["forward_return"].mean()),
        "median_forward_return": float(known["forward_return"].median()),
    }


def costs_exceed_threshold(config: LabelConfig, round_trip_cost: float) -> bool:
    """True when the label's threshold does not clear the cost of trading it."""
    return config.threshold <= round_trip_cost
