"""Dataset assembly and chronological splitting.

Three rules, each enforced in code rather than by convention:

**Split by time, never by row.** A random split lets the model train on
Wednesday and validate on Tuesday. Splits here are made on the timestamp
axis, so every coin is cut at the same instant.

**Purge and embargo around every boundary.** A training row at bar *t* is
labelled from bars up to *t + horizon*. If validation starts at *t + 1*,
that label already saw validation data. Every boundary therefore drops at
least ``horizon`` bars, plus a configurable extra margin.

**The holdout is locked.** :class:`LockedHoldout` refuses to hand over the
test set until the model has been explicitly frozen. Peeking is a
programming error, so it raises instead of warning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd

from features.pipeline import feature_columns
from models.labels import LabelConfig, make_labels

log = logging.getLogger(__name__)


class HoldoutViolation(RuntimeError):
    """Raised when the locked test set is touched before the model is frozen."""


@dataclass
class SplitConfig:
    #: Final share of the timeline reserved as an untouchable test set.
    test_fraction: float = 0.20
    #: Walk-forward folds inside the development period.
    n_folds: int = 5
    #: Extra bars dropped either side of a boundary, on top of the label
    #: horizon. Absorbs autocorrelation that outlives the label window.
    embargo_bars: int = 24
    #: A fold with fewer rows than this is not worth fitting.
    min_train_rows: int = 500
    min_val_rows: int = 100


def assemble(
    matrices: dict[str, pd.DataFrame],
    label_config: LabelConfig | None = None,
) -> pd.DataFrame:
    """Stack per-coin feature matrices into one labelled training table.

    Labels are computed per coin *before* stacking. Computing them after
    would run ``shift`` across the boundary between two coins and label
    BTC's last bar with ETH's first.
    """
    label_config = label_config or LabelConfig()
    labelled = []
    for coin, matrix in matrices.items():
        frame = make_labels(matrix, label_config)
        frame["coin"] = coin
        labelled.append(frame)

    stacked = pd.concat(labelled, ignore_index=True)
    stacked = stacked.sort_values(["ts_ms", "coin"]).reset_index(drop=True)
    return stacked


def usable_rows(dataset: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
    """Rows with a known label and enough populated features to be worth using."""
    features = features or feature_columns(dataset)
    known = dataset.loc[dataset["label_known"]].copy()
    # A row where most features are still warming up teaches nothing.
    populated = known[features].notna().mean(axis=1)
    return known.loc[populated > 0.7].reset_index(drop=True)


def split_timestamps(dataset: pd.DataFrame, fraction: float) -> int:
    """Timestamp that divides the data, so all coins are cut together."""
    stamps = np.sort(dataset["ts_ms"].unique())
    if len(stamps) < 2:
        raise ValueError("not enough distinct timestamps to split")
    index = int(len(stamps) * (1.0 - fraction))
    index = min(max(index, 1), len(stamps) - 1)
    return int(stamps[index])


@dataclass
class LockedHoldout:
    """The final slice of history, withheld until the model is frozen.

    ``release()`` raises unless :meth:`lock_model` has been called. This
    turns "do not look at the test set" from a rule people intend to follow
    into one the code will not let them break.
    """

    _data: pd.DataFrame
    boundary_ts_ms: int
    _locked: bool = False
    _released: bool = False

    def lock_model(self) -> None:
        self._locked = True

    @property
    def rows(self) -> int:
        return len(self._data)

    @property
    def span(self) -> tuple[int, int]:
        return int(self._data["ts_ms"].min()), int(self._data["ts_ms"].max())

    @property
    def was_released(self) -> bool:
        return self._released

    def release(self) -> pd.DataFrame:
        if not self._locked:
            raise HoldoutViolation(
                "The test set was requested before the model was locked. Finish "
                "training and call lock_model() first; any tuning done after "
                "seeing these rows invalidates the out-of-sample result."
            )
        self._released = True
        return self._data.copy()


def development_and_holdout(
    dataset: pd.DataFrame, config: SplitConfig | None = None, *, horizon_bars: int = 4
) -> tuple[pd.DataFrame, LockedHoldout]:
    """Cut the timeline into a development period and a locked test set."""
    config = config or SplitConfig()
    boundary = split_timestamps(dataset, config.test_fraction)

    interval_ms = _infer_interval_ms(dataset)
    embargo_ms = (horizon_bars + config.embargo_bars) * interval_ms

    # Development stops an embargo before the boundary so no training label
    # can reach across into the test period.
    development = dataset.loc[dataset["ts_ms"] < boundary - embargo_ms].reset_index(drop=True)
    holdout = dataset.loc[dataset["ts_ms"] >= boundary].reset_index(drop=True)

    log.info(
        "development: %d rows through %s | holdout: %d rows from %s (embargo %d bars)",
        len(development),
        pd.Timestamp(int(development["ts_ms"].max()), unit="ms", tz="UTC") if len(development) else "n/a",
        len(holdout),
        pd.Timestamp(boundary, unit="ms", tz="UTC"),
        horizon_bars + config.embargo_bars,
    )
    return development, LockedHoldout(holdout, boundary)


def walk_forward_folds(
    development: pd.DataFrame,
    config: SplitConfig | None = None,
    *,
    horizon_bars: int = 4,
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame, dict]]:
    """Expanding-window walk-forward folds.

    Fold *k* trains on everything before a cut and validates on the slice
    immediately after it. The training window expands rather than slides,
    because more history is genuinely useful and a rolling window throws
    away regimes the model should have seen.

    Yields ``(train, validate, info)``.
    """
    config = config or SplitConfig()
    stamps = np.sort(development["ts_ms"].unique())
    if len(stamps) < config.n_folds * 2:
        raise ValueError("not enough history for the requested number of folds")

    interval_ms = _infer_interval_ms(development)
    embargo_ms = (horizon_bars + config.embargo_bars) * interval_ms

    # Validation blocks tile the last half of the development period, so
    # early folds still get a substantial training base.
    first_cut = len(stamps) // 2
    edges = np.linspace(first_cut, len(stamps), config.n_folds + 1, dtype=int)

    for fold, (start, end) in enumerate(zip(edges[:-1], edges[1:]), start=1):
        val_start_ts = int(stamps[start])
        val_end_ts = int(stamps[min(end, len(stamps) - 1)])

        train = development.loc[development["ts_ms"] < val_start_ts - embargo_ms]
        validate = development.loc[
            (development["ts_ms"] >= val_start_ts) & (development["ts_ms"] <= val_end_ts)
        ]

        if len(train) < config.min_train_rows or len(validate) < config.min_val_rows:
            log.warning(
                "fold %d skipped: train=%d val=%d below minimum", fold, len(train), len(validate)
            )
            continue

        info = {
            "fold": fold,
            "train_rows": len(train),
            "val_rows": len(validate),
            "train_end_ts": int(train["ts_ms"].max()),
            "val_start_ts": val_start_ts,
            "val_end_ts": val_end_ts,
            "embargo_bars": horizon_bars + config.embargo_bars,
        }
        yield train.reset_index(drop=True), validate.reset_index(drop=True), info


def _infer_interval_ms(dataset: pd.DataFrame) -> int:
    stamps = np.sort(dataset["ts_ms"].unique())
    if len(stamps) < 2:
        return 3_600_000
    return int(np.median(np.diff(stamps)))


def prepare_xy(
    frame: pd.DataFrame, features: list[str]
) -> tuple[pd.DataFrame, np.ndarray]:
    """Feature matrix and label vector, with NaNs left intact.

    Gradient-boosted trees handle missing values natively by learning a
    default direction at each split. Imputing here would fabricate data --
    and for order-book features, "no collector was running" is genuine
    information, not a gap to paper over.
    """
    X = frame[features].astype("float64")
    y = frame["label"].to_numpy(dtype=np.int8)
    return X, y
