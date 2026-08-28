"""Shared primitives for point-in-time feature construction.

**The one rule:** a feature's value at bar *t* may depend only on bars at or
before *t*. Every helper here is causal by construction -- trailing rolling
windows, exponential weights, and backward shifts only. There is no
centred window, no interpolation across a gap, and no ``shift(-n)``
anywhere outside label construction.

:func:`features.pipeline.assert_point_in_time` verifies this empirically
rather than trusting the claim: it recomputes each feature on truncated
history and checks the value at *t* is bit-identical.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Rolling statistics need a minimum sample before they mean anything.
#: Below it we emit NaN rather than a number computed from three points,
#: because a confident-looking garbage feature is worse than a missing one.
MIN_PERIODS_FRACTION = 0.8


def min_periods(window: int) -> int:
    return max(2, int(window * MIN_PERIODS_FRACTION))


def log_returns(close: pd.Series) -> pd.Series:
    """Bar-over-bar log return. NaN at the first bar, never zero-filled."""
    return np.log(close / close.shift(1))


def rolling_z(series: pd.Series, window: int) -> pd.Series:
    """Trailing z-score. Standardises against the past, never the full sample.

    Using the whole sample's mean and standard deviation is one of the most
    common look-ahead leaks in published strategies: the normalisation
    itself carries information about the future.
    """
    mean = series.rolling(window, min_periods=min_periods(window)).mean()
    std = series.rolling(window, min_periods=min_periods(window)).std(ddof=1)
    return (series - mean) / std.replace(0.0, np.nan)


def rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    """Where the current value sits within its own trailing window, in [0, 1]."""
    return series.rolling(window, min_periods=min_periods(window)).rank(pct=True)


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Element-wise divide where a zero denominator yields NaN, not inf."""
    return numerator / denominator.replace(0.0, np.nan)


def clip_extremes(series: pd.Series, limit: float = 10.0) -> pd.Series:
    """Bound a standardised feature so one bad print cannot dominate a tree split."""
    return series.clip(lower=-limit, upper=limit)


def ensure_sorted(bars: pd.DataFrame) -> pd.DataFrame:
    """Sort by time and reject duplicates, which would corrupt every window."""
    if "ts_ms" not in bars.columns:
        raise ValueError("bars must carry a ts_ms column")
    out = bars.sort_values("ts_ms").reset_index(drop=True)
    if out["ts_ms"].duplicated().any():
        raise ValueError("duplicate timestamps in bars; deduplicate before featurising")
    return out


def prefix(frame: pd.DataFrame, tag: str) -> pd.DataFrame:
    """Namespace a feature block so its origin stays legible in the model."""
    return frame.rename(columns={c: f"{tag}_{c}" for c in frame.columns})
