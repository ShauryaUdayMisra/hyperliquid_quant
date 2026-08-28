"""Market-regime classification.

The regime label is itself a point-in-time feature: it is decided from
trailing efficiency and trailing volatility percentile only. A regime
labelled with hindsight ("this was a bull market") is one of the most
damaging leaks possible, because it encodes the answer.

Four regimes, deliberately few enough that per-regime performance
statistics have enough samples to mean something:

``trending_up``    directional and efficient, upward
``trending_down``  directional and efficient, downward
``ranging``        low efficiency, ordinary volatility
``high_volatility`` volatility in the top decile of its own history
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features.momentum import efficiency_ratio
from features.volatility import realized_vol
from features.base import ensure_sorted, min_periods, rolling_percentile

TRENDING = 0.40          # efficiency ratio above this counts as directional
HIGH_VOL_PERCENTILE = 0.85
REGIMES = ("trending_up", "trending_down", "ranging", "high_volatility", "unknown")


def classify(
    bars: pd.DataFrame,
    *,
    efficiency_window: int = 24,
    vol_window: int = 24,
    percentile_window: int = 720,
    trending_threshold: float = TRENDING,
    high_vol_percentile: float = HIGH_VOL_PERCENTILE,
) -> pd.DataFrame:
    """Label each bar's regime using only that bar and its past."""
    bars = ensure_sorted(bars)
    close = bars["close"]

    efficiency = efficiency_ratio(close, efficiency_window)
    vol = realized_vol(close, vol_window)
    vol_rank = rolling_percentile(vol, percentile_window)
    direction = np.sign(close - close.shift(efficiency_window))

    label = pd.Series("unknown", index=bars.index, dtype="object")
    known = efficiency.notna() & vol_rank.notna()

    ranging = known & (efficiency < trending_threshold)
    up = known & (efficiency >= trending_threshold) & (direction > 0)
    down = known & (efficiency >= trending_threshold) & (direction < 0)

    label[ranging] = "ranging"
    label[up] = "trending_up"
    label[down] = "trending_down"
    # Volatility overrides direction: a violent market behaves differently
    # regardless of which way it happens to be pointing.
    label[known & (vol_rank >= high_vol_percentile)] = "high_volatility"

    return pd.DataFrame(
        {
            "regime": label,
            "regime_efficiency": efficiency,
            "regime_vol_percentile": vol_rank,
            "regime_direction": direction,
        }
    )


def describe(row: pd.Series) -> str:
    """One plain sentence about the regime, for the email report."""
    regime = row.get("regime", "unknown")
    efficiency = row.get("regime_efficiency", float("nan"))
    vol_rank = row.get("regime_vol_percentile", float("nan"))

    if regime == "unknown":
        return "regime undetermined - not enough history yet"
    phrases = {
        "trending_up": "trending higher",
        "trending_down": "trending lower",
        "ranging": "range-bound",
        "high_volatility": "unusually volatile",
    }
    text = phrases.get(regime, regime)
    if np.isfinite(efficiency):
        text += f" (trend efficiency {efficiency:.2f}"
        if np.isfinite(vol_rank):
            text += f", volatility in the {vol_rank:.0%} percentile of the last 30 days"
        text += ")"
    return text
