"""Volatility features from completed bars only."""

from __future__ import annotations

import numpy as np
import pandas as pd

from features.base import (
    ensure_sorted,
    log_returns,
    min_periods,
    prefix,
    rolling_percentile,
    safe_divide,
)

BARS_PER_YEAR_1H = 365.25 * 24


def realized_vol(close: pd.Series, window: int, bars_per_year: float = BARS_PER_YEAR_1H) -> pd.Series:
    """Annualised standard deviation of trailing log returns."""
    returns = log_returns(close)
    return returns.rolling(window, min_periods=min_periods(window)).std(ddof=1) * np.sqrt(
        bars_per_year
    )


def parkinson_vol(high: pd.Series, low: pd.Series, window: int,
                  bars_per_year: float = BARS_PER_YEAR_1H) -> pd.Series:
    """High-low range estimator. Uses roughly 5x less data for the same precision."""
    log_hl = np.log(high / low) ** 2
    factor = 1.0 / (4.0 * np.log(2.0))
    return np.sqrt(
        log_hl.rolling(window, min_periods=min_periods(window)).mean() * factor * bars_per_year
    )


def garman_klass_vol(bars: pd.DataFrame, window: int,
                     bars_per_year: float = BARS_PER_YEAR_1H) -> pd.Series:
    """Garman-Klass: uses the full OHLC of each COMPLETED bar.

    Legitimate point-in-time because every component belongs to a bar that
    has already closed. It would leak instantly if applied to a forming bar,
    which is why the data layer refuses to store one.
    """
    log_hl = np.log(bars["high"] / bars["low"]) ** 2
    log_co = np.log(bars["close"] / bars["open"]) ** 2
    estimate = 0.5 * log_hl - (2.0 * np.log(2.0) - 1.0) * log_co
    return np.sqrt(
        estimate.rolling(window, min_periods=min_periods(window)).mean().clip(lower=0.0)
        * bars_per_year
    )


def atr(bars: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average true range, normalised by price so it is cross-coin comparable."""
    prev_close = bars["close"].shift(1)
    true_range = pd.concat(
        [
            bars["high"] - bars["low"],
            (bars["high"] - prev_close).abs(),
            (bars["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    smoothed = true_range.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    return smoothed / bars["close"]


def compute(bars: pd.DataFrame, bars_per_year: float = BARS_PER_YEAR_1H) -> pd.DataFrame:
    bars = ensure_sorted(bars)
    out = pd.DataFrame(index=bars.index)

    for window in (12, 24, 72, 168):
        out[f"rv_{window}"] = realized_vol(bars["close"], window, bars_per_year)

    out["parkinson_24"] = parkinson_vol(bars["high"], bars["low"], 24, bars_per_year)
    out["garman_klass_24"] = garman_klass_vol(bars, 24, bars_per_year)
    out["atr_14"] = atr(bars, 14)

    # Vol-of-vol: is the volatility itself unstable?
    out["vol_of_vol_72"] = out["rv_24"].rolling(72, min_periods=min_periods(72)).std(ddof=1)

    # Short vs long vol: >1 means volatility is expanding right now.
    out["vol_ratio_12_72"] = safe_divide(out["rv_12"], out["rv_72"])
    out["vol_ratio_24_168"] = safe_divide(out["rv_24"], out["rv_168"])

    # Where today's vol sits in its own trailing distribution. This is what
    # the regime classifier thresholds on.
    out["vol_percentile_168"] = rolling_percentile(out["rv_24"], 168)
    out["vol_percentile_720"] = rolling_percentile(out["rv_24"], 720)

    # Downside vs upside dispersion: skew of the recent return distribution.
    returns = log_returns(bars["close"])
    downside = returns.clip(upper=0.0)
    upside = returns.clip(lower=0.0)
    out["downside_vol_72"] = downside.rolling(72, min_periods=min_periods(72)).std(ddof=1)
    out["vol_skew_72"] = safe_divide(
        out["downside_vol_72"], upside.rolling(72, min_periods=min_periods(72)).std(ddof=1)
    )

    return prefix(out, "vol")
