"""Funding-rate and basis features.

Funding is the perp market's own crowding signal: persistently positive
funding means longs are paying to stay long, which is both a carry cost and
a positioning indicator.

Timing matters here. Hyperliquid funds hourly, and a rate is only knowable
once its hour has begun. :func:`attach_funding_to_bars` merges with
``direction="backward"``, so a bar is matched to the most recent funding
observation at or before its close -- never the next one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features.base import (
    ensure_sorted,
    min_periods,
    prefix,
    rolling_percentile,
    rolling_z,
    safe_divide,
)

#: Hyperliquid quotes funding per hour; 24 * 365.25 hours in a year.
HOURS_PER_YEAR = 24 * 365.25


def attach_funding_to_bars(bars: pd.DataFrame, funding: pd.DataFrame) -> pd.DataFrame:
    """Merge hourly funding onto bars without letting a future rate leak back.

    ``merge_asof`` with ``direction="backward"`` takes the last funding
    print at or before each bar's timestamp. A forward or nearest merge
    would hand the bar a rate that had not been published yet.
    """
    bars = ensure_sorted(bars)
    if funding is None or funding.empty:
        out = bars.copy()
        out["funding_rate"] = np.nan
        return out

    funding = funding.sort_values("ts_ms")[["ts_ms", "funding_rate"]]
    merged = pd.merge_asof(
        bars.sort_values("ts_ms"),
        funding,
        on="ts_ms",
        direction="backward",
        suffixes=("", "_hist"),
    )
    if "funding_rate_hist" in merged.columns:
        merged["funding_rate"] = merged["funding_rate"].fillna(merged["funding_rate_hist"])
        merged = merged.drop(columns=["funding_rate_hist"])
    return merged.reset_index(drop=True)


def compute(bars: pd.DataFrame) -> pd.DataFrame:
    bars = ensure_sorted(bars)
    out = pd.DataFrame(index=bars.index)

    if "funding_rate" not in bars.columns:
        return prefix(out, "fund")

    rate = bars["funding_rate"].astype("float64")
    out["rate"] = rate
    out["rate_annualized"] = rate * HOURS_PER_YEAR

    for window in (24, 168):
        out[f"rate_mean_{window}"] = rate.rolling(window, min_periods=min_periods(window)).mean()
        out[f"rate_z_{window}"] = rolling_z(rate, window)
        # Total carry actually paid over the window -- the real cost of
        # having held the position, not an annualised abstraction.
        out[f"cum_{window}"] = rate.rolling(window, min_periods=min_periods(window)).sum()

    out["rate_percentile_720"] = rolling_percentile(rate, 720)

    # How one-sided has funding been? +1 = longs paid every hour of the
    # window, -1 = shorts paid every hour.
    sign = np.sign(rate)
    out["sign_persistence_72"] = sign.rolling(72, min_periods=min_periods(72)).mean()

    # Is funding accelerating? Crowding building or unwinding.
    out["rate_change_24"] = rate - rate.shift(24)

    # Basis: mark versus oracle. Present only when asset-context data has
    # been joined on; absent in a candles-only backtest.
    if {"mark_px", "oracle_px"} <= set(bars.columns):
        basis = safe_divide(bars["mark_px"] - bars["oracle_px"], bars["oracle_px"])
        out["basis"] = basis
        out["basis_z_72"] = rolling_z(basis, 72)

    if "open_interest" in bars.columns:
        oi = bars["open_interest"].astype("float64")
        out["oi_change_24"] = oi / oi.shift(24) - 1.0
        out["oi_z_168"] = rolling_z(oi, 168)
        # Rising OI with rising price = new money; rising OI with falling
        # price = trapped longs.
        out["oi_price_divergence_24"] = np.sign(out["oi_change_24"]) * np.sign(
            bars["close"] / bars["close"].shift(24) - 1.0
        )

    return prefix(out, "fund")
