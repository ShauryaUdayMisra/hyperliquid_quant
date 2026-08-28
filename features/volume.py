"""Volume and participation features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from features.base import (
    ensure_sorted,
    log_returns,
    min_periods,
    prefix,
    rolling_percentile,
    rolling_z,
    safe_divide,
)


def compute(bars: pd.DataFrame) -> pd.DataFrame:
    bars = ensure_sorted(bars)
    out = pd.DataFrame(index=bars.index)

    volume = bars["volume"]
    notional = volume * bars["close"]

    for window in (24, 168):
        out[f"volume_z_{window}"] = rolling_z(volume, window)
        out[f"notional_z_{window}"] = rolling_z(notional, window)

    out["volume_percentile_168"] = rolling_percentile(volume, 168)
    out["volume_ratio_12_72"] = safe_divide(
        volume.rolling(12, min_periods=min_periods(12)).mean(),
        volume.rolling(72, min_periods=min_periods(72)).mean(),
    )

    if "trades" in bars.columns:
        trades = bars["trades"].astype("float64")
        out["trade_count_z_24"] = rolling_z(trades, 24)
        # Average trade size: rising size with flat count suggests larger
        # participants stepping in.
        out["avg_trade_size"] = safe_divide(volume, trades)
        out["avg_trade_size_z_72"] = rolling_z(out["avg_trade_size"], 72)

    # Amihud illiquidity: price move per unit of traded notional. High values
    # mean the book is thin and an order will cost more to execute.
    returns = log_returns(bars["close"]).abs()
    out["illiquidity_24"] = safe_divide(returns, notional).rolling(
        24, min_periods=min_periods(24)
    ).mean() * 1e9

    # Signed volume proxy: where the close sits inside the bar's range tells
    # us whether buyers or sellers finished in control.
    span = (bars["high"] - bars["low"]).replace(0.0, np.nan)
    close_location = ((bars["close"] - bars["low"]) - (bars["high"] - bars["close"])) / span
    out["close_location"] = close_location
    out["money_flow_24"] = (close_location * volume).rolling(
        24, min_periods=min_periods(24)
    ).sum() / volume.rolling(24, min_periods=min_periods(24)).sum()

    # Rolling VWAP distance, computed from completed bars only.
    typical = (bars["high"] + bars["low"] + bars["close"]) / 3.0
    for window in (24, 168):
        vwap = (
            (typical * volume).rolling(window, min_periods=min_periods(window)).sum()
            / volume.rolling(window, min_periods=min_periods(window)).sum()
        )
        out[f"vwap_distance_{window}"] = bars["close"] / vwap - 1.0

    return prefix(out, "vol_flow")
