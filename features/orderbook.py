"""Order-book microstructure features.

Input is the flat snapshot table written by the collector (one row per
price level). Snapshots are first collapsed to per-snapshot metrics, then
aggregated into the bar that contains them.

Causality: a bar stamped ``t`` covers ``[t, t + interval)``, so every
snapshot inside that window is known by the time the bar closes, which is
when a decision is made. Nothing from ``t + interval`` onward is used.

These features exist ONLY for periods when a collector was running. There
is no historical order-book endpoint on Hyperliquid, so backtests over old
data will find these columns entirely NaN -- which is correct and must not
be filled in with a guess.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features.base import ensure_sorted, min_periods, prefix, rolling_z, safe_divide


def snapshot_metrics(book: pd.DataFrame, depths: tuple[int, ...] = (1, 5, 20)) -> pd.DataFrame:
    """Collapse level rows into one row of metrics per (coin, snapshot)."""
    required = {"ts_ms", "coin", "side", "level", "px", "sz"}
    missing = required - set(book.columns)
    if missing:
        raise ValueError(f"order book frame missing columns: {sorted(missing)}")
    if book.empty:
        return pd.DataFrame(columns=["ts_ms", "coin"])

    bids = book[book["side"] == "bid"]
    asks = book[book["side"] == "ask"]

    best_bid = bids[bids["level"] == 0].set_index(["coin", "ts_ms"])["px"]
    best_ask = asks[asks["level"] == 0].set_index(["coin", "ts_ms"])["px"]
    bid_top = bids[bids["level"] == 0].set_index(["coin", "ts_ms"])["sz"]
    ask_top = asks[asks["level"] == 0].set_index(["coin", "ts_ms"])["sz"]

    frame = pd.DataFrame({"best_bid": best_bid, "best_ask": best_ask,
                          "bid_sz_top": bid_top, "ask_sz_top": ask_top})
    frame["mid"] = (frame["best_bid"] + frame["best_ask"]) / 2.0
    frame["spread_bps"] = (frame["best_ask"] - frame["best_bid"]) / frame["mid"] * 10_000

    # Microprice: the size-weighted fair value. Sitting above the mid means
    # the book is leaning bid-heavy and the next tick is more likely up.
    total_top = frame["bid_sz_top"] + frame["ask_sz_top"]
    microprice = (
        frame["best_bid"] * frame["ask_sz_top"] + frame["best_ask"] * frame["bid_sz_top"]
    ) / total_top.replace(0.0, np.nan)
    frame["microprice_offset_bps"] = (microprice - frame["mid"]) / frame["mid"] * 10_000

    for depth in depths:
        bid_depth = (
            bids[bids["level"] < depth].groupby(["coin", "ts_ms"])["sz"].sum()
        )
        ask_depth = (
            asks[asks["level"] < depth].groupby(["coin", "ts_ms"])["sz"].sum()
        )
        total = (bid_depth + ask_depth).replace(0.0, np.nan)
        # Imbalance in [-1, 1]: +1 is all bid, -1 is all ask.
        frame[f"imbalance_{depth}"] = (bid_depth - ask_depth) / total
        frame[f"depth_{depth}"] = bid_depth + ask_depth

    # Latency actually observed between the exchange stamp and our receipt.
    if "recv_ts_ms" in book.columns:
        latency = book.groupby(["coin", "ts_ms"])["recv_ts_ms"].first()
        frame["latency_ms"] = latency - frame.index.get_level_values("ts_ms")

    return frame.reset_index()


def aggregate_to_bars(
    snapshots: pd.DataFrame, bar_ts: pd.Series, interval_ms: int
) -> pd.DataFrame:
    """Bucket per-snapshot metrics into the bar that contains them.

    A snapshot at time ``s`` belongs to the bar ``floor(s / interval)``.
    Both the mean over the bar and its final value are kept: the mean is
    the steadier signal, the last value is what the next decision actually
    faces.
    """
    if snapshots.empty:
        return pd.DataFrame({"ts_ms": bar_ts.to_numpy()})

    working = snapshots.copy()
    working["bar_ts"] = (working["ts_ms"] // interval_ms) * interval_ms

    metric_columns = [
        c for c in working.columns
        if c not in {"ts_ms", "coin", "bar_ts", "best_bid", "best_ask"}
    ]
    grouped = working.groupby("bar_ts")[metric_columns]
    means = grouped.mean().add_suffix("_mean")
    lasts = grouped.last().add_suffix("_last")
    counts = grouped.size().rename("snapshot_count")

    merged = pd.concat([means, lasts, counts], axis=1).reset_index()
    merged = merged.rename(columns={"bar_ts": "ts_ms"})
    return pd.DataFrame({"ts_ms": bar_ts.to_numpy()}).merge(merged, on="ts_ms", how="left")


def compute(
    bars: pd.DataFrame,
    book_snapshots: pd.DataFrame | None,
    interval_ms: int,
) -> pd.DataFrame:
    """Order-book feature block, aligned to ``bars``."""
    bars = ensure_sorted(bars)
    if book_snapshots is None or book_snapshots.empty:
        # No collector was running for this period. Leave the block empty
        # rather than inventing neutral values that a model would learn from.
        return pd.DataFrame(index=bars.index)

    metrics = snapshot_metrics(book_snapshots)
    aligned = aggregate_to_bars(metrics, bars["ts_ms"], interval_ms)
    out = aligned.drop(columns=["ts_ms"]).set_index(bars.index)

    for column in ("imbalance_1_mean", "imbalance_5_mean", "imbalance_20_mean"):
        if column in out.columns:
            out[f"{column}_z_72"] = rolling_z(out[column], 72)
    if "spread_bps_mean" in out.columns:
        out["spread_z_72"] = rolling_z(out["spread_bps_mean"], 72)
    if "depth_20_mean" in out.columns:
        out["depth_change_24"] = out["depth_20_mean"] / out["depth_20_mean"].shift(24) - 1.0

    # Persistence of book pressure: a single lopsided snapshot is noise, a
    # sustained lean is information.
    if "imbalance_5_mean" in out.columns:
        out["imbalance_persistence_24"] = np.sign(out["imbalance_5_mean"]).rolling(
            24, min_periods=min_periods(24)
        ).mean()

    return prefix(out, "ob")
