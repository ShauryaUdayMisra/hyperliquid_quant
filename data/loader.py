"""Reads stored market data back out into the shapes the pipeline expects."""

from __future__ import annotations

import logging

import pandas as pd

from config.settings import SETTINGS, Settings
from data.database import MarketDatabase, ParquetStore

log = logging.getLogger(__name__)


def load_bars(
    coins: list[str],
    interval: str,
    *,
    store: ParquetStore | None = None,
    start_ms: int | None = None,
) -> dict[str, pd.DataFrame]:
    store = store or ParquetStore()
    if not store.has_data("candles"):
        raise RuntimeError(
            "no candle data on disk. Run `python main.py backfill` first."
        )
    out: dict[str, pd.DataFrame] = {}
    with MarketDatabase(store) as db:
        for coin in coins:
            sql = (
                "SELECT ts_ms, open, high, low, close, volume, trades FROM candles "
                "WHERE coin = ? AND interval = ?"
            )
            params: list[object] = [coin, interval]
            if start_ms is not None:
                sql += " AND ts_ms >= ?"
                params.append(start_ms)
            frame = db.query(sql + " ORDER BY ts_ms", params)
            if frame.empty:
                log.warning("no %s bars stored for %s", interval, coin)
                continue
            frame["coin"] = coin
            frame["interval"] = interval
            out[coin] = frame.reset_index(drop=True)
    if not out:
        raise RuntimeError(f"no bars found for {coins} at {interval}")
    return out


def load_funding(
    coins: list[str], *, store: ParquetStore | None = None
) -> dict[str, pd.DataFrame]:
    store = store or ParquetStore()
    if not store.has_data("funding"):
        return {}
    out = {}
    with MarketDatabase(store) as db:
        for coin in coins:
            frame = db.query(
                "SELECT ts_ms, coin, funding_rate, premium FROM funding "
                "WHERE coin = ? ORDER BY ts_ms",
                [coin],
            )
            if not frame.empty:
                out[coin] = frame.reset_index(drop=True)
    return out


def load_order_books(
    coins: list[str], *, store: ParquetStore | None = None
) -> dict[str, pd.DataFrame]:
    """Stored order-book snapshots, if a collector has been running.

    Usually empty for historical backtests: Hyperliquid serves no order-book
    history, so these features only exist for periods we observed live.
    """
    store = store or ParquetStore()
    if not store.has_data("orderbook"):
        return {}
    out = {}
    with MarketDatabase(store) as db:
        for coin in coins:
            frame = db.query(
                "SELECT ts_ms, recv_ts_ms, coin, side, level, px, sz, n_orders "
                "FROM orderbook WHERE coin = ? ORDER BY ts_ms",
                [coin],
            )
            if not frame.empty:
                out[coin] = frame.reset_index(drop=True)
    return out


def data_span(store: ParquetStore | None = None) -> dict[str, object]:
    store = store or ParquetStore()
    with MarketDatabase(store) as db:
        return db.table_summary().to_dict("records")
