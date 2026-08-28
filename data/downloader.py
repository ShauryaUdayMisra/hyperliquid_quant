"""Historical backfill.

Walks Hyperliquid's paginated history endpoints forward in time and writes
the result to Parquet via :class:`~data.database.ParquetStore`. Safe to
re-run: writes are upserts keyed on (coin, interval, ts_ms), and each run
resumes from the newest bar already on disk.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import pandas as pd

from config.settings import FUNDING_INTERVAL_MS, INTERVAL_MS, SETTINGS, Settings
from data.database import MarketDatabase, ParquetStore
from data.hyperliquid_client import HyperliquidInfoClient
from data.schemas import parse_candles, parse_funding_history

log = logging.getLogger(__name__)

#: Hyperliquid returns at most 500 funding records per request.
FUNDING_PAGE_LIMIT = 500

#: Overlap re-requested on an incremental run, so a bar that was still
#: forming when we last ran is re-fetched now that it has closed.
RESUME_OVERLAP_BARS = 2


@dataclass
class BackfillResult:
    coin: str
    interval: str | None
    requested_from_ms: int
    requested_to_ms: int
    rows_fetched: int = 0
    rows_new: int = 0
    pages: int = 0
    empty: bool = False

    def describe(self) -> str:
        label = f"{self.coin}" + (f" {self.interval}" if self.interval else " funding")
        if self.empty:
            return f"  {label:<14} no data returned for the requested window"
        return (
            f"  {label:<14} {self.rows_fetched:>7} rows fetched, {self.rows_new:>7} new, "
            f"{self.pages} page(s), through "
            f"{pd.Timestamp(self.requested_to_ms, unit='ms', tz='UTC'):%Y-%m-%d %H:%M}"
        )


class HistoricalDownloader:
    def __init__(
        self,
        client: HyperliquidInfoClient | None = None,
        store: ParquetStore | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or SETTINGS
        self.client = client or HyperliquidInfoClient(self.settings.hyperliquid)
        self.store = store or ParquetStore(self.settings.paths)

    # -- candles -----------------------------------------------------------

    def backfill_candles(
        self,
        coin: str,
        interval: str,
        *,
        start_ms: int | None = None,
        end_ms: int | None = None,
        resume: bool = True,
    ) -> BackfillResult:
        step = INTERVAL_MS.get(interval)
        if step is None:
            raise ValueError(f"unsupported interval '{interval}'")

        now = int(time.time() * 1000)
        end_ms = now if end_ms is None else end_ms
        if start_ms is None:
            start_ms = self._resume_point(coin, interval, step) if resume else None
        if start_ms is None:
            start_ms = now - self.settings.data.backfill_days * 86_400_000
        start_ms = (start_ms // step) * step

        result = BackfillResult(coin, interval, start_ms, end_ms)
        page_span = self.settings.hyperliquid.candle_page_limit * step
        cursor = start_ms

        while cursor <= end_ms:
            page_end = min(cursor + page_span - step, end_ms)
            raw = self.client.candle_snapshot(coin, interval, cursor, page_end)
            result.pages += 1

            if not raw:
                # Either the coin was not listed yet, or this window is dead
                # air. Step past it rather than spinning on the same request.
                cursor = page_end + step
                continue

            # ``drop_incomplete`` keeps the currently-forming bar out of storage.
            df = parse_candles(raw, now_ms=now)
            if df.empty:
                cursor = page_end + step
                continue

            result.rows_fetched += len(df)
            result.rows_new += self.store.upsert("candles", df)

            newest = int(df["ts_ms"].max())
            # Guard against a server that ignores our cursor: always advance.
            cursor = max(newest + step, cursor + step)

        result.empty = result.rows_fetched == 0
        log.info(result.describe().strip())
        return result

    def _resume_point(self, coin: str, interval: str, step: int) -> int | None:
        with MarketDatabase(self.store) as db:
            last = db.last_candle_ts(coin, interval)
        if last is None:
            return None
        return last - RESUME_OVERLAP_BARS * step

    # -- funding -----------------------------------------------------------

    def backfill_funding(
        self,
        coin: str,
        *,
        start_ms: int | None = None,
        end_ms: int | None = None,
        resume: bool = True,
    ) -> BackfillResult:
        now = int(time.time() * 1000)
        end_ms = now if end_ms is None else end_ms
        if start_ms is None and resume:
            with MarketDatabase(self.store) as db:
                last = db.last_funding_ts(coin)
            start_ms = None if last is None else last - RESUME_OVERLAP_BARS * FUNDING_INTERVAL_MS
        if start_ms is None:
            start_ms = now - self.settings.data.backfill_days * 86_400_000

        result = BackfillResult(coin, None, start_ms, end_ms)
        page_span = FUNDING_PAGE_LIMIT * FUNDING_INTERVAL_MS
        cursor = start_ms

        while cursor <= end_ms:
            page_end = min(cursor + page_span, end_ms)
            raw = self.client.funding_history(coin, cursor, page_end)
            result.pages += 1

            if not raw:
                cursor = page_end + FUNDING_INTERVAL_MS
                continue

            df = parse_funding_history(raw)
            # Funding is published at the top of the hour; anything stamped in
            # the future is corruption, not a prediction.
            df = df.loc[df["ts_ms"] <= now]
            if df.empty:
                cursor = page_end + FUNDING_INTERVAL_MS
                continue

            result.rows_fetched += len(df)
            result.rows_new += self.store.upsert("funding", df)

            newest = int(df["ts_ms"].max())
            cursor = max(newest + FUNDING_INTERVAL_MS, cursor + FUNDING_INTERVAL_MS)

        result.empty = result.rows_fetched == 0
        log.info(result.describe().strip())
        return result

    # -- orchestration -----------------------------------------------------

    def backfill_all(
        self,
        coins: list[str] | None = None,
        intervals: list[str] | None = None,
        *,
        days: int | None = None,
        resume: bool = True,
    ) -> list[BackfillResult]:
        coins = coins or list(self.settings.data.markets)
        intervals = intervals or list(self.settings.data.candle_intervals)
        start_ms = None
        if days is not None:
            start_ms = int(time.time() * 1000) - days * 86_400_000

        results: list[BackfillResult] = []
        for coin in coins:
            for interval in intervals:
                results.append(
                    self.backfill_candles(coin, interval, start_ms=start_ms, resume=resume)
                )
            results.append(self.backfill_funding(coin, start_ms=start_ms, resume=resume))
        return results

    def close(self) -> None:
        self.client.close()
