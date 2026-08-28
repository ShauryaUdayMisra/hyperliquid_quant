"""Live market-data collector.

Three concurrent jobs, all writing into in-memory buffers that are flushed
to Parquet on a timer:

* **trades** -- WebSocket subscription per coin (the only way to get public
  prints; there is no REST trade-history endpoint).
* **order book** -- REST ``l2Book`` snapshot per coin on a fixed cadence.
  Polling rather than streaming keeps the data volume bounded and gives
  evenly-spaced snapshots, which is what an order-book-imbalance feature
  wants anyway.
* **asset context** -- REST ``metaAndAssetCtxs`` poll for funding, open
  interest, mark and oracle prices.

Candles are topped up on a slower timer by reusing the historical
downloader, so live and historical bars go through exactly one code path.

Every order-book row stores both the exchange timestamp and our local
receipt time. The difference is observed latency, which Phase 2's execution
simulator needs and which cannot be reconstructed after the fact.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
import time
from dataclasses import dataclass, field

import pandas as pd

from config.settings import SETTINGS, Settings
from data.database import ParquetStore
from data.downloader import HistoricalDownloader
from data.hyperliquid_client import HyperliquidInfoClient, HyperliquidWebSocket
from data.schemas import (
    SchemaError,
    parse_l2_book,
    parse_meta_and_asset_ctxs,
    parse_trades,
)

log = logging.getLogger(__name__)


@dataclass
class CollectorStats:
    started_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    trades_written: int = 0
    orderbook_rows_written: int = 0
    asset_ctx_written: int = 0
    candles_written: int = 0
    flushes: int = 0
    ws_reconnects: int = 0
    errors: int = 0

    def describe(self) -> str:
        uptime = (int(time.time() * 1000) - self.started_ms) / 1000
        return (
            f"uptime={uptime:,.0f}s flushes={self.flushes} "
            f"trades={self.trades_written:,} book_rows={self.orderbook_rows_written:,} "
            f"asset_ctx={self.asset_ctx_written:,} candles={self.candles_written:,} "
            f"ws_reconnects={self.ws_reconnects} errors={self.errors}"
        )


class LiveCollector:
    def __init__(
        self,
        settings: Settings | None = None,
        store: ParquetStore | None = None,
        client: HyperliquidInfoClient | None = None,
    ) -> None:
        self.settings = settings or SETTINGS
        self.store = store or ParquetStore(self.settings.paths)
        self.client = client or HyperliquidInfoClient(self.settings.hyperliquid)
        self.coins = list(self.settings.data.markets)
        self.stats = CollectorStats()

        self._trade_buffer: list[pd.DataFrame] = []
        self._book_buffer: list[pd.DataFrame] = []
        self._ctx_buffer: list[pd.DataFrame] = []
        self._buffer_lock = asyncio.Lock()
        self._stop = asyncio.Event()
        self._ws: HyperliquidWebSocket | None = None

    # -- lifecycle ---------------------------------------------------------

    def request_stop(self) -> None:
        self._stop.set()
        if self._ws is not None:
            self._ws.stop()

    async def run(self, duration_s: float | None = None) -> CollectorStats:
        """Collect until stopped, or for ``duration_s`` seconds."""
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, self.request_stop)

        tasks = [
            asyncio.create_task(self._run_trades(), name="trades"),
            asyncio.create_task(self._run_orderbook(), name="orderbook"),
            asyncio.create_task(self._run_asset_ctx(), name="asset_ctx"),
            asyncio.create_task(self._run_flush(), name="flush"),
            asyncio.create_task(self._run_candles(), name="candles"),
        ]
        if duration_s is not None:
            tasks.append(asyncio.create_task(self._run_deadline(duration_s), name="deadline"))

        try:
            await self._stop.wait()
        finally:
            self.request_stop()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self._flush()
            self.client.close()
        return self.stats

    async def _run_deadline(self, duration_s: float) -> None:
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self._stop.wait(), timeout=duration_s)
        self.request_stop()

    async def _sleep(self, seconds: float) -> bool:
        """Sleep, returning False if a stop was requested during it."""
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            return True
        return False

    # -- jobs --------------------------------------------------------------

    async def _run_trades(self) -> None:
        self._ws = HyperliquidWebSocket(
            [{"type": "trades", "coin": coin} for coin in self.coins],
            self.settings.hyperliquid,
        )
        async for channel, data in self._ws.stream():
            if self._stop.is_set():
                break
            if channel == "_reconnected":
                self.stats.ws_reconnects += 1
                log.warning("websocket reconnected; trade stream had a gap")
                continue
            if channel != "trades" or not data:
                continue
            try:
                df = parse_trades(data)
            except SchemaError as exc:
                self.stats.errors += 1
                log.warning("bad trade payload: %s", exc)
                continue
            async with self._buffer_lock:
                self._trade_buffer.append(df)

    async def _run_orderbook(self) -> None:
        interval = self.settings.data.orderbook_snapshot_interval_s
        depth = self.settings.data.orderbook_depth
        while not self._stop.is_set():
            for coin in self.coins:
                if self._stop.is_set():
                    break
                try:
                    raw = await asyncio.to_thread(self.client.l2_book, coin)
                    recv_ms = int(time.time() * 1000)
                    df = parse_l2_book(raw, recv_ts_ms=recv_ms, depth=depth)
                except Exception as exc:  # noqa: BLE001
                    self.stats.errors += 1
                    log.warning("l2Book poll failed for %s: %s", coin, exc)
                    continue
                async with self._buffer_lock:
                    self._book_buffer.append(df)
            if not await self._sleep(interval):
                return

    async def _run_asset_ctx(self) -> None:
        while not self._stop.is_set():
            try:
                raw = await asyncio.to_thread(self.client.meta_and_asset_ctxs)
                recv_ms = int(time.time() * 1000)
                df = parse_meta_and_asset_ctxs(raw, recv_ts_ms=recv_ms, coins=self.coins)
            except Exception as exc:  # noqa: BLE001
                self.stats.errors += 1
                log.warning("asset context poll failed: %s", exc)
            else:
                async with self._buffer_lock:
                    self._ctx_buffer.append(df)
            if not await self._sleep(self.settings.data.asset_ctx_interval_s):
                return

    async def _run_candles(self) -> None:
        """Keep candle history current using the same code path as backfill."""
        downloader = HistoricalDownloader(
            client=self.client, store=self.store, settings=self.settings
        )
        # Let the first flush happen before competing for the rate limiter.
        if not await self._sleep(30.0):
            return
        while not self._stop.is_set():
            for coin in self.coins:
                for interval in self.settings.data.candle_intervals:
                    if self._stop.is_set():
                        return
                    try:
                        result = await asyncio.to_thread(
                            downloader.backfill_candles, coin, interval
                        )
                        self.stats.candles_written += result.rows_new
                    except Exception as exc:  # noqa: BLE001
                        self.stats.errors += 1
                        log.warning("candle top-up failed for %s %s: %s", coin, interval, exc)
            if not await self._sleep(60.0):
                return

    async def _run_flush(self) -> None:
        while not self._stop.is_set():
            if not await self._sleep(self.settings.data.flush_interval_s):
                return
            await self._flush()

    # -- flushing ----------------------------------------------------------

    async def _flush(self) -> None:
        async with self._buffer_lock:
            trades = self._trade_buffer
            books = self._book_buffer
            ctxs = self._ctx_buffer
            self._trade_buffer, self._book_buffer, self._ctx_buffer = [], [], []

        if not (trades or books or ctxs):
            return

        def write() -> tuple[int, int, int]:
            written = [0, 0, 0]
            if trades:
                df = pd.concat(trades, ignore_index=True).drop_duplicates(subset=["coin", "tid"])
                written[0] = self.store.append("trades", df)
            if books:
                written[1] = self.store.append("orderbook", pd.concat(books, ignore_index=True))
            if ctxs:
                written[2] = self.store.upsert("asset_ctx", pd.concat(ctxs, ignore_index=True))
            return tuple(written)  # type: ignore[return-value]

        try:
            n_trades, n_books, n_ctx = await asyncio.to_thread(write)
        except Exception as exc:  # noqa: BLE001
            self.stats.errors += 1
            log.error("flush failed, buffered data for this window is lost: %s", exc)
            return

        self.stats.trades_written += n_trades
        self.stats.orderbook_rows_written += n_books
        self.stats.asset_ctx_written += n_ctx
        self.stats.flushes += 1
        log.info("flushed | %s", self.stats.describe())
