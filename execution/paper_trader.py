"""Phase 6: the live paper-trading loop.

Real Hyperliquid market data in, simulated fills out. No keys, no orders,
no real capital -- the same :class:`~execution.paper_exchange.PaperExchange`
the backtest uses, driven by live prices instead of stored ones.

The loop deliberately reuses :class:`~strategy.signals.ModelStrategy` and
:class:`~backtest.engine.MarketView` rather than reimplementing the
decision logic. If live and backtest ran through different code, they would
drift apart and the backtest would stop being evidence about the live
system. Here they are literally the same functions.

One cycle:

    top up candles -> rebuild features -> settle funding -> check
    liquidation -> ask the strategy -> execute against the live book ->
    record everything -> save state

Everything is persisted each cycle, so a restart resumes rather than
silently starting a fresh $100k account.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal as signal_module
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest.engine import MarketView
from config.settings import INTERVAL_MS, SETTINGS, Settings
from data.database import MarketDatabase, ParquetStore
from data.downloader import HistoricalDownloader
from data.hyperliquid_client import HyperliquidInfoClient
from data.schemas import parse_l2_book, parse_meta_and_asset_ctxs
from execution.paper_exchange import PaperExchange
from execution.simulator import FillSimulator, MarketSnapshot
from execution.state_store import StateStore
from features.pipeline import FeatureConfig, build_universe
from models.predict import SignalGenerator
from models.train import TrainedModel
from risk.risk_engine import RiskEngine
from strategy.signals import ModelStrategy

log = logging.getLogger(__name__)


@dataclass
class CycleResult:
    ts_ms: int
    equity: float
    cash: float
    unrealized: float
    leverage: float
    open_positions: int
    orders_submitted: int
    fills: int
    decisions: list[Any] = field(default_factory=list)
    error: str | None = None

    def describe(self) -> str:
        if self.error:
            return f"cycle FAILED: {self.error}"
        return (
            f"equity ${self.equity:,.2f} | cash ${self.cash:,.2f} | "
            f"unrealised ${self.unrealized:+,.2f} | {self.open_positions} position(s) | "
            f"{self.leverage:.2f}x | {self.orders_submitted} order(s), {self.fills} fill(s)"
        )


class PaperTrader:
    def __init__(
        self,
        *,
        settings: Settings | None = None,
        model_path: Path | str | None = None,
        interval: str = "1h",
        long_threshold: float = 0.55,
        client: HyperliquidInfoClient | None = None,
        store: ParquetStore | None = None,
        exchange: PaperExchange | None = None,
    ) -> None:
        self.settings = settings or SETTINGS
        self.interval = interval
        self.interval_ms = INTERVAL_MS[interval]
        self.client = client or HyperliquidInfoClient(self.settings.hyperliquid)

        # "top:N" and "all" need the exchange to resolve. Done here rather
        # than in settings so importing config never touches the network.
        from data.universe import parse_spec, resolve as resolve_markets

        spec = self.settings.data.markets_spec
        if parse_spec(spec)[0] == "list":
            self.coins = list(self.settings.data.markets)
        else:
            self.coins = list(resolve_markets(
                spec, self.client, fallback=self.settings.data.markets
            ))
        if not self.coins:
            raise RuntimeError(
                f"MARKETS='{spec}' resolved to no markets; nothing to trade"
            )
        self.store = store or ParquetStore(self.settings.paths)
        self.downloader = HistoricalDownloader(
            client=self.client, store=self.store, settings=self.settings
        )

        model_path = Path(model_path or self.settings.paths.models / "model.pkl")
        if not model_path.exists():
            raise FileNotFoundError(
                f"no trained model at {model_path}. Run `python main.py train` first; "
                "the paper trader will not invent a signal."
            )
        self.model: TrainedModel = TrainedModel.load(model_path)
        self.generator = SignalGenerator(self.model, long_threshold=long_threshold)

        self.risk = RiskEngine(self.settings.risk, self.settings.execution)
        self.exchange = exchange or PaperExchange(
            self.settings.risk.starting_capital,
            config=self.settings.execution,
            simulator=FillSimulator(self.settings.execution),
        )

        self.state_store = StateStore(self.settings.paths.storage / "paper_account.json")
        restored = self.state_store.load_into(self.exchange)
        self._started_ms = int(restored.get("started_ms", time.time() * 1000))

        # The idle clock is restored, not restarted. Counting flat bars in
        # memory meant every redeploy handed the timer a fresh zero, so on a
        # service that redeploys more often than the limit it could never
        # fire and the system stayed flat indefinitely.
        self.strategy = ModelStrategy(
            self.generator, self.risk, {}, precompute=False,
            max_hold_ms=self.settings.strategy.max_hold_ms,
            max_idle_ms=self.settings.strategy.max_idle_ms,
            idle_since_ms=restored.get("idle_since_ms"),
        )
        self.feature_config = FeatureConfig(interval=interval)
        self._stop = asyncio.Event()
        self.cycles = 0
        self.errors = 0

    # -- lifecycle ---------------------------------------------------------

    def request_stop(self) -> None:
        self._stop.set()

    def close(self) -> None:
        self.client.close()

    # -- data --------------------------------------------------------------

    def _refresh_market_data(self) -> None:
        """Top up candles and funding from the exchange before deciding."""
        for coin in self.coins:
            self.downloader.backfill_candles(coin, self.interval)
            self.downloader.backfill_funding(coin)

    def _load_bars(self) -> dict[str, pd.DataFrame]:
        with MarketDatabase(self.store) as db:
            if not db.store.has_data("candles"):
                raise RuntimeError("no candle data stored; run a backfill first")
            bars = {}
            for coin in self.coins:
                # Only the tail is needed: features look back at most
                # MAX_LOOKBACK_BARS. Reading every stored bar for every coin
                # is affordable for three markets and is not for a hundred.
                frame = db.query(
                    "SELECT * FROM ("
                    "  SELECT ts_ms, open, high, low, close, volume, trades "
                    "  FROM candles WHERE coin = ? AND interval = ? "
                    "  ORDER BY ts_ms DESC LIMIT ?"
                    ") ORDER BY ts_ms",
                    [coin, self.interval, self.settings.data.live_lookback_bars],
                )
                frame["coin"] = coin
                frame["interval"] = self.interval
                bars[coin] = frame

            funding = {}
            if db.store.has_data("funding"):
                for coin in self.coins:
                    funding[coin] = db.query(
                        "SELECT * FROM ("
                        "  SELECT ts_ms, coin, funding_rate, premium FROM funding "
                        "  WHERE coin = ? ORDER BY ts_ms DESC LIMIT ?"
                        ") ORDER BY ts_ms",
                        [coin, self.settings.data.live_lookback_bars],
                    )

            # Order books too. If a collector has ever run, training saw
            # ob_* features and inference must see the same columns --
            # otherwise the model's feature list cannot be satisfied and the
            # strategy silently declines to signal on anything.
            books = {}
            if db.store.has_data("orderbook"):
                for coin in self.coins:
                    frame = db.query(
                        "SELECT ts_ms, recv_ts_ms, coin, side, level, px, sz, n_orders "
                        "FROM orderbook WHERE coin = ? ORDER BY ts_ms",
                        [coin],
                    )
                    if not frame.empty:
                        books[coin] = frame
        self._funding_frames = funding
        self._book_frames = books
        return bars

    def _live_snapshots(self, bars: dict[str, pd.DataFrame]) -> dict[str, MarketSnapshot]:
        """Build execution snapshots from the live book and the latest bar.

        The spread comes from the current order book rather than a constant,
        so a fill during a thin or fast market is charged what it would
        actually cost right then.
        """
        contexts: dict[str, dict[str, float]] = {}
        try:
            raw = self.client.meta_and_asset_ctxs()
            frame = parse_meta_and_asset_ctxs(
                raw, recv_ts_ms=int(time.time() * 1000), coins=self.coins
            )
            contexts = frame.set_index("coin").to_dict("index")
        except Exception as exc:  # noqa: BLE001
            log.warning("asset context poll failed; falling back to bar closes: %s", exc)

        snapshots: dict[str, MarketSnapshot] = {}
        for coin in self.coins:
            frame = bars[coin]
            if frame.empty:
                continue
            last = frame.iloc[-1]
            context = contexts.get(coin, {})

            mark = float(context.get("mark_px", last["close"]) or last["close"])
            half_spread = None
            try:
                book = parse_l2_book(
                    self.client.l2_book(coin), recv_ts_ms=int(time.time() * 1000), depth=1
                )
                bid = float(book[(book["side"] == "bid")]["px"].iloc[0])
                ask = float(book[(book["side"] == "ask")]["px"].iloc[0])
                mid = (bid + ask) / 2.0
                half_spread = max(0.0, (ask - bid) / 2.0 / mid)
                mark = mid
            except Exception as exc:  # noqa: BLE001
                log.warning("live book unavailable for %s: %s", coin, exc)

            snapshots[coin] = MarketSnapshot(
                ts_ms=int(time.time() * 1000),
                coin=coin,
                # The live price is what an order would touch right now, so
                # it plays the role the next bar's open plays in a backtest.
                open=mark,
                high=max(mark, float(last["high"])),
                low=min(mark, float(last["low"])),
                close=mark,
                volume=float(last["volume"]),
                interval_ms=self.interval_ms,
                funding_rate=float(context.get("funding", 0.0) or 0.0),
                mark_px=mark,
                half_spread=half_spread,
            )
        return snapshots

    # -- the cycle ---------------------------------------------------------

    def run_cycle(self) -> CycleResult:
        now_ms = int(time.time() * 1000)
        try:
            self._refresh_market_data()
            bars = self._load_bars()
            if any(frame.empty for frame in bars.values()):
                raise RuntimeError("a market has no stored bars yet")

            matrices = build_universe(
                bars,
                funding_by_coin=getattr(self, "_funding_frames", {}),
                book_by_coin=getattr(self, "_book_frames", {}),
                config=self.feature_config,
            )
            self.strategy.features = matrices
            snapshots = self._live_snapshots(bars)
            if not snapshots:
                raise RuntimeError("no live snapshots available")

            marks = {coin: snap.mark for coin, snap in snapshots.items()}

            # Funding and liquidation happen before any new decision, exactly
            # as they do in the backtest loop.
            self.exchange.apply_funding(now_ms, snapshots)
            self.exchange.check_liquidation(now_ms, snapshots)

            index = min(len(frame) for frame in bars.values()) - 1
            view = MarketView(
                index, now_ms, bars, snapshots, self.exchange, self.settings.risk, matrices
            )
            orders = list(self.strategy.on_bar(view))

            fills = 0
            for order in orders:
                snapshot = snapshots.get(order.coin)
                if snapshot is None:
                    continue
                if self.exchange.submit(order, snapshot, ts_ms=now_ms) is not None:
                    fills += 1

            equity = self.exchange.equity(marks)
            result = CycleResult(
                ts_ms=now_ms,
                equity=equity,
                cash=self.exchange.cash,
                unrealized=self.exchange.unrealized_pnl(marks),
                leverage=self.exchange.leverage(marks),
                open_positions=len(self.exchange.open_positions()),
                orders_submitted=len(orders),
                fills=fills,
                decisions=self.strategy.recent_decisions(now_ms - self.interval_ms),
            )

            self._record(result, marks)
            self.state_store.save(self.exchange, extra={
                "started_ms": self._started_ms,
                "idle_since_ms": self.strategy.idle_since_ms,
            })
            self.cycles += 1
            log.info("cycle %d | %s", self.cycles, result.describe())
            return result

        except Exception as exc:  # noqa: BLE001 - a bad cycle must not end the loop
            self.errors += 1
            log.exception("paper trading cycle failed")
            return CycleResult(now_ms, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, error=str(exc))

    # -- persistence -------------------------------------------------------

    def _record(self, result: CycleResult, marks: dict[str, float]) -> None:
        """Append this cycle to the performance record."""
        row = {
            "ts_ms": result.ts_ms,
            "equity": result.equity,
            "cash": result.cash,
            "unrealized_pnl": result.unrealized,
            "gross_notional": self.exchange.gross_notional(marks),
            "leverage": result.leverage,
            "open_positions": result.open_positions,
            "total_fees": self.exchange.total_fees,
            "total_funding": self.exchange.total_funding,
            "total_slippage": self.exchange.total_slippage,
            "liquidations": self.exchange.liquidation_count,
            "bankrupt": self.exchange.bankrupt,
            "profile": self.settings.risk.name,
        }
        self.store.upsert("equity", pd.DataFrame([row]))

        decisions = []
        for record in result.decisions:
            decisions.append({
                "ts_ms": record.ts_ms,
                "coin": record.coin,
                "probability": record.signal.probability,
                "confidence": record.signal.confidence,
                "direction": record.signal.direction,
                "regime": record.signal.regime,
                "action": record.action,
                "target_notional": record.target_notional,
                "current_notional": record.current_notional,
                "risk_verdict": record.risk.verdict.value,
                "risk_summary": record.risk.summary(),
                "top_features": "; ".join(f.describe() for f in record.signal.top_features[:5]),
            })
        if decisions:
            self.store.upsert("decisions", pd.DataFrame(decisions))

        new_fills = [f for f in self.exchange.fills if f.ts_ms >= result.ts_ms]
        if new_fills:
            self.store.upsert("paper_fills", pd.DataFrame([
                {
                    "ts_ms": f.ts_ms, "coin": f.coin, "fill_id": f"{f.ts_ms}-{i}",
                    "side": f.side.value, "size": f.size, "price": f.price,
                    "fee": f.fee, "slippage_cost": f.slippage_cost,
                    "realized_pnl": f.realized_pnl, "is_liquidation": f.is_liquidation,
                    "reason": f.context.reason,
                }
                for i, f in enumerate(new_fills)
            ]))

    # -- the loop ----------------------------------------------------------

    async def run(self, *, cycle_seconds: float | None = None, max_cycles: int | None = None):
        """Run until stopped, one decision per bar interval."""
        cycle_seconds = cycle_seconds or self.interval_ms / 1000.0
        loop = asyncio.get_running_loop()
        for sig in (signal_module.SIGINT, signal_module.SIGTERM):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, self.request_stop)

        log.info(
            "paper trader starting | profile=%s | markets=%s | model=%s | cycle=%.0fs",
            self.settings.risk.name, self.coins, self.model.backend_name, cycle_seconds,
        )

        results = []
        while not self._stop.is_set():
            results.append(await asyncio.to_thread(self.run_cycle))
            if max_cycles is not None and len(results) >= max_cycles:
                break
            # Wake shortly after each bar closes rather than on a raw timer,
            # so decisions line up with completed bars.
            delay = self._seconds_to_next_bar(cycle_seconds)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=delay)
            except asyncio.TimeoutError:
                continue
        self.close()
        return results

    def _seconds_to_next_bar(self, fallback: float) -> float:
        if self.interval_ms <= 0:
            return fallback
        now_ms = int(time.time() * 1000)
        next_close = ((now_ms // self.interval_ms) + 1) * self.interval_ms
        # A small delay past the close lets the exchange publish the final bar.
        return max(5.0, (next_close - now_ms) / 1000.0 + 15.0)
