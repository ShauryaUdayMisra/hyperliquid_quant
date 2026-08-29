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
    record everything -> save state -> retrain if the model is stale

The last step is what stops the model being frozen at whatever the first
boot fitted. On a schedule it refits on all stored history -- every bar the
live system has since been wrong about included -- and swaps the result in
without a restart. :mod:`models.scorecard` is the other half: it marks the
predictions this loop has already recorded against what the price actually
did, so "is it learning" is a number rather than a hope.

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
from features.pipeline import FeatureConfig, align_bars, build_universe
from models.predict import SignalGenerator
from models.retrain import RetrainOutcome, record as record_retrain, retrain_pair
from models.scorecard import LiveScorecard, load_scorecard
from models.train import TrainedModel, down_model_path
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
        short_threshold: float | None = None,
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
        self.model_path = model_path
        self.model: TrainedModel = TrainedModel.load(model_path)
        self.long_threshold = long_threshold
        self.short_threshold = (
            long_threshold if short_threshold is None else short_threshold
        )

        # The short side is optional and absent by default. Without it a low
        # P(up) means "do not buy", never "sell short" -- the two are not
        # complements, because between them sits "goes nowhere", which is
        # most bars. A short is only ever taken on a model trained to answer
        # that question.
        self.down_model_path = down_model_path(model_path)
        self.down_model: TrainedModel | None = None
        if self.down_model_path.exists():
            self.down_model = TrainedModel.load(self.down_model_path)
            log.info(
                "shorting enabled: %s, asking %s",
                self.down_model_path.name, self.down_model.label_config.name,
            )
        else:
            log.warning(
                "no down-model at %s: the system is long-only and can only sit "
                "out a falling market. Run `python main.py train` to build one.",
                self.down_model_path,
            )
        self.generator = self._build_generator()

        self.risk = RiskEngine(self.settings.risk, self.settings.execution)
        self.exchange = exchange or PaperExchange(
            self.settings.risk.starting_capital,
            config=self.settings.execution,
            simulator=FillSimulator(self.settings.execution),
        )

        self.state_store = StateStore(self.settings.paths.storage / "paper_account.json")
        restored = self.state_store.load_into(self.exchange)
        self._started_ms = int(restored.get("started_ms", time.time() * 1000))

        # The retrain clock runs from the model's own trained_at_ms, not from
        # boot. A restart therefore cannot postpone a retrain that is already
        # due -- on a service that redeploys several times a day, a clock
        # started at boot would mean the model never refits at all, which is
        # exactly the counter-versus-timestamp mistake the idle clock made.
        self.learning = self.settings.learning
        self._last_retrain_ms = int(
            restored.get("last_retrain_ms") or self.model.trained_at_ms
        )
        self.last_retrain: RetrainOutcome | None = None

        # The idle clock is restored, not restarted. Counting flat bars in
        # memory meant every redeploy handed the timer a fresh zero, so on a
        # service that redeploys more often than the limit it could never
        # fire and the system stayed flat indefinitely.
        self.strategy = ModelStrategy(
            self.generator, self.risk, {}, precompute=False,
            max_hold_ms=self.settings.strategy.max_hold_ms,
            min_hold_ms=self.settings.strategy.min_hold_ms,
            max_idle_ms=self.settings.strategy.max_idle_ms,
            idle_since_ms=restored.get("idle_since_ms"),
        )
        self.feature_config = FeatureConfig(interval=interval)
        self._stop = asyncio.Event()
        self.cycles = 0
        self.errors = 0

    def _build_generator(self) -> SignalGenerator:
        """The live generator, from whichever models are currently loaded."""
        return SignalGenerator(
            self.model,
            down_model=self.down_model,
            long_threshold=self.long_threshold,
            short_threshold=self.short_threshold,
        )

    # -- lifecycle ---------------------------------------------------------

    def request_stop(self) -> None:
        self._stop.set()

    def close(self) -> None:
        self.client.close()

    # -- data --------------------------------------------------------------

    def _refresh_market_data(self) -> None:
        """Top up candles and funding from the exchange before deciding.

        A market that cannot be refreshed is skipped, not fatal. This loop
        makes two requests per market, so a wide universe will eventually
        meet a 429 or a 500 from the exchange; letting that propagate killed
        the whole cycle, exited the trader, and left the supervisor
        restarting it into the same rate limit every thirty seconds.

        Skipping is safe because the strategy reads from storage, not from
        this call: a market whose top-up failed simply decides on the bars
        it already has, and the next cycle tries again. Silence is not
        acceptable though -- a market that keeps failing is a market whose
        data is going stale, so every failure is logged.
        """
        failed: list[str] = []
        for coin in self.coins:
            try:
                self.downloader.backfill_candles(coin, self.interval)
                self.downloader.backfill_funding(coin)
            except Exception as exc:  # noqa: BLE001 - one market must not stop the rest
                failed.append(coin)
                log.warning("could not refresh %s, deciding on stored bars: %s",
                            coin, exc)
        if failed:
            log.warning(
                "%d of %d market(s) went un-refreshed this cycle: %s",
                len(failed), len(self.coins), ", ".join(failed),
            )

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

        # Iterate the bars handed in, not self.coins: markets dropped
        # earlier in the cycle are absent here by design, and looking them
        # up by name raised a KeyError that undid the skip above.
        snapshots: dict[str, MarketSnapshot] = {}
        for coin, frame in bars.items():
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

    @staticmethod
    def _drop_empty_markets(
        bars: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        """Markets that actually have bars to decide on.

        A market with nothing stored is dropped for the cycle, not fatal.
        Across a wide universe there is almost always one coin mid-backfill
        or newly listed, and refusing to decide at all because of it stopped
        the other twenty-four from trading -- the same "one bad market halts
        everything" failure as an unhandled 429 in the refresh above.
        """
        empty = sorted(coin for coin, frame in bars.items() if frame.empty)
        if empty:
            log.warning(
                "%d of %d market(s) have no stored bars yet, skipping them "
                "this cycle: %s", len(empty), len(bars), ", ".join(empty),
            )
        return {coin: frame for coin, frame in bars.items() if not frame.empty}

    # -- the cycle ---------------------------------------------------------

    def run_cycle(self) -> CycleResult:
        now_ms = int(time.time() * 1000)
        try:
            self._refresh_market_data()
            bars = self._load_bars()

            # A market with nothing stored is dropped for this cycle, not
            # fatal. Across a wide universe there is almost always one coin
            # mid-backfill or newly listed, and refusing to decide at all
            # because of it stopped the other twenty-four from trading --
            # the same "one bad market halts everything" failure as an
            # unhandled 429 in the refresh above.
            bars = self._drop_empty_markets(bars)
            if not bars:
                raise RuntimeError(
                    "no market has stored bars yet; run a backfill first"
                )

            # One timestamp grid before any feature is computed. Markets are
            # topped up market-by-market and a 429 skips one entirely, so
            # their tails routinely differ in length -- which the cross-asset
            # block refuses to work with. Aligning the bars rather than the
            # matrices keeps the positional index below valid for both.
            bars = align_bars(bars)

            # Funding and books follow the surviving bars: a coin dropped
            # above must not reappear here with nothing to attach to.
            matrices = build_universe(
                bars,
                funding_by_coin={
                    c: f for c, f in getattr(self, "_funding_frames", {}).items()
                    if c in bars
                },
                book_by_coin={
                    c: f for c, f in getattr(self, "_book_frames", {}).items()
                    if c in bars
                },
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
            self.state_store.save(self.exchange, extra=self._state_extra())
            self.cycles += 1
            log.info("cycle %d | %s", self.cycles, result.describe())

            # Last, and deliberately after the cycle has been logged and
            # persisted: a retrain takes minutes, and nothing about this
            # cycle should be waiting on it.
            self._maybe_retrain(now_ms)
            return result

        except Exception as exc:  # noqa: BLE001 - a bad cycle must not end the loop
            self.errors += 1
            log.exception("paper trading cycle failed")
            return CycleResult(now_ms, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, error=str(exc))

    # -- learning ----------------------------------------------------------

    def _state_extra(self) -> dict[str, Any]:
        """Clocks that must survive a restart, saved beside the account."""
        return {
            "started_ms": self._started_ms,
            "idle_since_ms": self.strategy.idle_since_ms,
            "last_retrain_ms": self._last_retrain_ms,
            "model_trained_at_ms": self.model.trained_at_ms,
        }

    def retrain_due(self, now_ms: int) -> bool:
        every_ms = self.learning.every_ms
        if every_ms is None:
            return False
        return now_ms - self._last_retrain_ms >= every_ms

    def scorecard(self) -> LiveScorecard:
        """Mark the decisions this trader has already recorded.

        This is the honest out-of-sample number for the deployed model: its
        own calls, against what the price actually did. Unlike a backtest it
        cannot have been tuned, because the predictions were written down
        before the outcome existed.
        """
        return load_scorecard(
            coins=self.coins,
            interval=self.interval,
            interval_ms=self.interval_ms,
            label_config=self.model.label_config,
            entry_threshold=self.generator.long_threshold,
            store=self.store,
        )

    def _maybe_retrain(self, now_ms: int) -> list[RetrainOutcome] | None:
        """Refit on everything stored, if the model is due. Never raises.

        Run inline rather than on a worker thread. The cycle writes candles
        and reads them back through the same Parquet store, and a retrain
        reading the whole history underneath a concurrent write is a data
        race for no gain: decisions are hourly, so spending a few minutes of
        that hour learning costs nothing as long as it stays well inside the
        interval. If it ever does not, the next cycle simply starts late and
        the delay is visible in the log below.
        """
        if not self.retrain_due(now_ms):
            return None
        try:
            card = self.scorecard()
            log.info("before retraining -- %s", card.describe())

            # Both sides of the book, from one feature build. Each is
            # promoted on its own merit: a better long model is not held
            # back because the short model came out worse.
            outcomes = retrain_pair(
                coins=self.coins,
                interval=self.interval,
                model_path=self.model_path,
                incumbent=self.model,
                down_incumbent=self.down_model,
                settings=self.settings,
                learning=self.learning,
                store=self.store,
            )
            for outcome in outcomes:
                record_retrain(outcome, store=self.store)
                log.info("%s", outcome.describe())
                if outcome.promoted and outcome.model is not None:
                    self._swap_model(outcome.model, side=outcome.side)
            self.last_retrain = outcomes[0] if outcomes else None

            # The clock advances on any *completed* attempt, including a
            # rejection. A candidate that lost to the incumbent would lose
            # again on the same data an hour later; retrying every cycle
            # would burn the box for nothing. A "skipped" attempt -- too
            # little new data -- is cheap and is retried next cycle.
            if any(o.status != "skipped" for o in outcomes):
                self._last_retrain_ms = now_ms
                self.state_store.save(self.exchange, extra=self._state_extra())
            return outcomes
        except Exception as exc:  # noqa: BLE001 - learning must not stop trading
            log.exception("retrain step failed; trading continues on the current model")
            self._last_retrain_ms = now_ms
            return [RetrainOutcome(
                ts_ms=now_ms, status="failed", reason=f"{type(exc).__name__}: {exc}"
            )]

    def _swap_model(self, model: TrainedModel, *, side: str = "up") -> None:
        """Put a newly trained model behind the live strategy, in place.

        The strategy holds the generator, so replacing only ``self.model``
        would leave every decision still being made by the old one. The
        entry thresholds are carried across unchanged: retraining changes
        what a model has seen, never how confident it must be to trade.
        """
        if side == "down":
            self.down_model = model
        else:
            self.model = model
        self.generator = self._build_generator()
        self.strategy.generator = self.generator
        self.strategy._signal_cache = {}
        log.info(
            "live %s-model replaced: %s, %d features, trained through %s, "
            "walk-forward AUC %.4f",
            side, model.backend_name, len(model.features),
            pd.Timestamp(model.train_span[1], unit="ms", tz="UTC"),
            model.mean_val_auc,
        )

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
        log.info("learning | %s", self.learning.describe())

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
