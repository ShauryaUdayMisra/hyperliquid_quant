"""Event-driven backtest engine.

The loop is deliberately rigid about ordering, because ordering is where
look-ahead bias hides:

    for each bar i:
      1. execute orders decided on bar i-1, against bar i's OPEN
      2. settle funding at hour boundaries
      3. check liquidation against bar i's intrabar extremes
      4. mark to market at bar i's CLOSE and record equity
      5. ask the strategy for orders, showing it data up to bar i's close

A strategy therefore never trades on a price it used to decide. The earliest
a signal formed at bar i's close can be acted on is bar i+1's open, and it
pays spread, impact and latency to get there.

The engine holds no keys and cannot reach the network.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd

from config.settings import INTERVAL_MS, SETTINGS, RiskLimits
from execution.paper_exchange import ClosedTrade, Fill, Order, PaperExchange
from execution.simulator import FillSimulator, MarketSnapshot

log = logging.getLogger(__name__)


class LookaheadError(RuntimeError):
    """Raised when a strategy asks for data it cannot legally see."""


# --------------------------------------------------------------------------
# What a strategy is allowed to see
# --------------------------------------------------------------------------

class MarketView:
    """A read-only, point-in-time window onto the market and the account.

    Every accessor is clipped at the current bar. Asking for anything later
    raises rather than silently returning the future.
    """

    def __init__(
        self,
        index: int,
        ts_ms: int,
        bars: Mapping[str, pd.DataFrame],
        snapshots: Mapping[str, MarketSnapshot],
        exchange: PaperExchange,
        risk: RiskLimits,
        features: Mapping[str, pd.DataFrame] | None = None,
    ) -> None:
        self._i = index
        self.ts_ms = ts_ms
        self._bars = bars
        self._snapshots = snapshots
        self.exchange = exchange
        self.risk = risk
        self._features = features or {}

    @property
    def coins(self) -> list[str]:
        return list(self._bars)

    @property
    def bar_index(self) -> int:
        return self._i

    def snapshot(self, coin: str) -> MarketSnapshot:
        return self._snapshots[coin]

    def price(self, coin: str) -> float:
        return self._snapshots[coin].close

    def marks(self) -> dict[str, float]:
        return {coin: snap.mark for coin, snap in self._snapshots.items()}

    def history(self, coin: str, lookback: int | None = None) -> pd.DataFrame:
        """Bars up to and including the current one. Never beyond."""
        frame = self._bars[coin]
        end = self._i + 1
        start = 0 if lookback is None else max(0, end - lookback)
        return frame.iloc[start:end]

    def features(self, coin: str) -> pd.Series | None:
        """Point-in-time feature row for the current bar, if a pipeline ran."""
        frame = self._features.get(coin)
        if frame is None or self._i >= len(frame):
            return None
        return frame.iloc[self._i]

    def equity(self) -> float:
        return self.exchange.equity(self.marks())

    def position_size(self, coin: str) -> float:
        return self.exchange.position(coin).size

    def position_age_ms(self, coin: str) -> int | None:
        """How long the open position has been held, or None if flat.

        Read from the position itself rather than counted in the strategy,
        so a restarted live process inherits the true age from the restored
        account instead of resetting every holding clock to zero.
        """
        position = self.exchange.position(coin)
        if position.is_flat or position.opened_ts_ms is None:
            return None
        return max(0, self.ts_ms - int(position.opened_ts_ms))

    def notional_to_size(self, coin: str, notional: float) -> float:
        price = self.price(coin)
        return 0.0 if price <= 0 else notional / price


class Strategy(Protocol):
    name: str

    def on_bar(self, view: MarketView) -> Sequence[Order]:
        ...


# --------------------------------------------------------------------------
# Configuration and results
# --------------------------------------------------------------------------

@dataclass
class BacktestConfig:
    interval: str = "1h"
    starting_capital: float | None = None
    warmup_bars: int = 0
    #: Liquidate using each bar's adverse extreme rather than its close.
    intrabar_liquidation: bool = True
    #: Per-coin exchange maximum leverage, from Hyperliquid's meta endpoint.
    asset_max_leverage: dict[str, float] = field(default_factory=dict)
    #: Stop the run once the account is wiped out.
    stop_on_bankruptcy: bool = True


@dataclass
class BacktestResult:
    equity_curve: pd.DataFrame
    trades: list[ClosedTrade]
    fills: list[Fill]
    exchange: PaperExchange
    config: BacktestConfig
    risk: RiskLimits
    strategy_name: str
    bars_processed: int = 0
    halted_reason: str | None = None

    @property
    def starting_equity(self) -> float:
        return self.exchange.starting_capital

    @property
    def final_equity(self) -> float:
        return float(self.equity_curve["equity"].iloc[-1]) if len(self.equity_curve) else 0.0

    @property
    def total_return(self) -> float:
        return self.final_equity / self.starting_equity - 1.0

    def reconcile(self, tolerance: float = 1e-6) -> dict[str, Any]:
        """Prove the books balance.

        Two independent identities must hold at every step:

        * ``equity == cash + unrealised P&L`` (checked bar by bar), and
        * ``final equity - starting equity == realised P&L + unrealised P&L
          - fees - funding`` (checked once, from the fill ledger).

        If either drifts, the accounting is wrong and every downstream
        number is meaningless.
        """
        exchange = self.exchange
        curve = self.equity_curve
        identity_error = (
            float((curve["equity"] - curve["cash"] - curve["unrealized_pnl"]).abs().max())
            if len(curve)
            else 0.0
        )

        realized = sum(f.realized_pnl for f in exchange.fills)
        final_marks = {
            coin: float(curve[f"mark_{coin}"].iloc[-1])
            for coin in self._marked_coins()
        } if len(curve) else {}
        unrealized = exchange.unrealized_pnl(final_marks)
        expected_change = realized + unrealized - exchange.total_fees - exchange.total_funding
        actual_change = self.final_equity - self.starting_equity
        pnl_error = abs(expected_change - actual_change)

        return {
            "equity_identity_max_error": identity_error,
            "pnl_attribution_error": pnl_error,
            "realized_pnl": realized,
            "unrealized_pnl": unrealized,
            "fees": exchange.total_fees,
            "funding": exchange.total_funding,
            "slippage": exchange.total_slippage,
            "expected_equity_change": expected_change,
            "actual_equity_change": actual_change,
            "balanced": identity_error <= tolerance and pnl_error <= tolerance,
        }

    def _marked_coins(self) -> list[str]:
        return [c[5:] for c in self.equity_curve.columns if c.startswith("mark_")]


# --------------------------------------------------------------------------
# The engine
# --------------------------------------------------------------------------

class BacktestEngine:
    def __init__(
        self,
        bars: Mapping[str, pd.DataFrame],
        config: BacktestConfig | None = None,
        *,
        risk: RiskLimits | None = None,
        exchange: PaperExchange | None = None,
        features: Mapping[str, pd.DataFrame] | None = None,
    ) -> None:
        self.config = config or BacktestConfig()
        self.risk = risk or SETTINGS.risk
        self.bars = self._align(bars)
        self.features = features
        self.exchange = exchange or PaperExchange(
            self.config.starting_capital
            if self.config.starting_capital is not None
            else self.risk.starting_capital
        )
        self.interval_ms = INTERVAL_MS.get(self.config.interval, 3_600_000)

    # -- data preparation --------------------------------------------------

    @staticmethod
    def _align(bars: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """Restrict every coin to the timestamps they all share.

        An inner join is the only safe choice: forward-filling a missing bar
        would invent a price, and a coin whose data starts later would
        otherwise be silently back-filled with the first price it ever had.
        """
        if not bars:
            raise ValueError("no bars supplied")
        common: set[int] | None = None
        for coin, frame in bars.items():
            required = {"ts_ms", "open", "high", "low", "close", "volume"}
            missing = required - set(frame.columns)
            if missing:
                raise ValueError(f"{coin} bars missing columns: {sorted(missing)}")
            stamps = set(frame["ts_ms"].astype("int64"))
            common = stamps if common is None else (common & stamps)
        if not common:
            raise ValueError("coins share no common timestamps")

        aligned = {}
        keep = np.array(sorted(common), dtype=np.int64)
        for coin, frame in bars.items():
            sub = frame.loc[frame["ts_ms"].isin(keep)].sort_values("ts_ms").reset_index(drop=True)
            aligned[coin] = sub
        return aligned

    def _snapshot(self, coin: str, i: int) -> MarketSnapshot:
        row = self.bars[coin].iloc[i]
        funding = float(row["funding_rate"]) if "funding_rate" in row else 0.0
        half_spread = float(row["half_spread"]) if "half_spread" in row else None
        return MarketSnapshot(
            ts_ms=int(row["ts_ms"]),
            coin=coin,
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
            interval_ms=self.interval_ms,
            funding_rate=funding if np.isfinite(funding) else 0.0,
            half_spread=half_spread,
            max_asset_leverage=self.config.asset_max_leverage.get(coin),
        )

    # -- the loop ----------------------------------------------------------

    def run(self, strategy: Strategy) -> BacktestResult:
        coins = list(self.bars)
        n = len(self.bars[coins[0]])
        exchange = self.exchange
        leverages = self.config.asset_max_leverage

        rows: list[dict[str, Any]] = []
        pending: Sequence[Order] = []
        halted: str | None = None

        for i in range(n):
            snapshots = {coin: self._snapshot(coin, i) for coin in coins}
            ts_ms = snapshots[coins[0]].ts_ms

            # 1. Orders decided on the previous bar hit this bar's open.
            for order in pending:
                exchange.submit(order, snapshots[order.coin], ts_ms=ts_ms)
            pending = []

            # 2. Funding, on the hour.
            exchange.apply_funding(ts_ms, snapshots)

            # 3. Liquidation, against this bar's adverse extreme.
            exchange.check_liquidation(
                ts_ms,
                snapshots,
                leverages=leverages,
                use_intrabar_worst_case=self.config.intrabar_liquidation,
            )

            # 4. Mark to market at the close.
            marks = {coin: snap.mark for coin, snap in snapshots.items()}
            state = exchange.snapshot_state(marks)
            row = {"ts_ms": ts_ms, **state}
            for coin, mark in marks.items():
                row[f"mark_{coin}"] = mark
                row[f"pos_{coin}"] = exchange.position(coin).size
            rows.append(row)

            if exchange.bankrupt and self.config.stop_on_bankruptcy:
                halted = f"account wiped out at bar {i}"
                log.error(halted)
                break

            # 5. Decide, seeing only what has closed.
            if i >= self.config.warmup_bars and i < n - 1:
                view = MarketView(
                    i, ts_ms, self.bars, snapshots, exchange, self.risk, self.features
                )
                pending = strategy.on_bar(view) or []

        curve = pd.DataFrame(rows)
        if len(curve):
            curve["ts"] = pd.to_datetime(curve["ts_ms"], unit="ms", utc=True)
            curve["returns"] = curve["equity"].pct_change().fillna(0.0)

        return BacktestResult(
            equity_curve=curve,
            trades=list(exchange.closed_trades),
            fills=list(exchange.fills),
            exchange=exchange,
            config=self.config,
            risk=self.risk,
            strategy_name=getattr(strategy, "name", type(strategy).__name__),
            bars_processed=len(rows),
            halted_reason=halted,
        )
