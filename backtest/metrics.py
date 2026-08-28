"""Performance analytics.

Conventions that matter for honesty:

* Sharpe and Sortino annualise from the bar interval actually used, not a
  hardcoded 252 or 365. Getting this wrong is the single easiest way to
  inflate a crypto Sharpe by ~5x.
* Returns are simple period returns on equity, which already nets fees,
  funding and slippage. There is no "gross of costs" number anywhere.
* Drawdown is peak-to-trough on the equity curve, not on closed trades.
* Every ratio is undefined rather than infinite when its denominator is
  zero, and reports as NaN so it cannot be mistaken for a good result.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from execution.paper_exchange import ClosedTrade, Fill

SECONDS_PER_YEAR = 365.25 * 24 * 3600

#: Return dispersion below this is floating-point noise, not signal.
#: Dividing by it manufactures Sharpe ratios in the millions, which is
#: the classic way a broken backtest announces a "spectacular" edge.
MIN_RETURN_STD = 1e-12


def _safe_div(numerator: float, denominator: float) -> float:
    return float("nan") if denominator == 0 else numerator / denominator


def periods_per_year(interval_ms: int) -> float:
    return SECONDS_PER_YEAR / (interval_ms / 1000.0)


@dataclass
class PerformanceMetrics:
    bars: int = 0
    days: float = 0.0
    starting_equity: float = 0.0
    final_equity: float = 0.0
    total_pnl: float = 0.0
    total_return: float = 0.0
    cagr: float = float("nan")
    volatility_annual: float = float("nan")
    sharpe: float = float("nan")
    sortino: float = float("nan")
    calmar: float = float("nan")
    max_drawdown: float = 0.0
    max_drawdown_duration_bars: int = 0
    trades: int = 0
    win_rate: float = float("nan")
    profit_factor: float = float("nan")
    average_trade: float = float("nan")
    average_win: float = float("nan")
    average_loss: float = float("nan")
    largest_win: float = 0.0
    largest_loss: float = 0.0
    liquidations: int = 0
    exposure: float = 0.0
    turnover_annual: float = float("nan")
    total_fees: float = 0.0
    total_funding: float = 0.0
    total_slippage: float = 0.0
    cost_drag_bps_per_bar: float = float("nan")
    bankrupt: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def describe(self) -> str:
        def pct(x: float) -> str:
            return "n/a" if not np.isfinite(x) else f"{x:+.2%}"

        def num(x: float) -> str:
            return "n/a" if not np.isfinite(x) else f"{x:.2f}"

        return "\n".join(
            [
                f"  period            : {self.bars:,} bars ({self.days:,.1f} days)",
                f"  equity            : ${self.starting_equity:,.2f} -> ${self.final_equity:,.2f}",
                f"  total return      : {pct(self.total_return)}   P&L ${self.total_pnl:,.2f}",
                f"  CAGR              : {pct(self.cagr)}",
                f"  volatility (ann.) : {pct(self.volatility_annual)}",
                f"  Sharpe            : {num(self.sharpe)}",
                f"  Sortino           : {num(self.sortino)}",
                f"  Calmar            : {num(self.calmar)}",
                f"  max drawdown      : {self.max_drawdown:.2%} "
                f"({self.max_drawdown_duration_bars:,} bars underwater)",
                f"  trades            : {self.trades:,}   win rate {pct(self.win_rate)}",
                f"  profit factor     : {num(self.profit_factor)}",
                f"  avg trade         : ${self.average_trade:,.2f}  "
                f"(win ${self.average_win:,.2f} / loss ${self.average_loss:,.2f})",
                f"  exposure          : {self.exposure:.1%} of bars",
                f"  turnover (ann.)   : {num(self.turnover_annual)}x equity",
                f"  costs             : fees ${self.total_fees:,.2f} | "
                f"funding ${self.total_funding:,.2f} | slippage ${self.total_slippage:,.2f}",
                f"  liquidations      : {self.liquidations}"
                + ("   ACCOUNT WIPED OUT" if self.bankrupt else ""),
            ]
        )


def max_drawdown(equity: Sequence[float] | np.ndarray) -> tuple[float, int]:
    """Worst peak-to-trough decline, and the longest underwater stretch."""
    values = np.asarray(equity, dtype=float)
    if values.size == 0:
        return 0.0, 0
    peaks = np.maximum.accumulate(values)
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdowns = np.where(peaks > 0, (values - peaks) / peaks, 0.0)
    worst = float(drawdowns.min()) if drawdowns.size else 0.0

    longest = current = 0
    for value, peak in zip(values, peaks):
        current = current + 1 if value < peak else 0
        longest = max(longest, current)
    return abs(worst), longest


def sharpe_ratio(returns: np.ndarray, ppy: float, risk_free: float = 0.0) -> float:
    if returns.size < 2:
        return float("nan")
    excess = returns - risk_free / ppy
    std = float(excess.std(ddof=1))
    if not np.isfinite(std) or std < MIN_RETURN_STD:
        return float("nan")
    return float(excess.mean() / std * np.sqrt(ppy))


def sortino_ratio(returns: np.ndarray, ppy: float, risk_free: float = 0.0) -> float:
    if returns.size < 2:
        return float("nan")
    excess = returns - risk_free / ppy
    downside = excess[excess < 0]
    if downside.size == 0:
        return float("nan")
    # Downside deviation uses the full sample in the denominator on purpose:
    # dividing only by the losing periods flatters a strategy that rarely loses.
    dd = float(np.sqrt(np.sum(downside ** 2) / excess.size))
    if not np.isfinite(dd) or dd < MIN_RETURN_STD:
        return float("nan")
    return float(excess.mean() / dd * np.sqrt(ppy))


def compute_metrics(
    equity_curve: pd.DataFrame,
    trades: Sequence[ClosedTrade],
    fills: Sequence[Fill],
    *,
    interval_ms: int,
    starting_equity: float,
    liquidations: int = 0,
    bankrupt: bool = False,
    position_columns: Sequence[str] | None = None,
) -> PerformanceMetrics:
    metrics = PerformanceMetrics(starting_equity=starting_equity)
    if equity_curve is None or len(equity_curve) == 0:
        return metrics

    equity = equity_curve["equity"].to_numpy(dtype=float)
    metrics.bars = len(equity)
    metrics.final_equity = float(equity[-1])
    metrics.total_pnl = metrics.final_equity - starting_equity
    metrics.total_return = _safe_div(metrics.total_pnl, starting_equity)
    metrics.liquidations = liquidations
    metrics.bankrupt = bankrupt

    span_ms = int(equity_curve["ts_ms"].iloc[-1] - equity_curve["ts_ms"].iloc[0]) + interval_ms
    years = span_ms / 1000.0 / SECONDS_PER_YEAR
    metrics.days = span_ms / 1000.0 / 86_400.0

    returns = np.diff(equity) / np.where(equity[:-1] == 0, np.nan, equity[:-1])
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    ppy = periods_per_year(interval_ms)

    metrics.volatility_annual = (
        float(returns.std(ddof=1) * np.sqrt(ppy)) if returns.size > 1 else float("nan")
    )
    if np.isfinite(metrics.volatility_annual) and metrics.volatility_annual < MIN_RETURN_STD:
        metrics.volatility_annual = 0.0
    metrics.sharpe = sharpe_ratio(returns, ppy)
    metrics.sortino = sortino_ratio(returns, ppy)
    metrics.max_drawdown, metrics.max_drawdown_duration_bars = max_drawdown(equity)

    if years > 0 and starting_equity > 0 and metrics.final_equity > 0:
        metrics.cagr = float((metrics.final_equity / starting_equity) ** (1 / years) - 1)
    elif metrics.final_equity <= 0:
        metrics.cagr = -1.0
    metrics.calmar = _safe_div(metrics.cagr, metrics.max_drawdown)

    # -- trade statistics --
    metrics.trades = len(trades)
    if trades:
        pnls = np.array([t.net_pnl for t in trades], dtype=float)
        wins, losses = pnls[pnls > 0], pnls[pnls <= 0]
        metrics.win_rate = len(wins) / len(pnls)
        metrics.profit_factor = _safe_div(float(wins.sum()), float(abs(losses.sum())))
        metrics.average_trade = float(pnls.mean())
        metrics.average_win = float(wins.mean()) if wins.size else 0.0
        metrics.average_loss = float(losses.mean()) if losses.size else 0.0
        metrics.largest_win = float(pnls.max())
        metrics.largest_loss = float(pnls.min())

    # -- exposure and turnover --
    position_columns = position_columns or [
        c for c in equity_curve.columns if c.startswith("pos_")
    ]
    if position_columns:
        held = (equity_curve[position_columns].abs().sum(axis=1) > 1e-12).to_numpy()
        metrics.exposure = float(held.mean())

    traded_notional = sum(abs(f.size * f.price) for f in fills)
    average_equity = float(np.mean(equity))
    if average_equity > 0 and years > 0:
        metrics.turnover_annual = traded_notional / average_equity / years

    exchange_costs = 0.0
    for fill in fills:
        exchange_costs += fill.fee + fill.slippage_cost
    metrics.total_fees = sum(f.fee for f in fills)
    metrics.total_slippage = sum(f.slippage_cost for f in fills)
    metrics.total_funding = sum(t.funding for t in trades)
    if metrics.bars and average_equity > 0:
        metrics.cost_drag_bps_per_bar = (
            (exchange_costs + metrics.total_funding) / average_equity / metrics.bars * 10_000
        )
    return metrics


# --------------------------------------------------------------------------
# Regime breakdown
# --------------------------------------------------------------------------

def metrics_by_regime(
    equity_curve: pd.DataFrame,
    regimes: pd.Series,
    *,
    interval_ms: int,
) -> pd.DataFrame:
    """Per-regime return, volatility, Sharpe and share of time.

    ``regimes`` must be a point-in-time label aligned to ``equity_curve``:
    the regime as known at that bar's close, never a label that used later
    data to decide.
    """
    if len(equity_curve) == 0 or len(regimes) == 0:
        return pd.DataFrame(
            columns=["regime", "bars", "share", "total_return", "volatility", "sharpe", "max_dd"]
        )

    frame = equity_curve.copy().reset_index(drop=True)
    labels = pd.Series(regimes).reset_index(drop=True).reindex(range(len(frame)))
    frame["regime"] = labels.ffill().fillna("unknown").to_numpy()
    frame["ret"] = frame["equity"].pct_change().fillna(0.0)
    ppy = periods_per_year(interval_ms)

    rows = []
    for regime, group in frame.groupby("regime", sort=True):
        returns = group["ret"].to_numpy(dtype=float)
        equity = group["equity"].to_numpy(dtype=float)
        dd, _ = max_drawdown(equity)
        rows.append(
            {
                "regime": regime,
                "bars": len(group),
                "share": len(group) / len(frame),
                # Compounded return of the bars spent in this regime.
                "total_return": float(np.prod(1 + returns) - 1),
                "volatility": float(returns.std(ddof=1) * np.sqrt(ppy)) if len(returns) > 1 else float("nan"),
                "sharpe": sharpe_ratio(returns, ppy),
                "max_dd": dd,
            }
        )
    return pd.DataFrame(rows).sort_values("bars", ascending=False).reset_index(drop=True)
