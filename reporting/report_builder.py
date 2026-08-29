"""Builds the 6-hourly report.

Every sentence in the output is generated from a number that exists in the
performance record. There is no language model here and no invented
narrative: the "market reasoning" section is templated directly from the
regime label, the model probability and the top feature contributions.

When the model is flat or unconfident, the report says so plainly rather
than manufacturing a story, because a report that always sounds confident
is worse than useless.
"""

from __future__ import annotations

import html
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from backtest.metrics import compute_metrics, max_drawdown, periods_per_year, sharpe_ratio
from config.settings import SETTINGS, Settings
from data.database import MarketDatabase, ParquetStore
from execution.paper_exchange import PaperExchange
from features import regime as regime_module
from risk.risk_engine import RiskEngine

log = logging.getLogger(__name__)

WINDOW_MS = 6 * 3_600_000
DISCLAIMER = (
    "Simulated paper trading. No real capital at risk. Not financial advice."
)


def _fmt_usd(value: float) -> str:
    return f"${value:,.2f}"


def _fmt_signed(value: float) -> str:
    return f"{'+' if value >= 0 else '-'}${abs(value):,.2f}"


def _fmt_pct(value: float, digits: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:+.{digits}%}"


def _fmt_num(value: float, digits: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


@dataclass
class PositionLine:
    coin: str
    side: str
    size_usd: float
    size_base: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pct: float
    leverage: float
    funding_paid: float


@dataclass
class TradeLine:
    coin: str
    direction: str
    opened: str
    closed: str
    entry_price: float
    exit_price: float
    net_pnl: float
    fees: float
    funding: float
    liquidated: bool
    reason: str


@dataclass
class PlanLine:
    coin: str
    probability: float
    confidence: float
    direction: str
    regime: str
    target_notional: float
    current_notional: float
    action: str
    risk_summary: str
    drivers: list[tuple[str, float]] = field(default_factory=list)
    reasoning: str = ""


@dataclass
class ReportData:
    generated_ms: int
    window_start_ms: int
    window_end_ms: int
    profile: str
    starting_capital: float
    equity: float
    cash: float
    unrealized: float

    pnl_window: float
    pnl_today: float
    pnl_all_time: float

    window_metrics: dict[str, Any] = field(default_factory=dict)
    all_time_metrics: dict[str, Any] = field(default_factory=dict)
    positions: list[PositionLine] = field(default_factory=list)
    trades: list[TradeLine] = field(default_factory=list)
    plans: list[PlanLine] = field(default_factory=list)
    risk_status: dict[str, Any] = field(default_factory=dict)
    vetoes: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    #: Which model is deciding, how its own past calls turned out, and when
    #: it next refits. A model that retrains itself on a schedule makes a
    #: change in behaviour indistinguishable from a change in the market
    #: unless the report says which one changed.
    learning: dict[str, Any] = field(default_factory=dict)

    @property
    def total_return(self) -> float:
        return self.equity / self.starting_capital - 1.0 if self.starting_capital else 0.0


# --------------------------------------------------------------------------
# Narrative, generated strictly from numbers
# --------------------------------------------------------------------------

def market_reasoning(plan: PlanLine) -> str:
    """One grounded paragraph per market. No invention."""
    parts: list[str] = []

    regime_text = {
        "trending_up": "The market is trending higher",
        "trending_down": "The market is trending lower",
        "ranging": "The market is range-bound",
        "high_volatility": "The market is unusually volatile",
        "unknown": "The regime is undetermined (not enough history)",
    }.get(plan.regime, f"Regime: {plan.regime}")
    parts.append(regime_text + ".")

    if plan.confidence < 0.1:
        parts.append(
            f"The model is close to its base rate at P(up)={plan.probability:.3f}, "
            "so it has no meaningful opinion here."
        )
    else:
        lean = {
            "long": "an upward lean",
            "short": "a downward lean",
            "flat": "no directional lean",
        }[plan.direction]
        parts.append(
            f"The model puts P(up) at {plan.probability:.3f} with confidence "
            f"{plan.confidence:.2f}, {lean}."
        )

    if plan.drivers:
        described = ", ".join(
            f"{name} {'supporting' if value >= 0 else 'opposing'} ({value:+.3f})"
            for name, value in plan.drivers[:4]
        )
        parts.append(f"Largest contributors: {described}.")
    else:
        parts.append("No feature attribution was available for this decision.")

    if plan.current_notional > 0:
        parts.append(f"Currently holding {_fmt_usd(plan.current_notional)} of exposure.")
    else:
        parts.append("Currently flat.")

    if plan.target_notional > 0:
        parts.append(f"Target exposure {_fmt_usd(plan.target_notional)}; {plan.risk_summary}.")
    elif "veto" in plan.action.lower() or "skip" in plan.action.lower():
        parts.append(f"No exposure taken: {plan.action}.")
    else:
        parts.append("No new exposure planned.")

    return " ".join(parts)


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------

class ReportBuilder:
    def __init__(
        self,
        *,
        settings: Settings | None = None,
        store: ParquetStore | None = None,
        window_ms: int = WINDOW_MS,
    ) -> None:
        self.settings = settings or SETTINGS
        self.store = store or ParquetStore(self.settings.paths)
        self.window_ms = window_ms

    def build(
        self,
        *,
        exchange: PaperExchange,
        marks: dict[str, float],
        risk_engine: RiskEngine | None = None,
        latest_decisions: dict[str, Any] | None = None,
        learning: dict[str, Any] | None = None,
        now_ms: int | None = None,
    ) -> ReportData:
        now_ms = now_ms or int(pd.Timestamp.now("UTC").timestamp() * 1000)
        window_start = now_ms - self.window_ms
        day_start = (now_ms // 86_400_000) * 86_400_000

        equity = exchange.equity(marks)
        curve = self._load_equity_curve()

        data = ReportData(
            generated_ms=now_ms,
            window_start_ms=window_start,
            window_end_ms=now_ms,
            profile=self.settings.risk.name,
            starting_capital=exchange.starting_capital,
            equity=equity,
            cash=exchange.cash,
            unrealized=exchange.unrealized_pnl(marks),
            pnl_window=self._pnl_since(curve, window_start, equity),
            pnl_today=self._pnl_since(curve, day_start, equity),
            pnl_all_time=equity - exchange.starting_capital,
        )

        data.window_metrics = self._metrics_for(curve, window_start, exchange, now_ms)
        data.all_time_metrics = self._metrics_for(curve, 0, exchange, now_ms)
        data.positions = self._positions(exchange, marks, equity)
        data.trades = self._trades(exchange, window_start)
        data.plans = self._plans(latest_decisions or {}, exchange, marks)
        data.learning = learning or {}

        if risk_engine is not None:
            state = risk_engine.account_state(
                ts_ms=now_ms,
                equity=equity,
                gross_notional=exchange.gross_notional(marks),
                open_positions=len(exchange.open_positions()),
            )
            data.risk_status = risk_engine.status(state)
            data.vetoes = [
                f"{pd.Timestamp(ts, unit='ms', tz='UTC'):%H:%M} {coin}: {detail}"
                for ts, coin, detail in risk_engine.veto_log
                if ts >= window_start
            ]

        if curve.empty:
            data.notes.append(
                "No stored equity history yet - window statistics will fill in "
                "once the trader has been running for a few cycles."
            )
        if exchange.bankrupt:
            data.notes.append("THE ACCOUNT HAS BEEN WIPED OUT. Trading has stopped.")
        return data

    # -- data loading ------------------------------------------------------

    def _load_equity_curve(self) -> pd.DataFrame:
        if not self.store.has_data("equity"):
            return pd.DataFrame(columns=["ts_ms", "equity"])
        with MarketDatabase(self.store) as db:
            return db.query("SELECT * FROM equity ORDER BY ts_ms")

    @staticmethod
    def _pnl_since(curve: pd.DataFrame, since_ms: int, equity: float) -> float:
        if curve.empty:
            return 0.0
        earlier = curve.loc[curve["ts_ms"] <= since_ms]
        if earlier.empty:
            # No reading before the window: use the earliest we have and say
            # so, rather than pretending the window covered the whole run.
            baseline = float(curve["equity"].iloc[0])
        else:
            baseline = float(earlier["equity"].iloc[-1])
        return equity - baseline

    def _metrics_for(
        self, curve: pd.DataFrame, since_ms: int, exchange: PaperExchange, now_ms: int
    ) -> dict[str, Any]:
        window = curve.loc[curve["ts_ms"] >= since_ms].copy()
        if len(window) < 2:
            return {"rows": len(window)}

        trades = [
            t for t in exchange.closed_trades if t.closed_ts_ms >= since_ms
        ]
        interval_ms = int(np.median(np.diff(window["ts_ms"]))) if len(window) > 2 else 3_600_000
        metrics = compute_metrics(
            window,
            trades,
            [f for f in exchange.fills if f.ts_ms >= since_ms],
            interval_ms=max(interval_ms, 60_000),
            starting_equity=float(window["equity"].iloc[0]),
            liquidations=exchange.liquidation_count,
            bankrupt=exchange.bankrupt,
        )
        return metrics.to_dict()

    def _positions(
        self, exchange: PaperExchange, marks: dict[str, float], equity: float
    ) -> list[PositionLine]:
        lines = []
        for position in exchange.open_positions():
            price = marks.get(position.coin, position.entry_price)
            notional = position.notional(price)
            unrealized = position.unrealized_pnl(price)
            cost_basis = abs(position.size) * position.entry_price
            lines.append(PositionLine(
                coin=position.coin,
                side=position.direction,
                size_usd=notional,
                size_base=position.size,
                entry_price=position.entry_price,
                current_price=price,
                unrealized_pnl=unrealized,
                unrealized_pct=unrealized / cost_basis if cost_basis else 0.0,
                leverage=notional / equity if equity > 0 else 0.0,
                funding_paid=position.funding_paid,
            ))
        return lines

    def _trades(self, exchange: PaperExchange, since_ms: int) -> list[TradeLine]:
        lines = []
        for trade in exchange.closed_trades:
            if trade.closed_ts_ms < since_ms:
                continue
            lines.append(TradeLine(
                coin=trade.coin,
                direction=trade.direction,
                opened=f"{pd.Timestamp(trade.opened_ts_ms, unit='ms', tz='UTC'):%Y-%m-%d %H:%M}",
                closed=f"{pd.Timestamp(trade.closed_ts_ms, unit='ms', tz='UTC'):%Y-%m-%d %H:%M}",
                entry_price=trade.entry_price,
                exit_price=trade.exit_price,
                net_pnl=trade.net_pnl,
                fees=trade.fees,
                funding=trade.funding,
                liquidated=trade.liquidated,
                reason=trade.close_context.reason or trade.open_context.reason,
            ))
        return lines

    def _plans(
        self, latest: dict[str, Any], exchange: PaperExchange, marks: dict[str, float]
    ) -> list[PlanLine]:
        plans = []
        for coin, record in sorted(latest.items()):
            signal = record.signal
            position = exchange.position(coin)
            plan = PlanLine(
                coin=coin,
                probability=signal.probability,
                confidence=signal.confidence,
                direction=signal.direction,
                regime=signal.regime,
                target_notional=record.target_notional,
                current_notional=position.notional(marks.get(coin, position.entry_price)),
                action=record.action,
                risk_summary=record.risk.summary(),
                drivers=[(f.name, f.contribution) for f in signal.top_features],
            )
            plan.reasoning = market_reasoning(plan)
            plans.append(plan)
        return plans


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def learning_lines(learning: dict[str, Any]) -> list[str]:
    """The model paragraph, identical in the text and HTML reports.

    Written once because two renderings of the same facts drift, and the
    first thing to drift would be whichever one nobody reads.
    """
    lines: list[str] = []
    model = learning.get("model")
    if model:
        lines.append(
            f"Deciding: {model.get('backend', '?')} on {model.get('features', 0)} "
            f"features, asking {model.get('question', '?')}"
        )
        through = model.get("trained_through_ms")
        auc = model.get("val_auc")
        detail = f"Fitted on data through {pd.Timestamp(through, unit='ms', tz='UTC'):%Y-%m-%d}" \
            if through else "Training span unknown"
        if auc is not None:
            detail += f", walk-forward AUC {auc:.4f} (0.50 is a coin flip)"
        lines.append(detail)
    else:
        lines.append("No model artefact could be read.")

    shorting = learning.get("shorting") or {}
    if shorting:
        lines.append(
            f"Shorting: ON, asking {shorting.get('question', '?')}"
            if shorting.get("enabled")
            else "Shorting: OFF - long-only, so a falling market can only be sat out."
        )

    retrain = learning.get("retrain") or {}
    if not retrain.get("enabled"):
        lines.append("Retraining is OFF - this model is frozen at its first fit.")
    elif retrain.get("next_ms"):
        lines.append(
            f"Next refit on all stored history at "
            f"{pd.Timestamp(retrain['next_ms'], unit='ms', tz='UTC'):%Y-%m-%d %H:%M} UTC."
        )
    if retrain.get("last_outcome"):
        lines.append(f"Last refit: {retrain['last_outcome']}")

    for line in (learning.get("scorecard") or "").splitlines():
        if line.strip():
            lines.append(line.strip())
    return lines


def render_text(data: ReportData) -> str:
    """Plain-text fallback. Must contain everything the HTML does."""
    stamp = pd.Timestamp(data.generated_ms, unit="ms", tz="UTC")
    start = pd.Timestamp(data.window_start_ms, unit="ms", tz="UTC")
    lines = [
        "HYPERLIQUID PAPER TRADING - 6 HOUR REPORT",
        f"Generated {stamp:%Y-%m-%d %H:%M} UTC",
        f"Window {start:%Y-%m-%d %H:%M} -> {stamp:%H:%M} UTC",
        f"Risk profile: {data.profile}",
        "",
        f"Equity {_fmt_usd(data.equity)} vs {_fmt_usd(data.starting_capital)} start "
        f"({_fmt_pct(data.total_return)})",
        f"P&L all time {_fmt_signed(data.pnl_all_time)} | "
        f"today {_fmt_signed(data.pnl_today)} | window {_fmt_signed(data.pnl_window)}",
        "",
        "PERFORMANCE",
    ]
    for label, metrics in (("This window", data.window_metrics), ("All time", data.all_time_metrics)):
        if metrics.get("rows", 0) < 2:
            lines.append(f"  {label}: not enough history yet")
            continue
        lines.append(
            f"  {label}: return {_fmt_pct(metrics.get('total_return', float('nan')))} | "
            f"Sharpe {_fmt_num(metrics.get('sharpe', float('nan')))} | "
            f"Sortino {_fmt_num(metrics.get('sortino', float('nan')))} | "
            f"max DD {metrics.get('max_drawdown', 0):.2%} | "
            f"win rate {_fmt_pct(metrics.get('win_rate', float('nan')), 1)} | "
            f"profit factor {_fmt_num(metrics.get('profit_factor', float('nan')))}"
        )

    lines += ["", "OPEN POSITIONS"]
    if not data.positions:
        lines.append("  None.")
    for p in data.positions:
        lines.append(
            f"  {p.coin} {p.side.upper()} {_fmt_usd(p.size_usd)} @ {p.entry_price:,.2f} "
            f"-> {p.current_price:,.2f} | unrealised {_fmt_signed(p.unrealized_pnl)} "
            f"({_fmt_pct(p.unrealized_pct)}) | {p.leverage:.2f}x | "
            f"funding paid {_fmt_signed(-p.funding_paid)}"
        )

    lines += ["", "TRADES THIS WINDOW"]
    if not data.trades:
        lines.append("  No positions were closed this window.")
    for t in data.trades:
        flag = " [LIQUIDATED]" if t.liquidated else ""
        lines.append(
            f"  {t.coin} {t.direction} {t.opened} -> {t.closed} | "
            f"{t.entry_price:,.2f} -> {t.exit_price:,.2f} | "
            f"net {_fmt_signed(t.net_pnl)} (fees {_fmt_usd(t.fees)}, "
            f"funding {_fmt_signed(-t.funding)}){flag}"
        )

    lines += ["", "PLAN AND REASONING"]
    if not data.plans:
        lines.append("  No signals generated this window.")
    for plan in data.plans:
        lines += [
            f"  {plan.coin}: P(up)={plan.probability:.3f} confidence {plan.confidence:.2f} "
            f"leaning {plan.direction} | target {_fmt_usd(plan.target_notional)}",
            f"    drivers: " + (
                ", ".join(f"{n} {v:+.3f}" for n, v in plan.drivers[:5]) or "none available"
            ),
            f"    {plan.reasoning}",
        ]

    lines += ["", "RISK STATUS"]
    status = data.risk_status
    if status:
        dd_limit = status.get("drawdown_limit")
        dl_limit = status.get("daily_loss_limit")
        lines += [
            f"  drawdown {status.get('drawdown', 0):.2%} of "
            + (f"{dd_limit:.2%} limit" if dd_limit else "no limit (aggressive profile)"),
            f"  daily loss {status.get('daily_loss', 0):.2%} of "
            + (f"{dl_limit:.2%} limit" if dl_limit else "no limit (aggressive profile)"),
            f"  leverage {status.get('leverage', 0):.2f}x of {status.get('leverage_limit', 0):.2f}x",
            f"  positions {status.get('open_positions', 0)} of {status.get('max_open_positions', 0)}",
        ]
        if status.get("halted"):
            lines.append(f"  TRADING HALTED: {status['halted']}")
    lines.append(f"  risk vetoes this window: {len(data.vetoes)}")
    lines += [f"    {v}" for v in data.vetoes[:10]]

    if data.learning:
        lines += ["", "THE MODEL, AND HOW IT IS DOING"] + [
            f"  {line}" for line in learning_lines(data.learning)
        ]

    if data.notes:
        lines += ["", "NOTES"] + [f"  {n}" for n in data.notes]

    lines += ["", DISCLAIMER]
    return "\n".join(lines)


def render_html(data: ReportData) -> str:
    """HTML report. Inline styles only -- email clients strip stylesheets."""
    stamp = pd.Timestamp(data.generated_ms, unit="ms", tz="UTC")
    start = pd.Timestamp(data.window_start_ms, unit="ms", tz="UTC")
    positive = data.pnl_all_time >= 0
    accent = "#0b7a3b" if positive else "#a3162a"

    def esc(text: Any) -> str:
        return html.escape(str(text))

    def money_cell(value: float) -> str:
        colour = "#0b7a3b" if value >= 0 else "#a3162a"
        return f'<span style="color:{colour};font-weight:600">{_fmt_signed(value)}</span>'

    rows = []

    # 3. Positions
    if data.positions:
        rows.append(
            '<table role="presentation" width="100%" cellpadding="8" cellspacing="0" '
            'style="border-collapse:collapse;font-size:14px">'
            '<tr style="background:#f2f4f7;text-align:left">'
            "<th>Market</th><th>Side</th><th>Size</th><th>Entry</th><th>Now</th>"
            "<th>Unrealised</th><th>Leverage</th></tr>"
        )
        for p in data.positions:
            rows.append(
                f'<tr style="border-bottom:1px solid #e6e8eb">'
                f"<td><b>{esc(p.coin)}</b></td><td>{esc(p.side)}</td>"
                f"<td>{_fmt_usd(p.size_usd)}</td><td>{p.entry_price:,.2f}</td>"
                f"<td>{p.current_price:,.2f}</td><td>{money_cell(p.unrealized_pnl)} "
                f'<span style="color:#667085">({_fmt_pct(p.unrealized_pct)})</span></td>'
                f"<td>{p.leverage:.2f}x</td></tr>"
            )
        rows.append("</table>")
    else:
        rows.append('<p style="color:#667085">No open positions.</p>')
    positions_html = "".join(rows)

    trade_rows = []
    if data.trades:
        trade_rows.append(
            '<table role="presentation" width="100%" cellpadding="8" cellspacing="0" '
            'style="border-collapse:collapse;font-size:14px">'
            '<tr style="background:#f2f4f7;text-align:left">'
            "<th>Market</th><th>Direction</th><th>Entry &rarr; Exit</th>"
            "<th>Net P&amp;L</th><th>Fees</th><th>Funding</th></tr>"
        )
        for t in data.trades:
            flag = ' <span style="color:#a3162a;font-weight:700">LIQUIDATED</span>' if t.liquidated else ""
            trade_rows.append(
                f'<tr style="border-bottom:1px solid #e6e8eb">'
                f"<td><b>{esc(t.coin)}</b>{flag}</td><td>{esc(t.direction)}</td>"
                f"<td>{t.entry_price:,.2f} &rarr; {t.exit_price:,.2f}</td>"
                f"<td>{money_cell(t.net_pnl)}</td><td>{_fmt_usd(t.fees)}</td>"
                f"<td>{_fmt_signed(-t.funding)}</td></tr>"
            )
        trade_rows.append("</table>")
    else:
        trade_rows.append('<p style="color:#667085">No positions were closed this window.</p>')
    trades_html = "".join(trade_rows)

    plan_blocks = []
    for plan in data.plans:
        drivers = "".join(
            f'<li style="margin:2px 0">{esc(name)} '
            f'<span style="color:{"#0b7a3b" if value >= 0 else "#a3162a"}">{value:+.3f}</span></li>'
            for name, value in plan.drivers[:6]
        ) or '<li style="color:#667085">No attribution available.</li>'
        plan_blocks.append(
            f'<div style="border:1px solid #e6e8eb;border-radius:8px;padding:14px;margin-bottom:12px">'
            f'<div style="font-size:16px;font-weight:700">{esc(plan.coin)}'
            f'<span style="font-weight:400;color:#667085"> &mdash; leaning {esc(plan.direction)}</span></div>'
            f'<div style="margin:6px 0;font-size:14px">P(up) <b>{plan.probability:.3f}</b> '
            f"&middot; confidence <b>{plan.confidence:.2f}</b> &middot; regime <b>{esc(plan.regime)}</b> "
            f"&middot; target <b>{_fmt_usd(plan.target_notional)}</b></div>"
            f'<div style="font-size:13px;color:#344054;margin:8px 0">{esc(plan.reasoning)}</div>'
            f'<div style="font-size:13px"><b>Top contributing factors</b>'
            f'<ul style="margin:6px 0 0 18px;padding:0">{drivers}</ul></div>'
            f'<div style="font-size:12px;color:#667085;margin-top:8px">Risk: {esc(plan.risk_summary)}</div>'
            f"</div>"
        )
    plans_html = "".join(plan_blocks) or '<p style="color:#667085">No signals generated.</p>'

    status = data.risk_status
    def limit_row(label: str, used: float, limit: float | None) -> str:
        if limit is None:
            return (
                f'<tr><td style="padding:6px 8px">{label}</td>'
                f'<td style="padding:6px 8px">{used:.2%}</td>'
                f'<td style="padding:6px 8px;color:#667085">no limit (aggressive profile)</td></tr>'
            )
        share = min(1.0, used / limit) if limit else 0.0
        colour = "#a3162a" if share > 0.8 else ("#b54708" if share > 0.5 else "#0b7a3b")
        return (
            f'<tr><td style="padding:6px 8px">{label}</td>'
            f'<td style="padding:6px 8px;color:{colour};font-weight:600">{used:.2%}</td>'
            f'<td style="padding:6px 8px">of {limit:.2%} ({share:.0%} used)</td></tr>'
        )

    risk_html = (
        '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="font-size:14px">'
        + limit_row("Drawdown", status.get("drawdown", 0.0), status.get("drawdown_limit"))
        + limit_row("Daily loss", status.get("daily_loss", 0.0), status.get("daily_loss_limit"))
        + f'<tr><td style="padding:6px 8px">Leverage</td>'
          f'<td style="padding:6px 8px;font-weight:600">{status.get("leverage", 0):.2f}x</td>'
          f'<td style="padding:6px 8px">of {status.get("leverage_limit", 0):.2f}x</td></tr>'
        + f'<tr><td style="padding:6px 8px">Open positions</td>'
          f'<td style="padding:6px 8px;font-weight:600">{status.get("open_positions", 0)}</td>'
          f'<td style="padding:6px 8px">of {status.get("max_open_positions", 0)}</td></tr>'
        + "</table>"
    )
    if status.get("halted"):
        risk_html += (
            f'<p style="background:#fef3f2;border-left:4px solid #a3162a;padding:10px;'
            f'margin-top:10px;font-size:14px"><b>Trading halted:</b> {esc(status["halted"])}</p>'
        )
    veto_html = (
        '<ul style="margin:8px 0 0 18px;padding:0;font-size:13px;color:#344054">'
        + "".join(f"<li>{esc(v)}</li>" for v in data.vetoes[:10])
        + "</ul>"
    ) if data.vetoes else '<p style="color:#667085;font-size:13px">No risk vetoes this window.</p>'

    def metric_block(label: str, metrics: dict) -> str:
        if metrics.get("rows", 0) < 2:
            return f'<p style="color:#667085">{label}: not enough history yet.</p>'
        return (
            f'<p style="margin:4px 0;font-size:14px"><b>{label}</b> &mdash; '
            f'return {_fmt_pct(metrics.get("total_return", float("nan")))} &middot; '
            f'Sharpe {_fmt_num(metrics.get("sharpe", float("nan")))} &middot; '
            f'Sortino {_fmt_num(metrics.get("sortino", float("nan")))} &middot; '
            f'max DD {metrics.get("max_drawdown", 0):.2%} &middot; '
            f'win rate {_fmt_pct(metrics.get("win_rate", float("nan")), 1)} &middot; '
            f'profit factor {_fmt_num(metrics.get("profit_factor", float("nan")))}</p>'
        )

    notes_html = (
        '<div style="background:#fffaeb;border-left:4px solid #b54708;padding:10px;margin:16px 0">'
        + "".join(f'<p style="margin:4px 0;font-size:13px">{esc(n)}</p>' for n in data.notes)
        + "</div>"
    ) if data.notes else ""

    learning_html = (
        '<h2 style="font-size:16px;margin:24px 0 8px">The model, and how it is doing</h2>'
        '<div style="border:1px solid #e6e8eb;border-radius:8px;padding:12px;'
        'background:#fafbfc;font-size:13px;color:#344054">'
        + "".join(f"<div>{esc(line)}</div>" for line in learning_lines(data.learning))
        + "</div>"
    ) if data.learning else ""

    return f"""<div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;
     max-width:760px;margin:0 auto;padding:24px;color:#101828;background:#ffffff">
  <div style="border-bottom:3px solid {accent};padding-bottom:16px;margin-bottom:20px">
    <h1 style="margin:0;font-size:22px">Hyperliquid Paper Trading &mdash; 6 Hour Report</h1>
    <p style="margin:6px 0 0;color:#667085;font-size:13px">
      {start:%Y-%m-%d %H:%M} &rarr; {stamp:%Y-%m-%d %H:%M} UTC &middot;
      risk profile <b>{esc(data.profile)}</b>
    </p>
    <p style="margin:14px 0 0;font-size:28px;font-weight:700">{_fmt_usd(data.equity)}
      <span style="font-size:15px;font-weight:400;color:#667085">
        vs {_fmt_usd(data.starting_capital)} start</span></p>
    <p style="margin:4px 0 0;font-size:16px;color:{accent};font-weight:600">
      {_fmt_signed(data.pnl_all_time)} ({_fmt_pct(data.total_return)}) all time</p>
    <p style="margin:4px 0 0;font-size:14px;color:#344054">
      This window {_fmt_signed(data.pnl_window)} &middot; today {_fmt_signed(data.pnl_today)}
      &middot; cash {_fmt_usd(data.cash)} &middot; unrealised {_fmt_signed(data.unrealized)}</p>
  </div>
  {notes_html}
  <h2 style="font-size:16px;margin:20px 0 8px">Performance</h2>
  {metric_block("This window", data.window_metrics)}
  {metric_block("All time", data.all_time_metrics)}

  <h2 style="font-size:16px;margin:24px 0 8px">Current positions</h2>
  {positions_html}

  <h2 style="font-size:16px;margin:24px 0 8px">What it did this window</h2>
  {trades_html}

  <h2 style="font-size:16px;margin:24px 0 8px">How it plans to invest next</h2>
  {plans_html}

  <h2 style="font-size:16px;margin:24px 0 8px">Risk status</h2>
  {risk_html}
  {veto_html}
  {learning_html}

  <p style="margin-top:28px;padding-top:16px;border-top:1px solid #e6e8eb;
     color:#667085;font-size:12px">{DISCLAIMER}</p>
</div>"""
