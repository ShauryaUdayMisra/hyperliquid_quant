"""The risk engine. It has the last word.

A 99%-confidence model signal that breaches a limit is rejected. The engine
never trusts the model, never trusts the strategy, and records the reason
for every decision so a rejected trade is as auditable as an accepted one.

Checks, in the order they are applied:

1. **Drawdown halt** -- peak-to-trough equity beyond the limit stops trading.
2. **Daily loss halt** -- loss since 00:00 UTC beyond the limit stops trading
   until the next UTC day.
3. **Position count** -- no more than N concurrent positions.
4. **Position sizing** -- risk-per-trade budget divided by stop distance.
5. **Notional cap** -- no single position above the USD limit.
6. **Leverage cap** -- account gross notional over equity.
7. **Liquidation buffer** -- refuse an entry whose liquidation price sits
   inside the market's ordinary noise.

Both configured profiles run this same code. The ``aggressive`` profile
sets several limits to 1.0, which disables those halts by construction --
it does not remove them from the pipeline, and the liquidation buffer and
leverage cap still bind.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

import numpy as np

from config.settings import SETTINGS, ExecutionConfig, RiskLimits

log = logging.getLogger(__name__)

DAY_MS = 86_400_000


class Verdict(str, Enum):
    APPROVED = "approved"
    RESIZED = "resized"
    REJECTED = "rejected"


@dataclass
class RiskCheck:
    name: str
    passed: bool
    detail: str
    limit: float | None = None
    observed: float | None = None

    def describe(self) -> str:
        mark = "ok" if self.passed else "VETO"
        return f"[{mark}] {self.name}: {self.detail}"


@dataclass
class RiskDecision:
    """The engine's answer, with the full reasoning attached."""

    verdict: Verdict
    approved_notional: float
    requested_notional: float
    checks: list[RiskCheck] = field(default_factory=list)
    binding_constraint: str | None = None

    @property
    def approved(self) -> bool:
        return self.verdict is not Verdict.REJECTED and self.approved_notional > 0

    @property
    def vetoes(self) -> list[RiskCheck]:
        return [c for c in self.checks if not c.passed]

    def summary(self) -> str:
        if self.verdict is Verdict.REJECTED:
            reasons = "; ".join(c.detail for c in self.vetoes) or "rejected"
            return f"REJECTED: {reasons}"
        if self.verdict is Verdict.RESIZED:
            return (
                f"resized ${self.requested_notional:,.0f} -> "
                f"${self.approved_notional:,.0f} ({self.binding_constraint})"
            )
        return f"approved ${self.approved_notional:,.0f}"


@dataclass
class AccountState:
    """What the engine needs to know about the account right now."""

    equity: float
    gross_notional: float
    open_positions: int
    peak_equity: float
    day_start_equity: float
    ts_ms: int
    existing_notional: Mapping[str, float] = field(default_factory=dict)

    @property
    def drawdown(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, 1.0 - self.equity / self.peak_equity)

    @property
    def daily_loss(self) -> float:
        if self.day_start_equity <= 0:
            return 0.0
        return max(0.0, 1.0 - self.equity / self.day_start_equity)

    @property
    def leverage(self) -> float:
        return 0.0 if self.equity <= 0 else self.gross_notional / self.equity


class RiskEngine:
    def __init__(
        self,
        limits: RiskLimits | None = None,
        execution: ExecutionConfig | None = None,
        *,
        stop_atr_multiple: float = 2.0,
        min_stop_fraction: float = 0.005,
        liquidation_buffer_multiple: float = 1.5,
    ) -> None:
        self.limits = limits or SETTINGS.risk
        self.execution = execution or SETTINGS.execution
        #: Stop distance = this many ATRs. Wider stops mean smaller size for
        #: the same risk budget.
        self.stop_atr_multiple = stop_atr_multiple
        #: Floor on the stop distance, so a freakishly quiet hour cannot
        #: divide the risk budget by ~zero and demand an enormous position.
        self.min_stop_fraction = min_stop_fraction
        #: Refuse entries whose liquidation price is closer than this many
        #: stop distances away.
        self.liquidation_buffer_multiple = liquidation_buffer_multiple

        self._peak_equity: float | None = None
        self._day_start_equity: float | None = None
        self._current_day: int | None = None
        self.halted_reason: str | None = None
        self.veto_log: list[tuple[int, str, str]] = []

    # -- account tracking --------------------------------------------------

    def observe_equity(self, ts_ms: int, equity: float) -> None:
        """Update the running peak and the UTC-day baseline.

        Must be called every bar. The daily-loss limit is meaningless if the
        day's opening equity is never recorded.
        """
        day = ts_ms // DAY_MS
        if self._current_day is None or day != self._current_day:
            self._current_day = day
            self._day_start_equity = equity
            if self.halted_reason and "daily loss" in self.halted_reason:
                log.info("new UTC day: daily-loss halt lifted")
                self.halted_reason = None
        self._peak_equity = equity if self._peak_equity is None else max(self._peak_equity, equity)

    def account_state(
        self,
        *,
        ts_ms: int,
        equity: float,
        gross_notional: float,
        open_positions: int,
        existing_notional: Mapping[str, float] | None = None,
    ) -> AccountState:
        return AccountState(
            equity=equity,
            gross_notional=gross_notional,
            open_positions=open_positions,
            peak_equity=self._peak_equity if self._peak_equity is not None else equity,
            day_start_equity=(
                self._day_start_equity if self._day_start_equity is not None else equity
            ),
            ts_ms=ts_ms,
            existing_notional=existing_notional or {},
        )

    # -- sizing ------------------------------------------------------------

    def stop_distance_fraction(self, atr_fraction: float | None) -> float:
        """How far the stop sits from entry, as a fraction of price."""
        if atr_fraction is None or not np.isfinite(atr_fraction) or atr_fraction <= 0:
            # No volatility estimate: assume a wide stop, which sizes down.
            return max(self.min_stop_fraction, 0.02)
        return max(self.min_stop_fraction, atr_fraction * self.stop_atr_multiple)

    def target_notional(
        self,
        *,
        equity: float,
        confidence: float,
        atr_fraction: float | None,
    ) -> tuple[float, float]:
        """Position size from the risk budget. Returns (notional, stop_fraction).

        ``risk_per_trade`` is the share of equity lost if the stop is hit, so
        ``notional = equity * risk_per_trade / stop_distance``. Confidence
        scales it linearly: a marginal signal gets a small position.
        """
        stop_fraction = self.stop_distance_fraction(atr_fraction)
        budget = equity * self.limits.risk_per_trade * max(0.0, min(1.0, confidence))
        return budget / stop_fraction, stop_fraction

    # -- the gate ----------------------------------------------------------

    def evaluate(
        self,
        *,
        coin: str,
        state: AccountState,
        confidence: float,
        atr_fraction: float | None,
        is_new_position: bool = True,
        max_asset_leverage: float | None = None,
        bar_notional: float | None = None,
    ) -> RiskDecision:
        """Size a proposed trade and apply every limit. The answer is final."""
        checks: list[RiskCheck] = []
        requested, stop_fraction = self.target_notional(
            equity=state.equity, confidence=confidence, atr_fraction=atr_fraction
        )

        # -- hard halts --
        if state.equity <= 0:
            checks.append(RiskCheck("solvency", False, "account equity is zero or negative"))
            return self._reject(coin, state, requested, checks)

        if self.limits.drawdown_cap_active:
            breached = state.drawdown >= self.limits.max_portfolio_dd
            checks.append(RiskCheck(
                "max_drawdown", not breached,
                f"drawdown {state.drawdown:.2%} vs limit {self.limits.max_portfolio_dd:.2%}",
                self.limits.max_portfolio_dd, state.drawdown,
            ))
            if breached:
                self.halted_reason = f"max drawdown {state.drawdown:.2%} reached"
                return self._reject(coin, state, requested, checks)
        else:
            checks.append(RiskCheck(
                "max_drawdown", True, "no drawdown halt in this profile", 1.0, state.drawdown
            ))

        if self.limits.daily_loss_cap_active:
            breached = state.daily_loss >= self.limits.max_daily_loss
            checks.append(RiskCheck(
                "max_daily_loss", not breached,
                f"daily loss {state.daily_loss:.2%} vs limit {self.limits.max_daily_loss:.2%}",
                self.limits.max_daily_loss, state.daily_loss,
            ))
            if breached:
                self.halted_reason = f"daily loss {state.daily_loss:.2%} reached"
                return self._reject(coin, state, requested, checks)
        else:
            checks.append(RiskCheck(
                "max_daily_loss", True, "no daily-loss halt in this profile", 1.0, state.daily_loss
            ))

        if is_new_position and coin not in state.existing_notional:
            over = state.open_positions >= self.limits.max_open_positions
            checks.append(RiskCheck(
                "max_open_positions", not over,
                f"{state.open_positions} open vs limit {self.limits.max_open_positions}",
                float(self.limits.max_open_positions), float(state.open_positions),
            ))
            if over:
                return self._reject(coin, state, requested, checks)

        # -- sizing constraints: bind rather than reject --
        approved = requested
        binding: str | None = None

        # What this particular market can absorb without the cost of getting
        # in and out swamping the move being predicted. A flat dollar cap is
        # a different trade in every market: the same $10,000 is 0.01% of
        # BTC's hourly volume and 5.6% of a small-cap's, which is 9bp of
        # impact against 237bp. Sized here rather than in the strategy so
        # that, as with every other limit, the risk engine has the final say
        # and the reason is recorded in the decision.
        if bar_notional is not None and bar_notional > 0:
            liquidity_cap = self.execution.max_notional_for_impact(bar_notional)
            if approved > liquidity_cap:
                approved = liquidity_cap
                binding = "liquidity"
            checks.append(RiskCheck(
                "liquidity", True,
                f"${liquidity_cap:,.0f} keeps expected impact within "
                f"{self.execution.max_impact_bps:.0f}bp at "
                f"${bar_notional:,.0f}/bar of volume",
                liquidity_cap, requested,
            ))

        if approved > self.limits.max_position_usd:
            approved = min(approved, self.limits.max_position_usd)
            binding = "max_position_usd"
        checks.append(RiskCheck(
            "max_position_usd", True,
            f"${requested:,.0f} capped at ${self.limits.max_position_usd:,.0f}"
            if binding else f"${requested:,.0f} within ${self.limits.max_position_usd:,.0f}",
            self.limits.max_position_usd, requested,
        ))

        # Leverage headroom: what the account can still take on.
        current_for_coin = state.existing_notional.get(coin, 0.0)
        other_notional = state.gross_notional - current_for_coin
        headroom = self.limits.max_leverage * state.equity - other_notional
        if headroom <= 0:
            checks.append(RiskCheck(
                "max_leverage", False,
                f"already at {state.leverage:.2f}x vs limit {self.limits.max_leverage:.2f}x",
                self.limits.max_leverage, state.leverage,
            ))
            return self._reject(coin, state, requested, checks)
        if approved > headroom:
            approved = headroom
            binding = "max_leverage"
        checks.append(RiskCheck(
            "max_leverage", True,
            f"headroom ${headroom:,.0f} at {self.limits.max_leverage:.2f}x "
            f"(currently {state.leverage:.2f}x)",
            self.limits.max_leverage, state.leverage,
        ))

        # -- liquidation buffer --
        maintenance = self.execution.maintenance_margin_fraction(max_asset_leverage)
        implied_leverage = approved / state.equity if state.equity > 0 else float("inf")
        if implied_leverage > 0:
            # Adverse move that wipes margin down to the maintenance level.
            liquidation_move = (1.0 / implied_leverage - maintenance) / (1.0 - maintenance)
        else:
            liquidation_move = float("inf")
        safe = liquidation_move >= stop_fraction * self.liquidation_buffer_multiple
        checks.append(RiskCheck(
            "liquidation_buffer", safe,
            f"liquidation at {liquidation_move:.2%} adverse vs stop {stop_fraction:.2%} "
            f"(needs {self.liquidation_buffer_multiple:.1f}x clearance)",
            stop_fraction * self.liquidation_buffer_multiple, liquidation_move,
        ))
        if not safe:
            # Shrink to a size whose liquidation sits outside the stop, rather
            # than refusing outright -- a smaller position is still tradeable.
            required_move = stop_fraction * self.liquidation_buffer_multiple
            safe_leverage = 1.0 / (required_move * (1.0 - maintenance) + maintenance)
            safe_notional = max(0.0, safe_leverage * state.equity)
            if safe_notional < approved:
                approved = safe_notional
                binding = "liquidation_buffer"

        if approved <= 0:
            return self._reject(coin, state, requested, checks)

        verdict = Verdict.RESIZED if approved < requested - 1e-9 else Verdict.APPROVED
        decision = RiskDecision(verdict, approved, requested, checks, binding)
        log.debug("risk %s %s: %s", coin, verdict.value, decision.summary())
        return decision

    def _reject(self, coin, state, requested, checks) -> RiskDecision:
        decision = RiskDecision(Verdict.REJECTED, 0.0, requested, checks)
        for veto in decision.vetoes:
            self.veto_log.append((state.ts_ms, coin, veto.detail))
        log.info("risk VETO %s: %s", coin, decision.summary())
        return decision

    # -- reporting ---------------------------------------------------------

    def status(self, state: AccountState) -> dict[str, object]:
        """Current risk utilisation, for the dashboard and the email report."""
        return {
            "profile": self.limits.name,
            "equity": state.equity,
            "drawdown": state.drawdown,
            "drawdown_limit": self.limits.max_portfolio_dd if self.limits.drawdown_cap_active else None,
            "daily_loss": state.daily_loss,
            "daily_loss_limit": self.limits.max_daily_loss if self.limits.daily_loss_cap_active else None,
            "leverage": state.leverage,
            "leverage_limit": self.limits.max_leverage,
            "open_positions": state.open_positions,
            "max_open_positions": self.limits.max_open_positions,
            "halted": self.halted_reason,
            "vetoes_logged": len(self.veto_log),
        }
