"""Phase 5: the risk engine must be able to overrule anything."""

from __future__ import annotations

import numpy as np
import pytest

from config.settings import ExecutionConfig, resolve_risk_profile
from risk.risk_engine import AccountState, RiskEngine, Verdict

DAY = 86_400_000
EXEC = ExecutionConfig()


def engine(profile: str = "conservative", **kwargs) -> RiskEngine:
    return RiskEngine(resolve_risk_profile(profile), EXEC, **kwargs)


def state(engine_: RiskEngine, *, equity=100_000.0, gross=0.0, positions=0, ts=0, existing=None):
    engine_.observe_equity(ts, equity)
    return engine_.account_state(
        ts_ms=ts, equity=equity, gross_notional=gross,
        open_positions=positions, existing_notional=existing or {},
    )


def evaluate(engine_, st, *, confidence=1.0, atr=0.01, new=True):
    return engine_.evaluate(
        coin="BTC", state=st, confidence=confidence, atr_fraction=atr, is_new_position=new
    )


# ==========================================================================
# Sizing
# ==========================================================================

def test_position_size_follows_the_risk_budget_and_stop_distance() -> None:
    """notional = equity * risk_per_trade / stop_distance."""
    risk = engine("conservative")
    notional, stop = risk.target_notional(equity=100_000, confidence=1.0, atr_fraction=0.01)
    assert stop == pytest.approx(0.02)              # 2 x ATR
    assert notional == pytest.approx(100_000 * 0.0025 / 0.02)


def test_a_wider_stop_produces_a_smaller_position() -> None:
    risk = engine("conservative")
    tight, _ = risk.target_notional(equity=100_000, confidence=1.0, atr_fraction=0.005)
    wide, _ = risk.target_notional(equity=100_000, confidence=1.0, atr_fraction=0.05)
    assert wide < tight


def test_low_confidence_scales_the_position_down() -> None:
    risk = engine("conservative")
    strong, _ = risk.target_notional(equity=100_000, confidence=1.0, atr_fraction=0.01)
    weak, _ = risk.target_notional(equity=100_000, confidence=0.2, atr_fraction=0.01)
    assert weak == pytest.approx(strong * 0.2)


def test_a_missing_volatility_estimate_sizes_down_not_up() -> None:
    risk = engine("conservative")
    known, _ = risk.target_notional(equity=100_000, confidence=1.0, atr_fraction=0.005)
    unknown, _ = risk.target_notional(equity=100_000, confidence=1.0, atr_fraction=None)
    assert unknown < known


def test_an_absurdly_quiet_bar_cannot_demand_an_enormous_position() -> None:
    """A near-zero stop would otherwise divide the budget by ~nothing."""
    risk = engine("conservative")
    _, stop = risk.target_notional(equity=100_000, confidence=1.0, atr_fraction=1e-9)
    assert stop >= risk.min_stop_fraction


# ==========================================================================
# The limits
# ==========================================================================

def test_position_notional_is_capped() -> None:
    risk = engine("conservative")
    decision = evaluate(risk, state(risk), atr=0.0005)
    assert decision.verdict is Verdict.RESIZED
    assert decision.approved_notional <= risk.limits.max_position_usd


def test_leverage_headroom_binds_the_size() -> None:
    risk = engine("aggressive")
    # Already at 9.5x of a 10x limit: only 0.5x of equity remains.
    st = state(risk, gross=950_000, positions=2, existing={"ETH": 500_000, "SOL": 450_000})
    decision = evaluate(risk, st, atr=0.001)
    assert decision.approved_notional <= 50_000 + 1e-6


def test_a_fully_levered_account_is_rejected_outright() -> None:
    risk = engine("aggressive")
    st = state(risk, gross=1_000_000, positions=2, existing={"ETH": 600_000, "SOL": 400_000})
    decision = evaluate(risk, st, atr=0.01)
    assert decision.verdict is Verdict.REJECTED
    assert any(c.name == "max_leverage" for c in decision.vetoes)


def test_the_position_count_limit_blocks_a_fourth_market() -> None:
    risk = engine("conservative")
    st = state(risk, positions=3, existing={"ETH": 1000, "SOL": 1000, "DOGE": 1000})
    decision = evaluate(risk, st)
    assert decision.verdict is Verdict.REJECTED
    assert any(c.name == "max_open_positions" for c in decision.vetoes)


def test_an_existing_position_may_still_be_adjusted_at_the_count_limit() -> None:
    risk = engine("conservative")
    st = state(risk, positions=3, gross=3000, existing={"BTC": 1000, "ETH": 1000, "SOL": 1000})
    assert evaluate(risk, st, new=False).approved


# ==========================================================================
# Halts
# ==========================================================================

def test_the_daily_loss_halt_stops_new_trades() -> None:
    risk = engine("conservative")
    risk.observe_equity(0, 100_000)
    st = risk.account_state(ts_ms=3_600_000, equity=97_000, gross_notional=0, open_positions=0)
    decision = evaluate(risk, st)
    assert decision.verdict is Verdict.REJECTED
    assert any(c.name == "max_daily_loss" for c in decision.vetoes)
    assert "daily loss" in risk.halted_reason


def test_the_daily_halt_lifts_on_the_next_utc_day() -> None:
    risk = engine("conservative")
    risk.observe_equity(0, 100_000)
    evaluate(risk, risk.account_state(ts_ms=3_600_000, equity=97_000, gross_notional=0, open_positions=0))
    assert risk.halted_reason is not None
    st = state(risk, equity=97_000, ts=DAY)
    assert risk.halted_reason is None
    assert evaluate(risk, st).approved


def test_the_drawdown_halt_stops_trading_entirely() -> None:
    risk = engine("conservative")
    risk.observe_equity(0, 100_000)
    risk.observe_equity(DAY, 120_000)
    st = risk.account_state(ts_ms=2 * DAY, equity=100_000, gross_notional=0, open_positions=0)
    decision = evaluate(risk, st)
    assert decision.verdict is Verdict.REJECTED
    assert any(c.name == "max_drawdown" for c in decision.vetoes)


def test_a_zero_equity_account_can_do_nothing() -> None:
    risk = engine("aggressive")
    st = state(risk, equity=0.0)
    decision = evaluate(risk, st)
    assert decision.verdict is Verdict.REJECTED
    assert any(c.name == "solvency" for c in decision.vetoes)


# ==========================================================================
# The aggressive profile
# ==========================================================================

def test_the_aggressive_profile_disables_the_halts_but_keeps_the_checks() -> None:
    """Limits set to 1.0 switch a halt off; they do not remove it from the flow."""
    risk = engine("aggressive")
    risk.observe_equity(0, 100_000)
    risk.observe_equity(DAY, 200_000)
    st = risk.account_state(ts_ms=2 * DAY, equity=60_000, gross_notional=0, open_positions=0)
    decision = evaluate(risk, st, atr=0.02)

    assert decision.approved, "a 70% drawdown is survivable only because the halt is off"
    names = {c.name for c in decision.checks}
    assert {"max_drawdown", "max_daily_loss", "max_leverage", "liquidation_buffer"} <= names
    assert all(c.passed for c in decision.checks)


def test_the_aggressive_profile_still_enforces_leverage_and_liquidation() -> None:
    risk = engine("aggressive")
    st = state(risk)
    decision = evaluate(risk, st, atr=0.0001)
    assert decision.approved_notional <= risk.limits.max_leverage * st.equity + 1e-6
    assert decision.approved_notional <= risk.limits.max_position_usd + 1e-6


def test_full_risk_per_trade_is_bounded_by_the_other_limits() -> None:
    """risk_per_trade = 1.0 asks for an unbounded position; the caps hold."""
    risk = engine("aggressive")
    requested, _ = risk.target_notional(equity=100_000, confidence=1.0, atr_fraction=0.002)
    assert requested > 1_000_000                      # what sizing alone wants
    decision = evaluate(risk, state(risk), atr=0.002)
    assert decision.verdict is Verdict.RESIZED
    assert decision.approved_notional <= 100_000


# ==========================================================================
# The liquidation buffer
# ==========================================================================

def test_a_position_whose_liquidation_sits_inside_the_stop_is_shrunk() -> None:
    """At high leverage the stop would never be reached -- liquidation comes first."""
    risk = engine("aggressive", liquidation_buffer_multiple=1.5)
    decision = evaluate(risk, state(risk), confidence=1.0, atr=0.03)
    implied_leverage = decision.approved_notional / 100_000
    maintenance = EXEC.maintenance_margin_fraction(None)
    liquidation_move = (1 / implied_leverage - maintenance) / (1 - maintenance)
    assert liquidation_move >= 0.06 * 1.5 - 1e-6


def test_the_buffer_check_is_always_reported() -> None:
    risk = engine("aggressive")
    decision = evaluate(risk, state(risk), atr=0.01)
    buffer_check = next(c for c in decision.checks if c.name == "liquidation_buffer")
    assert "liquidation at" in buffer_check.detail


# ==========================================================================
# Auditability
# ==========================================================================

def test_every_decision_carries_its_full_reasoning() -> None:
    risk = engine("conservative")
    decision = evaluate(risk, state(risk))
    assert len(decision.checks) >= 5
    for check in decision.checks:
        assert check.detail
        assert check.describe().startswith(("[ok]", "[VETO]"))


def test_vetoes_are_logged_for_the_report() -> None:
    risk = engine("conservative")
    st = state(risk, positions=3, existing={"ETH": 1, "SOL": 1, "DOGE": 1})
    evaluate(risk, st)
    assert risk.veto_log
    assert risk.veto_log[-1][1] == "BTC"


def test_status_reports_utilisation_against_each_limit() -> None:
    risk = engine("conservative")
    status = risk.status(state(risk, gross=50_000, positions=2))
    assert status["profile"] == "conservative"
    assert status["leverage"] == pytest.approx(0.5)
    assert status["leverage_limit"] == 2.0
    assert status["max_open_positions"] == 3


def test_disabled_halts_report_as_none_not_as_a_number() -> None:
    status = engine("aggressive").status(state(engine("aggressive")))
    assert status["daily_loss_limit"] is None
    assert status["drawdown_limit"] is None
