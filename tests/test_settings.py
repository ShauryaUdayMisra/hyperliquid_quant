"""The safety guard is the single most important test in Phase 1."""

from __future__ import annotations

import pytest

from config.settings import (
    FORBIDDEN_ENV_VARS,
    RISK_PROFILES,
    INTERVAL_MS,
    SETTINGS,
    HyperliquidConfig,
    TradingCredentialsPresent,
    assert_no_trading_credentials,
    resolve_risk_profile,
)


@pytest.mark.parametrize("var", FORBIDDEN_ENV_VARS)
def test_signing_credentials_abort_startup(var: str) -> None:
    with pytest.raises(TradingCredentialsPresent, match="paper-trading only"):
        assert_no_trading_credentials({var: "0xdeadbeef"})


def test_blank_credential_is_not_a_credential() -> None:
    assert_no_trading_credentials({FORBIDDEN_ENV_VARS[0]: "   "})


def test_current_process_holds_no_signing_keys() -> None:
    assert_no_trading_credentials()


def test_paper_trading_flag_is_on() -> None:
    assert SETTINGS.paper_trading_only is True


def test_conservative_profile_matches_the_brief() -> None:
    risk = resolve_risk_profile("conservative")
    assert risk.starting_capital == 100_000
    assert risk.max_position_usd == 10_000
    assert risk.max_leverage == 2.0
    assert risk.risk_per_trade == 0.0025
    assert risk.max_daily_loss == 0.02
    assert risk.max_portfolio_dd == 0.10
    assert risk.max_open_positions == 3
    assert risk.daily_loss_cap_active and risk.drawdown_cap_active


def test_aggressive_profile_matches_the_requested_settings() -> None:
    risk = resolve_risk_profile("aggressive")
    assert risk.max_leverage == 10.0
    assert risk.risk_per_trade == 1.0
    assert risk.max_position_usd == 100_000
    assert risk.max_daily_loss == 1.0
    assert risk.max_portfolio_dd == 1.0
    assert risk.max_open_positions == 3
    # The circuit breakers are off, so liquidation is the only backstop left.
    assert not risk.daily_loss_cap_active
    assert not risk.drawdown_cap_active


def test_both_profiles_start_from_the_same_capital() -> None:
    assert {p.starting_capital for p in RISK_PROFILES.values()} == {100_000.0}


def test_unknown_profile_is_rejected_loudly() -> None:
    with pytest.raises(ValueError, match="unknown RISK_PROFILE"):
        resolve_risk_profile("yolo")


def test_maintenance_margin_follows_hyperliquid_half_initial_rule() -> None:
    execution = SETTINGS.execution
    # Hyperliquid: maintenance margin = half the initial margin at max leverage.
    assert execution.maintenance_margin_fraction(40.0) == pytest.approx(0.0125)
    assert execution.maintenance_margin_fraction(20.0) == pytest.approx(0.025)


def test_configured_intervals_are_all_known() -> None:
    for interval in SETTINGS.data.candle_intervals:
        assert interval in INTERVAL_MS


def test_verify_defaults_to_system_trust_store() -> None:
    assert HyperliquidConfig(ca_bundle=None).verify() is True
    assert HyperliquidConfig(ca_bundle="/tmp/ca.pem").verify() == "/tmp/ca.pem"
