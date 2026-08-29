"""Phase 7: the 6-hour report must contain every section, grounded in numbers."""

from __future__ import annotations

import re

import pandas as pd
import pytest

from conftest import BASE_MS, frictionless_config
from config.settings import ExecutionConfig, Paths, ReportConfig, resolve_risk_profile
from execution.paper_exchange import DecisionContext, Order, PaperExchange
from execution.simulator import FillSimulator, MarketSnapshot, Side
from models.predict import FeatureContribution, Signal
from reporting.emailer import Emailer, subject_line
from reporting.report_builder import (
    DISCLAIMER,
    PlanLine,
    ReportBuilder,
    market_reasoning,
    render_html,
    render_text,
)
from risk.risk_engine import RiskDecision, RiskEngine, Verdict
from strategy.signals import DecisionRecord

HOUR = 3_600_000


def snap(price: float) -> MarketSnapshot:
    return MarketSnapshot(BASE_MS, "BTC", price, price, price, price, 1e9, HOUR)


@pytest.fixture
def account():
    config = frictionless_config()
    exchange = PaperExchange(100_000.0, config=config, simulator=FillSimulator(config))
    exchange.submit(
        Order("BTC", Side.BUY, 0.5, context=DecisionContext(reason="test entry")),
        snap(100_000.0),
    )
    exchange.submit(
        Order("ETH", Side.BUY, 2.0, context=DecisionContext(reason="test entry")),
        MarketSnapshot(BASE_MS, "ETH", 3_000, 3_000, 3_000, 3_000, 1e9, HOUR),
    )
    exchange.submit(
        Order("ETH", Side.SELL, 2.0, context=DecisionContext(reason="test exit")),
        MarketSnapshot(BASE_MS + HOUR, "ETH", 3_300, 3_300, 3_300, 3_300, 1e9, HOUR),
    )
    return exchange


@pytest.fixture
def decisions():
    signal = Signal(
        "BTC", BASE_MS, probability=0.62, base_rate=0.48, direction="long",
        regime="trending_up",
        top_features=[
            FeatureContribution("mom_ret_24_z", 1.4, 0.31, "shap"),
            FeatureContribution("fund_rate_z_24", -0.8, -0.12, "shap"),
        ],
        label_question="P(return_4bar > +0.30%)",
    )
    record = DecisionRecord(
        ts_ms=BASE_MS, coin="BTC", signal=signal,
        risk=RiskDecision(Verdict.RESIZED, 8_000.0, 12_000.0, [], "max_position_usd"),
        target_notional=8_000.0, current_notional=50_000.0, action="long",
    )
    return {"BTC": record}


@pytest.fixture
def report(account, decisions, tmp_path):
    paths = Paths(
        root=tmp_path, storage=tmp_path / "storage",
        parquet=tmp_path / "storage" / "parquet",
        duckdb_file=tmp_path / "storage" / "db" / "m.duckdb",
        logs=tmp_path / "logs", models=tmp_path / "models",
    )
    from data.database import ParquetStore

    builder = ReportBuilder(store=ParquetStore(paths))
    risk = RiskEngine(resolve_risk_profile("conservative"), ExecutionConfig())
    risk.observe_equity(BASE_MS, 100_000.0)
    return builder.build(
        exchange=account,
        marks={"BTC": 110_000.0, "ETH": 3_300.0},
        risk_engine=risk,
        latest_decisions=decisions,
        now_ms=BASE_MS + 2 * HOUR,
    )


# ==========================================================================
# Content: every section the brief demands
# ==========================================================================

def test_header_carries_equity_and_pnl(report) -> None:
    assert report.starting_capital == 100_000.0
    assert report.equity > 100_000.0
    assert report.pnl_all_time == pytest.approx(report.equity - 100_000.0)
    assert report.window_end_ms - report.window_start_ms == 6 * HOUR


def test_open_positions_are_listed_with_everything_needed(report) -> None:
    assert len(report.positions) == 1
    position = report.positions[0]
    assert position.coin == "BTC"
    assert position.side == "long"
    assert position.entry_price == pytest.approx(100_000.0)
    assert position.current_price == pytest.approx(110_000.0)
    assert position.unrealized_pnl == pytest.approx(5_000.0)
    assert position.leverage > 0


def test_closed_trades_this_window_are_reported(report) -> None:
    assert len(report.trades) == 1
    trade = report.trades[0]
    assert trade.coin == "ETH"
    assert trade.net_pnl == pytest.approx(600.0)
    assert trade.entry_price == pytest.approx(3_000.0)
    assert trade.exit_price == pytest.approx(3_300.0)


def test_the_plan_section_states_probability_and_target(report) -> None:
    assert len(report.plans) == 1
    plan = report.plans[0]
    assert plan.probability == pytest.approx(0.62)
    assert plan.direction == "long"
    assert plan.target_notional == pytest.approx(8_000.0)
    assert plan.regime == "trending_up"


def test_the_factors_behind_each_decision_are_included(report) -> None:
    drivers = report.plans[0].drivers
    assert [name for name, _ in drivers] == ["mom_ret_24_z", "fund_rate_z_24"]


def test_risk_status_reports_usage_against_limits(report) -> None:
    status = report.risk_status
    assert status["profile"] == "conservative"
    assert "drawdown" in status and "daily_loss" in status
    assert status["leverage_limit"] == 2.0


# ==========================================================================
# The narrative must be grounded, never invented
# ==========================================================================

def test_reasoning_cites_the_regime_probability_and_drivers(report) -> None:
    text = report.plans[0].reasoning
    assert "trending higher" in text
    assert "0.62" in text
    assert "mom_ret_24_z" in text


def test_a_flat_model_is_described_plainly_not_dressed_up() -> None:
    plan = PlanLine(
        coin="SOL", probability=0.48, confidence=0.01, direction="flat",
        regime="ranging", target_notional=0.0, current_notional=0.0,
        action="flat: no signal", risk_summary="approved $0",
    )
    text = market_reasoning(plan)
    assert "no meaningful opinion" in text
    assert "range-bound" in text
    assert "Currently flat" in text


def test_missing_attribution_is_admitted_not_faked() -> None:
    plan = PlanLine("BTC", 0.7, 0.5, "long", "trending_up", 1000.0, 0.0, "long", "ok")
    assert "No feature attribution was available" in market_reasoning(plan)


def test_reasoning_contains_no_forecast_language() -> None:
    """The report describes what the model estimates, never what will happen."""
    plan = PlanLine(
        "BTC", 0.7, 0.5, "long", "trending_up", 1000.0, 0.0, "long", "ok",
        drivers=[("mom_ret_24_z", 0.3)],
    )
    text = market_reasoning(plan).lower()
    for forbidden in ("will rise", "will fall", "guaranteed", "certain to"):
        assert forbidden not in text


# ==========================================================================
# Rendering
# ==========================================================================

def test_text_report_contains_every_required_section(report) -> None:
    text = render_text(report)
    for heading in (
        "6 HOUR REPORT", "PERFORMANCE", "OPEN POSITIONS", "TRADES THIS WINDOW",
        "PLAN AND REASONING", "RISK STATUS",
    ):
        assert heading in text
    assert DISCLAIMER in text


def test_html_report_contains_every_required_section(report) -> None:
    html = render_html(report)
    for heading in (
        "Performance", "Current positions", "What it did this window",
        "How it plans to invest next", "Risk status",
    ):
        assert heading in html
    assert DISCLAIMER in html
    assert html.strip().startswith("<div")


def test_html_escapes_untrusted_text(report) -> None:
    report.plans[0].action = "<script>alert(1)</script>"
    report.plans[0].reasoning = "<img src=x onerror=alert(1)>"
    html = render_html(report)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html or "&lt;img" in html


def test_html_and_text_agree_on_the_headline_number(report) -> None:
    equity = f"${report.equity:,.2f}"
    assert equity in render_text(report)
    assert equity in render_html(report)


def test_positions_and_trades_render_when_empty(tmp_path) -> None:
    from data.database import ParquetStore

    paths = Paths(root=tmp_path, storage=tmp_path / "s", parquet=tmp_path / "s" / "p",
                  duckdb_file=tmp_path / "s" / "d.duckdb", logs=tmp_path / "l",
                  models=tmp_path / "m")
    config = frictionless_config()
    exchange = PaperExchange(100_000.0, config=config, simulator=FillSimulator(config))
    data = ReportBuilder(store=ParquetStore(paths)).build(
        exchange=exchange, marks={}, now_ms=BASE_MS
    )
    text = render_text(data)
    assert "None." in text
    assert "No positions were closed" in text
    assert "<div" in render_html(data)


# ==========================================================================
# Email
# ==========================================================================

def test_subject_summarises_the_account() -> None:
    subject = subject_line(112_500.0, 100_000.0, -1_200.0)
    assert "$112,500" in subject
    assert "+12.50%" in subject
    assert "-$1,200" in subject
    assert subject.startswith("[Paper]")


def test_message_is_multipart_with_a_text_fallback() -> None:
    emailer = Emailer(ReportConfig(
        smtp_host="smtp.example.com", smtp_user="a@b.c",
        smtp_app_password="secret", sender="a@b.c", recipient="d@e.f",
    ))
    message = emailer.build_message("subject", "plain body", "<p>html body</p>")
    assert message.is_multipart()
    types = {part.get_content_type() for part in message.walk()}
    assert "text/plain" in types and "text/html" in types
    assert message["To"] == "d@e.f"


def test_unconfigured_email_reports_what_is_missing(tmp_path) -> None:
    emailer = Emailer(ReportConfig(smtp_host="", smtp_app_password="", recipient=""),
                      fallback_dir=tmp_path)
    assert not emailer.configured
    missing = emailer.missing_settings()
    assert "SMTP_APP_PASSWORD" in missing and "REPORT_RECIPIENT" in missing


def test_a_failed_send_saves_the_report_and_does_not_raise(tmp_path) -> None:
    """Email must never take the trading loop down."""
    emailer = Emailer(ReportConfig(smtp_host="", smtp_app_password=""), fallback_dir=tmp_path)
    result = emailer.send("subject", "text", "<p>html</p>")
    assert result.sent is False
    assert result.saved_to is not None and result.saved_to.exists()
    assert "not configured" in result.error


def test_send_retries_with_backoff_then_gives_up(tmp_path, monkeypatch) -> None:
    emailer = Emailer(
        ReportConfig(smtp_host="smtp.example.com", smtp_user="a@b.c",
                     smtp_app_password="x", sender="a@b.c", recipient="d@e.f"),
        max_attempts=3, backoff_base_s=0.0, backoff_max_s=0.0, fallback_dir=tmp_path,
    )
    attempts = {"n": 0}

    def boom(message):
        attempts["n"] += 1
        raise OSError("connection refused")

    monkeypatch.setattr(emailer, "_deliver", boom)
    result = emailer.send("s", "t", "<p>h</p>")
    assert attempts["n"] == 3
    assert result.sent is False
    assert result.attempts == 3
    assert "connection refused" in result.error


def test_send_succeeds_after_a_transient_failure(tmp_path, monkeypatch) -> None:
    emailer = Emailer(
        ReportConfig(smtp_host="smtp.example.com", smtp_user="a@b.c",
                     smtp_app_password="x", sender="a@b.c", recipient="d@e.f"),
        max_attempts=3, backoff_base_s=0.0, backoff_max_s=0.0, fallback_dir=tmp_path,
    )
    attempts = {"n": 0}

    def flaky(message):
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise OSError("temporary failure")

    monkeypatch.setattr(emailer, "_deliver", flaky)
    result = emailer.send("s", "t", "<p>h</p>")
    assert result.sent is True
    assert result.attempts == 2


def test_credentials_never_appear_in_the_rendered_report(report) -> None:
    text = render_text(report) + render_html(report)
    assert "SMTP_APP_PASSWORD" not in text
    assert "password" not in text.lower()


# ==========================================================================
# The model paragraph
# ==========================================================================

def _learning(**overrides) -> dict:
    base = {
        "model": {
            "backend": "lightgbm",
            "question": "P(return_4bar > +0.30%)",
            "features": 106,
            "trained_through_ms": 1_767_225_600_000,
            "val_auc": 0.5042,
        },
        "retrain": {
            "enabled": True,
            "next_ms": 1_767_312_000_000,
            "last_outcome": "retrain promoted: fresher data",
        },
        "scorecard": "LIVE SCORECARD\n  resolved calls : 1,204\n  live AUC : 0.4980",
    }
    base.update(overrides)
    return base


def test_the_report_says_which_model_made_the_decisions() -> None:
    """The model refits itself on a schedule. Without this, a change in
    behaviour is indistinguishable from a change in the market."""
    from reporting.report_builder import learning_lines

    text = "\n".join(learning_lines(_learning()))
    assert "lightgbm" in text
    assert "0.5042" in text
    assert "coin flip" in text
    assert "live AUC : 0.4980" in text


def test_the_report_says_when_the_model_next_changes() -> None:
    from reporting.report_builder import learning_lines

    assert any("Next refit" in line for line in learning_lines(_learning()))


def test_a_frozen_model_is_reported_as_frozen() -> None:
    """Silence would read as "it is learning", which would be a lie."""
    from reporting.report_builder import learning_lines

    lines = learning_lines(_learning(retrain={"enabled": False}))
    assert any("Retraining is OFF" in line for line in lines)


def test_both_renderings_carry_the_model_section() -> None:
    """The plain-text fallback must contain everything the HTML does."""
    from reporting.report_builder import ReportData, render_html, render_text

    data = ReportData(
        generated_ms=1_767_225_600_000,
        window_start_ms=1_767_204_000_000,
        window_end_ms=1_767_225_600_000,
        profile="aggressive",
        starting_capital=100_000.0,
        equity=100_000.0,
        cash=100_000.0,
        unrealized=0.0,
        pnl_window=0.0,
        pnl_today=0.0,
        pnl_all_time=0.0,
        learning=_learning(),
    )
    for rendered in (render_text(data), render_html(data)):
        assert "lightgbm" in rendered
        assert "0.5042" in rendered


def test_a_report_still_renders_when_the_model_cannot_be_read() -> None:
    from reporting.report_builder import learning_lines

    lines = learning_lines({"model": None, "retrain": {"enabled": True}})
    assert any("No model artefact" in line for line in lines)
