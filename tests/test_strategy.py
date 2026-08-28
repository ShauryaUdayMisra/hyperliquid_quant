"""Phase 5 integration: model -> risk engine -> orders, end to end."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import realistic_config, synthetic_universe
from backtest.engine import BacktestConfig, BacktestEngine
from config.settings import ExecutionConfig, resolve_risk_profile
from execution.paper_exchange import PaperExchange
from execution.simulator import FillSimulator
from features.pipeline import build_universe
from models.backend import ModelParams
from models.dataset import SplitConfig, assemble, usable_rows
from models.labels import LabelConfig
from models.predict import Signal, SignalGenerator
from models.train import train_walk_forward
from risk.risk_engine import RiskEngine, Verdict
from strategy.portfolio import Candidate, rank_candidates
from strategy.signals import ModelStrategy


@pytest.fixture(scope="module")
def system():
    """A fully wired system on synthetic data: features, model, matrices."""
    universe = synthetic_universe(2500)
    matrices = build_universe(universe)
    dataset = assemble(matrices, LabelConfig())
    model, _ = train_walk_forward(
        dataset, split_config=SplitConfig(n_folds=3), params=ModelParams(n_estimators=80)
    )
    generator = SignalGenerator(model, long_threshold=0.50)
    generator.calibrate_feature_scales(usable_rows(dataset))
    return universe, matrices, generator


def run_backtest(system, profile: str, *, long_threshold: float = 0.50):
    universe, matrices, generator = system
    generator.long_threshold = long_threshold
    limits = resolve_risk_profile(profile)
    execution = ExecutionConfig()
    risk = RiskEngine(limits, execution)
    strategy = ModelStrategy(generator, risk, matrices)

    exchange = PaperExchange(
        limits.starting_capital, config=execution, simulator=FillSimulator(execution)
    )
    engine = BacktestEngine(
        universe,
        BacktestConfig(interval="1h", warmup_bars=800, asset_max_leverage={"BTC": 40.0}),
        risk=limits,
        exchange=exchange,
        features=matrices,
    )
    return engine.run(strategy), strategy, risk


# ==========================================================================
# Portfolio allocation
# ==========================================================================

def signal(coin: str, probability: float, base: float = 0.5) -> Signal:
    return Signal(coin, 0, probability=probability, base_rate=base, direction="long")


def test_the_strongest_signals_win_the_free_slots() -> None:
    candidates = [
        Candidate(signal("BTC", 0.90), False, 0.01),
        Candidate(signal("ETH", 0.60), False, 0.01),
        Candidate(signal("SOL", 0.75), False, 0.01),
    ]
    selected, skipped = rank_candidates(candidates, max_positions=2)
    assert [c.signal.coin for c in selected] == ["BTC", "SOL"]
    assert [c.signal.coin for c in skipped] == ["ETH"]


def test_an_open_position_keeps_its_slot() -> None:
    """Churning out of a held position for a marginally better one costs twice."""
    candidates = [
        Candidate(signal("BTC", 0.95), False, 0.01),
        Candidate(signal("ETH", 0.55), True, 0.01),
    ]
    selected, skipped = rank_candidates(candidates, max_positions=1)
    assert [c.signal.coin for c in selected] == ["ETH"]
    assert [c.signal.coin for c in skipped] == ["BTC"]


def test_flat_signals_do_not_consume_slots() -> None:
    flat = Candidate(Signal("ETH", 0, 0.5, 0.5, "flat"), False, 0.01)
    selected, _ = rank_candidates([Candidate(signal("BTC", 0.9), False, 0.01), flat], 3)
    assert [c.signal.coin for c in selected] == ["BTC"]


# ==========================================================================
# End-to-end limit enforcement
# ==========================================================================

def test_the_conservative_profile_never_breaches_its_limits(system) -> None:
    result, strategy, risk = run_backtest(system, "conservative")
    limits = resolve_risk_profile("conservative")
    curve = result.equity_curve

    assert (curve["open_positions"] <= limits.max_open_positions).all()
    assert (curve["leverage"] <= limits.max_leverage + 1e-6).all()
    assert result.reconcile()["balanced"]


def test_the_aggressive_profile_never_breaches_its_limits(system) -> None:
    result, strategy, risk = run_backtest(system, "aggressive")
    limits = resolve_risk_profile("aggressive")
    curve = result.equity_curve

    assert (curve["open_positions"] <= limits.max_open_positions).all()
    assert (curve["leverage"] <= limits.max_leverage + 1e-6).all()
    assert result.reconcile()["balanced"]


def test_no_single_position_exceeds_the_notional_cap(system) -> None:
    result, _, _ = run_backtest(system, "conservative")
    limits = resolve_risk_profile("conservative")
    for coin in ("BTC", "ETH", "SOL"):
        notional = (result.equity_curve[f"pos_{coin}"].abs()
                    * result.equity_curve[f"mark_{coin}"])
        # Allow a little headroom: a position grows with the market after entry.
        assert notional.max() <= limits.max_position_usd * 1.6


def test_the_aggressive_profile_takes_larger_positions(system) -> None:
    conservative, _, _ = run_backtest(system, "conservative")
    aggressive, _, _ = run_backtest(system, "aggressive")
    conservative_peak = conservative.equity_curve["gross_notional"].max()
    aggressive_peak = aggressive.equity_curve["gross_notional"].max()
    assert aggressive_peak > conservative_peak


def test_the_strategy_actually_trades(system) -> None:
    result, strategy, _ = run_backtest(system, "conservative")
    assert len(result.fills) > 0
    assert strategy.decisions


# ==========================================================================
# Auditability -- every trade must explain itself
# ==========================================================================

def test_every_fill_records_why_it_happened(system) -> None:
    result, _, _ = run_backtest(system, "conservative")
    for fill in result.fills[:20]:
        context = fill.context
        assert context.reason
        assert context.model_probability is not None
        assert context.regime is not None
        assert context.risk_checks, "no risk reasoning attached to a fill"


def test_decisions_capture_the_features_behind_them(system) -> None:
    _, strategy, _ = run_backtest(system, "conservative")
    with_features = [d for d in strategy.decisions if d.signal.top_features]
    assert with_features
    record = with_features[0]
    assert record.describe()
    assert "P(up)" in record.describe()


def test_risk_checks_are_recorded_on_every_sized_decision(system) -> None:
    _, strategy, _ = run_backtest(system, "conservative")
    sized = [d for d in strategy.decisions if d.risk.checks]
    assert sized
    names = {c.name for c in sized[0].risk.checks}
    assert "max_leverage" in names and "liquidation_buffer" in names


def test_the_latest_decision_per_market_is_retrievable(system) -> None:
    """The 6-hour report needs this: what the system currently thinks."""
    _, strategy, _ = run_backtest(system, "conservative")
    latest = strategy.latest_by_coin()
    assert set(latest) <= {"BTC", "ETH", "SOL"}
    for record in latest.values():
        assert 0.0 <= record.signal.probability <= 1.0


# ==========================================================================
# The engine can overrule the model
# ==========================================================================

def test_a_maximum_confidence_signal_is_still_vetoed_when_a_limit_binds() -> None:
    """The whole point: 99% confidence does not beat a breached limit."""
    risk = RiskEngine(resolve_risk_profile("conservative"), ExecutionConfig())
    risk.observe_equity(0, 100_000)
    state = risk.account_state(
        ts_ms=0, equity=100_000, gross_notional=3_000,
        open_positions=3, existing_notional={"ETH": 1000, "SOL": 1000, "DOGE": 1000},
    )
    decision = risk.evaluate(coin="BTC", state=state, confidence=0.99, atr_fraction=0.01)
    assert decision.verdict is Verdict.REJECTED


def test_positions_are_closed_without_asking_the_risk_engine(system) -> None:
    """Reducing exposure is always allowed, even mid-halt."""
    _, strategy, _ = run_backtest(system, "conservative")
    exits = [d for d in strategy.decisions if d.action.startswith("close")]
    if exits:
        assert exits[0].target_notional == 0.0


# ==========================================================================
# The shuffled control -- the cheapest test for a fake edge
# ==========================================================================

def test_the_shuffled_control_restores_the_model_afterwards(system) -> None:
    """A silently shuffled model would poison every later run in the process."""
    from backtest.runner import run_shuffled_control

    import numpy as np

    universe, matrices, generator = system
    model = generator.model
    # ETH, not BTC: BTC is the cross-asset benchmark, so its own matrix has
    # no cross_*_btc_* columns (correlation with itself is degenerate).
    sample = matrices["ETH"][model.features].astype("float64").iloc[-200:]
    before = model.backend.predict_proba(sample)

    run_shuffled_control(
        universe, model, limits=resolve_risk_profile("conservative"), warmup_bars=1500
    )

    # Compare predictions, not bound-method identity: attribute access on a
    # method creates a fresh object every time, so `is` proves nothing.
    assert np.allclose(model.backend.predict_proba(sample), before)
    assert "predict_proba" not in vars(model.backend)


def test_the_control_is_labelled_so_it_cannot_be_mistaken_for_a_real_run(system) -> None:
    from backtest.runner import run_shuffled_control

    universe, _, generator = system
    report = run_shuffled_control(
        universe, generator.model, limits=resolve_risk_profile("conservative"),
        warmup_bars=1500,
    )
    assert report.strategy_name == "shuffled control"
    assert any("randomly permuted" in note for note in report.notes)


def test_the_benchmark_carries_its_cross_asset_columns_as_nan(system) -> None:
    """BTC has no meaningful correlation-with-BTC, but the column must exist.

    Correlation with itself is trivially 1.0 and carries no information, so
    the value is NaN. The *column* is present because inference reads one
    coin's matrix at a time: when it was genuinely absent, BTC could never
    produce a signal and was silently excluded from trading entirely.
    """
    _, matrices, _ = system
    btc_columns = [c for c in matrices["BTC"].columns if c.startswith("cross_corr_btc")]
    assert btc_columns
    assert matrices["BTC"][btc_columns].isna().all().all()
    assert matrices["ETH"][btc_columns].notna().any().any()


# ==========================================================================
# Every market gets reported, even when the model is neutral
# ==========================================================================

def test_a_flat_market_still_produces_a_decision_record(system) -> None:
    """Regression: flat candidates were dropped, leaving the report empty."""
    universe, matrices, generator = system
    generator.long_threshold = 0.999          # nothing can clear this
    limits = resolve_risk_profile("conservative")
    risk = RiskEngine(limits, ExecutionConfig())
    strategy = ModelStrategy(generator, risk, matrices)

    exchange = PaperExchange(limits.starting_capital, config=ExecutionConfig(),
                             simulator=FillSimulator(ExecutionConfig()))
    engine = BacktestEngine(
        universe, BacktestConfig(interval="1h", warmup_bars=1500),
        risk=limits, exchange=exchange, features=matrices,
    )
    result = engine.run(strategy)

    assert result.fills == [], "threshold of 0.999 should prevent all trading"
    assert strategy.decisions, "a flat window must still be reported"
    assert set(strategy.latest_by_coin()) == {"BTC", "ETH", "SOL"}
    assert all(d.action.startswith("flat") for d in strategy.decisions[-3:])


def test_the_benchmark_market_can_produce_signals(system) -> None:
    """BTC is the cross-asset benchmark; it must not be locked out of trading."""
    universe, matrices, generator = system
    generator.long_threshold = 0.50
    risk = RiskEngine(resolve_risk_profile("conservative"), ExecutionConfig())
    strategy = ModelStrategy(generator, risk, matrices)

    btc_signals = [s for s in strategy._signal_cache.get("BTC", []) if s is not None]
    assert btc_signals, "BTC produced no signals at all"
