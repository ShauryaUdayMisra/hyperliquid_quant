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


def test_schema_drift_degrades_gracefully_instead_of_disabling_trading(system, caplog) -> None:
    """Regression: a column the model expected but the live matrix lacked
    made the strategy silently decline to signal on every market, so the
    deployed system did nothing at all and looked merely 'flat'."""
    import logging

    universe, matrices, generator = system
    risk = RiskEngine(resolve_risk_profile("conservative"), ExecutionConfig())

    # Drop three columns the model needs, as if inference loaded different
    # inputs from training.
    dropped = generator.model.features[:3]
    trimmed = {c: m.drop(columns=dropped) for c, m in matrices.items()}

    strategy = ModelStrategy(generator, risk, trimmed, precompute=False)
    with caplog.at_level(logging.WARNING):
        signal = strategy._signal_for("ETH", len(trimmed["ETH"]) - 1)

    assert signal is not None, "drift must not silently disable signalling"
    assert 0.0 <= signal.probability <= 1.0
    assert any("absent from the live matrix" in r.message for r in caplog.records)


def test_the_drift_warning_fires_once_per_market(system) -> None:
    universe, matrices, generator = system
    risk = RiskEngine(resolve_risk_profile("conservative"), ExecutionConfig())
    trimmed = {c: m.drop(columns=generator.model.features[:2]) for c, m in matrices.items()}
    strategy = ModelStrategy(generator, risk, trimmed, precompute=False)

    for _ in range(3):
        strategy._signal_for("ETH", len(trimmed["ETH"]) - 1)
    assert strategy._warned_missing == {"ETH"}


# ==========================================================================
# Activity rules: the holding cap and the idle timer
# ==========================================================================

def run_with_activity_rules(system, *, max_hold_ms=None, min_hold_ms=None,
                            max_idle_ms=None,
                            long_threshold=0.50, profile="conservative"):
    universe, matrices, generator = system
    generator.long_threshold = long_threshold
    limits = resolve_risk_profile(profile)
    execution = ExecutionConfig()
    risk = RiskEngine(limits, execution)
    strategy = ModelStrategy(
        generator, risk, matrices,
        max_hold_ms=max_hold_ms, min_hold_ms=min_hold_ms,
        max_idle_ms=max_idle_ms,
    )
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
    return engine.run(strategy), strategy


def holding_periods_ms(result) -> list[int]:
    return [
        int(t.closed_ts_ms) - int(t.opened_ts_ms)
        for t in result.trades
        if getattr(t, "opened_ts_ms", None) and getattr(t, "closed_ts_ms", None)
    ]


def test_no_position_outlives_the_holding_cap(system) -> None:
    cap = 6 * 3_600_000        # six hours, short enough to bind often
    result, _ = run_with_activity_rules(system, max_hold_ms=cap)
    held = holding_periods_ms(result)
    assert held, "the run produced no closed trades to measure"
    # The cap is evaluated on a bar, so a position can survive at most one
    # further bar before the closing order executes at the next open.
    assert max(held) <= cap + 2 * 3_600_000


def test_without_a_cap_positions_may_be_held_longer(system) -> None:
    cap = 6 * 3_600_000
    capped, _ = run_with_activity_rules(system, max_hold_ms=cap)
    uncapped, _ = run_with_activity_rules(system, max_hold_ms=None)
    # Guards against the cap being trivially satisfied because nothing was
    # ever held that long to begin with.
    assert max(holding_periods_ms(uncapped)) > max(holding_periods_ms(capped))


def test_the_idle_timer_forces_an_entry_the_model_did_not_ask_for(system) -> None:
    # A threshold this high means the model never asks to enter, so every
    # trade in the run exists only because the idle timer forced it.
    _, strategy = run_with_activity_rules(
        system, max_idle_ms=4 * 3_600_000, long_threshold=0.999
    )
    forced = [d for d in strategy.decisions if d.action.startswith("forced long")]
    assert forced, "the idle timer never fired"
    assert all(d.signal.probability < 0.999 for d in forced)


def test_without_an_idle_timer_a_silent_model_trades_nothing(system) -> None:
    result, strategy = run_with_activity_rules(
        system, max_idle_ms=None, long_threshold=0.999
    )
    assert not [d for d in strategy.decisions if d.action.startswith("forced long")]
    assert result.trades == []


def test_the_holding_cap_outranks_a_still_confident_model(system) -> None:
    # Threshold of 0.0 means every bar asks to be long, so any close at all
    # must have come from the cap rather than from a fading signal.
    _, strategy = run_with_activity_rules(
        system, max_hold_ms=3 * 3_600_000, long_threshold=0.0
    )
    closes = [d for d in strategy.decisions if "holding cap" in d.action]
    assert closes, "an always-long model was never closed by the cap"


def test_a_restarted_process_inherits_the_true_position_age(system) -> None:
    """The clock lives on the position, not in the strategy's memory.

    A live restart builds a new ModelStrategy against a restored account.
    If age were counted in the strategy, every restart would silently grant
    each open position a fresh 24 hours.
    """
    from backtest.engine import MarketView

    universe, matrices, _ = system
    limits = resolve_risk_profile("conservative")
    execution = ExecutionConfig()
    exchange = PaperExchange(
        limits.starting_capital, config=execution, simulator=FillSimulator(execution)
    )
    coin = next(iter(universe))
    opened = 1_700_000_000_000
    position = exchange.position(coin)
    position.size, position.entry_price, position.opened_ts_ms = 1.0, 100.0, opened

    engine = BacktestEngine(
        universe,
        BacktestConfig(interval="1h", warmup_bars=800, asset_max_leverage={"BTC": 40.0}),
        risk=limits, exchange=exchange, features=matrices,
    )
    now = opened + 30 * 3_600_000
    snapshots = {c: engine._snapshot(c, 900) for c in universe}
    view = MarketView(900, now, universe, snapshots, exchange, limits, matrices)

    assert view.position_age_ms(coin) == 30 * 3_600_000
    other = [c for c in universe if c != coin]
    if other:
        assert view.position_age_ms(other[0]) is None


def test_a_reduction_that_would_leave_dust_closes_instead(system) -> None:
    """The stub that a partial exit leaves behind must not be untradeable.

    A remainder below the minimum trade size can never be sold, so the
    position becomes permanent -- immune to the exit signal, the holding
    cap and everything else.
    """
    from backtest.engine import MarketView
    from execution.paper_exchange import DecisionContext
    from strategy.base import BaseStrategy

    universe, matrices, _ = system
    limits = resolve_risk_profile("conservative")
    execution = ExecutionConfig()
    exchange = PaperExchange(
        limits.starting_capital, config=execution, simulator=FillSimulator(execution)
    )
    coin = next(iter(universe))
    engine = BacktestEngine(
        universe,
        BacktestConfig(interval="1h", warmup_bars=800, asset_max_leverage={"BTC": 40.0}),
        risk=limits, exchange=exchange, features=matrices,
    )
    snapshots = {c: engine._snapshot(c, 900) for c in universe}
    view = MarketView(900, snapshots[coin].ts_ms, universe, snapshots,
                      exchange, limits, matrices)
    price = view.price(coin)

    position = exchange.position(coin)
    position.size = 1_000.0 / price          # a $1,000 position
    position.entry_price = price
    position.opened_ts_ms = view.ts_ms - 3_600_000

    # Ask to keep $4 of it -- below the $10 minimum trade size.
    orders = BaseStrategy.orders_to_reach(
        view, coin, 4.0 / price, DecisionContext(reason="trim")
    )
    assert len(orders) == 1
    assert orders[0].size == pytest.approx(position.size)   # all of it, not most


def test_an_existing_dust_position_can_still_be_closed(system) -> None:
    from backtest.engine import MarketView
    from execution.paper_exchange import DecisionContext
    from strategy.base import BaseStrategy

    universe, matrices, _ = system
    limits = resolve_risk_profile("conservative")
    execution = ExecutionConfig()
    exchange = PaperExchange(
        limits.starting_capital, config=execution, simulator=FillSimulator(execution)
    )
    coin = next(iter(universe))
    engine = BacktestEngine(
        universe,
        BacktestConfig(interval="1h", warmup_bars=800, asset_max_leverage={"BTC": 40.0}),
        risk=limits, exchange=exchange, features=matrices,
    )
    snapshots = {c: engine._snapshot(c, 900) for c in universe}
    view = MarketView(900, snapshots[coin].ts_ms, universe, snapshots,
                      exchange, limits, matrices)

    position = exchange.position(coin)
    position.size = 3.0 / view.price(coin)   # a $3 stub, below the minimum
    position.entry_price = view.price(coin)
    position.opened_ts_ms = view.ts_ms - 3_600_000

    orders = BaseStrategy.orders_to_reach(
        view, coin, 0.0, DecisionContext(reason="exit")
    )
    assert len(orders) == 1, "a position too small to trade is a position you cannot leave"
    assert orders[0].reduce_only


def test_the_idle_clock_survives_a_restart() -> None:
    """A redeploy must not hand the timer a fresh zero.

    The live service restarts on every deploy. With the clock counted in
    memory it reset each time, so on a service that redeploys more often
    than the idle limit the forced entry could never fire at all.
    """
    from unittest.mock import Mock

    from models.predict import SignalGenerator

    generator = Mock(spec=SignalGenerator)
    generator.long_threshold = 0.55
    generator.model = Mock(features=[])

    started = 1_700_000_000_000
    before = ModelStrategy(
        generator, Mock(), {}, precompute=False,
        max_idle_ms=6 * 3_600_000, idle_since_ms=started,
    )
    assert before.idle_ms(started + 5 * 3_600_000) == 5 * 3_600_000

    # A restart builds a new strategy and hands it the persisted timestamp.
    after = ModelStrategy(
        generator, Mock(), {}, precompute=False,
        max_idle_ms=6 * 3_600_000, idle_since_ms=before.idle_since_ms,
    )
    assert after.idle_ms(started + 5 * 3_600_000) == 5 * 3_600_000

    # Without the handover the clock would start over.
    naive = ModelStrategy(generator, Mock(), {}, precompute=False,
                          max_idle_ms=6 * 3_600_000)
    assert naive.idle_ms(started + 5 * 3_600_000) == 0


def test_a_forced_entry_fills_every_free_slot(system) -> None:
    """Opening one name and waiting is concentration, not diversification.

    With three markets and three slots, a forced entry from flat should put
    the account into all three, not just the highest-ranked one.
    """
    _, strategy = run_with_activity_rules(
        system, max_idle_ms=4 * 3_600_000, long_threshold=0.999
    )
    forced = [d for d in strategy.decisions if d.action.startswith("forced long")]
    assert forced, "the idle timer never fired"

    by_bar: dict[int, set[str]] = {}
    for record in forced:
        by_bar.setdefault(record.ts_ms, set()).add(record.coin)
    widest = max(len(coins) for coins in by_bar.values())
    assert widest > 1, "every forced entry opened a single market"
    assert widest <= 3


def test_a_forced_entry_does_not_exceed_the_position_limit(system) -> None:
    universe, matrices, generator = system
    generator.long_threshold = 0.999
    limits = resolve_risk_profile("conservative")
    risk = RiskEngine(limits, ExecutionConfig())
    strategy = ModelStrategy(
        generator, risk, matrices,
        max_idle_ms=3_600_000, idle_since_ms=1_000_000, precompute=True,
    )
    candidates = [
        Candidate(signal(coin, 0.30 + i / 100), False, 0.01)
        for i, coin in enumerate(["BTC", "ETH", "SOL", "DOGE", "HYPE"])
    ]
    chosen = strategy._forced_entries(candidates, ts_ms=10_000_000, held=0)
    assert len(chosen) == limits.max_open_positions

    # Slots already occupied are not double-filled.
    fewer = strategy._forced_entries(candidates, ts_ms=10_000_000, held=2)
    assert len(fewer) == limits.max_open_positions - 2


def test_a_position_is_not_closed_on_a_fade_before_the_minimum_hold(system) -> None:
    """The two activity rules were fighting each other.

    A forced entry is opened *below* the entry threshold by definition, so
    on the very next bar the same probability reads as "below the exit
    threshold" and closes it. 72.5% of round trips lasted exactly one hour.
    """
    _, churning = run_with_activity_rules(
        system, max_idle_ms=3_600_000, min_hold_ms=None, long_threshold=0.999
    )
    _, patient = run_with_activity_rules(
        system, max_idle_ms=3_600_000, min_hold_ms=4 * 3_600_000,
        long_threshold=0.999,
    )
    early = [d for d in patient.decisions if "minimum hold" in d.action]
    assert early, "the minimum hold never prevented an exit"

    churn_exits = [d for d in churning.decisions if d.action.startswith("close long")]
    patient_exits = [d for d in patient.decisions if d.action.startswith("close long")]
    assert len(patient_exits) < len(churn_exits)


def test_the_holding_cap_still_outranks_the_minimum_hold(system) -> None:
    """A minimum hold is an opinion; the cap and liquidation are risk."""
    _, strategy = run_with_activity_rules(
        system, max_hold_ms=2 * 3_600_000, min_hold_ms=12 * 3_600_000,
        max_idle_ms=3_600_000, long_threshold=0.999,
    )
    capped = [d for d in strategy.decisions if "holding cap" in d.action]
    assert capped, "the minimum hold suppressed the risk exit above it"


def test_a_market_gated_out_for_sparse_features_says_so(system, caplog) -> None:
    """Silence here is indistinguishable from "the model had no opinion".

    A market whose features are too sparse produces no decision record and
    no order, which on the dashboard looks exactly like a flat call. BTC
    disappeared from a 25-market universe this way with nothing logged.
    """
    import logging

    import numpy as np
    import pandas as pd

    universe, matrices, generator = system
    limits = resolve_risk_profile("conservative")
    strategy = ModelStrategy(
        generator, RiskEngine(limits, ExecutionConfig()), {}, precompute=False
    )

    coin = next(iter(matrices))
    frame = matrices[coin].tail(1).copy()
    for column in list(generator.model.features)[: int(len(generator.model.features) * 0.6)]:
        frame[column] = np.nan
    strategy.features = {coin: frame}

    with caplog.at_level(logging.WARNING, logger="strategy.signals"):
        assert strategy._signal_for(coin, 0) is None

    messages = [r.getMessage() for r in caplog.records]
    assert any("will not be scored" in m for m in messages), \
        "the market was dropped silently"
    assert any("70%" in m for m in messages), "the warning omits the gate"


def test_a_healthy_market_logs_no_warmup_warning(system, caplog) -> None:
    import logging

    universe, matrices, generator = system
    limits = resolve_risk_profile("conservative")
    strategy = ModelStrategy(
        generator, RiskEngine(limits, ExecutionConfig()), {}, precompute=False
    )
    coin = next(iter(matrices))
    strategy.features = {coin: matrices[coin].tail(1).copy()}

    with caplog.at_level(logging.WARNING, logger="strategy.signals"):
        assert strategy._signal_for(coin, 0) is not None
    assert not [r for r in caplog.records if "will not be scored" in r.getMessage()]
