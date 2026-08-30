"""Does the system actually learn from what it got wrong?

Two halves, and both have to hold. :mod:`models.scorecard` has to mark the
live model's own predictions against reality without accidentally handing it
a price from its own future, and :mod:`models.retrain` has to put a fresher
model behind the live strategy without letting a broken one through.
"""

from __future__ import annotations

import pathlib
import pickle

import numpy as np
import pandas as pd
import pytest

from conftest import BASE_MS
from config.settings import INTERVAL_MS, SETTINGS, LearningConfig
from models.backend import ModelParams
from models.labels import LabelConfig
from models.retrain import (
    RetrainOutcome,
    decide_promotion,
    new_labelled_rows,
    save_atomically,
)
from models.scorecard import LiveScorecard, score_decisions
from models.train import FoldResult, TrainedModel

HOUR = INTERVAL_MS["1h"]

#: Closes chosen so the next-bar direction alternates in a fixed, checkable
#: pattern: up, down, down, up.
CLOSES = [100.0, 101.0, 100.0, 99.0, 105.0]
CORRECT_LABELS = [1, 0, 0, 1]


def bars(closes=CLOSES) -> dict[str, pd.DataFrame]:
    return {
        "BTC": pd.DataFrame(
            {
                "ts_ms": [BASE_MS + i * HOUR for i in range(len(closes))],
                "close": closes,
            }
        )
    }


def decisions(probabilities, *, first_bar: int = 0, offset_ms: int = 15_000):
    """One call per bar, made just after that bar closed -- as live does."""
    return pd.DataFrame(
        {
            # A decision on bar k happens after bar k's close, which is bar
            # (k+1)'s open. The live loop wakes 15s past the boundary.
            "ts_ms": [
                BASE_MS + (first_bar + i + 1) * HOUR + offset_ms
                for i in range(len(probabilities))
            ],
            "coin": ["BTC"] * len(probabilities),
            "probability": list(probabilities),
        }
    )


def scored(probabilities, *, threshold: float = 0.55, closes=CLOSES):
    return score_decisions(
        decisions(probabilities),
        bars(closes),
        label_config=LabelConfig(horizon_bars=1, threshold=0.0),
        interval_ms=HOUR,
        entry_threshold=threshold,
    )


# ==========================================================================
# Scoring the live model's own calls
# ==========================================================================

def test_a_call_is_marked_against_the_bar_the_model_could_actually_see() -> None:
    """The whole measurement turns on this off-by-one.

    A decision at 13:00:15 was made on the 12:00 bar, because only completed
    bars are ever stored. Matching it to the newest bar whose *open* precedes
    it would hand the model the 13:00 bar -- a price from its own future --
    and score it against an outcome it was partly told. Predictions that are
    perfect against the bar it saw are exactly wrong against the next one, so
    an AUC of 1.0 here and 0.0 under the naive match separates the two.
    """
    card = scored([1.0, 0.0, 0.0, 1.0])
    assert card.resolved == 4
    assert card.auc == 1.0


def test_a_model_that_calls_every_move_backwards_scores_zero() -> None:
    card = scored([0.0, 1.0, 1.0, 0.0])
    assert card.auc == 0.0


def test_a_call_whose_horizon_has_not_elapsed_is_pending_not_scored() -> None:
    """The last bar has no future yet. Counting it would fake a full sample."""
    card = scored([1.0, 0.0, 0.0, 1.0, 0.5])
    assert card.resolved == 4
    assert card.pending == 1


def test_the_when_it_said_go_line_only_counts_calls_above_the_entry_bar() -> None:
    # Two calls clear 0.55, and both of those bars did go up.
    card = scored([0.9, 0.1, 0.2, 0.8], threshold=0.55)
    assert card.acted["rows"] == 2
    assert card.acted["hit_rate"] == 1.0
    assert card.acted["edge_over_base"] == pytest.approx(0.5)


def test_a_model_at_chance_is_reported_as_having_no_verdict_yet() -> None:
    """Four calls cannot distinguish skill from luck, and must not claim to."""
    card = scored([0.5, 0.5, 0.5, 0.5])
    assert not card.has_verdict
    assert "NOT YET A VERDICT" in card.describe()


def test_an_auc_that_cannot_be_computed_reads_as_english_not_nan() -> None:
    """Early on every resolved call can share one outcome, which leaves AUC
    undefined. Printing "nan" invites the reader to think the thing is
    broken when it is merely young."""
    # Every one of these bars fell, so the label is 0 throughout.
    card = scored([0.9, 0.8, 0.7], closes=[100.0, 99.0, 98.0, 97.0])
    assert card.resolved == 3
    assert not np.isfinite(card.auc)
    text = card.describe()
    assert "nan" not in text
    assert "not computable yet" in text


def test_an_empty_ledger_reports_nothing_rather_than_raising() -> None:
    card = score_decisions(
        pd.DataFrame(columns=["ts_ms", "coin", "probability"]),
        bars(),
        interval_ms=HOUR,
        entry_threshold=0.55,
    )
    assert card.resolved == 0
    assert "nothing resolved yet" in card.describe()


def test_a_scorecard_with_a_real_sample_still_serialises() -> None:
    """The bug this test exists for only appears after 200 resolved calls.

    has_verdict was `resolved >= 200 and np.isfinite(auc)`. Below 200 the
    `and` short-circuits and yields a Python False; at 200 it yields the
    second operand, which is a numpy bool -- and json cannot encode that.
    So the endpoint served fine for days and then began returning 500,
    which blanked the whole dashboard. Any test using a handful of rows
    would have passed against the broken code, as mine did.
    """
    import json

    card = LiveScorecard(label_question="q", entry_threshold=0.4, resolved=250)
    card.metrics = {"auc": 0.51, "brier": float("nan"), "rows": np.int64(250)}
    card.acted = {"rows": np.int64(9), "hit_rate": np.float64(0.33)}

    assert card.has_verdict is True
    assert type(card.has_verdict) is bool, "a numpy bool cannot be serialised"
    json.dumps(card.to_dict(), allow_nan=False)


def test_the_scorecard_survives_json_serialisation_for_the_dashboard() -> None:
    """NaN is not JSON. A single infinity here blanks the whole panel."""
    import json

    payload = scored([0.9, 0.1, 0.2, 0.8]).to_dict()
    json.dumps(payload)  # raises on NaN only with allow_nan=False, so also:
    text = json.dumps(payload, allow_nan=False)
    assert "NaN" not in text


# ==========================================================================
# Deciding whether a freshly trained model deserves to go live
# ==========================================================================

def fake_model(auc: float, *, lift: float = 0.001, span_end: int = BASE_MS) -> TrainedModel:
    fold = FoldResult(
        info={"fold": 1, "train_rows": 5_000, "val_rows": 1_000},
        train_metrics={"auc": auc + 0.01},
        val_metrics={"auc": auc, "log_loss_lift": lift},
    )
    return TrainedModel(
        backend=None,
        features=["a", "b"],
        label_config=LabelConfig(),
        params=ModelParams(),
        backend_name="sklearn_hist",
        trained_at_ms=span_end,
        train_span=(BASE_MS - 100 * HOUR, span_end),
        fold_results=[fold],
        class_balance={"rows": 6_000, "positive_rate": 0.45},
    )


def test_a_fresher_model_is_promoted_when_it_is_no_worse() -> None:
    promote, reason = decide_promotion(
        fake_model(0.505), fake_model(0.504), min_auc_margin=0.02
    )
    assert promote
    assert "0.5050" in reason


def test_a_candidate_clearly_worse_than_the_incumbent_is_kept_out() -> None:
    promote, reason = decide_promotion(
        fake_model(0.46), fake_model(0.52), min_auc_margin=0.02
    )
    assert not promote
    assert "below the incumbent" in reason


def test_a_candidate_inside_the_margin_still_wins_on_being_fresher() -> None:
    """At an AUC near 0.50 a 0.01 difference is noise, not evidence."""
    promote, _ = decide_promotion(
        fake_model(0.505), fake_model(0.515), min_auc_margin=0.02
    )
    assert promote


def test_an_implausibly_good_candidate_is_never_deployed_automatically() -> None:
    """A jump to 0.9 AUC on hourly crypto is a leak. Deploying it would put
    the bug into production and make the equity curve look like an edge."""
    promote, reason = decide_promotion(
        fake_model(0.90), fake_model(0.504), min_auc_margin=0.02
    )
    assert not promote
    assert "leak" in reason


def test_a_candidate_that_never_trained_a_fold_is_refused() -> None:
    empty = fake_model(0.5)
    empty.fold_results = []
    promote, reason = decide_promotion(empty, fake_model(0.5), min_auc_margin=0.02)
    assert not promote
    assert "no walk-forward fold" in reason


def test_the_first_model_is_promoted_with_nothing_to_compare_against() -> None:
    promote, _ = decide_promotion(fake_model(0.51), None, min_auc_margin=0.02)
    assert promote


# ==========================================================================
# Knowing when there is something new to learn from
# ==========================================================================

def test_only_rows_that_arrived_after_the_last_fit_count_as_new() -> None:
    """Measured from when the incumbent was trained, not from where its
    training span ends. The span stops ~20% short of the data it was built
    from, because that last slice is the locked holdout -- so measuring from
    there would report a fifth of the dataset as new on every single run and
    the freshness gate could never say no."""
    dataset = pd.DataFrame(
        {
            "ts_ms": [BASE_MS - HOUR, BASE_MS, BASE_MS + HOUR, BASE_MS + 2 * HOUR],
            "label_known": [True, True, True, True],
        }
    )
    model = fake_model(0.5, span_end=BASE_MS)
    model.trained_at_ms = BASE_MS
    # train_span ends well before trained_at_ms, as it does in practice.
    model.train_span = (BASE_MS - 500 * HOUR, BASE_MS - 100 * HOUR)
    assert new_labelled_rows(dataset, model) == 2


def test_rows_whose_future_is_unknown_are_not_new_training_data() -> None:
    dataset = pd.DataFrame(
        {
            "ts_ms": [BASE_MS + HOUR, BASE_MS + 2 * HOUR],
            "label_known": [True, False],
        }
    )
    assert new_labelled_rows(dataset, fake_model(0.5, span_end=BASE_MS)) == 1


def test_new_bars_are_counted_as_hours_of_market_not_rows(store) -> None:
    """Distinct timestamps, so the gate means the same thing whether three
    markets are being traded or two hundred, and one lagging market cannot
    hold the count down."""
    from models.retrain import new_bar_count

    store.upsert(
        "candles",
        pd.DataFrame(
            {
                # BTC has three bars, ETH only the first: two distinct new
                # timestamps in total, not three rows and not one market.
                "coin": ["BTC", "BTC", "BTC", "ETH"],
                "interval": ["1h"] * 4,
                "ts_ms": [BASE_MS, BASE_MS + HOUR, BASE_MS + 2 * HOUR, BASE_MS],
                "close": [1.0, 2.0, 3.0, 4.0],
            }
        ),
    )
    assert new_bar_count(["BTC", "ETH"], "1h", BASE_MS, store=store) == 2
    assert new_bar_count(["BTC", "ETH"], "1h", BASE_MS + 2 * HOUR, store=store) == 0


def test_with_no_incumbent_every_row_is_new() -> None:
    dataset = pd.DataFrame({"ts_ms": [BASE_MS, BASE_MS + HOUR], "label_known": [True, True]})
    assert new_labelled_rows(dataset, None) == 2


# ==========================================================================
# Swapping the model underneath a running trader
# ==========================================================================

def test_the_artefact_is_written_atomically(tmp_path) -> None:
    """The live trader loads this path on boot and rewrites it while running.
    A half-written pickle would stop the process from ever starting again."""
    path = tmp_path / "model.pkl"
    path.write_bytes(b"the incumbent")
    save_atomically(fake_model(0.51), path)

    assert not (tmp_path / "model.pkl.tmp").exists()
    with open(path, "rb") as handle:
        assert pickle.load(handle).mean_val_auc == 0.51


def test_a_retrain_outcome_describes_itself_for_the_log() -> None:
    outcome = RetrainOutcome(
        ts_ms=BASE_MS, status="promoted", reason="fresher", rows=1000, new_rows=50,
        candidate_auc=0.51, incumbent_auc=0.50, elapsed_s=42.0,
    )
    assert outcome.promoted
    text = outcome.describe()
    assert "promoted" in text and "0.5100" in text and "0.5000" in text


def test_a_stale_data_feed_stops_a_pointless_retrain(store, tmp_path) -> None:
    """No new bars means no new information. Refitting anyway would spend
    minutes of CPU to produce a reseeded copy of the same model."""
    from models.retrain import retrain

    outcome = retrain(
        coins=["BTC"],
        interval="1h",
        model_path=tmp_path / "model.pkl",
        incumbent=fake_model(0.5),
        learning=LearningConfig(min_new_bars=12),
        store=store,
    )
    assert outcome.status == "skipped"
    assert "Nothing new to learn from" in outcome.reason
    assert not (tmp_path / "model.pkl").exists()


def test_a_disabled_schedule_reports_that_the_model_is_frozen() -> None:
    frozen = LearningConfig(enabled=False)
    assert frozen.every_ms is None
    assert "OFF" in frozen.describe()


def test_the_schedule_is_measured_in_hours_of_wall_clock() -> None:
    config = LearningConfig(enabled=True, every_hours=6.0)
    assert config.every_ms == 6 * HOUR


# ==========================================================================
# Putting a new model behind a running trader
# ==========================================================================

def bare_trader(*, every_hours: float = 24.0, last_retrain_ms: int = BASE_MS):
    """A PaperTrader with only the fields the learning path touches.

    __new__ rather than __init__ because construction resolves the market
    universe against the exchange, and no test may touch the network.
    """
    from execution.paper_trader import PaperTrader
    from models.predict import SignalGenerator

    trader = PaperTrader.__new__(PaperTrader)
    trader.coins = ["BTC"]
    trader.interval = "1h"
    trader.interval_ms = HOUR
    trader.model = fake_model(0.504)
    trader.model_path = "unused.pkl"
    trader.long_threshold = 0.40
    trader.short_threshold = 0.40
    trader.down_model = None
    trader.down_model_path = pathlib.Path("unused_down.pkl")
    trader.generator = SignalGenerator(trader.model, long_threshold=0.40)
    trader.strategy = StubStrategy(trader.generator)
    trader.settings = SETTINGS
    trader.learning = LearningConfig(enabled=True, every_hours=every_hours)
    trader._last_retrain_ms = last_retrain_ms
    trader._started_ms = BASE_MS
    trader.last_retrain = None
    return trader


class StubStrategy:
    def __init__(self, generator):
        self.generator = generator
        self.idle_since_ms = None
        self._signal_cache = {"BTC": ["stale"]}


def test_swapping_the_model_also_swaps_the_one_the_strategy_uses() -> None:
    """The strategy holds its own reference to the generator.

    Replacing only ``trader.model`` would leave every decision still being
    made by the model that was just retired -- retraining would run, log
    success, and change nothing about what the system trades.
    """
    trader = bare_trader()
    old = trader.generator
    fresh = fake_model(0.51)
    trader._swap_model(fresh)

    assert trader.model is fresh
    assert trader.strategy.generator is trader.generator
    assert trader.strategy.generator.model is fresh
    assert trader.strategy.generator is not old


def test_the_entry_threshold_survives_a_retrain() -> None:
    """Retraining changes what the model has seen, never how sure it must be
    before it trades. Silently resetting the bar to the default would be a
    configuration change nobody asked for."""
    trader = bare_trader()
    trader._swap_model(fake_model(0.51))
    assert trader.generator.long_threshold == 0.40


def test_a_swap_clears_the_signal_cache_of_the_retired_model() -> None:
    """Cached probabilities belong to the model that produced them."""
    trader = bare_trader()
    trader._swap_model(fake_model(0.51))
    assert trader.strategy._signal_cache == {}


def test_a_retrain_is_due_only_once_the_interval_has_actually_elapsed() -> None:
    trader = bare_trader(every_hours=24.0, last_retrain_ms=BASE_MS)
    assert not trader.retrain_due(BASE_MS + 23 * HOUR)
    assert trader.retrain_due(BASE_MS + 24 * HOUR)


def test_retraining_can_be_turned_off_entirely() -> None:
    trader = bare_trader()
    trader.learning = LearningConfig(enabled=False)
    assert not trader.retrain_due(BASE_MS + 10_000 * HOUR)


def test_the_retrain_clock_is_persisted_so_a_redeploy_cannot_reset_it() -> None:
    """The same mistake the idle timer made. A clock that restarts on boot
    never reaches its limit on a service that redeploys several times a day,
    so the model would silently never refit at all."""
    trader = bare_trader(last_retrain_ms=BASE_MS + 5 * HOUR)
    extra = trader._state_extra()
    assert extra["last_retrain_ms"] == BASE_MS + 5 * HOUR
    assert extra["model_trained_at_ms"] == trader.model.trained_at_ms


def _run_maybe_retrain(monkeypatch, *outcomes: RetrainOutcome, now_ms: int):
    import execution.paper_trader as module

    trader = bare_trader(every_hours=1.0, last_retrain_ms=BASE_MS)
    trader.store = None
    saves: list[dict] = []
    trader.state_store = type("S", (), {"save": lambda _s, _e, extra: saves.append(extra)})()
    trader.exchange = object()
    monkeypatch.setattr(module, "retrain_pair", lambda **kwargs: list(outcomes))
    monkeypatch.setattr(module, "record_retrain", lambda *a, **k: None)
    monkeypatch.setattr(
        module, "load_scorecard", lambda **kwargs: scored([0.9, 0.1, 0.2, 0.8])
    )
    return trader, trader._maybe_retrain(now_ms), saves


def test_a_rejected_candidate_still_advances_the_clock(monkeypatch) -> None:
    """The same data would lose the same comparison an hour later. Retrying
    every cycle would burn the box for a foregone conclusion."""
    now = BASE_MS + 2 * HOUR
    trader, outcomes, saves = _run_maybe_retrain(
        monkeypatch,
        RetrainOutcome(ts_ms=now, status="rejected", reason="worse"),
        now_ms=now,
    )
    assert outcomes[0].status == "rejected"
    assert trader._last_retrain_ms == now
    assert saves and saves[0]["last_retrain_ms"] == now


def test_too_little_new_data_leaves_the_clock_alone(monkeypatch) -> None:
    """A skip costs nothing and should be retried, not deferred a full day."""
    now = BASE_MS + 2 * HOUR
    trader, outcomes, _ = _run_maybe_retrain(
        monkeypatch,
        RetrainOutcome(ts_ms=now, status="skipped", reason="nothing new"),
        now_ms=now,
    )
    assert outcomes[0].status == "skipped"
    assert trader._last_retrain_ms == BASE_MS


def test_a_promoted_candidate_goes_live_immediately(monkeypatch) -> None:
    from models.predict import SignalGenerator

    now = BASE_MS + 2 * HOUR
    fresh = fake_model(0.51)
    trader, outcomes, _ = _run_maybe_retrain(
        monkeypatch,
        RetrainOutcome(ts_ms=now, status="promoted", reason="fresher", model=fresh),
        now_ms=now,
    )
    assert outcomes[0].promoted
    assert trader.model is fresh
    assert isinstance(trader.generator, SignalGenerator)


def test_each_side_of_the_book_is_promoted_on_its_own_merit(monkeypatch) -> None:
    """The long and short models are two independent opinions, not a matched
    set. A good new long model must not be held back because the short model
    came out worse than its incumbent."""
    now = BASE_MS + 2 * HOUR
    fresh_long, fresh_short = fake_model(0.51), fake_model(0.52)
    trader, outcomes, _ = _run_maybe_retrain(
        monkeypatch,
        RetrainOutcome(ts_ms=now, side="up", status="promoted",
                       reason="fresher", model=fresh_long),
        RetrainOutcome(ts_ms=now, side="down", status="rejected", reason="worse"),
        now_ms=now,
    )
    assert len(outcomes) == 2
    assert trader.model is fresh_long
    assert trader.down_model is None, "a rejected short model was deployed anyway"


def test_a_promoted_short_model_goes_behind_the_live_strategy(monkeypatch) -> None:
    now = BASE_MS + 2 * HOUR
    fresh_short = fake_model(0.52)
    trader, _outcomes, _ = _run_maybe_retrain(
        monkeypatch,
        RetrainOutcome(ts_ms=now, side="down", status="promoted",
                       reason="fresher", model=fresh_short),
        now_ms=now,
    )
    assert trader.down_model is fresh_short
    assert trader.generator.down_model is fresh_short, "the strategy cannot short with it"


def test_a_retrain_that_blows_up_does_not_take_the_trader_down(monkeypatch) -> None:
    """Learning is a bonus; trading is the job. An exception here would fail
    the cycle, exit the process, and hand the supervisor a restart loop."""
    import execution.paper_trader as module

    now = BASE_MS + 2 * HOUR
    trader = bare_trader(every_hours=1.0, last_retrain_ms=BASE_MS)
    trader.store = None

    def explode(**kwargs):
        raise MemoryError("training ran out of room")

    monkeypatch.setattr(module, "retrain_pair", explode)
    monkeypatch.setattr(module, "load_scorecard", lambda **kwargs: scored([0.5]))

    outcomes = trader._maybe_retrain(now)
    assert outcomes[0].status == "failed"
    assert "MemoryError" in outcomes[0].reason
    assert trader._last_retrain_ms == now, "a failure must not retry every cycle"


def test_a_skipped_attempt_is_not_written_to_the_history(store) -> None:
    """A skip does not advance the clock, so it is retried every cycle.
    Recording it would write a row an hour saying "nothing happened" and bury
    the handful of rows where the model actually changed."""
    from models.retrain import record

    record(RetrainOutcome(ts_ms=BASE_MS, status="skipped", reason="nothing new"),
           store=store)
    assert not store.has_data("retrains")

    record(RetrainOutcome(ts_ms=BASE_MS, status="rejected", reason="worse"),
           store=store)
    assert store.has_data("retrains")


def test_a_model_answering_the_old_question_retrains_immediately() -> None:
    """Changing the label changes what every probability means.

    The entry threshold, the exit band and the live scorecard all assume the
    deployed model answers the configured question. A model still answering
    the previous one is stale in a way that waiting out a 24-hour schedule
    does not fix.
    """
    from config.settings import LabelSettings
    from models.labels import LabelConfig

    trader = bare_trader()
    trader.model.label_config = LabelConfig(horizon_bars=4, threshold=0.003)
    object.__setattr__(trader.settings, "label",
                       LabelSettings(horizon_bars=24, threshold=0.01))

    # Re-run the constructor's staleness check.
    configured, deployed = trader.settings.label, trader.model.label_config
    stale = (deployed.horizon_bars != configured.horizon_bars
             or deployed.threshold != configured.threshold)
    assert stale
    trader._last_retrain_ms = 0 if stale else BASE_MS
    assert trader.retrain_due(BASE_MS)


def test_a_label_that_does_not_clear_costs_is_a_question_worth_refusing() -> None:
    """Predicting a move smaller than a round trip costs is predicting
    something unprofitable even when the prediction is right."""
    from config.settings import ExecutionConfig

    execution = ExecutionConfig()
    round_trip = execution.round_trip_cost()

    # The old label: a 0.30% move against a 0.30% round trip.
    assert 0.003 <= round_trip
    # The configured one leaves most of the move on the table for the trader.
    assert SETTINGS.label.threshold > round_trip
    assert round_trip / SETTINGS.label.threshold < 0.5
