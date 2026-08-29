"""Refitting the deployed model on everything that has happened since.

A model trained once at first boot is a model that cannot learn. It never
sees a bar of the market it is trading: the same setup can go against it
every day for a month and the next prediction is identical to the first.
This module closes that loop -- on a schedule, the live trader refits on
all stored history, including every bar it has since been wrong about, and
swaps the result in.

What this is not: it does not manufacture an edge. The measured holdout AUC
of 0.504 says the features barely separate an up bar from a down one, and
refitting the same features on more of the same data will not change that.
What it does buy is that the model tracks the regime it is actually trading
in rather than the one that prevailed the day it was first fit, and that
its mistakes become training rows instead of being discarded.

Two decisions are worth spelling out.

**The question stays fixed.** The candidate inherits the incumbent's
:class:`~models.labels.LabelConfig`, parameters and backend preference.
Retraining changes what the model has *seen*, never what it is being asked
-- otherwise the entry threshold, the live scorecard and every stored
probability would quietly start meaning something else.

**The locked holdout is not touched.** Scoring it on every retrain would
turn a one-shot test set into a tuning set within a week. The candidate is
judged on walk-forward validation, and the honest out-of-sample record of
the incumbent is :mod:`models.scorecard` -- its own live calls, marked
against what the market actually did.

Promotion is deliberately biased toward the fresher model: a candidate is
rejected only when it looks broken (a leak-shaped AUC, no usable folds) or
is clearly worse than the incumbent. At an AUC near 0.50, differences
smaller than the margin are noise, and pretending otherwise would just be
selecting a model on coin flips.
"""

from __future__ import annotations

import logging
import os
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.settings import SETTINGS, LearningConfig, Settings
from models.train import SUSPICIOUS_AUC, TrainedModel

log = logging.getLogger(__name__)


@dataclass
class RetrainOutcome:
    """What one retrain attempt did, and why."""

    ts_ms: int
    status: str          # "promoted" | "rejected" | "skipped" | "failed"
    reason: str
    #: Which question this model answers -- "up" (longs) or "down" (shorts).
    #: Both sides refit in the same pass, so the history needs to tell them
    #: apart; without it two outcomes share a timestamp and one is lost to
    #: the upsert key.
    side: str = "up"
    rows: int = 0
    new_rows: int = 0
    span: tuple[int, int] | None = None
    candidate_auc: float = float("nan")
    candidate_lift: float = float("nan")
    incumbent_auc: float = float("nan")
    backend: str = ""
    warnings: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0
    model: TrainedModel | None = None

    @property
    def promoted(self) -> bool:
        return self.status == "promoted"

    def describe(self) -> str:
        head = f"retrain [{self.side}] {self.status}: {self.reason}"
        if self.status in {"skipped", "failed"}:
            return head
        return (
            f"{head} | {self.rows:,} rows ({self.new_rows:,} new) | "
            f"candidate AUC {self.candidate_auc:.4f} vs incumbent "
            f"{self.incumbent_auc:.4f} | {self.elapsed_s:.0f}s"
        )

    def as_row(self) -> dict[str, Any]:
        """One row for the ``retrains`` history table."""
        return {
            "ts_ms": self.ts_ms,
            "side": self.side,
            "status": self.status,
            "reason": self.reason,
            "rows": self.rows,
            "new_rows": self.new_rows,
            "span_start_ms": self.span[0] if self.span else 0,
            "span_end_ms": self.span[1] if self.span else 0,
            "candidate_auc": self.candidate_auc,
            "candidate_lift": self.candidate_lift,
            "incumbent_auc": self.incumbent_auc,
            "backend": self.backend,
            "warnings": " | ".join(self.warnings),
            "elapsed_s": self.elapsed_s,
        }


def save_atomically(model: TrainedModel, path: Path | str) -> Path:
    """Write the artefact via a temp file and a rename.

    The live trader loads this path on every boot and the retrain runs
    inside that same live process. A crash halfway through a plain write
    would leave a truncated pickle that no restart could load -- the trader
    would then refuse to start at all, which is a far worse failure than a
    stale model.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as handle:
        pickle.dump(model, handle)
    os.replace(tmp, path)
    return path


def decide_promotion(
    candidate: TrainedModel,
    incumbent: TrainedModel | None,
    *,
    min_auc_margin: float,
) -> tuple[bool, str]:
    """Promote, or keep the incumbent, and say which and why."""
    if not candidate.fold_results:
        return False, "no walk-forward fold produced a usable model"

    # Checked against the threshold itself rather than by matching the text
    # of a plausibility warning: the warning's wording is prose and would
    # eventually be reworded, silently disabling the guard.
    if candidate.mean_val_auc >= SUSPICIOUS_AUC:
        return False, (
            f"candidate AUC {candidate.mean_val_auc:.4f} is implausibly high for "
            "hourly crypto and is treated as a leak, not an edge; keeping the "
            "incumbent until a human has looked at it"
        )

    candidate_auc = candidate.mean_val_auc
    if not np.isfinite(candidate_auc):
        return False, "candidate validation AUC is undefined (a fold had one class only)"

    if incumbent is None:
        return True, "no incumbent to compare against"

    incumbent_auc = incumbent.mean_val_auc
    if not np.isfinite(incumbent_auc):
        return True, "incumbent has no usable validation AUC to defend itself with"

    if candidate_auc < incumbent_auc - min_auc_margin:
        return False, (
            f"candidate walk-forward AUC {candidate_auc:.4f} is more than "
            f"{min_auc_margin:.2f} below the incumbent's {incumbent_auc:.4f}"
        )
    return True, (
        f"candidate walk-forward AUC {candidate_auc:.4f} vs incumbent "
        f"{incumbent_auc:.4f}; trained on data through "
        f"{pd.Timestamp(candidate.train_span[1], unit='ms', tz='UTC'):%Y-%m-%d %H:%M}"
    )


def candidate_warnings(model: TrainedModel) -> list[str]:
    from models.train import plausibility_warnings

    return plausibility_warnings(model)


def new_labelled_rows(dataset: pd.DataFrame, incumbent: TrainedModel | None) -> int:
    """Labelled rows that did not exist when the incumbent was fitted.

    The cutoff is the incumbent's ``trained_at_ms`` -- the wall-clock moment
    it was fitted -- and not the end of its ``train_span``. The span ends
    where the *development* period ends, which is roughly 20% short of the
    data it was built from because the last slice is reserved as a locked
    holdout. Measuring from there counts that holdout as new every single
    time, so a fifth of the dataset always looks fresh and the "is there
    anything to learn from" gate can never say no.
    """
    if incumbent is None or dataset.empty:
        return int(len(dataset))
    fresh = dataset.loc[dataset["ts_ms"] > int(incumbent.trained_at_ms)]
    if "label_known" in fresh.columns:
        fresh = fresh.loc[fresh["label_known"]]
    return int(len(fresh))


def new_bar_count(
    coins: list[str], interval: str, since_ms: int, *, store: Any | None = None
) -> int:
    """Distinct bar timestamps stored since ``since_ms``, across ``coins``.

    Counted before any feature is built, because building features over a
    wide universe is minutes of work and there is no point spending it to
    discover that nothing has arrived. Distinct timestamps rather than rows
    so the answer means "hours of new market" regardless of whether three
    markets are being traded or two hundred, and so one lagging market does
    not drag the count down.
    """
    from data.database import MarketDatabase, ParquetStore

    store = store or ParquetStore()
    if not store.has_data("candles") or not coins:
        return 0
    placeholders = ", ".join("?" for _ in coins)
    with MarketDatabase(store) as db:
        frame = db.query(
            "SELECT count(DISTINCT ts_ms) AS bars FROM candles "
            f"WHERE interval = ? AND ts_ms > ? AND coin IN ({placeholders})",
            [interval, int(since_ms), *coins],
        )
    return int(frame["bars"].iloc[0]) if not frame.empty else 0


def build_features(
    coins: list[str], interval: str, *, store: Any | None = None
) -> dict[str, pd.DataFrame]:
    """Point-in-time features for every tradable market, from storage."""
    from data.loader import load_bars, load_funding, load_order_books
    from features.pipeline import FeatureConfig, align_bars, build_universe

    bars = align_bars(load_bars(coins, interval, store=store))
    return build_universe(
        bars,
        funding_by_coin=load_funding(list(bars), store=store),
        book_by_coin=load_order_books(list(bars), store=store),
        config=FeatureConfig(interval=interval),
    )


def retrain_pair(
    *,
    coins: list[str],
    interval: str,
    model_path: Path | str,
    incumbent: TrainedModel | None,
    down_incumbent: TrainedModel | None = None,
    settings: Settings | None = None,
    learning: LearningConfig | None = None,
    force: bool = False,
    dry_run: bool = False,
    store: Any | None = None,
) -> list[RetrainOutcome]:
    """Refit both sides of the book in one pass. Never raises.

    The long and short models answer different questions about the same
    feature matrix, so the matrix is built once and labelled twice. Each
    side is then promoted on its own merit: a good new long model is not
    held back because the short model came out worse, and vice versa. They
    were never a matched set -- they are two independent opinions.
    """
    from models.train import down_model_path

    settings = settings or SETTINGS
    learning = learning or settings.learning
    now_ms = int(time.time() * 1000)

    sides: list[tuple[str, Path, TrainedModel | None]] = [
        ("up", Path(model_path), incumbent),
        ("down", down_model_path(model_path), down_incumbent),
    ]
    # A side with no incumbent and no way to build one is simply absent --
    # a long-only deployment must not start reporting a failed short retrain
    # every day.
    sides = [s for s in sides if s[0] == "up" or s[2] is not None]

    # Build once, but only if at least one side has something new to learn.
    matrices: dict[str, pd.DataFrame] | None = None
    outcomes: list[RetrainOutcome] = []
    for side, path, model in sides:
        if matrices is None and not (force or _has_new_data(
            coins, interval, model, learning, store=store
        )):
            outcomes.append(RetrainOutcome(
                ts_ms=now_ms, side=side, status="skipped",
                reason=(
                    "no new bars since this model was fitted; refitting would "
                    "just reseed it"
                ),
            ))
            continue
        if matrices is None:
            try:
                matrices = build_features(coins, interval, store=store)
            except Exception as exc:  # noqa: BLE001 - never stop the trader
                log.exception("could not build features for the retrain")
                return outcomes + [RetrainOutcome(
                    ts_ms=now_ms, side=side, status="failed",
                    reason=f"{type(exc).__name__}: {exc}",
                )]
        outcomes.append(retrain(
            coins=coins, interval=interval, model_path=path, incumbent=model,
            settings=settings, learning=learning, force=force, dry_run=dry_run,
            store=store, matrices=matrices, side=side,
        ))
    return outcomes


def _has_new_data(
    coins: list[str],
    interval: str,
    incumbent: TrainedModel | None,
    learning: LearningConfig,
    *,
    store: Any | None = None,
) -> bool:
    if incumbent is None:
        return True
    return new_bar_count(
        coins, interval, incumbent.trained_at_ms, store=store
    ) >= learning.min_new_bars


def retrain(
    *,
    coins: list[str],
    interval: str,
    model_path: Path | str,
    incumbent: TrainedModel | None,
    settings: Settings | None = None,
    learning: LearningConfig | None = None,
    force: bool = False,
    dry_run: bool = False,
    store: Any | None = None,
    matrices: dict[str, pd.DataFrame] | None = None,
    side: str = "up",
) -> RetrainOutcome:
    """Refit on all stored history and promote the result if it earns it.

    Never raises. A retrain that fails must leave the live trader running on
    the model it already has -- an exception here would kill a cycle and,
    through the supervisor, restart the process into the same failure.
    """
    from data.database import ParquetStore
    from models.backend import ModelParams
    from models.dataset import SplitConfig, assemble
    from models.labels import LabelConfig
    from models.train import train_walk_forward

    settings = settings or SETTINGS
    learning = learning or settings.learning
    started = time.time()
    now_ms = int(time.time() * 1000)
    store = store or ParquetStore(settings.paths)

    try:
        # Cheap first: has anything happened since the incumbent was fitted?
        # This runs before the feature build precisely because the feature
        # build is the expensive part.
        if incumbent is not None and not force:
            new_bars = new_bar_count(
                coins, interval, incumbent.trained_at_ms, store=store
            )
            if new_bars < learning.min_new_bars:
                return RetrainOutcome(
                    ts_ms=now_ms,
                    side=side,
                    status="skipped",
                    reason=(
                        f"only {new_bars} new bar(s) since the model was fitted; "
                        f"{learning.min_new_bars} needed. Nothing new to learn from "
                        "-- refitting would just reseed the same model."
                    ),
                    elapsed_s=time.time() - started,
                )

        # The caller may hand these in. Both sides of the book train on the
        # identical feature matrix and differ only in the question asked, so
        # rebuilding it per side would double the expensive half of the work.
        if matrices is None:
            matrices = build_features(coins, interval, store=store)
        label_config = incumbent.label_config if incumbent else LabelConfig()
        params = incumbent.params if incumbent else ModelParams()
        # Keep the backend stable across a retrain unless it is unavailable.
        # Silently swapping LightGBM for scikit-learn would change the live
        # model's behaviour for a reason that has nothing to do with the data.
        prefer = "sklearn_hist" if (incumbent and incumbent.backend_name == "sklearn_hist") else None

        dataset = assemble(matrices, label_config)
        fresh = new_labelled_rows(dataset, incumbent)

        log.info(
            "retraining on %d rows across %d market(s), %d newer than the "
            "incumbent", len(dataset), len(matrices), fresh,
        )
        candidate, _holdout = train_walk_forward(
            dataset,
            label_config=label_config,
            split_config=SplitConfig(),
            params=params,
            prefer_backend=prefer,
        )
        # The holdout is deliberately left unread. See the module docstring.

        warnings = candidate_warnings(candidate)
        promote, reason = decide_promotion(
            candidate, incumbent, min_auc_margin=learning.min_auc_margin
        )
        outcome = RetrainOutcome(
            ts_ms=now_ms,
            side=side,
            status="promoted" if promote else "rejected",
            reason=reason,
            rows=int(len(dataset)),
            new_rows=fresh,
            span=candidate.train_span,
            candidate_auc=candidate.mean_val_auc,
            candidate_lift=candidate.mean_log_loss_lift,
            incumbent_auc=incumbent.mean_val_auc if incumbent else float("nan"),
            backend=candidate.backend_name,
            warnings=warnings,
            elapsed_s=time.time() - started,
            model=candidate if promote else None,
        )
        if promote and dry_run:
            outcome.reason = f"{reason} (dry run: nothing was written)"
            log.info("dry run: not writing the candidate to %s", model_path)
        elif promote:
            save_atomically(candidate, model_path)
            log.info("promoted a new model to %s", model_path)
        else:
            log.warning("kept the incumbent model: %s", reason)
        for warning in warnings:
            log.warning("candidate model: %s", warning)
        return outcome

    except Exception as exc:  # noqa: BLE001 - a failed retrain must not stop trading
        log.exception("retrain failed; continuing on the existing model")
        return RetrainOutcome(
            ts_ms=now_ms,
            side=side,
            status="failed",
            reason=f"{type(exc).__name__}: {exc}",
            elapsed_s=time.time() - started,
        )


def record(outcome: RetrainOutcome, store: Any | None = None) -> None:
    """Append the attempt to the ``retrains`` history. Best effort.

    A skip is not recorded. It is the absence of an event -- nothing was
    fitted and nothing changed -- and because a skip deliberately does not
    advance the retrain clock, it is retried every cycle. Writing one row an
    hour saying "no new data" would bury the handful of rows that record the
    model actually changing.
    """
    from data.database import ParquetStore

    if outcome.status == "skipped":
        return
    try:
        (store or ParquetStore()).upsert("retrains", pd.DataFrame([outcome.as_row()]))
    except Exception as exc:  # noqa: BLE001 - history is not worth a crash
        log.warning("could not record the retrain attempt: %s", exc)
