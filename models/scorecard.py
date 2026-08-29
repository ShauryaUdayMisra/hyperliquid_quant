"""Marking the live model's own predictions against what actually happened.

Retraining on fresh data is only half of learning from a mistake. The other
half is being able to say what the mistakes *were* -- and the deployed model
already writes down every opinion it forms: one row per market per bar in
the ``decisions`` table, probability included, whether or not it traded.

This module resolves that ledger. Each recorded probability is matched to
the bar the model was actually looking at when it spoke, and then to what
the price did over the label's horizon. The result is an out-of-sample
score of the live model on live data -- the only number here that owes
nothing to a backtest.

Three rules keep it honest:

**The bar the model saw, not the bar that existed.** A decision at 13:00:15
is made on the 12:00 bar: only completed bars are stored, so that is the
last one the feature pipeline could see. Matching on "the newest bar whose
open is before the decision" would silently hand the model the 13:00 bar --
a price from its own future -- and flatter the score. Bars are matched on
``close_ts_ms <= decision_ts``.

**The same question the model was trained on.** Outcomes come from
:func:`models.labels.make_labels` with the model's own
:class:`~models.labels.LabelConfig`, not a re-derivation. If the two
definitions drifted, the live score would be measuring a different question
from the one the model answers.

**Unfinished business stays unfinished.** A decision whose horizon has not
elapsed is counted as pending, never scored. Dropping it silently would
make the sample look complete when it is merely recent.

One caveat on reading the AUC: decisions taken at the same instant across
25 correlated markets are not 25 independent observations, so the effective
sample is smaller than the row count and the number is noisier than it
looks. It needs hundreds of hours before it says anything.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from models.labels import LabelConfig, make_labels

log = logging.getLogger(__name__)


def _json_safe(value: Any) -> Any:
    """Coerce pandas and numpy values into things ``json.dumps`` accepts.

    NaN is the one that matters: it is a perfectly ordinary metric here (an
    AUC with one class present, a lift with no rows) and it is not JSON.
    Interval objects from the calibration buckets are the other.
    """
    if isinstance(value, pd.DataFrame):
        return [_json_safe(record) for record in value.to_dict("records")]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, pd.Interval):
        return f"{value.left:.3f}-{value.right:.3f}"
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None or isinstance(value, str):
        return value
    return str(value)


@dataclass
class LiveScorecard:
    """How the deployed model has actually done, on its own recorded calls."""

    label_question: str
    entry_threshold: float
    resolved: int = 0
    pending: int = 0
    unmatched: int = 0
    first_ts_ms: int | None = None
    last_ts_ms: int | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    calibration: pd.DataFrame = field(default_factory=pd.DataFrame)
    by_coin: pd.DataFrame = field(default_factory=pd.DataFrame)
    #: The same metrics restricted to the calls the model was confident
    #: enough to act on. "Was it right when it said go" is a different and
    #: more interesting question than "was it right on average".
    acted: dict[str, float] = field(default_factory=dict)

    @property
    def auc(self) -> float:
        return float(self.metrics.get("auc", float("nan")))

    @property
    def has_verdict(self) -> bool:
        """Enough resolved rows for the numbers to mean anything at all."""
        return self.resolved >= 200 and np.isfinite(self.auc)

    def describe(self) -> str:
        if not self.resolved:
            return (
                f"live scorecard: nothing resolved yet "
                f"({self.pending} decision(s) still inside their horizon)"
            )
        def number(key: str, spec: str) -> str:
            value = self.metrics.get(key, float("nan"))
            return format(value, spec) if np.isfinite(value) else "not computable yet"

        lines = [
            f"LIVE SCORECARD - {self.label_question}",
            f"  resolved calls    : {self.resolved:,} "
            f"({self.pending:,} pending, {self.unmatched:,} unmatched)",
        ]
        if np.isfinite(self.auc):
            lines.append(f"  live AUC          : {self.auc:.4f}  (0.50 = coin flip)")
        else:
            # AUC needs both outcomes present. Early on, every resolved call
            # can share one -- printing "nan" invites the reader to think
            # something is broken when nothing is.
            lines.append(
                "  live AUC          : not computable yet - every resolved call so "
                "far had the same outcome"
            )
        lines += [
            f"  log-loss lift     : {number('log_loss_lift', '+.5f')}"
            f"  (vs always predicting the base rate)",
            f"  brier             : {number('brier', '.4f')}",
            f"  mean P(up) said   : {number('mean_prediction', '.3f')}"
            f"   actually up: {number('base_rate', '.3f')}",
        ]
        if self.acted.get("rows"):
            lines.append(
                f"  when it said go   : {int(self.acted['rows']):,} call(s) at "
                f"P(up) >= {self.entry_threshold:.2f}, right "
                f"{self.acted.get('hit_rate', float('nan')):.1%} of the time "
                f"(base rate {self.metrics.get('base_rate', float('nan')):.1%}, "
                f"edge {self.acted.get('edge_over_base', float('nan')):+.1%})"
            )
        else:
            lines.append(
                f"  when it said go   : no resolved call ever reached "
                f"P(up) >= {self.entry_threshold:.2f}"
            )
        if not self.has_verdict:
            lines.append(
                "  NOT YET A VERDICT - too few resolved calls to distinguish "
                "this from chance."
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """A JSON-safe summary, for the dashboard and the retrain history.

        Everything here is coerced through :func:`_json_safe` on the way out.
        A NaN metric or a pandas Interval left in the calibration table
        serialises to invalid JSON, and one script renders the whole
        dashboard -- so a single unserialisable value blanks every panel
        while the page and all its APIs still answer 200.
        """
        return {
            "label_question": self.label_question,
            "entry_threshold": self.entry_threshold,
            "resolved": self.resolved,
            "pending": self.pending,
            "unmatched": self.unmatched,
            "first_ts_ms": self.first_ts_ms,
            "last_ts_ms": self.last_ts_ms,
            "has_verdict": self.has_verdict,
            "metrics": _json_safe(self.metrics),
            "acted": _json_safe(self.acted),
            "calibration": _json_safe(self.calibration),
            "by_coin": _json_safe(self.by_coin),
        }


def _resolve(
    decisions: pd.DataFrame,
    bars: dict[str, pd.DataFrame],
    *,
    label_config: LabelConfig,
    interval_ms: int,
) -> pd.DataFrame:
    """Attach the realised outcome to every recorded probability.

    Returns the decisions frame with ``label``, ``forward_return`` and
    ``label_known`` columns. Rows whose bar could not be found are dropped
    and counted by the caller; rows still inside the horizon come back with
    ``label_known`` False.
    """
    resolved: list[pd.DataFrame] = []
    for coin, frame in bars.items():
        calls = decisions.loc[decisions["coin"] == coin]
        if calls.empty or frame.empty:
            continue

        # The same labels the trainer would have made from these bars.
        labelled = make_labels(
            frame[["ts_ms", "close"]].sort_values("ts_ms").reset_index(drop=True),
            label_config,
        )
        # A bar is only visible to a decision once it has closed. ts_ms is
        # the bar's OPEN, so the bar the model saw is the newest one whose
        # close is already in the past at decision time.
        labelled["visible_ms"] = labelled["ts_ms"] + interval_ms

        merged = pd.merge_asof(
            calls.sort_values("ts_ms"),
            labelled[["visible_ms", "ts_ms", "close", "label", "forward_return", "label_known"]]
            .rename(columns={"ts_ms": "bar_ts_ms"}),
            left_on="ts_ms",
            right_on="visible_ms",
            direction="backward",
        )
        resolved.append(merged)

    if not resolved:
        return pd.DataFrame(columns=[*decisions.columns, "label", "label_known"])
    return pd.concat(resolved, ignore_index=True)


def score_decisions(
    decisions: pd.DataFrame,
    bars: dict[str, pd.DataFrame],
    *,
    label_config: LabelConfig | None = None,
    interval_ms: int,
    entry_threshold: float,
) -> LiveScorecard:
    """Mark a decision ledger against realised prices."""
    from models.train import calibration_table, classification_metrics

    label_config = label_config or LabelConfig()
    card = LiveScorecard(
        label_question=label_config.name, entry_threshold=float(entry_threshold)
    )
    if decisions.empty or not bars:
        return card

    needed = {"ts_ms", "coin", "probability"}
    missing = needed - set(decisions.columns)
    if missing:
        raise ValueError(f"decision ledger is missing {sorted(missing)}")

    merged = _resolve(
        decisions.dropna(subset=["probability"]),
        bars,
        label_config=label_config,
        interval_ms=interval_ms,
    )
    if merged.empty:
        card.unmatched = int(len(decisions))
        return card

    card.unmatched = int(len(decisions) - len(merged) + merged["label_known"].isna().sum())
    known = merged.loc[merged["label_known"].fillna(False)].copy()
    card.pending = int(len(merged) - len(known))
    card.resolved = int(len(known))
    if known.empty:
        return card

    card.first_ts_ms = int(known["ts_ms"].min())
    card.last_ts_ms = int(known["ts_ms"].max())

    y = known["label"].to_numpy(dtype=int)
    p = known["probability"].to_numpy(dtype=float)
    card.metrics = classification_metrics(y, p)
    card.calibration = calibration_table(y, p)

    # Per market, because a single bad market is worth seeing separately
    # from a model that is uniformly at chance.
    grouped = known.groupby("coin", observed=True)
    card.by_coin = pd.DataFrame(
        {
            "calls": grouped.size(),
            "mean_predicted": grouped["probability"].mean(),
            "actual_rate": grouped["label"].mean(),
            "mean_forward_return": grouped["forward_return"].mean(),
        }
    ).reset_index().sort_values("calls", ascending=False)

    acted = known.loc[known["probability"] >= entry_threshold]
    if not acted.empty:
        base = float(known["label"].mean())
        card.acted = {
            "rows": int(len(acted)),
            "hit_rate": float(acted["label"].mean()),
            "edge_over_base": float(acted["label"].mean() - base),
            "mean_forward_return": float(acted["forward_return"].mean()),
        }
    return card


def load_scorecard(
    *,
    coins: list[str],
    interval: str,
    interval_ms: int,
    label_config: LabelConfig | None = None,
    entry_threshold: float,
    store: Any | None = None,
    since_ms: int | None = None,
) -> LiveScorecard:
    """Read the stored decision ledger and mark it. Never raises on absence."""
    from data.database import MarketDatabase, ParquetStore

    store = store or ParquetStore()
    label_config = label_config or LabelConfig()
    card = LiveScorecard(
        label_question=label_config.name, entry_threshold=float(entry_threshold)
    )
    if not store.has_data("decisions") or not store.has_data("candles"):
        return card

    with MarketDatabase(store) as db:
        sql = "SELECT ts_ms, coin, probability, action FROM decisions"
        params: list[Any] = []
        if since_ms is not None:
            sql += " WHERE ts_ms >= ?"
            params.append(since_ms)
        decisions = db.query(sql + " ORDER BY ts_ms", params)

        bars: dict[str, pd.DataFrame] = {}
        for coin in sorted(set(decisions["coin"]) if not decisions.empty else coins):
            frame = db.query(
                "SELECT ts_ms, close FROM candles WHERE coin = ? AND interval = ? "
                "ORDER BY ts_ms",
                [coin, interval],
            )
            if not frame.empty:
                bars[coin] = frame

    if decisions.empty or not bars:
        return card
    return score_decisions(
        decisions,
        bars,
        label_config=label_config,
        interval_ms=interval_ms,
        entry_threshold=entry_threshold,
    )
