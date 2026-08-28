"""The model-driven strategy.

Flow for every bar:

    features -> model probability -> direction -> portfolio slot ->
    risk engine -> position target -> order

The strategy proposes; the risk engine disposes. Nothing here can bypass
:class:`~risk.risk_engine.RiskEngine`, and every order carries the full
:class:`~execution.paper_exchange.DecisionContext` -- the features, the
probability, the regime and each risk check -- so any trade in the ledger
can be explained months later.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from backtest.engine import MarketView
from execution.paper_exchange import DecisionContext, Order
from models.predict import Signal, SignalGenerator
from risk.risk_engine import RiskDecision, RiskEngine, Verdict
from strategy.base import BaseStrategy
from strategy.portfolio import Candidate, rank_candidates

log = logging.getLogger(__name__)


@dataclass
class DecisionRecord:
    """One bar's full reasoning for one market, kept for the email report."""

    ts_ms: int
    coin: str
    signal: Signal
    risk: RiskDecision
    target_notional: float
    current_notional: float
    action: str
    atr_fraction: float | None = None

    def describe(self) -> str:
        return (
            f"{self.coin}: {self.action} | {self.signal.describe()} | "
            f"risk {self.risk.summary()}"
        )


class ModelStrategy(BaseStrategy):
    """Turns model probabilities into risk-approved position targets."""

    name = "model_strategy"

    def __init__(
        self,
        generator: SignalGenerator,
        risk_engine: RiskEngine,
        features_by_coin: dict[str, pd.DataFrame],
        *,
        exit_threshold: float | None = None,
        min_rebalance_fraction: float = 0.25,
        atr_column: str = "vol_atr_14",
        precompute: bool = True,
    ) -> None:
        self.generator = generator
        self.risk = risk_engine
        self.features = features_by_coin
        #: Below this probability an open long is closed. Defaults to a band
        #: just under the entry threshold, so a position is not opened and
        #: closed on noise straddling a single number.
        self.exit_threshold = (
            exit_threshold
            if exit_threshold is not None
            else generator.long_threshold - 0.05
        )
        #: Ignore rebalances smaller than this share of the current position.
        self.min_rebalance_fraction = min_rebalance_fraction
        self.atr_column = atr_column
        self.decisions: list[DecisionRecord] = []
        self._signal_cache: dict[str, list[Signal | None]] = {}
        if precompute:
            self._precompute_signals()

    def _precompute_signals(self) -> None:
        """Score every bar in one batched pass instead of one row at a time.

        This is equivalent to per-bar inference, not a shortcut around it:
        the model is frozen before the backtest starts, and each row's
        probability depends only on that row's (already point-in-time)
        features. Batching changes the arithmetic not at all -- only the
        number of Python-level calls, which dominates runtime.

        It is a backtest optimisation only. The live loop scores one bar at
        a time because that is all it has.
        """
        for coin, frame in self.features.items():
            required = self.generator.model.features
            missing = [c for c in required if c not in frame.columns]
            if missing:
                self._signal_cache[coin] = [None] * len(frame)
                continue

            populated = frame[required].notna().mean(axis=1).to_numpy()
            usable = populated >= 0.7
            cache: list[Signal | None] = [None] * len(frame)
            if usable.any():
                subset = frame.loc[usable]
                for position, signal in zip(np.flatnonzero(usable),
                                            self.generator.generate(subset, coin=coin)):
                    cache[int(position)] = signal
            self._signal_cache[coin] = cache
            log.debug("precomputed %d signals for %s", int(usable.sum()), coin)

    # -- helpers -----------------------------------------------------------

    def _feature_row(self, coin: str, index: int) -> pd.Series | None:
        frame = self.features.get(coin)
        if frame is None or index >= len(frame):
            return None
        return frame.iloc[index]

    def _atr_fraction(self, row: pd.Series | None) -> float | None:
        if row is None or self.atr_column not in row:
            return None
        value = row[self.atr_column]
        return float(value) if np.isfinite(value) else None

    def _signal_for(self, coin: str, index: int) -> Signal | None:
        cached = self._signal_cache.get(coin)
        if cached is not None:
            return cached[index] if index < len(cached) else None

        frame = self.features.get(coin)
        if frame is None or index >= len(frame):
            return None
        row = frame.iloc[[index]]
        missing = [c for c in self.generator.model.features if c not in row.columns]
        if missing:
            return None
        # A row that is still warming up cannot produce a trustworthy
        # probability, so no signal is generated at all.
        populated = row[self.generator.model.features].notna().mean(axis=1).iloc[0]
        if populated < 0.7:
            return None
        return self.generator.generate(row, coin=coin)[0]

    # -- the bar -----------------------------------------------------------

    def on_bar(self, view: MarketView) -> Sequence[Order]:
        equity = view.equity()
        self.risk.observe_equity(view.ts_ms, equity)

        candidates: list[Candidate] = []
        for coin in view.coins:
            signal = self._signal_for(coin, view.bar_index)
            if signal is None:
                continue
            candidates.append(
                Candidate(
                    signal=signal,
                    currently_held=abs(view.position_size(coin)) > 1e-12,
                    atr_fraction=self._atr_fraction(self._feature_row(coin, view.bar_index)),
                )
            )

        if not candidates:
            return []

        selected, skipped = rank_candidates(candidates, self.risk.limits.max_open_positions)

        # Markets the model is neutral on are not traded, but they are still
        # reported. The brief requires a line per market every window, and
        # "the model has no opinion here" is information, not an absence of
        # it. Without this, a fully flat window produces an empty report.
        for candidate in candidates:
            if candidate in selected or candidate in skipped:
                continue
            self.decisions.append(
                DecisionRecord(
                    ts_ms=view.ts_ms,
                    coin=candidate.signal.coin,
                    signal=candidate.signal,
                    risk=RiskDecision(Verdict.APPROVED, 0.0, 0.0, []),
                    target_notional=0.0,
                    current_notional=0.0,
                    action="flat: no signal",
                    atr_fraction=candidate.atr_fraction,
                )
            )

        for candidate in skipped:
            self.decisions.append(
                DecisionRecord(
                    ts_ms=view.ts_ms,
                    coin=candidate.signal.coin,
                    signal=candidate.signal,
                    risk=RiskDecision(Verdict.REJECTED, 0.0, 0.0, []),
                    target_notional=0.0,
                    current_notional=0.0,
                    action="skipped: no free position slot",
                    atr_fraction=candidate.atr_fraction,
                )
            )

        marks = view.marks()
        existing = {
            coin: abs(view.position_size(coin)) * marks[coin]
            for coin in view.coins
            if abs(view.position_size(coin)) > 1e-12
        }
        state = self.risk.account_state(
            ts_ms=view.ts_ms,
            equity=equity,
            gross_notional=sum(existing.values()),
            open_positions=len(existing),
            existing_notional=existing,
        )

        orders: list[Order] = []
        for candidate in selected:
            orders.extend(self._orders_for(view, candidate, state))
        return orders

    def _orders_for(self, view: MarketView, candidate: Candidate, state) -> list[Order]:
        signal = candidate.signal
        coin = signal.coin
        price = view.price(coin)
        current_size = view.position_size(coin)
        current_notional = abs(current_size) * price

        # -- decide the desired direction --
        if current_size > 0 and signal.probability < self.exit_threshold:
            desired = 0.0
            action = f"close long: P(up) {signal.probability:.3f} below exit {self.exit_threshold:.2f}"
        elif current_size < 0 and (signal.down_probability or 0.0) < self.exit_threshold:
            desired = 0.0
            action = "close short: down-signal faded"
        elif signal.direction == "long":
            desired = 1.0
            action = "long"
        elif signal.direction == "short":
            desired = -1.0
            action = "short"
        elif abs(current_size) > 1e-12:
            desired = np.sign(current_size)
            action = "hold"
        else:
            self.decisions.append(DecisionRecord(
                view.ts_ms, coin, signal,
                RiskDecision(Verdict.APPROVED, 0.0, 0.0, []),
                0.0, current_notional, "flat: no signal",
                candidate.atr_fraction,
            ))
            return []

        # -- closing needs no risk approval; reducing risk is always allowed --
        if desired == 0.0:
            self.decisions.append(DecisionRecord(
                view.ts_ms, coin, signal,
                RiskDecision(Verdict.APPROVED, 0.0, 0.0, []),
                0.0, current_notional, action, candidate.atr_fraction,
            ))
            return self.orders_to_reach(view, coin, 0.0, DecisionContext(
                reason=action,
                model_probability=signal.probability,
                model_confidence=signal.confidence,
                regime=signal.regime,
                features={f.name: f.contribution for f in signal.top_features},
                risk_checks=["exit: risk engine not consulted for reducing exposure"],
                target_notional=0.0,
            ))

        # -- everything that adds exposure goes through the risk engine --
        decision = self.risk.evaluate(
            coin=coin,
            state=state,
            confidence=signal.confidence,
            atr_fraction=candidate.atr_fraction,
            is_new_position=abs(current_size) <= 1e-12,
        )

        context = DecisionContext(
            reason=f"{action}: {signal.label_question} = {signal.probability:.3f}",
            model_probability=signal.probability,
            model_confidence=signal.confidence,
            regime=signal.regime,
            features={f.name: f.contribution for f in signal.top_features},
            risk_checks=[c.describe() for c in decision.checks],
            target_notional=decision.approved_notional,
        )

        self.decisions.append(DecisionRecord(
            view.ts_ms, coin, signal, decision,
            decision.approved_notional, current_notional,
            action if decision.approved else f"vetoed: {decision.summary()}",
            candidate.atr_fraction,
        ))

        if not decision.approved:
            return []

        target_size = desired * decision.approved_notional / price
        # Do not churn for a marginal adjustment; the costs outweigh it.
        if current_notional > 0:
            change = abs(target_size - current_size) * price
            if change < current_notional * self.min_rebalance_fraction:
                return []

        return self.orders_to_reach(view, coin, target_size, context)

    # -- reporting ---------------------------------------------------------

    def recent_decisions(self, since_ts_ms: int) -> list[DecisionRecord]:
        return [d for d in self.decisions if d.ts_ms >= since_ts_ms]

    def latest_by_coin(self) -> dict[str, DecisionRecord]:
        out: dict[str, DecisionRecord] = {}
        for record in self.decisions:
            out[record.coin] = record
        return out
