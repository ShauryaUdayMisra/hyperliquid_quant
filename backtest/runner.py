"""Phase 8: run the full system over history and report honestly.

This is the module that answers "does it work?". It deliberately makes the
answer hard to fake:

* the model is trained on the development period only, and the backtest is
  reported both overall and on the locked out-of-sample window,
* costs are always on,
* results are compared against buy-and-hold, because a strategy that
  underperforms holding the asset has not earned its complexity,
* :func:`implausibility_warnings` flags returns that are too good, and the
  printed report leads with them rather than burying them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from backtest.engine import BacktestConfig, BacktestEngine, BacktestResult
from backtest.metrics import PerformanceMetrics, compute_metrics, metrics_by_regime
from config.settings import INTERVAL_MS, SETTINGS, RiskLimits, Settings
from execution.paper_exchange import PaperExchange
from execution.simulator import FillSimulator
from features.pipeline import FeatureConfig, build_universe
from models.predict import SignalGenerator
from models.train import TrainedModel
from risk.risk_engine import RiskEngine
from strategy.baselines import AlwaysLongStrategy
from strategy.signals import ModelStrategy

log = logging.getLogger(__name__)

#: Annualised returns above this on a paper crypto backtest are a bug until
#: proven otherwise. The brief is explicit: do not celebrate, investigate.
SUSPICIOUS_ANNUAL_RETURN = 3.0
SUSPICIOUS_SHARPE = 3.0


@dataclass
class BacktestReport:
    strategy_name: str
    profile: str
    metrics: PerformanceMetrics
    regime_table: pd.DataFrame
    benchmark: PerformanceMetrics | None = None
    result: BacktestResult | None = None
    out_of_sample_metrics: PerformanceMetrics | None = None
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [
            "BACKTEST REPORT",
            f"  strategy      : {self.strategy_name}",
            f"  risk profile  : {self.profile}",
            "",
            self.metrics.describe(),
        ]

        if self.benchmark is not None:
            lines += [
                "",
                "  Versus buy-and-hold over the same period:",
                f"    strategy      {self.metrics.total_return:+.2%} "
                f"(Sharpe {self.metrics.sharpe:.2f}, max DD {self.metrics.max_drawdown:.2%})",
                f"    buy and hold  {self.benchmark.total_return:+.2%} "
                f"(Sharpe {self.benchmark.sharpe:.2f}, max DD {self.benchmark.max_drawdown:.2%})",
                f"    difference    {self.metrics.total_return - self.benchmark.total_return:+.2%}",
            ]

        if self.out_of_sample_metrics is not None:
            lines += [
                "",
                "  Out-of-sample window only (the model never saw this period):",
                f"    return {self.out_of_sample_metrics.total_return:+.2%} | "
                f"Sharpe {self.out_of_sample_metrics.sharpe:.2f} | "
                f"max DD {self.out_of_sample_metrics.max_drawdown:.2%} | "
                f"trades {self.out_of_sample_metrics.trades}",
            ]

        if len(self.regime_table):
            lines += ["", "  Performance by market regime:", ""]
            lines.append(
                f"    {'regime':<18}{'bars':>8}{'share':>9}{'return':>11}{'Sharpe':>9}{'max DD':>9}"
            )
            for _, row in self.regime_table.iterrows():
                sharpe = row["sharpe"]
                lines.append(
                    f"    {row['regime']:<18}{row['bars']:>8,}{row['share']:>8.1%}"
                    f"{row['total_return']:>11.2%}"
                    f"{(f'{sharpe:.2f}' if np.isfinite(sharpe) else 'n/a'):>9}"
                    f"{row['max_dd']:>9.2%}"
                )

        lines.append("")
        if self.warnings:
            lines.append("  WARNINGS - treat these results as unproven:")
            lines += [f"    - {w}" for w in self.warnings]
        else:
            lines.append("  No implausibility warnings raised.")
        if self.notes:
            lines += ["", "  Notes:"] + [f"    - {n}" for n in self.notes]
        return "\n".join(lines)


def implausibility_warnings(
    metrics: PerformanceMetrics, result: BacktestResult
) -> list[str]:
    """Reasons to disbelieve a good-looking backtest."""
    warnings: list[str] = []

    if np.isfinite(metrics.cagr) and metrics.cagr > SUSPICIOUS_ANNUAL_RETURN:
        warnings.append(
            f"CAGR of {metrics.cagr:.0%} is implausible for a paper crypto strategy. "
            "Assume look-ahead bias, unrealistic fills or overfitting until each is "
            "ruled out. Re-run the point-in-time check and the accounting proof."
        )
    if np.isfinite(metrics.sharpe) and metrics.sharpe > SUSPICIOUS_SHARPE:
        warnings.append(
            f"Sharpe of {metrics.sharpe:.2f} far exceeds what published systematic "
            "strategies achieve. This is far more likely to be a bug than an edge."
        )
    if metrics.trades < 30:
        warnings.append(
            f"Only {metrics.trades} closed trades. That is too small a sample to "
            "distinguish skill from luck, whatever the headline number says."
        )
    if metrics.total_fees + metrics.total_slippage == 0 and metrics.trades > 0:
        warnings.append(
            "Zero fees and zero slippage were charged. The cost model was disabled, "
            "so these results are not achievable."
        )
    if metrics.max_drawdown < 0.01 and metrics.trades > 10:
        warnings.append(
            f"A max drawdown of {metrics.max_drawdown:.2%} across {metrics.trades} "
            "trades is not realistic; check that positions were actually held."
        )
    if metrics.exposure < 0.02 and metrics.trades > 0:
        warnings.append(
            f"The strategy was in the market only {metrics.exposure:.1%} of the time; "
            "annualised statistics from so little exposure are unreliable."
        )
    if result.exchange.bankrupt:
        warnings.append("The account was wiped out. Any positive statistic before that point is moot.")
    return warnings


def run_backtest(
    bars_by_coin: dict[str, pd.DataFrame],
    model: TrainedModel,
    *,
    funding_by_coin: dict[str, pd.DataFrame] | None = None,
    book_by_coin: dict[str, pd.DataFrame] | None = None,
    settings: Settings | None = None,
    limits: RiskLimits | None = None,
    interval: str = "1h",
    long_threshold: float = 0.55,
    warmup_bars: int = 800,
    out_of_sample_from_ms: int | None = None,
    asset_max_leverage: dict[str, float] | None = None,
) -> BacktestReport:
    """Run the model strategy over history and produce an honest report."""
    settings = settings or SETTINGS
    limits = limits or settings.risk
    interval_ms = INTERVAL_MS[interval]

    matrices = build_universe(
        bars_by_coin,
        funding_by_coin=funding_by_coin,
        book_by_coin=book_by_coin,
        config=FeatureConfig(interval=interval),
    )

    generator = SignalGenerator(model, long_threshold=long_threshold)
    risk = RiskEngine(limits, settings.execution)
    strategy = ModelStrategy(
        generator, risk, matrices,
        max_hold_ms=settings.strategy.max_hold_ms,
        max_idle_ms=settings.strategy.max_idle_ms,
    )

    exchange = PaperExchange(
        limits.starting_capital,
        config=settings.execution,
        simulator=FillSimulator(settings.execution),
    )
    engine = BacktestEngine(
        bars_by_coin,
        BacktestConfig(
            interval=interval,
            warmup_bars=warmup_bars,
            asset_max_leverage=asset_max_leverage or {},
        ),
        risk=limits,
        exchange=exchange,
        features=matrices,
    )
    result = engine.run(strategy)

    metrics = compute_metrics(
        result.equity_curve, result.trades, result.fills,
        interval_ms=interval_ms,
        starting_equity=limits.starting_capital,
        liquidations=exchange.liquidation_count,
        bankrupt=exchange.bankrupt,
    )

    # Regime labels come from the benchmark market's own feature matrix and
    # are point-in-time, so this breakdown does not use hindsight.
    benchmark_coin = "BTC" if "BTC" in matrices else next(iter(matrices))
    regimes = matrices[benchmark_coin]["regime"].reset_index(drop=True)
    regimes = regimes.iloc[: len(result.equity_curve)]
    regime_table = metrics_by_regime(result.equity_curve, regimes, interval_ms=interval_ms)

    report = BacktestReport(
        strategy_name=result.strategy_name,
        profile=limits.name,
        metrics=metrics,
        regime_table=regime_table,
        benchmark=_buy_and_hold(bars_by_coin, settings, limits, interval, warmup_bars),
        result=result,
        warnings=implausibility_warnings(metrics, result),
    )

    if out_of_sample_from_ms is not None:
        report.out_of_sample_metrics = _out_of_sample(
            result, out_of_sample_from_ms, interval_ms, exchange
        )
        report.notes.append(
            "The out-of-sample window begins where the model's training data ends."
        )

    report.notes.append(
        f"Costs charged: {settings.execution.taker_fee:.3%} taker fee, "
        f"{settings.execution.default_half_spread:.4%} half-spread, square-root impact, "
        f"{settings.execution.latency_ms}ms latency, funding, and liquidation."
    )
    if model.backend_name != "lightgbm":
        report.notes.append(
            f"Model backend was '{model.backend_name}', not LightGBM. Results are "
            "comparable in kind but not identical to a LightGBM run."
        )
    return report


def run_shuffled_control(
    bars_by_coin: dict[str, pd.DataFrame],
    model: TrainedModel,
    *,
    seed: int = 0,
    **kwargs: Any,
) -> BacktestReport:
    """Re-run the backtest with the model's probabilities randomly permuted.

    The single most useful diagnostic in the whole system. The permutation
    keeps the distribution of predictions identical but destroys the mapping
    between a prediction and its bar. Anything the strategy still earns
    afterwards did not come from the model -- it came from long bias,
    position sizing, or an accident of the period.

    A strategy that does not clearly beat its own shuffled control has not
    demonstrated an edge, whatever its Sharpe ratio says.
    """
    rng = np.random.default_rng(seed)
    original = model.backend.predict_proba

    def shuffled(X):
        return rng.permutation(original(X))

    model.backend.predict_proba = shuffled
    try:
        report = run_backtest(bars_by_coin, model, **kwargs)
    finally:
        # Always restore, even if the run raises: a silently shuffled model
        # would poison every later result in the process. Deleting the
        # instance attribute (rather than reassigning it) puts the object
        # back exactly as it was, with the class method unshadowed.
        del model.backend.predict_proba

    report.strategy_name = "shuffled control"
    report.notes.append(
        "Predictions were randomly permuted. Treat this as the floor the real "
        "strategy must clear to have shown any edge."
    )
    return report


def _out_of_sample(result, from_ms, interval_ms, exchange) -> PerformanceMetrics | None:
    curve = result.equity_curve
    window = curve.loc[curve["ts_ms"] >= from_ms]
    if len(window) < 2:
        return None
    trades = [t for t in result.trades if t.closed_ts_ms >= from_ms]
    fills = [f for f in result.fills if f.ts_ms >= from_ms]
    return compute_metrics(
        window, trades, fills,
        interval_ms=interval_ms,
        starting_equity=float(window["equity"].iloc[0]),
        liquidations=exchange.liquidation_count,
        bankrupt=exchange.bankrupt,
    )


def _buy_and_hold(bars_by_coin, settings, limits, interval, warmup_bars) -> PerformanceMetrics:
    """Equal-weight buy-and-hold, charged the same costs."""
    exchange = PaperExchange(
        limits.starting_capital,
        config=settings.execution,
        simulator=FillSimulator(settings.execution),
    )
    engine = BacktestEngine(
        bars_by_coin,
        BacktestConfig(interval=interval, warmup_bars=warmup_bars),
        risk=limits,
        exchange=exchange,
    )
    per_coin = limits.starting_capital / max(1, len(bars_by_coin))
    result = engine.run(AlwaysLongStrategy(notional_per_coin=per_coin))
    return compute_metrics(
        result.equity_curve, result.trades, result.fills,
        interval_ms=INTERVAL_MS[interval],
        starting_equity=limits.starting_capital,
    )
