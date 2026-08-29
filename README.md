# Hyperliquid Paper-Trading Quant System

A research-grade **paper-trading** system for Hyperliquid perpetual futures.
It reads live market data, forms a probabilistic view, sizes a position, and
simulates the fill with realistic costs — spread, slippage, fees, funding,
partial fills, latency and liquidation.

It trades **$100,000 of entirely virtual money** and holds no keys capable of
moving funds. The configuration module refuses to start if a signing
credential is present in the environment.

```python
# config/settings.py — executed at import, every run
def assert_no_trading_credentials(env=None):
    found = [k for k in FORBIDDEN_ENV_VARS if (env.get(k) or "").strip()]
    if found:
        raise TradingCredentialsPresent(...)
```

**357 tests.** No look-ahead bias, verified empirically rather than asserted.

---

## The result, stated honestly

The system works. **The strategy does not.**

| Metric | Result |
| --- | --- |
| Walk-forward validation AUC | 0.540 |
| **Locked holdout AUC** | **0.504** — a coin flip |
| Out-of-sample return (conservative) | −1.08% |
| Out-of-sample return (aggressive, 10x) | −41.38% |
| Buy-and-hold over the same period | +17.60% |

Over 208 days of real BTC/ETH/SOL hourly data, the model found no
statistically meaningful edge. That is the honest finding, and this README
leads with it because a backtest that flatters itself is worthless.

The engineering below is what the project is actually about: building
infrastructure trustworthy enough that a negative result can be *believed*.

---

## What makes it trustworthy

### The accounting is proven, not assumed

`python main.py prove-accounting` runs seven scenarios where the expected
answer is computed by hand from the price path and fee schedule, entirely
independently of the engine:

```
PASS  2. Buy and hold, all costs off
      check                    hand-computed        engine        diff
      unrealised P&L              3,000.0000    3,000.0000    0.000000
PASS  6. Every cost switched on, 300 bars
      equity identity max error       0.0000        0.0000   -0.000000
      P&L attribution error           0.0000        0.0000   -0.000000
```

Two identities must hold: `equity == cash + unrealised` at *every* bar, and
`final − start == realised + unrealised − fees − funding` across the run.

### Look-ahead bias is disproven empirically

Claiming features are point-in-time is easy. `assert_point_in_time`
recomputes the entire feature matrix on *truncated* history and checks that
the value at bar *t* is bit-identical. If a feature peeked forward, hiding
the future changes it.

A companion test plants a deliberate leak to prove the checker can catch one:

```python
def leaky(bars):
    out = original(bars)
    out["mom_tomorrow"] = bars["close"].shift(-1) / bars["close"] - 1.0
    return out
# ... assert "mom_tomorrow" in leaking
```

**This caught a real bug.** An early training run scored a perfect **AUC
1.0000** — the model was reading `label` as a feature. Two independent guards
now prevent it, and the trainer raises `LabelLeakError` rather than reporting
a spectacular result.

### Results are tested against a random control

`main.py backtest --control` re-runs the identical strategy with the model's
predictions randomly permuted. Anything the strategy still earns did not come
from the model. A strategy that cannot beat its own shuffled control has
shown no edge, whatever its Sharpe ratio says.

### Implausible numbers are flagged, not celebrated

```
WARNINGS - treat these results as unproven:
  - Sharpe of 10.30 far exceeds what published systematic strategies
    achieve. This is far more likely to be a bug than an edge.
```

A separate bug produced a Sharpe of **1.0 × 10¹⁷** by dividing by
floating-point noise. It now returns `NaN` below a variance floor.

### The risk engine overrules the model

Seven checks — drawdown halt, daily-loss halt, position count, notional cap,
leverage headroom, liquidation buffer, solvency. A 99%-confidence signal is
rejected if any limit binds, and every check is recorded on the trade.

Liquidation follows Hyperliquid's real rule (maintenance margin = half the
initial margin at the asset's max leverage), checked against each bar's
*intrabar* extreme. A 10x BTC long liquidates at −8.86%.

---

## Architecture

```
data/       collection, backfill, Parquet + DuckDB, integrity checks
features/   106 point-in-time features across 7 families
models/     walk-forward training, purge + embargo, locked holdout
strategy/   probability → position target
risk/       hard limits; final say over the model
execution/  paper exchange, fill simulation, live loop, state persistence
backtest/   event-driven engine, metrics, accounting proof, controls
reporting/  6-hourly HTML email with per-market reasoning
dashboard/  read-only FastAPI view
```

**Ordering is the defence against look-ahead.** Each bar:

1. execute orders decided on bar *i−1*, against bar *i*'s **open**
2. settle funding at hour boundaries
3. check liquidation against bar *i*'s intrabar extremes
4. mark to market at bar *i*'s **close**
5. ask the strategy for orders, showing only data up to that close

A signal formed at a bar's close cannot trade on that same close.

The live loop reuses the *same* `ModelStrategy` and `MarketView` as the
backtest. Separate code paths would drift, and the backtest would stop being
evidence about the live system.

It also keeps learning. Every `RETRAIN_EVERY_HOURS` the trader refits on all
stored history — including every bar it has since been wrong about — and swaps
the new model in without a restart, keeping the incumbent if the candidate is
clearly worse or if its AUC has jumped into leak territory. And it marks its
own homework: `main.py scorecard` resolves every probability the live system
ever recorded against what the price actually did. That is the only
out-of-sample number here that cannot have been tuned, because each prediction
was written down before its outcome existed. Retraining does not manufacture
an edge — at a holdout AUC of 0.504 it cannot — but it stops the model being
frozen at the regime that happened to prevail the day it was first fit.

---

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

.venv/bin/python main.py status            # API reachability + safety checks
.venv/bin/python main.py prove-accounting  # verify the arithmetic
.venv/bin/python -m pytest                 # 446 tests

.venv/bin/python main.py backfill --days 400 --intervals 1h
.venv/bin/python main.py verify            # gaps + future timestamps
.venv/bin/python main.py train --interval 1h
.venv/bin/python main.py backtest --control
.venv/bin/python main.py scorecard         # mark the live model's own calls
```

Then `main.py paper` for live paper trading, `main.py dashboard` for the web
view. Deployment (Docker, systemd, Railway) is in [deploy/DEPLOY.md](deploy/DEPLOY.md).

Two risk profiles, switched with `RISK_PROFILE` in `.env`:

| | conservative | aggressive |
| --- | --- | --- |
| Max leverage | 2x | 10x |
| Risk per trade | 0.25% | 100% |
| Daily-loss halt | 2% | off |
| Drawdown halt | 10% | off |

The aggressive profile ended a 208-day backtest at **$54.95** — not from bad
direction calls, but from paying $231,646 in slippage at 1,098× annual
turnover. Liquidation remains enforced in both; loosening a limit disables a
halt, it does not remove it from the pipeline.

---

## Known limitations

- **Limit fills are optimistic.** Bar data cannot model queue position, so a
  limit order fills whenever the bar traded through it.
- **Liquidation is detected per-bar**, so it fires on the first breaching bar
  rather than mid-bar at the exact level.
- **Cross-margin stress uses every market's adverse extreme simultaneously** —
  deliberately pessimistic.
- **Order-book features exist only for observed periods.** Hyperliquid serves
  no order-book history, so they are NaN in historical backtests rather than
  filled with a guess.
- **Candle history caps near 5,000 bars** (~208 days at 1h).

---

## Stack

Python 3.11+, pandas, DuckDB + Parquet, LightGBM (scikit-learn
`HistGradientBoosting` fallback), FastAPI, APScheduler, httpx/websockets.

---

*Simulated paper trading. No real capital at risk. Not financial advice.*
