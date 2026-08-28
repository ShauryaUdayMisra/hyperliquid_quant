# CLAUDE.md

Context for Claude Code sessions in this repo. Read before changing anything.

## What this is

A **paper-trading** quant system for Hyperliquid perpetuals. Live market
data in, simulated fills out, $100,000 of virtual money.

## Non-negotiables

1. **No real trading, ever.** Read-only public market data. No keys that can
   move funds. `config/settings.py` runs `assert_no_trading_credentials()` at
   import and refuses to start if a signing credential is in the environment.
   Never add a subcommand or code path that could place a real order.
2. **No look-ahead bias.** Features at bar *t* use only bars ≤ *t*. The one
   exception is `models/labels.py`, the only module allowed to look forward,
   and only for training targets.
3. **The risk engine has final say.** A 99%-confidence signal is rejected if a
   limit binds. Never let the strategy bypass `risk/risk_engine.py`.
4. **Realistic execution.** Costs are always on. Never disable them to make a
   backtest look better.
5. **Implausible results are bugs.** A high Sharpe or a spectacular return is
   evidence of a leak until proven otherwise. Investigate; do not celebrate.

## Layout

```
data/       collection, backfill, Parquet + DuckDB, integrity checks
features/   106 point-in-time features (momentum, volatility, volume,
            funding, order book, cross-asset, regime)
models/     labels, dataset splits, walk-forward training, inference
strategy/   probability -> position target; baselines for testing
risk/       hard limits, position sizing, liquidation buffer
execution/  paper exchange, fill simulation, live loop, state persistence
backtest/   event-driven engine, metrics, accounting proof, controls
reporting/  6-hourly HTML email, scheduler
dashboard/  read-only FastAPI view
deploy/     Dockerfile lives at ROOT; start.sh, entrypoint.sh, DEPLOY.md here
```

## The invariants that matter

**Bar ordering** (`backtest/engine.py`). Per bar: execute orders decided on
the *previous* bar against this bar's **open**; settle funding; check
liquidation against intrabar extremes; mark to market at the **close**; then
ask the strategy for orders. A signal formed at a close cannot trade on that
same close.

**Live reuses backtest code.** `execution/paper_trader.py` builds a real
`MarketView` and calls the same `ModelStrategy.on_bar`. Do not fork this
logic — divergence would make the backtest stop being evidence about live.

**Accounting identities.** `equity == cash + unrealised` at every bar, and
`final − start == realised + unrealised − fees − funding`. Verified by
`python main.py prove-accounting` and in CI via `tests/test_accounting_proof.py`.

**Causality is proven, not asserted.** `features.pipeline.assert_point_in_time`
recomputes features on truncated history and compares. A companion test
plants a deliberate leak to prove the checker works.

## Commands

```bash
.venv/bin/python -m pytest              # 357 tests (+4 live, run with -m live)
.venv/bin/python main.py status         # API reachability + safety checks
.venv/bin/python main.py prove-accounting
.venv/bin/python main.py backfill --days 400 --intervals 1h
.venv/bin/python main.py verify         # gaps, future timestamps
.venv/bin/python main.py train --interval 1h
.venv/bin/python main.py backtest --control    # --control = shuffled baseline
.venv/bin/python main.py paper          # live paper trading
.venv/bin/python main.py report --test-report
.venv/bin/python main.py dashboard
```

## Gotchas that have already cost time

- **The Dockerfile must stay at the repo root.** PaaS builders only
  auto-detect it there. In `deploy/` it was silently ignored and Railway
  built a generic Python app instead.
- **Railway ignores `railway.json`'s `startCommand`** (config-as-code is
  deprecated, sunset 2026-12-01). The image's `CMD` must be the real
  entrypoint. Migrating to `.railway/railway.ts` is still outstanding.
- **Hyperliquid caps `candleSnapshot` at ~5,000 bars**, so 1h history stops
  near 208 days no matter what `--days` says. Funding history goes back
  further.
- **Funding timestamps are not exactly hourly** — they land at e.g.
  `06:00:00.019`. Gap and alignment checks need a jitter tolerance; a strict
  grid comparison reports thousands of gaps that do not exist.
- **Backfilling 1m/5m/1h at once triggers 429s.** Keep `CANDLE_INTERVALS=1h`
  in deployment.
- **The cross-asset benchmark (BTC) has no `cross_*_btc_*` values.**
  `build_universe` aligns all coins to a union of columns and fills NaN.
  Without that alignment BTC could never produce a signal at all.
- **LightGBM needs `libomp` on macOS** (no Homebrew here), so local runs fall
  back to scikit-learn `HistGradientBoosting`. Linux/Docker uses LightGBM.
  Results are comparable in kind, not identical — the backend is recorded on
  every artefact.
- **Never let label columns reach the feature list.** `label`,
  `forward_return` and `label_known` are excluded in two independent places.
  A missing exclusion once produced a perfect AUC 1.0000.

## Conventions

- Tests are behavioural and named as sentences describing the property.
  Comments explain *why*, not what.
- New features go in `features/`, must be causal, and must survive
  `assert_point_in_time`.
- Money is float USD; time is int64 UTC epoch **milliseconds** (`ts_ms`) as
  the source of truth, with `ts` derived for convenience.
- Ratios return `NaN`, never infinity, when their denominator is degenerate.

## Current status

Phases 1-8 of the original brief are built. 357 tests pass.

**The model has no demonstrated edge.** On 208 days of real BTC/ETH/SOL
hourly data:

| | |
| --- | --- |
| Locked holdout AUC | **0.504** (coin flip) |
| Out-of-sample return, conservative | −1.08% |
| Out-of-sample return, aggressive (10x) | −41.38% |
| Buy-and-hold, same period | +17.60% |

The aggressive profile ended a full backtest at **$54.95** — from paying
$231,646 in slippage at 1,098× annual turnover, not from bad direction calls.

Treat this as the honest baseline. The productive next step is a better
label (longer horizon, or funding-carry rather than direction) and more
regularisation, judged on walk-forward validation only — **not** further
tuning against the holdout, which has already been read once.

## Deployment

Live on Railway; runbook in `deploy/DEPLOY.md`. One service runs the trader
and the dashboard together against a single volume at `/app/storage`, which
holds the market data, the model, and the account state. Without that volume
every redeploy resets the account to $100k.
