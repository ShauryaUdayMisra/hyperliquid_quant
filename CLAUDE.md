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
   limit binds. Never let the strategy bypass `risk/risk_engine.py`. The
   *limits themselves* are the owner's to set, and as of 2026-08-28 he has
   set them wide open — see "How it is configured live" below. Structure
   stays; numbers are his call.
4. **Realistic execution.** Costs are always on. Never disable them to make a
   backtest look better.
5. **Implausible results are bugs.** A high Sharpe or a spectacular return is
   evidence of a leak until proven otherwise. Investigate; do not celebrate.

## Layout

```
data/       collection, backfill, Parquet + DuckDB, integrity checks
features/   106 point-in-time features (momentum, volatility, volume,
            funding, order book, cross-asset, regime)
models/     labels, dataset splits, walk-forward training, inference,
            scheduled retraining, live scorecard
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

**The model has to keep learning, and it has to be marked.** Trained once at
first boot, it never sees a bar of the market it trades: the same setup can go
against it every day and the next prediction is identical to the first.
`models/retrain.py` refits on all stored history every `RETRAIN_EVERY_HOURS`
and `PaperTrader._swap_model` puts the result behind the live strategy without
a restart. The retrain clock is `trained_at_ms`, not a counter, for the same
reason the idle clock is. Three things it deliberately does *not* do: it never
changes the question (label config, params and backend are inherited from the
incumbent, so probabilities keep meaning the same thing), it never reads the
locked holdout (that would turn a one-shot test set into a tuning set within a
week), and it never promotes a candidate whose AUC has jumped to leak territory.
Retraining does not manufacture an edge — at AUC 0.504 refitting the same
features on more of the same data will not change that. It buys currency with
the regime, and it turns mistakes into training rows.

`models/scorecard.py` is the other half, and the more important one:
`main.py scorecard` marks every probability the live system ever recorded
against what the price actually did. It is the only out-of-sample number in the
project that cannot have been tuned, because each prediction was written down
before its outcome existed. Two traps it avoids — a call at 13:00:15 is marked
against the **12:00** bar (only completed bars are stored, so matching on the
newest bar *open* would hand the model a price from its own future), and a call
whose horizon has not elapsed is pending, never scored.

**The activity rules can fight each other.** A forced entry is opened
*below* the entry threshold by definition, so on the very next bar the same
probability reads as "below the exit threshold" and closes it. That made
72.5% of round trips last exactly one hour at a 17.9% win rate: the spread
paid twice an hour for nothing, −99.91% over the backtest. `MIN_HOLD_HOURS`
is the floor that stops it. Measured, 3 markets, aggressive, $10k positions:
0h → 1,977 trades and −99.91%; 2h → 1,398 and −83.01%; 4h → 1,056 and
−60.35%; 8h → 721 and −42.66%. Turnover and ruin move together — pick the
point on that curve deliberately, and never remove the floor entirely.

**Activity rules sit outside the model** (`strategy/signals.py`,
`config.settings.StrategyConfig`). `MAX_HOLD_HOURS` force-closes any position
past its age regardless of how good the probability still looks; re-entry on
a later bar is allowed. `MAX_IDLE_HOURS` opens the strongest candidate after
that long holding nothing, even though it never cleared the entry threshold. Both buy activity, not accuracy — the forced trades are by
construction the ones the model was not confident enough to ask for. Both
clocks are timestamps, never counters: position age comes from
`Position.opened_ts_ms` and the idle clock is persisted in the state file's
`extra` blob and restored on boot. Counted in memory, each redeploy reset them
— and a service that redeploys more often than the idle limit could never fire
the timer at all.

**A position you cannot close is unbounded risk.** `orders_to_reach` refuses
trades under `min_trade_notional` to stop churn. Left alone, that guard also
refuses to close a sub-$10 stub, so a partial exit could strand a position
permanently — immune to the exit signal *and* the holding cap, which fired on
one such position every bar for 562 hours. Reductions that would leave less
than the minimum now go to zero, and full exits are never suppressed for size.

**Accounting identities.** `equity == cash + unrealised` at every bar, and
`final − start == realised + unrealised − fees − funding`. Verified by
`python main.py prove-accounting` and in CI via `tests/test_accounting_proof.py`.

**Causality is proven, not asserted.** `features.pipeline.assert_point_in_time`
recomputes features on truncated history and compares. A companion test
plants a deliberate leak to prove the checker works.

## Commands

```bash
.venv/bin/python -m pytest              # 451 tests (+4 live, run with -m live)
.venv/bin/python main.py status         # API reachability + safety checks
.venv/bin/python main.py prove-accounting
.venv/bin/python main.py backfill --days 400 --intervals 1h
.venv/bin/python main.py verify         # gaps, future timestamps
.venv/bin/python main.py train --interval 1h
.venv/bin/python main.py retrain           # refit the live model on fresh data
.venv/bin/python main.py scorecard         # mark the live model's own calls
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
- **The web server must bind before any slow startup work.** The platform
  starts its health check when the container starts, so a backfill that runs
  first and outlasts `healthcheckTimeout` kills a container that is working.
  At `MARKETS=top:25` the cold-volume backfill does exactly that. `start.sh`
  now starts uvicorn in the background first and waits on it last.
- **Do not normalise coin names from the exchange.** Hyperliquid lists
  `kPEPE`, `kBONK` and friends in mixed case. Upper-casing them in
  `data/universe.py` produced `KPEPE`, which no later API call could find:
  it could never be refreshed, never had bars, and took the whole cycle
  down. Explicit `MARKETS` values are still upper-cased for typing
  convenience; exchange-derived names are passed through untouched.
- **In a wide universe, one bad market must never stop the rest.** This has
  now bitten three times in three places: an unhandled 429 in
  `_refresh_market_data`, an empty frame in `run_cycle` ("a market has no
  stored bars yet" aborted the whole cycle while 24 markets had good data),
  and before that a missing feature column in `_signal_for`. The rule for
  anything that loops over `self.coins`: degrade that market, log it, keep
  going. Only refuse when *every* market is unusable.
- **A wide universe will meet the rate limit; do not let it be fatal.** The
  live loop makes two requests per market per cycle. `_refresh_market_data`
  skips a market it cannot refresh and logs it, rather than raising: an
  uncaught 429 exited the trader and the supervisor restarted it into the
  same rate limit every 30 seconds. The strategy reads from storage, so a
  skipped market decides on the bars it already has.
- **Backfilling 1m/5m/1h at once triggers 429s.** Keep `CANDLE_INTERVALS=1h`
  in deployment.
- **Markets are never equal in length; align the bars, not the matrices.**
  The cross-asset block raises `coins are not aligned` on frames of different
  lengths, and in real storage they always differ -- one market topped up a
  minute later, one skipped by a rate limit, one listed last week. That
  exception took the whole feature build down, which is the "one bad market
  stops the rest" failure yet again; a scheduled retrain met it every single
  time. `features.pipeline.align_bars` drops markets with a fraction of the
  deepest history (rather than truncating everyone to a new listing) and inner
  joins the rest. It must be applied to *bars*: the live loop and the backtest
  index a coin's feature frame by the same position they index its bars, so a
  matrix shorter than its bars would silently score the wrong row.
- **Freshness is measured from `trained_at_ms`, not `train_span[1]`.** The
  span ends where the *development* period ends, ~20% short of the data it was
  built from because the tail is the locked holdout. Measuring from there
  reports a fifth of the dataset as new on every run, so the "is there anything
  to learn from" gate can never say no and every cycle refits an identical
  model.
- **The cross-asset benchmark (BTC) has no `cross_*_btc_*` values.**
  `build_universe` aligns all coins to a union of columns and fills NaN.
  Without that alignment BTC could never produce a signal at all.
- **LightGBM needs `libomp` on macOS** (no Homebrew here), so local runs fall
  back to scikit-learn `HistGradientBoosting`. Linux/Docker uses LightGBM.
  Results are comparable in kind, not identical — the backend is recorded on
  every artefact.
- **A platform healthcheck cannot present a password.** `railway.json`'s
  `healthcheckPath` pointed at `/`, which is behind HTTP Basic auth and
  answers 401, so Railway killed every new deployment after exactly
  `healthcheckTimeout` (300s) while the app logs showed a healthy boot. The
  tell is deployments failing at a round number of seconds. `/health` is
  unauthenticated and returns only `{"status": "ok"}`.
- **A position too small to trade is a position you cannot leave.**
  `orders_to_reach` refuses trades under `min_trade_notional` to stop churn,
  which also meant it refused to *close* a sub-$10 stub. The holding cap
  fired on one such position every bar for 562 hours and was ignored every
  time. Reductions that would leave less than the minimum now go to zero,
  and exits are never suppressed for size.
- **Clocks must be timestamps, not counters.** The idle timer counted flat
  bars in memory, so every redeploy handed it zero and it could never reach
  its limit on a service that restarts often. It is now `idle_since_ms`,
  persisted in the state file's `extra` blob; position age likewise comes
  from `Position.opened_ts_ms`.
- **One script renders the whole dashboard.** A single syntax error in it
  blanks every panel while the page still returns 200 and all six APIs still
  return 200 — the failure looks like a data problem and is not one. A
  redeclared `const` did exactly this. `tests/test_dashboard.py` now runs
  `node --check` over the page script; it skips where node is absent, so run
  the suite locally before trusting a dashboard change.
- **A syntax check is not enough: the page must survive a broken panel.**
  Every fetch used to sit in one `Promise.all`, which rejects the moment any
  member does — so one endpoint returning 500 blanked all six panels, with
  the page and the other five APIs still answering 200. Identical symptom to
  the syntax error above, different cause, and `node --check` cannot see it.
  Endpoints are now fetched through `get(url, fallback)`, which never
  rejects, each render runs inside `safely(...)`, and whatever failed is
  named in the banner — a silently missing panel reads as "nothing is
  happening", which is a different and far more misleading claim than "this
  did not load". `tests/test_dashboard.py` executes the real page script
  under node with a stubbed DOM and a deliberately failing endpoint; that
  test fails against the old code, which is the only reason to trust it.
  **Adding a new panel means adding it to `get(...)`, never to a bare
  `fetch` — one new API must not be able to take the other five down.**
- **numpy scalars are not JSON, and `and` returns its second operand.**
  `has_verdict` was `resolved >= 200 and np.isfinite(auc)`. Under 200 rows
  the `and` short-circuits to a Python `False` and serialises fine; from the
  200th row on it yields a `numpy.bool_` and `/api/learning` began returning
  500 — days after deploy, with the tests passing, because every test used a
  handful of rows. numpy 2 also renamed `np.bool_`, so the error reads
  `Object of type bool is not JSON serializable`, which sounds impossible.
  Anything crossing the JSON boundary goes through
  `models.scorecard._json_safe` as a whole payload, never field by field,
  and bools are checked before ints there because a Python bool *is* an int.
  Test JSON paths at a realistic sample size, not with three rows.
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

## How it is configured live (2026-08-28)

Shaurya asked to "remove all these guardrails" and to force the system to
trade rather than sit flat. Railway variables now:

| Variable | Value | Effect |
| --- | --- | --- |
| `RISK_PROFILE` | `aggressive` | 10x leverage, 100% of equity risked per trade, $100k max position, **daily-loss and drawdown halts off**. Liquidation still applies and is now the only backstop. |
| `SIGNAL_THRESHOLD` | `0.40` | Was 0.55. Fires on ~25% of bars instead of ~7%. |
| `MAX_HOLD_HOURS` | `24` | Nothing is held longer; re-entry allowed. |
| `MAX_IDLE_HOURS` | `0.75` | Force an entry after 45min holding nothing. Shorter than the 1h decision interval, so in practice it means "never be flat at a decision". |
| `MARKETS` | `top:25` | The 25 most traded perps, resolved at startup. `all` (~176) is supported; see DEPLOY.md for the cost. |
| `MAX_OPEN_POSITIONS` | `20` | Held at once. A forced entry fills every free slot, not just one. |
| `MAX_POSITION_USD` | `10000` | Per position. 20 x $10k = $200k gross on $100k equity, i.e. 2x used of the 10x allowed. |
| `MIN_HOLD_HOURS` | `4` | A fading probability may not close a position younger than this. Risk exits ignore it. |

He was shown the measured consequences and chose this anyway. **Do not
quietly re-tighten these.** If the account collapses or liquidates, that is
the configuration behaving as measured — report it plainly, do not treat it
as a defect and do not "fix" it without being asked.

The two things that were *not* removed, and should not be:

- **Liquidation stays.** His own earlier instruction, and with the halts off
  it is the only thing between a bad run and zero.
- **Costs, causality checks and the accounting proof stay.** These do not
  restrain the trading; they are what makes the reported numbers true.
- **Real trading stays impossible.** Non-negotiable 1 is not his to waive by
  implication; it would need an explicit, unambiguous request, and the answer
  is still no.

## Current status

Phases 1-8 of the original brief are built. 451 tests pass.

**The model has no demonstrated edge.** On 208 days of real BTC/ETH/SOL
hourly data:

| | |
| --- | --- |
| Locked holdout AUC | **0.504** (coin flip) |
| Out-of-sample return, conservative | −1.08% |
| Out-of-sample return, aggressive (10x) | −41.38% |
| Out-of-sample, aggressive + forced activity | **−54.07%** (2 liquidations) |
| Buy-and-hold, same period | +17.60% |

The aggressive profile ended a full backtest at **$54.95** — from paying
$231,646 in slippage at 1,098× annual turnover, not from bad direction calls.

Lowering the entry threshold and forcing activity was measured, not guessed.
Out-of-sample, on the conservative profile: 0.55 → −1.59% over 13 trades,
0.45 → −3.45% over 63, 0.40 → −3.74% over 105, and 0.40 with a 24h holding cap
and a 6-bar idle timer → −4.29% over 167. More trading, monotonically more
loss. On the aggressive profile the same settings end the full backtest at
**$4.43** with two liquidations. This is what a 0.504-AUC model does when you
let it trade more: it pays spread and fees to express a coin flip.

Treat this as the honest baseline. The productive next step is a better
label (longer horizon, or funding-carry rather than direction) and more
regularisation, judged on walk-forward validation only — **not** further
tuning against the holdout, which has already been read once.

## The dashboard

Read-only. Times are rendered in `DISPLAY_TIMEZONE` (default `Asia/Kolkata`)
and so is the daily P&L boundary — both ends, so the label and the number
cannot disagree. Everything stored and computed is UTC; this is presentation
only. `/api/state` reports the live configuration (markets, per-position cap,
thresholds, hold window) because configuration the reader cannot see is
configuration they cannot trust. `/api/learning` does the same for the model:
which one is deciding, when it next refits, every refit it has attempted, and
the live scorecard. A model that changes itself on a schedule makes a change in
behaviour indistinguishable from a change in the market unless the page says
which one moved.

## Deployment

Live on Railway; runbook in `deploy/DEPLOY.md`. One service runs the trader
and the dashboard together against a single volume at `/app/storage`, which
holds the market data, the model, and the account state. Without that volume
every redeploy resets the account to $100k.
