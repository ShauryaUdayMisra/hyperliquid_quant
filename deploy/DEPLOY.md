# Running continuously on an always-on host

The system is useless on a laptop that sleeps. Beyond the obvious uptime
argument, there is one that is specific to this project:

> **Order-book snapshots and trade prints cannot be backfilled.** Hyperliquid
> serves historical candles and funding on request, but there is no
> historical endpoint for book depth or individual prints. That data exists
> only if a collector was listening at the time. Every day the collector is
> off is a day of Phase 3 features that can never be recovered.

Deploy the collector first, even though Phases 3-8 are not built yet. It
starts accumulating the one dataset that has to be captured live.

---

## 1. Pick a host

| Option | Cost | Notes |
| --- | --- | --- |
| **Small cloud VM** (Hetzner CX22, DigitalOcean, Vultr, Lightsail) | ~$5-7/mo | Recommended. 2 vCPU / 4 GB / 40 GB is ample. |
| Oracle Cloud always-free ARM | free | Generous, but availability is erratic and the account can be reclaimed. |
| Raspberry Pi 4/5 at home | one-off | Works well. Vulnerable to your power and internet, and to the same firewall you have now. |

Any Linux host with 2 GB RAM and 25 GB disk is enough for the collector.
Phase 4 model training wants 4 GB.

**A cloud VM also solves your FortiGate problem.** A datacentre host has
clean egress to `api.hyperliquid.xyz`, so `HL_CA_BUNDLE` stays unset and the
TLS interception you are hitting locally simply does not occur.

## 2. Provision (fresh Ubuntu 24.04)

```bash
sudo adduser --system --group --home /opt/hyperliquid-quant quant
sudo apt update && sudo apt install -y python3-venv python3-pip git
```

## 3. Get the code onto the host

The repo is not under version control yet. Two paths:

**Git (preferred)** — makes updates a one-liner forever after:

```bash
# on your Mac, once
git init && git add -A && git commit -m "Hyperliquid paper-trading system"
# push to a PRIVATE GitHub repo, then on the server:
sudo -u quant git clone git@github.com:<you>/hyperliquid-quant.git /opt/hyperliquid-quant
```

**rsync** — no GitHub account needed:

```bash
rsync -avz --exclude .venv --exclude storage --exclude .env \
  ~/hyperliquid-quant/ user@your-server:/opt/hyperliquid-quant/
```

Never copy `.env` or `certs/` to the server. `.gitignore` already excludes
both; recreate `.env` on the host in the next step.

## 4. Configure

```bash
cd /opt/hyperliquid-quant
sudo -u quant cp .env.example .env
sudo -u quant nano .env
```

Set at minimum:

- `RISK_PROFILE` — `aggressive` or `conservative`
- `MARKETS`, `CANDLE_INTERVALS`
- Leave `HL_CA_BUNDLE` commented out (you should not need it here)
- Phase 7 only: `SMTP_APP_PASSWORD`, `REPORT_SENDER`, `REPORT_ENABLED=true`

Lock it down — it will hold a Gmail app password from Phase 7 onward:

```bash
sudo chmod 600 .env && sudo chown quant:quant .env
```

## 5. Install and verify BEFORE starting anything

```bash
sudo -u quant python3 -m venv .venv
sudo -u quant .venv/bin/pip install -r requirements.txt

sudo -u quant .venv/bin/python -m pytest          # 208 offline tests
sudo -u quant .venv/bin/python main.py status     # API reachability
sudo -u quant .venv/bin/python -m pytest -m live  # 4 real-API tests
```

`main.py status` must print the perp market count and live mids. If the live
tests pass here, that is the Phase 1 verification that has never yet run on
your Mac.

## 6. Backfill history, then start collecting

```bash
sudo -u quant .venv/bin/python main.py backfill --days 180
sudo -u quant .venv/bin/python main.py verify
```

Then run the collector permanently. **Pick one** of the two:

**Systemd (simplest on a dedicated VM):**

```bash
sudo cp deploy/hlquant-collector.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now hlquant-collector
sudo systemctl status hlquant-collector
```

**Docker (if you would rather not manage a venv):**

```bash
docker compose -f deploy/docker-compose.yml up -d --build
```

Both restart automatically on crash and on host reboot, and both give the
collector 90 seconds on shutdown to flush buffered data to Parquet.

## 6b. Train a model, then start paper trading

The collector only gathers data. Trading needs a model, and reports need SMTP.

```bash
# Train with walk-forward validation. Read the WARNINGS section of the output.
sudo -u quant .venv/bin/python main.py train --interval 1h

# Backtest it, with the shuffled-prediction control for comparison.
sudo -u quant .venv/bin/python main.py backtest --control

# Verify the email format before trusting the schedule.
sudo -u quant .venv/bin/python main.py report --test-report --print-text
```

Only once the backtest is credible, start the trader:

```bash
docker compose -f deploy/docker-compose.yml up -d paper-trader dashboard
```

The trader decides once per bar close, persists its account to
`storage/paper_account.json` after every cycle, and resumes from that file on
restart rather than silently resetting to $100k.

### System-level report fallback

APScheduler only fires while the process is alive. Add a cron entry so a
report still goes out if the trader is down:

```cron
0 0,6,12,18 * * * cd /opt/hyperliquid-quant && .venv/bin/python main.py report >> logs/report-cron.log 2>&1
```

On macOS use a launchd plist with `StartCalendarInterval` for the same hours.

## 7. Confirm it is actually alive

```bash
# systemd
journalctl -u hlquant-collector -f
# docker
docker compose -f deploy/docker-compose.yml logs -f

# either way, the real check: is stored data advancing?
sudo -u quant .venv/bin/python main.py summary
```

A healthy collector logs a `flushed | uptime=... trades=... book_rows=...`
line roughly every 60 seconds, and `summary` shows the `end` timestamp for
`trades` and `orderbook` moving forward. If `errors=` climbs steadily,
something is wrong.

## 8. Disk and backups

At the default cadence (3 markets, 20 book levels, one snapshot per 10s)
expect roughly **1-2 GB per month**, mostly order-book rows. A 40 GB disk
holds well over a year. To reduce it, raise `ORDERBOOK_SNAPSHOT_INTERVAL_S`
or lower `ORDERBOOK_DEPTH` — but remember this data is unrecoverable, so
prefer more disk over less data.

`storage/` is the only irreplaceable state. Back it up:

```bash
rsync -avz user@your-server:/opt/hyperliquid-quant/storage/ ~/hlquant-backup/
```

## 9. Updating later

```bash
cd /opt/hyperliquid-quant && sudo -u quant git pull
sudo -u quant .venv/bin/pip install -r requirements.txt
sudo systemctl restart hlquant-collector
```

## Security notes

- The system holds **no keys that can move funds**, and `config/settings.py`
  aborts at import if one appears in the environment. Do not add exchange
  API keys to this host for any reason.
- Do not expose the Phase 7 dashboard to the internet. The compose file
  binds it to `127.0.0.1` deliberately; reach it over an SSH tunnel:
  `ssh -L 8000:localhost:8000 user@your-server`.
- `.env` will hold a Gmail app password. Keep it `chmod 600`, and use an
  app password rather than your account password so it can be revoked
  independently.

---

# Deploying to Railway

Railway is the quickest way to get this running 24/7 with a public dashboard
URL. One service runs both the trader and the dashboard, because they share a
filesystem and Railway attaches one volume per service.

Expect roughly **$5/month** on the Hobby plan; this workload is small but it
runs continuously, so it does not fit the free trial credit indefinitely.

## 1. Push the repo to GitHub, privately

The repo is not under version control yet:

```bash
git init && git add -A
git commit -m "Hyperliquid paper-trading system"
gh repo create hyperliquid-quant --private --source=. --push
```

`.gitignore` already excludes `.env`, `storage/`, `certs/` and `*.pkl`, so no
secrets and no data go up. Confirm with `git status --porcelain` before
pushing if you want to be sure.

## 2. Create the service

On railway.app: **New Project → Deploy from GitHub repo** → pick the repo.

Railway reads [`railway.json`](../railway.json), which points at
the root `Dockerfile` and runs `deploy/start.sh`. No further build config.

## 3. Add a volume — do this before the first successful deploy

**Service → Settings → Volumes → Add volume**, mount path:

```
/app/storage
```

Everything irreplaceable lives here: the market data, the trained model, the
account state and the equity history. Without a volume, every redeploy resets
your account to $100k and throws away the track record.

## 4. Environment variables

**Service → Variables.** At minimum:

| Variable | Value | Why |
| --- | --- | --- |
| `DASHBOARD_PASSWORD` | a long random string | **Required.** The app refuses to start on a public host without it. |
| `DASHBOARD_USER` | `admin` | Optional, defaults to `admin`. |
| `DISPLAY_TIMEZONE` | `Asia/Kolkata` | Timezone every time on the dashboard is shown in, and whose midnight starts "today" for the daily P&L. Presentation only — the trader is UTC throughout. An unknown name falls back to UTC with a warning. |
| `MIN_HOLD_HOURS` | `4` | A fading probability may not close a position younger than this. Risk exits ignore it. |
| `RISK_PROFILE` | `conservative` or `aggressive` | Which limits to enforce. |
| `MARKETS` | `BTC,ETH,SOL` | Markets to trade. Also accepts `top:25` (the N most traded perps) and `all` (every live perp, ~176). Ranked forms are resolved against the exchange at startup and fall back to any literal coins if it is unreachable. |
| `MAX_OPEN_POSITIONS` | profile default | How many markets may be held at once. |
| `LIVE_LOOKBACK_BARS` | `1000` | Bars of history the live loop reads per market. Must exceed `features.pipeline.MAX_LOOKBACK_BARS` (720). |
| `TRADE_INTERVAL` | `1h` | Decision cadence. |
| `BACKFILL_DAYS` | `400` | Hyperliquid serves ~5,000 candles, so 1h data caps near 208 days regardless. |
| `SIGNAL_THRESHOLD` | `0.55` | P(up) a market must clear to open a long. |
| `RISK_PROFILE` | `conservative` | `aggressive` removes the daily-loss and drawdown halts and raises leverage to 10x. Liquidation still applies and is the only backstop left. |
| `MAX_HOLD_HOURS` | `24` | Force-close a position past this age. Re-entry allowed. 0 disables. |
| `MAX_IDLE_HOURS` | `6` | Force an entry after this long holding nothing. 0 disables. |
| `RETRAIN_ENABLED` | `true` | Refit the live model on a schedule. `false` freezes it at whatever the first boot trained. |
| `RETRAIN_EVERY_HOURS` | `24` | How often. Timed from the model's own `trained_at_ms`, so a redeploy cannot postpone a refit that is already due. |
| `RETRAIN_MIN_NEW_BARS` | `12` | New bars needed before a refit is worth the CPU. Counted as distinct timestamps, so it means "hours of new market" at any universe size. |
| `RETRAIN_MIN_AUC_MARGIN` | `0.02` | A candidate this far below the incumbent's walk-forward AUC is rejected. |

`SIGNAL_THRESHOLD` sets how often the system trades, not how well. The model's
base rate is 0.33, so on 208 days of BTC/ETH/SOL history the default 0.55 fires
on 6.9% of bars, 0.45 on 16.6%, and 0.40 on 24.9%. Lowering it buys frequency
with weaker evidence, and every extra round trip pays spread, impact and fees.
Given a holdout AUC of 0.504, expect a lower threshold to lose money faster
rather than find an edge. There is no down-model, so the strategy is long-only:
a low P(up) means "do not buy", never "sell short".

For the 6-hour emails, add `REPORT_ENABLED=true`, `REPORT_RECIPIENT`,
`REPORT_SENDER`, `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_APP_PASSWORD`.

`MARKETS=all` is supported but costs real time on a cold volume: roughly 23
requests per market for a full history, so ~4,000 requests and half an hour of
backfill before the first trade, and a much larger feature build every cycle.
`top:25` gets most of the diversification for a twentieth of the work.

Set these in Railway's UI, not in a committed file.

## 5. Get the URL

**Settings → Networking → Generate Domain.** You get something like
`hyperliquid-quant-production.up.railway.app`.

Open it and the browser prompts for the username and password from step 4.

Railway injects `$PORT`; `start.sh` binds to it automatically.

## 6. What the first boot does

`deploy/start.sh` is idempotent, so a redeploy skips whatever is already on
the volume:

1. backfills candles and funding if `storage/parquet/candles` is absent,
2. trains a model if `storage/model.pkl` is absent (a few minutes),
3. starts the paper trader, restarting it if it ever exits,
4. serves the dashboard in the foreground.

From then on the trader keeps the model current itself: every
`RETRAIN_EVERY_HOURS` it refits on all stored history — including every bar it
has since been wrong about — and swaps the result in without a restart. Nothing
in `start.sh` needs to change for that; step 2 only ever runs once.

First deploy takes ~5-10 minutes because of steps 1 and 2. Watch **Deployments
→ View Logs**. The health check allows 300 seconds before the container is
considered up, which is why `healthcheckTimeout` is set high.

## What the dashboard shows

- **P&L over three horizons** — lifetime, today (since 00:00 UTC), and the
  rolling last hour. Each reads "not enough history" rather than guessing when
  there are too few readings to answer honestly.
- **Equity curve** with the starting capital drawn as a dashed baseline.
- **Open positions** with entry, notional and unrealised P&L.
- **Every buy and sell** — time, side, size, price, fee, realised P&L, and the
  reason the trade was taken.
- **Closed round trips** with net P&L after fees and funding.
- **Current reasoning per market**, generated by the same template the 6-hour
  email uses, so the page and the inbox never tell different stories.
- **What it has learned** — which model is deciding, when it next refits, the
  history of every refit (promoted or rejected, and why), and the live
  scorecard: the model's own recorded calls marked against what the price
  actually did. That last number is the only out-of-sample figure on the page
  that cannot have been tuned, because every prediction was written down
  before its outcome existed. It needs hundreds of resolved calls before it
  says anything, and says so until then.

It refreshes every 30 seconds and has no write endpoints — it cannot place,
modify or cancel anything.

## Security

The dashboard exposes your full account and its reasoning. HTTP Basic over
Railway's TLS is adequate for a private paper account; it is not adequate for
anything you would mind a determined stranger reading. The app hard-fails at
startup if `RAILWAY_ENVIRONMENT` is set and `DASHBOARD_PASSWORD` is not, so a
forgotten password takes the service down rather than quietly publishing it.

---

# The live deployment (recorded 2026-08-28)

Live at **https://hyperliquidquant-production.up.railway.app**
(HTTP Basic: `admin` / the value in `.railway_dashboard_password`, gitignored).

Railway project `welcoming-rejoicing`, environment `production`, service
`hyperliquid_quant`, US West. One service runs the trader and the dashboard
together against a 5 GB volume at `/app/storage`.

Administer it with the CLI rather than the web UI — it is already linked:

```bash
railway logs                      # runtime logs
railway logs --build              # build logs
railway variables                 # list; --set K=V --skip-deploys to change
railway redeploy --yes            # apply changes
railway volume list               # confirm the mount is attached
```

## Four things that broke on the first real deploy

Recorded because each cost time and none was obvious from the error.

1. **Railway ignored the Dockerfile in `deploy/`.** PaaS builders only
   auto-detect a Dockerfile at the *repository root*. It fell back to its own
   Python heuristics and built an image that ran none of the bootstrap. The
   Dockerfile now lives at the root.
2. **Railway ignored `railway.json`'s `startCommand`.** Config-as-code is
   deprecated (sunset 2026-12-01), so the container fell through to the
   image's `CMD` and ran the data collector instead of the system. The `CMD`
   is now `bash deploy/start.sh`.
3. **A volume mounts as root over the build-time `chown`.** The container ran
   as an unprivileged user and could not write. `deploy/entrypoint.sh` now
   starts as root purely to fix ownership, then drops to `quant` via `gosu`.
4. **Rate limiting, then a poisoned bootstrap.** Backfilling 1m/5m/1h at once
   drew HTTP 429s and left a partial candle store. The bootstrap saw the
   directory, skipped the backfill, and training died with `no usable
   labelled rows` — under `set -e` that crash-looped invisibly. The backfill
   is idempotent and now always runs, a training failure no longer kills the
   container, and `CANDLE_INTERVALS=1h` keeps request volume sane.
