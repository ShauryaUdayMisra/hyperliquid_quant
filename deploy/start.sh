#!/usr/bin/env bash
# Entrypoint for a single-container deployment (Railway, Fly, any PaaS).
#
# The trader writes market data and account state; the dashboard reads it.
# They must share a filesystem, which on most platforms means one service and
# one volume. So this runs both: the trader in the background, the web server
# in the foreground.

set -uo pipefail   # deliberately NOT -e: see the failure handling below

PORT="${PORT:-8000}"
INTERVAL="${TRADE_INTERVAL:-1h}"
BACKFILL_DAYS="${BACKFILL_DAYS:-400}"
# Railway attaches ONE volume per service, so the model lives under the same
# mount as the data. Otherwise every redeploy would retrain.
MODEL_PATH="${MODEL_PATH:-storage/model.pkl}"
# P(up) a market must clear before the strategy will open a long. Lower means
# more trades on weaker evidence, not better ones -- see DEPLOY.md.
THRESHOLD="${SIGNAL_THRESHOLD:-0.55}"

echo "==> Hyperliquid paper trading (simulated fills only, no real orders)"
python -c "from config.settings import SETTINGS; print('   ', SETTINGS.risk.describe())"

# 1. Market data. Always run: the backfill resumes from whatever is already
#    stored, so it is cheap when the volume is current and self-healing when
#    a previous run was cut short by rate limiting. Skipping it merely
#    because the directory existed left a half-written store that then
#    failed training with an unhelpful error.
echo "==> Backfilling up to ${BACKFILL_DAYS} days at ${INTERVAL} (resumes if present)..."
python main.py backfill --days "${BACKFILL_DAYS}" --intervals "${INTERVAL}" \
  || echo "!! backfill reported errors (usually rate limiting); continuing with what landed"

echo "==> Data on disk:"
python main.py summary 2>/dev/null | sed 's/^/    /' || true

# 2. Model. Never fabricate one; if training cannot succeed, say so loudly
#    and still bring the dashboard up so the failure is visible rather than
#    hidden behind a restart loop.
HAVE_MODEL=0
if [ -f "${MODEL_PATH}" ]; then
  echo "==> Using existing ${MODEL_PATH}"
  HAVE_MODEL=1
else
  echo "==> No model at ${MODEL_PATH}. Training with walk-forward validation..."
  if python main.py train --interval "${INTERVAL}" --output "${MODEL_PATH}"; then
    HAVE_MODEL=1
  else
    echo "!! TRAINING FAILED. The dashboard will start, but no trading will"
    echo "!! happen until a model exists. The usual cause is too little"
    echo "!! stored history -- check the data summary above."
  fi
fi

# 3. Trader in the background, but only with a model behind it.
TRADER_PID=""
if [ "${HAVE_MODEL}" = "1" ]; then
  (
    while true; do
      echo "==> starting paper trader"
      python main.py paper --interval "${INTERVAL}" --model "${MODEL_PATH}" \
        --threshold "${THRESHOLD}" \
        || echo "!! paper trader exited ($?); restarting in 30s"
      sleep 30
    done
  ) &
  TRADER_PID=$!
else
  echo "==> paper trader NOT started (no model)"
fi

# Stop the trader cleanly on SIGTERM so the last cycle's buffers reach the
# volume instead of being dropped.
trap 'echo "==> shutting down"; [ -n "${TRADER_PID}" ] && kill -TERM "${TRADER_PID}" 2>/dev/null; wait' TERM INT

# 4. Dashboard in the foreground. It binds 0.0.0.0 because the platform's
#    router reaches the container from outside; DASHBOARD_PASSWORD is what
#    keeps it private, not the bind address.
echo "==> dashboard on :${PORT}"
exec uvicorn dashboard.app:app --host 0.0.0.0 --port "${PORT}" --log-level warning
