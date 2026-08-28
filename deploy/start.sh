#!/usr/bin/env bash
# Entrypoint for a single-container deployment (Railway, Fly, any PaaS).
#
# The trader writes market data and account state; the dashboard reads it.
# They must therefore share a filesystem, which on most PaaS platforms means
# sharing one service and one volume. So this script runs both: the trader
# in the background, the web server in the foreground as PID 1's child.
#
# The bootstrap is idempotent -- a redeploy with a mounted volume skips
# straight to trading rather than re-downloading and re-training.

set -euo pipefail

PORT="${PORT:-8000}"
INTERVAL="${TRADE_INTERVAL:-1h}"
BACKFILL_DAYS="${BACKFILL_DAYS:-400}"
# Railway attaches ONE volume per service, so the model lives under the
# same mount as the data. Otherwise every redeploy would retrain.
MODEL_PATH="${MODEL_PATH:-storage/model.pkl}"

echo "==> Hyperliquid paper trading (simulated fills only, no real orders)"
python -c "from config.settings import SETTINGS; print('   ', SETTINGS.risk.describe())"

# 1. Market data. Skipped when a volume already holds it.
if [ ! -d storage/parquet/candles ]; then
  echo "==> No stored candles. Backfilling ${BACKFILL_DAYS} days at ${INTERVAL}..."
  python main.py backfill --days "${BACKFILL_DAYS}" --intervals "${INTERVAL}"
else
  echo "==> Candle store found; the trader will top it up on its first cycle."
fi

# 2. Model. Trading without one is not something to paper over with a default.
if [ ! -f "${MODEL_PATH}" ]; then
  echo "==> No model at ${MODEL_PATH}. Training with walk-forward validation..."
  python main.py train --interval "${INTERVAL}" --output "${MODEL_PATH}"
else
  echo "==> Using existing ${MODEL_PATH}"
fi

# 3. Trader in the background, restarted if it ever exits.
(
  while true; do
    echo "==> starting paper trader"
    python main.py paper --interval "${INTERVAL}" --model "${MODEL_PATH}" || \
      echo "!! paper trader exited ($?); restarting in 30s"
    sleep 30
  done
) &
TRADER_PID=$!

# Stop the trader cleanly when the platform sends SIGTERM, so the last
# cycle's buffers are flushed to the volume instead of dropped.
trap 'echo "==> shutting down"; kill -TERM "${TRADER_PID}" 2>/dev/null || true; wait' TERM INT

# 4. Dashboard in the foreground. Binds 0.0.0.0 because the platform's
#    router reaches the container from outside; the password is what keeps
#    it private, not the bind address.
echo "==> dashboard on :${PORT}"
exec uvicorn dashboard.app:app --host 0.0.0.0 --port "${PORT}" --log-level warning
