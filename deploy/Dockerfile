# Single image for every phase of the system.
#
# Python 3.12 rather than 3.14: LightGBM and scikit-learn ship prebuilt
# wheels for 3.12, so the image builds in seconds instead of compiling from
# source. The code targets 3.11+ and runs identically on both.
FROM python:3.12-slim

# libgomp1 is LightGBM's OpenMP runtime (needed from Phase 4 onward).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first so code edits do not invalidate the layer cache.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Never run the collector as root. The storage volume is chowned to match.
RUN useradd --create-home --uid 10001 quant \
    && mkdir -p /app/storage /app/logs \
    && chown -R quant:quant /app
USER quant

ENV PYTHONUNBUFFERED=1

# PaaS routers inject the port to listen on. 8000 is only the local default.
ENV PORT=8000
EXPOSE 8000

# tini reaps zombies and forwards SIGTERM, so `docker stop` triggers the
# collector's graceful shutdown and flushes buffered data instead of
# dropping the last window.
ENTRYPOINT ["tini", "--"]

# Default: collector only. Railway overrides this with deploy/start.sh, which
# runs the trader and the dashboard together against one shared volume.
CMD ["python", "main.py", "collect"]
