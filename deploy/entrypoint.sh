#!/usr/bin/env bash
# Runs as root for exactly as long as it takes to fix volume ownership,
# then drops to the unprivileged `quant` user and hands over.
#
# This exists because a mounted volume arrives owned by root at RUN time,
#overwriting whatever the Dockerfile chowned at BUILD time. Without this the
# container starts, tries to write market data, and dies with a permission
# error that gives no hint that volumes are involved.

set -euo pipefail

for dir in /app/storage /app/logs /app/models; do
  mkdir -p "${dir}"
  chown -R quant:quant "${dir}" 2>/dev/null || \
    echo "warning: could not chown ${dir}; continuing" >&2
done

exec gosu quant "$@"
