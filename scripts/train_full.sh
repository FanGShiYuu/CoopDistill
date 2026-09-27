#!/usr/bin/env bash
set -euo pipefail

python -u -m coopdistill.trainer \
  --variant full \
  --seed "${SEED:-23}" \
  --device "${DEVICE:-cuda}" \
  --workers "${WORKERS:-6}" \
  --output "${OUTPUT:-runs/coopdistill_seed23}"
