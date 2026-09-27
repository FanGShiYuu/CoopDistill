#!/usr/bin/env bash
set -euo pipefail

python -u -m coopdistill.trainer \
  --variant direct_mappo \
  --seed "${SEED:-23}" \
  --device "${DEVICE:-cuda}" \
  --workers "${WORKERS:-6}" \
  --output "${MAPPO_OUTPUT:-runs/direct_mappo_seed23}"

python -u -m coopdistill.comparison madqn \
  --hard-root benchmarks/default \
  --seed "${SEED:-23}" \
  --device "${DEVICE:-cuda}" \
  --output "${MADQN_OUTPUT:-runs/discrete_madqn_seed23.json}"
