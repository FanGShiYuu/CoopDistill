#!/usr/bin/env bash
set -euo pipefail

python -m coopdistill.evaluate \
  --workers "${WORKERS:-8}" \
  --output results/reproduced.json
