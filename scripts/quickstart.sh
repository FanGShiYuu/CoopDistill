#!/usr/bin/env bash
set -euo pipefail

python -m unittest discover -s tests -v
python -m coopdistill.evaluate \
  --limit "${LIMIT:-2}" \
  --workers "${WORKERS:-0}" \
  --output results/quickstart.json
