#!/usr/bin/env bash
set -euo pipefail

python -m coopdistill.evaluate \
  --methods pg,coopdistill,fcfs_reservation,auction_reservation,nash_bargaining_mpc \
  --workers "${WORKERS:-8}" \
  --output results/reproduced.json
