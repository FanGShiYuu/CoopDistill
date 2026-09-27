# CoopDistill

Official review release for **CoopDistill: Reference-Guided Multi-Agent Policy
Distillation for Cooperative Driving**.

CoopDistill retains a potential-game (PG) optimizer as a cooperative reference,
learns bounded acceleration, timing, and lateral residuals with a shared MAPPO
actor and centralized critic, prioritizes informative interactions during
training, and compares learned proposals against PG before execution.

This repository is a compact reproduction package. It contains the complete
model used for the main experiment, the default four-leg intersection benchmark,
a pretrained checkpoint, and the independent comparison methods used in the
paper. Paper-writing utilities, cluster-specific launch files, intermediate
search runs, and ablation checkpoints are intentionally excluded.

## Included methods

- **CoopDistill**: PG-guided residual MAPPO with curriculum sampling and
  comparative fallback fusion.
- **Potential Game**: structured cooperative reference controller.
- **FCFS reservation**.
- **Auction reservation**.
- **Nash-bargaining MPC**.
- **Direct MAPPO**: continuous-action MARL without the PG solver.
- **Discrete MADQN**: discrete-action independent multi-agent Q-learning.

All methods use the same vehicle dynamics, legal origin-destination routes,
episode horizon, collision definition, and evaluation cases.

## Environment

The default benchmark is an unsignalized four-leg intersection under
right-hand traffic. Each case contains six to eight vehicles on legal routes,
with a CAV penetration rate of 0.5 or 0.75. HDVs use heterogeneous IDM-like
car-following and conflict-yielding behavior. The simulation step is 0.1 s, the
decision interval is 0.5 s, and the episode horizon is 60 s.

An episode is successful only when every vehicle reaches its destination within
the horizon without a collision or invalid motion. `collision` counts episodes
with a CAV-involved collision, `cleared` is the fraction of vehicles reaching
their destinations, and `delay` is measured against free-flow travel time with
the horizon assigned to unfinished vehicles.

## Installation

```bash
conda create -n coopdistill python=3.11 -y
conda activate coopdistill
pip install -e .
```

PyTorch must match the CUDA version of the target machine for training. The
pretrained checkpoint and evaluation also run on CPU.

## Quick verification

The quickstart checks the benchmark and checkpoint, then evaluates Potential
Game and the complete CoopDistill model on two cases:

```bash
bash scripts/quickstart.sh
```

Equivalent commands on Windows:

```powershell
python -m unittest discover -s tests -v
python -m coopdistill.evaluate \
  --methods pg,coopdistill --limit 2 --workers 0 \
  --output results/quickstart.json
```

## Reproduce the default benchmark

Evaluate the provided full-model checkpoint and optimization/rule baselines on
all 120 cases:

```bash
WORKERS=8 bash scripts/reproduce_main.sh
```

The command writes per-method aggregates to `results/reproduced.json`. Runtime
depends mainly on the Nash-bargaining MPC baseline; `--limit N` can be used for
a shorter check. Learning-baseline reference results are provided in
`results/expected_main_table.json`; the training commands below reproduce their
checkpoints from scratch.

Reference values from the paper are:

| Method | Success (%) | Collision (%) | Cleared (%) | Delay (s) |
|---|---:|---:|---:|---:|
| FCFS reservation | 13.3 | 0.0 | 64.7 | 22.1 |
| Auction reservation | 1.7 | 0.0 | 42.1 | 31.5 |
| Nash-bargaining MPC | 71.7 | 0.0 | 85.9 | 13.9 |
| Direct MAPPO | 70.8 | 0.0 | 83.8 | 13.5 |
| Discrete MADQN | 0.0 | 0.8 | 45.5 | 33.7 |
| **CoopDistill** | **85.8** | **0.0** | **92.2** | **11.2** |

Small numerical differences can occur across PyTorch, SciPy, and multiprocessing
versions. The case definitions and pretrained full-model checkpoint are fixed.

## Train CoopDistill

The full paper configuration is the default:

```bash
DEVICE=cuda SEED=23 bash scripts/train_full.sh
```

The output directory contains the protocol, training log, validation-selected
checkpoint, and test results. For a CPU pipeline check without full training:

```bash
python -m coopdistill.trainer \
  --variant full --seed 23 --device cpu --workers 0 \
  --smoke --output runs/smoke
```

## Train learning baselines

```bash
DEVICE=cuda SEED=23 bash scripts/train_baselines.sh
```

Static baselines require no training and can be evaluated independently:

```bash
python -m coopdistill.comparison static \
  --hard-root benchmarks/default \
  --methods fcfs_reservation,auction_reservation,nash_bargaining_mpc \
  --workers 8 --output results/static_baselines.json
```

## Repository layout

```text
coopdistill/              environment, PG, MAPPO, fusion, and baselines
benchmarks/default/       fixed default evaluation cases and protocol
checkpoints/              pretrained full CoopDistill actor
scripts/                  quickstart, evaluation, and training commands
tests/                    release integrity and smoke tests
results/                  expected main-table metrics
```

## Review release

The code is provided to support method inspection and result reproduction during
review. The repository uses anonymous author metadata; identifying information
can be added after the review process.
