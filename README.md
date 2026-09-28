# CoopDistill

Official review release for **CoopDistill: Reference-Guided Multi-Agent Policy
Distillation for Cooperative Driving**.

CoopDistill retains a potential-game (PG) optimizer as an internal cooperative
reference, learns bounded acceleration, timing, and lateral residuals with a
shared MAPPO actor and centralized critic, prioritizes informative interactions
during training, and applies comparative fallback before execution.

This compact repository contains the complete model used for the main
experiment, the fixed four-leg intersection benchmark, a pretrained checkpoint,
and the code needed to train and evaluate CoopDistill. Paper-writing utilities,
cluster launch files, intermediate experiments, ablation checkpoints, and
unrelated controller implementations are intentionally excluded.

## Environment

The default benchmark is an unsignalized four-leg intersection under
right-hand traffic. Each case contains six to eight vehicles on legal routes,
with a CAV penetration rate of 0.5 or 0.75. HDVs use heterogeneous IDM-like
car-following and conflict-yielding behavior. The simulation step is 0.1 s, the
decision interval is 0.5 s, and the episode horizon is 60 s.

An episode is successful only when every vehicle reaches its destination within
the horizon without a collision or invalid motion. `collision` counts episodes
with a CAV-involved collision, `cleared_fraction` is the fraction of vehicles
reaching their destinations, and `delay_censored_s` is measured against
free-flow travel time with the horizon assigned to unfinished vehicles.

## Installation

```bash
conda create -n coopdistill python=3.11 -y
conda activate coopdistill
pip install -e .
```

PyTorch must match the CUDA version of the target machine for training. The
pretrained checkpoint and evaluation also run on CPU.

## Quick verification

The quickstart verifies the benchmark and checkpoint, then evaluates the
complete CoopDistill model on two cases:

```bash
bash scripts/quickstart.sh
```

Equivalent commands on Windows:

```powershell
python -m unittest discover -s tests -v
python -m coopdistill.evaluate --limit 2 --workers 0 --output results/quickstart.json
```

## Reproduce the main model

Evaluate the provided full-model checkpoint on all 120 fixed cases:

```bash
WORKERS=8 bash scripts/reproduce_main.sh
```

The command writes aggregate and per-episode metrics to
`results/reproduced.json`. The reference checkpoint produces approximately
85.8% successful episodes, 0.0% CAV-involved collision episodes, 92.2% cleared
vehicles, and 11.2 s censored delay. Small numerical differences can occur
across PyTorch, SciPy, and multiprocessing versions.

## Train CoopDistill

The full paper configuration is the default:

```bash
DEVICE=cuda SEED=23 bash scripts/train_full.sh
```

The output directory contains the protocol, training log, validation-selected
checkpoint, trajectories, and test results. For a CPU pipeline check without
full training:

```bash
python -m coopdistill.trainer \
  --variant full --seed 23 --device cpu --workers 0 \
  --smoke --output runs/smoke
```

## Repository layout

```text
coopdistill/              environment, PG reference, MAPPO, fusion, and metrics
benchmarks/default/       fixed default evaluation cases and protocol
checkpoints/              pretrained full CoopDistill actor
scripts/                  quick verification, evaluation, and training commands
tests/                    release integrity and smoke tests
```

## Review release

The code is provided to support method inspection and result reproduction during
review. The repository uses anonymous author metadata; identifying information
can be added after the review process.
