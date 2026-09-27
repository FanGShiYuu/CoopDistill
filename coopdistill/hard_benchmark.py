"""Prepare and evaluate the fixed default benchmark. This entry point never trains."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict
import hashlib
import json
import multiprocessing
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .environment import Config
from .hard_cases import ANCHOR_SEED, FAMILIES, case_from_dict, generate_candidates, select_cases
from .trainer import Actor, VARIANTS, atomic_json, initialize_worker, mean_metrics, rollout

SEEDS = (11, 23, 37)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(main_root, root, per_family):
    main_root, root = Path(main_root).resolve(), Path(root).resolve()
    if (root / 'protocol.json').exists():
        raise FileExistsError('Preparation already complete; resume the screen directly')
    protocols = [read(main_root / v / f'seed_{s}' / 'protocol.json') for v in VARIANTS for s in SEEDS]
    originals = sorted((main_root / 'source').glob('*.py'))
    source_hash = hashlib.sha256(b''.join(p.read_bytes() for p in originals)).hexdigest()
    if any(p['code_sha256'] != source_hash for p in protocols):
        raise ValueError('Saved source snapshot does not match the trained models')
    if any(sha(Path(__file__).parent / p.name) != sha(p) for p in originals):
        raise ValueError('Evaluation must use the original source snapshot')
    cfg = Config(**protocols[0]['environment'])
    if any(p['environment'] != asdict(cfg) for p in protocols):
        raise ValueError('Source models used different environments')
    tasks = []
    for variant in VARIANTS:
        for seed in SEEDS:
            parent = main_root / variant / f'seed_{seed}'
            checkpoint = parent / 'best_actor.pt'
            actor = Actor()
            actor.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
            result = read(parent / 'results.json')
            if (result['status'] != 'complete' or result['variant'] != variant or
                    result['training_seed'] != seed):
                raise ValueError(f'Incomplete source run: {parent}')
            tasks.append(dict(index=len(tasks), variant=variant, training_seed=seed,
                              checkpoint=f'models/{variant}_{seed}.pt', sha256=sha(checkpoint),
                              best_update=result['best_update']))
    anchor_data = next(c for c in protocols[0]['test'] if c['seed'] == ANCHOR_SEED)
    candidates, rejected = generate_candidates(case_from_dict(anchor_data), cfg, per_family)
    (root / 'models').mkdir(parents=True, exist_ok=True)
    for task in tasks:
        shutil.copy2(main_root / task['variant'] / f"seed_{task['training_seed']}" / 'best_actor.pt',
                     root / task['checkpoint'])
    # runtime/ is assembled from the original run's source by the submit script.
    atomic_json(root / 'candidates.json', candidates)
    atomic_json(root / 'protocol.json', dict(
        source_run=str(main_root), anchor=anchor_data, environment=asdict(cfg),
        source_code_sha256=source_hash, candidates_sha256=sha(root / 'candidates.json'),
        target_cases=120, target_pg_failure_fraction=.2, candidates_per_family=per_family,
        rejection_counts=rejected, tasks=tasks, training_enabled=False,
        inference_device='cpu; identical rollout implementation to the original evaluation',
        runtime_sha256={p.name: sha(p) for p in Path(__file__).parent.glob('*.py')},
        interpretation='Fixed default intersection benchmark for reproducible comparison.',
        selection='PG only: target 20% failures, otherwise highest PG delay; family-balanced; never inspect policy outcomes.',
        safety='Original collision, boundary, dynamics, horizon and fallback checks unchanged.'))
    print(f'PREPARED candidates={len(candidates)} frozen_models={len(tasks)} output={root}', flush=True)


def verify_runtime(root):
    protocol = read(root / 'protocol.json')
    for name, expected in protocol['runtime_sha256'].items():
        if sha(Path(__file__).parent / name) != expected:
            raise ValueError(f'Frozen evaluation source changed: {name}')
    return protocol


def trace_diagnostics(trace):
    history = trace['history']
    speed = np.asarray([h['speed'] for h in history])
    finished = np.asarray([h['finished'] for h in history])
    dt = np.diff([h['time'] for h in history], prepend=0.)
    waiting = (speed < .2) & ~finished
    tail = []
    for i in range(speed.shape[1]):
        duration = 0.
        for t in range(len(history)-1, -1, -1):
            if not waiting[t, i]:
                break
            duration += dt[t]
        tail.append(float(duration))
    return dict(wait_time_s=(waiting * dt[:, None]).sum(0).tolist(),
                terminal_continuous_wait_s=tail,
                terminal_unfinished_ids=np.flatnonzero(~finished[-1]).tolist(),
                terminal_poses=history[-1]['poses'])


def evaluate_one(task):
    item, cfg, variant, checkpoint, path, manifest_hash = task
    path = Path(path)
    case = case_from_dict(item['case'])
    weights = torch.load(checkpoint, map_location='cpu', weights_only=True) if checkpoint else None
    result = rollout((case, cfg, variant, weights, case.seed, True, True))
    record = dict(seed=case.seed, fingerprint=case.fingerprint, manifest_sha256=manifest_hash,
                  variant=variant, summary=result['summary'], wall_s=result['wall_s'],
                  diagnostics=trace_diagnostics(result['trace']))
    # A completed case always has a saved trajectory, including unsuccessful cases.
    atomic_json(path.with_name(path.stem + '_trace.json'), result['trace'])
    atomic_json(path, record)
    return case.seed


def run_cases(items, cfg, variant, checkpoint, folder, manifest_hash, workers):
    folder.mkdir(parents=True, exist_ok=True)
    todo = []
    for item in items:
        case = case_from_dict(item['case'])
        path = folder / f'{case.seed}.json'
        if path.exists():
            stored = read(path)
            if (stored['fingerprint'] != case.fingerprint or stored['manifest_sha256'] != manifest_hash
                    or stored['variant'] != variant):
                raise ValueError(f'Cannot reuse mismatched case: {path}')
            if path.with_name(path.stem + '_trace.json').exists():
                continue
        todo.append((item, cfg, variant, checkpoint, path, manifest_hash))
    start = time.monotonic()
    completed = len(items) - len(todo)
    print(f'EVALUATE variant={variant} complete={completed}/{len(items)} workers={workers}', flush=True)
    with ProcessPoolExecutor(workers, initializer=initialize_worker,
                             mp_context=multiprocessing.get_context('spawn')) as executor:
        pending = {executor.submit(evaluate_one, task) for task in todo}
        while pending:
            done, pending = wait(pending, timeout=30., return_when=FIRST_COMPLETED)
            for future in done:
                future.result()
                completed += 1
            print(f'PROGRESS variant={variant} complete={completed}/{len(items)} '
                  f'elapsed_s={time.monotonic()-start:.1f}', flush=True)
    return [read(folder / f"{item['case']['seed']}.json") for item in items]


def screen(root, workers):
    protocol = verify_runtime(root)
    candidates = read(root / 'candidates.json')
    if sha(root / 'candidates.json') != protocol['candidates_sha256']:
        raise ValueError('Candidate pool changed after preparation')
    rows = run_cases(candidates, Config(**protocol['environment']), 'pg', None,
                     root / 'pg_candidates', sha(root / 'candidates.json'), workers)
    records = [dict(item, pg=row['summary'], pg_diagnostics=row['diagnostics'])
               for item, row in zip(candidates, rows)]
    selected = select_cases(records, protocol['target_cases'], protocol['target_pg_failure_fraction'])
    manifest = dict(cases=selected, protocol_sha256=sha(root / 'protocol.json'),
                    candidate_count=len(records), selected_count=len(selected),
                    candidate_pg=mean_metrics([r['pg'] for r in records]),
                    selected_pg=mean_metrics([r['pg'] for r in selected]),
                    selected_pg_failures=sum(not r['pg']['success'] for r in selected),
                    candidate_families={f: dict(count=sum(r['family'] == f for r in records),
                        **mean_metrics([r['pg'] for r in records if r['family'] == f])) for f in FAMILIES})
    atomic_json(root / 'manifest.json', manifest)
    print('SCREEN_COMPLETE ' + json.dumps({k: v for k, v in manifest.items() if k != 'cases'}), flush=True)


def evaluate(root, index, workers):
    protocol = verify_runtime(root)
    manifest = read(root / 'manifest.json')
    if manifest['protocol_sha256'] != sha(root / 'protocol.json'):
        raise ValueError('Protocol changed after PG screening')
    task = protocol['tasks'][index]
    checkpoint = root / task['checkpoint']
    if sha(checkpoint) != task['sha256']:
        raise ValueError('Frozen checkpoint changed')
    directory = root / task['variant'] / f"seed_{task['training_seed']}"
    rows = run_cases(manifest['cases'], Config(**protocol['environment']), task['variant'],
                     str(checkpoint), directory / 'cases', sha(root / 'manifest.json'), workers)
    pg = [item['pg'] for item in manifest['cases']]
    policy = [row['summary'] for row in rows]
    atomic_json(directory / 'results.json', dict(
        status='complete', variant=task['variant'], training_seed=task['training_seed'],
        best_update=task['best_update'], checkpoint_sha256=task['sha256'],
        manifest_sha256=sha(root / 'manifest.json'), unique_test_cases=len(pg),
        pg=pg, policy=policy, aggregate=dict(pg=mean_metrics(pg), policy=mean_metrics(policy))))
    print('EVALUATION_COMPLETE ' + json.dumps(mean_metrics(policy)), flush=True)


def collect(root):
    import csv
    protocol = verify_runtime(root)
    manifest = read(root / 'manifest.json')
    pg = [c['pg'] for c in manifest['cases']]
    if manifest['protocol_sha256'] != sha(root / 'protocol.json'):
        raise ValueError('Protocol changed after PG screening')
    found, missing = [], []
    for task in protocol['tasks']:
        path = root / task['variant'] / f"seed_{task['training_seed']}" / 'results.json'
        if not path.exists():
            missing.append(task)
            continue
        data = read(path)
        if (data['status'] != 'complete' or data['variant'] != task['variant'] or
                data['training_seed'] != task['training_seed'] or
                data['manifest_sha256'] != sha(root / 'manifest.json') or
                data['checkpoint_sha256'] != task['sha256'] or
                [r['fingerprint'] for r in data['policy']] != [r['fingerprint'] for r in pg]):
            raise ValueError(f'Unpaired or stale results: {path}')
        found.append(data)
    table = [dict(method='pg', training_seeds=0, unique_cases=len(pg), **mean_metrics(pg))]
    details = {}
    metrics = ('success', 'collision', 'cav_average_speed', 'delay_censored_s', 'episode_ttcp_lt_1p5')
    for variant in VARIANTS:
        runs = [r for r in found if r['variant'] == variant]
        if not runs:
            continue
        table.append(dict(method=variant, training_seeds=len(runs), unique_cases=len(pg),
                          **mean_metrics([p for r in runs for p in r['policy']])))
        paired = {}
        for metric in metrics:
            delta = np.array([[p[metric]-g[metric] for p, g in zip(r['policy'], pg)] for r in runs])
            rng = np.random.default_rng(109)
            boots = [float(delta[np.ix_(rng.integers(len(runs), size=len(runs)),
                                        rng.integers(len(pg), size=len(pg)))].mean()) for _ in range(2000)]
            paired[metric] = dict(mean=float(delta.mean()), ci95=np.quantile(boots, [.025, .975]).tolist(),
                                  per_training_seed=delta.mean(1).tolist())
        families = {}
        for family in FAMILIES:
            ix = [i for i, c in enumerate(manifest['cases']) if c['family'] == family]
            if ix:
                families[family] = dict(unique_cases=len(ix), pg=mean_metrics([pg[i] for i in ix]),
                    policy=mean_metrics([r['policy'][i] for r in runs for i in ix]))
        failures = [i for i, row in enumerate(pg) if not row['success']]
        successes = [i for i, row in enumerate(pg) if row['success']]
        details[variant] = dict(paired_vs_pg=paired, families=families,
            recovered_pg_failures=sum(r['policy'][i]['success'] for r in runs for i in failures),
            pg_failure_evaluations=len(failures)*len(runs),
            lost_pg_successes=sum(not r['policy'][i]['success'] for r in runs for i in successes),
            pg_success_evaluations=len(successes)*len(runs))
    payload = dict(complete=not missing, complete_runs=len(found), expected_runs=len(protocol['tasks']),
                   missing=missing, table=table, details=details,
                   interpretation=protocol['interpretation'], manifest_sha256=sha(root / 'manifest.json'))
    atomic_json(root / 'summary.json', payload)
    with (root / 'summary.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader(); writer.writerows(table)
    print(json.dumps(payload, indent=2), flush=True)
    if missing:
        raise SystemExit(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'screen', 'evaluate', 'collect'))
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--main-root', type=Path)
    parser.add_argument('--per-family', type=int, default=120)
    parser.add_argument('--index', type=int)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.action == 'prepare':
        if args.main_root is None or args.per_family < 40:
            parser.error('prepare requires --main-root and at least 40 candidates per family')
        prepare(args.main_root, args.root, args.per_family)
    elif args.action == 'screen':
        screen(args.root, args.workers)
    elif args.action == 'evaluate':
        if args.index is None or not 0 <= args.index < len(VARIANTS)*len(SEEDS):
            parser.error('evaluate requires a valid --index')
        evaluate(args.root, args.index, args.workers)
    else:
        collect(args.root)


if __name__ == '__main__':
    main()
