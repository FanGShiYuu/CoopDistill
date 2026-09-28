"""Shared-actor MAPPO with centralized value function and masked agent PPO.

CPU workers advance independent geometric episodes; the requested CUDA device
performs minibatch learning. Raw Gaussian actions are stored before the tanh
decoder, so the PPO likelihood is consistent with the collected policy.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict, replace
import hashlib
import json
import os
import multiprocessing
from pathlib import Path
import signal
import time

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from .environment import Config, IntersectionEnv
from .scenarios import Case, MAX_VEHICLES, make_case, suite

VARIANTS = ('full', 'no_lateral', 'no_curriculum', 'no_fallback',
            'no_warmstart', 'no_residual')
STOP = False


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(44, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 3))
        self.log_std = nn.Parameter(torch.full((3,), -.8))
        nn.init.orthogonal_(self.net[-1].weight, gain=.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        mean = self.net(x)
        return Normal(mean, self.log_std.clamp(-3., .5).exp().expand_as(mean))


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(8*44, 192), nn.Tanh(), nn.Linear(192, 128), nn.Tanh(), nn.Linear(128, 1))

    def forward(self, x):
        return self.net(x.flatten(start_dim=1)).squeeze(-1)


def dimensions(variant):
    active = np.ones(3, dtype=np.float32)
    if variant == 'no_lateral':
        active[2] = 0.
    if variant == 'no_residual':
        active[1] = 0.  # Direct acceleration has no PG timing correction.
    return active


def mode(variant):
    return {'no_fallback': 'no_fallback', 'no_residual': 'absolute'}.get(variant, 'full')


def initialize_worker():
    torch.set_num_threads(1)


def rollout(task):
    case, cfg, variant, weights, random_seed, deterministic, record = task
    torch.manual_seed(random_seed)
    torch.set_num_threads(1)
    env = IntersectionEnv(case, cfg, record=record)
    actor = Actor()
    if weights is not None:
        actor.load_state_dict(weights)
    actor.eval()
    dim = torch.as_tensor(dimensions(variant))
    observations, masks, actions, logps, rewards = [], [], [], [], []
    start = time.monotonic()
    with torch.no_grad():
        while not env.done:
            obs = env.observe(include_pg=True)
            mask = env.mask()
            if variant == 'pg':
                action = np.zeros((8, 3), dtype=np.float32)
                logp = np.zeros(8)
                reward, _ = env.step(mode='pg')
            else:
                dist = actor(torch.as_tensor(obs))
                raw = dist.mean if deterministic else dist.sample()
                action = (raw*dim).numpy()
                logp = (dist.log_prob(raw)*dim).sum(-1).numpy()
                reward, _ = env.step(action, mode=mode(variant), lateral=variant != 'no_lateral')
            observations.append(obs)
            masks.append(mask)
            actions.append(action)
            logps.append(logp)
            rewards.append(reward)
    return dict(obs=np.asarray(observations), mask=np.asarray(masks), actions=np.asarray(actions),
                logp=np.asarray(logps), rewards=np.asarray(rewards, dtype=np.float32),
                summary=env.summary(), wall_s=time.monotonic()-start,
                trace={'case': case.to_dict(), 'history': env.history} if record else None)


def pg_scan(task):
    case, cfg = task
    return rollout((case, cfg, 'pg', None, case.seed, True, False))['summary']


def warm_examples(task):
    case, cfg, variant, samples, max_states = task
    env = IntersectionEnv(case, cfg)
    rng = np.random.default_rng(case.seed+451)
    observations, labels, weights = [], [], []
    examined = improved = 0
    # Sample real PG states at fixed intervals, not hand-labelled deadlock actions.
    stride = max(1, int(cfg.horizon_s/(cfg.dt*cfg.decision_steps*max_states)))
    step = 0
    while not env.done:
        if step % stride == 0 and examined < max_states:
            pg = env.pg()[0]
            reference = env.preview(pg)
            best_score = reference.progress_score()
            best = np.zeros((8, 3), dtype=np.float32)
            if variant == 'no_residual':
                best[:env.n, 0] = np.arctanh(np.clip((pg+1.)/3., -.995, .995))
            found = False
            candidates = rng.normal(0., .85, size=(samples, 8, 3)).astype(np.float32)
            for raw in candidates:
                raw *= dimensions(variant)
                candidate, acceleration = env.proposal(raw, lateral=variant != 'no_lateral', absolute=variant == 'no_residual')
                prediction = candidate.preview(acceleration)
                if (prediction.collision or prediction.boundary or prediction.dynamic_violation):
                    continue
                if prediction.progress_score() > best_score+1e-4:
                    best, best_score, found = raw.copy(), prediction.progress_score(), True
            obs = env.observe()
            for i in np.flatnonzero(env.mask()):
                observations.append(obs[i]); labels.append(best[i]); weights.append(1. if found else .2)
            examined += 1
            improved += int(found)
        env.step(mode='pg')
        step += 1
    return dict(obs=np.asarray(observations), labels=np.asarray(labels), weights=np.asarray(weights),
                examined=examined, improved=improved)


def map_tasks(executor, function, tasks):
    if not executor:
        return [function(t) for t in tasks]
    pending = {executor.submit(function, t): i for i, t in enumerate(tasks)}
    results = [None]*len(tasks)
    start = time.monotonic()
    last_log = start
    while pending:
        done, _ = wait(pending, timeout=60., return_when=FIRST_COMPLETED)
        for future in done:
            index = pending.pop(future)
            results[index] = future.result()
        now = time.monotonic()
        if now-last_log >= 60.:
            print(f'WORKERS phase={function.__name__} completed={len(tasks)-len(pending)}/{len(tasks)} '
                  f'elapsed_s={now-start:.1f}', flush=True)
            last_log = now
    return results


def gae(rewards, values, gamma=.99, lam=.95):
    # The finite-horizon task ends at its specified deadline, not an arbitrary
    # training truncation; time remaining is observed, terminal value is zero.
    advantage = np.zeros_like(rewards)
    running = 0.
    for t in reversed(range(len(rewards))):
        next_v = values[t+1] if t+1 < len(values) else 0.
        delta = rewards[t]+gamma*next_v-values[t]
        running = delta+gamma*lam*running
        advantage[t] = running
    return advantage, advantage+values


def update(actor, critic, actor_opt, critic_opt, episodes, variant, device, epochs, minibatch, anchor=None):
    obs = torch.as_tensor(np.concatenate([e['obs'] for e in episodes]), device=device)
    mask = torch.as_tensor(np.concatenate([e['mask'] for e in episodes]), device=device)
    acts = torch.as_tensor(np.concatenate([e['actions'] for e in episodes]), device=device)
    oldlp = torch.as_tensor(np.concatenate([e['logp'] for e in episodes]), device=device, dtype=torch.float32)
    with torch.no_grad():
        values = critic(obs).cpu().numpy()
    advantages, returns, at = [], [], 0
    for e in episodes:
        n = len(e['rewards'])
        a, r = gae(e['rewards'], values[at:at+n])
        advantages.extend(a); returns.extend(r); at += n
    adv = torch.as_tensor(np.asarray(advantages), device=device, dtype=torch.float32)
    ret = torch.as_tensor(np.asarray(returns), device=device, dtype=torch.float32)
    live = mask.any(-1)
    if live.any():
        adv = (adv-adv[live].mean())/(adv[live].std(unbiased=False)+1e-6)
    dim = torch.as_tensor(dimensions(variant), device=device)
    logs = []
    for _ in range(epochs):
        order = torch.randperm(len(obs), device=device)
        for start in range(0, len(obs), minibatch):
            idx = order[start:start+minibatch]
            dist = actor(obs[idx])
            lp = (dist.log_prob(acts[idx])*dim).sum(-1)
            ratio = (lp-oldlp[idx]).exp()
            unclipped = ratio*adv[idx, None]
            clipped = ratio.clamp(.8, 1.2)*adv[idx, None]
            valid = mask[idx].float()
            count = valid.sum().clamp_min(1.)
            entropy = ((dist.entropy()*dim).sum(-1)*valid).sum()/count
            policy = -(torch.minimum(unclipped, clipped)*valid).sum()/count-.005*entropy
            imitation = torch.zeros((), device=device)
            if anchor is not None:
                ao, al, aw = anchor
                draw = torch.randint(len(ao), (min(minibatch, len(ao)),), device=device)
                imitation = (((actor(ao[draw]).mean-al[draw])**2*dim).mean(-1)*aw[draw]).mean()
                policy = policy+.02*imitation
            value_loss = .5*(critic(obs[idx])-ret[idx]).square().mean()
            actor_opt.zero_grad(set_to_none=True); policy.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), .5); actor_opt.step()
            critic_opt.zero_grad(set_to_none=True); value_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 1.); critic_opt.step()
            with torch.no_grad():
                kl = (((ratio-1)-(lp-oldlp[idx]))*valid).sum()/count
                clipfrac = (((ratio-1).abs()>.2)*valid).sum()/count
            logs.append([policy.item(), value_loss.item(), entropy.item(), kl.item(), clipfrac.item(), imitation.item()])
    names = ('actor_loss', 'critic_loss', 'entropy', 'approx_kl', 'clip_fraction', 'anchor_loss')
    return dict(zip(names, np.mean(logs, axis=0).tolist()))


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def weights_cpu(actor):
    return {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}


def mean_metrics(rows):
    keys = ('collision', 'total_collision', 'success', 'cleared_fraction', 'cav_average_speed',
            'delay_censored_s', 'episode_ttcp_lt_1p5', 'episode_ttcp_lt_0p5',
            'boundary', 'dynamic_violation', 'lateral_replans', 'accept_rate', 'pg_solver_failures')
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def evaluate(cases, cfg, variant, actor, executor, trace=False):
    weights = weights_cpu(actor) if actor is not None else None
    tasks = [(c, cfg, variant, weights, c.seed, True, trace and index in (0, len(cases)//2, len(cases)-1))
             for index, c in enumerate(cases)]
    rows = map_tasks(executor, rollout, tasks)
    return rows


def validation_rank(metrics):
    return (-metrics['collision']-metrics['boundary']-metrics['dynamic_violation'],
            metrics['success'], -metrics['delay_censored_s'], metrics['cav_average_speed'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=VARIANTS, default='full')
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--updates', type=int, default=240)
    parser.add_argument('--episodes', type=int, default=12)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--ppo-epochs', type=int, default=4)
    parser.add_argument('--minibatch', type=int, default=256)
    parser.add_argument('--pool', type=int, default=120)
    parser.add_argument('--hard-count', type=int, default=32)
    parser.add_argument('--warm-cases', type=int, default=12)
    parser.add_argument('--warm-states', type=int, default=8)
    parser.add_argument('--warm-candidates', type=int, default=8)
    parser.add_argument('--warm-epochs', type=int, default=16)
    parser.add_argument('--val-every', type=int, default=30)
    parser.add_argument('--test-per-stratum', type=int, default=3)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if not 0 <= args.seed < 100:
        parser.error('seed must be in [0,100) to keep manifest namespaces disjoint')
    if args.updates*args.episodes >= 50000:
        parser.error('training seed namespace exhausted')
    if args.smoke:
        args.updates, args.episodes, args.pool, args.hard_count = 1, 2, 3, 1
        args.warm_cases, args.warm_states, args.warm_candidates, args.warm_epochs = 1, 1, 2, 1
        args.val_every, args.test_per_stratum = 1, 1
    device = torch.device(args.device)
    torch.set_num_threads(1)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA, but no GPU is available. No silent CPU fallback.')
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    cfg = Config(horizon_s=2., preview_steps=1) if args.smoke else Config()
    val_cases = suite('validation', 1)
    test_cases = suite('test', args.test_per_stratum)
    if args.smoke:
        val_cases, test_cases = val_cases[:1], test_cases[:2]
    sources = sorted(Path(__file__).parent.glob('*.py'))
    code_hash = hashlib.sha256(b''.join(p.read_bytes() for p in sources)).hexdigest()
    protocol = dict(args=vars(args), environment=asdict(cfg), code_sha256=code_hash,
                    validation=[c.to_dict() for c in val_cases], test=[c.to_dict() for c in test_cases],
                    note='Finite prediction safety checks are not a global non-degradation guarantee.')
    # JSON turns dataclass tuples into lists. Compare the same representation
    # on resume instead of incorrectly treating that conversion as a change.
    protocol = json.loads(json.dumps(protocol))
    protocol_path = out/'protocol.json'
    if protocol_path.exists():
        previous = json.loads(protocol_path.read_text())
        if previous != protocol:
            raise RuntimeError('Refusing to mix different protocols/code in one run directory')
    atomic_json(protocol_path, protocol)
    print(f'START variant={args.variant} seed={args.seed} device={device} workers={args.workers}', flush=True)
    actor, critic = Actor().to(device), Critic().to(device)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=3e-4)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=1e-3)
    executor = ProcessPoolExecutor(args.workers, initializer=initialize_worker,
                                   mp_context=multiprocessing.get_context('spawn')) if args.workers else None
    global STOP
    def stop_handler(*_):
        global STOP
        STOP = True
        print('STOP requested: finishing current update and saving checkpoint.', flush=True)
    signal.signal(signal.SIGTERM, stop_handler)
    if hasattr(signal, 'SIGUSR1'):
        signal.signal(signal.SIGUSR1, stop_handler)
    started = time.monotonic()
    try:
        pool_path = out/'curriculum_pool.json'
        train_base = 1_000_000+args.seed*100_000
        if pool_path.exists():
            pool_rows = json.loads(pool_path.read_text())
        else:
            print(f'CURRICULUM reference scan cases={args.pool}', flush=True)
            pool_cases = [make_case(train_base+i) for i in range(args.pool)]
            pool_rows = map_tasks(executor, pg_scan, [(c, cfg) for c in pool_cases])
            atomic_json(pool_path, pool_rows)
        ranked = sorted(pool_rows, key=lambda r: (r['collision'], 1-r['success'], r['delay_censored_s']), reverse=True)
        hard_seeds = [r['seed'] for r in ranked[:args.hard_count]]
        warm_seeds = [train_base+50_000+i for i in range(args.warm_cases)]
        anchor = None
        if args.variant != 'no_warmstart':
            data_path = out/'warmstart.npz'
            if data_path.exists():
                with np.load(data_path) as data:
                    arrays = [data[k] for k in ('obs', 'labels', 'weights')]
            else:
                print(f'WARMSTART search cases={len(warm_seeds)} candidates={args.warm_candidates}', flush=True)
                warm = map_tasks(executor, warm_examples,
                    [(make_case(s), cfg, args.variant, args.warm_candidates, args.warm_states) for s in warm_seeds])
                arrays = [np.concatenate([w[k] for w in warm]) for k in ('obs', 'labels', 'weights')]
                np.savez_compressed(data_path, **dict(zip(('obs', 'labels', 'weights'), arrays)))
                atomic_json(out/'warmstart_statistics.json', dict(states=sum(w['examined'] for w in warm),
                            improved=sum(w['improved'] for w in warm), examples=len(arrays[0])))
            anchor = tuple(torch.as_tensor(a, dtype=torch.float32, device=device) for a in arrays)
        checkpoint_path = out/'last.pt'
        first_update, best_rank, best_update = 0, None, -1
        best_state = weights_cpu(actor)
        if checkpoint_path.exists():
            saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
            actor.load_state_dict(saved['actor']); critic.load_state_dict(saved['critic'])
            actor_opt.load_state_dict(saved['actor_opt']); critic_opt.load_state_dict(saved['critic_opt'])
            first_update, best_rank, best_update = saved['update'], saved['best_rank'], saved['best_update']
            best_state = {k: v.cpu() for k, v in saved['best_actor'].items()}
            rng.bit_generator.state = saved['numpy_rng']
            torch.set_rng_state(saved['torch_rng'].cpu())
            if device.type == 'cuda' and saved.get('cuda_rng') is not None:
                torch.cuda.set_rng_state(saved['cuda_rng'].cpu())
            print(f'RESUME update={first_update}', flush=True)
        else:
            if anchor is not None:
                ao, al, aw = anchor
                for epoch in range(args.warm_epochs):
                    order = torch.randperm(len(ao), device=device)
                    losses = []
                    for start in range(0, len(ao), args.minibatch):
                        idx = order[start:start+args.minibatch]
                        dim = torch.as_tensor(dimensions(args.variant), device=device)
                        loss = (((actor(ao[idx]).mean-al[idx])**2*dim).mean(-1)*aw[idx]).mean()
                        actor_opt.zero_grad(set_to_none=True); loss.backward(); actor_opt.step()
                        losses.append(loss.item())
                    print(f'WARMSTART epoch={epoch+1} mse={np.mean(losses):.6f}', flush=True)
        def checkpoint(completed):
            temporary = out/'last.pt.tmp'
            torch.save(dict(actor=actor.state_dict(), critic=critic.state_dict(), actor_opt=actor_opt.state_dict(),
                critic_opt=critic_opt.state_dict(), update=completed, best_rank=best_rank,
                best_update=best_update, best_actor=best_state, numpy_rng=rng.bit_generator.state,
                torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state() if device.type == 'cuda' else None), temporary)
            temporary.replace(checkpoint_path)
        completed = first_update
        for update_idx in range(first_update, args.updates):
            tick = time.monotonic()
            fraction = .25+.25*update_idx/max(args.updates-1, 1)
            cases = []
            for k in range(args.episodes):
                s = train_base+1000+update_idx*args.episodes+k
                if args.variant != 'no_curriculum' and rng.random() < fraction:
                    s = int(rng.choice(hard_seeds))
                cases.append(make_case(s))
            weights = weights_cpu(actor)
            tasks = [(c, cfg, args.variant, weights, args.seed*100000+update_idx*args.episodes+k, False, False)
                     for k, c in enumerate(cases)]
            episodes = map_tasks(executor, rollout, tasks)
            statistics = update(actor, critic, actor_opt, critic_opt, episodes, args.variant, device,
                                args.ppo_epochs, args.minibatch, anchor)
            statistics.update(update=update_idx+1, environment_steps=sum(len(e['rewards']) for e in episodes),
                              mean_return=float(np.mean([e['rewards'].sum() for e in episodes])),
                              seconds=time.monotonic()-tick, training=mean_metrics([e['summary'] for e in episodes]))
            if (update_idx+1) % args.val_every == 0 or update_idx == 0 or update_idx+1 == args.updates:
                val = evaluate(val_cases, cfg, args.variant, actor, executor)
                metrics = mean_metrics([r['summary'] for r in val])
                atomic_json(out/'validation'/f'update_{update_idx+1:04d}.json',
                            [r['summary'] for r in val])
                rank = validation_rank(metrics)
                statistics['validation'] = metrics
                if best_rank is None or rank > tuple(best_rank):
                    best_state, best_rank, best_update = weights_cpu(actor), rank, update_idx+1
                    atomic_json(out/'best_validation.json', dict(update=best_update, metrics=metrics))
                print(f'VALIDATION update={update_idx+1} {json.dumps(metrics)}', flush=True)
            with (out/'training.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(statistics)+'\n')
            completed = update_idx+1
            checkpoint(completed)
            eta = statistics['seconds']*(args.updates-completed)/3600
            print(f'TRAIN update={completed}/{args.updates} return={statistics["mean_return"]:.4f} '
                  f'seconds={statistics["seconds"]:.1f} rough_train_hours_left={eta:.2f}', flush=True)
            if STOP:
                print('CHECKPOINTED; resume with identical command.', flush=True)
                return
        actor.load_state_dict(best_state)
        torch.save(best_state, out/'best_actor.pt')
        print(f'TEST frozen_checkpoint_update={best_update} unique_cases={len(test_cases)}', flush=True)
        pg = evaluate(test_cases, cfg, 'pg', None, executor, trace=True)
        policy = evaluate(test_cases, cfg, args.variant, actor, executor, trace=True)
        for label, rows in (('pg', pg), ('policy', policy)):
            for row in rows:
                if row['trace'] is not None:
                    atomic_json(out/'trajectories'/f'{label}_{row["summary"]["seed"]}.json', row['trace'])
        result = dict(status='complete', variant=args.variant, training_seed=args.seed, best_update=best_update,
                      completed_updates=completed, unique_test_cases=len(test_cases), protocol_sha256=code_hash,
                      elapsed_s=time.monotonic()-started,
                      pg=[e['summary'] for e in pg], policy=[e['summary'] for e in policy],
                      aggregate={'pg': mean_metrics([e['summary'] for e in pg]),
                                 'policy': mean_metrics([e['summary'] for e in policy])})
        atomic_json(out/'results.json', result)
        print('COMPLETE '+json.dumps(result['aggregate']), flush=True)
    finally:
        if executor:
            executor.shutdown(wait=True, cancel_futures=True)


if __name__ == '__main__':
    main()
