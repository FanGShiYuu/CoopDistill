"""A small, timed geometry/PG preflight, not evidence of policy improvement."""
import argparse
import json
import time
from pathlib import Path
from .environment import IntersectionEnv
from .scenarios import make_case


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', type=int, default=5)
    parser.add_argument('--output', default='intersection_preflight.json')
    args = parser.parse_args()
    rows = []
    start = time.monotonic()
    for index in range(args.cases):
        env = IntersectionEnv(make_case(5000+index, count=4+index % 5,
                                       density='dense' if index % 2 else 'ordinary'))
        while not env.done:
            env.step(mode='pg')
        rows.append(env.summary())
        print(json.dumps(rows[-1]), flush=True)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'elapsed_s': time.monotonic()-start, 'rows': rows}, indent=2))
    if any(row['boundary'] or row['dynamic_violation'] for row in rows):
        raise SystemExit('Reference-route geometry/actuation failed; do not train.')


if __name__ == '__main__':
    main()
