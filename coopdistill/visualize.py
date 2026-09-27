"""Render predefined test traces without editing trajectories or choosing winners."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.animation import FuncAnimation, PillowWriter
from .environment import Config, rectangle_corners


def draw_map(ax):
    ax.set_facecolor('#e8ece9')
    ax.add_patch(Rectangle((-95, -10.5), 190, 21, color='#d1d5d8'))
    ax.add_patch(Rectangle((-10.5, -95), 21, 190, color='#d1d5d8'))
    ax.add_patch(Rectangle((-14, -14), 28, 28, color='#d1d5d8'))
    for offset in (-7, -3.5, 3.5, 7):
        for side in (-1, 1):
            ax.plot([side*14, side*95], [offset, offset], '--', c='white', lw=.8)
            ax.plot([offset, offset], [side*14, side*95], '--', c='white', lw=.8)
    for side in (-1, 1):
        ax.plot([side*14, side*95], [0, 0], c='#d9a62d', lw=1.1)
        ax.plot([0, 0], [side*14, side*95], c='#d9a62d', lw=1.1)
    ax.set_aspect('equal'); ax.set_xlim(-65, 65); ax.set_ylim(-65, 65)
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')


def render(pg_file, policy_file, output):
    records = [json.loads(Path(p).read_text()) for p in (pg_file, policy_file)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    for ax, label, record in zip(axes, ('PG', 'PG + residual MAPPO'), records):
        draw_map(ax)
        ax.set_title(label)
        poses = np.asarray([r['poses'] for r in record['history']])
        for i, spec in enumerate(record['case']['vehicles']):
            color = plt.cm.tab10(i)
            ax.plot(poses[:, i, 0], poses[:, i, 1], c=color, lw=1.6,
                    label=f'{i}: {spec["origin"]}->{spec["destination"]} '+('CAV' if spec['cav'] else 'HDV'))
            ax.add_patch(Polygon(rectangle_corners(poses[0, i], Config()), fc=color, ec='black', lw=.5))
        ax.legend(fontsize=6, loc='upper right')
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix('.png'), dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    patches = []
    for ax, label, record in zip(axes, ('PG', 'PG + residual MAPPO'), records):
        draw_map(ax); ax.set_title(label)
        group = []
        for i, spec in enumerate(record['case']['vehicles']):
            patch = Polygon(np.zeros((4, 2)), fc=plt.cm.tab10(i), ec='black' if spec['cav'] else 'white', lw=1.)
            ax.add_patch(patch); group.append(patch)
        patches.append(group)
    count = max(len(r['history']) for r in records)
    def frame(k):
        artists = []
        for ax, group, record, label in zip(axes, patches, records, ('PG', 'PG + residual MAPPO')):
            state = record['history'][min(k, len(record['history'])-1)]
            ax.set_title(f'{label}: t={state["time"]:.1f}s')
            for i, patch in enumerate(group):
                patch.set_xy(rectangle_corners(state['poses'][i], Config()))
                patch.set_visible(not state['finished'][i]); artists.append(patch)
        return artists
    animation = FuncAnimation(fig, frame, frames=range(0, count, 3), interval=100)
    animation.save(output.with_suffix('.gif'), writer=PillowWriter(fps=10))
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('directory'); args = p.parse_args()
    directory = Path(args.directory)
    for pg in sorted((directory/'trajectories').glob('pg_*.json')):
        policy = pg.with_name(pg.name.replace('pg_', 'policy_', 1))
        if policy.exists():
            render(pg, policy, directory/'figures'/pg.stem.replace('pg_', 'case_'))
