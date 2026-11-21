"""Plot attention latency against channel bandwidth for the main comparison (SVG and PNG).

    python -m tools.plot_comparison --output-dir docs/assets            # from the committed reference
    python -m tools.plot_comparison --runs-dir outputs/main --output-dir outputs/main/tables
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, INK_2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
# Dense designs are context (neutral ink); POP-PIM configurations take categorical slots 1-3.
STYLE = {
    'Dense D4': dict(color='#9b9a95', linestyle=(0, (4, 3)), marker='s'),
    'Dense D8': dict(color='#52514e', linestyle='-', marker='s'),
    'POP-PIM': dict(color='#1baf7a', linestyle='-', marker='o'),
    'POP-PIM-Q': dict(color='#eb6834', linestyle='-', marker='o'),
    'POP-PIM-S': dict(color='#2a78d6', linestyle='-', marker='o'),
}


def plot_main(rows, stem):
    """``rows`` are tools.analysis.derive() rows of the main suite; writes ``stem``.svg and ``stem``.png."""
    series = {}
    for row in rows:
        if row['configuration'] in STYLE:
            series.setdefault(row['configuration'], []).append((row['bandwidth_GBs'], row['latency_ms']))
    fig, ax = plt.subplots(figsize=(7.2, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for name, style in STYLE.items():
        if name not in series:
            continue
        points = sorted(series[name], reverse=True)
        x, y = zip(*points)
        ax.plot(x, y, linewidth=2, markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5, label=name, **style)
        ax.annotate(name, (x[-1], y[-1]), xytext=(8, 0), textcoords='offset points', va='center',
                    fontsize=9, color=INK_2)
    ax.set_xscale('log', base=2)
    ax.invert_xaxis()
    ticks = sorted({row['bandwidth_GBs'] for row in rows}, reverse=True)
    ax.set_xticks(ticks, [f'{t:g}' for t in ticks])
    ax.minorticks_off()
    ax.set_xlim(ticks[0] * 1.08, ticks[-1] / 1.55)
    ax.set_ylim(0, max(r['latency_ms'] for r in rows if r['configuration'] in STYLE) * 1.08)
    ax.set_xlabel('HBM3 channel bandwidth (GB/s)', color=INK_2)
    ax.set_ylabel('Attention latency (ms)', color=INK_2)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.legend(frameon=False, fontsize=9, loc='upper left', labelcolor=INK_2)
    ax.set_title('One attention layer, one decode token, N = 32K (D16/D32 stay within 3% of D8)',
                 fontsize=10, color=INK, loc='left')
    fig.tight_layout()
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in ('svg', 'png'):
        fig.savefig(stem.with_suffix(f'.{extension}'), dpi=180, facecolor=SURFACE)
    plt.close(fig)


def main(argv=None):
    from .analysis import derive
    from .rebuild_tables import load
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs-dir', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    _, records = load('main', args.runs_dir)
    plot_main([derive(r) for r in records], args.output_dir / 'latency_vs_bandwidth')
    print(args.output_dir / 'latency_vs_bandwidth.svg')


if __name__ == '__main__':
    main()
