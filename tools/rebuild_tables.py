"""Rebuild report tables from committed reference records or from new simulation runs.

    python -m tools.rebuild_tables --suite main --output-dir outputs/tables/main
    python -m tools.rebuild_tables --suite main --runs-dir outputs/main --output-dir outputs/main/tables

Writes ``<suite>.csv`` (one full-precision row per run), ``<suite>.md`` (summary tables) and, for the main
suite, ``latency_vs_bandwidth.svg/png``. Latency includes the VS scan; area is the known-partial area of
``tools.area_model``, not a total chip area.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .analysis import derive
from .common import ROOT, save
from .run_simulation import REFRESH
from .suites import ALL_BW, REFERENCE, SUITES

MAIN_ORDER = ['Dense D4', 'Dense D8', 'Dense D16', 'Dense D32', 'POP-PIM', 'POP-PIM-Q', 'POP-PIM-S']


def load(suite, runs_dir=None):
    path = Path(runs_dir) / 'runs.json' if runs_dir else ROOT / 'results' / REFERENCE[suite] / 'runs.json'
    records = json.loads(path.read_text())
    if not runs_dir and suite == 'quick':
        wanted = {c.name for c in SUITES['quick']()}
        records = [r for r in records if r['name'] in wanted]
    return path, records


def write_csv(path, rows):
    keys = []
    for row in rows:
        keys += [k for k in row if k not in keys]
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows({k: row.get(k, '') for k in keys} for row in rows)


def table(head, rows, align=None):
    align = align or ['---'] + ['---:'] * (len(head) - 1)
    return '\n'.join(['| ' + ' | '.join(head) + ' |', '| ' + ' | '.join(align) + ' |']
                     + ['| ' + ' | '.join(map(str, r)) + ' |' for r in rows]) + '\n'


def by(rows, *keys):
    return {tuple(r[k] for k in keys): r for r in rows}


def main_report(rows):
    bws = [b for b in ALL_BW if any(r['bandwidth_GBs'] == b for r in rows)]
    names = [n for n in MAIN_ORDER if any(r['configuration'] == n for r in rows)]
    idx = by(rows, 'configuration', 'bandwidth_GBs')
    out = ['# Main comparison', '',
           'One Llama-3.1-8B attention layer, one decode token, N = 32,768, one HBM3 channel (64 banks), per-bank '
           '(PIM-aware) refresh. POP-PIM rows use the S path with the measured 8-thread CPU Top-k. Latency includes '
           'the VS threshold scan. Area is the known partial area per channel (not total chip area).', '',
           '## Attention latency (ms)', '',
           table(['Configuration'] + [f'{b:g} GB/s' for b in bws],
                 [[n] + [f"{idx[(n, b)]['latency_ms']:.3f}" if (n, b) in idx else '' for b in bws] for n in names])]
    if ('Dense D8', bws[0]) in idx:
        out += ['## Speed-up over Dense D8 (latency ratio)', '',
                table(['Configuration'] + [f'{b:g} GB/s' for b in bws],
                      [[n] + [f"{idx[('Dense D8', b)]['latency_ms'] / idx[(n, b)]['latency_ms']:.2f}x"
                              if (n, b) in idx and ('Dense D8', b) in idx else '' for b in bws] for n in names])]
    first, last = bws[0], bws[-1]
    out += ['## Bytes, area and area x delay', '',
            table(['Configuration', 'Channel MiB', 'Host link MiB', 'Known partial area (mm²)',
                   f'A×D @{first:g} (mm²·ms)', f'A×D @{last:g} (mm²·ms)'],
                  [[n, f"{idx[(n, first)]['channel_MiB']:.2f}", f"{idx[(n, first)]['host_link_MiB']:.3f}",
                    f"{idx[(n, first)]['known_partial_area_mm2']:.3f}", f"{idx[(n, first)]['area_delay_mm2_ms']:.1f}",
                    f"{idx[(n, last)]['area_delay_mm2_ms']:.1f}" if (n, last) in idx else '']
                   for n in names if (n, first) in idx])]
    return '\n'.join(out)


def ablation_report(rows):
    bws = [b for b in ALL_BW if any(r['bandwidth_GBs'] == b for r in rows)]
    out = ['# GS / VS / CT ablation', '',
           'Same workload and per-bank refresh as the main comparison. S path: host Top-k (GPU 0.2 ms assumed, or the '
           'measured CPU Top-k); L path: local E24/P32 selector. `h` = hardware VS rule (RTL), otherwise the exact '
           '(ideal) rule. Latency includes the hardware VS scan.', '']
    for design, title in (('POP_S_GPU', 'S path, GPU Top-k (assumed 0.2 ms)'), ('POP_S_CPU', 'S path, measured CPU Top-k'),
                          ('POP_E24P32', 'L path, local selector E24/P32')):
        sel = [r for r in rows if r['design'] == design and r['host_link_GBs'] in ('', 32.0)]
        if not sel:
            continue
        idx = by(sel, 'variant', 'bandwidth_GBs')
        variants = list(dict.fromkeys(r['variant'] for r in sel))
        out += [f'## {title}: latency (ms) and channel MiB at {bws[0]:g} GB/s', '',
                table(['Variant'] + [f'{b:g} GB/s' for b in bws] + ['Channel MiB', 'Area (mm²)'],
                      [[v] + [f"{idx[(v, b)]['latency_ms']:.3f}" if (v, b) in idx else '' for b in bws]
                       + [f"{idx[(v, bws[0])]['channel_MiB']:.2f}" if (v, bws[0]) in idx else '',
                          f"{idx[(v, bws[0])]['known_partial_area_mm2']:.3f}" if (v, bws[0]) in idx else '']
                       for v in variants])]
    link = [r for r in rows if r['design'] == 'POP_S_GPU' and r['bandwidth_GBs'] == bws[0]]
    links = sorted({r['host_link_GBs'] for r in link}, reverse=True)
    if len(links) > 1:
        idx = by(link, 'variant', 'host_link_GBs')
        variants = list(dict.fromkeys(r['variant'] for r in link if r['host_link_GBs'] != 32.0))
        out += [f'## Host link sweep (S path, GPU Top-k, {bws[0]:g} GB/s channel): latency (ms)', '',
                table(['Variant'] + [f'{x:g} GB/s link' for x in links] + ['Host link MiB'],
                      [[v] + [f"{idx[(v, x)]['latency_ms']:.3f}" if (v, x) in idx else '' for x in links]
                       + [f"{idx[(v, links[0])]['host_link_MiB']:.3f}"] for v in variants])]
    return '\n'.join(out)


def refresh_report(rows):
    policies = [p for p in REFRESH if any(r['refresh'] == p for r in rows)]
    designs = list(dict.fromkeys(r['design'] for r in rows))
    idx = by(rows, 'design', 'refresh')
    return '\n'.join(['# Refresh policies', '',
                      'Attention latency (ms) at 41.6 GB/s, N = 32,768. `off` removes refresh entirely and is an '
                      'optimistic bound; `all-bank` is the strict REFab controller of the 0.1.0 comparison; `per-bank` '
                      'is the PIM-aware single-bank refresh used by the main results (round rule kept).', '',
                      table(['Design'] + policies,
                            [[d] + [f"{idx[(d, p)]['latency_ms']:.3f}" if (d, p) in idx else '' for p in policies]
                             for d in designs]),
                      'Known partial area (mm²) with the per-bank scheduler: '
                      + ', '.join(f"{d} {idx[(d, 'per-bank')]['known_partial_area_mm2']:.3f}"
                                  for d in designs if (d, 'per-bank') in idx) + '.', ''])


def quick_report(rows):
    return '\n'.join(['# Quick check', '',
                      table(['Configuration', 'Bandwidth (GB/s)', 'Latency (ms)', 'Channel MiB', 'Area (mm²)'],
                            [[r['configuration'], f"{r['bandwidth_GBs']:g}", f"{r['latency_ms']:.3f}",
                              f"{r['channel_MiB']:.2f}", f"{r['known_partial_area_mm2']:.3f}"] for r in rows])])


REPORTS = dict(main=main_report, ablation=ablation_report, refresh=refresh_report, quick=quick_report)


def rebuild(suite, output_dir, runs_dir=None, figures=True):
    source, records = load(suite, runs_dir)
    rows = [derive(r) for r in records]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / f'{suite}.csv', rows)
    (output_dir / f'{suite}.md').write_text(REPORTS[suite](rows))
    if figures and suite == 'main':
        from .plot_comparison import plot_main
        plot_main(rows, output_dir / 'latency_vs_bandwidth')
    save(output_dir / 'scope.json', dict(suite=suite, records=len(rows), source=str(source),
         latency='simulated attention output time plus the hardware VS scan (125 MHz)',
         area='known partial area per channel (tools.area_model); total chip area and energy are not reported'))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--suite', choices=list(REPORTS), default='main')
    parser.add_argument('--runs-dir', type=Path, help='output directory of a new run; default: committed reference')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--no-figures', action='store_true')
    args = parser.parse_args(argv)
    rows = rebuild(args.suite, args.output_dir, args.runs_dir, not args.no_figures)
    print(f'Rebuilt {len(rows)} rows in {args.output_dir}')


if __name__ == '__main__':
    main()
