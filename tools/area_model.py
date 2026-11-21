"""Known-partial area model (mm^2 per HBM3 channel, 64 banks) for Dense and POP-PIM designs.

"Known partial" means: freshly mapped Nangate45 cells of the near-bank scorer logic, plus the retained
SRAM/queue estimates and proxies of the supporting blocks each design needs, plus the logic and SRAM
added by GS, VS, CT and the per-bank refresh scheduler. Blocks shared by every design (exact QK/softmax/AV
backend, CDC/PHY, clocking, placement and routing) and the DRAM itself are excluded, so this is not a
total chip area.
"""
from __future__ import annotations

import csv
import json
from functools import lru_cache

from .common import ROOT
from .run_simulation import POP_HOST, REFRESH, VARIANTS, headroom

N = 32768


@lru_cache(maxsize=1)
def components():
    data = json.loads((ROOT / 'data/area/extension_components.json').read_text())
    with (ROOT / 'data/calibration/area_breakdown.csv').open() as handle:
        data['dense_support'] = {r['design']: r for r in csv.DictReader(handle) if r['variant'] == 'itrs-hp'}
    with (ROOT / 'data/calibration/pop_area_components.csv').open() as handle:
        data['pop_support'] = {f"POP_E{r['engines']}P{r['ports']}": r for r in csv.DictReader(handle)
                               if r['variant'] == 'itrs-hp' and (r['engines'], r['ports']) in (('4', '4'), ('24', '32'))}
    return data


def sram(size_bytes):
    return components()['sram_cacti_itrs_hp'][str(size_bytes)]['area_mm2']


def _pow2(x):
    n = 1
    while n < x:
        n *= 2
    return n


def base_area(design):
    """Scorer cells x 64 banks plus each design's supporting components (no refresh scheduler)."""
    c = components()
    if design.startswith('D'):
        ref = c['dense_support'][design]
        cells = 64 * c['dense_scorer_16B_port'][design]['cell_area_um2'] / 1e6
        support = sum(float(ref[k]) for k in ('input_query_SRAM_mm2', 'scheduler_proxy_mm2', 'router_queue_proxy_mm2'))
        return dict(scorer_cells_64banks_mm2=cells, supporting_mm2=support)
    cells = 64 * c['pop_score_bank']['cell_area_um2'] / 1e6
    if design in POP_HOST:
        # S path: response/tag queue proxy and the output staging RTL; no local selector SRAM.
        ref = c['pop_support']['POP_E4P4']
        support = float(ref['response_queue_proxy_mm2']) + c['s_path_staging']['cell_area_um2'] / 1e6
    else:
        ref = c['pop_support'][design]
        support = sum(float(ref[k]) for k in ('response_queue_proxy_mm2', 'score_sram_mm2', 'candidate_sram_mm2'))
    return dict(scorer_cells_64banks_mm2=cells, supporting_mm2=support)


def refresh_scheduler(design, refresh):
    """Per-bank refresh decision logic; the synthesized variant follows the design's drain headroom."""
    policy = REFRESH[refresh]
    if policy[1] != 'pim_pb':
        return 0.0
    variants = components()['refresh_scheduler']['variants']
    name = 'relaxed_H2' if policy[3] else ('round_H6' if headroom(design) >= 6 else 'round_H2')
    return variants[name]['cell_area_um2'] / 1e6


def variant_area(design, variant):
    """Area added (or removed) by a POP selection variant; empty for Dense."""
    if not design.startswith('POP_'):
        return {}
    c = components()
    source, budget, mask, compact = VARIANTS[variant]
    shared = source != 'base'
    out = {}
    gs_unit = 64 * c['gs_unit']['cell_area_um2'] / 1e6
    if design in POP_HOST:
        # Selection state returned by the host and kept near memory: a 4-bit head mask per (KV head, token)
        # for per-head selection, one bit per (KV head, token) for GS.
        out['membership_store_mm2'] = sram(8 * N * (1 if shared else 4) // 8)
        if compact:
            out['gs_unit_64banks_mm2'] = gs_unit
            out['staging_delta_mm2'] = (c['staging_w16']['cell_area_um2'] - c['staging_w64']['cell_area_um2']) / 1e6
    elif shared:
        if design != 'POP_E24P32':
            raise ValueError('the GS selector-SRAM model covers the 32-port E24/P32 selector only')
        ref = c['pop_support'][design]
        # Score SRAM holds 8 shared rows instead of 32; candidate SRAM holds 8 x K entries; same 32 ports.
        new = 32 * sram(_pow2(8 * N * 1.5 / 32)) + 32 * sram(_pow2(8 * budget * 3 / 32))
        out['selector_sram_delta_mm2'] = new - float(ref['score_sram_mm2']) - float(ref['candidate_sram_mm2'])
        out['gs_unit_64banks_mm2'] = gs_unit
    if mask:
        # One VS engine per KV head: logic, a 4 KiB histogram and a bin-code store
        # (GS: K ranks x 4 heads x 1 B; per-head lists: 4 heads x K x {token, bin} 4 B).
        out['vs_engines_logic_mm2'] = 8 * c['vs_engine']['cell_area_um2'] / 1e6
        out['vs_histogram_sram_mm2'] = 8 * sram(4096)
        out['vs_bincode_sram_mm2'] = 8 * sram(_pow2(budget * 4) if shared else _pow2(4 * budget * 4))
    return out


def area(design, variant='base', refresh='per-bank'):
    """Itemized known-partial area; ``known_partial_mm2`` is the sum."""
    parts = base_area(design)
    parts['refresh_scheduler_mm2'] = refresh_scheduler(design, refresh)
    parts.update(variant_area(design, variant))
    parts['known_partial_mm2'] = sum(parts.values())
    return parts


def report(output_dir):
    """Write the itemized area of every configuration used by the suites, and the synthesized block table."""
    import csv as _csv
    from pathlib import Path
    from .suites import SUITES
    seen = {}
    for suite in ('main', 'ablation', 'refresh'):
        for case in SUITES[suite]():
            key = (case.design, case.variant, case.refresh)
            if key not in seen:
                seen[key] = dict(design=case.design, variant=case.variant if case.pop else '', refresh=case.refresh,
                                 **area(*key))
    rows = list(seen.values())
    keys = []
    for row in rows:
        keys += [k for k in row if k not in keys]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'area_breakdown.csv').open('w', newline='') as handle:
        writer = _csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows({k: row.get(k, '') for k in keys} for row in rows)
    c = components()
    blocks = [('Dense D8 scorer, 16-byte ports', c['dense_scorer_16B_port']['D8'], 64, 'rtl/dense/dense_score_port16.sv'),
              ('POP score bank (A2 scorer, query mux, GQA4 frontend)', c['pop_score_bank'], 64, c['pop_score_bank']['rtl']),
              ('GS unit', c['gs_unit'], 64, c['gs_unit']['rtl']), ('VS engine (logic)', c['vs_engine'], 8, c['vs_engine']['rtl']),
              ('S-path staging, 64-bit packets', c['s_path_staging'], 1, c['s_path_staging']['rtl']),
              ('S-path staging, 16-bit packets (CT)', c['staging_w16'], 1, c['staging_w16']['rtl']),
              ('Refresh scheduler (round rule, H=2)', c['refresh_scheduler']['variants']['round_H2'], 1, c['refresh_scheduler']['rtl']),
              ('POP score bank with GS unit (integration check)', c['bank_gs'], 64, c['bank_gs']['rtl'])]
    lines = ['# Known partial area', '',
             'Per HBM3 channel. Mapped Nangate45 cells (Yosys 0.9 + ABC, pre-layout OpenSTA) plus CACTI 45 nm itrs-hp SRAM '
             'estimates and retained support proxies; blocks common to every design are excluded. `area_breakdown.csv` '
             'itemizes every configuration of the main, ablation and refresh suites.', '',
             '## Selected configurations (per-bank refresh)', '',
             '| Configuration | Scorer cells ×64 | Support | Refresh scheduler | GS/VS/CT and selection state | Total (mm²) |',
             '|---|---:|---:|---:|---:|---:|']
    for label, design, variant in (('Dense D4', 'D4', 'base'), ('Dense D8', 'D8', 'base'), ('Dense D16', 'D16', 'base'),
                                   ('Dense D32', 'D32', 'base'), ('POP-PIM (S path)', 'POP_S_CPU', 'base'),
                                   ('POP-PIM-Q (S path)', 'POP_S_CPU', 'gs6144_vs002h_ct'),
                                   ('POP-PIM-S (S path)', 'POP_S_CPU', 'gs4096_vs005h_ct'),
                                   ('POP-PIM E24/P32 (L path)', 'POP_E24P32', 'base'),
                                   ('GS on E24/P32 (L path)', 'POP_E24P32', 'gs4096')):
        a = area(design, variant)
        extra = sum(v for k, v in a.items() if k not in ('known_partial_mm2', 'scorer_cells_64banks_mm2',
                                                         'supporting_mm2', 'refresh_scheduler_mm2'))
        lines.append(f"| {label} | {a['scorer_cells_64banks_mm2']:.3f} | {a['supporting_mm2']:.3f} | "
                     f"{a['refresh_scheduler_mm2']:.4f} | {f'{extra:+.3f}' if extra else '—'} | {a['known_partial_mm2']:.3f} |")
    lines += ['', '## Synthesized blocks', '',
              '| Block | Cell area (µm²) | Flip-flops | Instances per channel | 125 MHz slack (ns) | RTL |',
              '|---|---:|---:|---:|---:|---|']
    for label, b, n, rtl in blocks:
        slack = b.get('slack_125MHz_ns')
        lines.append(f"| {label} | {b['cell_area_um2']:,.0f} | {b['flip_flops']:,} | {n} | "
                     f"{'' if slack is None else f'{slack:.3f}'} | `{rtl}` |")
    lines += ['', 'The S-path staging is timed at 250 MHz (output lanes). The area model adds the standalone GS unit '
              '(64 × 11,726 µm²) although the integrated bank grew by only 1,818 µm², which keeps the GS cost conservative.', '']
    (out / 'README.md').write_text('\n'.join(lines))
    return rows


def main(argv=None):
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    rows = report(args.output_dir)
    print(f'Wrote {len(rows)} configurations to {args.output_dir}')


if __name__ == '__main__':
    main()
