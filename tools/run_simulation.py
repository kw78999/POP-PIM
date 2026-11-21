"""Run Dense and POP-PIM architectural simulations: one decode token, one attention layer.

Each case is a fresh Ramulator HBM3 simulation with the adapters built by ``tools.build``. Cases are the
cross product of ``--designs``, ``--variants`` (POP only), ``--bandwidths`` and ``--host-links`` (S path
only). Every case writes ``config.yaml``, ``metrics.json``, ``record.json`` and ``run.log`` into its own
folder; ``runs.json`` in the output directory collects all records.

Examples::

    python -m tools.run_simulation --designs D8 POP_S_CPU --variants POP-PIM-S \\
        --bandwidths 41.6 5.2 --output-dir outputs/demo
    python -m tools.run_simulation --designs D1 D2 D4 D8 POP_E4P4 POP_E24P32 \\
        --refresh all-bank --output-dir outputs/all_bank_refresh
"""
from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from .common import BUILD, ROOT, run, save, sha
from . import prepare_inputs

N_DEFAULT = 32768
DENSE = ('D1', 'D2', 'D4', 'D8', 'D16', 'D32')
POP_LOCAL = {'POP_E4P4': (4, 4), 'POP_E24P32': (24, 32)}   # local selector: engines, SRAM ports
POP_HOST = ('POP_S_GPU', 'POP_S_CPU')                        # S path: Top-k on the host
DESIGNS = DENSE + tuple(POP_LOCAL) + POP_HOST

# name: (replay input, budget K, V-keep mask key or None, compact transport)
VARIANTS = {
    'base':             ('base', 4096, None, False),
    'base_vs002':       ('base', 4096, 'base_0.002_exact', False),
    'base_vs002h':      ('base', 4096, 'base_0.002_hw', False),
    'gs4096':           ('gs4096', 4096, None, False),
    'gs6144':           ('gs6144', 6144, None, False),
    'gs4096_vs005':     ('gs4096', 4096, 'gs4096_0.005_exact', False),
    'gs6144_vs002':     ('gs6144', 6144, 'gs6144_0.002_exact', False),
    'gs4096_vs005h':    ('gs4096', 4096, 'gs4096_0.005_hw', False),
    'gs6144_vs002h':    ('gs6144', 6144, 'gs6144_0.002_hw', False),
    'gs4096_ct':        ('gs4096', 4096, None, True),
    'gs6144_vs002_ct':  ('gs6144', 6144, 'gs6144_0.002_exact', True),
    'gs4096_vs005_ct':  ('gs4096', 4096, 'gs4096_0.005_exact', True),
    'gs6144_vs002h_ct': ('gs6144', 6144, 'gs6144_0.002_hw', True),
    'gs4096_vs005h_ct': ('gs4096', 4096, 'gs4096_0.005_hw', True),
}
NAMED = {'POP-PIM': 'base', 'POP-PIM-S': 'gs4096_vs005h_ct', 'POP-PIM-Q': 'gs6144_vs002h_ct'}
LABELS = {v: k for k, v in NAMED.items()}

# CLI name: (RefreshManager, controller policy, round rule, relaxed drain)
REFRESH = {
    'per-bank':         ('No', 'pim_pb', True, False),
    'per-bank-relaxed': ('No', 'pim_pb', False, True),
    'all-bank':         ('AllBankHBM3', 'strict', None, None),
    'all-bank-wc':      ('AllBankHBM3', 'wc_ab', None, None),
    'off':              ('No', 'strict', None, None),
}
DEFAULT_REFRESH = 'per-bank'

# Channel bandwidth -> burst length override of the retained HBM3 5.2 Gbps preset (2 pseudochannels x 32 B).
BANDWIDTHS = {41.6: 2, 20.8: 4, 10.4: 8, 5.2: 16}

HBM_PS = 769                 # HBM3 controller clock period
NREFI = 5070                 # refresh interval in controller cycles (checked by refresh_single_bank)
POSTPONE = PULLIN = 8        # per-bank refresh debt window, in tREFI
POP_LATENCY = 43             # calibrated A2 scorer pipeline latency (scorer cycles at 125 MHz)
HOST_LINK = dict(startup_ps=5_000_000, sync_ps=2_000_000, output_lanes=4, output_period_ps=4000,
                 lane_depth=16, dma_depth=16)


def resolve_variant(name):
    name = NAMED.get(name, name)
    if name not in VARIANTS:
        raise ValueError(f'unknown variant {name!r}; choose from {sorted(VARIANTS) + sorted(NAMED)}')
    return name


def host_topk_ps(design, variant, override_ms=None):
    """External Top-k service time for the S path (GPU assumption or measured CPU median)."""
    if override_ms is not None:
        return round(override_ms * 1e9)
    data = json.loads((ROOT / 'data/calibration/host_topk.json').read_text())
    if design == 'POP_S_GPU':
        return round(data['gpu_assumed_ms'] * 1e9)
    source, budget, _, _ = VARIANTS[variant]
    rows = 32 if source == 'base' else 8
    for item in data['cpu_measured']:
        if (item['rows'], item['K']) == (rows, budget):
            return round(item['median_ms'] * 1e9)
    raise ValueError(f'no measured CPU Top-k for {rows} rows x K={budget}; pass --host-topk-ms')


def headroom(design):
    """Refresh drain headroom: longest uninterruptible scorer window, in tREFI, plus one period."""
    if design.startswith('POP_'):
        window = math.ceil(POP_LATENCY * 8000 / HBM_PS)
    else:
        timing = json.loads((ROOT / 'data/calibration/tile_schedules.json').read_text())[design]
        window = math.ceil((timing['ready'][-1] + 1) * 8000 / HBM_PS)
    return math.ceil(window / NREFI) + 1


@dataclass(frozen=True)
class Case:
    design: str
    variant: str = 'base'
    bandwidth: float = 41.6
    refresh: str = DEFAULT_REFRESH
    context: int = N_DEFAULT
    link_GBs: float = 32.0
    topk_ms: float | None = None
    mode: str = 'full'

    @property
    def pop(self):
        return self.design.startswith('POP_')

    @property
    def host(self):
        return self.design in POP_HOST

    def validate(self):
        if self.design not in DESIGNS:
            raise ValueError(f'unknown design {self.design}')
        if self.bandwidth not in BANDWIDTHS:
            raise ValueError(f'bandwidth must be one of {sorted(BANDWIDTHS)} GB/s')
        if self.refresh not in REFRESH:
            raise ValueError(f'refresh must be one of {sorted(REFRESH)}')
        if self.mode not in ('full', 'score'):
            raise ValueError('mode must be full or score')
        if self.pop:
            if self.context != N_DEFAULT:
                raise ValueError('the included POP replay is fixed at N=32768')
            if VARIANTS[self.variant][3] and not self.host:
                raise ValueError(f'{self.variant}: compact transport applies to the S path (POP_S_GPU/POP_S_CPU) only')
        else:
            if self.variant != 'base':
                raise ValueError('variants apply to POP designs only')
            if self.context not in (32768, 65536, 131072):
                raise ValueError('Dense supports N = 32768, 65536 or 131072')
        if not self.host and (self.link_GBs != 32.0 or self.topk_ms is not None):
            raise ValueError('host link and host Top-k apply to the S path only')

    @property
    def name(self):
        parts = [self.design]
        if self.pop:
            parts.append(LABELS.get(self.variant, self.variant))
        parts.append(f'bw{self.bandwidth:g}')
        if self.host and self.link_GBs != 32.0:
            parts.append(f'link{self.link_GBs:g}')
        if self.topk_ms is not None:
            parts.append(f'topk{self.topk_ms:g}ms')
        if self.refresh != DEFAULT_REFRESH:
            parts.append(self.refresh)
        if self.context != N_DEFAULT:
            parts.append(f'n{self.context}')
        if self.mode != 'full':
            parts.append(self.mode)
        return '_'.join(parts)


def _yaml(params):
    return ''.join(f'  {k}: {json.dumps(v)}\n' for k, v in params.items())


def inputs_for(case, replay_dir=None):
    """Input files read by one case (replay scores/indices, V-keep mask, or Dense tile schedule)."""
    if not case.pop:
        return dict(tile_schedule=ROOT / f'data/calibration/schedules/{case.design}.txt')
    source, _, mask, _ = VARIANTS[case.variant]
    scores, expected = prepare_inputs.input_paths(source, replay_dir)
    files = dict(score_file=scores, expected_file=expected)
    if mask:
        files['vkeep_file'] = prepare_inputs.mask_path(mask, replay_dir)
    return files


def configuration(case, folder, replay_dir=None):
    """Ramulator YAML for one case. ``folder`` receives metrics and the command trace."""
    case.validate()
    manager, policy, round_rule, relaxed = REFRESH[case.refresh]
    files = inputs_for(case, replay_dir)
    source, budget, mask, compact = VARIANTS[case.variant]
    frontend = dict(impl='Phase13Common', output=str(folder), design='A2-S1' if case.pop else case.design,
                    mode=case.mode, replicas=1, coupled_release=False, context_length=case.context,
                    budget=budget if case.pop else 4096)
    if case.pop:
        engines, ports = POP_LOCAL.get(case.design, (4, 4))
        frontend.update(ii=22.0, latency=POP_LATENCY, selector_engines=engines, selector_ports=ports,
                        score_file=str(files['score_file']), expected_file=str(files['expected_file']))
    else:
        rows = json.loads((ROOT / 'data/calibration/port_rtl.json').read_text())
        rtl = next(row for row in rows if row['design'] == case.design and row['mode'] == 0)
        frontend.update(ii=rtl['score_II'], latency=round(rtl['latency_mean']),
                        tile_schedule=str(files['tile_schedule']), selector_engines=4, selector_ports=4)
    extra = {}
    if case.host:
        extra.update(offload=True, link_GBs=case.link_GBs, startup_ps=HOST_LINK['startup_ps'],
                     sync_ps=HOST_LINK['sync_ps'], external_topk_ps=host_topk_ps(case.design, case.variant, case.topk_ms),
                     output_lanes=HOST_LINK['output_lanes'], output_period_ps=HOST_LINK['output_period_ps'],
                     lane_depth=HOST_LINK['lane_depth'], dma_depth=HOST_LINK['dma_depth'])
    if mask:
        extra['vkeep_file'] = str(files['vkeep_file'])
    if case.host and compact:
        # One 16-bit shared score per (KV head, token), position-implicit; indices return as an N-bit bitmap.
        extra.update(shared_export=True, packet_wire_bytes=2, index_return_bytes=8 * case.context // 8)
    elif case.host and source != 'base':
        # Shared selection without compact transport: per-head packets, 8 index lists come back.
        extra.update(index_return_bytes=8 * budget * 4)
    text = 'Frontend:\n'
    if policy != 'strict':
        text += '  refresh_drain: true\n'
    text += _yaml(frontend) + _yaml(extra)
    memory = (ROOT / 'configs/hbm3.yaml').read_text()
    memory = memory.replace('@REFRESH@', manager).replace('@TRACE@', json.dumps(str(folder / 'command_trace.bin')))
    block = f'    refresh_policy: {policy}\n'
    if policy == 'pim_pb':
        h = headroom(case.design)
        low = min(0, POSTPONE - h - 1) if relaxed else POSTPONE - h - 1
        block += (f'    refresh_postpone: {POSTPONE}\n    refresh_pullin: {PULLIN}\n'
                  f'    refresh_headroom: {h}\n    refresh_round_rule: {str(round_rule).lower()}\n'
                  f'    refresh_drain_low: {low}\n')
    memory = memory.replace('    impl: PopPIMHBM3\n', '    impl: PopPIMHBM3\n' + block, 1)
    nbl = BANDWIDTHS[case.bandwidth]
    if nbl != 2:
        memory = memory.replace('      preset: HBM3_5.2Gbps\n', f'      preset: HBM3_5.2Gbps\n      nBL: {nbl}\n', 1)
    return text + memory


def binary(case):
    return BUILD / 'bin' / ('poppim_pop' if case.pop else 'poppim_dense')


def execute(case, out, keep_trace=True, audit_trace=False, replay_dir=None):
    folder = out / case.name
    folder.mkdir(parents=True, exist_ok=False)
    config = folder / 'config.yaml'
    config.write_text(configuration(case, folder, replay_dir))
    exe = binary(case)
    run([exe, '-f', config], folder / 'run.log', timeout=7200)
    metrics = json.loads((folder / 'metrics.json').read_text())
    trace = folder / 'command_trace.bin'
    record = dict(name=case.name, design=case.design, variant=case.variant if case.pop else None,
                  label=LABELS.get(case.variant) if case.pop else None, bandwidth_GBs=case.bandwidth,
                  nBL=BANDWIDTHS[case.bandwidth], refresh=case.refresh, context=case.context,
                  budget=VARIANTS[case.variant][1] if case.pop else None, mode=case.mode,
                  host_link_GBs=case.link_GBs if case.host else None,
                  host_topk_ps=host_topk_ps(case.design, case.variant, case.topk_ms) if case.host else None,
                  refresh_headroom=headroom(case.design) if REFRESH[case.refresh][1] == 'pim_pb' else None,
                  metrics=metrics, config_sha256=sha(config), binary_sha256=sha(exe),
                  library_sha256=sha(BUILD / 'ramulator/lib/libramulator.so'), metrics_sha256=sha(folder / 'metrics.json'),
                  trace_sha256=sha(trace) if trace.exists() else None,
                  input_sha256={(str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else p.name): sha(p)
                                for p in inputs_for(case, replay_dir).values()})
    if audit_trace:
        from .audit_simulation import audit
        record['audit'] = audit(record, folder)
    if not keep_trace and trace.exists():
        trace.unlink()
        record['trace_retained'] = False
    save(folder / 'record.json', record)
    print(f"{case.name:48s} simulated {metrics['attention_output_ps'] / 1e9:9.3f} ms   DRAM read "
          f"{metrics['read_bytes'] / 2**20:7.2f} MiB", flush=True)
    return record


def run_cases(cases, out, jobs=2, keep_traces=True, audit_traces=False):
    """Validate every case first, then run them; returns the list of records (also saved as runs.json).

    ``audit_traces`` audits each raw command trace right after its run (before an optional discard).
    """
    out = Path(out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'{out}: use a fresh, empty output directory')
    names = [c.name for c in cases]
    if len(set(names)) != len(names):
        raise ValueError('duplicate cases')
    for case in cases:
        configuration(case, out / case.name)
    for case in cases:
        if not binary(case).is_file():
            raise FileNotFoundError('Simulator binaries are missing; run python -m tools.build first')
    if any(c.pop for c in cases):
        prepare_inputs.prepare(quiet=True)
    out.mkdir(parents=True, exist_ok=True)
    # Longest (Dense, low bandwidth) cases first keeps the worker pool busy.
    order = sorted(cases, key=lambda c: (c.pop, -BANDWIDTHS[c.bandwidth], -c.context))
    with ThreadPoolExecutor(max(1, jobs)) as pool:
        done = {r['name']: r for r in pool.map(lambda c: execute(c, out, keep_traces, audit_traces), order)}
    records = [done[c.name] for c in cases]
    save(out / 'runs.json', records)
    print(f'Completed {len(records)} simulations in {out}', flush=True)
    return records


def grid(designs, variants=('base',), bandwidths=(41.6,), refresh=DEFAULT_REFRESH, context=N_DEFAULT,
         links=(32.0,), topk_ms=None, mode='full'):
    cases = []
    for design in designs:
        pop = design.startswith('POP_')
        for variant in (variants if pop else ('base',)):
            variant = resolve_variant(variant)
            if pop and VARIANTS[variant][3] and design not in POP_HOST:
                continue  # compact transport exists only on the S path
            for bandwidth in bandwidths:
                for link in (links if design in POP_HOST else (32.0,)):
                    cases.append(Case(design, variant, bandwidth, refresh, context, link,
                                      topk_ms if design in POP_HOST else None, mode))
    return cases


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--designs', nargs='+', choices=DESIGNS, default=['D8', 'POP_S_CPU'])
    parser.add_argument('--variants', nargs='+', default=['POP-PIM-S'],
                        help='POP selection variants (POP-PIM, POP-PIM-S, POP-PIM-Q or an internal id); ignored for Dense')
    parser.add_argument('--bandwidths', nargs='+', type=float, default=[41.6], choices=sorted(BANDWIDTHS),
                        help='channel bandwidth in GB/s')
    parser.add_argument('--refresh', choices=list(REFRESH), default=DEFAULT_REFRESH)
    parser.add_argument('--context', type=int, default=N_DEFAULT, help='Dense only: 32768, 65536 or 131072')
    parser.add_argument('--host-links', nargs='+', type=float, default=[32.0], help='S path host link GB/s')
    parser.add_argument('--host-topk-ms', type=float, help='override the S-path external Top-k time')
    parser.add_argument('--mode', choices=['full', 'score'], default='full', help='score: stop after scoring')
    parser.add_argument('--jobs', type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument('--discard-traces', action='store_true', help='delete command traces after each run')
    parser.add_argument('--audit', action='store_true', help='audit each raw trace right after its run')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true', help='print the resolved configurations only')
    args = parser.parse_args(argv)
    try:
        cases = grid(args.designs, args.variants, args.bandwidths, args.refresh, args.context,
                     args.host_links, args.host_topk_ms, args.mode)
        if args.dry_run:
            for case in cases:
                print(f'--- {case.name}\n' + configuration(case, args.output_dir.resolve() / case.name))
            return
        run_cases(cases, args.output_dir, args.jobs, not args.discard_traces, args.audit)
    except (ValueError, FileExistsError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
