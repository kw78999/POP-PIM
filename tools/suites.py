"""Named experiment suites shared by tools.reproduce and tools.rebuild_tables."""
from __future__ import annotations

from .run_simulation import BANDWIDTHS, REFRESH, VARIANTS, Case, grid

ALL_BW = sorted(BANDWIDTHS, reverse=True)          # 41.6, 20.8, 10.4, 5.2 GB/s
MAIN_POP = ('POP-PIM', 'POP-PIM-Q', 'POP-PIM-S')
NON_CT = [v for v, spec in VARIANTS.items() if not spec[3]]
LINK_VARIANTS = ('base', 'gs6144_vs002', 'gs6144_vs002h', 'gs6144_vs002_ct', 'gs6144_vs002h_ct',
                 'gs4096_vs005_ct', 'gs4096_vs005h_ct')


def quick():
    """Three cases at the nominal 41.6 GB/s channel (about one minute on four cores)."""
    return grid(['D8']) + grid(['POP_S_CPU'], ['POP-PIM', 'POP-PIM-S'])


def main():
    """Dense D4-D32 vs POP-PIM, POP-PIM-Q, POP-PIM-S (measured CPU host Top-k) at four channel bandwidths."""
    return grid(['D4', 'D8', 'D16', 'D32'], bandwidths=ALL_BW) + grid(['POP_S_CPU'], MAIN_POP, ALL_BW)


def ablation():
    """GS/VS/CT ablation on the S path (GPU Top-k assumption) and the local-selector L path, host link sweep,
    and the ideal-VS CPU variants."""
    return (grid(['POP_S_GPU'], list(VARIANTS), ALL_BW)
            + grid(['POP_E24P32'], NON_CT, ALL_BW)
            + grid(['POP_S_GPU'], LINK_VARIANTS, links=(8.0, 2.0, 0.5))
            + grid(['POP_S_CPU'], ['gs6144_vs002_ct', 'gs4096_vs005_ct'], ALL_BW))


def refresh():
    """Refresh policies (off, all-bank, all-bank work-conserving, per-bank, per-bank relaxed) at 41.6 GB/s."""
    designs = ['D1', 'D2', 'D4', 'D8', 'D16', 'D32', 'POP_E4P4', 'POP_E24P32', 'POP_S_GPU']
    return [case for policy in REFRESH for case in grid(designs, refresh=policy)]


SUITES = dict(quick=quick, main=main, ablation=ablation, refresh=refresh)
# Committed reference records for each suite (quick is a subset of main).
REFERENCE = dict(quick='main', main='main', ablation='ablation', refresh='refresh')


def cases(name) -> list[Case]:
    return SUITES[name]()
