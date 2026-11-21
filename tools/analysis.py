"""Derived metrics for simulation records: reported latency, byte ledger, area and area x delay."""
from __future__ import annotations

import json
from functools import lru_cache

from poppim.variants import vs_scan_cycles
from .area_model import area
from .common import ROOT
from .run_simulation import LABELS, VARIANTS

MiB = 2 ** 20
VS_CLOCK_MHZ = 125.0     # the VS engines and score banks are timed at 125 MHz (pre-layout STA)
DESIGN_LABELS = {'D1': 'Dense D1', 'D2': 'Dense D2', 'D4': 'Dense D4', 'D8': 'Dense D8', 'D16': 'Dense D16',
                 'D32': 'Dense D32', 'POP_E4P4': 'POP-PIM L path E4/P4', 'POP_E24P32': 'POP-PIM L path E24/P32',
                 'POP_S_GPU': 'POP-PIM S path, GPU Top-k (assumed)', 'POP_S_CPU': 'POP-PIM S path, CPU Top-k (measured)'}


@lru_cache(maxsize=1)
def _masks():
    return json.loads((ROOT / 'data/replay/variants/manifest.json').read_text())['vkeep']


def scan_cycles(variant):
    """VS threshold-scan cycles added serially before the V phase (hardware VS rule only).

    The exact-rule ("ideal VS") variants are an idealization without a scan cost.
    """
    mask = VARIANTS[variant][2] if variant else None
    if not mask or not mask.endswith('_hw'):
        return 0
    return vs_scan_cycles(_masks()[mask]['bstar'])


def display_name(record):
    if not record['design'].startswith('POP_'):
        return DESIGN_LABELS[record['design']]
    label = LABELS.get(record['variant'])
    if record['design'] == 'POP_S_CPU' and label:
        return label
    return f"{DESIGN_LABELS[record['design']]}: {label or record['variant']}"


def derive(record):
    """Flat row of the quantities used in the reports."""
    m = record['metrics']
    pop = record['design'].startswith('POP_')
    variant = record.get('variant') or 'base'
    sim_ms = m['attention_output_ps'] / 1e9
    cycles = scan_cycles(variant) if pop else 0
    latency = sim_ms + cycles / VS_CLOCK_MHZ / 1e3
    two_phase = bool(m.get('phase21_two_phase'))
    sel_k = (m.get('phase21_selected_k_reads', 0) * 32 if two_phase else m.get('selected_kv_read_requests', 0) * 16) / MiB
    sel_v = (m.get('phase21_selected_v_reads', 0) * 32 if two_phase else m.get('selected_kv_read_requests', 0) * 16) / MiB
    export = m.get('score_wire_bytes', 0) / MiB
    index = m.get('phase21_index_return_bytes', 0) / MiB if m.get('offload') else 0.0
    parts = area(record['design'], variant, record['refresh'])
    row = dict(name=record['name'], design=record['design'], configuration=display_name(record),
               variant=variant if pop else '', bandwidth_GBs=record['bandwidth_GBs'], refresh=record['refresh'],
               context=record['context'], host_link_GBs=record.get('host_link_GBs') or '',
               host_topk_ms=(record['host_topk_ps'] / 1e9) if record.get('host_topk_ps') else '',
               latency_ms=latency, simulated_latency_ms=sim_ms, vs_scan_cycles=cycles,
               score_ready_ms=m['last_score_ps'] / 1e9, topk_ready_ms=m.get('topk_ready_ps', 0) / 1e9 if pop else '',
               channel_MiB=m['read_bytes'] / MiB + export + index, dram_read_MiB=m['read_bytes'] / MiB,
               metadata_MiB=m.get('metadata_read_requests', 0) * 32 / MiB if pop else 0.0,
               selected_K_MiB=sel_k if pop else '', selected_V_MiB=sel_v if pop else '',
               host_link_MiB=export + index, union_tokens=m.get('selected_union_tokens', '') if pop else '',
               kept_v_tokens=m.get('phase21_kept_v_tokens', '') if two_phase else '',
               ACT=m['ACT'], RD=m['RD'], REFab=m['REF'], REFsb=m.get('REFsb', 0),
               known_partial_area_mm2=parts['known_partial_mm2'])
    row['area_delay_mm2_ms'] = row['known_partial_area_mm2'] * latency
    row.update({f'area_{k}': v for k, v in parts.items() if k != 'known_partial_mm2'})
    return row
