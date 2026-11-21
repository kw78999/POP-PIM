"""Uncalibrated analytical cycles/traffic, NEVER measured PIM performance.

All bandwidths are effective aggregate GB/s (decimal). No technology area or
energy coefficients are invented. Index wire format is uint32, Python int64.
"""
from dataclasses import dataclass, asdict
import math
import numpy as np
from poppim.memory.bank_mapper import map_banks


@dataclass(frozen=True)
class HardwareConfig:
    pim_mhz: float = 500
    popcount_lanes: int = 8
    bitwise_lanes: int = 8
    fp_units: int = 4
    score_pipeline_cycles: int = 8
    bank_read_GBs: float = 8
    host_link_GBs: float = 32
    internal_link_GBs: float = 256
    local_selector_mhz: float = 500
    near_selector_mhz: float = 500
    near_compare_lanes: int = 32
    host_gpu_mhz: float = 1000
    host_gpu_compare_lanes: int = 256
    sync_ns: float = 500
    pilot_per_bank: int = 8
    index_bytes: int = 4
    kv_element_bytes: int = 2

    def __post_init__(self):
        if any(v <= 0 for v in asdict(self).values()):
            raise ValueError('all hardware assumptions must be positive')


def heap_comparison_bound(n, k):
    # Streaming min-heap, includes initial insert and final heap output sorting.
    if not k:
        return 0
    levels = math.ceil(math.log2(max(k, 2)))
    return int(n*(1+2*levels) + 2*k*levels)


def estimate(context, budget, banks, q_heads=4, kv_heads=2, dim=128, group=64,
             mapping='striped', mode='host', quotas=None, control_bytes=0,
             synchronization_rounds=0, hw=None):
    hw = hw or HardwareConfig()
    if (context < 1 or not 0 <= budget <= context or banks < 1 or
            q_heads < 1 or kv_heads < 1 or q_heads % kv_heads or
            dim < 1 or group not in (16,32,64) or context >= 2**32):
        raise ValueError('invalid architecture dimensions')
    if mode not in ('host', 'bank_local', 'bank_adaptive', 'near_memory'):
        raise ValueError('invalid architecture mode')
    loads = np.bincount(map_banks(context, banks, mapping), minlength=banks)
    words, reps = math.ceil(dim/32), math.ceil(dim/group)
    segments = math.ceil(dim/group) if group == 16 else words
    metadata_per_token = 8*words + 4*reps
    internal_bytes = context*kv_heads*metadata_per_token
    # Assume bank-local metadata fetched once and reused across GQA queries.
    read_ns = max(loads)*kv_heads*metadata_per_token / hw.bank_read_GBs
    per_score_cycles = (math.ceil(5*segments/hw.bitwise_lanes) +
                        math.ceil(7*segments/hw.popcount_lanes) +
                        math.ceil((9*reps+1)/hw.fp_units))
    score_cycles = max(loads)*q_heads*per_score_cycles + hw.score_pipeline_cycles
    compute_ns = score_cycles*1000/hw.pim_mhz
    query_bytes = banks*q_heads*(8*words+8*reps)
    query_ns = query_bytes/hw.internal_link_GBs
    score_ns = read_ns + compute_ns
    score_element_bytes = 2 if dim == 128 else 4
    score_bytes = context*q_heads*score_element_bytes
    index_bytes = budget*q_heads*hw.index_bytes
    # Serialized conservative model: query -> score -> selection/return -> indices.
    # C streams transfer and selector concurrently; no N-sized score buffer.
    bank_to_near = 0
    memory_host = 0
    select_ns = 0.
    transfer_ns = 0.
    candidate_bytes = 0
    comparisons = 0
    if mode == 'host':
        memory_host = score_bytes
        transfer_ns = score_bytes/hw.host_link_GBs
        comparisons = q_heads*heap_comparison_bound(context, budget)
        select_ns = comparisons/hw.host_gpu_compare_lanes*1000/hw.host_gpu_mhz
        candidate_bytes = budget*q_heads*(score_element_bytes+hw.index_bytes)
        output_ns = 0.  # indices are already at host; no second external transfer
    elif mode == 'near_memory':
        bank_to_near = score_bytes
        memory_host = index_bytes
        comparisons = q_heads*heap_comparison_bound(context, budget)
        select_ns = comparisons/hw.near_compare_lanes*1000/hw.near_selector_mhz
        transfer_ns = score_bytes/hw.internal_link_GBs
        candidate_bytes = budget*q_heads*(score_element_bytes+hw.index_bytes)
        output_ns = index_bytes/hw.host_link_GBs
    else:
        if quotas is None:
            raise ValueError('local model requires actual per-head quotas')
        quota = np.asarray(quotas)
        if (quota.shape != (q_heads,banks) or np.any(quota < 0) or
                np.any(quota > loads) or np.any(quota.sum(-1) != budget)):
            raise ValueError('invalid quotas')
        per_bank = [sum(heap_comparison_bound(int(loads[b]), int(quota[h,b]))
                        for h in range(q_heads)) for b in range(banks)]
        comparisons = int(sum(per_bank))
        select_ns = max(per_bank)*1000/hw.local_selector_mhz
        candidate_bytes = int(quota.sum())*(score_element_bytes+hw.index_bytes)
        memory_host = index_bytes
        # B2 score survey + threshold/count/quota controller traffic stays internal.
        bank_to_near = int(control_bytes)
        transfer_ns = control_bytes/hw.internal_link_GBs
        if mode == 'bank_adaptive':
            # Threshold count pass rereads stored FP16 scores in every bank.
            read_ns += score_bytes/(banks*hw.bank_read_GBs)
            score_ns = read_ns + compute_ns
            select_ns += max(loads)*q_heads*1000/hw.local_selector_mhz
            # Coordinator sorts bounded pilot set; comparison upper bound.
            pilot_n = int(sum(min(int(n), hw.pilot_per_bank) for n in loads))
            pilot_comparisons = q_heads*pilot_n*math.ceil(math.log2(max(pilot_n,2)))
            select_ns += pilot_comparisons/hw.near_compare_lanes*1000/hw.near_selector_mhz
        output_ns = index_bytes/hw.host_link_GBs
    communication_ns = query_ns+transfer_ns+output_ns
    if mode == 'near_memory':
        service_ns = max(select_ns, transfer_ns)+output_ns
    else:
        service_ns = select_ns+transfer_ns+output_ns
    sync_ns = hw.sync_ns*(1+synchronization_rounds)
    total_ns = query_ns+score_ns+service_ns+sync_ns
    # B2 needs score SRAM for the survey/reallocation second pass, explicitly paid.
    staging = score_bytes if mode == 'bank_adaptive' else 0
    pilot_buffer = (q_heads*banks*hw.pilot_per_bank*score_element_bytes if mode == 'bank_adaptive' else 0)
    local_heap_peak = (int(np.max(np.asarray(quotas).sum(0)))*(score_element_bytes+hw.index_bytes)
                       if quotas is not None else 0)
    return {
        'source': 'Analytical Model (uncalibrated assumptions)', 'mode': mode,
        'pim_score_ns': float(score_ns), 'pim_compute_ns': float(compute_ns),
        'dram_read_ns': float(read_ns), 'selection_ns': float(select_ns),
        'communication_ns': float(communication_ns), 'sync_ns': float(sync_ns),
        'query_broadcast_ns': float(query_ns),
        'score_delivery_or_control_ns': float(transfer_ns),
        'output_index_transfer_ns': float(output_ns),
        'selection_and_transfer_service_ns': float(service_ns),
        'total_retrieval_ns': float(total_ns),
        'estimated_retrievals_per_second': float(1e9/total_ns),
        'bank_internal_read_bytes': int(internal_bytes+(staging if mode == 'bank_adaptive' else 0)),
        'bank_internal_write_bytes': int(staging),
        'host_score_buffer_bytes': int(score_bytes if mode == 'host' else 0),
        'bank_to_near_bytes': int(bank_to_near), 'memory_to_host_bytes': int(memory_host),
        'selected_index_bytes': int(index_bytes),
        'selected_kv_read_bytes_logical': int(budget*q_heads*dim*2*hw.kv_element_bytes),
        'query_broadcast_bytes': int(query_bytes),
        'candidate_buffer_logical_bytes': int(candidate_bytes),
        'score_staging_bytes': int(staging), 'pilot_buffer_bytes': int(pilot_buffer),
        'local_candidate_peak_bank_bytes': int(local_heap_peak),
        'heap_comparison_upper_bound': int(comparisons),
        'load_imbalance_max_over_mean': float(max(loads)/(context/banks)),
        'bank_utilization_load_only': float(context/(banks*max(loads))),
        'popcount_operators': banks*hw.popcount_lanes,
        'fp_issue_units': banks*hw.fp_units,
        'selector_compare_lanes': (banks if mode.startswith('bank_') else
                                  hw.near_compare_lanes if mode == 'near_memory' else 0),
        'score_unit_area_um2': None, 'selector_area_um2': None,
        'popcount_area_um2': None, 'fp_weighted_score_area_um2': None,
        'local_topk_area_um2': None, 'global_topk_area_um2': None,
        'buffer_area_um2': None, 'controller_area_um2': None, 'total_area_um2': None,
        'dram_energy_pj': None, 'pim_energy_pj': None, 'selection_energy_pj': None,
        'communication_energy_pj': None, 'total_energy_pj': None,
        'area_energy_source': 'Unavailable: no synthesis/library/activity calibration',
    }
