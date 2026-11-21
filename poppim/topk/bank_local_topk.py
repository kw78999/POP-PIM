"""B1 fixed and B2 survey-adaptive quotas, both WITHOUT global merge.

B2 sends pilot scores, broadcasts threshold, exchanges counts and quotas.
It does not claim exactness. Survey control traffic/synchronizations are exposed.
"""
import numpy as np
from poppim.topk.host_topk import validate, host_topk


def allocate_quotas(budget, capacities, weights=None):
    cap = np.asarray(capacities, dtype=np.int64)
    if np.any(cap < 0) or budget < 0 or budget > cap.sum():
        raise ValueError('invalid quota budget/capacity')
    weight = cap.astype(float) if weights is None else np.asarray(weights, dtype=float)
    if weight.shape != cap.shape or not np.isfinite(weight).all() or np.any(weight < 0):
        raise ValueError('invalid quota weights')
    quota = np.zeros_like(cap)
    # Capped largest remainder apportionment, stable bank-ID tie break.
    remaining = int(budget)
    while remaining:
        active = quota < cap
        w = np.where(active, weight, 0.)
        if not w.sum():
            w = np.where(active, cap-quota, 0.).astype(float)
        shares = remaining*w/w.sum()
        add = np.minimum(np.floor(shares).astype(np.int64), cap-quota)
        if not add.sum():
            order = np.lexsort((np.arange(len(cap)), -(shares-np.floor(shares))))
            for b in order:
                if quota[b] < cap[b] and remaining:
                    quota[b] += 1
                    remaining -= 1
        else:
            quota += add
            remaining -= int(add.sum())
    return quota


def bank_local_topk(scores, budget, bank_ids, num_banks, adaptive=False,
                    valid=None, pilot_per_bank=8):
    s, v = validate(scores, budget, valid)
    banks = np.asarray(bank_ids)
    if (num_banks < 1 or banks.shape != s.shape or banks.dtype.kind not in 'iu'
            or np.any(banks < 0) or np.any(banks >= num_banks) or pilot_per_bank < 1):
        raise ValueError('invalid bank IDs or pilot size')
    ids_by_bank = [np.flatnonzero(v & (banks == b)) for b in range(num_banks)]
    capacities = np.array([len(ids) for ids in ids_by_bank])
    control_bytes = 0
    threshold = None
    pilot_count = 0
    weights = None
    if adaptive and budget and v.sum():
        pilots = []
        for ids in ids_by_bank:
            if len(ids):
                offsets = np.linspace(0, len(ids)-1, min(pilot_per_bank, len(ids)), dtype=int)
                pilots.extend(s[ids[offsets]].tolist())
        pilot_count = len(pilots)
        rank = max(1, int(np.ceil(len(pilots)*budget/v.sum())))
        threshold = float(np.sort(pilots)[-rank])
        weights = np.array([np.count_nonzero(s[ids] >= threshold) for ids in ids_by_bank])
        # Per-bank unicast threshold, count, quota; pilot scores need no IDs.
        control_bytes = pilot_count*s.itemsize + num_banks*(s.itemsize+4+4)
    quota = allocate_quotas(budget, capacities, weights)
    selected = []
    for ids, amount in zip(ids_by_bank, quota):
        local_order = host_topk(s[ids], int(amount))
        selected.extend(ids[local_order].tolist())
    # Token-order transport only; no global score comparison or global merge.
    return np.array(sorted(selected), dtype=np.int64), {
        'quotas': quota, 'control_bytes': int(control_bytes),
        'threshold': threshold, 'pilot_count': pilot_count,
        'synchronization_rounds': 2 if control_bytes else 0,
    }
