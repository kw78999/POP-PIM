"""Exact FP16 radix selection. Bit keys preserve values, not quantization.

Scores are replayed from bounded, explicitly costed score storage. Four nibble
histogram passes, one gather pass, then deterministic candidate merge ranking.
NaN/Inf are rejected, including masked entries, matching the existing contract.
"""
import numpy as np
from poppim.topk.host_topk import validate


def ordered_keys(scores):
    s = np.asarray(scores)
    if s.dtype != np.float16 or not np.isfinite(s).all():
        raise ValueError('ordered keys require finite FP16 stored scores')
    bits = s.view(np.uint16)
    # Both zero signs compare equal; the token index breaks that tie.
    bits = np.where((bits & 0x7fff) == 0, np.uint16(0), bits).astype(np.uint16)
    return np.where(bits & 0x8000, ~bits, bits ^ np.uint16(0x8000)).astype(np.uint16)


def merge_rank(ids, keys):
    """Instrumented stable merge sort of candidates; actual comparison counts."""
    entries = [(int(key), int(idx)) for idx,key in zip(ids,keys)]
    comparisons = reads = writes = 0
    width = 1
    while width < len(entries):
        target = []
        for start in range(0,len(entries),2*width):
            left = entries[start:start+width]
            right = entries[start+width:start+2*width]
            i = j = 0
            while i < len(left) and j < len(right):
                comparisons += 1
                reads += 2
                a,b = left[i],right[j]
                if a[0] > b[0] or (a[0] == b[0] and a[1] < b[1]):
                    target.append(a); i += 1
                else:
                    target.append(b); j += 1
                writes += 1
            target.extend(left[i:]); target.extend(right[j:])
            reads += len(left)-i+len(right)-j
            writes += len(left)-i+len(right)-j
        entries = target
        width *= 2
    return np.array([i for _,i in entries],dtype=np.int64), {
        'comparisons':comparisons,'rank_sram_reads':reads,'rank_sram_writes':writes}


def radix_topk(scores,budget,valid=None,reserve_current=False,radix_bits=4,
               chunk_size=1024,num_banks=1,bank_ids=None,return_stats=False,output_mode="sorted"):
    if output_mode not in ("sorted", "unordered", "position"):
        raise ValueError("unknown radix output mode")
    s = np.asarray(scores)
    ordered_keys(s)  # dtype and non-finite policy, even if budget is zero
    if reserve_current:
        if not len(s) or budget < 1:
            raise ValueError('invalid current-token budget')
        mask = np.ones(len(s),dtype=bool) if valid is None else np.asarray(valid,dtype=bool).copy()
        if mask.shape != s.shape or not mask[-1]:
            raise ValueError('invalid current-token mask')
        mask[-1] = False
        selected,stats = radix_topk(s,budget-1,mask,False,radix_bits,chunk_size,
                                   num_banks,bank_ids,True,output_mode)
        selected = np.r_[selected,np.int64(len(s)-1)]
        return (selected,stats) if return_stats else selected
    s,v = validate(s,budget,valid)
    if radix_bits not in (1,2,4,8) or chunk_size < 1 or num_banks < 1:
        raise ValueError('invalid radix/chunk/bank configuration')
    banks = np.arange(len(s)) % num_banks if bank_ids is None else np.asarray(bank_ids)
    if banks.shape != s.shape or banks.dtype.kind not in 'iu' or np.any(banks<0) or np.any(banks>=num_banks):
        raise ValueError('invalid bank IDs')
    size = 1 << radix_bits
    stats = dict(algorithm='radix',score_reads=0,score_writes=len(s),histogram_updates=0,
        histogram_reads=0,histogram_writes=0,reduction_adds=0,synchronizations=0,
        prefix_tests=0,comparisons=0,passes=0,boundary_key=None,
        candidate_peak=0,score_staging_bytes=2*len(s),histogram_bytes=4*num_banks*size,
        candidate_buffer_bytes=(12 if output_mode=='sorted' else 4)*budget,rank_sram_reads=0,rank_sram_writes=0,
        radix_bits=radix_bits,bank_pass_updates=[],output_mode=output_mode,
        compaction_writes=0)
    if not budget:
        selected = np.empty(0,dtype=np.int64)
        return (selected,stats) if return_stats else selected
    prefix = 0
    prefix_mask = 0
    rank = int(budget)
    for shift in range(16-radix_bits,-1,-radix_bits):
        hist = np.zeros((num_banks,size),dtype=np.int64)
        for start in range(0,len(s),chunk_size):
            end = min(len(s),start+chunk_size)
            key = ordered_keys(s[start:end])
            active = v[start:end] & ((key & prefix_mask) == prefix)
            np.add.at(hist,(banks[start:end][active],((key[active]>>shift)&(size-1))),1)
        updates = int(hist.sum())
        stats['bank_pass_updates'].append(hist.sum(1).tolist())
        stats['histogram_updates'] += updates
        stats['histogram_reads'] += updates
        stats['histogram_writes'] += updates+num_banks*size
        stats['reduction_adds'] += max(0,num_banks-1)*size
        stats['prefix_tests'] += int(v.sum())
        stats['score_reads'] += len(s)
        stats['passes'] += 1
        stats['synchronizations'] += 1
        global_hist = hist.sum(0)
        for digit in range(size-1,-1,-1):
            if rank > global_hist[digit]: rank -= int(global_hist[digit])
            else: break
        prefix |= digit << shift
        prefix_mask |= (size-1) << shift
    boundary = prefix
    selected_ids = []
    tie_ids = []
    ties_remaining = rank
    for start in range(0,len(s),chunk_size):
        end = min(len(s),start+chunk_size)
        key = ordered_keys(s[start:end])
        ids = np.arange(start,end)
        if output_mode != 'sorted':
            strict = v[start:end] & (key>boundary)
            equal = v[start:end] & (key==boundary)
            # Stable stream compaction: inclusive prefix counts resolve boundary ties.
            take_ties = equal & (np.cumsum(equal) <= ties_remaining)
            ties_remaining -= int(take_ties.sum())
            selected_ids.extend(ids[strict | take_ties].tolist())
            continue
        selected_ids.extend(ids[v[start:end] & (key>boundary)].tolist())
        # Only the required lowest-ID ties; no N-sized tie candidate buffer.
        remaining = rank-len(tie_ids)
        if remaining > 0:
            tie_ids.extend(ids[v[start:end] & (key==boundary)][:remaining].tolist())
    selected_ids.extend(tie_ids)
    ids = np.array(selected_ids,dtype=np.int64)
    assert len(ids) == budget
    if output_mode == 'sorted':
        selected,ranking = merge_rank(ids,ordered_keys(s[ids]))
    else:
        selected,ranking = ids,dict(comparisons=0,rank_sram_reads=0,rank_sram_writes=0)
        stats['compaction_writes'] = len(ids)
    stats.update(ranking)
    stats.update(boundary_key=int(boundary),boundary_ties_selected=rank,
                 candidate_peak=len(ids),score_reads=stats['score_reads']+len(s),
                 prefix_tests=stats['prefix_tests']+int(v.sum()))
    return (selected,stats) if return_stats else selected
