"""Common retrieval interface, optional original PopKV current-token contract."""
import numpy as np
from poppim.memory.bank_mapper import map_banks
from poppim.topk.host_topk import host_topk
from poppim.topk.bank_local_topk import bank_local_topk
from poppim.topk.near_memory_topk import near_memory_topk


def run_retrieval(scores, budget, mode='host', num_banks=32,
                  mapping='striped', valid=None, reserve_current=False, selector='heap'):
    scores = np.asarray(scores)
    if scores.ndim != 1:
        raise ValueError('run_retrieval expects one query head; loop heads explicitly')
    if budget < 0 or (reserve_current and (not len(scores) or budget < 1)):
        raise ValueError('invalid budget/current-token contract')
    v = np.ones(len(scores), dtype=bool) if valid is None else np.asarray(valid, dtype=bool).copy()
    if v.shape != scores.shape:
        raise ValueError('mask shape mismatch')
    if reserve_current:
        if not v[-1]:
            raise ValueError('current token cannot be masked')
        v[-1] = False
    k = budget - int(reserve_current)
    banks = map_banks(len(scores), num_banks, mapping)
    details = {}
    if mode == 'host':
        selected = host_topk(scores, k, v)
    elif mode == 'near_memory':
        if selector == 'heap':
            selected = near_memory_topk(scores, k, v)
        elif selector == 'radix':
            from poppim.topk.radix_topk import radix_topk
            selected = radix_topk(scores,k,v,num_banks=num_banks,bank_ids=banks)
        elif selector in ('radix_unordered', 'radix_position'):
            from poppim.topk.radix_selection_only import radix_selection_only
            selected = radix_selection_only(scores,k,v,output_mode=selector.split('_')[1])
        elif selector == 'hierarchical':
            from poppim.topk.hierarchical_topk import hierarchical_topk
            selected = hierarchical_topk(scores,k,v)
        else:
            raise ValueError('unknown near-memory selector')
    elif mode in ('bank_local', 'bank_adaptive'):
        selected, details = bank_local_topk(scores, k, banks, num_banks,
                                             mode == 'bank_adaptive', v)
    else:
        raise ValueError('unknown retrieval mode')
    if reserve_current:
        selected = np.r_[selected, np.int64(len(scores)-1)]
    return {'indices': selected, 'mode': mode, **details}


def run_streaming_near_memory(encoded_query, encoded_keys, budget,
                              head_to_kv=None, chunk_size=1024):
    """Direct chunk scorer -> C heaps, without materializing [B,Hq,N] scores.

    Prefix-only contract; bank hardware timing is modeled separately.
    """
    from poppim.scoring.pim_scorer import score_chunks
    from poppim.topk.near_memory_topk import StreamingTopK
    if not 0 <= budget <= encoded_keys.sign.shape[1] or chunk_size < 1:
        raise ValueError('invalid budget or chunk size')
    batch,heads = encoded_query.sign.shape[:2]
    selectors = [[StreamingTopK(budget) for _ in range(heads)] for _ in range(batch)]
    for start,chunk in score_chunks(encoded_query,encoded_keys,head_to_kv,chunk_size):
        ids = np.arange(start,start+chunk.shape[-1],dtype=np.int64)
        for b in range(batch):
            for h in range(heads):
                selectors[b][h].feed(chunk[b,h],ids)
    return np.array([[selector.result() for selector in row] for row in selectors],dtype=np.int64)
