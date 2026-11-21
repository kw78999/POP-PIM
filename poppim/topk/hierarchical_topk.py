"""Exact tile Top-k retains min(K,tile_size); smaller quotas are approximate."""
import numpy as np
from poppim.topk.host_topk import validate
from poppim.topk.radix_topk import ordered_keys,merge_rank
from poppim.topk.near_memory_topk import StreamingTopK


def hierarchical_topk(scores,budget,valid=None,tile_size=256,candidate_limit=None,return_stats=False):
    s,v=validate(scores,budget,valid)
    key=ordered_keys(s)
    if tile_size<1 or (candidate_limit is not None and candidate_limit<1):
        raise ValueError('invalid tile size/candidate quota')
    exact=candidate_limit is None or candidate_limit>=min(budget,tile_size)
    # Functional candidates are forwarded tile by tile; no all-tile candidate array.
    selector=StreamingTopK(budget)
    count=comparisons=0
    max_tile=0
    for start in range(0,len(s),tile_size):
        ids=np.flatnonzero(v[start:start+tile_size])+start
        order,stats=merge_rank(ids,key[ids])
        local_k=min(budget,len(ids),candidate_limit if candidate_limit is not None else budget)
        candidates=np.sort(order[:local_k])
        selector.feed(s[candidates],candidates)
        count+=len(candidates); comparisons+=stats['comparisons']
        max_tile=max(max_tile,len(ids))
    if selector.seen < budget:
        raise ValueError('limited hierarchical candidates cannot fill requested budget')
    selected=selector.result()
    stats=dict(algorithm='hierarchical_exact' if exact else 'hierarchical_approximate',
        exact_guarantee=exact,tile_sort_comparisons=comparisons,candidates_forwarded=count,
        tile_buffer_bytes=12*max_tile,candidate_buffer_bytes=6*budget,
        score_staging_bytes=0,tile_size=tile_size,
        note='local exactness requires min(K,eligible tile count); global merge heap separately modeled')
    return (selected,stats) if return_stats else selected
