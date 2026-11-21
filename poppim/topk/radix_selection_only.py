"""Exact selection with stable position compaction; no score ranking.

Unordered is an interface guarantee (set only). This implementation naturally
emits increasing token IDs, also satisfying the stronger position contract.
"""
from poppim.topk.radix_topk import radix_topk


def radix_selection_only(scores, budget, valid=None, reserve_current=False,
                         output_mode='unordered', **kwargs):
    if output_mode not in ('unordered', 'position'):
        raise ValueError('selection-only requires unordered or position output')
    return radix_topk(scores,budget,valid,reserve_current,
                      output_mode=output_mode,**kwargs)
