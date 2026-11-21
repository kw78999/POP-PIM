"""Exact materialized threshold baseline, separate from streaming C design."""
import numpy as np
from poppim.topk.host_topk import validate


def threshold_topk(scores,budget,valid=None):
    s,v = validate(scores,budget,valid)
    ids = np.flatnonzero(v)
    if budget == 0:
        return np.empty(0,dtype=np.int64)
    threshold = np.partition(s[ids],len(ids)-budget)[len(ids)-budget]
    strict = ids[s[ids] > threshold]
    ties = ids[s[ids] == threshold][:budget-len(strict)]
    selected = np.r_[strict,ties]
    return selected[np.lexsort((selected,-s[selected].astype(np.float32)))].astype(np.int64)
