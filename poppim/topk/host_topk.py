"""Deterministic global top-k: descending stored score, ascending token ID."""
import numpy as np


def validate(scores, budget, valid=None):
    s = np.asarray(scores)
    if s.ndim != 1 or s.dtype.kind != 'f' or not np.isfinite(s).all():
        raise ValueError('scores must be a finite 1D floating array')
    if not isinstance(budget, (int, np.integer)) or budget < 0:
        raise ValueError('budget must be a nonnegative integer')
    v = np.ones(len(s), dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    if v.shape != s.shape or budget > v.sum():
        raise ValueError('invalid mask or budget exceeds eligible tokens')
    return s, v


def host_topk(scores, budget, valid=None):
    s, v = validate(scores, budget, valid)
    ids = np.flatnonzero(v)
    return ids[np.lexsort((ids, -s[ids].astype(np.float32)))[:budget]].astype(np.int64)


def accuracy(reference, selected):
    a, b = set(map(int, reference)), set(map(int, selected))
    return {'recall': len(a & b)/len(a) if a else 1.0,
            'exact_match': int(a == b)}
