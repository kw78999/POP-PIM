"""Exact streaming heap selector, O(k) candidates, no full-score buffer."""
import heapq
import numpy as np
from poppim.topk.host_topk import validate


class StreamingTopK:
    def __init__(self, budget):
        if not isinstance(budget, (int, np.integer)) or budget < 0:
            raise ValueError('invalid budget')
        self.budget = int(budget)
        self.heap = []
        self.seen = 0
        self.last_id = -1

    def feed(self, scores, ids):
        scores, ids = np.asarray(scores), np.asarray(ids)
        if (scores.ndim != 1 or ids.shape != scores.shape or ids.dtype.kind not in 'iu'
                or not np.isfinite(scores).all() or np.any(ids < 0)):
            raise ValueError('invalid stream chunk')
        # Require ordered disjoint token chunks; no O(N) duplicate-ID set.
        if len(ids) and (ids[0] <= self.last_id or np.any(ids[1:] <= ids[:-1])):
            raise ValueError('IDs must be strictly increasing across chunks')
        if len(ids):
            self.last_id = int(ids[-1])
        self.seen += len(ids)
        for s, idx in zip(scores, ids):
            item = (float(s), -int(idx))
            if len(self.heap) < self.budget:
                heapq.heappush(self.heap, item)
            elif self.budget and item > self.heap[0]:
                heapq.heapreplace(self.heap, item)

    def result(self):
        if self.seen < self.budget:
            raise ValueError('stream has fewer eligible tokens than budget')
        return np.array([-i for _, i in sorted(self.heap, reverse=True)], dtype=np.int64)


def near_memory_topk(scores, budget, valid=None, chunk_size=1024):
    s, v = validate(scores, budget, valid)
    if chunk_size < 1:
        raise ValueError('chunk_size must be positive')
    selector = StreamingTopK(budget)
    for start in range(0, len(s), chunk_size):
        ids = np.flatnonzero(v[start:start+chunk_size]) + start
        selector.feed(s[ids], ids)
    return selector.result()
