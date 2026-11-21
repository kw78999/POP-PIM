"""Explicit binary min-heap, actual control path and logical SRAM events.

Each event: (operation, logical addresses). The selector pipeline replays the
same events, including inter-level and inter-token state dependencies.
"""
import numpy as np
from poppim.topk.host_topk import validate
from poppim.topk.radix_topk import ordered_keys,merge_rank


def measured_heap(scores,budget,valid=None,record_events=True):
    s,v = validate(scores,budget,valid)
    keys = ordered_keys(s)
    heap = []
    events = []
    comparisons = reads = writes = updates = 0
    level_updates = {}
    per_token = []
    def event(op,addresses):
        nonlocal comparisons,reads,writes
        if op=='compare': comparisons += 1
        elif op=='read': reads += len(addresses)
        elif op=='write': writes += len(addresses)
        if record_events: events.append((op,tuple(addresses)))
    def less(a,b):
        event('compare',())
        return a < b
    for idx in np.flatnonzero(v):
        start = len(events)
        before = comparisons
        item = (int(keys[idx]),-int(idx))
        if len(heap) < budget:
            pos = len(heap); heap.append(item); event('write',(pos,))
            while pos:
                parent = (pos-1)//2
                event('read',(pos,parent))
                if not less(heap[pos],heap[parent]): break
                heap[pos],heap[parent] = heap[parent],heap[pos]
                event('write',(pos,parent))
                pos = parent
        elif budget:
            event('read',(0,))
            if less(heap[0],item):
                updates += 1
                heap[0]=item; event('write',(0,))
                pos = 0
                while 2*pos+1 < budget:
                    left,right = 2*pos+1,2*pos+2
                    event('read',(left,))
                    child=left
                    if right < budget:
                        event('read',(right,))
                        if less(heap[right],heap[left]): child=right
                    event('read',(pos,))
                    if not less(heap[child],heap[pos]): break
                    heap[pos],heap[child]=heap[child],heap[pos]
                    event('write',(pos,child))
                    level=int(np.floor(np.log2(pos+1)))
                    level_updates[level]=level_updates.get(level,0)+1
                    pos=child
        per_token.append((int(idx),start,len(events),comparisons-before))
    ids=np.array([-i for _,i in heap],dtype=np.int64)
    rank,sort_stats=merge_rank(ids,keys[ids])
    stats=dict(algorithm='heap',
        heap_comparisons=comparisons,heap_updates=updates,heap_sram_reads=reads,
        heap_sram_writes=writes,score_reads=int(v.sum()),score_writes=0,
        candidate_buffer_bytes=6*budget,score_staging_bytes=0,
        comparisons_per_score=comparisons/max(1,int(v.sum())),level_updates=level_updates,
        events=events,per_token=per_token,**sort_stats)
    # merge_rank's comparison field is rank-only; restore full total explicitly.
    stats['comparisons']=comparisons+sort_stats['comparisons']
    return rank,stats
