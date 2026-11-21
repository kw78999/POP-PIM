"""Phase 3 address/event model. Separate score/counter SRAM, bounded outstanding
requests, counter forwarding option, and an explicit stable compactor schedule.
Not RTL-equivalent and not a DRAM/PIM scheduler. No measured hardware frequency.
"""
from dataclasses import dataclass,asdict
import heapq
import math
import numpy as np
from poppim.simulator.selector_pipeline import PipelineConfig,Ports,stage_inputs,merge_sort_cycles
from poppim.topk.radix_topk import radix_topk,ordered_keys,merge_rank


@dataclass(frozen=True)
class SelectionConfig(PipelineConfig):
    counter_banks: int = 8
    counter_forwarding: bool = False
    interleave_shards: bool = False

    def __post_init__(self):
        # Parent validation predates boolean options.
        PipelineConfig(**{k:getattr(self,k) for k in PipelineConfig.__dataclass_fields__})
        if self.counter_banks<1:raise ValueError('counter banks must be positive')


def reduce_histogram(cfg, start):
    """Read every private counter and reduce equal digits with dependent adds.

    SRAM counter reads share physical banks; register-forwarded counters use the
    same bounded read fabric. A node cannot issue before both children complete.
    The final descending rank scan is serial and charged separately.
    """
    size = 1 << cfg.radix_bits
    ports = [Ports(cfg.read_ports) for _ in range(cfg.counter_banks)]
    adders = Ports(min(cfg.issue_width, cfg.comparators))
    roots = []
    for digit in range(size):
        ready = []
        for shard in range(cfg.histogram_shards):
            address = (digit * cfg.histogram_shards + shard if cfg.interleave_shards
                       else shard * size + digit)
            ready.append(ports[address % cfg.counter_banks].issue(start, cfg.sram_latency))
        while len(ready) > 1:
            next_level = []
            for i in range(0, len(ready), 2):
                next_level.append(ready[i] if i + 1 == len(ready) else
                                  adders.issue(max(ready[i:i+2]), cfg.counter_latency))
            ready = next_level
        roots.extend(ready)
    return max(roots) + size * cfg.counter_latency + cfg.sync_cycles - start


def compact_schedule(keys,valid,boundary,ties,k,cfg,start):
    """Ascending input groups; registered prefix-sum + banked compact writes.

    Every group waits for earlier group's tie count and compact offset commit.
    Width is bounded by comparator and SRAM ports. Outstanding group reads are
    bounded by FIFO, so banking alone cannot yield unbounded issue throughput.
    Output is a separate drain; no uncharged gather/output overlap.
    """
    width=max(1,min(cfg.issue_width,cfg.comparators,cfg.sram_banks*cfg.read_ports,cfg.fifo_depth))
    rp=[Ports(cfg.read_ports) for _ in range(cfg.sram_banks)]
    wp=[Ports(cfg.write_ports) for _ in range(cfg.sram_banks)]
    now=start;offset=0;prefix_cycles=0;write_cycles=0;read_cycles=0
    for base in range(0,len(keys),width):
        end=min(base+width,len(keys))
        read_done=max(rp[i%cfg.sram_banks].issue(now,cfg.sram_latency) for i in range(base,end))
        read_cycles+=read_done-now
        decision=read_done+cfg.comparator_latency
        chosen=[]
        for i in range(base,end):
            if valid[i] and (keys[i]>boundary or (keys[i]==boundary and ties>0)):
                chosen.append(i)
                if keys[i]==boundary:ties-=1
        # Inclusive scan across lanes + registered offset/tie-count update.
        prefix_latency=math.ceil(math.log2(width))+1
        ready=decision+prefix_latency;prefix_cycles+=prefix_latency
        done=ready
        for j in range(len(chosen)):
            done=max(done,wp[(offset+j)%cfg.sram_banks].issue(ready,cfg.sram_latency))
        write_cycles+=done-ready;offset+=len(chosen);now=done
    assert offset==k
    return dict(gather_cycles=now-start,compaction_prefix_cycles=prefix_cycles,
                compaction_write_cycles=write_cycles,gather_read_cycles=read_cycles,
                compaction_lanes=width,compaction_writes=k)


def simulate_selection(scores,budget,config=None,valid=None,output_mode='unordered',reserve_current=False):
    cfg=config or SelectionConfig()
    s=np.asarray(scores)
    selected,stats=radix_topk(s,budget,valid,reserve_current,radix_bits=cfg.radix_bits,
                            return_stats=True,output_mode=output_mode)
    keys=ordered_keys(s);v=np.ones(len(s),dtype=bool) if valid is None else np.asarray(valid,dtype=bool).copy()
    k=budget-int(reserve_current)
    if reserve_current:v[-1]=False
    if output_mode == 'sorted':
        # The new compactor feeds candidates in position order, unlike the
        # legacy gather's strict-candidates-then-ties order. Count actual merges.
        compact_ids = np.sort(selected[:-1] if reserve_current else selected)
        _, ranking_stats = merge_rank(compact_ids, keys[compact_ids])
        stats.update(ranking_stats)
    size=1<<cfg.radix_bits
    now,input_stalls,input_peak=stage_inputs(len(s),cfg);ingest=now
    score_ports=[Ports(cfg.read_ports) for _ in range(cfg.sram_banks)]
    prefix=mask=0;rank=k;updates=hazards=fifo_stalls=hist_peak=0;passes=[];threshold_cycles=0;hist_cycles=0
    for shift in range(16-cfg.radix_bits,-1,-cfg.radix_bits) if k else []:
        begin=now
        rp=[Ports(cfg.read_ports) for _ in range(cfg.counter_banks)]
        wp=[Ports(cfg.write_ports) for _ in range(cfg.counter_banks)]
        ready_table=np.full((cfg.histogram_shards,size),now,dtype=np.int64)
        units=Ports(min(cfg.issue_width,cfg.comparators))
        now+=math.ceil(cfg.histogram_shards*size/(cfg.counter_banks*cfg.write_ports))
        finish=source=now;pending=[];hist=np.zeros(size,dtype=np.int64)
        for i,key in enumerate(keys):
            while pending and pending[0]<=source:heapq.heappop(pending)
            if len(pending)>=cfg.fifo_depth:
                release=heapq.heappop(pending);fifo_stalls+=max(0,release-source);source=max(source,release)
                while pending and pending[0]<=source:heapq.heappop(pending)
            rd=score_ports[i%cfg.sram_banks].issue(source,cfg.sram_latency)
            ready=units.issue(rd,cfg.comparator_latency)
            done=ready
            if v[i] and (int(key)&mask)==prefix:
                shard=i%cfg.histogram_shards;digit=(int(key)>>shift)&(size-1)
                hist[digit]+=1;updates+=1
                req=max(ready,int(ready_table[shard,digit]));hazards+=req-ready
                address=(digit*cfg.histogram_shards+shard if cfg.interleave_shards else shard*size+digit)
                bank=address%cfg.counter_banks
                if cfg.counter_forwarding:
                    # Private register counters with bypassed latest value. One update
                    # per bank write port per cycle; recurrence latency remains paid.
                    done=wp[bank].issue(req,cfg.counter_latency)
                else:
                    cr=rp[bank].issue(req,cfg.sram_latency)
                    done=wp[bank].issue(cr+cfg.counter_latency,cfg.sram_latency)
                ready_table[shard,digit]=done
            finish=max(finish,done);heapq.heappush(pending,done)
            hist_peak=max(hist_peak,len(pending))
        reduction=reduce_histogram(cfg,finish)
        hist_cycles+=finish-begin;threshold_cycles+=reduction;now=finish+reduction;passes.append(now-begin)
        for digit in range(size-1,-1,-1):
            if rank>hist[digit]:rank-=int(hist[digit])
            else:break
        prefix|=digit<<shift;mask|=(size-1)<<shift
    gather=compact_schedule(keys,v,prefix,rank,k,cfg,now) if k else dict(gather_cycles=0,compaction_prefix_cycles=0,compaction_write_cycles=0,gather_read_cycles=0,compaction_lanes=0,compaction_writes=0)
    ranking=merge_sort_cycles(k,cfg) if output_mode=='sorted' else 0
    # Reservation is one explicit write before drain. The functional current ID is last.
    current_write=cfg.sram_latency if reserve_current else 0
    output=math.ceil(4*budget/min(cfg.output_bytes_per_cycle,4*cfg.sram_banks*cfg.read_ports))
    cycles=now+gather['gather_cycles']+ranking+current_write+output
    counter_count=cfg.histogram_shards*size
    # Counter-forwarded design stores counters in registers, not SRAM. Report
    # logical counter traffic separately so reduced SRAM traffic is not free area.
    score_reads=len(passes)*len(s)+(len(s) if k else 0)
    counter_reads=updates+len(passes)*counter_count
    counter_writes=len(passes)*counter_count+updates
    reads=score_reads+stats['rank_sram_reads']+budget
    writes=len(s)+budget+stats['rank_sram_writes']
    if not cfg.counter_forwarding:
        reads+=counter_reads
        writes+=counter_writes
    return dict(algorithm='radix',source='Dependency-aware selector simulation; not RTL cycles',output_mode=output_mode,
        cycles=cycles,selection_cycles=cycles-output,ingest_cycles=ingest,
        histogram_cycles=hist_cycles,threshold_cycles=threshold_cycles,histogram_pass_cycles=passes,
        **gather,ranking_cycles=ranking,current_write_cycles=current_write,output_cycles=output,
        sram_reads=reads,sram_writes=writes,counter_updates=updates,
        score_sram_reads=score_reads,score_sram_writes=len(s),
        counter_logical_reads=counter_reads,counter_logical_writes=counter_writes,
        counter_sram_reads=0 if cfg.counter_forwarding else counter_reads,
        counter_sram_writes=0 if cfg.counter_forwarding else counter_writes,
        candidate_sram_reads=stats['rank_sram_reads']+budget,
        candidate_sram_writes=stats['rank_sram_writes']+budget,
        sram_read_bytes=2*score_reads+(0 if cfg.counter_forwarding else 4*counter_reads)
            +(6 if output_mode=='sorted' else 4)*(stats['rank_sram_reads']+budget),
        sram_write_bytes=2*len(s)+(0 if cfg.counter_forwarding else 4*counter_writes)
            +(6 if output_mode=='sorted' else 4)*(stats['rank_sram_writes']+budget),
        sram_access_unit='one addressed record; score 16b, counter 32b, sorted candidate 48b, index 32b',
        counter_hazard_stall_cycles=hazards,counter_wait_definition='sum of request waits; not unique wall stalls',
        pipeline_backpressure_cycles=fifo_stalls,input_backpressure_cycles=input_stalls,fifo_peak_records=input_peak,
        histogram_fifo_peak_records=hist_peak,
        score_staging_bytes=2*len(s),candidate_buffer_bytes=(12 if output_mode=='sorted' else 4)*budget,
        histogram_bytes=4*counter_count,counter_sram_bytes=0 if cfg.counter_forwarding else 4*counter_count,
        forwarding_register_bits=32*counter_count if cfg.counter_forwarding else 0,
        compaction_register_bits=cfg.fifo_depth*(16+32+1),fifo_bytes=6*cfg.fifo_depth,
        input_scores_per_cycle=len(s)/max(1,cycles),estimated_ns=cycles*1000/cfg.clock_mhz,
        selected_indices=selected,config=asdict(cfg))
