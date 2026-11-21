"""Deterministic dependency-aware discrete-event selector resource model.

This models a specified microarchitecture, not DRAM timing or synthesized RTL.
Heap state commits serialize successive inputs. Histogram counters have explicit
RMW hazards; per-bank SRAM ports and bounded ingress FIFO constrain scheduling.
"""
from dataclasses import dataclass,asdict
import heapq
import math
import numpy as np
from poppim.topk.radix_topk import ordered_keys,radix_topk
from poppim.topk.instrumented_heap import measured_heap


@dataclass(frozen=True)
class PipelineConfig:
    comparators: int = 32
    issue_width: int = 32
    comparator_latency: int = 2
    clock_mhz: float = 500
    sram_banks: int = 8
    read_ports: int = 1
    write_ports: int = 1
    sram_latency: int = 1
    fifo_depth: int = 16
    input_bytes_per_cycle: float = 64
    output_bytes_per_cycle: float = 16
    histogram_shards: int = 8
    counter_latency: int = 1
    sync_cycles: int = 8
    radix_bits: int = 4
    sort_lanes: int = 8

    def __post_init__(self):
        if any(v<=0 for v in asdict(self).values()) or self.radix_bits not in (1,2,4,8):
            raise ValueError('invalid positive pipeline parameters')


class Ports:
    def __init__(self,count): self.ready=[0]*count
    def issue(self,earliest,latency=1):
        i=min(range(len(self.ready)),key=self.ready.__getitem__)
        start=max(int(earliest),self.ready[i])
        self.ready[i]=start+1
        return start+latency


def stage_inputs(n,cfg):
    """Bounded outstanding score writes, bank conflicts and ingress handshake."""
    ports=[Ports(cfg.write_ports) for _ in range(cfg.sram_banks)]
    pending=[]; producer=0.; stalls=peak=finish=0
    for i in range(n):
        arrival=int(producer)
        while pending and pending[0]<=arrival: heapq.heappop(pending)
        if len(pending)>=cfg.fifo_depth:
            release=heapq.heappop(pending)
            stalls+=max(0,release-arrival)
            arrival=max(arrival,release); producer=max(producer,float(arrival))
            while pending and pending[0]<=arrival: heapq.heappop(pending)
        done=ports[i%cfg.sram_banks].issue(arrival,cfg.sram_latency)
        heapq.heappush(pending,done)
        finish=max(finish,done); peak=max(peak,len(pending))
        producer+=2/cfg.input_bytes_per_cycle
    return finish,stalls,peak


def ingress(service_cycles,config):
    """Finite FIFO, serial state machine consumer, backpressured producer.

    fifo_depth counts waiting+in-service records. Producer advances only when a
    record has been accepted. An output FIFO is modeled separately by drain time.
    """
    completions=[]
    server=0
    producer=0.
    stalls=idle=peak=0
    interval=2/config.input_bytes_per_cycle
    for duration in service_cycles:
        arrival=int(math.floor(producer))
        while completions and completions[0] <= arrival: heapq.heappop(completions)
        if len(completions)>=config.fifo_depth:
            earliest=heapq.heappop(completions)
            stalls+=max(0,earliest-arrival)
            arrival=max(arrival,earliest)
            producer=max(producer,float(arrival))
            while completions and completions[0] <= arrival: heapq.heappop(completions)
        start=max(server,arrival)
        idle+=max(0,arrival-server)
        server=start+max(1,int(duration))
        heapq.heappush(completions,server)
        peak=max(peak,len(completions))
        producer+=interval
    return dict(stream_cycles=server,input_backpressure_cycles=stalls,
                input_idle_cycles=idle,fifo_peak_records=peak)


def merge_sort_cycles(k,config):
    """Level barriers + candidate SRAM bandwidth, bottom-up parallel merge.

    Independent merge engines limited by lane count and bank read/write ports.
    Each output in a merge has one comparison and dependent mux. No divide-by-
    arbitrary comparator count: one engine emits <=1/compare-latency cycles.
    """
    cycles=0
    width=1
    while width<k:
        independent=math.ceil(k/(2*width))
        engines=min(config.sort_lanes,config.comparators,config.issue_width,
                    independent,max(1,config.sram_banks*config.read_ports//2),
                    config.sram_banks*config.write_ports)
        cycles+=math.ceil(k/engines)*(config.comparator_latency+config.sram_latency)+config.sync_cycles
        width*=2
    return cycles


def simulate_heap(scores,budget,config=None,valid=None):
    cfg=config or PipelineConfig()
    selected,stats=measured_heap(scores,budget,valid,record_events=True)
    durations=[]
    # Each input must commit all heap writes before the next root inspection.
    # Comparisons within this non-speculative heap are sequential, not 512-wide.
    for _,begin,end,_ in stats['per_token']:
        now=0
        for op,addresses in stats['events'][begin:end]:
            if op=='compare': now+=cfg.comparator_latency
            else:
                counts=np.bincount(np.array(addresses,dtype=int)%cfg.sram_banks,
                                   minlength=cfg.sram_banks)
                ports=cfg.read_ports if op=='read' else cfg.write_ports
                now+=int(max((int(x)+ports-1)//ports for x in counts))+cfg.sram_latency-1
        durations.append(max(now,1))
    flow=ingress(durations,cfg)
    ranking=merge_sort_cycles(budget,cfg)
    output=math.ceil(4*budget/cfg.output_bytes_per_cycle)
    cycles=flow['stream_cycles']+ranking+output
    total_cmp=stats['comparisons']
    return {**{k:v for k,v in stats.items() if k not in ('events','per_token','level_updates')},
        **flow,'algorithm':'heap','source':'Dependency-aware selector simulation',
        'cycles':cycles,'selection_cycles':cycles-output,'ranking_cycles':ranking,
        'output_cycles':output,'estimated_ns':cycles*1000/cfg.clock_mhz,
        'ideal_comparison_cycles':math.ceil(total_cmp/min(cfg.comparators,cfg.issue_width)),
        'effective_comparisons_per_cycle':total_cmp/max(1,cycles-output),
        'input_scores_per_cycle':len(durations)/max(1,cycles),
        'counter_hazard_stall_cycles':0,'synchronizations':math.ceil(math.log2(max(budget,1))),
        'sram_reads':stats['heap_sram_reads']+stats['rank_sram_reads'],
        'sram_writes':stats['heap_sram_writes']+stats['rank_sram_writes'],
        'fifo_bytes':6*cfg.fifo_depth,'rank_workspace_bytes':6*budget,
        'selected_indices':selected,'config':asdict(cfg)}


def simulate_radix(scores,budget,config=None,valid=None):
    cfg=config or PipelineConfig()
    selected,stats=radix_topk(scores,budget,valid,radix_bits=cfg.radix_bits,return_stats=True)
    s=np.asarray(scores)
    v=np.ones(len(s),dtype=bool) if valid is None else np.asarray(valid,dtype=bool)
    keys=ordered_keys(s)
    size=1<<cfg.radix_bits
    score_read=[Ports(cfg.read_ports) for _ in range(cfg.sram_banks)]
    # Initial ingest stages the whole prefix. Capacity is reported, never implicit.
    now,input_stalls,input_peak=stage_inputs(len(s),cfg)
    ingest_cycles=now
    threshold=0; mask=0; rank=int(budget)
    pass_cycles=[]; hazards=0; updates=0; pipeline_stalls=0
    total_score_reads=0
    for shift in range(16-cfg.radix_bits,-1,-cfg.radix_bits) if budget else []:
        start=now
        # Counter SRAM is separate from score staging SRAM, with same port config.
        hread=[Ports(cfg.read_ports) for _ in range(cfg.sram_banks)]
        hwrite=[Ports(cfg.write_ports) for _ in range(cfg.sram_banks)]
        counter_ready=np.full((cfg.histogram_shards,size),now,dtype=np.int64)
        prefix_units=Ports(min(cfg.issue_width,cfg.comparators))
        now+=math.ceil(cfg.histogram_shards*size/(cfg.sram_banks*cfg.write_ports))
        finish=now
        pending=[]; source_time=now
        hist=np.zeros(size,dtype=np.int64)
        for i,key in enumerate(keys):
            while pending and pending[0]<=source_time: heapq.heappop(pending)
            if len(pending)>=cfg.fifo_depth:
                released=heapq.heappop(pending)
                pipeline_stalls+=max(0,released-source_time)
                source_time=max(source_time,released)
                while pending and pending[0]<=source_time: heapq.heappop(pending)
            read_finish=score_read[i%cfg.sram_banks].issue(source_time,cfg.sram_latency)
            ready=prefix_units.issue(read_finish,cfg.comparator_latency)
            finish=max(finish,ready)
            if not v[i] or (int(key)&mask)!=threshold:
                heapq.heappush(pending,ready)
                continue
            shard=i%cfg.histogram_shards
            digit=(int(key)>>shift)&(size-1)
            hist[digit]+=1; updates+=1
            request=max(ready,int(counter_ready[shard,digit]))
            hazards+=max(0,int(counter_ready[shard,digit])-ready)
            bank=(shard*size+digit)%cfg.sram_banks
            rd=hread[bank].issue(request,cfg.sram_latency)
            wr=hwrite[bank].issue(rd+cfg.counter_latency,cfg.sram_latency)
            counter_ready[shard,digit]=wr
            finish=max(finish,wr)
            heapq.heappush(pending,wr)
        total_score_reads+=len(s)
        # Tree reduction and descending prefix rank scan. Includes histogram reads.
        reduction=(math.ceil(math.log2(cfg.histogram_shards))*
                   (math.ceil(size/max(1,min(cfg.issue_width,cfg.sram_banks*cfg.read_ports)))+
                    cfg.counter_latency+cfg.sram_latency)+
                   size+cfg.sync_cycles)
        now=finish+reduction
        for digit in range(size-1,-1,-1):
            if rank>hist[digit]: rank-=int(hist[digit])
            else: break
        threshold|=digit<<shift; mask|=(size-1)<<shift
        pass_cycles.append(now-start)
    # Replay/gather, bounded K output buffer and stable tie-ID handling.
    gather=(math.ceil(len(s)/min(cfg.issue_width,cfg.comparators,
               cfg.sram_banks*cfg.read_ports))+cfg.comparator_latency+
            math.ceil(budget/(cfg.sram_banks*cfg.write_ports))) if budget else 0
    ranking=merge_sort_cycles(budget,cfg)
    output=math.ceil(4*budget/cfg.output_bytes_per_cycle)
    cycles=now+gather+ranking+output
    return {**{k:v for k,v in stats.items() if k!='bank_pass_updates'},
        'source':'Dependency-aware selector simulation','cycles':cycles,
        'selection_cycles':cycles-output,'ranking_cycles':ranking,'output_cycles':output,
        'ingest_cycles':ingest_cycles,'histogram_pass_cycles':pass_cycles,'gather_cycles':gather,
        'estimated_ns':cycles*1000/cfg.clock_mhz,
        'ideal_comparison_cycles':math.ceil(stats['comparisons']/min(cfg.comparators,cfg.issue_width)),
        'counter_hazard_stall_cycles':hazards,
        'counter_wait_definition':'sum of per-request counter dependency waits, not unique wall-clock stalls',
        'pipeline_backpressure_cycles':pipeline_stalls,
        'effective_comparisons_per_cycle':stats['comparisons']/max(1,cycles-output),
        'input_scores_per_cycle':len(s)/max(1,cycles),
        'reduction_adds':stats['passes']*(cfg.histogram_shards-1)*size,
        'sram_reads':total_score_reads+(len(s) if budget else 0)+updates+stats['rank_sram_reads']+stats['passes']*cfg.histogram_shards*size,
        'sram_writes':len(s)+updates+stats['passes']*cfg.histogram_shards*size+stats['rank_sram_writes'],
        'histogram_bytes':cfg.histogram_shards*size*4,
        'fifo_bytes':6*cfg.fifo_depth,
        'input_backpressure_cycles':input_stalls,
        'input_idle_cycles':0,'fifo_peak_records':input_peak,
        'selected_indices':selected,'config':asdict(cfg)}


def simulate_hierarchical(scores,budget,config=None,valid=None,tile_size=256):
    """One bounded tile sorter feeding an exact global heap, serial phases.

    Exact local quota is min(K,tile count). For K>=tile size there is no candidate
    traffic reduction. This deliberately avoids unmodeled multi-tile parallelism.
    """
    from poppim.topk.host_topk import validate
    from poppim.topk.radix_topk import merge_rank
    from dataclasses import replace
    s,v=validate(scores,budget,valid)
    cfg=config or PipelineConfig()
    keys=ordered_keys(s)
    forwarded=[]; local_cycles=local_comparisons=local_reads=local_writes=0
    for start in range(0,len(s),tile_size):
        ids=np.flatnonzero(v[start:start+tile_size])+start
        order,stats=merge_rank(ids,keys[ids])
        forwarded.extend(sorted(order[:min(budget,len(ids))].tolist()))
        local_comparisons+=stats['comparisons']
        local_reads+=stats['rank_sram_reads'];local_writes+=stats['rank_sram_writes']
        local_cycles+=merge_sort_cycles(len(ids),replace(cfg,sort_lanes=1))
    compact=s[np.array(forwarded,dtype=np.int64)]
    global_result=simulate_heap(compact,budget,cfg)
    selected=np.array(forwarded,dtype=np.int64)[global_result['selected_indices']]
    total=local_cycles+global_result['cycles']
    return {**global_result,'algorithm':'hierarchical','selected_indices':selected,
        'cycles':total,'selection_cycles':local_cycles+global_result['selection_cycles'],
        'local_tile_cycles':local_cycles,'tile_size':tile_size,
        'candidates_forwarded':len(forwarded),'comparisons':local_comparisons+global_result['comparisons'],
        'sram_reads':local_reads+global_result['sram_reads'],
        'sram_writes':local_writes+global_result['sram_writes'],
        'tile_buffer_bytes':12*min(len(s),tile_size),
        'effective_comparisons_per_cycle':(local_comparisons+global_result['comparisons'])/max(1,total-global_result['output_cycles']),
        'estimated_ns':total*1000/cfg.clock_mhz,
        'input_scores_per_cycle':len(s)/max(1,total),
        'source':'Dependency-aware selector simulation; serial exact tile/global phases'}
