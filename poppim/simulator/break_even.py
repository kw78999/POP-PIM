"""Placement comparison with dependency-model cycles, explicit phase overlap.

Host selection is a matched-resource hardware proxy, NOT a measured GPU kernel.
The legacy optimistic GPU model is preserved separately for sensitivity only.
"""
from dataclasses import replace
from poppim.simulator.cost_model import estimate
from poppim.simulator.selector_pipeline import PipelineConfig,simulate_heap,simulate_radix


def compose_latency(model,n,k,config,link_GBs,external_output_GBs=None,overlap=True):
    if link_GBs<=0 or (external_output_GBs is not None and external_output_GBs<=0):
        raise ValueError('bandwidth must be positive')
    cycle_ns=1000/config.clock_mhz
    transfer=2*n/link_GBs
    core=model['selection_cycles']*cycle_ns
    ingest=model.get('ingest_cycles',0)*cycle_ns
    # Radix cannot begin histogram until score staging is complete.
    if model['algorithm']=='radix':
        core-=ingest
        service=(max(transfer,ingest) if overlap else transfer+ingest)+core
    else:
        # Aggregate streaming max is an analytical composition of dependency-aware
        # service work; the per-token FIFO model supplies detailed baseline stalls.
        service=max(transfer,core) if overlap else transfer+core
    output=model['output_cycles']*cycle_ns
    if external_output_GBs is not None:
        output=max(output,4*k/external_output_GBs)  # physical output port caps link usage
    return dict(input_transfer_ns=transfer,selector_core_ns=core,staging_ingest_ns=ingest,
                index_output_ns=output,selection_service_ns=service+output)


def compare_placement(n,k,banks,host_model,near_model,host_cfg=None,near_cfg=None,
                      host_link_GBs=32,internal_link_GBs=256,overlap=True,q_heads=4):
    host_cfg=host_cfg or PipelineConfig()
    near_cfg=near_cfg or PipelineConfig()
    # Models describe one query-head engine; head engines are time-multiplexed.
    # Resources and cycles are not silently multiplied into four parallel units.
    bank_score_GBs=banks*8. # explicitly assumed per-bank score-buffer bandwidth
    a=compose_latency(host_model,n,k,host_cfg,min(host_link_GBs,bank_score_GBs),None,overlap)
    c=compose_latency(near_model,n,k,near_cfg,min(internal_link_GBs,bank_score_GBs),host_link_GBs,overlap)
    common=estimate(n,k,banks,q_heads=q_heads)
    # The chosen frame-level score/selection barrier needs finite score retention
    # across four heads before their serialized selector service. Pay its write
    # service and capacity. Input transfer above also pays reads through its cap.
    phase_score_bytes=2*n*q_heads
    phase_score_write_ns=phase_score_bytes/bank_score_GBs
    base=common['pim_score_ns']+common['query_broadcast_ns']+common['sync_ns']+phase_score_write_ns
    a_total=base+q_heads*a['selection_service_ns']
    c_total=base+q_heads*c['selection_service_ns']
    # Different algorithms/configs remain visible in resource columns.
    return dict(source='Analytical Estimate composed with dependency-aware selector simulation',
        host_selector_model='matched-resource proxy; not actual GPU timing',
        context=n,budget=k,banks=banks,q_heads=q_heads,algorithm=near_model['algorithm'],
        host_link_GBs=host_link_GBs,internal_link_GBs=internal_link_GBs,overlap=int(overlap),
        host_comparators=host_cfg.comparators,near_comparators=near_cfg.comparators,
        host_clock_mhz=host_cfg.clock_mhz,near_clock_mhz=near_cfg.clock_mhz,
        sram_banks=near_cfg.sram_banks,read_ports=near_cfg.read_ports,write_ports=near_cfg.write_ports,
        pim_score_ns=common['pim_score_ns'],common_ns=base,
        common_bank_score_buffer_bytes=phase_score_bytes,
        common_bank_score_write_ns=phase_score_write_ns,
        host_selection_service_ns=q_heads*a['selection_service_ns'],
        near_selection_service_ns=q_heads*c['selection_service_ns'],
        host_total_ns=a_total,near_total_ns=c_total,near_faster=int(c_total<a_total),
        latency_ratio_c_over_a=c_total/a_total,
        host_external_score_bytes=2*n*q_heads,near_external_index_bytes=4*k*q_heads,
        selected_kv_bytes=common['selected_kv_read_bytes_logical'],
        near_score_staging_bytes=near_model['score_staging_bytes'],
        near_candidate_buffer_bytes=near_model['candidate_buffer_bytes'],
        near_histogram_bytes=near_model.get('histogram_bytes',0),
        near_fifo_bytes=near_model.get('fifo_bytes',0),
        near_area_um2=None,near_energy_pj=None,
        equal_area_power_claim='not established; equal listed resources only',
        legacy_gpu_ideal_total_ns=estimate(n,k,banks,mode='host')['total_retrieval_ns'])
