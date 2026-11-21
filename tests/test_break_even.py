from dataclasses import replace
import numpy as np
from poppim.simulator.selector_pipeline import PipelineConfig,simulate_radix
from poppim.simulator.break_even import compare_placement,compose_latency


def test_matched_resource_break_even_and_output_bytes():
    n,k=256,16
    s=np.random.default_rng(9).normal(size=n).astype(np.float16)
    cfg=PipelineConfig(sram_banks=8)
    model=simulate_radix(s,k,cfg)
    slow=compare_placement(n,k,16,model,model,cfg,cfg,host_link_GBs=.5)
    fast=compare_placement(n,k,16,model,model,cfg,cfg,host_link_GBs=256)
    assert slow['near_faster']==1
    assert slow['near_total_ns']<slow['host_total_ns']
    assert fast['near_total_ns']>=fast['host_total_ns']
    assert slow['host_external_score_bytes']==2*n*4
    assert slow['near_external_index_bytes']==4*k*4
    assert slow['near_area_um2'] is None
    no_overlap=compare_placement(n,k,16,model,model,cfg,cfg,host_link_GBs=.5,overlap=False)
    assert no_overlap['near_total_ns']>=slow['near_total_ns']


def test_large_budget_can_reverse_traffic_and_break_even():
    s=np.arange(128,dtype=np.float16)
    cfg=PipelineConfig()
    model=simulate_radix(s,128,cfg)
    result=compare_placement(128,128,16,model,model,cfg,cfg,host_link_GBs=.1)
    assert result['near_external_index_bytes']>result['host_external_score_bytes']
    assert result['near_total_ns']>result['host_total_ns']
