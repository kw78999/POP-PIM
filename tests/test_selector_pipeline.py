from dataclasses import replace
import numpy as np
import pytest
from poppim.simulator.selector_pipeline import (PipelineConfig,simulate_heap,
    simulate_radix,ingress,stage_inputs)
from poppim.topk.host_topk import host_topk


@pytest.mark.parametrize('k',[0,1,8,64])
def test_model_returns_exact_selection_and_finite_fifo(k):
    s=np.random.default_rng(2).normal(size=64).astype(np.float16)
    cfg=PipelineConfig(fifo_depth=4)
    for model in [simulate_heap,simulate_radix]:
        result=model(s,k,cfg)
        np.testing.assert_array_equal(result['selected_indices'],host_topk(s,k))
        assert result['cycles']>=result['output_cycles']
        assert result['fifo_peak_records']<=cfg.fifo_depth


def test_heap_comparator_count_does_not_remove_state_dependency():
    s=np.arange(128,dtype=np.float16) # frequent heap updates
    a=simulate_heap(s,16,PipelineConfig(comparators=32,issue_width=32))
    b=simulate_heap(s,16,PipelineConfig(comparators=512,issue_width=512))
    assert a['cycles']==b['cycles']
    assert a['ideal_comparison_cycles']>b['ideal_comparison_cycles']
    assert a['heap_updates']==112
    assert a['input_backpressure_cycles']>0
    assert a['effective_comparisons_per_cycle']<1


def test_sram_ports_counter_hazards_and_bandwidth_constraints():
    s=np.ones(128,dtype=np.float16)
    cfg=PipelineConfig(sram_banks=2,histogram_shards=2,fifo_depth=4)
    slow=simulate_radix(s,16,cfg)
    fast=simulate_radix(s,16,replace(cfg,sram_banks=8,histogram_shards=8,fifo_depth=32))
    assert slow['counter_hazard_stall_cycles']>0
    assert fast['cycles']<slow['cycles']
    assert simulate_radix(s,16,replace(cfg,comparator_latency=5))['cycles']>slow['cycles']
    assert stage_inputs(128,replace(cfg,input_bytes_per_cycle=.5))[0]>stage_inputs(128,cfg)[0]
    a=ingress([10]*20,cfg)
    b=ingress([10]*20,replace(cfg,fifo_depth=1))
    assert a['fifo_peak_records']<=4 and b['fifo_peak_records']==1
    assert b['input_backpressure_cycles']>=a['input_backpressure_cycles']
