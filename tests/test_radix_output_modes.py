from dataclasses import replace
import numpy as np
import pytest
from poppim.simulator.selection_pipeline import SelectionConfig,simulate_selection
from poppim.topk.radix_topk import radix_topk

@pytest.mark.parametrize('banks',[1,4,16])
def test_cycle_conservation_and_resource_capacity(banks):
    s=np.random.default_rng(4).integers(-2,3,257).astype(np.float16)
    cfg=SelectionConfig(sram_banks=banks,counter_banks=banks,interleave_shards=True)
    for mode in ['sorted','unordered','position']:
        r=simulate_selection(s,64,cfg,output_mode=mode)
        assert r['cycles']==sum(r[x] for x in ['ingest_cycles','histogram_cycles','threshold_cycles','gather_cycles','ranking_cycles','current_write_cycles','output_cycles'])
        assert r['compaction_writes']==64
        assert r['fifo_peak_records']<=cfg.fifo_depth
        assert r['input_scores_per_cycle']==len(s)/r['cycles']
        np.testing.assert_array_equal(np.sort(r['selected_indices']),np.sort(radix_topk(s,64)))
        if mode!='sorted':assert r['ranking_cycles']==0 and r['candidate_buffer_bytes']==256


def test_counter_forwarding_changes_dependency_not_selection():
    s=np.ones(1024,np.float16);c=SelectionConfig(counter_banks=8,interleave_shards=True,counter_latency=2)
    a=simulate_selection(s,128,c);b=simulate_selection(s,128,replace(c,counter_forwarding=True))
    np.testing.assert_array_equal(a['selected_indices'],b['selected_indices'])
    assert b['cycles']<a['cycles']
    assert a['sram_reads']-b['sram_reads']==a['counter_logical_reads']
    assert b['counter_sram_reads']==b['counter_sram_writes']==0
    assert b['forwarding_register_bits']>0


def test_current_and_zero_budget_model():
    s=np.array([-2,-1,0,1],np.float16)
    for k in [0,1,4]:
        for reserve in ([False,True] if k else [False]):
            r=simulate_selection(s,k,reserve_current=reserve)
            assert len(r['selected_indices'])==k
            assert r['current_write_cycles']==int(reserve)


@pytest.mark.parametrize('bits',[1,2,4,8])
def test_radix_width_and_access_accounting(bits):
    s=np.random.default_rng(57).normal(size=64).astype(np.float16)
    r=simulate_selection(s,13,SelectionConfig(radix_bits=bits))
    assert len(r['histogram_pass_cycles'])==16//bits
    assert r['score_sram_reads']==(16//bits+1)*len(s)
    assert r['sram_reads']==sum(r[x] for x in
        ['score_sram_reads','counter_sram_reads','candidate_sram_reads'])
    assert r['sram_writes']==sum(r[x] for x in
        ['score_sram_writes','counter_sram_writes','candidate_sram_writes'])


def test_reduction_cannot_outrun_counter_read_ports():
    from poppim.simulator.selection_pipeline import reduce_histogram
    c=SelectionConfig(histogram_shards=32,counter_banks=1)
    assert reduce_histogram(c,0)>=32*16
