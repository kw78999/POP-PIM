import itertools
import numpy as np
import pytest
from poppim.topk.host_topk import host_topk
from poppim.topk.radix_topk import radix_topk,ordered_keys
from poppim.topk.hierarchical_topk import hierarchical_topk
from poppim.topk.instrumented_heap import measured_heap
from poppim.simulator.functional import run_retrieval


def test_every_finite_fp16_key_and_zero():
    s=np.arange(65536,dtype=np.uint16).view(np.float16)
    s=s[np.isfinite(s)]
    keys=ordered_keys(s)
    oracle=np.argsort(s.astype(np.float32),kind='stable')
    assert np.all(keys[oracle][1:]>=keys[oracle][:-1])
    assert keys[s==0][0]==keys[s==0][1]
    assert len(np.unique(keys))==len(np.unique(s))
    np.testing.assert_array_equal(radix_topk(s,4096),host_topk(s,4096))


def test_exhaustive_small_sorting_oracle():
    for values in itertools.product([-1.,-0.,0.,1.],repeat=4):
        s=np.array(values,dtype=np.float16)
        for k in range(5):
            expected=np.array(sorted(range(4),key=lambda i:(-float(s[i]),i))[:k])
            np.testing.assert_array_equal(radix_topk(s,k,chunk_size=3),expected)


@pytest.mark.parametrize('n',[0,1,7,65,513])
@pytest.mark.parametrize('bits',[1,2,4,8])
def test_exact_algorithms_masks_ties_banks(n,bits):
    s=(np.random.default_rng(71).integers(-8,9,n)/8).astype(np.float16)
    valid=np.arange(n)%3!=1
    for k in set([0,min(1,int(valid.sum())),int(valid.sum())//2,int(valid.sum())]):
        a=host_topk(s,k,valid)
        r,stats=radix_topk(s,k,valid,radix_bits=bits,chunk_size=7,num_banks=16,return_stats=True)
        np.testing.assert_array_equal(r,a)
        np.testing.assert_array_equal(hierarchical_topk(s,k,valid,tile_size=17),a)
        np.testing.assert_array_equal(measured_heap(s,k,valid)[0],a)
        assert stats['candidate_peak']<=k
        if k: assert stats['score_reads']==len(s)*(16//bits+1)


def test_integration_current_mask_and_dtype_policy():
    s=np.array([1,-0.,0.,9,-1],dtype=np.float16)
    valid=[True,True,True,False,True]
    for k in [1,2,4]:
        expected=run_retrieval(s,k,valid=valid,reserve_current=True)['indices']
        for selector in ['radix','hierarchical']:
            actual=run_retrieval(s,k,'near_memory',valid=valid,reserve_current=True,selector=selector)['indices']
            np.testing.assert_array_equal(actual,expected)
        np.testing.assert_array_equal(radix_topk(s,k,valid,reserve_current=True),expected)
    for value in [np.nan,np.inf,-np.inf]:
        with pytest.raises(ValueError): radix_topk(np.array([value],dtype=np.float16),0,valid=[False])
    with pytest.raises(ValueError): radix_topk(np.ones(3,dtype=np.float32),1)
    with pytest.raises(ValueError): radix_topk(s,6)


def test_hierarchy_exactness_condition_and_counterexample():
    s=np.zeros(64,dtype=np.float16); s[:8]=np.arange(8)+10
    exact,stats=hierarchical_topk(s,8,tile_size=16,return_stats=True)
    np.testing.assert_array_equal(exact,host_topk(s,8))
    assert stats['exact_guarantee']
    approximate,stats=hierarchical_topk(s,8,tile_size=16,candidate_limit=4,return_stats=True)
    assert not stats['exact_guarantee']
    assert len(set(exact)&set(approximate))==4
