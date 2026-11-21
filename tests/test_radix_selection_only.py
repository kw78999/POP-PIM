import numpy as np
import pytest
from poppim.topk.radix_topk import radix_topk
from poppim.topk.radix_selection_only import radix_selection_only
from poppim.simulator.functional import run_retrieval

@pytest.mark.parametrize('mode',['unordered','position'])
@pytest.mark.parametrize('reserve',[False,True])
def test_masks_ties_boundaries(mode,reserve):
    rng=np.random.default_rng(2028)
    for n in [1,7,32,129]:
        for distribution in ['random','ties','zero']:
            s=(rng.normal(size=n) if distribution=='random' else rng.integers(-2,3,n) if distribution=='ties' else np.resize([0.,-0.],n)).astype(np.float16)
            valid=rng.random(n)>.3;valid[-1]=True
            for k in sorted(set([int(reserve),min(3,int(valid.sum())),int(valid.sum())])):
                if reserve and not k:continue
                ref=radix_topk(s,k,valid,reserve_current=reserve)
                got,stat=radix_selection_only(s,k,valid,reserve,mode,chunk_size=3,return_stats=True)
                assert np.array_equal(np.sort(ref),got)
                assert len(got)==k and len(set(got))==k
                assert valid[got].all()
                assert stat['rank_sram_writes']==stat['rank_sram_reads']==stat['comparisons']==0
                if reserve:assert got[-1]==n-1
                result=run_retrieval(s,k,'near_memory',valid=valid,reserve_current=reserve,selector='radix_'+mode)
                np.testing.assert_array_equal(got,result['indices'])

@pytest.mark.parametrize('bad',[np.inf,-np.inf,np.nan])
def test_invalid_even_if_masked(bad):
    with pytest.raises(ValueError):radix_selection_only(np.array([bad,1],np.float16),1,[False,True])

def test_exhaustive_finite_fp16_keys_selection():
    bits=np.arange(65536,dtype=np.uint16);s=bits.view(np.float16);s=s[np.isfinite(s)]
    for k in [0,1,4096,len(s)//2,len(s)]:
        np.testing.assert_array_equal(np.sort(radix_topk(s,k)),radix_selection_only(s,k))
