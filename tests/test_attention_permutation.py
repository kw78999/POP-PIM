import numpy as np
from poppim.simulator.attention_reference import indexed_attention,rope
from poppim.topk.radix_topk import radix_topk


def test_attention_identity_mask_rope_current_and_pairs():
    rng=np.random.default_rng(8);n=100;d=32
    q=rope(rng.normal(size=d),n-1).astype(np.float16)
    k=rope(rng.normal(size=(n,d)),np.arange(n)).astype(np.float16);v=rng.normal(size=(n,d)).astype(np.float16)
    s=rng.integers(-2,3,n).astype(np.float16);mask=np.ones(n,bool);mask[::3]=False;mask[-1]=True
    ids=radix_topk(s,40,mask,True)
    bias=np.linspace(-1,1,n);slot=rng.random(len(ids))>.2;slot[-1]=True
    ref=indexed_attention(q,k,v,ids,slot,np.float64,np.float64,bias=bias,split_tokens=11)
    for order in [np.arange(len(ids)),np.argsort(ids),rng.permutation(len(ids))]:
        got=indexed_attention(q,k,v,ids[order],slot[order],np.float64,np.float64,bias=bias,split_tokens=11)
        np.testing.assert_allclose(got,ref,atol=2e-15,rtol=2e-14)
        got32=indexed_attention(q,k,v,ids[order],slot[order],bias=bias,split_tokens=11)
        np.testing.assert_allclose(got32,ref,atol=2e-6,rtol=2e-5)
    wrong_v=v[::-1].copy()
    assert np.max(np.abs(indexed_attention(q,k,wrong_v,ids,slot,bias=bias)-ref))>.01
    # Reapplying positional rotation by compacted slot is not valid.
    wrong_k=rope(k,np.arange(n)[::-1]).astype(np.float16)
    assert np.max(np.abs(indexed_attention(q,wrong_k,v,ids,slot,bias=bias)-ref))>.001
    # Slot masks have to move with token IDs, even if the selected set is exact.
    order=rng.permutation(len(ids))
    wrong_mask=indexed_attention(q,k,v,ids[order],slot,bias=bias,split_tokens=11)
    assert np.max(np.abs(wrong_mask-ref))>.001
