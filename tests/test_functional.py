import ctypes
import math
import numpy as np
import pytest
from poppim.encoding.bitpacking import (pack_bits, unpack_bits, encode_query,
                                       encode_keys)
from poppim.scoring.reference import (score_encoded, dense_encoded_oracle,
                                      popcount32, gqa_mapping, fma32)
from poppim.scoring.pim_scorer import score_chunks
from poppim.topk.host_topk import host_topk, accuracy
from poppim.topk.near_memory_topk import near_memory_topk, StreamingTopK
from poppim.topk.threshold_topk import threshold_topk
from poppim.topk.bank_local_topk import bank_local_topk, allocate_quotas
from poppim.memory.bank_mapper import map_banks
from poppim.simulator.functional import run_retrieval, run_streaming_near_memory
from poppim.simulator.cost_model import HardwareConfig, estimate


@pytest.mark.parametrize('dim', [1,16,31,32,33,64,96,127,128])
def test_pack_roundtrip_and_popcount(dim):
    bits = np.random.default_rng(dim).integers(0,2,(2,3,dim)).astype(bool)
    packed = pack_bits(bits)
    assert packed.dtype == np.uint32
    np.testing.assert_array_equal(unpack_bits(packed,dim), bits)
    np.testing.assert_array_equal(popcount32(packed).sum(-1),bits.sum(-1))


def test_upstream_hand_case_and_empty_buckets():
    # PopKV/tests/test_popkv.py hand case, same signs and magnitude threshold.
    q = encode_query(np.array([[[1,-1,3,-3,0]]],dtype=np.float32),32)
    assert q.sign.item() == 0b10101
    assert q.magnitude.item() == 0b01100
    assert q.low.item() == np.float32(2/3)
    assert q.delta.item() == np.float32(3-np.float32(2/3))
    for value in [-2,0,2]:
        x = np.full((1,1,128),value,dtype=np.float16)
        for encoder in [encode_query,encode_keys]:
            encoded = encoder(x)
            assert np.all(encoded.low == abs(value))
            assert np.all(encoded.delta == 0)


def scalar_encoding_oracle(values, group, token_group=None):
    # Deliberately scalar implementation, independent from vectorized port.
    sign, mag = values >= 0, np.zeros(values.shape,dtype=bool)
    lo, de = [], []
    for a in range(values.shape[0]):
        lows, deltas = [], []
        for t in range(values.shape[1]):
            for d in range(values.shape[2]):
                v = float(values[a,t,d])
                pool = (values[a,t] if token_group is None else
                        values[a,t//token_group*token_group:(t//token_group+1)*token_group,d])
                mag[a,t,d] = v > float(max(pool))*.5 if v >= 0 else v < float(min(pool))*.5
            rowlo, rowde = [], []
            for start in range(0,values.shape[-1],group):
                low = [abs(float(values[a,t,d])) for d in range(start,min(start+group,values.shape[-1])) if not mag[a,t,d]]
                high = [abs(float(values[a,t,d])) for d in range(start,min(start+group,values.shape[-1])) if mag[a,t,d]]
                l = sum(low)/len(low) if low else sum(high)/len(high)
                h = sum(high)/len(high) if high else l
                rowlo.append(l); rowde.append(h-l)
            lows.append(rowlo); deltas.append(rowde)
        lo.append(lows); de.append(deltas)
    return sign,mag,np.array(lo),np.array(de)


@pytest.mark.parametrize('group',[16,32,64])
@pytest.mark.parametrize('token_group',[8,16,32,64,128])
def test_encoding_against_scalar_oracle(group,token_group):
    rng = np.random.default_rng(31)
    # Dyadic inputs ensure warp sums and scalar sums agree before division.
    k = (rng.integers(-16,17,(2,37,128))/4).astype(np.float16)
    for x, enc, tg in [(k,encode_keys(k,group,token_group),token_group),
                       (k[:,:2],encode_query(k[:,:2],group),None)]:
        s,m,l,d = scalar_encoding_oracle(x,group,tg)
        np.testing.assert_array_equal(enc.sign,pack_bits(s))
        np.testing.assert_array_equal(enc.magnitude,pack_bits(m))
        dtype = np.float16 if tg else np.float32
        np.testing.assert_allclose(enc.low,l.astype(dtype),rtol=2e-7,atol=2e-7)
        np.testing.assert_allclose(enc.delta,d.astype(dtype),rtol=2e-7,atol=3e-7)


@pytest.mark.parametrize('group',[16,32,64])
@pytest.mark.parametrize('dim',[33,64,127,128])
def test_score_dense_oracle_precision_gqa_and_chunking(group,dim):
    rng = np.random.default_rng(123)
    q = encode_query(rng.normal(size=(2,4,dim)).astype(np.float16),group)
    k = encode_keys(rng.normal(size=(2,37,dim)).astype(np.float16),group,16)
    actual = score_encoded(q,k,chunk_size=7)
    dense = dense_encoded_oracle(q,k)
    tolerance = 5e-6*(1+np.abs(dense))
    if dim == 128:
        tolerance += .501*np.spacing(np.abs(actual)).astype(np.float64)
    assert np.all(np.abs(actual.astype(float)-dense) <= tolerance)
    assert actual.dtype == (np.float16 if dim == 128 else np.float32)
    np.testing.assert_array_equal(actual,score_encoded(q,k,chunk_size=64))
    streamed = np.concatenate([chunk for _,chunk in score_chunks(q,k,chunk_size=9)],axis=-1)
    np.testing.assert_array_equal(actual,streamed)
    np.testing.assert_array_equal(gqa_mapping(4,2),[0,0,1,1])


def test_fma_is_single_rounding():
    a,b,c = np.float32(1+2**-23),np.float32(1-2**-23),np.float32(-1)
    assert np.float32(a*b+c) == 0
    assert fma32(a,b,c) == np.float32(-2**-46)


def test_direct_chunk_score_to_global_selector():
    rng = np.random.default_rng(171)
    q = encode_query(rng.normal(size=(2,4,128)).astype(np.float16))
    k = encode_keys(rng.normal(size=(2,73,128)).astype(np.float16))
    score = score_encoded(q,k)
    for budget in [0,1,17,73]:
        stream = run_streaming_near_memory(q,k,budget,chunk_size=11)
        for b in range(2):
            for h in range(4):
                np.testing.assert_array_equal(stream[b,h],host_topk(score[b,h],budget))


@pytest.mark.parametrize('n,banks',[(1,4),(7,16),(37,8),(128,16)])
@pytest.mark.parametrize('mapping',['striped','contiguous'])
def test_selection_boundaries_ties_masks(n,banks,mapping):
    s = np.random.default_rng(7).integers(-2,3,n).astype(np.float16)
    for budget in sorted(set([0,1,min(n,3),n])):
        a = run_retrieval(s,budget,'host',banks,mapping)['indices']
        c = run_retrieval(s,budget,'near_memory',banks,mapping)['indices']
        np.testing.assert_array_equal(a,c)
        np.testing.assert_array_equal(a,threshold_topk(s,budget))
        for mode in ['bank_local','bank_adaptive']:
            selected = run_retrieval(s,budget,mode,banks,mapping)['indices']
            assert len(selected) == len(set(selected)) == budget
            if budget == n:
                assert accuracy(a,selected)['recall'] == 1
    valid = np.arange(n)%2 == 0
    np.testing.assert_array_equal(host_topk(s,int(valid.sum()),valid),
                                  near_memory_topk(s,int(valid.sum()),valid))
    assert len(host_topk(np.empty(0,dtype=np.float16),0)) == 0
    np.testing.assert_array_equal(host_topk(np.zeros(n,dtype=np.float16),n),np.arange(n))


def test_concentration_and_adaptive_is_not_assumed_exact():
    n,banks,budget = 128,8,16
    ids = map_banks(n,banks)
    s = (np.arange(n)/1000).astype(np.float16)
    s[ids == 0] += 10
    ref = host_topk(s,budget)
    fixed, _ = bank_local_topk(s,budget,ids,banks)
    adaptive, details = bank_local_topk(s,budget,ids,banks,True)
    assert accuracy(ref,fixed)['recall'] == 2/16
    assert accuracy(ref,adaptive)['recall'] >= accuracy(ref,fixed)['recall']
    assert details['control_bytes'] > 0
    assert details['synchronization_rounds'] == 2
    # Unobserved sharp outliers: surveys are no exact-global guarantee.
    s = np.zeros(64,dtype=np.float16)
    s[[1,9,17,25]] = 10
    ids = map_banks(64,4,'contiguous')
    got,_ = bank_local_topk(s,4,ids,4,True,pilot_per_bank=1)
    assert accuracy(host_topk(s,4),got)['recall'] < 1


def test_current_token_and_masks():
    s = np.array([9,8,7,-10],dtype=np.float16)
    for mode in ['host','near_memory','bank_local','bank_adaptive']:
        np.testing.assert_array_equal(run_retrieval(s,1,mode,reserve_current=True)['indices'],[3])
        result = run_retrieval(s,2,mode,valid=[True,False,True,True],reserve_current=True)
        assert result['indices'][-1] == 3
        assert 1 not in result['indices']


def test_invalid_inputs():
    for score,k in [(np.array([np.nan]),1),(np.zeros(2),3),(np.zeros(2),-1)]:
        with pytest.raises(ValueError): host_topk(score,k)
        with pytest.raises(ValueError): near_memory_topk(score,k)
    for q,k in [(3,2),(0,1),(4,0)]:
        with pytest.raises(ValueError): gqa_mapping(q,k)
    with pytest.raises(ValueError): encode_query(np.zeros((1,1,0),dtype=np.float32))
    with pytest.raises(ValueError): map_banks(2,0)
    with pytest.raises(ValueError): allocate_quotas(4,[1,2])
    with pytest.raises(ValueError): run_retrieval(np.ones(3),1,valid=[False])
    sel = StreamingTopK(1)
    sel.feed([1.],[0])
    with pytest.raises(ValueError): sel.feed([2.],[0])
    with pytest.raises(ValueError): StreamingTopK(1).result()


def test_cost_accounting_and_resource_sensitivity():
    a = estimate(4096,512,32)
    c = estimate(4096,512,32,mode='near_memory')
    assert a['memory_to_host_bytes'] == 4096*4*2
    assert c['memory_to_host_bytes'] == 512*4*4
    assert c['bank_to_near_bytes'] == a['memory_to_host_bytes']
    assert c['candidate_buffer_logical_bytes'] == 512*4*6
    assert a['selector_compare_lanes'] == 0
    assert a['total_area_um2'] is None and a['total_energy_pj'] is None
    assert estimate(4096,512,64)['pim_compute_ns'] < a['pim_compute_ns']
    assert estimate(4096,512,32,hw=HardwareConfig(fp_units=8))['pim_compute_ns'] < a['pim_compute_ns']
    assert estimate(4096,512,32,hw=HardwareConfig(pim_mhz=250))['pim_compute_ns'] > a['pim_compute_ns']
