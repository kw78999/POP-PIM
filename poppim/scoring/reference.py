"""Packed-popcount PopKV scoring with actual libm fmaf (round-to-nearest).

This reproduces the source arithmetic, not an assertion of CUDA bit identity:
CUDA --use_fast_math division/rsqrt and compiler contractions remain unverified.
"""
import ctypes
import ctypes.util
import numpy as np
from poppim.encoding.bitpacking import unpack_bits

_libm = ctypes.CDLL(ctypes.util.find_library('m'))
_fmaf = _libm.fmaf
_fmaf.argtypes = [ctypes.c_float] * 3
_fmaf.restype = ctypes.c_float
_fma_ufunc = np.frompyfunc(_fmaf, 3, 1)
_LUT = np.array([bin(i).count('1') for i in range(256)], dtype=np.int32)


def fma32(a, b, c):
    return np.asarray(_fma_ufunc(a, b, c), dtype=np.float32)


def popcount32(words):
    w = np.ascontiguousarray(words, dtype=np.uint32)
    return _LUT[w.view(np.uint8).reshape(*w.shape, 4)].sum(-1, dtype=np.int32)


def gqa_mapping(query_heads, kv_heads):
    if kv_heads <= 0 or query_heads <= 0 or query_heads % kv_heads:
        raise ValueError('query heads must be a positive multiple of KV heads')
    return np.arange(query_heads, dtype=np.int64) // (query_heads // kv_heads)


def _mapping(q, k, mapping):
    if q.dim != k.dim or q.group != k.group:
        raise ValueError('Q/K dimensions and representative groups must match')
    mapping = gqa_mapping(q.sign.shape[1], k.sign.shape[0]) if mapping is None else np.asarray(mapping)
    if (mapping.shape != (q.sign.shape[1],) or mapping.dtype.kind not in 'iu'
            or np.any(mapping < 0) or np.any(mapping >= k.sign.shape[0])):
        raise ValueError('invalid GQA head mapping')
    return mapping


def score_encoded(q, k, head_to_kv=None, chunk_size=1024):
    """[batch,query heads,tokens]; D128 emits FP16, other D emits FP32."""
    mapping = _mapping(q, k, head_to_kv)
    if chunk_size < 1:
        raise ValueError('chunk_size must be positive')
    n = k.sign.shape[1]
    dtype = np.float16 if q.dim == 128 else np.float32
    out = np.empty((*q.sign.shape[:2], n), dtype=dtype)
    for start in range(0, n, chunk_size):
        end = min(n, start+chunk_size)
        for b in range(q.sign.shape[0]):
            for h, kv in enumerate(mapping):
                total = np.zeros(end-start, dtype=np.float32)
                for rep, rs in enumerate(range(0, q.dim, q.group)):
                    re = min(q.dim, rs+q.group)
                    counts = [np.zeros(end-start, dtype=np.int32) for _ in range(4)]
                    for word in range(rs//32, (re-1)//32+1):
                        begin = max(rs, word*32)-word*32
                        finish = min(re, word*32+32)-word*32
                        mask = np.uint32(((1 << (finish-begin))-1) << begin)
                        qs, qm = q.sign[b,h,word], q.magnitude[b,h,word] & mask
                        ks = k.sign[kv,start:end,word]
                        km = k.magnitude[kv,start:end,word] & mask
                        match = ~(qs ^ ks) & mask
                        for j, selected in enumerate((np.full(end-start, mask),
                                                      np.full(end-start, qm), km, qm & km)):
                            counts[j] += 2*popcount32(match & selected)-popcount32(selected)
                    cs, cq, ck, cqk = (x.astype(np.float32) for x in counts)
                    ql, dq = q.low[b,h,rep], q.delta[b,h,rep]
                    kl = k.low[kv,start:end,rep].astype(np.float32)
                    dk = k.delta[kv,start:end,rep].astype(np.float32)
                    # CUDA nested fmaf; coefficient products round to FP32 first.
                    v = ((dq*dk)*cqk).astype(np.float32)
                    v = fma32(ql*dk, ck, v)
                    v = fma32(dq*kl, cq, v)
                    v = fma32(ql*kl, cs, v)
                    total += v
                out[b,h,start:end] = total*np.float32(1/np.sqrt(q.dim))
    if not np.isfinite(out).all():
        raise ValueError('score storage overflow or non-finite result')
    return out


def dense_encoded_oracle(q, k, head_to_kv=None, chunk_size=1024):
    """Independent FP64 dense reconstruction of *same encoded metadata*.

    Not dense original QK, nor CUDA validation. O(chunk_size*D) workspace.
    """
    mapping = _mapping(q, k, head_to_kv)
    def reconstruct(x):
        s = np.where(unpack_bits(x.sign, x.dim), 1., -1.)
        m = unpack_bits(x.magnitude, x.dim)
        low = np.repeat(x.low.astype(np.float64), x.group, -1)[...,:x.dim]
        delta = np.repeat(x.delta.astype(np.float64), x.group, -1)[...,:x.dim]
        return s*(low + m*delta)
    qhat = reconstruct(q)
    out = np.empty((*q.sign.shape[:2], k.sign.shape[1]), dtype=np.float64)
    for start in range(0, k.sign.shape[1], chunk_size):
        khat = reconstruct(k.slice_tokens(start, start+chunk_size))
        for h, kv in enumerate(mapping):
            out[:,h,start:start+chunk_size] = qhat[:,h] @ khat[kv].T / np.sqrt(q.dim)
    return out
