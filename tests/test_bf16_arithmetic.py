"""Independent Fraction oracle checks, plus preservation/reference checks."""
from bisect import bisect_left
from fractions import Fraction
import numpy as np
from poppim.numeric.bf16 import (bf16_bits_from_fp32, widen, bf16, mul_bits,
                         add_bits, fma_bits)


def test_all_bf16_patterns_roundtrip():
    u = np.arange(65536, dtype=np.uint16)
    expected = u.copy()
    nan = ((u & 0x7f80) == 0x7f80) & ((u & 127) != 0)
    expected[nan] = 0x7fc0
    np.testing.assert_array_equal(bf16_bits_from_fp32(widen(u)), expected)


def test_rne_ties_and_signed_zero():
    a = np.array([1 + 1/256, 1 + 3/256, -1 - 1/256,
                  2.**-134, 3*2.**-134, 0., -0.], dtype=np.float32)
    np.testing.assert_array_equal(bf16_bits_from_fp32(a), [0x3f80, 0x3f82, 0xbf80, 0, 2, 0, 0x8000])
    assert mul_bits(0x8000, 0x3f80) == 0x8000
    assert add_bits(0x8000, 0x8000) == 0x8000
    assert add_bits(0x8000, 0) == 0
    assert fma_bits(0x8000, 0x3f80, 0x8000) == 0x8000


def test_fused_and_separate_differ():
    a,b,c = map(int, bf16_bits_from_fp32([1+1/128, 1-1/128, -1]))
    assert add_bits(mul_bits(a,b),c) == 0
    assert float(widen(fma_bits(a,b,c))) == -2.**-14


def test_no_fp32_double_rounding():
    # Exact product lies at a BF16 midpoint; the tiny c breaks the tie.
    a,b,c = map(int, bf16_bits_from_fp32([1.5, 1.0234375, 2.**-30]))
    assert float(widen(fma_bits(a,b,c))) == 1.5390625
    assert float(bf16(np.float32(1.5*1.0234375 + 2.**-30))) == 1.53125


def test_final_fp16_ties_use_smaller_token_index():
    from poppim.topk.host_topk import host_topk
    np.testing.assert_array_equal(host_topk(np.array([1,1,2,1],dtype=np.float16),4),[2,0,1,3])


def test_special_values():
    assert mul_bits(0,0x7f80) == 0x7fc0
    assert add_bits(0x7f80,0xff80) == 0x7fc0
    assert fma_bits(0x7f80,0x3f80,0xff80) == 0x7fc0
    assert mul_bits(0x7f7f,0x4000) == 0x7f80
    assert mul_bits(0x0001,0x3f00) == 0
    assert mul_bits(0x0003,0x3f00) == 2
    assert fma_bits(0,0x7f80,0x3f80) == 0x7fc0


def test_against_independent_fraction_oracle():
    # All positive finite BF16 numbers; compare exact rational distances.
    grid = [Fraction(float(v)) for v in widen(np.arange(0x7f80,dtype=np.uint16))]
    overflow = grid[-1] + Fraction(2)**119
    def rounded(value):
        if not value:
            return 0
        sign = 0x8000 if value < 0 else 0
        value = abs(value)
        if value >= overflow:
            return sign | 0x7f80
        if value > grid[-1]:
            return sign | 0x7f7f
        j = bisect_left(grid,value)
        if j == 0:
            return sign
        lo,hi = value-grid[j-1],grid[j]-value
        chosen = j-1 if lo<hi or (lo==hi and (j-1)%2==0) else j
        return sign | chosen
    rng = np.random.default_rng(8741)
    u = rng.integers(0,65536,(2400,3),dtype=np.uint16)
    u = u[np.all((u & 0x7f80)!=0x7f80,axis=1)]
    for a,b,c in u:
        af,bf,cf = (Fraction(float(widen(x))) for x in (a,b,c))
        for got,exact in [(mul_bits(a,b),af*bf),(add_bits(a,b),af+bf),
                          (fma_bits(a,b,c),af*bf+cf)]:
            expected = rounded(exact)
            # Exact cancellation is +0; product with signed zero may be -0.
            assert got==expected or (got & 0x7fff)==expected==0
