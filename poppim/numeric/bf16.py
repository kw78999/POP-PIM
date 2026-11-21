"""CPU IEEE-like RNE BF16 emulator; arithmetic uses exact Python integers.

BF16 is carried as uint16 bits, or exactly widened FP32 values. No native BF16,
DPI, GPU, Torch, or host floating-point contraction is required. NaNs canonicalize.
"""
import numpy as np


def bf16_bits_from_fp32(x):
    """One RNE conversion from binary32, gradual subnormals, quiet canonical NaN."""
    x = np.asarray(x, dtype=np.float32)
    u = x.view(np.uint32)
    wide = u.astype(np.uint64)
    rounded = ((wide + np.uint64(0x7fff) + ((wide >> np.uint64(16)) & np.uint64(1))) >> np.uint64(16)).astype(np.uint16)
    nan = ((u & 0x7f800000) == 0x7f800000) & ((u & 0x007fffff) != 0)
    return np.where(nan, np.uint16(0x7fc0), rounded).astype(np.uint16)


def widen(bits):
    u = np.asarray(bits, dtype=np.uint16).astype(np.uint32) << np.uint32(16)
    return u.view(np.float32)


def bf16(x):
    return widen(bf16_bits_from_fp32(x))


def _decode(u):
    u = int(u)
    s, exp, frac = u >> 15, (u >> 7) & 255, u & 127
    return s, (128 + frac if exp else frac), (exp - 134 if exp else -133)


def _round_shift(n, shift):
    if shift <= 0:
        return n << -shift
    q, r = divmod(n, 1 << shift)
    half = 1 << (shift - 1)
    return q + int(r > half or (r == half and q & 1))


def round_exact(n, exponent, zero_sign=0):
    """Round exact signed integer * 2**exponent directly to BF16 once."""
    if n == 0:
        return zero_sign << 15
    sign = int(n < 0)
    n = abs(n)
    k = n.bit_length() - 1 + exponent
    if k < -126:
        significand = _round_shift(n, -133 - exponent)
        return (sign << 15) | significand  # 128 naturally encodes min normal.
    significand = _round_shift(n, k - 7 - exponent)
    if significand == 256:
        significand >>= 1
        k += 1
    if k > 127:
        return (sign << 15) | 0x7f80
    return (sign << 15) | ((k + 127) << 7) | (significand - 128)


def _nan(u):
    return (u & 0x7f80) == 0x7f80 and (u & 127) != 0


def mul_bits(a, b):
    a, b = int(a), int(b)
    if _nan(a) or _nan(b):
        return 0x7fc0
    sa, ma, ea = _decode(a)
    sb, mb, eb = _decode(b)
    if (a & 0x7fff) == 0x7f80 or (b & 0x7fff) == 0x7f80:
        if (a & 0x7fff) == 0 or (b & 0x7fff) == 0:
            return 0x7fc0
        return ((sa ^ sb) << 15) | 0x7f80
    return round_exact((-1 if sa ^ sb else 1) * ma * mb, ea + eb, sa ^ sb)


def add_bits(a, b):
    a, b = int(a), int(b)
    if _nan(a) or _nan(b):
        return 0x7fc0
    if (a & 0x7fff) == 0x7f80 or (b & 0x7fff) == 0x7f80:
        if (a & 0x7fff) == (b & 0x7fff) == 0x7f80 and (a ^ b) & 0x8000:
            return 0x7fc0
        return a if (a & 0x7fff) == 0x7f80 else b
    sa, ma, ea = _decode(a)
    sb, mb, eb = _decode(b)
    exponent = min(ea, eb)
    n = ((-ma if sa else ma) << (ea - exponent)) + ((-mb if sb else mb) << (eb - exponent))
    return round_exact(n, exponent, int(sa and sb))


def fma_bits(a, b, c):
    """True fused BF16 FMA: exact a*b+c, one BF16 RNE; no FP32 bridge."""
    a, b, c = int(a), int(b), int(c)
    if _nan(a) or _nan(b) or _nan(c):
        return 0x7fc0
    sa, ma, ea = _decode(a)
    sb, mb, eb = _decode(b)
    sc, mc, ec = _decode(c)
    ai, bi, ci = [(v & 0x7fff) == 0x7f80 for v in (a, b, c)]
    sp = sa ^ sb
    if ai or bi:
        if (a & 0x7fff) == 0 or (b & 0x7fff) == 0 or (ci and sp != sc):
            return 0x7fc0
        return (sp << 15) | 0x7f80
    if ci:
        return c
    exponent = min(ea + eb, ec)
    n = ((-ma * mb if sp else ma * mb) << (ea + eb - exponent)) + ((-mc if sc else mc) << (ec - exponent))
    return round_exact(n, exponent, int(sp and sc))


def apply_bits(op, *args):
    arrays = np.broadcast_arrays(*(np.asarray(x, dtype=np.uint16) for x in args))
    data = np.fromiter((op(*xs) for xs in zip(*(a.flat for a in arrays))), dtype=np.uint16, count=arrays[0].size)
    return data.reshape(arrays[0].shape)


def mul(a, b):
    return widen(apply_bits(mul_bits, bf16_bits_from_fp32(a), bf16_bits_from_fp32(b)))


def add(a, b):
    return widen(apply_bits(add_bits, bf16_bits_from_fp32(a), bf16_bits_from_fp32(b)))


def fma(a, b, c):
    return widen(apply_bits(fma_bits, bf16_bits_from_fp32(a), bf16_bits_from_fp32(b), bf16_bits_from_fp32(c)))
