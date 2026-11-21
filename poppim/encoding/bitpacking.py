"""PopKV sign/magnitude metadata; Q FP32, K FP16 representatives.

Ported from PopKV 623349ca, popkv.py and csrc/popkv_ext.cu.
uint32 here has the same bits as upstream int32; no additional quantization.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Encoded:
    sign: np.ndarray
    magnitude: np.ndarray
    low: np.ndarray
    delta: np.ndarray
    dim: int
    group: int

    def slice_tokens(self, start, end):
        return Encoded(*(a[:, start:end] for a in
                         (self.sign, self.magnitude, self.low, self.delta)),
                       self.dim, self.group)


def pack_bits(bits):
    bits = np.asarray(bits, dtype=bool)
    words = (bits.shape[-1] + 31) // 32
    padded = np.pad(bits, [(0, 0)] * (bits.ndim - 1) +
                    [(0, words * 32 - bits.shape[-1])])
    weights = np.left_shift(np.uint32(1), np.arange(32, dtype=np.uint32))
    return (padded.reshape(*bits.shape[:-1], words, 32) * weights).sum(
        axis=-1, dtype=np.uint32)


def unpack_bits(words, dim):
    return ((words[..., None] >> np.arange(32, dtype=np.uint32)) & 1).reshape(
        *words.shape[:-1], -1)[..., :dim].astype(bool)


def _representatives(values, mag, group, warp):
    lows, deltas = [], []
    for start in range(0, values.shape[-1], group):
        v = np.abs(values[..., start:start + group])
        m = mag[..., start:start + group]
        sums, counts = [], []
        for mask in (~m, m):
            data = np.where(mask, v, np.float32(0))
            if warp:
                # CUDA warp shuffle-down; G64 pairs lane i and lane i+32 first.
                width = 16 if group == 16 else 32
                pad = group - data.shape[-1]
                data = np.pad(data, [(0, 0)] * (data.ndim-1) + [(0, pad)])
                lane = data[..., :width].copy()
                if group == 64:
                    lane += data[..., 32:]
                offset = width // 2
                while offset:
                    previous = lane.copy()
                    lane[..., :width-offset] = (previous[..., :width-offset] +
                                               previous[..., offset:])
                    offset //= 2
                sums.append(lane[..., 0])
            else:
                # Generic CUDA query prelude visits dimensions sequentially.
                total = np.zeros(data.shape[:-1], dtype=np.float32)
                for i in range(data.shape[-1]):
                    total += data[..., i]
                sums.append(total)
            counts.append(mask.sum(axis=-1))
        lc, hc = counts
        low = (sums[0] / np.maximum(lc, 1).astype(np.float32)).astype(np.float32)
        high = (sums[1] / np.maximum(hc, 1).astype(np.float32)).astype(np.float32)
        low = np.where(lc > 0, low, high)
        high = np.where(hc > 0, high, low)
        lows.append(low)
        deltas.append((high-low).astype(np.float32))
    return np.stack(lows, -1), np.stack(deltas, -1)


def _validate(values, group, kind):
    x = np.asarray(values)
    if (x.ndim != 3 or min(x.shape) <= 0 or group not in (16, 32, 64)
            or x.dtype not in (np.dtype('float16'), np.dtype('float32'))):
        raise ValueError(f'{kind} needs nonempty 3D FP16/FP32 input and G=16/32/64')
    if not np.isfinite(x).all():
        raise ValueError('non-finite encoding input')
    return x.astype(np.float32)


def encode_query(query, group=64):
    q = _validate(query, group, 'Q')
    sign = q >= 0
    mag = np.where(sign, q > q.max(-1, keepdims=True)*np.float32(.5),
                   q < q.min(-1, keepdims=True)*np.float32(.5))
    low, delta = _representatives(q, mag, group, warp=q.shape[-1] == 128)
    return Encoded(pack_bits(sign), pack_bits(mag), low, delta, q.shape[-1], group)


def encode_keys(keys, group=64, token_group=64):
    k = _validate(keys, group, 'K')
    if not 1 <= token_group <= 4096:
        raise ValueError('token_group must be in [1,4096]')
    if group != 32 and token_group not in (8, 16, 32, 64, 128):
        raise ValueError('upstream specialized G16/G64 requires specialized token_group')
    sign = k >= 0
    mag = np.empty_like(sign)
    for start in range(0, k.shape[1], token_group):
        chunk = k[:, start:start+token_group]
        mag[:, start:start+token_group] = np.where(
            sign[:, start:start+token_group],
            chunk > chunk.max(1, keepdims=True)*np.float32(.5),
            chunk < chunk.min(1, keepdims=True)*np.float32(.5))
    low, delta = _representatives(k, mag, group, warp=True)
    low, delta = low.astype(np.float16), delta.astype(np.float16)
    if not np.isfinite(low).all() or not np.isfinite(delta).all():
        raise ValueError('K representatives overflow FP16')
    return Encoded(pack_bits(sign), pack_bits(mag), low, delta, k.shape[-1], group)
