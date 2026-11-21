"""Byte-frugal selection variants of the POP-PIM score replay.

GQA-shared selection (GS)
    The four query heads that share a KV head (GQA4) merge their BF16 approximate logits as
    ``((s0 + s1) + (s2 + s3)) * 0.25``, every operation rounded to BF16 (RNE), and one Top-K is
    chosen per KV head. Per-head exact QK/softmax is unchanged. The GS unit RTL
    (``rtl/gs/gs_shared_score.sv``) is checked bit-exactly against :func:`gs_aggregate`.

Mass-guided V skipping (VS)
    After exact QK over the selected keys, a V row is read only if some head of the KV group needs
    the token to keep at least ``1 - delta`` of its selected attention mass. The masks depend on the
    exact logits of the activation snapshot, so they are stored (bit-packed) under
    ``data/replay/variants``; :func:`unpack_mask` restores the simulator layout.

The replay is one Llama-3.1-8B decode step (32 query heads, 8 KV heads) at N = 32,768.
"""
from __future__ import annotations

import numpy as np

QUERY_HEADS = 32
KV_HEADS = 8
GROUP = QUERY_HEADS // KV_HEADS


def bf16_to_f32(bits):
    """Widen BF16 bit patterns to float32 values."""
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(values):
    """Round float32 values to BF16 bit patterns (round to nearest, ties to even; finite inputs)."""
    u = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32).astype(np.uint64)
    rounding = ((u >> 16) & 1) + 0x7FFF
    return ((u + rounding) >> 16).astype(np.uint16)


def _round(values):
    return bf16_to_f32(f32_to_bf16(values))


def gs_aggregate(score_bits):
    """Return the GQA-shared BF16 score bits, shape ``[KV_HEADS, N]``, from per-head bits ``[32, N]``."""
    scores = bf16_to_f32(score_bits)
    if scores.ndim != 2 or scores.shape[0] != QUERY_HEADS:
        raise ValueError('expected per-query-head scores of shape [32, N]')
    f = scores.reshape(KV_HEADS, GROUP, -1)
    s01 = _round(f[:, 0] + f[:, 1])
    s23 = _round(f[:, 2] + f[:, 3])
    total = _round(s01 + s23)
    return f32_to_bf16(total * np.float32(0.25))


def stable_topk(values, k):
    """Top-``k`` per row (score descending, index ascending, +0 == -0); each row returned in index order."""
    values = np.asarray(values, dtype=np.float32)
    rows, n = values.shape
    if not 1 <= k <= n:
        raise ValueError('k must be in [1, N]')
    ids = np.arange(n)
    out = np.empty((rows, k), dtype=np.int32)
    for row in range(rows):
        order = np.lexsort((ids, -(values[row] + np.float32(0.0))))[:k]
        out[row] = np.sort(order)
    return out


def replay_variant(score_bits, budget, shared):
    """Per-head replay inputs ``(scores [32, N] uint16, expected [32, budget] int32)`` for one variant.

    ``shared=False`` keeps each head's own scores and Top-k. ``shared=True`` gives every head of a
    KV group the aggregated score and the group's single Top-k list.
    """
    score_bits = np.asarray(score_bits, dtype=np.uint16)
    if not shared:
        return score_bits, stable_topk(bf16_to_f32(score_bits), budget)
    shared_bits = gs_aggregate(score_bits)
    shared_idx = stable_topk(bf16_to_f32(shared_bits), budget)
    return np.repeat(shared_bits, GROUP, axis=0), np.repeat(shared_idx, GROUP, axis=0)


def union_tokens(expected):
    """Unique selected tokens per KV head (the K/V rows a per-group gather must read)."""
    expected = np.asarray(expected)
    return [int(np.unique(expected[GROUP * g:GROUP * (g + 1)]).size) for g in range(KV_HEADS)]


def unpack_mask(packed, n):
    """Restore a ``[KV_HEADS, n]`` uint8 V-keep mask from its ``numpy.packbits`` form."""
    bits = np.unpackbits(np.frombuffer(packed, dtype=np.uint8))
    if bits.size != KV_HEADS * n:
        raise ValueError('mask size does not match the replay shape')
    return bits.reshape(KV_HEADS, n)


def vs_scan_cycles(bstar):
    """VS engine threshold-scan cycles before the V phase can start.

    One engine per KV head scans its four heads' 256-bin histograms from the top bin down to the
    chosen bin ``b*``; the RTL takes ``1069 + sum(255 - b*)`` cycles per group (checked on all 16
    groups of the GS variants). Engines run in parallel, so the slowest group sets the delay.
    """
    if len(bstar) != QUERY_HEADS:
        raise ValueError('expected one b* per query head')
    return max(1069 + sum(255 - int(b) for b in bstar[GROUP * g:GROUP * (g + 1)]) for g in range(KV_HEADS))
