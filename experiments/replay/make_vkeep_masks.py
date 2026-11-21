"""Regenerate the V-keep masks and VS RTL test vectors from the activation snapshot.

The committed masks (data/replay/variants) and test vectors (data/rtl_vectors/vs) depend on the exact QK
logits of one Llama-3.1-8B layer-2 decode step at 32K PG19 context. This script recomputes them from that
snapshot (a torch file with ``q`` [1, 32, 128], ``k`` and ``v`` [8, 32768, 128]) and checks every byte
against the committed hashes. It needs torch and the replay inputs from ``python -m tools.prepare_inputs``.

    python experiments/replay/make_vkeep_masks.py --activation latency_activation.pt --output-dir outputs/vkeep

Exact rule: per query head, the smallest score-ordered prefix of its selected tokens whose softmax mass is at
least 1 - delta. Hardware rule (rtl/vs/vs_mass_threshold.sv): 256 bins of 1/8 nat below an anchor, a
fixed-point mass histogram and a conservative FP32 threshold scan; bit-exact here through exact rational
FP32 rounding. In both, a V row of a KV group is read if any of its four heads keeps the token.
"""
import argparse
import hashlib
import json
import math
import struct
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.prepare_inputs import manifest, prepare  # noqa: E402

N, HQ, HKV, D, MARGIN = 32768, 32, 8, 128, 6.0
T = json.loads((ROOT / 'rtl/vs/vs_tables.json').read_text())
T1, T2 = T['T1'], T['T2']
E = [np.float32(x) for x in T['E']]
E0P = np.float32(T['E0P'])
MASKS = (('base', 0.002), ('gs4096', 0.005), ('gs6144', 0.002))


def f32(x):
    return np.float32(x)


def rne32(fr):
    """Round an exact Fraction to FP32 (round to nearest even)."""
    if fr == 0:
        return f32(0.0)
    f = float(fr)
    if Fraction(f) == fr:
        return f32(f)
    a = f32(f)
    candidates = [a, np.nextafter(a, f32(np.inf)), np.nextafter(a, f32(-np.inf))]
    return min(candidates, key=lambda c: (abs(Fraction(float(c)) - fr), int(np.float32(c).view(np.uint32)) & 1))


def fma32(a, b, c):
    return rne32(Fraction(float(a)) * Fraction(float(b)) + Fraction(float(c)))


def head_hist(logits, C):
    """Bin codes (uint8), histogram S[256], clipped-tail count and saturation flag for one head."""
    X = np.array([math.floor(float(fma32(x, f32(2.0 ** 19), C))) for x in logits], dtype=np.int64)
    tail, sat = X < 65536, X >= 2 ** 24
    codes = np.where(tail, 0, np.where(sat, 255, (X >> 16) & 255)).astype(np.uint8)
    S = np.zeros(256, dtype=np.int64)
    for x in X[~tail & ~sat]:
        f = int(x) & 0xffff
        S[(int(x) >> 16) & 255] += (T1[f >> 8] * T2[f & 255]) >> 16
    assert S.max() < 2 ** 30
    return codes, S, int(tail.sum()), bool(sat.any())


def bstar(S, c0, sat, KD):
    if sat:
        return 0
    acc = f32(0.0)
    for b in range(1, 256):
        acc = fma32(f32(int(S[b])), E[b], acc)
    threshold = fma32(fma32(f32(c0), E0P, acc), KD, f32(0.0))
    cum = f32(0.0)
    for b in range(255, 0, -1):
        cum = fma32(f32(int(S[b])), E[b], cum)
        if cum >= threshold:
            return b
    return 0


def bf16(u):
    return (u.astype(np.uint32) << 16).view(np.float32)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--activation', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir
    (out / 'vs_vectors').mkdir(parents=True, exist_ok=True)
    spec = manifest()
    inputs = prepare(quiet=True)['inputs']
    snapshot = torch.load(args.activation, map_location='cpu')
    q, k, v = snapshot['q'][0].double(), snapshot['k'].double(), snapshot['v'].double()
    h2kv = torch.arange(HQ) // 4
    logits = torch.einsum('hd,hnd->hn', q, k[h2kv]) / math.sqrt(D)
    logits32 = logits.numpy().astype(np.float32)          # backend exact-QK logits as FP32
    report = dict(masks={}, vectors={})
    for name, delta in MASKS:
        idx = np.fromfile(inputs[name]['expected'], dtype=np.int32).reshape(HQ, -1)
        scores = bf16(np.fromfile(inputs[name]['scores'], dtype=np.uint16).reshape(HQ, N))
        exact = np.zeros((HKV, N), dtype=np.uint8)
        for h in range(HQ):
            sel = torch.from_numpy(idx[h].astype(np.int64))
            w = torch.softmax(logits[h, sel], -1)
            sw, si = w.sort(descending=True)
            n = int((sw.cumsum(0) < 1 - delta).sum()) + 1
            exact[h // 4, sel[si[:n]].numpy()] = 1
        KD = f32(T['KD'][str(delta)])
        hw = np.zeros((HKV, N), dtype=np.uint8)
        for g in range(HKV):
            heads = range(4 * g, 4 * g + 4)
            Cs, b = [], []
            for h in heads:
                sel = idx[h]
                group = scores[4 * g:4 * g + 4][:, sel] if name != 'base' else scores[h, sel]
                C = f32((255 - 8 * (float(group.max()) + MARGIN)) * 65536)
                codes, S, c0, sat = head_hist(logits32[h, sel], C)
                b.append(bstar(S, c0, sat, KD))
                Cs.append(C)
                hw[g, sel[codes >= b[-1]]] = 1
            if name != 'base':
                sel = idx[4 * g]
                vector = (struct.pack('<4I', *[int(np.float32(c).view(np.uint32)) for c in Cs])
                          + struct.pack('<2I', int(KD.view(np.uint32)), int(E0P.view(np.uint32)))
                          + struct.pack('<I', len(sel))
                          + np.stack([logits32[h, sel] for h in heads], 1).astype('<f4').tobytes()
                          + struct.pack('<4I', *b) + hw[g, sel].astype(np.uint8).tobytes())
                file = f'{name}_kv{g}.bin'
                (out / 'vs_vectors' / file).write_bytes(vector)
                committed = ROOT / 'data/rtl_vectors/vs' / file
                report['vectors'][file] = committed.read_bytes() == vector
        assert (hw >= exact).all(), 'hardware rule must keep a superset of the exact rule'
        for rule, mask in (('exact', exact), ('hw', hw)):
            key = f'{name}_{delta}_{rule}'
            data = mask.tobytes()
            (out / f'vkeep_{key}.bin').write_bytes(data)
            report['masks'][key] = dict(kept_v_tokens=int(mask.sum()),
                                        identical_to_release=digest(data) == spec['vkeep'][key]['unpacked_sha256'])
        print(name, delta, {k: v for k, v in report['masks'].items() if k.startswith(name)}, flush=True)
    (out / 'report.json').write_text(json.dumps(report, indent=1) + '\n')
    ok = all(m['identical_to_release'] for m in report['masks'].values()) and all(report['vectors'].values())
    print('All masks and test vectors are identical to the release data.' if ok else
          'Some regenerated files differ from the release data; see report.json')
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
