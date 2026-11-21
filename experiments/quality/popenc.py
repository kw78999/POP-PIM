"""PopKV (A2, Gtok=64, Gdim=64) reference scoring in torch, used by the quality evaluation.

Matches PopKV popkv_reference_scores semantics (FP32 reconstruction); hardware BF16 DAG differs only
in rounding (validated separately in the release). Used for quality/byte studies on real activations.
"""
import math, torch

GT, GD = 64, 64

def encode_keys(k, gt=GT, gd=GD):
    """k [Hkv,N,D] -> sign(bool), mag(bool), low[Hkv,N,D/gd], delta[Hkv,N,D/gd] (per-token coefficients)."""
    k = k.float(); H, N, D = k.shape
    G = math.ceil(N / gt); pad = G * gt - N
    x = torch.nn.functional.pad(k, (0, 0, 0, pad), value=float('nan')) if pad else k
    g = x.view(H, G, gt, D)
    mn = torch.nan_to_num(g, nan=float('inf')).amin(2).repeat_interleave(gt, 1)[:, :N]
    mx = torch.nan_to_num(g, nan=-float('inf')).amax(2).repeat_interleave(gt, 1)[:, :N]
    sign = k >= 0
    mag = torch.where(sign, k > mx * .5, k < mn * .5)
    low, delta = reps(k, mag, gd)
    return sign, mag, low, delta

def reps(x, mag, gd):
    sh = (*x.shape[:-1], x.shape[-1] // gd, gd)
    a, m = x.abs().view(sh), mag.view(sh)
    lc, hc = (~m).sum(-1), m.sum(-1)
    lo = (a * ~m).sum(-1) / lc.clamp_min(1); hi = (a * m).sum(-1) / hc.clamp_min(1)
    lo2 = torch.where(lc > 0, lo, torch.where(hc > 0, hi, torch.zeros_like(lo)))
    hi2 = torch.where(hc > 0, hi, lo2)
    return lo2, hi2 - lo2

def encode_query(q, gd=GD):
    q = q.float()
    mn, mx = q.amin(-1, keepdim=True), q.amax(-1, keepdim=True)
    sign = q >= 0
    mag = torch.where(sign, q > mx * .5, q < mn * .5)
    low, delta = reps(q, mag, gd)
    return sign, mag, low, delta

def recon(sign, mag, low, delta, gd=GD):
    l = low.repeat_interleave(gd, -1); h = (low + delta).repeat_interleave(gd, -1)
    return torch.where(sign, 1.0, -1.0) * torch.where(mag, h, l)

def scores_from(qenc, kenc, head_to_kv, gd=GD):
    """q [Hq,D] enc, k [Hkv,N,D] enc -> [Hq,N] approximate scores."""
    qh = recon(*qenc, gd)                       # [Hq,D]
    kh = recon(*kenc, gd)                       # [Hkv,N,D]
    D = qh.shape[-1]
    return torch.einsum('hd,hnd->hn', qh, kh[head_to_kv]) / math.sqrt(D)

def sign_only_scores(qenc, kenc, head_to_kv):
    """Coarse score from sign planes only (16B/token): weighted sign agreement with query magnitudes.
    Uses the query's reconstructed vector (query-side info is free) and key sign only."""
    qh = recon(*qenc)                            # [Hq,D]
    ks = torch.where(kenc[0], 1.0, -1.0)         # [Hkv,N,D]
    return torch.einsum('hd,hnd->hn', qh, ks[head_to_kv])
