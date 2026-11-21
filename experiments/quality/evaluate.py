"""Quality of POP-PIM selection on CPU: Llama-3.1-8B, PG19, 32K context, teacher-forced decoding.

Usage (see experiments/quality/README.md):
    python experiments/quality/evaluate.py --pg19-dir /path/to/pg19/test --out outputs/quality


Methods (layers 0,1 dense; every other layer sparse; current token always attended):
  full                      dense attention
  ph:K                      POP-PIM baseline: per-query-head PopKV(A2) Top-(K-1) history + current; KV gather = union of 4 heads
  gs:K                      group-shared: mean of the 4 heads' approximate logits per KV head, one Top-(K-1) per KV head
  gs:K:vs:D                 + two-phase K->V gather; V read only for tokens needed to keep >= 1-D of each head's
                            selected-set mass (union over the group); dropped tokens are removed and renormalized

The 'full' run also measures, at every step and every sparse layer, attention-mass recall, relative output
error versus dense attention, unique KV tokens and kept-V tokens for a grid of selection policies (metric mode).
Scores use the FP32 PopKV reconstruction (Gtok=64, Gdim=64); hardware BF16 rounding is validated elsewhere.
"""
import argparse, json, math, sys, time, csv
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from popenc import encode_keys, encode_query, recon

HQ, HKV, D, GT = 32, 8, 128, 64
H2KV = torch.arange(HQ) // 4


def parse(m):
    p = m.split(':')
    if p[0] == 'full': return dict(kind='full', name=m)
    d = dict(kind=p[0], K=int(p[1]), name=m, vs=None)
    if len(p) > 2 and p[2] == 'vs': d['vs'] = float(p[3])
    return d


METRIC_GRID = (['ph:1024', 'ph:2048', 'ph:4096'] +
               [f'gs:{k}' for k in (2048, 4096, 6144, 8192)] +
               [f'gs:{k}:vs:{d}' for k in (4096, 6144) for d in (0.002, 0.005, 0.01)] +
               ['ph:4096:vs:0.002', 'oracle:4096'])


def load_stream(tok, need, pg19_dir):
    stream = []
    files = sorted(Path(pg19_dir).glob('*.txt'), key=lambda x: x.name)
    if not files:
        raise SystemExit(f'no *.txt files in {pg19_dir}')
    for f in files:
        stream.extend(tok(f.read_text(encoding='utf-8', errors='ignore'), add_special_tokens=False)['input_ids'])
        stream.append(tok.eos_token_id)
        if len(stream) >= need: break
    return stream


class KCache:
    """Per-layer BF16 K/V plus FP32 PopKV-reconstructed keys (incrementally re-encoded tail group)."""
    def __init__(self, k, v, cap):
        n = k.shape[1]; self.n = n
        self.k = torch.zeros(HKV, cap, D, dtype=torch.bfloat16); self.k[:, :n] = k
        self.v = torch.zeros(HKV, cap, D, dtype=torch.bfloat16); self.v[:, :n] = v
        self.kh = torch.zeros(HKV, cap, D)
        self.kh[:, :n] = recon(*encode_keys(k.float()))

    def mark(self):
        # Memory-frugal reset point (container limit 32 GB): only the partial tail group's encoding changes on append.
        self.n0 = self.n; self.g0 = (self.n // GT) * GT; self.tail = self.kh[:, self.g0:self.n].clone(); return self

    def reset(self):
        self.n = self.n0; self.kh[:, self.g0:self.n0] = self.tail; return self

    def append(self, k, v):
        j = self.n; self.k[:, j] = k; self.v[:, j] = v; self.n += 1
        g0 = (j // GT) * GT
        self.kh[:, g0:self.n] = recon(*encode_keys(self.k[:, g0:self.n].float()))


def gdot(q, K):
    """q [HQ,D], K [HKV,n,D] (any dtype) -> [HQ,n] fp32 dot products without expanding K per query head."""
    return torch.einsum('gjd,gnd->gjn', q.view(HKV, 4, D), K.float()).reshape(HQ, -1)


def gmix(w, V):
    """w [HQ,n], V [HKV,n,D] -> [HQ,D]"""
    return torch.einsum('gjn,gnd->gjd', w.view(HKV, 4, -1), V.float()).reshape(HQ, D)


def select(m, s_hist, budget_hist):
    """Return per-query-head index tensor [HQ, k] over history tokens."""
    budget_hist = min(budget_hist, s_hist.shape[-1])
    if m['kind'] == 'ph':
        return s_hist.topk(budget_hist, -1).indices
    if m['kind'] == 'gs':
        agg = s_hist.view(HKV, 4, -1).mean(1)
        return agg.topk(budget_hist, -1).indices[H2KV]
    raise ValueError(m)


def sparse_attend(q, K, V, idx, n_hist, vs=None, ex_full=None):
    """q [HQ,D] fp32; K,V [HKV,n,D] bf16 (n = n_hist+1 incl. current); idx [HQ,k] history indices.
    Returns output [HQ,D], unique tokens (sum over KV heads), kept-V tokens (sum over KV heads)."""
    cur = torch.full((HQ, 1), n_hist, dtype=idx.dtype)
    idx = torch.cat([idx, cur], 1)
    Kg = K[H2KV[:, None], idx].float(); Vg = V[H2KV[:, None], idx].float()   # [HQ,k,D]
    logit = torch.einsum('hd,hkd->hk', q, Kg) / math.sqrt(D)
    w = torch.softmax(logit, -1)
    uniq = sum(int(torch.unique(idx[4 * g:4 * g + 4]).numel()) for g in range(HKV))
    kept = uniq
    if vs is not None:
        keep = torch.zeros_like(w, dtype=torch.bool)
        sw, si = w.sort(-1, descending=True); c = sw.cumsum(-1)
        nkeep = (c < 1 - vs).sum(-1) + 1
        rank_keep = torch.arange(w.shape[1])[None, :] < nkeep[:, None]
        keep.scatter_(1, si, rank_keep)
        kept = 0
        for g in range(HKV):          # V read once per KV head: union over its 4 heads
            gi = idx[4 * g:4 * g + 4]; gk = keep[4 * g:4 * g + 4]
            u = torch.unique(gi[gk]); kept += int(u.numel())
            member = torch.isin(gi, u)  # any head's need makes the V row available to all heads in the group
            keep[4 * g:4 * g + 4] = member
        w = w * keep; w = w / w.sum(-1, keepdim=True)
    out = torch.einsum('hk,hkd->hd', w, Vg)
    return out, uniq, kept, idx


def metric_mode(q, cache, rows, layer, step):
    n = cache.n; n_hist = n - 1
    K, V = cache.k[:, :n], cache.v[:, :n]
    ex = gdot(q, K) / math.sqrt(D)
    p = torch.softmax(ex, -1)
    full = gmix(p, V)
    qh = recon(*encode_query(q))
    s = gdot(qh, cache.kh[:, :n_hist]) / math.sqrt(D)
    fn = full.norm(dim=-1)
    for name in METRIC_GRID:
        m = parse(name)
        if m['kind'] == 'oracle':
            idx = ex[:, :n_hist].topk(min(m['K'] - 1, n_hist), -1).indices
        else:
            idx = select(m, s, m['K'] - 1)
        out, uniq, kept, fidx = sparse_attend(q, K, V, idx, n_hist, m['vs'])
        err = ((out - full).norm(dim=-1) / fn)
        mass = p.gather(1, fidx).sum(-1)
        rows.append(dict(step=step, layer=layer, method=name, err_mean=err.mean().item(), err_max=err.max().item(),
                         mass_mean=mass.mean().item(), mass_min=mass.min().item(), unique_kv=uniq, kept_v=kept, n=n))


def decode_step(model, caches, token, pos, m, apply_rope, metric_rows=None, step=0):
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    mm = model.model
    hidden = mm.embed_tokens(torch.tensor([[token]]))
    cos, sin = mm.rotary_emb(hidden, torch.tensor([[pos]]))
    stats = dict(uniq=0, kept=0, layers=0)
    for li, layer in enumerate(mm.layers):
        at = layer.self_attn
        h = layer.input_layernorm(hidden)
        q = at.q_proj(h).view(1, 1, HQ, D).transpose(1, 2)
        k = at.k_proj(h).view(1, 1, HKV, D).transpose(1, 2)
        v = at.v_proj(h).view(1, 1, HKV, D).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        c = caches[li]; c.append(k[0, :, 0], v[0, :, 0])
        qf = q[0, :, 0].float()
        n = c.n
        if m['kind'] == 'full' or li < 2:
            w = torch.softmax(gdot(qf, c.k[:, :n]) / math.sqrt(D), -1)
            out = gmix(w, c.v[:, :n])
            if metric_rows is not None and li >= 2: metric_mode(qf, c, metric_rows, li, step)
        else:
            qh = recon(*encode_query(qf))
            s = gdot(qh, c.kh[:, :n - 1]) / math.sqrt(D)
            idx = select(m, s, m['K'] - 1)
            out, uniq, kept, _ = sparse_attend(qf, c.k[:, :n], c.v[:, :n], idx, n - 1, m['vs'])
            stats['uniq'] += uniq; stats['kept'] += kept; stats['layers'] += 1
        hidden = hidden + at.o_proj(out.reshape(1, 1, HQ * D).to(hidden.dtype))
        hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
    logits = model.lm_head(mm.norm(hidden)).float()[0, -1]
    return logits, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--context', type=int, default=32768)
    ap.add_argument('--tokens', type=int, default=64)
    ap.add_argument('--blocks', type=str, default='0,1,2,3')
    ap.add_argument('--methods', default='full,ph:4096,gs:4096,gs:6144,gs:4096:vs:0.005,gs:6144:vs:0.002')
    ap.add_argument('--metric-every', type=int, default=32, help='metric mode every E-th step of the full run')
    ap.add_argument('--out', required=True)
    ap.add_argument('--chunk', type=int, default=4096)
    ap.add_argument('--pg19-dir', required=True, help='directory of PG19 test-split books as *.txt')
    ap.add_argument('--model', default='meta-llama/Llama-3.1-8B', help='Hugging Face id or local path')
    ap.add_argument('--threads', type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = a.model
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation='sdpa').eval()
    blocks = [int(x) for x in a.blocks.split(',')]
    size = a.context + a.tokens
    stream = load_stream(tok, (max(blocks) + 1) * size, a.pg19_dir)
    methods = [parse(x) for x in a.methods.split(',')]
    (out / 'config.json').write_text(json.dumps(vars(a) | dict(model=path, stream_tokens=len(stream), metric_grid=METRIC_GRID), indent=1))
    nll_f = open(out / 'nll.csv', 'a'); nll_w = csv.writer(nll_f)
    if nll_f.tell() == 0: nll_w.writerow(['block', 'step', 'method', 'nll', 'unique_kv_per_layer', 'kept_v_per_layer'])
    met_f = open(out / 'metrics.csv', 'a'); met_w = None
    torch.manual_seed(0)
    with torch.no_grad():
        for b in blocks:
            blk = stream[b * size:(b + 1) * size]
            done = set()
            if (out / 'nll.csv').exists():
                import collections
                cnt = collections.Counter((int(r[0]), r[2]) for r in csv.reader(open(out / 'nll.csv')) if r[0] != 'block')
                done = {k for k, v in cnt.items() if v >= a.tokens}
            todo = [m for m in methods if (b, m['name']) not in done]
            if not todo: continue
            prefix = torch.tensor([blk[:a.context - 1]])
            cur_ids = blk[a.context - 1:a.context - 1 + a.tokens]; tgt = blk[a.context:a.context + a.tokens]
            t0 = time.time()
            from transformers import DynamicCache
            pkv = DynamicCache()
            for s0 in range(0, prefix.shape[1], a.chunk):
                model(prefix[:, s0:s0 + a.chunk], past_key_values=pkv, use_cache=True, logits_to_keep=1)
            print(f'[block {b}] prefill {time.time() - t0:.0f}s', flush=True)
            base = []
            for li in range(len(model.model.layers)):
                lay = pkv.layers[li]
                base.append(KCache(lay.keys[0], lay.values[0], a.context + a.tokens + 1).mark())
            del pkv
            for m in todo:
                caches = [c.reset() for c in base]; t0 = time.time(); nlls = []
                for t, (x, y) in enumerate(zip(cur_ids, tgt)):
                    rows = [] if (m['kind'] == 'full' and t % a.metric_every == 0) else None
                    logits, st = decode_step(model, caches, x, a.context - 1 + t, m, None, rows, t)
                    nll = (torch.logsumexp(logits, -1) - logits[y]).item(); nlls.append(nll)
                    L = max(st['layers'], 1)
                    nll_w.writerow([b, t, m['name'], nll, st['uniq'] / L, st['kept'] / L]); nll_f.flush()
                    if rows:
                        for r in rows: r['block'] = b
                        if met_w is None:
                            met_w = csv.DictWriter(met_f, fieldnames=list(rows[0]))
                            if met_f.tell() == 0: met_w.writeheader()
                        met_w.writerows(rows); met_f.flush()
                print(f'[block {b}] {m["name"]:22s} ppl {math.exp(sum(nlls) / len(nlls)):.4f} ({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
