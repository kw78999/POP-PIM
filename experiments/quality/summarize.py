"""Summarize an evaluate.py run: per-method attention-output error/mass recall and perplexity.

    python experiments/quality/summarize.py --run-dir outputs/quality --output-dir outputs/quality/summary

Writes ``quality_metrics.csv`` (relative L2 attention-output error vs dense attention, attention-mass recall,
unique KV and kept-V tokens per layer, over every sampled step and sparse layer) and ``quality_ppl.csv``
(teacher-forced perplexity on the steps every method completed, with a paired bootstrap 95% interval of the
NLL difference to full attention).
"""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def read(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def write(path, rows):
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metrics(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['method']].append(row)
    out = []
    for method in sorted(groups):
        g = groups[method]
        col = lambda k: np.array([float(r[k]) for r in g])
        out.append(dict(method=method, err_mean=float(col('err_mean').mean()), err_p95=float(np.percentile(col('err_mean'), 95)),
                        err_max=float(col('err_max').max()), mass_mean=float(col('mass_mean').mean()),
                        mass_min=float(col('mass_min').min()), unique_kv=float(col('unique_kv').mean()),
                        kept_v=float(col('kept_v').mean()), samples=len(g)))
    return out


def perplexity(rows, bootstrap=2000, seed=0):
    nll = defaultdict(lambda: defaultdict(list))
    for row in rows:
        nll[row['method']][(int(row['block']), int(row['step']))].append(float(row['nll']))
    methods = sorted(nll)
    common = sorted(set.intersection(*(set(nll[m]) for m in methods)))   # steps completed by every method
    value = {m: np.array([np.mean(nll[m][s]) for s in common]) for m in methods}
    rng = np.random.default_rng(seed)
    out = []
    for m in methods:
        diff = value[m] - value['full']
        boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(bootstrap)]
        mine = [r for r in rows if r['method'] == m]
        out.append(dict(method=m, tokens=len(common), ppl=math.exp(value[m].mean()), dnll_vs_full=float(diff.mean()),
                        dnll_ci_lo=float(np.percentile(boots, 2.5)), dnll_ci_hi=float(np.percentile(boots, 97.5)),
                        unique_kv_per_layer=float(np.mean([float(r['unique_kv_per_layer']) for r in mine])),
                        kept_v_per_layer=float(np.mean([float(r['kept_v_per_layer']) for r in mine]))))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-dir', type=Path, required=True, help='--out directory of evaluate.py')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write(args.output_dir / 'quality_metrics.csv', metrics(read(args.run_dir / 'metrics.csv')))
    write(args.output_dir / 'quality_ppl.csv', perplexity(read(args.run_dir / 'nll.csv')))
    print(f'Wrote {args.output_dir}/quality_metrics.csv and quality_ppl.csv')


if __name__ == '__main__':
    main()
