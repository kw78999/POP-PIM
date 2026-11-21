# Model quality summary

Llama-3.1-8B, PG19 at 32K context, CPU, teacher-forced decoding, 30 sparse layers (layers 0 and 1 dense). Produced
by `experiments/quality/evaluate.py` with `run_config.json` and summarized by `experiments/quality/summarize.py`.

| Configuration | Method | Attention-output error, mean / p95 / max | Selected mass, mean | Perplexity (64 tokens) | ΔNLL vs full, 95% CI |
|---|---|---:|---:|---:|---:|
| Dense | `full` | 0 | 1 | 22.39 | — |
| POP-PIM | `ph:4096` | 0.166 / 0.394 / 7.90 | 0.906 | 22.73 | [−0.051, 0.076] |
| GS only | `gs:4096` | 0.139 / 0.232 / 1.95 | 0.910 | 22.71 | [−0.026, 0.051] |
| POP-PIM-S | `gs:4096:vs:0.005` | 0.145 / 0.242 / 2.11 | 0.910 | 22.57 | [−0.035, 0.047] |
| POP-PIM-Q | `gs:6144:vs:0.002` | 0.105 / 0.173 / 1.71 | 0.934 | 22.50 | [−0.026, 0.033] |

Error is the relative L2 distance of each sparse layer's attention output from dense attention, over 4 decode steps ×
30 layers = 120 samples per method; the selected mass is the dense softmax mass captured by the selected token set
(before any V row is skipped), so VS does not change it. The quality run uses the exact VS rule; on the snapshot the
hardware rule keeps 4.6% (S) and 5.5% (Q) more V rows, a superset, so it skips no more attention mass than the
exact rule. `quality_metrics.csv` also lists Top-1,024/2,048, GS 2,048/8,192,
other δ values and an oracle Top-4,096.

Perplexity covers only the 64 tokens that every method completed, so it is a small-sample check: all differences to
full attention lie inside their paired-bootstrap 95% intervals.
