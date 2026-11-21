# Refresh policies

Attention latency (ms) at 41.6 GB/s, N = 32,768. `off` removes refresh entirely and is an optimistic bound; `all-bank` is the strict REFab controller of the 0.1.0 comparison; `per-bank` is the PIM-aware single-bank refresh used by the main results (round rule kept).

| Design | per-bank | per-bank-relaxed | all-bank | all-bank-wc | off |
| --- | ---: | ---: | ---: | ---: | ---: |
| D1 | 23.337 | 19.268 | 41.494 | 39.486 | 18.399 |
| D2 | 16.128 | 10.587 | 24.100 | 23.398 | 10.163 |
| D4 | 12.067 | 7.347 | 15.870 | 15.548 | 7.347 |
| D8 | 7.345 | 7.345 | 18.225 | 19.544 | 7.345 |
| D16 | 7.345 | 7.345 | 31.259 | 23.870 | 7.345 |
| D32 | 7.345 | 7.345 | 16.494 | 18.924 | 7.345 |
| POP_E4P4 | 23.394 | 23.222 | 23.775 | 23.675 | 23.024 |
| POP_E24P32 | 9.238 | 9.060 | 9.629 | 9.524 | 8.900 |
| POP_S_GPU | 7.403 | 7.227 | 7.738 | 7.690 | 7.031 |

Known partial area (mm²) with the per-bank scheduler: D1 3.786, D2 4.200, D4 4.994, D8 6.585, D16 9.955, D32 16.540, POP_E4P4 10.793, POP_E24P32 24.527, POP_S_GPU 9.652.
