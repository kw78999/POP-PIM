# Main architectural comparison

One-layer attention latency; known partial area excludes unimplemented/unknown blocks.

| Design | Context | Refresh | Attention (ms) | Scorer cells, 64 banks (mm²) | Known partial area (mm²) |
|---|---:|---|---:|---:|---:|
| D1 | 32768 | True | 41.494400 | 0.535456 | 3.774173 |
| D2 | 32768 | True | 24.100080 | 0.950450 | 4.189168 |
| D4 | 32768 | True | 15.870288 | 1.737895 | 4.976613 |
| D8 | 32768 | True | 18.224537 | 3.377681 | 6.616398 |
| POP_E4P4 | 32768 | True | 23.774929 | 1.673834 | 10.803492 |
| POP_E24P32 | 32768 | True | 9.629112 | 1.673834 | 24.538036 |
