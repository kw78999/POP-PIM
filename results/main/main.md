# Main comparison

One Llama-3.1-8B attention layer, one decode token, N = 32,768, one HBM3 channel (64 banks), per-bank (PIM-aware) refresh. POP-PIM rows use the S path with the measured 8-thread CPU Top-k. Latency includes the VS threshold scan. Area is the known partial area per channel (not total chip area).

## Attention latency (ms)

| Configuration | 41.6 GB/s | 20.8 GB/s | 10.4 GB/s | 5.2 GB/s |
| --- | ---: | ---: | ---: | ---: |
| Dense D4 | 12.067 | 13.770 | 18.786 | 31.430 |
| Dense D8 | 7.345 | 7.356 | 14.361 | 27.497 |
| Dense D16 | 7.345 | 7.355 | 13.971 | 27.671 |
| Dense D32 | 7.345 | 7.349 | 14.396 | 27.578 |
| POP-PIM | 9.991 | 10.064 | 10.447 | 14.408 |
| POP-PIM-Q | 7.908 | 7.984 | 8.227 | 9.072 |
| POP-PIM-S | 6.435 | 6.510 | 6.710 | 7.174 |

## Speed-up over Dense D8 (latency ratio)

| Configuration | 41.6 GB/s | 20.8 GB/s | 10.4 GB/s | 5.2 GB/s |
| --- | ---: | ---: | ---: | ---: |
| Dense D4 | 0.61x | 0.53x | 0.76x | 0.87x |
| Dense D8 | 1.00x | 1.00x | 1.00x | 1.00x |
| Dense D16 | 1.00x | 1.00x | 1.03x | 0.99x |
| Dense D32 | 1.00x | 1.00x | 1.00x | 1.00x |
| POP-PIM | 0.74x | 0.73x | 1.37x | 1.91x |
| POP-PIM-Q | 0.93x | 0.92x | 1.75x | 3.03x |
| POP-PIM-S | 1.14x | 1.13x | 2.14x | 3.83x |

## Bytes, area and area x delay

| Configuration | Channel MiB | Host link MiB | Known partial area (mm²) | A×D @41.6 (mm²·ms) | A×D @5.2 (mm²·ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense D4 | 128.00 | 0.000 | 4.994 | 60.3 | 157.0 |
| Dense D8 | 128.00 | 0.000 | 6.585 | 48.4 | 181.1 |
| Dense D16 | 128.00 | 0.000 | 9.955 | 73.1 | 275.5 |
| Dense D32 | 128.00 | 0.000 | 16.540 | 121.5 | 456.1 |
| POP-PIM | 55.84 | 8.500 | 9.652 | 96.4 | 139.1 |
| POP-PIM-Q | 29.33 | 0.531 | 11.892 | 94.0 | 107.9 |
| POP-PIM-S | 21.60 | 0.531 | 11.008 | 70.8 | 79.0 |
