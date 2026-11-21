# Known partial area

Per HBM3 channel. Mapped Nangate45 cells (Yosys 0.9 + ABC, pre-layout OpenSTA) plus CACTI 45 nm itrs-hp SRAM estimates and retained support proxies; blocks common to every design are excluded. `area_breakdown.csv` itemizes every configuration of the main, ablation and refresh suites.

## Selected configurations (per-bank refresh)

| Configuration | Scorer cells ×64 | Support | Refresh scheduler | GS/VS/CT and selection state | Total (mm²) |
|---|---:|---:|---:|---:|---:|
| Dense D4 | 1.745 | 3.239 | 0.0104 | — | 4.994 |
| Dense D8 | 3.336 | 3.239 | 0.0104 | — | 6.585 |
| Dense D16 | 6.706 | 3.239 | 0.0104 | — | 9.955 |
| Dense D32 | 13.290 | 3.239 | 0.0104 | — | 16.540 |
| POP-PIM (S path) | 8.598 | 0.264 | 0.0104 | +0.780 | 9.652 |
| POP-PIM-Q (S path) | 8.598 | 0.264 | 0.0104 | +3.020 | 11.892 |
| POP-PIM-S (S path) | 8.598 | 0.264 | 0.0104 | +2.136 | 11.008 |
| POP-PIM E24/P32 (L path) | 8.598 | 15.919 | 0.0104 | — | 24.527 |
| GS on E24/P32 (L path) | 8.598 | 15.919 | 0.0104 | -10.949 | 13.578 |

## Synthesized blocks

| Block | Cell area (µm²) | Flip-flops | Instances per channel | 125 MHz slack (ns) | RTL |
|---|---:|---:|---:|---:|---|
| Dense D8 scorer, 16-byte ports | 52,118 | 4,138 | 64 | 5.488 | `rtl/dense/dense_score_port16.sv` |
| POP score bank (A2 scorer, query mux, GQA4 frontend) | 134,344 | 11,929 | 64 | 5.330 | `rtl/bank/gqa4_score_bank.sv` |
| GS unit | 11,726 | 946 | 64 | 5.557 | `rtl/gs/gs_shared_score.sv` |
| VS engine (logic) | 25,977 | 450 | 8 | 0.395 | `rtl/vs/vs_mass_threshold.sv` |
| S-path staging, 64-bit packets | 41,322 | 5,312 | 1 |  | `rtl/transport/score_stream_interface.sv` |
| S-path staging, 16-bit packets (CT) | 11,760 | 1,472 | 1 | 7.430 | `rtl/transport/score_stream_interface_w.sv` |
| Refresh scheduler (round rule, H=2) | 10,425 | 533 | 1 | 4.699 | `rtl/refresh/pim_refresh_sched.sv` |
| POP score bank with GS unit (integration check) | 136,162 | 12,815 | 64 | 5.219 | `rtl/bank/gqa4_score_bank_gs.sv` |

The S-path staging is timed at 250 MHz (output lanes). The area model adds the standalone GS unit (64 × 11,726 µm²) although the integrated bank grew by only 1,818 µm², which keeps the GS cost conservative.
