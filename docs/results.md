# Interpreting the results

Every latency is the simulated attention service time of **one Llama-3.1-8B layer for one decode token** at
N = 32,768 on one HBM3 channel. It is not an end-to-end decode latency, and no number here is a silicon measurement.

| Quantity | Evidence | Limits |
|---|---|---|
| Attention latency, DRAM commands, bytes | Ramulator 2 HBM3 simulation with an RTL-calibrated scorer, the live GQA router RTL, PIM-aware per-bank refresh, and a real 32K PG19 score/activation snapshot | One layer-2 snapshot; completion (exact QK/softmax/AV) is a service model shared by all designs |
| VS scan time | Cycle count of the VS RTL on the snapshot (added serially at 125 MHz) | Hardware-rule variants only; the exact-rule ("ideal") variants carry no scan cost |
| Host Top-k (S path) | GPU: assumed 0.2 ms. CPU: measured 8-thread median on the development host | Re-measure with `tools.measure_host_topk`; host and link timing are models, not a measured system |
| Scorer, GS, VS, staging, refresh-scheduler area and timing | Yosys 0.9 + ABC mapping to Nangate45, pre-layout OpenSTA | No placement/routing, clock tree, or DRAM-process implementation |
| SRAM area | CACTI 45 nm itrs-hp estimates | Model estimates, not compiler macros |
| Known partial area | Mapped cells + SRAM estimates + declared support proxies (`tools/area_model.py`) | Excludes blocks common to all designs (exact-attention backend, CDC/PHY, clocking, layout); not a total chip area |
| Attention-output error, perplexity | CPU Llama-3.1-8B on PG19 at 32K context, 30 sparse layers, teacher forcing, FP32 PopKV scores and the exact VS rule | Error: 4 decode steps × 30 layers = 120 samples per method. Perplexity: only 64 tokens; every difference lies inside its 95% bootstrap interval. The hardware VS rule keeps a superset of the exact rule's tokens |
| Energy, power, total chip area | Not measured | No value is reported or implied |

## What is compared

- **Dense D4–D32** read all 128 MiB of K/V per layer-token. Above about 20 GB/s they saturate on the shared
  softmax/AV backend near 7.35 ms; D16/D32 add area for at most a 2.7% gain over D8.
- **POP-PIM** (per-head selection) reads 40 B/token of metadata plus the union of four per-head Top-4,096 lists.
- **POP-PIM-S/Q** add GS, VS and CT. Their quality contract differs from Dense: selection is exact over
  approximate stored scores, and VS bounds the dropped attention mass by δ per head. The error and perplexity
  results above, not the latency tables, are the evidence for output quality.

The main table uses the **S path with the measured CPU Top-k**, so no GPU assumption enters it. The ablation table
also reports the GPU assumption and the on-chip E24/P32 selector.

## Assumptions to keep in mind

- **Per-bank refresh** keeps a round rule as the conservative assumption; whether a given HBM3 device allows the
  per-bank postponement it relies on has to be checked against that device's datasheet. The `refresh` suite reports
  all-bank, work-conserving, relaxed, and no-refresh variants next to it; `off` is an optimistic bound only.
- **Lower channel bandwidths** are emulated by a longer burst (`nBL`) of the same preset, not by a different device.
- **One snapshot.** The replay and the V-keep masks come from a single layer-2 decode step; other layers, prompts or
  models can select differently.
- **Area accounting** follows one consistent rule across designs (see `tools/area_model.py`). GS on the S path is
  counted only with CT, because without CT the host receives all per-head scores and can aggregate them itself. The
  conservative standalone GS unit area is used even though the integrated bank synthesis grew by less.

## Reference results and provenance

| Location | Content |
|---|---|
| `results/main/` | Dense D4–D32 vs POP-PIM, POP-PIM-Q, POP-PIM-S at 41.6/20.8/10.4/5.2 GB/s (28 runs) |
| `results/ablation/` | GS/VS/CT ablation on the S path (GPU and CPU Top-k) and the E24/P32 L path, host-link sweep (121 runs) |
| `results/refresh/` | Five refresh policies for D1–D32, POP E4/P4, E24/P32 and the S path (45 runs) |
| `results/quality/` | Attention-output error and perplexity summaries |
| `results/area/` | Itemized known partial area of every reported configuration and the synthesized blocks |
| `results/reference/all_bank_refresh_v0.1/` | The 0.1.0 comparison (all-bank refresh, D1–D8 vs E4/P4, E24/P32) as released then |
| `results/reference/refresh_dense/`, `no_refresh_diagnostic/` | Retained 0.1.0 Dense finite-tile and no-refresh evidence |

Every reference run was produced by this repository's build, and its raw command trace was audited
(`tools.audit_simulation`) before it was discarded. `runs.json` keeps the full metrics, the configuration, binary,
library and input hashes, and the audit summary. `tools.reproduce` reruns a suite and compares it with these
records metric by metric. The 0.1.0 numbers are reproduced by the `all-bank` rows of the refresh suite.
