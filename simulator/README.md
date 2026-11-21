# Architectural simulator

Ramulator 2 frontends for one decode token of one attention layer on one HBM3 channel. Build with
`python -m tools.build`; run cases with `python -m tools.run_simulation` or whole suites with `python -m tools.reproduce`.

| File | Role |
|---|---|
| `dense/frontend.cpp` | Dense D1–D32: replays the measured finite-tile scorer schedule, then full K/V attention service |
| `pop/frontend.cpp` | POP-PIM: live Verilated GQA4 router RTL, calibrated scorer, real score replay, selected K/V gather, two-phase K→V with V-keep masks |
| `common/offload.inc` | S path: output lanes, staging FIFOs, host link, external Top-k, index return, compact transport |
| `common/selector_cluster_sram.h` | L path: exact shared selector engines/ports model |
| `common/controller.cpp` | HBM3 controller with PIM bank exclusion and the refresh policies (`strict`, `wc_ab`, `pim_pb`) |
| `common/hbm3_scoped.cpp` | Corrected HBM3 model: pseudochannel-scoped PREA/REF and same-bank REFsb timing |
| `tests/` | Refresh scope/boundary (`refresh.cpp`, `refresh_single_bank.cpp`) and GQA RTL (`gqa.cpp`) checks run by the build |

Internal registration names (`Phase13Common`, `Phase20HBM3`) and some metric keys keep their development identifiers
for comparability with archived evidence. [docs/architecture.md](../docs/architecture.md) describes the clocks,
bandwidth and precision contracts.
