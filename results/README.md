# Results inventory

| Location | Evidence | Rebuild / rerun |
|---|---|---|
| `main/` | Dense D4/D8/D16/D32 vs POP-PIM, POP-PIM-Q, POP-PIM-S (S path, measured CPU Top-k) at 41.6/20.8/10.4/5.2 GB/s | `tools.reproduce --suite main` |
| `ablation/` | GS/VS/CT variants on the S path (GPU assumption and CPU) and on the E24/P32 L path; host-link sweep | `tools.reproduce --suite ablation` |
| `refresh/` | Per-bank, relaxed per-bank, all-bank, work-conserving all-bank and no refresh for Dense and POP | `tools.reproduce --suite refresh` |
| `area/` | Itemized known partial area of every configuration in the suites; synthesized block summary | `tools.area_model`, `tools.synthesize_extensions` |
| `quality/` | Attention-output error and perplexity of the selection variants (Llama-3.1-8B, PG19, 32K) | `experiments/quality/` |
| `reference/all_bank_refresh_v0.1/` | The 0.1.0 comparison: D1–D8 vs POP-PIM E4/P4, E24/P32 with all-bank refresh, as released then | Same numbers from the `all-bank` rows of `refresh/` |
| `reference/refresh_dense/` | Retained 0.1.0 Dense finite-tile measurements, mapped-area/STA evidence and validation records | — |
| `reference/no_refresh_diagnostic/` | Retained 0.1.0 no-refresh comparison, explicitly optimistic | — |

Each suite folder holds `runs.json` (one record per run: case parameters, full metrics, configuration/binary/library/
input hashes and the summary of the raw-trace audit), `<suite>.csv` (derived quantities in full precision) and
`<suite>.md` (summary tables), all regenerable with `python -m tools.rebuild_tables --suite <suite>`. Raw command
traces were audited when they were produced and are not committed.

Latency is the simulated attention output time of one layer for one decode token, plus the VS scan where the
hardware VS rule applies. Area is the known partial area per channel; total chip area, power and energy are not
reported. See [docs/results.md](../docs/results.md) for the measurement scope.
