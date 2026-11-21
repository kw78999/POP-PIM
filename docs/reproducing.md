# Reproduction guide

Run every command from the repository root with the Python environment from the README activated. Each step
writes into a fresh directory under `outputs/` or `build/`, both ignored by Git; committed reference data is never
overwritten.

| Step | Needs | Time (4 cores) |
|---|---|---|
| [1. CPU checks and reference tables](#1-cpu-checks-and-reference-tables) | Python 3.10+ | about 1 min |
| [2. Build the simulator](#2-build-the-simulator) | C++20 compiler, CMake, Make, Verilator | about 5 min |
| [3. Simulation suites](#3-simulation-suites) | step 2 | 1 min (quick) to about 1 h (all suites) |
| [4. RTL checks](#4-rtl-checks-of-the-new-blocks) | Verilator | about 3 min |
| [5. Area and timing](#5-area-and-timing) | Yosys, ABC, OpenSTA, optionally CACTI | about 30 min |
| [6. Host Top-k](#6-host-top-k-on-your-machine) | C++ compiler | under 1 min |
| [7. Optional: model quality](#7-optional-model-quality) | torch, transformers, Llama-3.1-8B, PG19 | many CPU hours |

**Toolchain.** The reference results were produced with GCC 10.5, CMake 3.16, Verilator 4.028, Yosys 0.9 (with its
ABC), OpenSTA 2.0.17 and Python 3.11.13 on Ubuntu 20.04. These are the Ubuntu 20.04 package versions of GCC 10,
Verilator and Yosys. Other versions were not tested: `tools.reproduce` reports any simulated metric that differs from
the reference, and mapped synthesis areas depend on the Yosys/ABC version. `CXX` selects the compiler; `POPPIM_TOOLCHAIN_ROOT` may point to an extracted toolchain that has
`usr/bin` and `usr/share/verilator`; `POPPIM_BUILD_DIR` moves `build/`.

## 1. CPU checks and reference tables

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q
python -m tools.verify_replay
python -m tools.prepare_inputs
python -m tools.rebuild_tables --suite main --output-dir outputs/tables/main
python -m tools.check_release
```

The tests cover the functional encoders, scorers and selectors with boundary and tie cases, integer BF16 oracles
(with an independent compiled C++ scalar oracle when `g++` exists), the GS replay and V-keep masks against their
committed hashes, the configuration builder against configurations of the archived development runs, the area
model, and the invariants of every committed reference record. `verify_replay` checks the stable Top-k of all 32
heads. `prepare_inputs` regenerates the GS replay variants from `data/replay/scores_bf16.bin`, unpacks the
V-keep masks and checks every hash. `rebuild_tables` derives the report tables from the committed records; it does
not run a simulation.

## 2. Build the simulator

```bash
sudo apt-get install -y build-essential g++-10 cmake verilator   # Ubuntu 20.04/22.04
export CXX=g++-10
python -m tools.build --jobs 4
```

The build uses only vendored source (no download). It compiles Ramulator 2, Verilates the GQA4 frontend, links the
Dense and POP adapters, and runs three checks: the preserved upstream pseudochannel refresh defect next to the
corrected model with exact nRP/nRFC boundaries, the exact single-bank refresh boundaries (nRP 19, nRC 63, nRFCSB 260,
nRREFD 11 controller cycles), and the live GQA metadata/router RTL. Logs, hashes and check results are under
`build/`. Rebuild after moving the checkout.

## 3. Simulation suites

```bash
python -m tools.reproduce --suite quick        # 3 cases
python -m tools.reproduce --suite main         # 28 cases
python -m tools.reproduce --suite ablation     # 121 cases
python -m tools.reproduce --suite refresh      # 45 cases
```

`tools.reproduce` runs the cases in parallel (`--jobs`), audits each raw command trace right after its run, compares
every metric with the committed reference in `results/`, and writes report tables to `outputs/<suite>/tables/`.
It exits with an error if any case differs. Command traces are kept only for `quick` (`--keep-traces` keeps all; a
32K Dense trace is about 100 MB).

| Suite | Cases |
|---|---|
| `quick` | Dense D8, POP-PIM and POP-PIM-S (S path, CPU Top-k) at 41.6 GB/s |
| `main` | Dense D4/D8/D16/D32 and POP-PIM, POP-PIM-Q, POP-PIM-S (CPU Top-k) at 41.6, 20.8, 10.4, 5.2 GB/s |
| `ablation` | All 14 variants on the S path with the GPU Top-k assumption, the 9 non-CT variants on the E24/P32 L path, a host-link sweep (8, 2, 0.5 GB/s), and the ideal-VS CPU variants |
| `refresh` | D1–D32, POP E4/P4, E24/P32 and the S path under `per-bank`, `per-bank-relaxed`, `all-bank`, `all-bank-wc` and `off` refresh |

Low-bandwidth Dense cases dominate the runtime (about 8 minutes each at 5.2 GB/s).

### Individual cases

```bash
python -m tools.run_simulation --designs D8 POP_S_CPU --variants POP-PIM POP-PIM-S \
  --bandwidths 41.6 5.2 --audit --output-dir outputs/custom
python -m tools.run_simulation --designs POP_S_GPU --variants gs4096_ct --host-links 2 \
  --output-dir outputs/link2
python -m tools.run_simulation --designs D4 --context 131072 --refresh all-bank --output-dir outputs/d4_128k
python -m tools.rebuild_tables --suite main --runs-dir outputs/main --output-dir outputs/main/tables
python -m tools.audit_simulation --input-dir outputs/custom
```

`--dry-run` prints the resolved Ramulator configurations without running. Variants accept `POP-PIM`, `POP-PIM-S`,
`POP-PIM-Q` or an internal id from `tools/run_simulation.py: VARIANTS`; they apply to POP designs only, and compact
transport (`*_ct`) only to the S path. The POP replay is fixed at N = 32,768; Dense also supports 64K and 128K.

Each case folder holds `config.yaml`, `metrics.json`, `record.json`, `run.log` and, unless discarded,
`command_trace.bin` (24-byte little-endian records: uint64 time in ps, uint64 bank mask, uint64 command:
1 ACT, 2 PRE, 3 PREA, 4 RD, 5 WR, 6 REFab, 7 other, 8 REFsb). The audit re-derives command counts, row
conservation, scorer/selector/host-transfer work, two-phase K/V reads against the masks, REFab timing, and REFsb
timing, debt window and round rule from the trace, and checks every file and binary hash.

## 4. RTL checks of the new blocks

```bash
python -m tools.verify_rtl
```

Builds four Verilator testbenches: the per-bank refresh scheduler against a cycle-by-cycle transliteration of the
simulator's controller policy (11 parameter and stress cases), the exact BF16 ×0.25 unit against the BF16 FMA on
all 65,536 inputs, the GS unit on the full 32K × 32-head replay with random input gaps and back-pressure (262,144
shared scores, bit-exact), and the VS engine on 16 KV groups of real logits (b* per head, 81,920 keep bits and scan
cycles). The original Dense, popcount scorer and GQA checks remain available:

```bash
python -m tools.calibrate_dense --lanes 1 2 4 8 --output-dir outputs/rtl_dense
python -m tools.calibrate_pop --samples 4096 --output-dir outputs/rtl_pop
```

They regenerate the seed-9042 Dense corpus (numerical, back-pressure, reset and 64 drained 16-score tiles per design,
which must match `data/calibration/`) and the seed-7007 popcount scorer corpus (scorer II of 22 cycles). Calibration
files under `data/` are never overwritten by these commands.

## 5. Area and timing

```bash
sudo apt-get install -y yosys     # plus OpenSTA (the `sta` binary)
python -m tools.fetch_liberty     # pinned Nangate45 typical Liberty, checked by SHA-256
python -m tools.synthesize_dense --lanes 1 2 4 8 16 32 --output-dir outputs/ppa_dense
python -m tools.synthesize_pop --output-dir outputs/ppa_pop
python -m tools.synthesize_extensions --output-dir outputs/ppa_extensions
python -m tools.reproduce_cacti --fetch --extension-sizes --output-dir outputs/cacti_extensions
```

`synthesize_dense` and `synthesize_pop` map the 16-byte-port Dense scorers and the integrated GQA4 score bank.
`synthesize_extensions` maps the GS unit, the VS engine, both S-path staging widths, the GS score bank and three
refresh-scheduler variants, and compares each mapped cell area with `data/area/extension_components.json`.
`reproduce_cacti --extension-sizes` derives the eight SRAM macros of the GS/VS/CT area model from the retained
CACTI configuration (`--executable /path/to/cacti` instead of `--fetch` uses a local build of the pinned revision;
without `--extension-sizes` it reruns the retained configurations under `data/area/cacti/`).

`tools/area_model.py` turns these components into the known partial area of every design and variant. All of this
is pre-layout logic area plus SRAM estimates; it excludes blocks common to all designs and is not a chip area.

## 6. Host Top-k on your machine

```bash
python -m tools.measure_host_topk
python -m tools.run_simulation --designs POP_S_CPU --variants POP-PIM-S --host-topk-ms 0.9 \
  --output-dir outputs/my_cpu
```

The benchmark times the S-path selection (8 threads, spawn/join included) for 32 rows × Top-4,096 (POP-PIM),
8 rows × Top-4,096 (POP-PIM-S) and 8 rows × Top-6,144 (POP-PIM-Q), checks the selected sets against the replay,
and prints them next to the recorded development-host values. Pass your own median with `--host-topk-ms`.

## 7. Optional: model quality

[`experiments/quality/`](../experiments/quality/README.md) evaluates attention-output error and perplexity of the
selection variants on Llama-3.1-8B with PG19 at 32K context. It needs the model weights, the PG19 test books,
`torch` and `transformers`, more than 32 GB of memory and many CPU hours; it is not part of the simulator
reproduction. [`experiments/replay/make_vkeep_masks.py`](../experiments/replay/make_vkeep_masks.py) regenerates the
committed V-keep masks and VS test vectors from the activation snapshot and checks them byte for byte.
