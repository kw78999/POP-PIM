# Release preparation validation

These records describe checks actually executed from the prepared source. They are distinct from historical validation files and from future GitHub CI results.

- `pytest.log` / `pytest.xml`: 120 CPU correctness tests passed in the fresh Python 3.11 environment.
- `numpy2_bf16.log`: all seven BF16 arithmetic checks also passed with NumPy 2.2.6, in addition to the pinned NumPy 1.26.4 environment.
- `refresh.jsonl`: preserved upstream defect demonstration, corrected pseudochannel behavior, and exact nRP/nRFC boundaries passed.
- `gqa.json`: live GQA RTL tests passed, including stale/duplicate response rejection and independent release behavior.
- `dense_*_checks.json`: fresh D1/D2/D4/D8 numerical, backpressure, reset and finite-tile checks; all 64 observed tiles per design match the retained schedules.
- `pop_checks.json`: fresh 4,096-vector corpus in normal, backpressure and reset modes; all outputs bit exact, streaming II=22 cycles.
- `ppa_d8_*`: actual D8 port-wrapper synthesis/STA rerun. Its mapping differs from the retained core-only area, as disclosed in the reproduction guide.
- `ppa_pop_*`: actual integrated GQA4+scorer bank synthesis/STA rerun; mapped area 134,344.364 µm², with 5.330049-ns slack at 125 MHz. The comparison retains its original component model.
- `cacti_models.json`: all 32 retained SRAM configurations rerun; every macro dimension matches its retained output exactly.
- `build_manifest.json`: freshly built executable/library hashes and source hashes.
- `summary.json`: environment, runtime, comparison and preservation summary.

The prepared source was rebuilt against the vendored library, rather than linking the original simulator binary. A locally extracted external GCC/Verilator/Yosys/STA toolchain was used; its local filesystem path in recorded command lines is machine provenance, not a source dependency. The external tools are not redistributed.

Full fresh architectural results and raw-command audit summaries are under `results/main/`. The main table uses retained supporting-area models, not a full-chip implementation. D16/D32 were not repeated during release preparation. The CACTI run used an isolated copy of the existing pinned local executable; its optional network fetch/build branch was not executed. The GitHub workflow is supplied; it has not yet run on GitHub.
