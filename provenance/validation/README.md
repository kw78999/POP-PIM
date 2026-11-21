# Release validation (0.2.0)

Checks actually executed while preparing this release, from this repository's source, with GCC 10.5, Verilator 4.028,
Yosys 0.9, OpenSTA 2.0.17 and Python 3.11 (`summary.json` has the exact versions). Local paths are replaced by
`<toolchain>`/`<scratch>`. They are not GitHub CI results; the workflow in `.github/workflows/ci.yml` has not run yet.

| Record | What was checked |
|---|---|
| `pytest.log` | The complete test suite: 166 tests passed |
| `build_manifest.json`, `refresh.jsonl`, `refresh_single_bank.jsonl`, `gqa.json` | Offline source build; upstream refresh defect vs corrected model with exact nRP/nRFC boundaries; exact single-bank refresh boundaries; live GQA RTL |
| `summary.json` → `binaries_identical_to_development_build` | The rebuilt `poppim_dense`/`poppim_pop`, Ramulator library and GQA archive have the same SHA-256 as the development build that produced the per-bank refresh and byte-frugal results, also after the comment-only cleanup of the simulator source |
| `config_comparison.json` | The configuration builder reproduces the YAML of all 274 archived development runs it supports, apart from file paths |
| `archive_comparison.json` | All 194 runs of the `main`, `ablation` and `refresh` reference suites, regenerated with `tools.run_simulation`, against the archived development runs: 181 identical, 13 identical in every shared metric (their archived build predates the `phase21_*` counters), none different |
| `summary.json` → `reference_suites`, `quick_suite` | Each reference run passed the raw-trace audit before its trace was discarded; `tools.reproduce --suite quick` reproduced its cases from a clean output directory |
| `rtl_extensions.json` | `tools.verify_rtl`: refresh scheduler (11 cases), exact ×0.25 (65,536 inputs), GS unit (262,144 scores), VS engine (16 groups, 81,920 keep bits, scan cycles) |
| `synthesis_extensions.json`, `synthesis_extensions_after_comment_cleanup.json` | `tools.synthesize_extensions`: all 9 blocks reproduce their recorded mapped-cell areas; after a comment-only cleanup of the GS, VS, staging and refresh-scheduler RTL, the 8 affected blocks were synthesized again with identical results |
| `cacti_extensions.json` | `tools.reproduce_cacti --extension-sizes`: the eight SRAM macros of the area model, identical areas |
| `vkeep_regeneration.json` | `experiments/replay/make_vkeep_masks.py` on the activation snapshot: all six masks and 16 VS test vectors byte-identical |
| `pip_freeze.txt` | Resolved Python environment |

`v0.1.0/` keeps the validation records of the previous release unchanged.
