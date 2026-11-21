# 0.1.0 comparison (all-bank refresh)

This was the main comparison of release 0.1.0 and is kept unchanged. The current simulator reproduces every
metric of these six runs in the `all-bank` rows of `results/refresh/` (checked by `tests/test_reference_results.py`);
the 0.1.0 configurations name the DRAM model `Phase17HBM3`, which the current source calls `Phase20HBM3` with the
same all-bank behavior. The 0.1.0 area column uses the retained (pre-0.2.0) component values.

Six cases were freshly rerun from the prepared source: D1/D2/D4/D8 and POP-PIM E4/P4, E24/P32. All use the same corrected pseudochannel HBM model, strict refresh-priority controller, 64 banks, N=32,768, K=4,096, and declared clocks/backend.

`runs.json` contains full numerical metrics and hashes. `raw/<design>/` retains compact metrics, resolved configurations, run logs and records. `audit.json` records the independent audit of every original command trace. Large traces and binaries remain generated outputs, not committed files. Resolved configurations are frozen measurement provenance and contain the run's absolute paths; regenerate portable configurations with `tools.run_simulation` instead of executing those historical path strings directly.

`main_comparison.csv` and `comparison.md` were derived by the 0.1.0 `tools.rebuild_tables`. Area inputs are retained mapped-cell and supporting-component evidence, not newly fabricated total chip area. The separate D8 PPA rerun is a flow validation and does not replace a single component in this consistently defined reference area table.

Dense scalar metrics reproduce the retained final Dense audit exactly. POP results are new matched-refresh reruns; neither the old manuscript rankings nor no-refresh numbers are substituted. Both paths have their declared, different quality contracts; the replay is not a new PPL evaluation.
