# Development history and separated material

This repository is the release of a longer development effort. Exploratory studies, reports, debugging scripts,
manuscript drafts and large raw outputs were kept out of it; what remains is the code and compact evidence that
reproduce the reported results.

| Stage | What is in this release | What was separated |
|---|---|---|
| Functional placement, selector and scorer studies | Core algorithms, oracles and tests (`poppim/`, `tests/`) | Chronological reports, intermediate scripts |
| Scoring/GQA integration, context and port sweeps | RTL, calibration fixtures, Dense finite-tile evidence (`results/reference/refresh_dense/`) | Older rankings computed with an earlier HBM refresh model |
| Refresh-scope audit and the 0.1.0 comparison | Corrected pseudochannel model; the 0.1.0 all-bank comparison (`results/reference/all_bank_refresh_v0.1/`) | Manuscript drafts based on the older refresh model |
| PIM-aware per-bank refresh | Controller policy, scheduler RTL, exact REFsb timing checks; the `refresh` suite | Long-context POP replays (64K/128K) and an escalation policy variant that were not carried forward |
| Byte-frugal dataflow (GS, VS, CT) and bandwidth sweep | Simulator support, GS/VS/staging RTL with tests, area model, the `main` and `ablation` suites, quality summary | Abstract drafts, snapshot probes, design-space scans of the VS binning |
| Bank-area reduction study (tag compression, router SRAM, query broadcast) | — | The whole study: it is exploratory and not part of POP-PIM-S/Q as reported |

Development phase numbers survive only in a few internal identifiers (`Phase13Common`, `Phase20HBM3`, metric keys
`phase20_*`/`phase21_*`) so that archived evidence stays comparable; nothing in this repository depends on the
separated material.

The integrated simulator sources are those of the development build that produced the per-bank refresh and
byte-frugal results. Rebuilt from this repository with the same compiler, the Dense and POP executables are
byte-identical to that build, and every simulation configuration the tools generate matches the archived runs
apart from file paths (`tests/test_simulation_config.py` keeps seven of them as fixtures). The reference records in
`results/main`, `results/ablation` and `results/refresh` were regenerated with this repository's tools and compared
metric by metric with the archived development runs.

Earlier manuscript drafts compare designs under an older refresh implementation; their numbers are historical and are
not results of this release. A later large artifact bundle, if published, belongs in a GitHub release or an archival
service with its own manifest and checksums.
