# Changelog

## 0.2.0 — prepared 2026-11-21

- POP-PIM-S and POP-PIM-Q: GQA-shared selection (GS), two-phase K→V gather with mass-guided V skipping (VS) and
  compact host transport (CT), evaluated on the S path (host Top-k) and the E24/P32 local selector.
- PIM-aware per-bank refresh (REFsb) in the controller, with exact single-bank timing checks, a refresh-scheduler RTL
  and four comparison policies (relaxed, all-bank, work-conserving all-bank, off). Per-bank refresh is the new default.
- Channel-bandwidth sweep (41.6/20.8/10.4/5.2 GB/s) and host-link sweep; measured CPU Top-k for the S path.
- New RTL with Verilator checks: GS unit, VS engine, 64/16-bit S-path staging, GS score bank, refresh scheduler.
- Known-partial area model covering the new blocks; synthesis and CACTI reproduction of their components.
- `tools.reproduce` runs named suites (quick, main, ablation, refresh), audits every raw trace and compares each run
  with the committed reference; all reference records were regenerated with this release.
- Optional model-quality evaluation (attention-output error, perplexity) and regeneration of the activation-derived
  V-keep masks.
- The 0.1.0 comparison moved to `results/reference/all_bank_refresh_v0.1/`; it is reproduced by the `all-bank` rows of
  the refresh suite. Exploratory studies and development reports stay outside the repository.

## 0.1.0 — prepared 2026-10-30

- Standalone research release of Dense D1–D8 versus POP-PIM E4/P4 and E24/P32 under a corrected all-bank refresh
  model, with portable builds, frozen replay inputs, RTL calibration, trace auditing and table reconstruction.

The project license and final publication metadata remain maintainer decisions. These entries describe prepared
releases, not a published GitHub release or an accepted paper.
