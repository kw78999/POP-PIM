# Reproduction inputs

`replay/scores_bf16.bin` stores a fixed [32, 32768] array of little-endian uint16 BF16 score bits. `replay/topk_indices_i32.bin` stores [32, 4096] little-endian int32 selected token indices, sorted by token position. Stable selection uses score descending and index ascending ties. These files total 2.5 MiB and contain no model weights, book text, or token IDs. Their hashes and provenance/scope are in `replay/manifest.json`.

The scores originate from a saved real PG19 layer-2 pilot snapshot, encoded with token/dimension groups of 64. The architectural selector chooses exactly 4,096 entries from all 32,768 scores, without forcing the current token. This service experiment differs from a GPU decode policy that selects 4,095 historical entries plus the current token. The original capture record is retained in `../provenance/replay_capture.json`; its separate GPU-policy union diagnostic is not a quality validation of this replay. Regenerating the original activation requires the separately maintained model/data/capture environment; the distributed frozen score fixture is the reproducible input for this hardware study.

`calibration/schedules/D*.txt` contains one input-start/output-ready cycle pair per score in a drained 16-score tile. `tile_schedules.json` links these schedules to 64 validated RTL tiles; `port_rtl.json` retains streaming, backpressure, reset, and tile-run statistics. Timing is at the declared 125-MHz scorer clock, with independent Q and K 16-byte ports.

`calibration/hardware_summary.csv` records mapped scorer cell area and pre-layout timing. `area_breakdown.csv` supplies Dense supporting-area estimates; `pop_area_components.csv` supplies POP selector SRAM/support components. `area/cacti/` contains the retained configurations/logs behind SRAM estimates. The source CACTI revision is recorded separately; the release does not automatically rerun the complete supporting-area flow.

## Byte-frugal variants and new components

`replay/variants/manifest.json` describes the derived replay inputs of POP-PIM-S/Q. The GQA-shared (GS) scores and
Top-4,096/6,144 indices are regenerated from `replay/scores_bf16.bin` by `python -m tools.prepare_inputs` and checked
against the recorded SHA-256; they are not stored. The six V-keep masks (exact and hardware VS rule for the three
selections) depend on exact QK logits of the snapshot and are stored bit-packed (`vkeep_*.bits`, 32 KiB each, 1 bit
per KV head and token); `prepare_inputs` unpacks them into `build/replay/masks/`. The manifest also records the kept
token counts, the superset check, the per-head threshold bins b* (which set the VS scan time) and the error of each
mask on the snapshot. `rtl_vectors/vs/` holds the VS engine test vectors of the 16 KV groups of the GS selections
(FP32 logits, anchors, expected b* and keep bits). `experiments/replay/make_vkeep_masks.py` regenerates both from the
activation snapshot.

`calibration/host_topk.json` records the S-path host Top-k: an assumed 0.2 ms on a GPU, and medians measured with
`tools/bench/host_topk.cpp` on the development CPU. `area/extension_components.json` collects every mapped-cell and
SRAM component of `tools/area_model.py`: the 16-byte-port Dense scorers and the integrated POP score bank
(freshly mapped), the refresh scheduler, S-path staging, GS unit, VS engine and GS bank, and the eight CACTI macros.

Inputs are frozen for reproducibility. Run calibration tools into a new `outputs/` directory; changing a committed input requires a reviewed new measurement and manifest.
