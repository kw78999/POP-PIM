# Activation-derived replay data

The simulator replay (`data/replay/scores_bf16.bin`, `topk_indices_i32.bin`) holds BF16 PopKV scores and Top-4,096
indices of one Llama-3.1-8B layer-2 decode step at 32K PG19 context. The GS variants are derived from those scores
alone (`python -m tools.prepare_inputs`). The V-keep masks of VS (`data/replay/variants/*.bits`) and the VS engine
test vectors (`data/rtl_vectors/vs/`) also need the exact QK logits of that step, so they are committed as derived
data.

`make_vkeep_masks.py` recomputes them from the activation snapshot (`q`, `k`, `v` tensors of that step) and compares
every mask and vector byte for byte with the release:

```bash
pip install torch            # CPU build is enough
python -m tools.prepare_inputs
python experiments/replay/make_vkeep_masks.py --activation /path/to/latency_activation.pt --output-dir outputs/vkeep
```

The snapshot (128 MiB) is produced by the [PopKV](https://github.com/kw78999/PopKV) GPU PG19 latency run and is not
distributed in this repository. The hardware rule is evaluated with exact rational FP32 rounding; the whole run takes
under a minute on a CPU.
