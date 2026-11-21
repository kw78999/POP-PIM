# Experiments

| Path | What it does | Needs |
|---|---|---|
| `run_comparison.py` | Synthetic CPU comparison of the functional encoders, scorers and selectors (shared scores, bank mapping, selection placement) | Python only |
| [`quality/`](quality/README.md) | Attention-output error and perplexity of the POP-PIM selection variants on Llama-3.1-8B, PG19, 32K context | torch, transformers, model weights, PG19, many CPU hours |
| [`replay/`](replay/README.md) | Regenerates the activation-derived V-keep masks and VS RTL test vectors and checks them against the release | torch, the layer-2 activation snapshot |

The architectural simulations are run with `tools.reproduce` / `tools.run_simulation`, not from this folder.
