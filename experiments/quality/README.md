# Model quality of the selection variants (optional)

`evaluate.py` runs Llama-3.1-8B on CPU over PG19 at 32K context with teacher-forced decoding. Layers 0 and 1 stay
dense; the other 30 layers attend only to the selected tokens (the current token is always attended). Methods:

| Method | Meaning | Hardware configuration |
|---|---|---|
| `full` | dense attention | Dense |
| `ph:K` | per-query-head PopKV Top-(K−1) + current; KV gather = union of the 4 heads | POP-PIM (`ph:4096`) |
| `gs:K` | mean of the 4 heads' approximate logits per KV head, one Top-(K−1) | GS |
| `gs:K:vs:δ` | GS + two-phase K→V with V skipped beyond 1 − δ of each head's selected mass (exact rule) | POP-PIM-S (`gs:4096:vs:0.005`), POP-PIM-Q (`gs:6144:vs:0.002`) |

Scores use the FP32 PopKV reconstruction (token/dimension groups of 64), so they differ from the BF16 hardware DAG
only in rounding. The run also records, every 32nd step of the full-attention pass, the relative L2 error of each
sparse layer's attention output against dense attention and the selected attention mass for a grid of methods.

## Run

```bash
pip install torch transformers accelerate
python experiments/quality/evaluate.py --pg19-dir /path/to/pg19/test --out outputs/quality \
    --blocks 0 --tokens 64
python experiments/quality/summarize.py --run-dir outputs/quality --output-dir outputs/quality/summary
```

`--model` takes a Hugging Face id (default `meta-llama/Llama-3.1-8B`, gated: accept its license first) or a local
path. `--pg19-dir` is a folder of PG19 test-split books as `*.txt`; consecutive books are concatenated into
32,768 + 64-token blocks. In the recorded run (8 threads, more than 32 GB of memory) one dense prefill per block took
17–19 minutes; the methods then share its cache, 1–2 minutes each for 64 tokens, except `full` with its metric
pass (about 35 minutes). The run appends to `metrics.csv` and `nll.csv` and skips methods a block already finished.

## Recorded result

`results/quality/` holds the summary of the development run (`--blocks 0,1,2,3`). Every method finished block 0;
block 1 finished only `full`, so perplexity covers the 64 tokens of block 0, while the attention-output error covers
4 sampled steps × 30 layers = 120 samples per method (blocks 0 and 1). Perplexity differences to full attention all
lie inside their 95% paired-bootstrap intervals; treat them as "no measurable degradation on a small sample", not as
a quality ranking.
