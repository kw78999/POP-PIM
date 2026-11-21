"""Verify fixed BF16 replay size, finiteness, stable Top-k, and file hashes."""
import argparse
import json
from pathlib import Path
import numpy as np
from .common import ROOT, save, sha


def verify():
    score_path = ROOT / 'data/replay/scores_bf16.bin'
    index_path = ROOT / 'data/replay/topk_indices_i32.bin'
    assert score_path.stat().st_size == 32 * 32768 * 2
    assert index_path.stat().st_size == 32 * 4096 * 4
    bits = np.fromfile(score_path, dtype='<u2').reshape(32, 32768)
    scores = (bits.astype(np.uint32) << 16).view(np.float32)
    expected = np.fromfile(index_path, dtype='<i4').reshape(32, 4096)
    assert np.isfinite(scores).all(), 'Nonfinite replay score'
    ids = np.arange(32768)
    for head in range(32):
        actual = np.lexsort((ids, -scores[head]))[:4096]
        # The replay stores the selected set in ascending token-position order.
        np.testing.assert_array_equal(np.sort(actual), expected[head])
    result = dict(status='passed', heads=32, context=32768, budget=4096,
                  tie_rule='score descending, index ascending', files={
                  str(p.relative_to(ROOT)): sha(p) for p in [score_path, index_path]})
    manifest = ROOT / 'data/replay/manifest.json'
    if manifest.exists():
        declared = json.loads(manifest.read_text())
        assert result['files'] == declared['files'], 'Replay differs from release input'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = verify()
    if args.output:
        save(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
