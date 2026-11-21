"""Measure the S-path host Top-k on this machine (8 threads) for the three replay shapes.

Compiles tools/bench/host_topk.cpp, checks the selected membership against the replay, and prints medians
next to the values recorded in data/calibration/host_topk.json. The recorded values are not changed; pass a
new value to ``tools.run_simulation --host-topk-ms`` to simulate with your own measurement.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .common import BUILD, ROOT, run, save
from . import prepare_inputs

SHAPES = [('base', 4096, 1), ('gs4096', 4096, 4), ('gs6144', 6144, 4)]   # input, K, row stride (32 or 8 rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=BUILD / 'validation/host_topk.json')
    args = parser.parse_args(argv)
    exe = BUILD / 'bin/host_topk'
    exe.parent.mkdir(parents=True, exist_ok=True)
    run([os.environ.get('CXX', 'g++'), '-O2', '-std=c++17', '-pthread', ROOT / 'tools/bench/host_topk.cpp', '-o', exe],
        BUILD / 'logs/host_topk_build.log')
    inputs = prepare_inputs.prepare(quiet=True)['inputs']
    recorded = json.loads((ROOT / 'data/calibration/host_topk.json').read_text())['cpu_measured']
    results = []
    for name, k, stride in SHAPES:
        log = BUILD / f'logs/host_topk_{name}.json'
        run([exe, inputs[name]['scores'], inputs[name]['expected'], 32768, k, stride], log)
        result = json.loads(log.read_text().splitlines()[-1])
        ref = next(r for r in recorded if (r['rows'], r['K']) == (result['rows'], k))
        result['recorded_median_ms'] = ref['median_ms']
        results.append(result)
        print(f"{result['rows']:2d} rows x Top-{k}: median {result['median_ms']:.3f} ms "
              f"(recorded {ref['median_ms']:.3f} ms on the development host)", flush=True)
    save(args.output, results)


if __name__ == '__main__':
    main()
