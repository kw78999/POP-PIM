"""Run a named experiment suite, audit it, compare it with the committed reference and rebuild its tables.

    python -m tools.reproduce --suite quick      # 3 cases, about a minute on four cores
    python -m tools.reproduce --suite main       # 28 cases, Dense vs POP-PIM/-Q/-S at four bandwidths
    python -m tools.reproduce --suite ablation   # 121 cases, GS/VS/CT ablation and host-link sweep
    python -m tools.reproduce --suite refresh    # 45 cases, refresh policies

Every raw command trace is audited right after its run (tools.audit_simulation). Traces are kept only for
``quick`` unless ``--keep-traces`` is given; a 32K Dense trace is about 100 MB. Results go to
``outputs/<suite>`` by default; the directory must not exist yet.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .common import BUILD, ROOT, save
from .rebuild_tables import rebuild
from .run_simulation import run_cases
from .suites import REFERENCE, SUITES, cases

# Integer counters must match exactly; this tolerance only applies to floating-point metrics.
REL_TOL = 1e-12


def compare(records, reference_path):
    """Compare new metrics with the committed reference, case by case."""
    if not reference_path.exists():
        return dict(status='no reference', reference=str(reference_path))
    reference = {r['name']: r for r in json.loads(reference_path.read_text())}
    rows, identical, missing = [], 0, []
    for record in records:
        ref = reference.get(record['name'])
        if ref is None:
            missing.append(record['name'])
            continue
        new, old = record['metrics'], ref['metrics']
        diffs = {}
        for key in sorted(set(new) | set(old)):
            a, b = new.get(key), old.get(key)
            if a == b:
                continue
            if isinstance(a, float) and isinstance(b, float) and abs(a - b) <= REL_TOL * max(abs(a), abs(b)):
                continue
            diffs[key] = [a, b]
        same = not diffs
        identical += same
        rows.append(dict(name=record['name'], identical=same,
                         latency_ms=new['attention_output_ps'] / 1e9, reference_latency_ms=old['attention_output_ps'] / 1e9,
                         differing_metrics=diffs))
    status = 'identical' if identical == len(rows) and not missing else 'differs'
    return dict(status=status, reference=str(reference_path.relative_to(ROOT)), compared=len(rows),
                identical=identical, missing_from_reference=missing, cases=rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--suite', choices=list(SUITES), default='quick')
    parser.add_argument('--output-dir', type=Path, help='default: outputs/<suite>')
    parser.add_argument('--jobs', type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument('--keep-traces', action='store_true', help='keep every raw command trace')
    args = parser.parse_args(argv)
    out = (args.output_dir or ROOT / 'outputs' / args.suite).resolve()
    if out.exists() and any(out.iterdir()):
        parser.error(f'{out} is not empty; choose a fresh --output-dir')
    if not (BUILD / 'bin/poppim_pop').is_file():
        parser.error('Simulator binaries are missing; run python -m tools.build first')
    selected = cases(args.suite)
    print(f'Suite {args.suite}: {len(selected)} simulations with {args.jobs} parallel jobs -> {out}', flush=True)
    records = run_cases(selected, out, args.jobs, keep_traces=args.keep_traces or args.suite == 'quick',
                        audit_traces=True)
    result = compare(records, ROOT / 'results' / REFERENCE[args.suite] / 'runs.json')
    save(out / 'comparison_to_reference.json', result)
    rebuild(args.suite, out / 'tables', out)
    print('Tables report latency including the VS scan of the hardware VS rule (simulated time + scan).')
    print((out / 'tables' / f'{args.suite}.md').read_text())
    if result['status'] == 'identical':
        print(f"All {result['compared']} cases are identical to {result['reference']}. Tables: {out / 'tables'}")
    elif result['status'] == 'no reference':
        print(f'No committed reference for this suite. Tables: {out / "tables"}')
    else:
        print(f"{result['compared'] - result['identical']} of {result['compared']} cases differ from "
              f"{result['reference']}; see {out / 'comparison_to_reference.json'}. A different compiler or "
              'library version can change scheduling details; report the toolchain with any difference.')
        raise SystemExit(1)


if __name__ == '__main__':
    main()
