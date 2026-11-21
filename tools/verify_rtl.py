"""Verilator checks of the byte-frugal RTL blocks and the per-bank refresh scheduler.

* ``refresh``: rtl/refresh/pim_refresh_sched.sv, cycle by cycle against a transliteration of the simulator
  controller's per-bank refresh policy, over 11 parameter/stress cases (random bank activity);
* ``quarter``: the exact BF16 x0.25 unit of the GS unit against the BF16 FMA, all 65,536 inputs;
* ``gs``: rtl/gs/gs_shared_score.sv on the full 32K x 32-head replay with random input gaps and output
  back-pressure, bit-exact against the GS replay used by the simulator;
* ``vs``: rtl/vs/vs_mass_threshold.sv against the bit-exact software reference vectors in
  data/rtl_vectors/vs (b* per head, keep bit per rank, scan cycles).

Requires Verilator and a C++ compiler (the development toolchain: Verilator 4.028, GCC 10).
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

from .common import BUILD, HF, ROOT, run, save
from . import prepare_inputs

FLAGS = ['--cc', '--exe', '-Wno-fatal', '-Wno-WIDTH', '-Wno-UNUSED', '-Wno-PINCONNECTEMPTY', '-O2',
         f'-I{HF}', f'-I{HF / "RISCV"}']
HARDFLOAT = [HF / p for p in ['HardFloat_primitives.v', 'HardFloat_rawFN.v', 'isSigNaNRecFN.v', 'fNToRecFN.v',
             'recFNToFN.v', 'iNToRecFN.v', 'recFNToIN.v', 'mulAddRecFN.v', 'compareRecFN.v', 'RISCV/HardFloat_specialize.v']]
FMA = ROOT / 'rtl/scoring/bf16_fma_pipeline.sv'
GS = ROOT / 'rtl/gs/gs_shared_score.sv'
# name, NREFI, P (postpone), U (pull-in), H (headroom), LOW, ROUND, seed, cycles, heavy traffic
REFRESH_CASES = [
    ('h2_round', 5070, 8, 8, 2, 5, 1, 11, 400000, False), ('h3_round', 5070, 8, 8, 3, 4, 1, 12, 400000, False),
    ('h4_round', 5070, 8, 8, 4, 3, 1, 13, 400000, False), ('h6_round', 5070, 8, 8, 6, 1, 1, 14, 400000, False),
    ('h2_relaxed', 5070, 8, 8, 2, 0, 0, 15, 400000, False), ('h6_relaxed', 5070, 8, 8, 6, 0, 0, 16, 400000, False),
    ('short_nrefi', 320, 8, 8, 2, 5, 1, 17, 400000, False), ('tight_window', 320, 3, 1, 2, 0, 1, 18, 400000, False),
    ('heavy_h2_round', 5070, 8, 8, 2, 5, 1, 19, 600000, True), ('heavy_h6_round', 5070, 8, 8, 6, 1, 1, 20, 600000, True),
    ('heavy_h2_relaxed', 5070, 8, 8, 2, 0, 0, 21, 600000, True)]


def compiler():
    return os.environ.get('CXX', 'g++')


def verilate(name, top, sources, extra=(), cflags=''):
    folder = BUILD / 'rtl_verify' / name
    folder.mkdir(parents=True, exist_ok=True)
    command = ['verilator', *FLAGS, *extra, '--top-module', top, '--Mdir', folder, *map(str, sources)]
    if cflags:
        command[1:1] = ['-CFLAGS', cflags]
    run(command, folder / 'verilate.log')
    cxx = compiler()
    run(['make', '-C', folder, '-f', f'V{top}.mk', f'CXX={cxx}', f'LINK={cxx}', '-j2'], folder / 'make.log')
    return folder / f'V{top}'


def last_json(path):
    return json.loads([line for line in path.read_text().splitlines() if line.startswith('{')][-1])


def refresh_case(case):
    name, nrefi, p, u, h, low, rnd, seed, cycles, heavy = case
    defines = f'-O2 -DP_NREFI={nrefi} -DP_P={p} -DP_U={u} -DP_H={h} -DP_LOW={low} -DP_ROUND={rnd} -DP_SEED={seed} -DP_CYCLES={cycles}'
    exe = verilate(f'refresh_{name}', 'pim_refresh_sched',
                   [ROOT / 'rtl/refresh/pim_refresh_sched.sv', ROOT / 'rtl/refresh/tests/tb_pim_refresh_sched.cpp'],
                   [f'-GNREFI={nrefi}', f'-GP={p}', f'-GU={u}', f'-GH={h}', f'-GLOW={low}', f'-GROUND={rnd}'],
                   defines + (' -DP_HEAVY' if heavy else ''))
    log = exe.parent / 'result.json'
    run([exe], log)
    result = last_json(log)
    result['case'] = name
    assert result['passed'], f'refresh scheduler mismatch in {name}'
    return result


def test_refresh(jobs):
    with ThreadPoolExecutor(jobs) as pool:
        results = list(pool.map(refresh_case, REFRESH_CASES))
    return dict(cases=len(results), passed=all(r['passed'] for r in results), results=results)


def test_quarter():
    exe = verilate('bf16_quarter', 'quarter_check', [*HARDFLOAT, FMA, GS, ROOT / 'rtl/gs/tests/quarter_check.sv',
                                                    ROOT / 'rtl/gs/tests/tb_quarter.cpp'])
    run([exe], exe.parent / 'result.json')
    result = last_json(exe.parent / 'result.json')
    assert result['inputs'] == 65536 and result['mismatch'] == 0
    return result


def test_gs():
    inputs = prepare_inputs.prepare(quiet=True)['inputs']
    exe = verilate('gs_shared_score', 'gs_shared_score', [*HARDFLOAT, FMA, GS, ROOT / 'rtl/gs/tests/tb_gs.cpp'])
    run([exe, inputs['base']['scores'], inputs['gs4096']['scores']], exe.parent / 'result.json')
    result = last_json(exe.parent / 'result.json')
    assert result['mismatch'] == 0 and result['pending'] == 0 and result['bit_exact'] == 8 * 32768
    return result


def test_vs():
    exe = verilate('vs_mass_threshold', 'vs_mass_threshold',
                   [*HARDFLOAT, ROOT / 'rtl/vs/vs_mass_threshold.sv', ROOT / 'rtl/vs/tests/tb_vs.cpp'],
                   [f'-I{ROOT / "rtl/vs"}'])
    vectors = sorted((ROOT / 'data/rtl_vectors/vs').glob('*.bin'))
    run([exe, *vectors], exe.parent / 'result.json')
    lines = (exe.parent / 'result.json').read_text().splitlines()
    result = last_json(exe.parent / 'result.json')
    result['scan_cycles'] = {line.split()[0].rsplit('/', 1)[-1]: int(line.split('scan_cycles')[1].split()[0])
                             for line in lines if 'scan_cycles' in line}
    assert result['kv_groups'] == len(vectors) == 16 and result['fail'] == 0
    assert result['keep_bits_match'] == result['keep_bits_checked']
    return result


TESTS = dict(refresh=test_refresh, quarter=test_quarter, gs=test_gs, vs=test_vs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tests', nargs='+', choices=list(TESTS), default=list(TESTS))
    parser.add_argument('--jobs', type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args(argv)
    results = {}
    for name in args.tests:
        print(f'Running RTL check: {name}', flush=True)
        results[name] = TESTS[name](args.jobs) if name == 'refresh' else TESTS[name]()
        summary = {k: v for k, v in results[name].items() if k not in ('results', 'scan_cycles')}
        print(f'  passed: {json.dumps(summary)}', flush=True)
    save(BUILD / 'validation/rtl_extensions.json', results)
    print(f'All {len(results)} RTL checks passed; details in {BUILD / "validation/rtl_extensions.json"}')


if __name__ == '__main__':
    main()
