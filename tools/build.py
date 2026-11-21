"""Build the vendored memory model, the Dense/POP architectural adapters and their C++/RTL checks offline."""
from __future__ import annotations
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from .common import BUILD, ROOT, env, run, save, sha, verilator_root


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=int, default=2)
    args = parser.parse_args(argv)
    if sys.version_info < (3, 10):
        parser.error('Python 3.10 or newer is required; use the activated reproduction environment')
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    ram = ROOT / 'third_party/ramulator'
    vendor = BUILD / 'ramulator'
    cxx = os.environ.get('CXX', 'g++')
    probe = subprocess.run([cxx, '-std=c++20', '-x', 'c++', '-fsyntax-only', '-'],
                           input='#include <ranges>\nint main(){}\n', text=True, env=env(),
                           capture_output=True)
    if probe.returncode:
        parser.error('CXX must support C++20 <ranges>; select GCC 10+ or a compatible compiler')
    run(['cmake', '-S', ram, '-B', vendor, f'-DCMAKE_CXX_COMPILER={cxx}'], BUILD / 'logs/cmake.log')
    run(['cmake', '--build', vendor, '--target', 'ramulator', '-j', args.jobs], BUILD / 'logs/ramulator.log')
    gqa = BUILD / 'gqa'
    gqa.mkdir(parents=True, exist_ok=True)
    run(['verilator', '--cc', '--top-module', 'gqa4_frontend', '--prefix', 'GQA1', '--Mdir', gqa,
         '-Wno-fatal', '-GBUFFERS=1', *sorted((ROOT / 'rtl/gqa').glob('*.sv'))], BUILD / 'logs/gqa_verilate.log')
    run(['make', '-C', gqa, '-f', 'GQA1.mk', f'-j{args.jobs}', 'OPT_FAST=-O2', 'OPT_SLOW=-O2'], BUILD / 'logs/gqa_make.log')
    vr = verilator_root()
    includes = ['-I' + str(p) for p in [gqa, vr / 'include', ram / 'src',
                ram / 'ext/yaml-cpp/include', ram / 'ext/spdlog/include', ram / 'ext/argparse/include']]
    runtime = [vr / 'include/verilated.cpp']
    version = subprocess.check_output(['verilator', '--version'], env=env(), text=True)
    major = int(re.search(r'Verilator\s+(\d+)', version).group(1))
    if major >= 5:
        runtime.append(vr / 'include/verilated_threads.cpp')
    library = ['-L' + str(vendor / 'lib'), '-Wl,-rpath,' + str(vendor / 'lib'), '-lramulator', vendor / 'yaml-cpp/libyaml-cpp.a', '-pthread']
    bindir = BUILD / 'bin'
    bindir.mkdir(parents=True, exist_ok=True)
    common = ROOT / 'simulator/common'
    for design in ['dense', 'pop']:
        print(f'Building {design} adapter', flush=True)
        run([cxx, '-O2', '-std=c++20', *includes, common / 'hbm3_scoped.cpp',
             ROOT / f'simulator/{design}/frontend.cpp', common / 'controller.cpp', ram / 'src/main.cpp',
             gqa / 'GQA1__ALL.a', *runtime, *library, '-o', bindir / f'poppim_{design}'], BUILD / f'logs/{design}.log')
    run([cxx, '-O2', '-std=c++20', *includes, common / 'hbm3_scoped.cpp',
         ROOT / 'simulator/tests/refresh.cpp', *library, '-o', bindir / 'refresh_test'], BUILD / 'logs/refresh_build.log')
    run([bindir / 'refresh_test'], BUILD / 'validation/refresh.jsonl')
    run([cxx, '-O2', '-std=c++20', *includes, common / 'hbm3_scoped.cpp',
         ROOT / 'simulator/tests/refresh_single_bank.cpp', *library, '-o', bindir / 'refresh_single_bank'],
        BUILD / 'logs/refresh_single_bank_build.log')
    run([bindir / 'refresh_single_bank'], BUILD / 'validation/refresh_single_bank.jsonl')
    run([cxx, '-O2', '-std=c++17', *includes, ROOT / 'simulator/tests/gqa.cpp',
         gqa / 'GQA1__ALL.a', *runtime, '-pthread', '-o', bindir / 'gqa_test'], BUILD / 'logs/gqa_test_build.log')
    run([bindir / 'gqa_test'], BUILD / 'validation/gqa.json')
    save(BUILD / 'build_manifest.json', {'binaries': {p.name: sha(p) for p in bindir.iterdir() if p.is_file()},
         'ramulator_sha256': sha(vendor / 'lib/libramulator.so'),
         'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in (ROOT / 'simulator').rglob('*') if p.is_file()}})
    print(f"Build and refresh/GQA checks passed: {bindir}", flush=True)


if __name__ == '__main__':
    main()
