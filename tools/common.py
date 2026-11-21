from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
BUILD = Path(os.environ.get('POPPIM_BUILD_DIR', ROOT / 'build')).resolve()
HF = ROOT / 'third_party/hardfloat/source'
LIB = Path(os.environ.get('POPPIM_LIBERTY', ROOT / 'third_party/nangate45/NangateOpenCellLibrary_typical.lib'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def env():
    values = os.environ.copy()
    values.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    # Optional user-supplied toolchain. No path to the development workspace is assumed.
    if prefix := values.get('POPPIM_TOOLCHAIN_ROOT'):
        prefix = Path(prefix).resolve()
        values['PATH'] = str(prefix / 'usr/bin') + os.pathsep + values.get('PATH', '')
        values['LD_LIBRARY_PATH'] = str(prefix / 'usr/lib/x86_64-linux-gnu') + os.pathsep + values.get('LD_LIBRARY_PATH', '')
        values['VERILATOR_ROOT'] = str(prefix / 'usr/share/verilator')
        values['TCL_LIBRARY'] = str(prefix / 'usr/share/tcltk/tcl8.6')
    return values


def run(command, log, timeout=3600):
    command = list(map(str, command))
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log.open('w') as output:
        try:
            subprocess.run(command, cwd=ROOT, env=env(), stdout=output,
                           stderr=subprocess.STDOUT, check=True, timeout=timeout)
        except subprocess.CalledProcessError:
            raise RuntimeError(f'Command failed; see {log}') from None
    save(str(log) + '.command.json', {'command': command, 'seconds': time.monotonic() - started})


def hardfloat_sources():
    return [HF / p for p in ['HardFloat_primitives.v', 'HardFloat_rawFN.v', 'isSigNaNRecFN.v',
            'fNToRecFN.v', 'recFNToFN.v', 'iNToRecFN.v', 'recFNToRecFN.v',
            'mulAddRecFN.v', 'RISCV/HardFloat_specialize.v']]


def dense_sources():
    return hardfloat_sources() + [ROOT / 'rtl/scoring/bf16_fma_pipeline.sv', ROOT / 'rtl/dense/dense_score.sv']


def verilator_root():
    result = subprocess.check_output(['verilator', '-V'], env=env(), text=True)
    import re
    matches = re.findall(r'^\s*VERILATOR_ROOT\s*=\s*(.+?)\s*$', result, re.M)
    if not matches:
        raise RuntimeError('Cannot discover VERILATOR_ROOT from verilator -V')
    return Path(env().get('VERILATOR_ROOT') or matches[-1])
