"""Rerun retained SRAM estimator configurations with an explicitly selected CACTI.

``--extension-sizes`` instead derives the eight macros used by the GS/VS/CT area model (4 KiB to 512 KiB) from the
retained phase14 itrs-hp 4 KiB configuration, changing only the capacity, and compares them with
data/area/extension_components.json.
"""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
from .common import ROOT, BUILD, run, save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, help='Local CACTI built from the pinned source revision')
    parser.add_argument('--fetch', action='store_true', help='Explicitly fetch and build the pinned public CACTI source')
    parser.add_argument('--case', help='Optional configuration folder name, e.g. itrs-hp_65536')
    parser.add_argument('--extension-sizes', action='store_true', help='Reproduce the GS/VS/CT SRAM macros instead')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error('use a fresh output directory')
    if args.fetch and args.executable:
        parser.error('select --fetch or --executable')
    if args.fetch:
        metadata = json.loads((ROOT/'third_party/versions.json').read_text())['cacti']
        vendor = ROOT/'downloads/cacti'
        if vendor.exists():
            parser.error('downloads/cacti already exists; use its executable explicitly or a clean checkout')
        vendor.parent.mkdir(parents=True, exist_ok=True)
        run(['git','clone','--no-checkout',metadata['url'],vendor],BUILD/'logs/cacti_clone.log')
        run(['git','-C',vendor,'checkout','--detach',metadata['revision']],BUILD/'logs/cacti_checkout.log')
        run(['make','-C',vendor,'-j2'],BUILD/'logs/cacti_build.log')
        executable = vendor/'cacti'
    elif args.executable:
        executable = args.executable.resolve()
    else:
        parser.error('provide --executable or explicitly choose --fetch')
    if not executable.is_file():
        parser.error('CACTI executable not found')
    out.mkdir(parents=True, exist_ok=True)
    if args.extension_sizes:
        template = ROOT/'data/area/cacti/phase14/itrs-hp_4096/cache.cfg'
        recorded = json.loads((ROOT/'data/area/extension_components.json').read_text())['sram_cacti_itrs_hp']
        cases = []
        for size in recorded:
            config = out/'extension'/f'itrs-hp_{size}'/'template.cfg'
            config.parent.mkdir(parents=True, exist_ok=True)
            config.write_text(re.sub(r'-size \(bytes\) \d+', f'-size (bytes) {size}', template.read_text()))
            cases.append(config)
    else:
        cases = sorted((ROOT/'data/area/cacti').rglob('*.cfg'))
        if args.case:
            cases = [path for path in cases if path.parent.name == args.case]
    if not cases:
        parser.error('no matching SRAM configuration')
    results = []
    for config in cases:
        relative = config.relative_to(out) if args.extension_sizes else config.relative_to(ROOT/'data/area/cacti')
        folder = out/relative.parent
        folder.mkdir(parents=True, exist_ok=True)
        # CACTI loads process data relative to its executable/source directory.
        log = folder/'run.log'
        local_config = folder/'cache.cfg'
        shutil.copy2(config, local_config)
        with log.open('w') as output:
            subprocess.run([str(executable),'-infile',str(local_config)],cwd=executable.parent,
                           stdout=output,stderr=subprocess.STDOUT,check=True,timeout=120)
        text = log.read_text()
        dimensions = re.search(r'Cache height x width \(mm\):\s*([\d.eE+-]+)\s*x\s*([\d.eE+-]+)',text)
        if not dimensions:
            raise ValueError(f'No SRAM dimensions in {log}')
        results.append(dict(case=str(relative.parent),area_mm2=float(dimensions[1])*float(dimensions[2]),
                            config_sha256=sha(config),log_sha256=sha(log),executable_sha256=sha(executable)))
    if args.extension_sizes:
        for result in results:
            size = result['case'].split('_')[-1]
            result['recorded_area_mm2'] = recorded[size]['area_mm2']
            result['identical_area'] = abs(result['area_mm2'] - recorded[size]['area_mm2']) < 1e-12
    save(out/'models.json',results)
    if args.extension_sizes and not all(r['identical_area'] for r in results):
        raise SystemExit(f'Some SRAM areas differ from data/area/extension_components.json; see {out}/models.json')
    print(f'Reran {len(results)} SRAM configurations: {out}')


if __name__=='__main__':
    main()
