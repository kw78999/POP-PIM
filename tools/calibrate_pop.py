"""Validate the two-context popcount BF16 scorer on a generated oracle corpus."""
import argparse
import csv
import json
import statistics
import struct
from pathlib import Path
import numpy as np
from .common import ROOT, BUILD, HF, hardfloat_sources, run, save, sha
from poppim.numeric.retrieval import GROUP, TEST_FRAME, counts, group_ops, coefficient_bits, add_bits, mul_bits


def generate(path, samples):
    rng = np.random.default_rng(7007)
    frames = np.zeros(samples, dtype=TEST_FRAME)
    for group in range(2):
        data = frames['group'][:, group]
        for name in ['qs','qm','ks','km']:
            data[name] = rng.integers(0, 2**64, samples, dtype=np.uint64)
        data['mask'] = np.uint64(2**64-1)
        for name in ['ql','dq']:
            data[name] = np.abs(rng.normal(size=samples)).astype(np.float32).view(np.uint32)
        for name in ['kl','dk']:
            data[name] = np.abs(rng.normal(size=samples)).astype(np.float16).view(np.uint16)
        data['norm'] = np.float32(1/np.sqrt(128)).view(np.uint32)
        # Include signed zero, subnormals, overflow, infinity and NaN conversions.
        data['ql'][:8] = [0,0x80000000,1,0x00800000,0x7f7fffff,0x7f800000,0xff800000,0x7fc00001]
        for row in data:
            row['counts'] = counts(row)
            row['expected'][:9] = group_ops(row, 'A')
    for i in range(samples):
        total = add_bits(int(frames['group']['expected'][i,0,8]), int(frames['group']['expected'][i,1,8]))
        frames['final'][i] = [total, mul_bits(total, coefficient_bits(frames['group'][i,0])[-1])]
    with path.open('wb') as output:
        output.write(struct.pack('<II', samples, 9))
        output.write(frames.tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=4096)
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 512:
        parser.error('use at least 512 samples')
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error('use a fresh output directory')
    out.mkdir(parents=True, exist_ok=True)
    vector = out / 'vectors.bin'
    generate(vector, args.samples)
    folder = BUILD / 'pop_rtl'
    folder.mkdir(parents=True, exist_ok=True)
    driver = (ROOT/'rtl/scoring/tests/score7_tb.cpp').read_text().replace('Vscore7_top','VPopScore')
    (folder/'driver.cpp').write_text(driver)
    sources = hardfloat_sources() + [ROOT/'rtl/primitives'/name for name in ['popcount32.sv','bitwise_count.sv']] + [ROOT/'rtl/scoring'/name for name in ['bf16_fma_pipeline.sv','bf16_conversions.sv','score7_group.sv','score7_top.sv']]
    run(['verilator','--cc','--exe','-Wno-fatal','-Wno-WIDTH','--top-module','score7_top',
         '--prefix','VPopScore','--Mdir',folder,'-I'+str(HF),'-I'+str(HF/'RISCV'),
         '-GCONTEXTS=2','-GARCH_B=0','-GRAW_BF16=0',*sources,folder/'driver.cpp'],folder/'verilator.log')
    run(['make','-C',folder,'-f','VPopScore.mk',f'-j{args.jobs}'],folder/'make.log')
    checks = []
    for mode in [0,1,2]:
        result = out / f'mode{mode}'
        result.mkdir()
        run([folder/'VPopScore',vector,result/'score',mode,args.samples],result/'run.log')
        stats = json.loads((result/'score_stats.json').read_text())
        assert stats['all_bit_exact'] and stats['outputs'] == args.samples
        if mode == 1:
            assert stats['output_stall_cycles'] > 0
        if mode == 2:
            assert stats['resets'] == 3
        frames = list(csv.DictReader((result/'score_frames.csv').open()))
        accepted = [int(row['accepted_cycle']) for row in frames[256:-8]]
        checks.append(dict(mode=mode,**stats,mean_II=statistics.mean(b-a for a,b in zip(accepted,accepted[1:]))))
    save(out/'checks.json',dict(samples=args.samples,vector_sha256=sha(vector),checks=checks,
                               scope='Generated descriptor and special-value corpus; not a model-quality measurement'))
    assert abs(checks[0]['mean_II']-22) < 0.05, 'Pop scorer calibration changed'
    print(f'Pop BF16 RTL checks passed: {out}',flush=True)


if __name__ == '__main__':
    main()
