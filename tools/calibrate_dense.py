"""Regenerate finite-tile RTL timing with numerical, stall, and reset checks."""
import argparse
import csv
import json
import statistics
import struct
from pathlib import Path
import numpy as np
from .common import ROOT, BUILD, HF, dense_sources, run, save, sha
from poppim.numeric.dense import dense, NORM, bf16_bits_from_fp32


def vectors(lanes, folder):
    path = folder / f'D{lanes}.bin'
    rng = np.random.default_rng(9042)
    n = 9216
    q = bf16_bits_from_fp32(rng.normal(size=(n, 128)).astype(np.float16).astype(np.float32))
    k = bf16_bits_from_fp32(rng.normal(size=(n, 128)).astype(np.float16).astype(np.float32))
    edge = np.array([0,0x8000,1,0x8001,0x7f,0x80,0x81,0x3f7f,0x3f80,0x3f81,0xbf80,0x7f7f,0xff7f,0x7f80,0xff80,0x7fc0,0x7f81], np.uint16)
    q[8192:] = rng.choice(edge, (1024,128))
    k[8192:] = rng.choice(edge, (1024,128))
    q[8192:8256] = 0x8000
    k[8192:8256] = 0x3f80
    q[8256:8320] = np.where(np.arange(128)%2, 0xbf80, 0x3f80)
    k[8256:8320] = 0x3f80
    with path.open('wb') as handle:
        handle.write(struct.pack('<IIIH', n, lanes, 128+4*lanes, NORM))
        for a, b in zip(q, k):
            handle.write(a.astype('<u2').tobytes())
            handle.write(b.astype('<u2').tobytes())
            handle.write(np.asarray(dense(a, b, lanes), dtype='<u2').tobytes())
    return path


def build(lanes, jobs):
    folder = BUILD / f'dense_rtl/D{lanes}'
    folder.mkdir(parents=True, exist_ok=True)
    prefix = f'VDense{lanes}'
    driver = (ROOT / 'rtl/dense/tests/dense_tb.cpp').read_text().replace('Vdense_score', prefix)
    driver = driver.replace(prefix+' d;', prefix+' d;const unsigned PORT_L=std::min(L,8u);')
    driver = driver.replace('beat+1)*L','beat+1)*PORT_L').replace('beat*L','beat*PORT_L')
    driver = driver.replace('l<L;l++)if(beat*PORT_L+l<128)','l<PORT_L;l++)if(beat*PORT_L+l<128)')
    driver = driver.replace('d.in_valid=next<n&&(ihold||mode==0||((rng>>20)&7)!=0);',
        'd.in_valid=next<n&&(mode!=3||next/16==outputs/16)&&(ihold||mode==0||mode==3||((rng>>20)&7)!=0);')
    driver = driver.replace('d.out_ready=mode==0||','d.out_ready=mode==0||mode==3||')
    assert 'PORT_L' in driver and 'next/16==outputs/16' in driver
    (folder / 'driver.cpp').write_text(driver)
    run(['verilator','--cc','--exe','-Wno-fatal','-Wno-WIDTH','--top-module','dense_score_port16',
         '--prefix',prefix,'--Mdir',folder,'-I'+str(HF),'-I'+str(HF/'RISCV'),f'-GLANES={lanes}',
         '-CFLAGS','-std=c++17 -O2',*dense_sources(),ROOT/'rtl/dense/dense_score_port16.sv',folder/'driver.cpp'],folder/'verilator.log')
    run(['make','-C',folder,'-f',prefix+'.mk',f'-j{jobs}'],folder/'make.log')
    return folder / prefix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lanes', type=int, nargs='+', choices=[1,2,4,8,16,32], default=[1,2,4,8])
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error('use a fresh output directory')
    out.mkdir(parents=True, exist_ok=True)
    rows, schedules = [], {}
    reference = json.loads((ROOT/'data/calibration/tile_schedules.json').read_text())
    for lanes in args.lanes:
        print(f'Validating Dense D{lanes}', flush=True)
        binary = build(lanes, args.jobs)
        vector = vectors(lanes, out)
        for mode, limit in [(0,9216),(1,2048),(2,1024),(3,1024)]:
            folder = out / f'D{lanes}_{mode}'
            folder.mkdir()
            run([binary,vector,folder/'score',mode,limit],folder/'run.log')
            stats = json.loads((folder/'score_stats.json').read_text())
            assert stats['bit_exact'] and stats['outputs'] == limit
            frames = list(csv.DictReader((folder/'score_frames.csv').open()))
            window = frames[128:-8]
            accepted = [int(frame['accepted']) for frame in window]
            rows.append(dict(design=f'D{lanes}',mode=mode,**stats,
                             score_II=statistics.mean(b-a for a,b in zip(accepted,accepted[1:])),
                             vectors_sha256=sha(vector),binary_sha256=sha(binary)))
            if mode == 3:
                tiles = []
                for start in range(0,len(frames),16):
                    tile = frames[start:start+16]
                    origin = int(tile[0]['start'])
                    tiles.append(dict(start=[int(f['start'])-origin for f in tile],ready=[int(f['valid'])-origin for f in tile]))
                assert len(tiles) == 64 and all(tile==tiles[0] for tile in tiles)
                assert all(tiles[0][key] == reference[f'D{lanes}'][key] for key in ['start','ready']), 'Calibration changed; review before replacing reference inputs'
                schedules[f'D{lanes}'] = dict(**tiles[0], validated_tiles=len(tiles))
        save(out/'rtl_checks.json',rows)
        save(out/'tile_schedules.json',schedules)
    print(f'Numerical, backpressure, reset and tile checks passed: {out}',flush=True)


if __name__ == '__main__':
    main()
