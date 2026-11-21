"""Mapped-cell synthesis and pre-layout STA of the blocks added by GS, VS, CT and per-bank refresh.

Same flow as tools.synthesize_dense/synthesize_pop: Yosys + ABC to the pinned Nangate45 typical Liberty
(BUF_X1 drive, 5 fF load, ``synth -noshare``), cell areas summed from the Liberty, OpenSTA with 50 ps clock
uncertainty and 100 ps I/O delays. Results are compared with data/area/extension_components.json.

    python -m tools.fetch_liberty
    python -m tools.synthesize_extensions --output-dir outputs/ppa_extensions
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .common import HF, LIB, ROOT, env, hardfloat_sources, run, save, sha

HFS = hardfloat_sources() + [HF / 'recFNToIN.v', HF / 'compareRecFN.v']
FMA = ROOT / 'rtl/scoring/bf16_fma_pipeline.sv'
REFRESH = ROOT / 'rtl/refresh/pim_refresh_sched.sv'
REFRESH_VARIANTS = {'round_H2': dict(NREFI=5070, P=8, U=8, H=2, LOW=5, ROUND=1),
                    'round_H6': dict(NREFI=5070, P=8, U=8, H=6, LOW=1, ROUND=1),
                    'relaxed_H2': dict(NREFI=5070, P=8, U=8, H=2, LOW=0, ROUND=0)}


def cells():
    table = {}
    for body in re.split(r'\n\s*cell\s*\(', LIB.read_text())[1:]:
        name = body.split(')', 1)[0].strip().strip('"')
        match = re.search(r'\barea\s*:\s*([\d.]+)', body)
        if match:
            table[name] = (float(match[1]), bool(re.search(r'\bff\s*\(', body)))
    return table


def abc():
    return shutil.which('berkeley-abc', path=env()['PATH']) or shutil.which('yosys-abc', path=env()['PATH'])


def sta(folder, top, netlist, mhz, inputs='[delete_from_list [all_inputs] [get_ports clk]]'):
    script = folder / f'timing_{mhz:.3f}MHz.tcl'
    script.write_text(f'''read_liberty {LIB}
read_verilog {netlist}
link_design {top}
create_clock -name clk -period {1000 / mhz:.9f} [get_ports clk]
set_clock_uncertainty 0.05 [get_clocks clk]
set_input_delay 0.1 -clock clk {inputs}
set_output_delay 0.1 -clock clk [all_outputs]
set_load 5 [all_outputs]
report_checks -path_delay max -digits 6
report_worst_slack -digits 6
exit
''')
    log = script.with_suffix('.log')
    run(['sta', '-exit', script], log, 1800)
    text = log.read_text()
    assert 'Error:' not in text, text[-800:]
    return float(re.findall(r'worst slack\s+(-?[\d.]+)', text)[-1])


def fmax(folder, top, netlist, resolution):
    lo, hi = 25.0, 2000.0
    while hi - lo > resolution:
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if sta(folder, top, netlist, mid) >= 0 else (lo, mid)
    return lo, hi


def normalize(folder):
    """Split concatenated alias assignments for OpenSTA."""
    out = []
    for line in (folder / 'mapped.v').read_text().splitlines():
        match = re.match(r'\s*assign\s+\{(.+)\}\s*=\s*\{(.+)\};', line)
        if match:
            left = [x.strip() for x in match[1].split(',')]
            right = [x.strip() for x in match[2].split(',')]
            out += [f'  assign {a}  = {b} ;' for a, b in zip(left, right)]
        else:
            out.append(line)
    (folder / 'mapped_sta.v').write_text('\n'.join(out) + '\n')
    return folder / 'mapped_sta.v'


def yosys(folder, top, read, params=None, flatten_first=False, check=True, post_abc='clean\nflatten\nclean -purge',
          final_clean=True):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'abc.constr').write_text('set_driving_cell BUF_X1\nset_load 5\n')
    chparam = f"chparam {' '.join(f'-set {k} {v}' for k, v in params.items())} {top}\n" if params else ''
    (folder / 'synth.ys').write_text(f'''{read}
{chparam}hierarchy {'-check ' if check else ''}-top {top}
proc
opt
memory_dff
memory_collect
memory_map
opt
synth -noshare -top {top}
{'flatten' + chr(10) + 'opt' + chr(10) if flatten_first else ''}dfflibmap -liberty {LIB}
abc -exe {abc()} -liberty {LIB} -constr {folder}/abc.constr
{post_abc}
splitnets -ports
{'clean' + chr(10) if final_clean else ''}stat -liberty {LIB}
write_json {folder}/mapped.json
write_verilog -noattr -noexpr {folder}/mapped.v
''')
    run(['yosys', '-s', folder / 'synth.ys'], folder / 'yosys.log', 7200)
    module = json.loads((folder / 'mapped.json').read_text())['modules'][top]
    table = cells()
    assert all(c['type'] in table for c in module['cells'].values()), 'blackbox or unmapped cell'
    return dict(cell_area_um2=sum(table[c['type']][0] for c in module['cells'].values()),
                flip_flops=int(sum(table[c['type']][1] for c in module['cells'].values())), cells=len(module['cells']))


def logic_block(out, name, top, sources, params=None):
    folder = out / name
    read = f'read_verilog -sv -I{HF} -I{HF / "RISCV"} -I{ROOT / "rtl/vs"} ' + ' '.join(map(str, sources))
    result = dict(name=name, top=top, params=params or {}, **yosys(folder, top, read, params))
    net = normalize(folder)
    result.update(slack_125MHz_ns=sta(folder, top, net, 125.0), slack_250MHz_ns=sta(folder, top, net, 250.0))
    result['fmax_pass_MHz'], result['fmax_fail_MHz'] = fmax(folder, top, net, 2.0)
    return result


def refresh_scheduler(out, name):
    folder = out / f'refresh_{name}'
    result = dict(name=name, params=REFRESH_VARIANTS[name],
                  **yosys(folder, 'pim_refresh_sched', f'read_verilog -sv {REFRESH}', REFRESH_VARIANTS[name], check=False))
    net = folder / 'mapped.v'
    result['fmax_pass_MHz'], result['fmax_fail_MHz'] = fmax(folder, 'pim_refresh_sched', net, 1.0)
    result.update(slack_125MHz_ns=sta(folder, 'pim_refresh_sched', net, 125.0),
                  slack_250MHz_ns=sta(folder, 'pim_refresh_sched', net, 250.0))
    return result


def s_path_staging(out):
    folder = out / 's_path_staging'
    src = ROOT / 'rtl/transport/score_stream_interface.sv'
    result = dict(name='s_path_staging', **yosys(folder, 'score_stream_interface', f'read_verilog -sv {src}', check=False))
    result['slack_250MHz_ns'] = sta(folder, 'score_stream_interface', folder / 'mapped.v', 250.0)
    return result


def bank_gs(out):
    folder = out / 'bank_gs'
    sources = (hardfloat_sources() + [ROOT / 'rtl/primitives' / n for n in ['popcount32.sv', 'bitwise_count.sv']]
               + [ROOT / 'rtl/scoring' / n for n in ['bf16_fma_pipeline.sv', 'bf16_conversions.sv', 'score7_group.sv', 'score7_top.sv']]
               + [ROOT / 'rtl/gqa' / n for n in ['metadata_tile_stream_gqa4.sv', 'tile_allocator_gqa4.sv', 'head_score_router_gqa4.sv', 'gqa4_frontend.sv']]
               + [GS_RTL, ROOT / 'rtl/bank/gqa4_score_bank_gs.sv'])
    read = f'read_verilog -sv -I{HF} -I{HF / "RISCV"} ' + ' '.join(map(str, dict.fromkeys(sources)))
    result = dict(name='bank_gs', **yosys(folder, 'gqa4_score_bank_gs', read, flatten_first=True, post_abc='clean -purge',
                                  final_clean=False))
    ports = '[get_ports {rst enable n_local* bank_id* q_* norms* req_ready rsp_* read_head* score_ready}]'
    result['slack_125MHz_ns'] = sta(folder, 'gqa4_score_bank_gs', normalize(folder), 125.0, ports)
    return result


GS_RTL = ROOT / 'rtl/gs/gs_shared_score.sv'
JOBS = {
    'gs_unit': lambda out: logic_block(out, 'gs_unit', 'gs_shared_score', hardfloat_sources() + [FMA, GS_RTL]),
    'vs_engine': lambda out: logic_block(out, 'vs_engine', 'vs_mass_threshold', HFS + [ROOT / 'rtl/vs/vs_mass_threshold.sv'], dict(RW=13)),
    'staging_w64': lambda out: logic_block(out, 'staging_w64', 'score_stream_interface_w',
                                           [ROOT / 'rtl/transport/score_stream_interface_w.sv'], dict(W=64)),
    'staging_w16': lambda out: logic_block(out, 'staging_w16', 'score_stream_interface_w',
                                           [ROOT / 'rtl/transport/score_stream_interface_w.sv'], dict(W=16)),
    's_path_staging': s_path_staging,
    'bank_gs': bank_gs,
    **{f'refresh_{k}': (lambda k: lambda out: refresh_scheduler(out, k))(k) for k in REFRESH_VARIANTS},
}


def recorded(name):
    data = json.loads((ROOT / 'data/area/extension_components.json').read_text())
    if name.startswith('refresh_'):
        return data['refresh_scheduler']['variants'][name[len('refresh_'):]]
    return data[name]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--blocks', nargs='+', choices=list(JOBS), default=list(JOBS))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args(argv)
    if not LIB.is_file():
        parser.error('Liberty file missing; run python -m tools.fetch_liberty')
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error('use a fresh output directory')
    out.mkdir(parents=True)
    with ThreadPoolExecutor(args.jobs) as pool:
        results = dict(zip(args.blocks, pool.map(lambda name: JOBS[name](out), args.blocks)))
    comparison = {}
    for name, result in results.items():
        ref = recorded(name)
        comparison[name] = dict(cell_area_um2=result['cell_area_um2'], recorded_cell_area_um2=ref['cell_area_um2'],
                                identical_area=abs(result['cell_area_um2'] - ref['cell_area_um2']) < 1e-6,
                                flip_flops=result['flip_flops'], recorded_flip_flops=ref['flip_flops'])
        print(f"{name:20s} {result['cell_area_um2']:12.3f} um2 (recorded {ref['cell_area_um2']:12.3f})"
              f"  FF {result['flip_flops']}", flush=True)
    save(out / 'summary.json', dict(liberty_sha256=sha(LIB), results=results, comparison=comparison))
    if not all(c['identical_area'] for c in comparison.values()):
        print('Some areas differ from the recorded values; tool versions (Yosys 0.9, ABC) and the Liberty file '
              'must match the development flow for identical mapping.')
        raise SystemExit(1)
    print(f'All {len(results)} blocks reproduce the recorded mapped-cell areas: {out}')


if __name__ == '__main__':
    main()
