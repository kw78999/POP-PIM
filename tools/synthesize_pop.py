"""Reproduce mapped GQA4+scorer bank area and pre-layout timing."""
import argparse,json,re,shutil
from pathlib import Path
from .common import ROOT,HF,LIB,env,run,save,sha,hardfloat_sources
ABC=None

def sources():
 return hardfloat_sources()+[ROOT/'rtl/primitives'/n for n in ['popcount32.sv','bitwise_count.sv']]+[ROOT/'rtl/scoring'/n for n in ['bf16_fma_pipeline.sv','bf16_conversions.sv','score7_group.sv','score7_top.sv']]

def synth(out):
 lib=LIB;p=out/'synthesis/FullBank';p.mkdir(parents=True,exist_ok=True)
 inputs=sources()+[ROOT/'rtl/gqa'/name for name in ['metadata_tile_stream_gqa4.sv', 'tile_allocator_gqa4.sv', 'head_score_router_gqa4.sv', 'gqa4_frontend.sv']]+[ROOT/'rtl/bank/gqa4_score_bank.sv']
 (p/'abc.constr').write_text('set_driving_cell BUF_X1\nset_load 5\n')
 ys='read_verilog -sv -I'+str(HF)+' -I'+str(HF/'RISCV')+' '+' '.join(str(x) for x in inputs)+f'''
hierarchy -check -top gqa4_score_bank
proc
opt
memory_dff
memory_collect
memory_map
opt
synth -noshare -top gqa4_score_bank
flatten
opt
dfflibmap -liberty {lib}
abc -exe {ABC} -liberty {lib} -constr {p}/abc.constr
clean -purge
splitnets -ports
stat -liberty {lib}
write_json {p}/mapped.json
write_verilog -noattr -noexpr {p}/mapped.v
'''
 (p/'synth.ys').write_text(ys)
 if not (p/'mapped.json').exists():run(['yosys','-s',p/'synth.ys'],p/'yosys.log',300)
 cells={}
 for body in re.split(r'\n\s*cell\s*\(',lib.read_text())[1:]:
  n=body.split(')',1)[0].strip().strip('"');m=re.search(r'\barea\s*:\s*([\d.]+)',body)
  if m:cells[n]=(float(m[1]),bool(re.search(r'\bff\s*\(',body)))
 module=json.loads((p/'mapped.json').read_text())['modules']['gqa4_score_bank'];assert all(c['type'] in cells for c in module['cells'].values())
 area=sum(cells[c['type']][0] for c in module['cells'].values());ff=sum(cells[c['type']][1] for c in module['cells'].values())
 normalized=[];aliases=0
 for line in (p/'mapped.v').read_text().splitlines():
  match=re.match(r'\s*assign\s+\{(.+)\}\s*=\s*\{(.+)\};',line)
  if match:
   lhs=[x.strip() for x in match[1].split(',')];rhs=[x.strip() for x in match[2].split(',')]
   assert len(lhs)==len(rhs) and all(x.startswith('\\') for x in lhs+rhs)
   normalized += [f'  assign {a}  = {b} ;' for a,b in zip(lhs,rhs)];aliases+=1
  else:normalized.append(line)
 (p/'mapped_sta.v').write_text('\n'.join(normalized)+'\n')
 save(p/'sta_netlist_normalization.json',dict(concatenated_alias_assignments_split=aliases,logic_changed=False,scope='Scalar bit aliases only, equivalent per-bit continuous assignments'))
 (p/'sta.tcl').write_text(f'''read_liberty {lib}
read_verilog {p}/mapped_sta.v
link_design gqa4_score_bank
create_clock -name clk -period 8 [get_ports clk]
set_clock_uncertainty 0.05 [get_clocks clk]
set_input_delay 0.1 -clock clk [get_ports {{rst enable n_local* bank_id* q_* norms* req_ready rsp_* read_head* score_ready}}]
set_output_delay 0.1 -clock clk [all_outputs]
set_load 5 [all_outputs]
report_checks -path_delay max -group_count 3 -digits 6
report_worst_slack -digits 6
exit
''');run(['sta','-exit',p/'sta.tcl'],p/'sta.log',90)
 txt=(p/'sta.log').read_text();assert 'Error:' not in txt,txt[-1000:];slack=float(re.findall(r'worst slack\s+(-?[\d.]+)',txt)[-1])
 result=dict(top='gqa4_score_bank',cell_area_um2=area,flip_flops=ff,blackboxes=0,slack_125MHz_ns=slack,timing_pass_125MHz=slack>=0,library_sha256=sha(lib),scope='Actual A2+32-head query mux+GQA4 metadata ownership+8KV allocator+global32-head router; external Q storage/response queues/selector/CDC/clock/layout excluded')
 save(out/'full_bank_ppa.json',result);print(result,flush=True)

def main():
 global ABC
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
 out=a.output_dir.resolve()
 if out.exists() and any(out.iterdir()):p.error('use a fresh output directory')
 if not LIB.exists():p.error('run python -m tools.fetch_liberty first')
 ABC=shutil.which('berkeley-abc',path=env()['PATH']) or shutil.which('yosys-abc',path=env()['PATH'])
 if not ABC:p.error('ABC executable not found')
 synth(out)

if __name__=='__main__':main()
