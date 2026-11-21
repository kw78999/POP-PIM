"""Mapped Dense scorer synthesis and pre-layout STA under preserved constraints."""
import argparse, collections, csv, json, re, shutil
from pathlib import Path
from .common import ROOT, BUILD, HF, LIB, env, sha, run, dense_sources
CELLS={}
OUT=None
ABC=None

def synth(lanes, port16=False):
 name=f'D{lanes}';folder=OUT/'synthesis'/name;folder.mkdir(parents=True,exist_ok=True);wrapper=folder/'wrapper.sv'
 wrapper.write_text('''module dense9_ppa(input wire clk,rst,in_valid,first_beat,last_beat,input wire [47:0] in_tag,
 input wire [%d:0] q_bits,k_bits,input wire [15:0] normalization,input wire out_ready,
 output wire in_ready,out_valid,output wire [47:0] out_tag,output wire [15:0] score_bits);
 dense_score #(.LANES(%d)) core(.clk(clk),.rst(rst),.in_valid(in_valid),.first_beat(first_beat),.last_beat(last_beat),.in_tag(in_tag),
 .q_bits(q_bits),.k_bits(k_bits),.normalization(normalization),.out_ready(out_ready),.in_ready(in_ready),.out_valid(out_valid),.out_tag(out_tag),.score_bits(score_bits));
 endmodule\n'''%(16*lanes-1,lanes))
 if port16:
  w=wrapper.read_text().replace(f'[{16*lanes-1}:0] q_bits,k_bits','[127:0] q_bits,k_bits').replace('dense_score #(', 'dense_score_port16 #(')
  wrapper.write_text(w)
 ys='read_verilog -sv -I'+str(HF)+' -I'+str(HF/'RISCV')+' '+' '.join(map(str,dense_sources()+([ROOT/'rtl/dense/dense_score_port16.sv'] if port16 else [])+[wrapper]))+'\n'
 (folder/'abc.constr').write_text('set_driving_cell BUF_X1\nset_load 5\n')
 ys+=f'''hierarchy -top dense9_ppa\nproc\nopt\nmemory_dff\nmemory_collect\nmemory_map\nopt\nsynth -noshare -top dense9_ppa
dfflibmap -liberty {LIB}
abc -exe {ABC} -liberty {LIB} -constr {folder}/abc.constr
clean
stat -liberty {LIB}
write_json {folder}/hierarchy.json
flatten
clean -purge
splitnets -ports
clean
write_json {folder}/mapped.json
write_verilog -noattr -noexpr {folder}/mapped.v
'''
 (folder/'synth.ys').write_text(ys);print('synth',name,flush=True);run(['yosys','-s',folder/'synth.ys'],folder/'yosys.log')
 mods=json.loads((folder/'hierarchy.json').read_text())['modules'];areas=collections.Counter();ffs=collections.Counter();num=collections.Counter();fma=[]
 def walk(mod,path=''):
  m=mods[mod];assert not int(m.get('attributes',{}).get('blackbox',0));alias=collections.defaultdict(list)
  for n,v in m.get('netnames',{}).items():
   for bit in v['bits']:alias[bit].append(n)
  for n,c in m['cells'].items():
   p=path+'.'+n;typ=c['type']
   if typ in mods:
    if typ=='bf16_fma_pipeline':fma.append(p)
    walk(typ,p);continue
   assert typ in CELLS,(p,typ);a,ff=CELLS[typ];names=' '.join(x for bit in c.get('connections',{}).get('Q',[]) for x in alias[bit])
   if '.fma.' in p:cat='FMA pipeline registers' if ff else 'FMA datapath'
   elif ff:
    if 'val[' in names:cat='Partial accumulation and reduction registers'
    elif 'buffer_' in names:cat='Input beat buffer registers'
    elif any(x in names for x in ['results','tags','norms']):cat='Output and metadata registers'
    else:cat='Controller scoreboard and tag registers'
   else:cat='Operand mux and control combinational'
   areas[cat]+=a;ffs[cat]+=ff;num[cat]+=1
 walk('dense9_ppa');assert len(fma)==lanes
 flat=json.loads((folder/'mapped.json').read_text())['modules']['dense9_ppa'];area=sum(CELLS[c['type']][0] for c in flat['cells'].values());assert abs(area-sum(areas.values()))<1e-5
 timing=[]
 def sta(freq):
  period=1000/freq;p=folder/f'timing_{freq:.3f}MHz.tcl';p.write_text(f'''read_liberty {LIB}
read_verilog {folder}/mapped.v
link_design dense9_ppa
create_clock -name clk -period {period:.9f} [get_ports clk]
set_clock_uncertainty 0.05 [get_clocks clk]
set_input_delay 0.1 -clock clk [get_ports {{rst in_valid first_beat last_beat in_tag* q_bits* k_bits* normalization* out_ready}}]
set_output_delay 0.1 -clock clk [all_outputs]
set_load 5 [all_outputs]
report_checks -path_delay max -group_count 3 -digits 6
report_worst_slack -digits 6
exit
''');log=p.with_suffix('.log');run(['sta','-exit',p],log);s=log.read_text();assert 'Error:' not in s,s[-1500:];slack=float(re.findall(r'worst slack\s+(-?[\d.]+)',s)[-1]);timing.append(dict(design=name,frequency_MHz=freq,slack_ns=slack,timing_pass=slack>=0,evidence='Pre-layout STA'));return slack
 for f in [125,150,200,250,300]:sta(f)
 lo,hi=25.,800.
 while hi-lo>.5:
  mid=(lo+hi)/2
  if sta(mid)>=0:lo=mid
  else:hi=mid
 (folder/'timing.json').write_text(json.dumps(timing,indent=2)+'\n')
 r=dict(design=name,mac_instances=lanes,contexts=2,total_cell_area_um2=area,datapath_area_um2=areas['FMA datapath'],fma_pipeline_register_area_um2=areas['FMA pipeline registers'],register_area_um2=sum(areas[c] for c in areas if ffs[c]>0),control_mux_area_um2=areas['Operand mux and control combinational'],flip_flops=sum(ffs.values()),fmax_pass_MHz=lo,fmax_fail_MHz=hi,slack_125_ns=timing[0]['slack_ns'],timing_pass_125=timing[0]['timing_pass'],within_A2_area_budget=area<=24304.154,blackboxes=0,library_sha256=sha(LIB),evidence='Actual RTL Synthesis / Pre-layout STA')
 (folder/'metrics.json').write_text(json.dumps(r,indent=2)+'\n');print(name,'area',area,'Fmax',lo,flush=True)
 return r,[dict(design=name,category=c,area_um2=areas[c],flip_flops=ffs[c],cells=num[c],evidence='Actual RTL Synthesis') for c in areas]

def main():
 global OUT,ABC
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--lanes',type=int,nargs='+',choices=[1,2,4,8,16,32],default=[1,2,4,8])
 p.add_argument('--output-dir',type=Path,required=True)
 a=p.parse_args();OUT=a.output_dir.resolve()
 if OUT.exists() and any(OUT.iterdir()):p.error('use a fresh output directory')
 if not LIB.exists():p.error('run python -m tools.fetch_liberty first, or set POPPIM_LIBERTY')
 ABC=shutil.which('berkeley-abc',path=env()['PATH']) or shutil.which('yosys-abc',path=env()['PATH'])
 if ABC is None:p.error('ABC executable not found')
 for body in re.split(r'\n\s*cell\s*\(',LIB.read_text())[1:]:
  name=body.split(')',1)[0].strip().strip('"');m=re.search(r'\barea\s*:\s*([\d.]+)',body)
  if m:CELLS[name]=(float(m[1]),bool(re.search(r'\bff\s*\(',body)))
 rows=[];breakdown=[]
 for lanes in a.lanes:
  row,parts=synth(lanes,port16=True);rows.append(row);breakdown.extend(parts)
 for name,data in [('hardware.csv',rows),('cell_breakdown.csv',breakdown)]:
  with (OUT/name).open('w',newline='') as handle:
   w=csv.DictWriter(handle,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 print(OUT)

if __name__=='__main__':main()
