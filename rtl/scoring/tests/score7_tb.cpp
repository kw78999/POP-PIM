#include "Vscore7_top.h"
#include "verilated.h"
#if VM_TRACE
#include "verilated_vcd_c.h"
#include <memory>
#endif
#include <fstream>
#include <vector>
#include <deque>
#include <map>
#include <cstdint>
#include <stdexcept>
#include <iostream>
#include <algorithm>
#pragma pack(push,1)
struct Group{uint64_t qs,qm,ks,km,mask;uint32_t ql,dq,norm;uint16_t kl,dk;int8_t count[4];uint16_t expected[11];};
struct Frame{Group g[2];uint16_t final[2];};
#pragma pack(pop)
static_assert(sizeof(Frame)==168);
struct Pending{uint64_t cycle,tag;int node;};
uint64_t tag(int f){return(uint64_t(f%4)<<40)|(uint64_t(f%2)<<32)|uint32_t(f);}
void check(bool b,const std::string&s){if(!b)throw std::runtime_error(s);}
int main(int argc,char**argv){try{
 Verilated::commandArgs(argc,argv);check(argc>=4,"vectors prefix mode");
 std::ifstream in(argv[1],std::ios::binary);uint32_t n,gn;in.read((char*)&n,4);in.read((char*)&gn,4);std::vector<Frame>v(n);in.read((char*)v.data(),n*sizeof(Frame));check(bool(in),"vectors truncated");
 int mode=std::stoi(argv[3]);int limit=argc>4?std::stoi(argv[4]):n;n=std::min(n,unsigned(limit));
 std::vector<uint64_t>start(n),valid_at(n,~uint64_t(0)),issued(n),completed(n);std::vector<int>count(n);
 std::deque<Pending>pending,pc;std::vector<uint64_t>outcycles,issuecycles;std::vector<uint8_t>occupancy;
 std::ofstream trace(std::string(argv[2])+"_trace.csv"),frames(std::string(argv[2])+"_frames.csv");
 trace<<"cycle,event,frame,node,bits\n";frames<<"frame,start_cycle,valid_cycle,accepted_cycle,latency,bits,fp_requests\n";
 auto event=[&](uint64_t c,const char*s,int f,int node,int bits){if(f<32)trace<<c<<","<<s<<","<<f<<","<<node<<","<<bits<<"\n";};
 Vscore7_top d;
#if VM_TRACE
 Verilated::traceEverOn(true);std::unique_ptr<VerilatedVcdC>vcd;
 if(argc>5){vcd=std::make_unique<VerilatedVcdC>();d.trace(vcd.get(),99);vcd->open(argv[5]);}
 uint64_t timestamp=0;
#endif
 d.clk=0;d.rst=1;d.in_valid=0;d.out_ready=0;d.eval();auto tick=[&](){
#if VM_TRACE
 if(vcd)vcd->dump(timestamp++);
#endif
 d.clk=1;d.eval();
#if VM_TRACE
 if(vcd)vcd->dump(timestamp++);
#endif
 d.clk=0;d.eval();};tick();tick();d.rst=0;
 uint32_t rng=7009;uint32_t next=0,rep=0,outputs=0,resets=0;uint64_t issues=0,completes=0,counts=0,outs=0,dep=0,ctx=0;bool hold=false,ohold=false;uint64_t heldtag=0;uint16_t heldbits=0;
 for(uint64_t cycle=0;cycle<30000000;cycle++){
  if(mode==2&&(cycle==23||cycle==157||cycle==1013)){
   d.in_valid=0;d.out_ready=0;d.rst=1;tick();d.rst=0;d.eval();resets++;hold=ohold=false;pending.clear();pc.clear();occupancy.push_back(0);next=outputs;rep=0;
   for(uint32_t f=outputs;f<n;f++){issued[f]=completed[f]=0;count[f]=0;valid_at[f]=~uint64_t(0);}continue;
  }
  rng=rng*1664525u+1013904223u;
  d.in_valid=next<n&&(hold||mode==0||((rng>>20)&7)!=0);
  if(next<n){auto&g=v[next].g[rep];d.first_group=rep==0;d.last_group=rep==1;d.in_tag=tag(next);d.q_sign=g.qs;d.q_mag=g.qm;d.k_sign=g.ks;d.k_mag=g.km;d.dim_mask=g.mask;d.q_low=g.ql;d.q_delta=g.dq;d.k_low=g.kl;d.k_delta=g.dk;d.normalization=g.norm;}
  d.out_ready=mode==0||(((rng>>12)&7)!=0&&!(cycle%257>=100&&cycle%257<140));d.eval();
  check(!ohold||(d.out_valid&&d.out_tag==heldtag&&d.score_bits==heldbits),"unstable output");ohold=d.out_valid&&!d.out_ready;heldtag=d.out_tag;heldbits=d.score_bits;
  outs+=d.output_stall;dep+=d.dependency_stall;ctx+=d.context_stall;occupancy.push_back(pending.size());
  if(d.fp_complete_valid){
   check(!pending.empty(),"unrequested completion");auto p=pending.front();pending.pop_front();
   check(cycle==p.cycle+3&&d.fp_complete_tag==p.tag&&d.fp_complete_node==p.node,"result tag or latency mismatch");
   int f=uint32_t(p.tag),node=p.node;uint16_t want=node<int(2*gn)?v[f].g[node/gn].expected[node%gn]:v[f].final[node-2*gn];
   if(d.fp_complete_bits!=want){std::cerr<<"mismatch cycle="<<cycle<<" frame="<<f<<" node="<<node<<" got="<<std::hex<<d.fp_complete_bits<<" want="<<want<<"\n";return 2;}
   completed[f]|=uint64_t(1)<<node;completes++;event(cycle,"COMPLETE",f,node,d.fp_complete_bits);
  }
  if(d.fp_issue_valid){
   int f=uint32_t(d.fp_issue_tag),node=d.fp_issue_node;check(f<int(n)&&d.fp_issue_tag==tag(f)&&node<int(2*gn+2),"issue tag invalid");
   check(!(issued[f]&(uint64_t(1)<<node)),"duplicate node issue");
   uint64_t deps=0;
   if(node==int(2*gn))deps=(uint64_t(1)<<(gn-1))|(uint64_t(1)<<(2*gn-1));
   else if(node==int(2*gn+1))deps=uint64_t(1)<<(2*gn);
   else {int g=node/gn,o=node%gn;uint32_t mask=0;
    if(gn==9){uint32_t ds[9]={0,0,0,0,1u<<0,1u<<2,(1u<<1)|(1u<<4),(1u<<3)|(1u<<5),(1u<<6)|(1u<<7)};mask=ds[o];}
    else {uint32_t ds[11]={0,0,0,0,1u<<0,1u<<1,1u<<2,1u<<3,(1u<<4)|(1u<<5),(1u<<6)|(1u<<7),(1u<<8)|(1u<<9)};mask=ds[o];}
    deps=uint64_t(mask)<<(g*gn);if(o>=4&&o<8)check(count[f]>g,"weighted op issued before POPCOUNT");
   }
   check((completed[f]&deps)==deps,"dependency violation");issued[f]|=uint64_t(1)<<node;issues++;pending.push_back({cycle,d.fp_issue_tag,node});issuecycles.push_back(cycle);event(cycle,"ISSUE",f,node,0);
  }
  if(d.count_valid){
   check(!pc.empty(),"unexpected count");auto p=pc.front();pc.pop_front();check(p.tag==d.count_tag&&p.node==d.count_rep&&cycle==p.cycle+6,"count tag latency");
   int f=uint32_t(p.tag);for(int j=0;j<4;j++)check(int8_t(d.count_values>>(8*j))==v[f].g[p.node].count[j],"count bits");count[f]++;counts++;event(cycle,"COUNT",f,p.node,0);
  }
  if(d.out_valid){
   int f=uint32_t(d.out_tag);check(f==int(outputs)&&d.out_tag==tag(f),"output order");check(d.score_bits==v[f].final[1],"final bits");
   if(valid_at[f]==~uint64_t(0))valid_at[f]=cycle;
   if(d.out_ready){check(issued[f]==(uint64_t(1)<<(2*gn+2))-1&&completed[f]==issued[f]&&count[f]==2,"request count/drain");
    frames<<f<<","<<start[f]<<","<<valid_at[f]<<","<<cycle<<","<<valid_at[f]-start[f]<<","<<d.score_bits<<","<<2*gn+2<<"\n";outcycles.push_back(cycle);outputs++;}
  }
  hold=d.in_valid&&!d.in_ready;
  if(d.in_valid&&d.in_ready){pc.push_back({cycle,tag(next),int(rep)});if(rep==0)start[next]=cycle;event(cycle,"INPUT",next,rep,0);if(rep==1){next++;rep=0;}else rep=1;}
  tick();
  if(outputs==n&&d.empty){
   check(pending.empty()&&pc.empty(),"pending drain");uint64_t ws=outcycles.at(n>512?256:0),we=outcycles.at(n>512?n-9:n-1),stages=0;
   for(uint64_t c=ws;c<we;c++)stages+=occupancy.at(c);
   auto ni=std::lower_bound(issuecycles.begin(),issuecycles.end(),we)-std::lower_bound(issuecycles.begin(),issuecycles.end(),ws);
   std::ofstream stats(std::string(argv[2])+"_stats.json");stats<<"{\"outputs\":"<<outputs<<",\"cycles\":"<<cycle+1<<",\"issues\":"<<issues<<",\"completes\":"<<completes<<",\"count_completions\":"<<counts<<",\"resets\":"<<resets<<",\"output_stall_cycles\":"<<outs<<",\"dependency_stall_cycles\":"<<dep<<",\"context_stall_cycles\":"<<ctx<<",\"steady_issue_utilization\":"<<double(ni)/(we-ws)<<",\"stage_occupancy\":"<<double(stages)/(3*(we-ws))<<",\"all_bit_exact\":true,\"fp_latency\":3}\n";
   std::cout<<"PASS outputs="<<outputs<<" cycles="<<cycle+1<<" issues="<<issues<<" completes="<<completes<<" resets="<<resets<<"\n";return 0;
  }
 }
 throw std::runtime_error("timeout");
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
