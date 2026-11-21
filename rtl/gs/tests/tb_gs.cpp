// GS unit vs the simulator's GS input (tools.prepare_inputs: replay/gs4096/scores.bin, rows 4g) on the full snapshot,
// head-major 4-token tiles, random input gaps and output back-pressure.
#include "Vgs_shared_score.h"
#include "verilated.h"
#include <cstdio>
#include <cstdint>
#include <vector>
#include <random>
#include <deque>
int main(int argc,char**argv){
 Verilated::commandArgs(argc,argv);
 const int N=32768,H=32;
 std::vector<uint16_t> base(size_t(H)*N),gs(size_t(H)*N);
 FILE*f=fopen(argv[1],"rb");fread(base.data(),2,base.size(),f);fclose(f);
 f=fopen(argv[2],"rb");fread(gs.data(),2,gs.size(),f);fclose(f);
 struct In{uint64_t tag;uint16_t bits;};std::vector<In> in;
 struct Ex{int kv,tok;};std::deque<Ex> exp;
 for(int kv=0;kv<8;kv++)for(int t0=0;t0<N;t0+=4){
  for(int ph=0;ph<4;ph++)for(int t=t0;t<t0+4;t++){
   int h=kv*4+ph,bank=t>>12,ol=t&4095;
   uint64_t tag=(uint64_t(h)<<40)|(uint64_t(kv)<<32)|(uint64_t(ol)<<6)|bank;
   in.push_back({tag,base[size_t(h)*N+t]});}
  for(int t=t0;t<t0+4;t++)exp.push_back({kv,t});}
 Vgs_shared_score m;std::mt19937 rng(1);
 m.clk=0;m.rst=1;m.in_valid=0;m.out_ready=0;for(int i=0;i<4;i++){m.clk=!m.clk;m.eval();}
 m.rst=0;size_t p=0,ok=0,bad=0;uint64_t cyc=0;
 while((p<in.size()||!exp.empty())&&cyc<50000000){
  bool v=p<in.size()&&(rng()%8!=0);
  m.in_valid=v;if(v){m.in_tag=in[p].tag;m.in_bits=in[p].bits;}
  m.out_ready=(rng()%4!=0);
  m.clk=0;m.eval();
  bool fire=v&&m.in_ready,pop=m.out_valid&&m.out_ready;uint64_t ot=m.out_tag;uint16_t ob=m.out_bits;
  m.clk=1;m.eval();cyc++;
  if(fire)p++;
  if(pop){Ex e=exp.front();exp.pop_front();
   int head=(ot>>40)&31,ol=(ot>>6)&4095,bank=ot&63,tok=bank*4096+ol;
   uint16_t want=gs[size_t(e.kv*4)*N+e.tok];
   if(head==e.kv*4&&tok==e.tok&&ob==want)ok++;else{if(bad<10)printf("mismatch kv%d tok%d got h%d tok%d %04x want %04x\n",e.kv,e.tok,head,tok,ob,want);bad++;}}
 }
 printf("{\"test\":\"gs_shared_score\",\"outputs\":%zu,\"bit_exact\":%zu,\"mismatch\":%zu,\"pending\":%zu,\"cycles\":%lu}\n",ok+bad,ok,bad,exp.size(),cyc);
 return bad||!exp.empty();}
