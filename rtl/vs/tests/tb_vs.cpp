// VS engine vs the bit-exact software reference (experiments/replay/make_vkeep_masks.py) for every KV group of the GS variants.
// SRAM models: hist 1024x30 and bincode 2^13 x 32 (byte enables), 1-cycle read, write visible next cycle.
#include "Vvs_mass_threshold.h"
#include "verilated.h"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <random>
struct Mem{std::vector<uint32_t> d;uint32_t q=0;};
int main(int argc,char**argv){
 Verilated::commandArgs(argc,argv);
 int fails=0,cases=0;long keep_ok=0,keep_n=0;
 for(int a=1;a<argc;a++){
  FILE*f=fopen(argv[a],"rb");if(!f)continue;
  uint32_t C[4],KD,E0,KS;fread(C,4,4,f);fread(&KD,4,1,f);fread(&E0,4,1,f);fread(&KS,4,1,f);
  std::vector<uint32_t> lg(size_t(KS)*4);fread(lg.data(),4,lg.size(),f);
  uint32_t bs[4];fread(bs,4,4,f);std::vector<uint8_t> kp(KS);fread(kp.data(),1,KS,f);fclose(f);
  Vvs_mass_threshold m;Mem hist,bc;hist.d.assign(1024,0xdeadbeef&0x3fffffff);bc.d.assign(1<<13,0);
  std::mt19937 rng(a);
  for(int i=0;i<4;i++)m.c_coef[i]=C[i];m.k_delta=KD;m.e0p=E0;
  auto tick=[&](){
   m.clk=0;m.eval();
   bool hre=m.h_re,hwe=m.h_we,bre=m.b_re,bwe=m.b_we;uint32_t hra=m.h_raddr,hwa=m.h_waddr,hwd=m.h_wdata,bra=m.b_raddr,bwa=m.b_waddr,bwd=m.b_wdata,be=m.b_be;
   m.clk=1;m.eval();
   if(hre)m.h_rdata=hist.d[hra];
   if(bre)m.b_rdata=bc.d[bra];
   if(hwe)hist.d[hwa]=hwd;
   if(bwe){uint32_t x=bc.d[bwa];for(int b=0;b<4;b++)if(be>>b&1)x=(x&~(0xffu<<(8*b)))|(bwd&(0xffu<<(8*b)));bc.d[bwa]=x;}
   m.eval();};
  m.rst=1;m.clear=0;m.scan_start=0;m.lg_valid=0;m.kq_valid=0;tick();tick();m.rst=0;
  m.clear=1;tick();m.clear=0;while(m.busy)tick();
  // phase K: rank-major, head-minor, random gaps
  size_t p=0;while(p<lg.size()){bool v=rng()%5!=0;m.lg_valid=v;if(v){m.lg_head=p%4;m.lg_rank=p/4;m.lg_logit=lg[p];}
   m.clk=0;m.eval();bool fire=v&&m.lg_ready;tick();if(fire)p++;}
  m.lg_valid=0;for(int i=0;i<6;i++)tick();
  m.scan_start=1;tick();m.scan_start=0;long sc=0;while(!m.scan_done&&sc<100000){tick();sc++;}
  bool good=m.scan_done;uint32_t got[4];for(int h=0;h<4;h++){got[h]=(m.bstar>>(8*h))&255;good&=got[h]==bs[h];}
  // phase V: keep query per rank (pipelined, 2-cycle latency)
  std::vector<int> out;size_t r=0;int pending=0;
  while(out.size()<KS){m.kq_valid=r<KS;if(r<KS)m.kq_rank=r;m.clk=0;m.eval();bool ov=m.kq_out_valid;int kk=m.kq_keep;tick();if(r<KS)r++;if(ov)out.push_back(kk);}
  long kok=0;for(size_t i=0;i<KS;i++)kok+=out[i]==kp[i];keep_ok+=kok;keep_n+=KS;
  good&=kok==(long)KS;cases++;if(!good)fails++;
  printf("%s KS=%u b*=[%u %u %u %u] want [%u %u %u %u] keep %ld/%u scan_cycles %ld %s\n",argv[a],KS,got[0],got[1],got[2],got[3],bs[0],bs[1],bs[2],bs[3],kok,KS,sc,good?"PASS":"FAIL");
 }
 printf("{\"test\":\"vs_mass_threshold\",\"kv_groups\":%d,\"fail\":%d,\"keep_bits_checked\":%ld,\"keep_bits_match\":%ld}\n",cases,fails,keep_n,keep_ok);
 return fails!=0;}
