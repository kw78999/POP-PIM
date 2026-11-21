// Cycle-by-cycle comparison of pim_refresh_sched.sv against a transliteration of the simulator
// controller's update_refresh_obligations()/try_refresh() (same order, keys, and state updates).
#include "Vpim_refresh_sched.h"
#include "verilated.h"
#include <array>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#ifndef P_NREFI
#error parameters
#endif
struct Ref {
  int64_t P=P_P,U=P_U,H=P_H,LOW=P_LOW; bool round=P_ROUND; uint64_t nREFI=P_NREFI;
  std::array<uint64_t,64> next_due{}; std::array<int64_t,64> due{},issued{}; std::array<bool,64> drain{};
  int target=-1; uint64_t clk=0; bool violation=false;
  Ref(){for(int b=0;b<64;b++)next_due[b]=nREFI*uint64_t((b>>1)+1)/32;}
  // returns bank or -1; is_pre set
  int tick(uint64_t compute,uint64_t refreshing,uint64_t quiet,uint64_t queued,uint64_t ready,uint64_t open,bool&is_pre){
    clk++;
    for(int b=0;b<64;b++){
      while(clk>=next_due[b]){due[b]++;next_due[b]+=nREFI;}
      int64_t debt=due[b]-issued[b]; if(debt>P)violation=true;
      if(debt>=P-H)drain[b]=true; else if(debt<=LOW)drain[b]=false;
    }
    std::array<int64_t,2> least{INT64_MAX,INT64_MAX};
    for(int b=0;b<64;b++)least[b&1]=std::min(least[b&1],issued[b]);
    auto bit=[](uint64_t m,int b){return (m>>b)&1;};
    auto eligible=[&](int b){
      if(bit(compute,b)||bit(refreshing,b))return false;
      if(due[b]-issued[b]-1<-U)return false;
      if(round&&issued[b]>least[b&1])return false;
      return drain[b]||(bit(quiet,b)&&!bit(queued,b));
    };
    int order[64],n=0;
    if(target>=0&&eligible(target))order[n++]=target; else target=-1;
    int rest[64],r=0;for(int b=0;b<64;b++)if(b!=target&&eligible(b))rest[r++]=b;
    std::sort(rest,rest+r,[&](int x,int y){if(drain[x]!=drain[y])return drain[x]>drain[y];
      int64_t dx=due[x]-issued[x],dy=due[y]-issued[y];if(dx!=dy)return dx>dy;return x<y;});
    for(int i=0;i<r;i++)order[n++]=rest[i];
    for(int i=0;i<n;i++){int b=order[i];if(!bit(ready,b))continue;
      is_pre=bit(open,b);
      if(!is_pre){issued[b]++;target=-1;}else target=b;
      return b;}
    return -1;
  }
};
int main(int argc,char**argv){
  Verilated::commandArgs(argc,argv);
  Vpim_refresh_sched dut; Ref ref; std::mt19937_64 rng(P_SEED);
  uint64_t compute=0,refreshing=0,quiet=0,queued=0,open=0; uint64_t issued=0,pres=0,drains=0;
  dut.rst=1;dut.clk=0;dut.eval();dut.clk=1;dut.eval();dut.clk=0;dut.rst=0;dut.eval();
  const uint64_t cycles=P_CYCLES;
  for(uint64_t c=0;c<cycles;c++){
    // Correlated random activity: flip a few banks per cycle; long busy phases stress drains.
    for(int k=0;k<3;k++){int b=rng()%64;compute^=uint64_t(1)<<b;}
    for(int k=0;k<2;k++){int b=rng()%64;quiet^=uint64_t(1)<<b;}
    queued=rng()&rng();refreshing=rng()&rng()&rng();open=rng();
    uint64_t ready=rng()|rng();
    if(c%7000<3000)compute|=rng()|rng(); // periodic heavy compute windows
#ifdef P_HEAVY
    quiet&=rng()&rng()&rng()&rng(); if(c%40000<30000)compute|=rng()|rng()|rng();
#endif
    dut.compute=compute;dut.refreshing=refreshing;dut.quiet=quiet;dut.queued=queued;dut.ready=ready;dut.bank_open=open;
    dut.eval();
    bool pre=false;int rb=ref.tick(compute,refreshing,quiet,queued,ready,open,pre);
    uint64_t rdrain=0;for(int b=0;b<64;b++)if(ref.drain[b])rdrain|=uint64_t(1)<<b;
    bool ok=(dut.req_valid==(rb>=0))&&(rb<0||(dut.req_bank==rb&&dut.req_is_pre==pre))&&dut.drain==rdrain;
    if(!ok){std::printf("{\"passed\":false,\"cycle\":%llu,\"rtl_valid\":%d,\"rtl_bank\":%d,\"ref_bank\":%d}\n",(unsigned long long)c,dut.req_valid,dut.req_bank,rb);return 1;}
    if(rb>=0){if(pre)pres++;else issued++;} drains+=__builtin_popcountll(rdrain);
    dut.clk=1;dut.eval();dut.clk=0;dut.eval();
    // The RTL flags violations on the registered edge; compare after the edge.
    if(bool(dut.violation)!=ref.violation){std::printf("{\"passed\":false,\"cycle\":%llu,\"violation_mismatch\":true}\n",(unsigned long long)c);return 1;}
  }
  std::printf("{\"passed\":true,\"cycles\":%llu,\"refsb\":%llu,\"refresh_pre\":%llu,\"drain_bank_cycles\":%llu,\"violation\":%s,\"NREFI\":%d,\"P\":%d,\"U\":%d,\"H\":%d,\"LOW\":%d,\"ROUND\":%d,\"seed\":%d}\n",
    (unsigned long long)cycles,(unsigned long long)issued,(unsigned long long)pres,(unsigned long long)drains,ref.violation?"true":"false",P_NREFI,P_P,P_U,P_H,P_LOW,P_ROUND,P_SEED);
  return 0;
}
