// Phase7-only activity harness for unchanged Phase6 RTL; identical primary metadata.
#include "Vscore7_top.h"
#include "verilated.h"
#include "verilated_vcd_c.h"
#include <fstream>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <iostream>
#pragma pack(push,1)
struct Group{uint64_t qs,qm,ks,km,mask;uint32_t ql,dq,norm;uint16_t kl,dk;};
struct Frame{Group g[2];uint32_t fp32;uint16_t fp16;};
#pragma pack(pop)
static_assert(sizeof(Frame)==118);
int main(int argc,char**argv){try{
 Verilated::commandArgs(argc,argv);Verilated::traceEverOn(true);std::ifstream f(argv[1],std::ios::binary);uint32_t h,n;f.read((char*)&h,4);f.read((char*)&n,4);std::vector<Frame>v(512);f.read((char*)v.data(),v.size()*sizeof(Frame));if(!f)throw std::runtime_error("metadata");
 Vscore7_top d;VerilatedVcdC trace;d.trace(&trace,99);trace.open(argv[2]);uint64_t ts=0;d.clk=0;d.rst=1;d.in_valid=0;d.out_ready=1;d.eval();
 auto tick=[&](){trace.dump(ts++);d.clk=1;d.eval();trace.dump(ts++);d.clk=0;d.eval();};tick();tick();d.rst=0;unsigned next=0,rep=0,outputs=0;
 for(unsigned cycle=0;cycle<100000;cycle++){
  d.in_valid=next<512;
  if(next<512){auto&g=v[next].g[rep];d.first_group=rep==0;d.last_group=rep==1;d.in_tag=next;d.q_sign=g.qs;d.q_mag=g.qm;d.k_sign=g.ks;d.k_mag=g.km;d.dim_mask=g.mask;d.q_low=g.ql;d.q_delta=g.dq;d.normalization=g.norm;d.k_low=g.kl;d.k_delta=g.dk;}
  d.eval();if(d.out_valid){if(d.out_tag!=outputs||d.out_fp32!=v[outputs].fp32||d.out_fp16!=v[outputs].fp16)throw std::runtime_error("FP32 baseline bit mismatch");outputs++;}
  if(d.in_valid&&d.in_ready){if(rep){next++;rep=0;}else rep=1;}tick();
  if(outputs==512&&d.empty){trace.close();std::cout<<"PASS FP32 baseline outputs="<<outputs<<" cycles="<<cycle+1<<"\n";return 0;}
 }
 throw std::runtime_error("timeout");
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
