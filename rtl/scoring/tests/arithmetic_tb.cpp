#include "Vscore7_top.h"
#include "verilated.h"
#include <fstream>
#include <vector>
#include <deque>
#include <stdexcept>
#include <iostream>
#include <cstdint>
#pragma pack(push,1)
struct Row{uint16_t a,b,c,result;uint8_t flags;uint32_t f32;uint16_t f16;uint8_t count;uint16_t c32,c16,cc;};
#pragma pack(pop)
static_assert(sizeof(Row)==22);
int main(int argc,char**argv){try{
 std::ifstream f(argv[1],std::ios::binary);uint32_t n;f.read((char*)&n,4);std::vector<Row>v(n);f.read((char*)v.data(),n*sizeof(Row));if(!f)throw std::runtime_error("vectors");
 Vscore7_top d;d.clk=0;d.rst=1;d.in_valid=0;d.eval();auto tick=[&](){d.clk=1;d.eval();d.clk=0;d.eval();};tick();tick();d.rst=0;std::deque<int>q;uint64_t checked=0;
 for(unsigned c=0;c<n+3;c++){
  d.in_valid=c<n;if(c<n){auto&r=v[c];d.a=r.a;d.b=r.b;d.c=r.c;d.fp32_bits=r.f32;d.fp16_bits=r.f16;d.count=r.count;}d.eval();
  if(c<n){auto&r=v[c];uint16_t canonical=(r.a&0x7fff)?r.a:0;uint16_t key=(canonical&0x8000)?uint16_t(~canonical):uint16_t(canonical^0x8000);if(d.sort_key!=key||d.sort_finite!=((r.a&0x7f80)!=0x7f80))throw std::runtime_error("BF16 sort key/finite classifier");if(d.converted32!=r.c32||d.converted16!=r.c16||d.converted_count!=r.cc){std::cerr<<"conversion row "<<c<<"\n";return 2;}}
  if(d.out_valid){if(q.empty())throw std::runtime_error("unsolicited result");int k=q.front();q.pop_front();auto&r=v[k];if(d.result!=r.result||d.flags!=r.flags){std::cerr<<"arithmetic row="<<k<<" a="<<std::hex<<r.a<<" b="<<r.b<<" c="<<r.c<<" got="<<d.result<<" expected="<<r.result<<" flags="<<int(d.flags)<<" wanted_flags="<<int(r.flags)<<"\n";return 3;}checked++;}
  if(d.in_valid)q.push_back(c);tick();
 }
 if(checked!=n||!q.empty())throw std::runtime_error("drain");std::cout<<"PASS BF16 FMA/MUL/ADD bits+flags="<<checked<<" conversions="<<3*n<<"\n";
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
