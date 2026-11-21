#include "Vdense_score.h"
#include "verilated.h"
#if VM_TRACE
#include "verilated_vcd_c.h"
#include <memory>
#endif
#include <fstream>
#include <vector>
#include <deque>
#include <cstdint>
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <type_traits>
template<class T>uint64_t get(const T&x,int start,int n){uint64_t v=0;for(int j=0;j<n;j++){if constexpr(std::is_integral<T>::value)v|=uint64_t((x>>(start+j))&1)<<j;else v|=uint64_t((x[(start+j)/32]>>((start+j)%32))&1)<<j;}return v;}
template<class T>void clear(T&x){if constexpr(std::is_integral<T>::value)x=0;else for(unsigned j=0;j<sizeof(x)/4;j++)x[j]=0;}
template<class T>void put(T&x,int start,uint16_t v){for(int j=0;j<16;j++){if constexpr(std::is_integral<T>::value){auto m=T(1)<<(start+j);x=(x&~m)|((T((v>>j)&1))<<(start+j));}else{unsigned bit=start+j;auto m=uint32_t(1)<<(bit%32);x[bit/32]=(x[bit/32]&~m)|(uint32_t((v>>j)&1)<<(bit%32));}}}
struct Frame{uint16_t q[128],k[128];std::vector<uint16_t>expected;};struct Pending{uint64_t cycle,tag;int node;};
void check(bool b,const std::string&s){if(!b)throw std::runtime_error(s);}
uint64_t tag(int f){return (uint64_t(f%4)<<40)|(uint64_t(f%2)<<32)|uint32_t(f);}
int main(int argc,char**argv){try{
 Verilated::commandArgs(argc,argv);check(argc>=4,"vectors prefix mode [limit] [vcd]");std::ifstream f(argv[1],std::ios::binary);uint32_t n,L,R;uint16_t norm;f.read((char*)&n,4);f.read((char*)&L,4);f.read((char*)&R,4);f.read((char*)&norm,2);std::vector<Frame>v(n);
 for(auto&x:v){x.expected.resize(R);f.read((char*)x.q,256);f.read((char*)x.k,256);f.read((char*)x.expected.data(),2*R);}check(bool(f),"vector file");if(argc>4)n=std::min(n,unsigned(std::stoi(argv[4])));int mode=std::stoi(argv[3]);
 Vdense_score d;
#if VM_TRACE
 Verilated::traceEverOn(true);std::unique_ptr<VerilatedVcdC>trace;if(argc>5){trace=std::make_unique<VerilatedVcdC>();d.trace(trace.get(),99);trace->open(argv[5]);}uint64_t ts=0;
#endif
 d.clk=0;d.rst=1;d.in_valid=0;d.out_ready=0;d.eval();auto tick=[&](){
#if VM_TRACE
 if(trace)trace->dump(ts++);
#endif
 d.clk=1;d.eval();
#if VM_TRACE
 if(trace)trace->dump(ts++);
#endif
 d.clk=0;d.eval();};tick();tick();d.rst=0;
 std::vector<uint64_t>start(n),valid(n,~uint64_t(0));std::vector<std::vector<bool>>issued(n,std::vector<bool>(R)),done=issued;std::vector<unsigned>count(n);std::vector<std::deque<Pending>>pending(L);
 unsigned next=0,beat=0,outputs=0,resets=0;uint64_t issues=0,completes=0,outputstall=0,inputstall=0;bool ihold=false,ohold=false;uint64_t heldtag=0;uint16_t heldbits=0;uint32_t rng=9019;
 std::ofstream frames(std::string(argv[2])+"_frames.csv"),events(std::string(argv[2])+"_trace.csv");frames<<"frame,start,valid,accepted,latency,bits\n";events<<"cycle,event,frame,lane,node,bits\n";
 for(uint64_t c=0;c<20000000;c++){
  if(mode==2&&(c==71||c==313||c==1009)){d.in_valid=0;d.rst=1;tick();d.rst=0;resets++;ihold=ohold=false;for(auto&p:pending)p.clear();next=outputs;beat=0;for(unsigned i=outputs;i<n;i++){std::fill(issued[i].begin(),issued[i].end(),false);std::fill(done[i].begin(),done[i].end(),false);count[i]=0;valid[i]=~uint64_t(0);}continue;}
  rng=rng*1664525u+1013904223u;d.in_valid=next<n&&(ihold||mode==0||((rng>>20)&7)!=0);d.out_ready=mode==0||(((rng>>10)&7)!=0&&!(c%257>=100&&c%257<140));d.normalization=norm;
  if(next<n){d.first_beat=beat==0;d.last_beat=(beat+1)*L>=128;d.in_tag=tag(next);clear(d.q_bits);clear(d.k_bits);for(unsigned l=0;l<L;l++)if(beat*L+l<128){put(d.q_bits,16*l,v[next].q[beat*L+l]);put(d.k_bits,16*l,v[next].k[beat*L+l]);}}
  d.eval();check(!ohold||(d.out_valid&&d.out_tag==heldtag&&d.score_bits==heldbits),"backpressure hold");ohold=d.out_valid&&!d.out_ready;heldtag=d.out_tag;heldbits=d.score_bits;outputstall+=ohold;inputstall+=d.in_valid&&!d.in_ready;
  for(unsigned l=0;l<L;l++){
   if(get(d.fp_complete,l,1)){check(!pending[l].empty(),"unrequested completion");auto p=pending[l].front();pending[l].pop_front();int frame=uint32_t(p.tag);auto tg=get(d.complete_tags,48*l,48);int node=get(d.complete_nodes,8*l,8),bits=get(d.complete_bits,16*l,16);check(c==p.cycle+3&&tg==p.tag&&node==p.node,"pipeline tag/latency");if(bits!=v[frame].expected[node]){std::cerr<<"mismatch frame="<<frame<<" node="<<node<<" lane="<<l<<" got="<<std::hex<<bits<<" want="<<v[frame].expected[node]<<"\n";return 2;}done[frame][node]=true;completes++;if(frame<8)events<<c<<",COMPLETE,"<<frame<<","<<l<<","<<node<<","<<bits<<"\n";}
   if(get(d.fp_issue,l,1)){auto tg=get(d.issue_tags,48*l,48);int frame=uint32_t(tg),node=get(d.issue_nodes,8*l,8);check(frame<int(n)&&node<int(R)&&tg==tag(frame),"issue tag bounds");check(!issued[frame][node],"duplicate FP node");issued[frame][node]=true;count[frame]++;issues++;pending[l].push_back({c,tg,node});if(frame<8)events<<c<<",ISSUE,"<<frame<<","<<l<<","<<node<<",0\n";}
  }
  if(d.out_valid){int frame=uint32_t(d.out_tag);check(frame==int(outputs)&&d.out_tag==tag(frame)&&d.score_bits==v[frame].expected.back(),"output score/order");if(valid[frame]==~uint64_t(0))valid[frame]=c;if(d.out_ready){check(count[frame]==R&&std::all_of(done[frame].begin(),done[frame].end(),[](bool x){return x;}),"score request/completion count");frames<<frame<<","<<start[frame]<<","<<valid[frame]<<","<<c<<","<<valid[frame]-start[frame]<<","<<d.score_bits<<"\n";outputs++;}}
  ihold=d.in_valid&&!d.in_ready;if(d.in_valid&&d.in_ready){if(beat==0)start[next]=c;if((beat+1)*L>=128){next++;beat=0;}else beat++;}tick();
  if(outputs==n&&d.empty){for(auto&p:pending)check(p.empty(),"pipeline drain");std::ofstream s(std::string(argv[2])+"_stats.json");s<<"{\"outputs\":"<<outputs<<",\"cycles\":"<<c+1<<",\"issues\":"<<issues<<",\"completes\":"<<completes<<",\"lanes\":"<<L<<",\"requests_per_score\":"<<R<<",\"output_stall_cycles\":"<<outputstall<<",\"input_stall_cycles\":"<<inputstall<<",\"resets\":"<<resets<<",\"bit_exact\":true}\n";std::cout<<"PASS D"<<L<<" outputs="<<outputs<<" cycles="<<c+1<<" requests="<<issues<<" resets="<<resets<<"\n";return 0;}
 }
 throw std::runtime_error("timeout");
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
