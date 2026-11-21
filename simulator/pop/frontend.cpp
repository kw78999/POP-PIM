// Live GQA4 metadata ownership/router RTL + calibrated scorer and completion model.
#include "../common/gqa_hw.h"
#include "frontend/frontend.h"
#include "dram/dram.h"
#include "dram_controller/controller.h"
#include "dram_controller/plugin.h"
#include "../common/resources.h"
#include "../common/selector_cluster_sram.h"
#include <array>
#include "../common/diagnostics.h"
#include <fstream>
#include <set>
#include <map>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <climits>
#include <iostream>
namespace phase13 {
struct Counters {uint64_t act=0,pre=0,prea=0,wr=0,ref=0,refsb=0,rd=0,hits=0,misses=0,conflicts=0,refresh_bank_cycles=0;};
inline Counters counters;
inline std::ofstream* trace_sink=nullptr;
}
namespace Ramulator {
class Phase13Commands final:public IControllerPlugin,public Implementation {
 RAMULATOR_REGISTER_IMPLEMENTATION(IControllerPlugin,Phase13Commands,"Phase13Commands","Non-mutating HBM3 command/first-service row counters")
 std::ofstream command_trace;
 IDRAM* dram=nullptr; std::map<uint64_t,int> classification;
 public:
 void init() override {std::string p=param<std::string>("trace_file").default_val("");if(!p.empty())command_trace.open(p,std::ios::binary);phase13::trace_sink=&command_trace;}
 void setup(IFrontEnd*,IMemorySystem*) override {m_ctrl=cast_parent<IDRAMController>();dram=m_ctrl->m_dram;}
 void update(bool found,ReqBuffer::iterator& it) override {
  if(!found)return;
  auto name=dram->m_commands(it->command);auto& s=phase13::counters;
  if(command_trace){uint64_t stamp=poppim_hw::hw.cycle*769,mask=0,code=name=="ACT"?1:name=="PRE"?2:name=="PREA"?3:(name=="RD"||name=="RDA")?4:(name=="WR"||name=="WRA")?5:name=="REFab"?6:name=="REFsb"?8:7;for(int b=0;b<64;b++)if(poppim_hw::hw.targets(it->addr_vec,b))mask|=uint64_t(1)<<b;command_trace.write((char*)&stamp,8);command_trace.write((char*)&mask,8);command_trace.write((char*)&code,8);}
  if(name=="ACT")s.act++;
  if(name=="PREA")s.prea++;
  if(name=="WR"||name=="WRA")s.wr++;
  if(name=="PRE"||name=="PREA")s.pre++;
  if(name=="REFsb") {s.refsb++;s.refresh_bank_cycles+=dram->m_timing_vals("nRFCSB");}
  if(name=="REFab") {s.ref++;unsigned n=0;for(int b=0;b<64;b++)if(poppim_hw::hw.targets(it->addr_vec,b))n++;s.refresh_bank_cycles+=uint64_t(n)*dram->m_timing_vals("nRFC");}
  if(it->type_id==Request::Type::Read){
   auto key=uint64_t(it->addr);
   if(!classification.count(key))classification[key]=name=="ACT"?1:(name=="PRE"||name=="PREA"?2:0);
   if(name=="RD"||name=="RDA") {s.rd++;int c=classification.at(key);if(c==0)s.hits++;if(c==1)s.misses++;if(c==2)s.conflicts++;classification.erase(key);}
  }
 }
};
class Phase13Frontend final:public IFrontEnd,public Implementation {
 RAMULATOR_REGISTER_IMPLEMENTATION(IFrontEnd,Phase13Frontend,"Phase13Common","Common bounded 64-bank architectural adapter; preserved HBM3 and exact selector")
 struct Task {int kv=0;std::vector<int> tokens;std::array<int,4> counts{};int scores=0;};
 struct Bank {
  int kv=0,tile=0,state=0,next=0,done=0,outstanding=0;
  bool finished=false;
  std::vector<uint64_t> addresses;
  std::array<uint64_t,16> ready{};std::array<bool,16> consumed{};
  uint64_t until=0;
  std::vector<Task> tasks;size_t task=0;
 };
 std::array<Bank,64> banks;
 std::unique_ptr<poppim::Cluster> cluster;
 std::vector<uint16_t> scores;std::vector<int> expected;
 std::string output,design,mode,score_file,expected_file;
 uint64_t metadata_base=4*1024*1024;
 int N=32768,K=4096,replicas=1,latency=43,engines=4,ports=4,mac_units=1024,exp_units=16,reduction_units=64,local_ports=16;
 double ii=22;
 bool pop=false,finished=false,selected=false;
 bool refresh_drain=false;
 uint64_t cycle=0,next_score=8000,next_selector=4000;
 uint64_t reads=0,callbacks=0,k_reads=0,v_reads=0,meta_reads=0,selected_reads=0,rejected=0,peak=0;
 uint64_t score_count=0,score_jobs=0,first_score=0,last_score=0,topk_ready=0,first_selected=0,last_selected=0,softmax_ready=0,av_ready=0,attention_ready=0;
 uint64_t first_read=0,last_read=0,latency_sum=0,bank_busy_ticks=0,input_wait_ticks=0,output_wait_ticks=0,refresh_wait_ticks=0;
 uint64_t backend_free=0,backend_busy=0,backend_qk=0,backend_softmax=0,backend_av=0,backend_queue_wait=0,final_norm=0;
 uint64_t unique_selected=0,exact_qk_macs=0,av_macs=0,exp_ops=0,reduction_ops=0,rescale_macs=0;
 std::ofstream milestones;
 // VS: two-phase K->V gather with mass-guided V skipping.
 bool two_phase=false,vphase=false;std::string vkeep_file;std::vector<uint8_t> vkeep;
 std::array<std::map<int,unsigned>,8> kv_membership;
 uint64_t p21_k_reads=0,p21_v_reads=0,p21_kphase_end=0,p21_vphase_start=0,p21_kept_v_tokens=0,p21_v_pairs=0;
#include "../common/pop_control.inc"
#include "../common/offload.inc"
 static uint64_t ceildiv(uint64_t a,uint64_t b){return (a+b-1)/b;}
 uint64_t address(int b,uint64_t byte)const {uint64_t line=byte/32,row=line/32,col=line%32;int pch=b&1,bg=(b>>1)&3,rank=(b>>3)&1,ba=(b>>4)&3;return (((((row*4+ba)*2+rank)*32+col)*4+bg)*2+pch)*32;}
 uint64_t now()const{return cycle*769;}
 void setup_addresses(int b) {
  auto&x=banks[b];x.addresses.clear();x.next=x.done=x.outstanding=0;
  if(selected){
   if(x.task>=x.tasks.size()){x.finished=true;return;}
   auto&t=x.tasks[x.task];for(int token:t.tokens)for(int which=0;which<2;which++)if(!two_phase||which==(vphase?1:0))for(int beat=0;beat<8;beat++)x.addresses.push_back((uint64_t(which)*8*N/64+uint64_t(t.kv)*N/64+token/64)*256+beat*32);
  }else{
   uint64_t base=uint64_t(x.kv)*(N/64)*(pop?40:256)+uint64_t(x.tile)*(pop?160:1024);
   if(pop)base+=metadata_base;
   if(x.state==2)base+=uint64_t(8)*N/64*256;
   for(int j=0;j<(pop?5:32);j++)x.addresses.push_back(base+j*32);
  }
 }
 void advance(int b){auto&x=banks[b];if(selected){x.task++;x.state=0;setup_addresses(b);return;}x.tile++;if(x.tile==N/64/4){x.tile=0;x.kv++;}if(x.kv==8){x.finished=true;return;}x.state=0;setup_addresses(b);}
 void schedule_scores(int b) {
  auto&x=banks[b];uint64_t start=ceildiv(now(),8000)*8000;
  std::vector<uint64_t> free(replicas,start);std::vector<double> issue_phase(replicas,0);uint64_t input_free=start,output_free=start;
  x.consumed.fill(false);
  // Explicit SRAM input port: 16 B/bank/125MHz; no free operand broadcast.
  // Reuse is on HBM reads only. Every head/engine pays operand-port service.
  for(int h=0;h<4;h++)for(int t=0;t<4;t++){
   int e=std::min_element(free.begin(),free.end())-free.begin();
   uint64_t issue=std::max(free[e],input_free);
   uint64_t input_cost=ceildiv(pop?40:256,16)*8000;
   input_free=issue+input_cost;
   double old_phase=issue_phase[e];issue_phase[e]+=ii;
   // Fractional measured mean II uses an integer 22/23-cycle issue pattern.
   free[e]=issue+uint64_t(std::ceil(issue_phase[e])-std::ceil(old_phase))*8000;
   uint64_t ready=std::max(issue+input_cost+uint64_t(latency)*8000,output_free);
   output_free=ready+8000; // one BF16 score/bank/scorer tick
   x.ready[h*4+t]=ready;
   first_score=first_score?std::min(first_score,ready):ready;last_score=std::max(last_score,ready);score_count++;
  }
  x.until=*std::max_element(x.ready.begin(),x.ready.end());x.state=1;score_jobs++;
 }
 uint64_t mac_time(uint64_t ops)const {
  if(!ops)return 0;
  // Both arithmetic issue and local operand bandwidth constrain service.
  return std::max(ceildiv(ops,mac_units),ceildiv(ops*4,uint64_t(local_ports)*16))*2000+8000;
 }
 void schedule_backend(int b) {
  auto&x=banks[b];std::array<int,4> counts{};int count=0;
  if(selected){counts=x.tasks[x.task].counts;count=x.tasks[x.task].scores;}else{counts.fill(4);count=16;}
  int heads=0;for(int c:counts)heads+=c>0;
  uint64_t start=std::max(now(),backend_free);backend_queue_wait+=start-now();
  bool doqk=!two_phase||!vphase,doav=!two_phase||vphase;
  uint64_t qk=selected&&doqk?mac_time(uint64_t(count)*128):0;
  if(selected&&doqk){
   // Parallel partial products require an explicit128-way exact-QK reduction.
   // Seven balanced levels, finite64-add reduction issue, then score scaling.
   qk+=ceildiv(uint64_t(count)*127,reduction_units)*2000+7*6000;
   qk+=ceildiv(uint64_t(count),16)*2000+8000;
  }
  uint64_t red=uint64_t(count)*2+heads*2;
  uint64_t ex=count+heads; // token exponential and running-max rescale
  uint64_t soft=doqk?ceildiv(red,reduction_units)*2000+6000+ceildiv(ex,exp_units)*2000+12000:0;
  uint64_t av=doav?mac_time(uint64_t(count+heads)*128):0;
  if(doqk){exact_qk_macs+=selected?uint64_t(count)*128:0;exp_ops+=ex;reduction_ops+=red+(selected?uint64_t(count)*127:0);}
  if(doav){av_macs+=uint64_t(count)*128;rescale_macs+=uint64_t(heads)*128;if(vphase)p21_v_pairs+=count;}
  backend_qk+=qk;backend_softmax+=soft;backend_av+=av;backend_busy+=qk+soft+av;
  softmax_ready=std::max(softmax_ready,start+qk+soft);
  backend_free=start+qk+soft+av;av_ready=backend_free;x.until=backend_free;x.state=3;
 }
 void bank_edge(int b) {
  bank_edge_inner(b);
  // PIM hint: waiting on the shared backend needs neither DRAM access nor the scorer.
  const auto&x=banks[b];
  poppim_hw::hw.quiet[b]=(!pop||selected)&&(x.finished||(x.state==3&&now()<x.until));
 }
 bool draining(int b)const{return refresh_drain&&phase20diag::drain_reads&&phase17diag::refresh_pending[b];}
 void bank_edge_inner(int b) {
  if(pop&&!selected){pop_edge13(b);return;}
  auto&x=banks[b];if(x.finished){poppim_hw::hw.compute[b]=false;return;}
  bool refresh=cycle<poppim_hw::hw.refresh_until[b];if(refresh)refresh_wait_ticks++;
  // Existing controller excludes DRAM commands during scorer occupancy.
  poppim_hw::hw.compute[b]=!selected&&x.state==1&&now()<x.until;
  if(poppim_hw::hw.compute[b])bank_busy_ticks++;
  if(x.state==1){
   bool drained=!pop||std::all_of(x.consumed.begin(),x.consumed.end(),[](bool z){return z;});
   if(now()<x.until)return;
   if(!drained){output_wait_ticks++;return;}
   if(mode=="score"||pop){advance(b);return;}
   x.state=2;setup_addresses(b); // V only after its QK tile is ready
  }
  if(x.state==3){if(now()>=x.until)advance(b);else return;}
  if(x.finished)return;
  if(x.state==0||x.state==2){
   if(x.done==int(x.addresses.size())){
    if(refresh)return;
    if(selected){first_selected=first_selected?std::min(first_selected,now()):now();last_selected=std::max(last_selected,now());schedule_backend(b);}
    else if(x.state==2)schedule_backend(b);
    else {
     if(refresh_drain&&phase17diag::refresh_pending[b]){phase17diag::drain_wait_bank_ticks++;return;}
     schedule_scores(b);
    }
    return;
   }
   if(!selected&&x.state==0)input_wait_ticks++;
   if(x.next<int(x.addresses.size())&&x.outstanding<2&&!refresh&&!draining(b)){
    uint64_t submitted=cycle;
    Request req(address(b,x.addresses[x.next]),Request::Type::Read,b,[this,b,submitted](Request&r){
     int mapped=r.addr_vec[1]+(r.addr_vec[3]<<1)+(r.addr_vec[2]<<3)+(r.addr_vec[4]<<4);
     if(mapped!=b)throw std::runtime_error("bank address mismatch");
     auto&z=banks[b];z.done++;z.outstanding--;callbacks++;last_read=now();latency_sum+=cycle-submitted;
    });
    if(m_memory_system->send(req)){
     x.next++;x.outstanding++;reads++;if(!first_read)first_read=now();
     if(selected){selected_reads++;if(vphase)p21_v_reads++;else p21_k_reads++;}else if(pop)meta_reads++;else if(x.state==2)v_reads++;else k_reads++;
     peak=std::max(peak,reads-callbacks);
    }else rejected++;
   }
  }
 }
 bool available(int h,int token) {
  if(pop)return pop_available13(h,token);
  int b=token%64;const auto&x=banks[b];int local=token/64-x.tile*4;
  return !x.finished&&x.state==1&&h/4==x.kv&&local>=0&&local<4&&!x.consumed[(h%4)*4+local]&&x.ready[(h%4)*4+local]<=now();
 }
 uint16_t take(int h,int token){if(pop)return pop_take13(h,token);if(!available(h,token))throw std::runtime_error("selector score unavailable");auto&x=banks[token%64];x.consumed[(h%4)*4+token/64-x.tile*4]=true;return scores.at(h*N+token);}
 void begin_selected(){
  if(!offload)for(auto&j:cluster->jobs){if(j.result.size()!=size_t(K))throw std::runtime_error("K count");for(int i=0;i<K;i++)if(j.result[i]!=expected[j.head*K+i])throw std::runtime_error("exact Top-k mismatch");}
  topk_ready=now();
  if(mode=="score"){attention_ready=0;finished=true;return;}
  selected=true;
  std::ofstream union_file(output+"/selected_union.csv");union_file<<"kv_head,unique_tokens\n";
  for(auto&x:banks){x.finished=false;x.state=0;x.tasks.clear();x.task=0;}
  for(int kv=0;kv<8;kv++){
   std::map<int,unsigned> membership;for(int h=0;h<4;h++)for(int token:(offload?external_indices[kv*4+h]:cluster->jobs[kv*4+h].result))membership[token]|=1u<<h;
   unique_selected+=membership.size();union_file<<kv<<','<<membership.size()<<'\n';kv_membership[kv]=membership;
  }
  build_tasks(false);
 }
 void build_tasks(bool vonly){
  for(auto&x:banks){x.finished=false;x.state=0;x.tasks.clear();x.task=0;}
  for(int kv=0;kv<8;kv++){
   for(int b=0;b<64;b++){
    Task t;t.kv=kv;
    for(auto [token,mask]:kv_membership[kv])if(token%64==b&&(!vonly||vkeep.at(size_t(kv)*N+token))){if(vonly)p21_kept_v_tokens++;t.tokens.push_back(token);for(int h=0;h<4;h++)if(mask&(1u<<h)){t.counts[h]++;t.scores++;}if(t.tokens.size()==4){banks[b].tasks.push_back(t);t=Task{};t.kv=kv;}}
    if(!t.tokens.empty())banks[b].tasks.push_back(t);
   }
  }
  for(int b=0;b<64;b++){poppim_hw::hw.compute[b]=false;setup_addresses(b);}
 }
 public:
 void init()override {
  output=param<std::string>("output").required();design=param<std::string>("design").required();mode=param<std::string>("mode").required();ii=param<double>("ii").required();latency=param<int>("latency").required();replicas=param<int>("replicas").default_val(1);pop=design=="A2-S1";
  refresh_drain=param<bool>("refresh_drain").default_val(false);
  N=param<int>("context_length").default_val(32768);K=param<int>("budget").default_val(4096);
  if(N<256||N>131072||N%256||K<1||K>=N)throw std::runtime_error("unsupported context/budget");
  metadata_base=std::max(uint64_t(4*1024*1024),uint64_t(8)*N/64*512);
  if(metadata_base+uint64_t(8)*N/64*40>16*1024*1024)throw std::runtime_error("HBM bank capacity exceeded");
  coupled_release=param<bool>("coupled_release").default_val(false);engines=param<int>("selector_engines").default_val(4);ports=param<int>("selector_ports").default_val(4);
  mac_units=param<int>("mac_units").default_val(1024);local_ports=param<int>("local_ports").default_val(16);
  vkeep_file=param<std::string>("vkeep_file").default_val("");two_phase=!vkeep_file.empty();
  if(two_phase){vkeep.resize(size_t(8)*N);std::ifstream vf(vkeep_file,std::ios::binary);vf.read((char*)vkeep.data(),vkeep.size());if(!vf)throw std::runtime_error("missing vkeep file");}
  offload=param<bool>("offload").default_val(false);ideal_sink=param<bool>("ideal_sink").default_val(false);
  if(pop){
   score_file=param<std::string>("score_file").required();expected_file=param<std::string>("expected_file").required();
   scores.resize(32*N);expected.resize(32*K);std::ifstream f(score_file,std::ios::binary),g(expected_file,std::ios::binary);f.read((char*)scores.data(),scores.size()*2);g.read((char*)expected.data(),expected.size()*4);if(!f||!g)throw std::runtime_error("missing selector trace");
   std::vector<poppim::Job> jobs(32);for(int h=0;h<32;h++){jobs[h].n=N;jobs[h].k=K;jobs[h].head=h;jobs[h].preferred=-1;}
   if(!offload)cluster=std::make_unique<poppim::Cluster>(jobs,engines,ports,ports,16,4,1,false);if(cluster){cluster->word_model=true;cluster->word_bits=32;cluster->sram_latency=2;cluster->sram_port_II=1;}
  }
  if(offload)init_offload();
  if(pop)initialize_pop13();
  for(int b=0;b<64;b++)setup_addresses(b);
 }
 uint64_t progress_signature=~uint64_t(0);
 void progress_watchdog(){
  if(!pop||offload||cycle%1000000!=0)return;
  uint64_t work=0;for(auto& e:cluster->engines)work+=e.work;
  uint64_t signature=score_count+reads+callbacks+cluster->ingested+cluster->returned+rtl_operands+rtl_releases+work;
  if(signature==progress_signature){
   std::ofstream d(output+"/no_progress.json");d<<"{\"hbm_cycles\":"<<cycle<<",\"score_count\":"<<score_count<<",\"operands\":"<<rtl_operands<<",\"router_enqueues\":"<<rtl_router_enqueues<<",\"router_dequeues\":"<<rtl_router_dequeues<<",\"selector_ingested\":"<<cluster->ingested<<",\"selector_returned\":"<<cluster->returned<<",\"engine_work_ticks\":"<<work<<",\"engines\":[";
   for(int e=0;e<engines;e++){if(e)d<<",";auto& x=cluster->engines[e];d<<"{\"engine\":"<<e<<",\"job\":"<<x.job<<",\"state\":"<<(x.core?int(x.core->state):-1)<<",\"produced\":"<<(x.job>=0?cluster->jobs[x.job].produced:-1)<<"}";}d<<"]}\n";d.close();
   throw std::runtime_error("No progress over 1M HBM cycles: inspect no_progress.json");
  }
  progress_signature=signature;
 }
 std::chrono::steady_clock::time_point wall_start=std::chrono::steady_clock::now();
 void tick()override {
  cycle++;progress_watchdog();
  if(cycle%5000000==0){double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-wall_start).count();double fraction=std::min(1.,double(score_count)/(32.*N));std::cerr<<"PROGRESS context="<<N<<" budget="<<K<<" score_percent="<<fraction*100<<" elapsed_s="<<seconds<<" score_remaining_estimate_s="<<(fraction>0?seconds*(1-fraction)/fraction:-1)<<" selected_backend="<<selected<<" hbm_cycles="<<cycle<<std::endl;}
 if(cycle>1000000000)throw std::runtime_error("1B HBM-cycle watchdog");if(finished)return;
  while(now()>=next_selector){
   if(offload&&!selected)offload_tick();
   if(pop&&!offload&&!selected&&!cluster->finished())cluster->tick([this](int h,int t){return available(h,t);},[this](int h,int t){return take(h,t);});
   next_selector+=4000;
  }
  while(now()>=next_score){for(int b=0;b<64;b++)bank_edge(b);next_score+=8000;}
  if(offload&&!selected&&external_ready&&now()>=external_ready){begin_selected();return;}
  if(pop&&!offload&&!selected&&cluster->finished()){begin_selected();return;}
  if((!pop||selected)&&std::all_of(banks.begin(),banks.end(),[](const Bank&x){return x.finished;})){
   if(two_phase&&selected&&!vphase){
    // Global barrier: V selection needs final softmax statistics of every selected K row.
    if(now()<backend_free)return;
    p21_kphase_end=now();vphase=true;build_tasks(true);p21_vphase_start=now();return;
   }
   if(mode=="full"){
    final_norm=std::max(ceildiv(uint64_t(32)*128,mac_units),ceildiv(uint64_t(32)*128*8,uint64_t(local_ports)*16))*2000+8000;
    attention_ready=std::max(now(),backend_free)+final_norm;
    // Final normalization uses the common completion backend after all tiles.
    if(now()<attention_ready){backend_free=attention_ready;mode="finalizing";return;}
   }
   if(mode=="finalizing"&&now()<attention_ready)return;
   finished=true;
  }
 }
 bool is_finished()override{return finished;}
 void finalize()override {
  if(phase13::trace_sink)phase13::trace_sink->flush();
  if(score_count!=uint64_t(32)*N||reads!=callbacks)throw std::runtime_error("workload/read conservation");
  if(pop&&!offload&&(cluster->ingested!=uint64_t(32)*N||cluster->returned!=uint64_t(32)*K))throw std::runtime_error("selector conservation");
  if(pop&&(rtl_operands!=uint64_t(64)*N||rtl_router_enqueues!=uint64_t(32)*N||rtl_router_dequeues!=uint64_t(32)*N||rtl_releases!=uint64_t(2)*N))throw std::runtime_error("RTL integrated conservation");
  if(pop&&!offload){
   std::ofstream ef(output+"/selector_engines.csv");ef<<"engine,busy_ticks,work_ticks,read_stall_ticks,counter_stall_ticks,input_stall_ticks,output_stall_ticks\n";
   for(int e=0;e<engines;e++){auto& a=cluster->engines[e];ef<<e<<","<<a.busy<<","<<a.work<<","<<a.read_stall<<","<<a.counter_stall<<","<<a.input_stall<<","<<a.output_stall<<"\n";}
   std::ofstream jf(output+"/selector_jobs.csv");jf<<"head,start_tick,completion_tick\n";
   for(auto& j:cluster->jobs)jf<<j.head<<","<<j.start<<","<<j.completion<<"\n";
  }
  auto&s=phase13::counters;std::ofstream f(output+"/metrics.json");f<<std::setprecision(17);
  f<<"{\n";
  bool first=true;auto num=[&](const char*k,auto v){if(!first)f<<",\n";first=false;f<<"  \""<<k<<"\": "<<v;};
  num("priority_compute_stall_hbm_cycles",phase17diag::priority_compute_stalls);
  num("priority_timing_stall_hbm_cycles",phase17diag::priority_timing_stalls);
  num("priority_queue_peak",phase17diag::priority_peak);
  num("refresh_drain_wait_bank_score_ticks",phase17diag::drain_wait_bank_ticks);
  num("refresh_drain_enabled",refresh_drain?1:0);
  if(phase20diag::policy!=0){
   int64_t dmax=INT64_MIN,dmin=INT64_MAX,imax=INT64_MIN,imin=INT64_MAX;
   for(int b=0;b<64;b++){dmax=std::max(dmax,phase20diag::final_debt[b]);dmin=std::min(dmin,phase20diag::final_debt[b]);imax=std::max(imax,phase20diag::issued[b]);imin=std::min(imin,phase20diag::issued[b]);}
   num("phase20_refresh_policy",phase20diag::policy);num("REFsb",s.refsb);
   num("phase20_refsb_urgent",phase20diag::refsb_urgent);num("phase20_refsb_quiet",phase20diag::refsb_quiet);num("phase20_refresh_pre",phase20diag::refresh_pre);
   num("phase20_wc_bypass_cycles",phase20diag::wc_bypass_cycles);num("phase20_drain_bank_hbm_cycles",phase20diag::urgent_bank_cycles);
   num("phase20_max_debt",phase20diag::max_debt);num("phase20_debt_violations",phase20diag::violations);
   num("phase20_final_debt_max",phase20diag::policy==2?dmax:0);num("phase20_final_debt_min",phase20diag::policy==2?dmin:0);
   num("phase20_refsb_per_bank_min",phase20diag::policy==2?imin:0);num("phase20_refsb_per_bank_max",phase20diag::policy==2?imax:0);
   num("phase20_postpone",phase20diag::postpone);num("phase20_pullin",phase20diag::pullin);num("phase20_headroom",phase20diag::headroom);
   num("phase20_round_rule",phase20diag::round_rule?1:0);num("phase20_nREFI",phase20diag::nREFI);
  }
  num("metadata_base_per_bank",metadata_base);num("context_length",N);num("budget",K);num("rtl_local_length_bits",12);
  num("hbm_period_ps",769);num("score_period_ps",8000);num("selector_period_ps",4000);num("backend_period_ps",2000);num("hbm_cycles",cycle);num("global_end_ps",now());num("score_clock_ticks",(next_score/8000)-1);num("selector_clock_ticks",cluster?cluster->cycle:0);
  num("score_count",score_count);num("score_tiles",score_jobs);num("replicas_per_bank",replicas);num("active_score_engines",64*replicas);num("score_ii",ii);num("score_lower_bound_ps",std::ceil(32.0*N*ii*8000/(64*replicas)));num("first_score_ps",first_score);num("last_score_ps",last_score);num("topk_ready_ps",topk_ready);num("first_selected_kv_ready_ps",first_selected);num("last_selected_kv_ready_ps",last_selected);num("softmax_ready_ps",softmax_ready);num("av_ready_ps",av_ready);num("attention_output_ps",attention_ready);
  num("read_requests",reads);num("read_bytes",reads*32);num("k_read_requests",k_reads);num("v_read_requests",v_reads);num("metadata_read_requests",meta_reads);num("selected_kv_read_requests",selected_reads);num("callbacks",callbacks);num("outstanding_peak",peak);num("read_admission_stalls",rejected);num("first_read_ps",first_read);num("last_read_ps",last_read);num("read_latency_sum_hbm_cycles",latency_sum);
  num("ACT",s.act);num("PRE",s.pre);num("REF",s.ref);num("RD",s.rd);num("row_hits",s.hits);num("row_misses",s.misses);num("row_conflicts",s.conflicts);num("refresh_issued_bank_hbm_cycles",s.refresh_bank_cycles);num("refresh_observed_bank_score_ticks",refresh_wait_ticks);num("bank_busy_score_ticks",bank_busy_ticks);num("score_input_wait_bank_ticks",input_wait_ticks);num("score_output_wait_bank_ticks",output_wait_ticks);num("bank_command_conflict_probes",poppim_hw::hw.blocked);
  num("backend_busy_ps",backend_busy);num("backend_qk_ps",backend_qk);num("backend_softmax_ps",backend_softmax);num("backend_av_ps",backend_av);num("backend_queue_wait_sum_ps",backend_queue_wait);num("final_normalization_ps",final_norm);num("selected_union_tokens",unique_selected);num("selected_exact_qk_macs",exact_qk_macs);num("av_macs",av_macs);num("online_rescale_macs",rescale_macs);num("exp_ops",exp_ops);num("reduction_ops",reduction_ops);
  num("rtl_operands",rtl_operands);num("rtl_metadata_releases",rtl_releases);num("rtl_release_before_output_drain",rtl_release_independent);num("rtl_router_enqueues",rtl_router_enqueues);num("rtl_router_dequeues",rtl_router_dequeues);num("rtl_context_wait_bank_ticks",rtl_context_wait);num("rtl_router_full_bank_ticks",rtl_router_full);num("coupled_barrier_wait_bank_ticks",rtl_barrier_wait);num("rtl_skipped_quiescent_edges",rtl_skipped_idle_edges);num("metadata_buffers_per_bank",META_BUFFERS);num("router_heads",32);num("router_depth",4);num("selector_engines",engines);num("selector_ports",ports);num("coupled_release_enabled",coupled_release?1:0);
  num("WR",s.wr);num("PREA",s.prea);num("PRE_single",s.pre-s.prea);num("write_requests",0);num("write_bytes",0);
  if(cluster){num("score_sram_reads",cluster->score_reads);num("score_sram_writes",cluster->score_writes);num("candidate_sram_reads",cluster->candidate_reads);num("candidate_sram_writes",cluster->candidate_writes);num("counter_reads",cluster->counter_reads);num("counter_writes",cluster->counter_writes);num("histogram_updates",cluster->histogram_updates);num("radix_passes",cluster->radix_passes);num("selector_comparisons",cluster->comparisons);num("index_generation",cluster->index_generation);num("keep_reads",cluster->keep_reads);num("keep_writes",cluster->keep_writes);}
  num("phase21_two_phase",two_phase?1:0);num("phase21_selected_k_reads",p21_k_reads);num("phase21_selected_v_reads",p21_v_reads);num("phase21_kept_v_tokens",p21_kept_v_tokens);num("phase21_v_pairs",p21_v_pairs);num("phase21_kphase_end_ps",p21_kphase_end);num("phase21_vphase_start_ps",p21_vphase_start);
  num("phase21_shared_export",shared_export?1:0);num("phase21_packet_wire_bytes",packet_wire_bytes);num("phase21_merged_packets",merged);num("phase21_link_packets",delivered-merged);num("phase21_index_return_bytes",index_return_bytes);
  num("offload",offload?1:0);num("score_output_fifo_capacity_packets",fifo_capacity);num("score_output_fifo_peak_packets",fifo_peak);num("dma_queue_depth",dma_depth);num("dma_queue_peak",dma_peak);num("output_interface_stall_ticks",interface_stall);num("score_output_packets",delivered);num("score_payload_bytes",delivered*2);num("score_wire_bytes",(delivered-merged)*packet_wire_bytes);num("score_transfer_first_ps",transfer_first);num("score_transfer_last_ps",transfer_last);num("topk_start_ps",transfer_last);num("external_topk_service_ps",external_topk_ps);num("index_transfer_ps",index_transfer);num("external_indices_ready_ps",external_ready);num("output_fifo_writes",fifo_writes);num("output_fifo_reads",fifo_reads);num("offload_startup_ps",startup_ps);num("sync_ps",sync_ps);
  num("selector_ingested",cluster?cluster->ingested:0);num("selector_returned",cluster?cluster->returned:0);f<<"\n}\n";
 }
};
}
