#include "base/base.h"
#include "dram/dram.h"
#include <iostream>
#include <stdexcept>
using namespace Ramulator;
void test(const char* name,bool fixed){
 auto cfg=YAML::Load("DRAM:\n  impl: "+std::string(name)+"\n  org:\n    preset: HBM3_8Gb_2R\n    channel: 1\n  timing:\n    preset: HBM3_5.2Gbps\n");
 auto impl=Factory::create_implementation("DRAM",name,cfg,nullptr);
 auto d=dynamic_cast<IDRAM*>(impl);if(!d)throw std::runtime_error("DRAM cast");
 AddrVec_t a={0,0,0,0,0,0,0},b={0,1,0,0,0,0,0},pc={0,0,-1,-1,-1,-1,-1};
 d->issue_command(d->m_commands("ACT"),a);for(int i=0;i<100;i++)d->tick();
 d->issue_command(d->m_commands("ACT"),b);for(int i=0;i<100;i++)d->tick();
 d->issue_command(d->m_commands("PREA"),pc);for(int i=0;i<3;i++)d->tick();
 bool trp=!d->check_ready(d->m_commands("REFab"),pc);
 for(int i=0;i<100;i++)d->tick();
 bool isolated=d->get_preq_command(d->m_commands("RD"),b)==d->m_commands("RD");
 d->issue_command(d->m_commands("REFab"),pc);for(int i=0;i<3;i++)d->tick();
 bool trfc=!d->check_ready(d->m_commands("ACT"),a);
 std::cout<<"{\"implementation\":\""<<name<<"\",\"other_pseudochannel_row_preserved\":"<<(isolated?"true":"false")
          <<",\"PREA_to_REF_tRP_enforced\":"<<(trp?"true":"false")
          <<",\"tRFC_enforced\":"<<(trfc?"true":"false")
          <<",\"expected_result_confirmed\":"<<((isolated==fixed&&trp==fixed&&trfc)?"true":"false")<<"}"<<std::endl;
 if(isolated!=fixed||trp!=fixed||!trfc)throw std::runtime_error("Unexpected scope/timing result");
}
void boundaries(){
 auto cfg=YAML::Load("DRAM:\n  impl: Phase20HBM3\n  org:\n    preset: HBM3_8Gb_2R\n    channel: 1\n  timing:\n    preset: HBM3_5.2Gbps\n");
 auto impl=Factory::create_implementation("DRAM","Phase20HBM3",cfg,nullptr);
 auto d=dynamic_cast<IDRAM*>(impl);
 AddrVec_t a={0,0,0,0,0,0,0},pc={0,0,-1,-1,-1,-1,-1};
 d->issue_command(d->m_commands("ACT"),a);
 for(int i=0;i<200;i++)d->tick();
 d->issue_command(d->m_commands("PREA"),pc);
 int rp=d->m_timing_vals("nRP"),rfc=d->m_timing_vals("nRFC");
 for(int i=0;i<rp;i++){
  if(d->check_ready(d->m_commands("REFab"),pc))throw std::runtime_error("REF before tRP");
  d->tick();
 }
 if(!d->check_ready(d->m_commands("REFab"),pc))throw std::runtime_error("REF blocked at tRP");
 d->issue_command(d->m_commands("REFab"),pc);
 for(int i=0;i<rfc;i++){
  if(d->check_ready(d->m_commands("ACT"),a))throw std::runtime_error("ACT before tRFC");
  d->tick();
 }
 if(!d->check_ready(d->m_commands("ACT"),a))throw std::runtime_error("ACT blocked at tRFC");
 std::cout<<"{\"timing_boundaries_passed\":true,\"nRP\":"<<rp<<",\"nRFC\":"<<rfc<<"}"<<std::endl;
}
int main(){try{boundaries();test("HBM3",false);test("Phase20HBM3",true);}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
