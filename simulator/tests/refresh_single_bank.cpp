// Exact-cycle boundaries for the same-bank single-bank refresh (REFsb) constraints that
// the per-bank refresh policy relies on (nRP, nRC, nRFCSB, nRREFD), checked on Phase20HBM3.
#include "base/base.h"
#include "dram/dram.h"
#include <iostream>
#include <stdexcept>
using namespace Ramulator;
static IDRAM* make(){
 auto cfg=YAML::Load("DRAM:\n  impl: Phase20HBM3\n  org:\n    preset: HBM3_8Gb_2R\n    channel: 1\n  timing:\n    preset: HBM3_5.2Gbps\n");
 auto d=dynamic_cast<IDRAM*>(Factory::create_implementation("DRAM","Phase20HBM3",cfg,nullptr));
 if(!d)throw std::runtime_error("DRAM cast");return d;
}
static void expect_block(IDRAM* d,const char* cmd,const AddrVec_t& a,int cycles,const char* what){
 for(int i=0;i<cycles;i++){if(d->check_ready(d->m_commands(cmd),a))throw std::runtime_error(std::string(what)+": ready too early at +"+std::to_string(i));d->tick();}
 if(!d->check_ready(d->m_commands(cmd),a))throw std::runtime_error(std::string(what)+": still blocked at boundary");
}
int main(){try{
 // Retained REFab boundaries.
 {auto d=make();AddrVec_t a={0,0,0,0,0,0,0},pc={0,0,-1,-1,-1,-1,-1};
  d->issue_command(d->m_commands("ACT"),a);for(int i=0;i<200;i++)d->tick();
  d->issue_command(d->m_commands("PREA"),pc);expect_block(d,"REFab",pc,d->m_timing_vals("nRP"),"PREA->REFab nRP");
  d->issue_command(d->m_commands("REFab"),pc);expect_block(d,"ACT",a,d->m_timing_vals("nRFC"),"REFab->ACT nRFC");}
 auto d=make();
 int rp=d->m_timing_vals("nRP"),rfcsb=d->m_timing_vals("nRFCSB"),rrefd=d->m_timing_vals("nRREFD"),rc=d->m_timing_vals("nRC");
 AddrVec_t b0={0,0,0,0,0,-1,-1},r0={0,0,0,0,0,7,0},b1={0,0,0,1,0,-1,-1},r1={0,0,0,1,0,9,0},pc={0,0,-1,-1,-1,-1,-1};
 // ACT->REFsb same bank needs nRC even though the bank must also be precharged.
 d->issue_command(d->m_commands("ACT"),r0);
 if(d->get_preq_command(d->m_commands("REFsb"),b0)!=d->m_commands("PRE"))throw std::runtime_error("REFsb must require a closed bank");
 for(int i=0;i<200;i++)d->tick();
 d->issue_command(d->m_commands("PRE"),r0);expect_block(d,"REFsb",b0,rp,"PRE->REFsb nRP");
 d->issue_command(d->m_commands("REFsb"),b0);
 // Different bank in the same rank: nRREFD to the next REFsb. Same bank: nRFCSB to ACT/REFsb.
 expect_block(d,"REFsb",b1,rrefd,"REFsb->REFsb(other bank) nRREFD");
 {auto e=make();e->issue_command(e->m_commands("REFsb"),b0);expect_block(e,"ACT",r0,rfcsb,"REFsb->ACT(same bank) nRFCSB");}
 {auto e=make();e->issue_command(e->m_commands("REFsb"),b0);expect_block(e,"REFsb",b0,rfcsb,"REFsb->REFsb(same bank) nRFCSB");}
 {auto e=make();e->issue_command(e->m_commands("REFsb"),b0);expect_block(e,"REFab",pc,rfcsb,"REFsb->REFab nRFCSB");}
 {auto e=make();e->issue_command(e->m_commands("ACT"),r1);for(int i=0;i<200;i++)e->tick();
  // A REFsb to a closed bank is not delayed by an open row in another bank beyond rank rules.
  if(e->get_preq_command(e->m_commands("REFsb"),b0)!=e->m_commands("REFsb"))throw std::runtime_error("independent bank");}
 std::cout<<"{\"phase20_refresh_boundaries_passed\":true,\"nRP\":"<<rp<<",\"nRC\":"<<rc<<",\"nRFCSB\":"<<rfcsb<<",\"nRREFD\":"<<rrefd<<",\"nREFI\":"<<d->m_timing_vals("nREFI")<<"}"<<std::endl;
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
