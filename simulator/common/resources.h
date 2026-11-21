#pragma once
#include <vector>
#include <cstdint>
#include <fstream>
#include <stdexcept>
namespace poppim_hw {
// Single-channel HBM3_8Gb_2R, 2 pseudochannels x 2 ranks x 4 BG x 4 banks.
struct Shared {
    uint64_t cycle=0,commands=0,command_cycles=0,last_command_cycle=0,blocked=0;
    std::vector<bool> compute=std::vector<bool>(64,false);
    std::vector<uint64_t> refresh_until=std::vector<uint64_t>(64,0);
    // Near-bank PIM controller hint; true while the bank waits on the shared backend
    // (no DRAM access or scorer occupancy is needed until that wait ends).
    std::vector<bool> quiet=std::vector<bool>(64,false);
    std::ofstream trace;
    bool full_trace=true;
    void event(const char* op,int job=-1,int token=-1,int bank=-1,int value=0){
        if(trace && full_trace)trace<<cycle<<","<<cycle*769<<","<<op<<","<<job<<","<<token<<","<<bank<<","<<value<<"\n";
    }
    bool targets(const std::vector<int>&a,int b){
        int v[]={0,b&1,(b>>3)&1,(b>>1)&3,(b>>4)&3};
        for(int i=0;i<5;i++)if(a[i]>=0&&a[i]!=v[i])return false;
        return true;
    }
    bool blocked_by_compute(const std::vector<int>&a){
        for(int b=0;b<64;b++)if(compute[b]&&targets(a,b))return true;
        return false;
    }
    void issued(int command,const std::vector<int>&a,int refresh_id,int nRFC,int refsb_id=-1,int nRFCSB=0){
        if(blocked_by_compute(a))throw std::runtime_error("DRAM command overlapped PIM bank occupancy");
        commands++;
        if(last_command_cycle!=cycle){command_cycles++;last_command_cycle=cycle;}
        if(command==refresh_id)for(int b=0;b<64;b++)if(targets(a,b))refresh_until[b]=cycle+nRFC;
        if(refsb_id>=0&&command==refsb_id)for(int b=0;b<64;b++)if(targets(a,b))refresh_until[b]=cycle+nRFCSB;
    }
};
inline Shared hw;
}
