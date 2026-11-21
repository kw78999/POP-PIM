// Phase7 BF16 selector cycle model. BF16 finite classification, zero canonicalization, stable token ties.
#pragma once
#include <vector>
#include <cstdint>
#include <stdexcept>
namespace poppim {
inline uint16_t ordered(uint16_t b){if((b&0x7fff)==0)b=0;return b&0x8000?uint16_t(~b):b^0x8000;}
struct Core {
    int n,k,budget,state=1,load=0,scan=0,eligible=0,out=0,clear=0;
    int prefix=0,mask=0,boundary=0,rank=0,ties=0,shift=12,digit=0,hvalue=0,tkey=0;
    bool reserve=false,error=false,done=false,tkeep=false;
    std::vector<uint16_t> keys;
    std::vector<bool> keep;
    std::vector<int> hist,ids;
    uint64_t cycles=0;
    Core(int n_,int k_,bool r=false):n(n_),k(k_-int(r)),budget(k_),reserve(r),
        keys(n_),keep(n_),hist(16){
        error=k_>n_||(r&&(n_==0||k_==0));if(error||n_==0)state=10;
    }
    bool in_ready()const{return state==1;}
    bool out_valid()const{return state==8||state==9;}
    int index()const{return state==9?n-1:state==8?ids.at(out):0;}
    bool last()const{return state==9||(state==8&&out+1==k&&!reserve);}
    int counter_address()const{return state==3?clear:state==13?digit:(tkey>>shift)&15;}
    void tick(bool iv=false,uint16_t bits=0,bool ik=true,bool ready=true){
        if(done)return;cycles++;
        switch(state){
        case 1:if(iv){keys[load]=ordered(bits);keep[load]=ik&&!(reserve&&load==n-1);
            eligible+=keep[load];error|=(bits&0x7f80)==0x7f80||(reserve&&load==n-1&&!ik);
            if(++load==n)state=2;}break;
        case 2:if(error||eligible<k){error=true;state=10;}
            else if(k==0)state=reserve?9:10;
            else{rank=k;clear=0;state=3;}break;
        case 3:hist[clear++]=0;if(clear==16){scan=0;state=11;}break;
        case 11:case 12:tkey=keys[scan];tkeep=keep[scan];state=state==11?4:6;break;
        case 4:if(tkeep&&(tkey&mask)==prefix)hist[(tkey>>shift)&15]++;
            if(++scan==n){digit=15;state=13;}else state=11;break;
        case 13:hvalue=hist[digit];state=5;break;
        case 5:if(rank>hvalue){rank-=hvalue;digit--;state=13;}
            else{prefix|=digit<<shift;mask|=15<<shift;
                if(shift==0){boundary=prefix;ties=rank;scan=0;state=12;}
                else{shift-=4;clear=0;state=3;}}break;
        case 6:if(tkeep&&(tkey>boundary||(tkey==boundary&&ties>0))){
            ids.push_back(scan);if(tkey==boundary)ties--;}
            if(++scan==n)state=8;else state=12;break;
        case 8:if(ready&&++out==k)state=reserve?9:10;break;
        case 9:if(ready)state=10;break;
        case 10:done=true;state=0;break;
        default:throw std::runtime_error("bad selector state");
        }
    }
};
}
