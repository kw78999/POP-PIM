#pragma once
#include "bf16_core.h"
#include <deque>
#include <memory>
#include <functional>
#include <algorithm>
#include <numeric>
namespace poppim {
struct Job {
    int n=0,k=0,preferred=-1,head=0,bank=-1,produced=0;
    bool assigned=false,finished=false;
    uint64_t start=0,completion=0;
    std::deque<uint16_t> fifo;
    std::vector<int> result;
};
struct Engine {
    uint64_t phase_cycles[16]={0},phase_read_wait[16]={0};
    int job=-1,prefetch=-1;
    bool read_pending=false,cache_valid=false;uint64_t due=0,pending_word=0,cache_word=0;
    std::unique_ptr<Core> core;
    uint64_t busy=0,work=0,read_stall=0,counter_stall=0,input_stall=0,output_stall=0;
};
struct Cluster {
    int engine_count,score_banks,counter_banks,fifo_depth,input_width,output_width;
    bool double_buffer;
    uint64_t cycle=0,fifo_stall=0,link_stall=0,ingested=0,returned=0;
    uint64_t score_reads=0,score_writes=0,candidate_reads=0,candidate_writes=0,counter_reads=0,counter_writes=0,histogram_updates=0,radix_passes=0,comparisons=0,index_generation=0,keep_reads=0,keep_writes=0;
    unsigned rr=0;
    bool word_model=false,private_memory=false;
    int word_bits=32,sram_latency=2,sram_port_II=1;
    std::vector<uint64_t> read_free,write_free;
    int max_n=0,max_k=0;
    uint64_t score_word(int e,int i)const{return (uint64_t(e)*max_n*2+i*2)/(word_bits/8);}
    uint64_t candidate_word(int e,int i)const{return (uint64_t(engine_count)*max_n*2+uint64_t(e)*max_k*4+i*4)/(word_bits/8);}

    std::function<bool(int)> selection_allowed=[](int){return true;};
    std::function<void(int,int)> fifo_blocked=[](int,int){};
    std::vector<Job> jobs;
    std::vector<Engine> engines;
    Cluster(std::vector<Job> j,int e,int s,int c,int f,int iw=1,int ow=1,bool db=false):
        engine_count(e),score_banks(s),counter_banks(c),fifo_depth(f),input_width(iw),
        output_width(ow),double_buffer(db),jobs(std::move(j)),engines(e){
        if(e<1||s<1||c<1||f<1||iw<1||ow<1)throw std::runtime_error("bad cluster config");
        read_free.resize(s);write_free.resize(s);
        for(auto&j:jobs){max_n=std::max(max_n,j.n);max_k=std::max(max_k,j.k);}

    }
    int next_job(int engine){
        for(unsigned j=0;j<jobs.size();j++)if(!jobs[j].assigned&&
            (jobs[j].preferred<0||jobs[j].preferred==engine)){
            jobs[j].assigned=true;return j;
        }return -1;
    }
    bool active(int job)const{return jobs[job].assigned&&!jobs[job].finished;}
    bool finished()const{
        return std::all_of(jobs.begin(),jobs.end(),[](const Job&j){return j.finished;});
    }
    // offer(j, token) supplies a score only if actually ready; take consumes it.
    // Input FIFO is registered: newly offered scores cannot be loaded this tick.
    void tick(const std::function<bool(int,int)>&available,
              const std::function<uint16_t(int,int)>&take,bool allow_select=true,bool output_enable=true){
        std::vector<bool> read(score_banks),write(score_banks),counter(counter_banks);
        int outputs=output_enable?output_width:0;
        for(int offset=0;offset<engine_count;offset++){
            int e=(rr+offset)%engine_count;auto&eng=engines[e];
            if(!eng.core){
                int j=eng.prefetch>=0?eng.prefetch:next_job(e);eng.prefetch=-1;
                if(j>=0){eng.job=j;eng.core=std::make_unique<Core>(jobs[j].n,jobs[j].k);
                    jobs[j].start=cycle;}continue; // paid start edge
            }
            auto&core=*eng.core;auto&job=jobs[eng.job];eng.busy++;eng.phase_cycles[core.state]++;
            int bank=e%score_banks,cb=e%counter_banks;
            bool grant=true;
            bool reading=core.state==11||core.state==12||(core.state==8&&outputs>0);
            bool chosen=core.tkeep&&(core.tkey>core.boundary||
                                    (core.tkey==core.boundary&&core.ties>0));
            bool writing=(core.state==1&&!job.fifo.empty())||(core.state==6&&chosen);
            bool counting=core.state==3||core.state==13||
                          (core.state==4&&core.tkeep&&(core.tkey&core.mask)==core.prefix);
            uint64_t write_word=0;
            if(word_model){
                bool need=core.state==11||core.state==12||core.state==8;
                uint64_t word=core.state==8?candidate_word(e,core.out):score_word(e,core.scan);
                if(need){
                    bank=private_memory?e%score_banks:word%score_banks;
                    if(eng.read_pending&&cycle>=eng.due){eng.cache_valid=true;eng.cache_word=eng.pending_word;eng.read_pending=false;}
                    if(!eng.cache_valid||eng.cache_word!=word){
                        if(!eng.read_pending&&!read[bank]&&read_free[bank]<=cycle){
                            if(core.state==8)candidate_reads++;else score_reads++;
                            read[bank]=true;read_free[bank]=cycle+sram_port_II;
                            eng.read_pending=true;eng.pending_word=word;eng.due=cycle+sram_latency;
                        }
                        eng.read_stall++;eng.phase_read_wait[core.state]++;grant=false;
                    }
                }
                reading=false; // the request itself was reserved above; cache reads have no bank request
                if(writing){
                    write_word=core.state==1?score_word(e,core.load):candidate_word(e,core.ids.size());
                    bank=private_memory?e%score_banks:write_word%score_banks;
                    if(write[bank]||write_free[bank]>cycle){eng.read_stall++;eng.phase_read_wait[core.state]++;grant=false;}
                }
            }else{
                if(reading&&read[bank]){eng.read_stall++;eng.phase_read_wait[core.state]++;grant=false;}
                if(writing&&write[bank]){eng.read_stall++;eng.phase_read_wait[core.state]++;grant=false;}
            }
            if(counting&&counter[cb]){eng.counter_stall++;grant=false;}
            if(core.state==2&&(!allow_select||!selection_allowed(eng.job)))grant=false;
            bool iv=core.in_ready()&&!job.fifo.empty();
            bool ov=core.out_valid(),ready=outputs>0;
            if(core.in_ready()&&!iv)eng.input_stall++;
            if(ov&&!ready)eng.output_stall++;
            if(grant){
                if(reading)read[bank]=true;
                if(writing){
                    if(core.state==1)score_writes++;else candidate_writes++;
                    write[bank]=true;
                    if(word_model){write_free[bank]=cycle+sram_port_II;
                        for(auto&other:engines)if(other.cache_valid&&other.cache_word==write_word)other.cache_valid=false;}
                }
                if(counting)counter[cb]=true;
                uint16_t bits=iv?job.fifo.front():0;
                if(iv){job.fifo.pop_front();ingested++;}
                if(ov&&ready){job.result.push_back(core.index());outputs--;returned++;}
                if((!core.in_ready()||iv)&&(!ov||ready))eng.work++;
                if(core.state==3){counter_writes++;if(core.clear==0)radix_passes++;}
                if(core.state==13)counter_reads++;
                if(core.state==4&&core.tkeep){comparisons++;if((core.tkey&core.mask)==core.prefix){counter_reads++;counter_writes++;histogram_updates++;}}
                if(core.state==5)comparisons++;
                if(core.state==6&&core.tkeep){comparisons++;if(core.tkey<=core.boundary)comparisons++;if(chosen)index_generation++;}
                if(core.state==11||core.state==12)keep_reads++;
                if(iv)keep_writes++;
                core.tick(iv,bits,true,ready);
                if(core.done){if(core.error)throw std::runtime_error("selector error");
                    job.finished=true;job.completion=cycle+1;eng.core.reset();eng.job=-1;}
            }
            if(double_buffer&&eng.prefetch<0)eng.prefetch=next_job(e);
        }
        int inputs=input_width;
        for(int offset=0;offset<engine_count;offset++){
            int e=(rr+offset)%engine_count;
            for(int j:{engines[e].job,engines[e].prefetch}){
                if(j<0)continue;auto&job=jobs[j];
                if(job.produced>=job.n||!available(j,job.produced))continue;
                int capacity=(j==engines[e].prefetch&&double_buffer)?job.n:fifo_depth;
                // A prefetched full frame retains its capacity after activation.
                if(double_buffer)capacity=job.n;
                if(int(job.fifo.size())>=capacity){fifo_stall++;fifo_blocked(j,job.produced);continue;}
                if(inputs==0){link_stall++;continue;}
                job.fifo.push_back(take(j,job.produced++));inputs--;
            }
        }
        rr=(rr+1)%engine_count;cycle++;
    }
};
}
