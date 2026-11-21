#pragma once
#include "rtl_core.h"
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
    int job=-1,prefetch=-1;
    std::unique_ptr<Core> core;
    uint64_t busy=0,work=0,read_stall=0,counter_stall=0,input_stall=0,output_stall=0;
};
struct Cluster {
    int engine_count,score_banks,counter_banks,fifo_depth,input_width,output_width;
    bool double_buffer;
    uint64_t cycle=0,fifo_stall=0,link_stall=0,ingested=0,returned=0;
    unsigned rr=0;
    std::function<bool(int)> selection_allowed=[](int){return true;};
    std::function<void(int,int)> fifo_blocked=[](int,int){};
    std::vector<Job> jobs;
    std::vector<Engine> engines;
    Cluster(std::vector<Job> j,int e,int s,int c,int f,int iw=1,int ow=1,bool db=false):
        engine_count(e),score_banks(s),counter_banks(c),fifo_depth(f),input_width(iw),
        output_width(ow),double_buffer(db),jobs(std::move(j)),engines(e){
        if(e<1||s<1||c<1||f<1||iw<1||ow<1)throw std::runtime_error("bad cluster config");
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
            auto&core=*eng.core;auto&job=jobs[eng.job];eng.busy++;
            int bank=e%score_banks,cb=e%counter_banks;
            bool grant=true;
            bool reading=core.state==11||core.state==12||(core.state==8&&outputs>0);
            bool chosen=core.tkeep&&(core.tkey>core.boundary||
                                    (core.tkey==core.boundary&&core.ties>0));
            bool writing=(core.state==1&&!job.fifo.empty())||(core.state==6&&chosen);
            bool counting=core.state==3||core.state==13||
                          (core.state==4&&core.tkeep&&(core.tkey&core.mask)==core.prefix);
            if(reading&&read[bank]){eng.read_stall++;grant=false;}
            if(writing&&write[bank]){eng.read_stall++;grant=false;}
            if(counting&&counter[cb]){eng.counter_stall++;grant=false;}
            if(core.state==2&&(!allow_select||!selection_allowed(eng.job)))grant=false;
            bool iv=core.in_ready()&&!job.fifo.empty();
            bool ov=core.out_valid(),ready=outputs>0;
            if(core.in_ready()&&!iv)eng.input_stall++;
            if(ov&&!ready)eng.output_stall++;
            if(grant){
                if(reading)read[bank]=true;
                if(writing)write[bank]=true;
                if(counting)counter[cb]=true;
                uint16_t bits=iv?job.fifo.front():0;
                if(iv){job.fifo.pop_front();ingested++;}
                if(ov&&ready){job.result.push_back(core.index());outputs--;returned++;}
                if((!core.in_ready()||iv)&&(!ov||ready))eng.work++;
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
