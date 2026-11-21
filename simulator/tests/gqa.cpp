#include "../common/gqa_hw.h"
#include <vector>
#include <random>
#include <iostream>
#include <set>
struct Pending {uint64_t due,tag;uint16_t bits;};
int main(){
 uint64_t total_operand=0,total_scores=0,total_release=0,total_stale=0,full_events=0,early_release=0;
 for(unsigned seed=1;seed<=8;seed++){
  GQA d;unsigned n=seed%2?7:31,bank=seed*7%64;std::mt19937 rng(seed);reset13(d,n,seed*1000);
  std::vector<std::pair<uint64_t,Response13>> responses;std::deque<Pending> events;
  std::map<unsigned,unsigned> captures;std::set<std::pair<unsigned,unsigned>> seen_req;
  std::array<unsigned,32> got{};uint64_t issued=0,returned=0,releases=0;bool injected=false,duplicated=false;Response13 duplicate{};
  for(uint64_t t=0;t<200000;t++){
   d.req_ready=(rng()%4!=0)&&responses.size()<8;d.rsp_valid=0;d.operand_ready=rng()%5!=0&&events.size()<2;d.score_in_valid=0;
   d.read_head=rng()%32;d.score_ready=t>500&&rng()%3!=0;
   if(!events.empty()&&events.front().due<=t){d.score_in_valid=1;d.score_in_tag=events.front().tag;d.score_in_bits=events.front().bits;}
   int response=-1;
   for(int j=int(responses.size())-1;j>=0;j--)if(responses[j].first<=t){response=j;break;}
   if(t==8){Response13 stale{0,0,seed*1000-1,0,0};d.rsp_valid=1;fill13(d,bank,stale);injected=true;response=-2;}
   else if(!duplicated&&returned>3){d.rsp_valid=1;fill13(d,bank,duplicate);duplicated=true;response=-2;}
   else if(response>=0){d.rsp_valid=1;fill13(d,bank,responses[response].second);duplicate=responses[response].second;returned++;}
   d.eval();
   if(d.req_valid&&d.req_ready){Response13 r{unsigned(d.req_slot),unsigned(d.req_beat),unsigned(d.req_tag),unsigned(d.req_kv),unsigned(d.req_byte)};if(r.kv>=8||!seen_req.insert({r.tag,r.beat}).second)throw std::runtime_error("duplicate/wrong KV request");responses.push_back({t+2+rng()%40,r});}
   if(d.operand_valid&&d.operand_ready){verify_operand13(d,bank);captures[d.operand_tile_tag]++;total_operand++;
    if(!d.operand_group){unsigned h=d.operand_head,local=d.operand_local;events.push_back({t+43,(uint64_t(h)<<40)|(uint64_t(h/4)<<32)|(local*64+bank),uint16_t(h*100+local)});issued++;}
   }
   if(d.score_in_valid&&d.score_in_ready)events.pop_front();
   if(d.score_in_valid&&!d.score_in_ready)full_events++;
   if(d.score_valid&&d.score_ready){unsigned h=d.read_head,local=got[h]++;if(d.score_tag!=((uint64_t(h)<<40)|(uint64_t(h/4)<<32)|(local*64+bank))||d.score_bits!=h*100+local)throw std::runtime_error("head/token router misdelivery");total_scores++;}
   if(response>=0)responses.erase(responses.begin()+response);
   edge13(d);
   if(d.stale_response)total_stale++;
   if(d.released){unsigned tag=d.released_tag;unsigned tile=(tag-seed*1000)%((n+3)/4);unsigned count=std::min(4u,n-tile*4);if(captures[tag]!=8*count)throw std::runtime_error("premature/missing fourth consumer release");releases++;total_release++;if(!events.empty()||!d.stream_idle)early_release++;}
   unsigned sum=0;for(unsigned g:got)sum+=g;
   if(sum==32*n&&d.stream_idle&&events.empty()&&responses.empty()){if(issued!=32*n||releases!=8*((n+3)/4)||!injected||!duplicated)throw std::runtime_error("unit conservation");for(unsigned g:got)if(g!=n)throw std::runtime_error("missing head");break;}
   if(t==199999)throw std::runtime_error("unit watchdog");
  }
  // Reset must invalidate FIFO contents and ownership before a new epoch.
  reset13(d,n,seed*1000+500);if(d.score_valid||d.occupied)throw std::runtime_error("reset did not clear ownership/router");
 }
 if(!full_events||total_stale!=16||!early_release)throw std::runtime_error("coverage missing");
 std::cout<<"{\"status\":\"Passed\",\"buffers\":"<<META_BUFFERS<<",\"trials\":8,\"operand_handshakes\":"<<total_operand<<",\"routed_scores\":"<<total_scores<<",\"tile_releases\":"<<total_release<<",\"rejected_stale_duplicate_responses\":"<<total_stale<<",\"router_full_cycles\":"<<full_events<<",\"independent_release_events\":"<<early_release<<"}\n";
}
