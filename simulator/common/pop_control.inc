// Included inside the POP frontend (Phase13Frontend). RTL control and routing are live in every Pop run.
 struct Score13 {uint64_t due,tag;uint16_t bits;int engine;bool captured=false;};
 struct Pop13 {
  std::unique_ptr<GQA>d;std::deque<Response13>responses;std::deque<Score13>pending;
  std::vector<uint64_t> engine_next;std::vector<int> contexts;
  uint64_t input_free=0,group1_ready=0;uint64_t active_tag=0;
  int outstanding=0,last_tile=-1;bool dequeue=false;unsigned dequeue_head=0;uint64_t dequeue_tag=0;
  std::map<int,int> unconsumed;
 };
 std::array<Pop13,64> controls;
 bool coupled_release=false;
 uint64_t rtl_operands=0,rtl_releases=0,rtl_release_independent=0,rtl_router_enqueues=0,rtl_router_dequeues=0,rtl_context_wait=0,rtl_router_full=0,rtl_barrier_wait=0,rtl_skipped_idle_edges=0;
 void initialize_pop13(){for(int b=0;b<64;b++){auto&x=controls[b];x.d=std::make_unique<GQA>();reset13(*x.d,N/64,1);x.engine_next.resize(replicas);x.contexts.resize(replicas);}}
 void pop_edge13(int b){
  auto&x=controls[b];auto&d=*x.d;
  bool refresh=cycle<poppim_hw::hw.refresh_until[b];if(refresh)refresh_wait_ticks++;
  bool computing=false;for(auto&e:x.pending)computing|=e.due>next_score;
  poppim_hw::hw.compute[b]=computing;if(computing)bank_busy_ticks++;
  d.score_in_valid=0;d.req_ready=0;d.rsp_valid=!x.responses.empty();d.operand_ready=0;d.score_ready=x.dequeue;d.read_head=x.dequeue?x.dequeue_head:0;
  if(d.rsp_valid)fill13(d,b,x.responses.front());
  if(!x.pending.empty()&&x.pending.front().captured&&x.pending.front().due<=next_score){
   d.score_in_valid=1;d.score_in_tag=x.pending.front().tag;d.score_in_bits=x.pending.front().bits;
  }
  d.eval();
  if(x.dequeue&&(!d.score_valid||d.score_tag!=x.dequeue_tag))throw std::runtime_error("RTL dequeue changed");
  bool drain=draining(b);if(drain)phase17diag::drain_wait_bank_ticks++;
  if(d.req_valid&&x.outstanding<2&&!refresh&&!drain){
   Response13 response{unsigned(d.req_slot),unsigned(d.req_beat),unsigned(d.req_tag),unsigned(d.req_kv),unsigned(d.req_byte)};
   uint64_t byte=metadata_base+uint64_t(response.kv)*(N/64)*40+response.byte;uint64_t submitted=cycle;
   Request req(address(b,byte),Request::Type::Read,b,[this,b,response,submitted](Request&r){
    int mapped=r.addr_vec[1]+(r.addr_vec[3]<<1)+(r.addr_vec[2]<<3)+(r.addr_vec[4]<<4);if(mapped!=b)throw std::runtime_error("RTL metadata bank mapping");
    auto&z=controls[b];if(z.responses.size()>=2)throw std::runtime_error("RTL response queue overflow");z.responses.push_back(response);callbacks++;last_read=now();latency_sum+=cycle-submitted;
   });
   if(m_memory_system->send(req)){d.req_ready=1;x.outstanding++;reads++;meta_reads++;if(!first_read)first_read=now();peak=std::max(peak,reads-callbacks);}else rejected++;
  }
  int engine=-1;
  if(d.operand_valid&&!refresh){
   int tile=(d.operand_head/4)*(N/64/4)+d.operand_local/4;
   bool barrier=coupled_release&&x.last_tile>=0&&tile!=x.last_tile&&x.unconsumed[x.last_tile]!=0;
   if(barrier)rtl_barrier_wait++;
   if(!d.operand_group){
    for(int e=0;e<replicas;e++)if(x.contexts[e]<2&&x.engine_next[e]<=next_score&&next_score>=x.input_free){engine=e;break;}
    // Refresh drain: no new scorer context while this bank owes an urgent refresh.
    d.operand_ready=engine>=0&&!barrier&&!drain;
    if(engine<0&&!barrier)rtl_context_wait++;
   }else d.operand_ready=next_score>=x.group1_ready;
  }
  if(!d.operand_valid&&!d.stream_idle)input_wait_ticks++;
  d.eval();
  bool input=d.operand_valid&&d.operand_ready;
  if(input){
   verify_operand13(d,b);rtl_operands++;
   unsigned h=d.operand_head,token=d.operand_local*64+b;uint64_t tag=(uint64_t(h)<<40)|(uint64_t(h/4)<<32)|token;
   if(!d.operand_group){
    if(engine<0)throw std::runtime_error("no scorer context");
    x.contexts[engine]++;x.engine_next[engine]=next_score+uint64_t(ii)*8000;x.input_free=next_score+24000;x.group1_ready=next_score+16000;x.active_tag=tag;
    x.pending.push_back({next_score+uint64_t(latency)*8000,tag,scores.at(h*N+token),engine,false});
    int tile=(h/4)*(N/64/4)+d.operand_local/4;x.last_tile=tile;x.unconsumed[tile]++;
   }else{
    if(tag!=x.active_tag)throw std::runtime_error("group tag mismatch");bool found=false;
    for(auto&e:x.pending)if(e.tag==tag){e.captured=true;found=true;break;}if(!found)throw std::runtime_error("lost scorer context");
   }
  }
  if(d.score_in_valid&&d.score_in_ready){
   if(!first_score)first_score=next_score;last_score=next_score;score_count++;rtl_router_enqueues++;if(offload)router_order[b].push_back({unsigned(x.pending.front().tag>>40),unsigned(uint32_t(x.pending.front().tag))});x.contexts[x.pending.front().engine]--;x.pending.pop_front();
  }else if(d.score_in_valid){rtl_router_full++;output_wait_ticks++;}
  if(x.dequeue){unsigned h=x.dequeue_tag>>40,token=uint32_t(x.dequeue_tag);int tile=(h/4)*(N/64/4)+(token/64)/4;
   if(--x.unconsumed[tile]<0)throw std::runtime_error("tile output credit underflow");rtl_router_dequeues++;
  }
  bool response=d.rsp_valid;
  // Structurally quiescent full slots have no internal autonomous transitions.
  bool idle_stall=!input&&!response&&!d.req_ready&&!d.score_in_ready&&!x.dequeue&&d.filled==((1u<<META_BUFFERS)-1)&&!d.released;
  if(idle_stall)rtl_skipped_idle_edges++;else edge13(d);
  x.dequeue=false;
  if(response){x.responses.pop_front();x.outstanding--;}
  if(d.stale_response)throw std::runtime_error("unexpected stale response in integrated run");
  if(d.released&&!idle_stall){rtl_releases++;if(x.unconsumed[int(d.released_tag)-1]>0)rtl_release_independent++;}
 }
 bool pop_available13(int h,int token){
  auto&x=controls[token%64];if(x.dequeue)return false;auto&d=*x.d;d.read_head=h;d.score_ready=0;d.eval();
  if(!d.score_valid)return false;
  uint64_t expected_tag=(uint64_t(h)<<40)|(uint64_t(h/4)<<32)|unsigned(token);
  if(d.score_tag!=expected_tag)return false;
  if(d.score_bits!=scores.at(h*N+token))throw std::runtime_error("RTL routed bits mismatch");return true;
 }
 uint16_t pop_take13(int h,int token){if(!pop_available13(h,token))throw std::runtime_error("RTL output not available");auto&x=controls[token%64];x.dequeue=true;x.dequeue_head=h;x.dequeue_tag=x.d->score_tag;return x.d->score_bits;}
