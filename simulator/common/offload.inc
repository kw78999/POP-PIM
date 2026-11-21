 struct Packet15{unsigned head,token;uint16_t bits;uint64_t due=0;};
 bool offload=false,ideal_sink=false;uint64_t fifo_capacity=64,fifo_peak=0,dma_depth=16,dma_peak=0,interface_stall=0,delivered=0,transfer_first=0,transfer_last=0,external_topk_ps=0,external_ready=0,index_transfer=0,startup_ps=5000000,sync_ps=2000000,link_free=0,fifo_writes=0,fifo_reads=0;
 double link_GBs=32;unsigned output_rr=0;int lane_depth=16,enabled_lanes=4;uint64_t output_period_ps=4000,next_output_ps=4000;
 std::array<std::deque<std::pair<unsigned,unsigned>>,64> router_order;
 std::array<std::deque<Packet15>,4> output_lanes;std::deque<Packet15> dma;
 std::vector<uint16_t> received;std::vector<unsigned char> received_valid;
 std::array<std::vector<int>,32> external_indices;
 // Compact transport (CT): aggregated (GQA-shared) export and implicit-tag packets.
 bool shared_export=false;uint64_t packet_wire_bytes=8,merged=0,index_return_bytes=0;
 void init_offload(){
  link_GBs=param<double>("link_GBs").default_val(32);startup_ps=param<uint64_t>("startup_ps").default_val(5000000);sync_ps=param<uint64_t>("sync_ps").default_val(2000000);external_topk_ps=param<uint64_t>("external_topk_ps").default_val(200000000);lane_depth=param<int>("lane_depth").default_val(16);dma_depth=param<uint64_t>("dma_depth").default_val(16);enabled_lanes=param<int>("output_lanes").default_val(4);output_period_ps=param<uint64_t>("output_period_ps").default_val(4000);next_output_ps=output_period_ps;fifo_capacity=enabled_lanes*lane_depth;
  shared_export=param<bool>("shared_export").default_val(false);packet_wire_bytes=param<uint64_t>("packet_wire_bytes").default_val(8);index_return_bytes=param<uint64_t>("index_return_bytes").default_val(uint64_t(32)*K*4);
  if(link_GBs<=0||lane_depth<1||dma_depth<1||enabled_lanes<1||enabled_lanes>4||output_period_ps<4000||output_period_ps%4000!=0)throw std::runtime_error("finite output config");received.resize(32*N);received_valid.resize(32*N);
 }
 void receive_packet(Packet15 p,uint64_t stamp){
  unsigned i=p.head*N+p.token;if(i>=received.size()||received_valid[i])throw std::runtime_error("duplicate/bad head token");
  if(p.bits!=scores[i])throw std::runtime_error("BF16 transfer corruption");received[i]=p.bits;received_valid[i]=1;delivered++;if(!transfer_first)transfer_first=stamp;transfer_last=stamp;
 }
 void merge_packet(Packet15 p){
  // Non-leader heads are summed into the KV-group score in the output stage and never cross the link.
  unsigned i=p.head*N+p.token;if(i>=received.size()||received_valid[i])throw std::runtime_error("duplicate/bad merged head token");
  if(p.bits!=scores[i])throw std::runtime_error("merged score mismatch");received[i]=p.bits;received_valid[i]=1;delivered++;merged++;
 }
 void external_select(){
  if(received!=scores||delivered!=uint64_t(32)*N)throw std::runtime_error("incomplete score reconstruction");
  std::ofstream sf(output+"/received_scores.bin",std::ios::binary);sf.write((char*)received.data(),received.size()*2);sf.close();
  // CPU functional implementation only. GPU scenarios use this as a correctness stand-in,
  // with explicitly parameterized device service time; no GPU execution is claimed.
  for(int h=0;h<32;h++){
   std::vector<int> ids(N);std::iota(ids.begin(),ids.end(),0);
   auto cmp=[&](int a,int b){auto x=poppim::ordered(received[h*N+a]),y=poppim::ordered(received[h*N+b]);return x!=y?x>y:a<b;};
   std::nth_element(ids.begin(),ids.begin()+K,ids.end(),cmp);ids.resize(K);std::sort(ids.begin(),ids.end());external_indices[h]=ids;
   for(int i=0;i<K;i++)if(ids[i]!=expected[h*K+i])throw std::runtime_error("external stable membership mismatch");
  }
  index_transfer=ideal_sink?0:startup_ps+uint64_t(std::ceil(double(index_return_bytes)*1000/link_GBs));
  external_ready=transfer_last+external_topk_ps+index_transfer+sync_ps+(ideal_sink?8000:0); // ideal diagnostic waits for final RTL dequeue edge
 }
 void offload_tick(){
  if(next_selector<next_output_ps)return;next_output_ps+=output_period_ps;
  uint64_t stamp=next_selector;
  while(!dma.empty()&&dma.front().due<=stamp){auto p=dma.front();dma.pop_front();receive_packet(p,p.due);}
  if(!ideal_sink){
   for(int lane=0;lane<enabled_lanes;lane++)if(!output_lanes[lane].empty()&&dma.size()<dma_depth){
    auto p=output_lanes[lane].front();output_lanes[lane].pop_front();fifo_reads++;
    if(!link_free)link_free=stamp+startup_ps;
    p.due=std::max(link_free,stamp)+uint64_t(std::ceil(double(packet_wire_bytes)*1000/link_GBs));link_free=p.due;dma.push_back(p);dma_peak=std::max(dma_peak,uint64_t(dma.size()));
   }
  }
  bool pushed[4]={false,false,false,false};
  for(int offset=0;offset<64;offset++){
   unsigned b=(output_rr+offset)%64;auto& q=router_order[b];if(q.empty()||controls[b].dequeue)continue;
   if(shared_export){auto [h0,t0]=q.front();if(h0%4!=0){if(!pop_available13(h0,t0))continue;Packet15 m{h0,t0,pop_take13(h0,t0),0};q.pop_front();merge_packet(m);continue;}}
   int lane=b%enabled_lanes;if(!ideal_sink&&(pushed[lane]||output_lanes[lane].size()>=size_t(lane_depth))){interface_stall++;continue;}
   auto [h,t]=q.front();if(!pop_available13(h,t))continue;
   Packet15 p{h,t,pop_take13(h,t),0};q.pop_front();
   if(ideal_sink)receive_packet(p,stamp);else{output_lanes[lane].push_back(p);fifo_writes++;pushed[lane]=true;}
  }
  output_rr=(output_rr+1)%64;uint64_t occupancy=0;for(auto& q:output_lanes)occupancy+=q.size();fifo_peak=std::max(fifo_peak,occupancy);
  if(delivered==uint64_t(32)*N&&!external_ready)external_select();
 }
