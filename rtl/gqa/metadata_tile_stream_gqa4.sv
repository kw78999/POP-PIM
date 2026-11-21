// Bank-local metadata stream. Q registers and the unmodified A2 are external.
// A slot is reserved at allocation, filled by tagged 32B responses, and released
// only after all four GQA heads capture every two-group operand. No score reorder RAM.
module metadata_tile_stream_gqa4 #(
 parameter BUFFERS=2, TILE=4, IW=12, REUSE=1,
 parameter TB=TILE*40, NB=(TB+47)/32, BW=$clog2(NB+1), TW=$clog2(TILE+1)
)(
 input wire clk,rst,enable,
 input wire alloc_valid, output wire alloc_ready,
 input wire [IW-1:0] alloc_start, input wire [TW-1:0] alloc_count,
 input wire [2:0] alloc_kv, input wire [31:0] alloc_tag,
 output reg req_valid, input wire req_ready,
 output reg req_slot, output reg [BW-1:0] req_beat,
 output reg [31:0] req_tag, output reg [2:0] req_kv,
 output reg [IW+6:0] req_byte,
 input wire rsp_valid, output wire rsp_ready,
 input wire rsp_slot, input wire [BW-1:0] rsp_beat,
 input wire [31:0] rsp_tag, input wire [255:0] rsp_data,
 output wire operand_valid, input wire operand_ready,
 output wire [IW-1:0] operand_local,
 output wire [4:0] operand_head, output wire operand_group,
 output wire [63:0] k_sign,k_mag, output wire [15:0] k_low,k_delta,
 output wire [31:0] operand_tile_tag,
 output wire [BUFFERS-1:0] occupied,filled,
 output reg released, output reg [31:0] released_tag,
 output reg stale_response, output wire idle
);
 reg [BUFFERS-1:0] live,complete;
 reg [TB*8-1:0] payload[0:BUFFERS-1];
 reg [IW-1:0] starts[0:BUFFERS-1];
 reg [TW-1:0] counts[0:BUFFERS-1];
 reg [2:0] kv[0:BUFFERS-1];reg [31:0] tags[0:BUFFERS-1];
 reg [BW-1:0] issue_count[0:BUFFERS-1],bursts[0:BUFFERS-1];
 reg [NB-1:0] returned[0:BUFFERS-1];
 reg read_slot,write_slot;
 reg [1:0] head_phase;reg group_phase;reg [TW-1:0] token;
 wire [TW:0] remaining_token=counts[read_slot]-1;
 wire [IW+6:0] start_byte=alloc_start*40;
 wire [4:0] alloc_pad=TILE==2?start_byte[4:0]:5'b0;
 assign alloc_ready=enable&&!live[write_slot];
 assign rsp_ready=1'b1;
 assign occupied=live;assign filled=complete;
 assign operand_valid=live[read_slot]&&complete[read_slot];
 assign operand_local=starts[read_slot]+token;
 assign operand_head={kv[read_slot],head_phase};
 assign operand_group=group_phase;
 assign operand_tile_tag=tags[read_slot];
 wire [159:0] group_data=payload[read_slot] >> ((token*40+group_phase*20)*8);
 assign k_sign=group_data[63:0];assign k_mag=group_data[127:64];
 assign k_low=group_data[143:128];assign k_delta=group_data[159:144];
 assign idle=!(|live)&&!req_valid;
 integer i,j;reg picked;
 always @(posedge clk)begin
  if(rst)begin
   live<=0;complete<=0;read_slot<=0;write_slot<=0;
   head_phase<=0;group_phase<=0;token<=0;req_valid<=0;released<=0;stale_response<=0;
   req_slot<=0;req_beat<=0;req_tag<=0;req_kv<=0;req_byte<=0;released_tag<=0;
   for(i=0;i<BUFFERS;i=i+1)begin issue_count[i]<=0;returned[i]<=0;bursts[i]<=0;end
  end else begin
   released<=0;stale_response<=0;
   if(alloc_valid&&alloc_ready)begin
    live[write_slot]<=1;complete[write_slot]<=0;starts[write_slot]<=alloc_start;
    counts[write_slot]<=alloc_count;kv[write_slot]<=alloc_kv;tags[write_slot]<=alloc_tag;
    issue_count[write_slot]<=0;returned[write_slot]<=0;
    bursts[write_slot]<=(alloc_count*40+alloc_pad+31)/32;
    if(BUFFERS==2)write_slot<=!write_slot;
   end
   // A registered one-entry request FIFO; each reserved slot bounds outstanding reads.
   if(req_valid&&req_ready)begin
    req_valid<=0;issue_count[req_slot]<=issue_count[req_slot]+1;
   end
   if(!req_valid)begin
    picked=0;
    for(i=0;i<BUFFERS;i=i+1)begin
     if(!picked&&live[i]&&issue_count[i]<bursts[i])begin
      picked=1;req_valid<=1;req_slot<=i;req_tag<=tags[i];req_kv<=kv[i];req_beat<=issue_count[i];
      req_byte<=((starts[i]*40)/32+issue_count[i])*32;
     end
    end
   end
   if(rsp_valid)begin
    if(rsp_slot<BUFFERS&&live[rsp_slot]&&tags[rsp_slot]==rsp_tag&&rsp_beat<bursts[rsp_slot]&&!returned[rsp_slot][rsp_beat])begin
     returned[rsp_slot][rsp_beat]<=1;
     if((returned[rsp_slot]|( {{(NB-1){1'b0}},1'b1}<<rsp_beat))==(({{(NB-1){1'b0}},1'b1}<<bursts[rsp_slot])-1))complete[rsp_slot]<=1;
    end else stale_response<=1;
   end
   if(operand_valid&&operand_ready)begin
    if(!group_phase)group_phase<=1;
    else begin
     group_phase<=0;
     if(token==remaining_token)begin
      token<=0;
      if(head_phase!=3)begin
       head_phase<=head_phase+1;
       if(!REUSE)begin complete[read_slot]<=0;issue_count[read_slot]<=0;returned[read_slot]<=0;end
      end
      else begin
       head_phase<=0;live[read_slot]<=0;complete[read_slot]<=0;
       released<=1;released_tag<=tags[read_slot];
       if(BUFFERS==2)read_slot<=!read_slot;
      end
     end else token<=token+1;
    end
   end
  end
 end
 // Byte enables are constant for 4/8-token tiles; 2-token packed tiles can start
 // halfway through a DRAM line. Padding bytes are never exposed as operands.
 genvar s,b;
 generate for(s=0;s<BUFFERS;s=s+1)begin:slot_data
  for(b=0;b<TB;b=b+1)begin:bytes
   wire pad16=TILE==2&&starts[s][1];
   wire beat_match=pad16?rsp_beat==(b+16)/32:rsp_beat==b/32;
   wire [7:0] incoming=pad16?rsp_data[((b+16)%32)*8+:8]:rsp_data[(b%32)*8+:8];
   always @(posedge clk)begin
    if(!rst&&rsp_valid&&rsp_slot==s&&live[s]&&tags[s]==rsp_tag&&rsp_beat<bursts[s]&&!returned[s][rsp_beat]&&beat_match)
     payload[s][b*8+:8]<=incoming;
   end
  end
 end endgenerate
endmodule
