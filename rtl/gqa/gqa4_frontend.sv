// Synthesizable GQA4 ownership, eight-KV allocator, and global32-head score router.
// Metadata releases after operand handshakes; score FIFO lifetime is independent.
module gqa4_frontend #(parameter BUFFERS=1, DEPTH=4)(
 input wire clk,rst,enable,prefetch,input wire [11:0] n_local,input wire [31:0] epoch,
 output wire req_valid,input wire req_ready,output wire req_slot,output wire [2:0] req_beat,
 output wire [31:0] req_tag,output wire [2:0] req_kv,output wire [18:0] req_byte,
 input wire rsp_valid,input wire rsp_slot,input wire [2:0] rsp_beat,input wire [31:0] rsp_tag,input wire [255:0] rsp_data,
 output wire operand_valid,input wire operand_ready,output wire [11:0] operand_local,
 output wire [4:0] operand_head,output wire operand_group,output wire [63:0] k_sign,k_mag,output wire [15:0] k_low,k_delta,
 output wire [31:0] operand_tile_tag,output wire released,output wire [31:0] released_tag,
 output wire stale_response,output wire [BUFFERS-1:0] occupied,filled,output wire allocation_done,output wire stream_idle,
 input wire score_in_valid,output wire score_in_ready,input wire [47:0] score_in_tag,input wire [15:0] score_in_bits,
 input wire [4:0] read_head,output wire score_valid,input wire score_ready,output wire [47:0] score_tag,output wire [15:0] score_bits,
 output wire [32*$clog2(DEPTH+1)-1:0] occupancy
);
 wire av,ar;wire [11:0] ast;wire [3:0] ac;wire [2:0] ak;wire [31:0] at;
 tile_allocator_gqa4 alloc(clk,rst,enable,prefetch,|occupied,n_local,4'd4,epoch,av,ar,ast,ac,ak,at,allocation_done);
 metadata_tile_stream_gqa4 #(.BUFFERS(BUFFERS)) stream(.clk(clk),.rst(rst),.enable(enable),
  .alloc_valid(av),.alloc_ready(ar),.alloc_start(ast),.alloc_count(ac[2:0]),.alloc_kv(ak),.alloc_tag(at),
  .req_valid(req_valid),.req_ready(req_ready),.req_slot(req_slot),.req_beat(req_beat),.req_tag(req_tag),.req_kv(req_kv),.req_byte(req_byte),
  .rsp_valid(rsp_valid),.rsp_ready(),.rsp_slot(rsp_slot),.rsp_beat(rsp_beat),.rsp_tag(rsp_tag),.rsp_data(rsp_data),
  .operand_valid(operand_valid),.operand_ready(operand_ready),.operand_local(operand_local),.operand_head(operand_head),.operand_group(operand_group),
  .k_sign(k_sign),.k_mag(k_mag),.k_low(k_low),.k_delta(k_delta),.operand_tile_tag(operand_tile_tag),.occupied(occupied),.filled(filled),
  .released(released),.released_tag(released_tag),.stale_response(stale_response),.idle(stream_idle));
 head_score_router_gqa4 #(.DEPTH(DEPTH)) router(.clk(clk),.rst(rst),.in_valid(score_in_valid),.in_ready(score_in_ready),.in_tag(score_in_tag),.in_bits(score_in_bits),
  .read_head(read_head),.out_valid(score_valid),.out_tag(score_tag),.out_bits(score_bits),.out_ready(score_ready),.occupancy(occupancy));
endmodule
