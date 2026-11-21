// GS bank: unchanged A2 bank, GS unit inserted between the scorer and the 32-head router (only the 8 leader-head
// FIFOs are used; the router is kept unchanged, so its area is not credited back).
module gqa4_score_bank_gs(
 input wire clk,rst,enable,input wire [11:0] n_local,input wire [5:0] bank_id,
 input wire [4095:0] q_signs,q_mags,input wire [2047:0] q_lows,q_deltas,input wire [1023:0] norms,
 output wire req_valid,input wire req_ready,output wire req_slot,output wire [2:0] req_beat,output wire [31:0] req_tag,output wire [2:0] req_kv,output wire [18:0] req_byte,
 input wire rsp_valid,input wire rsp_slot,input wire [2:0] rsp_beat,input wire [31:0] rsp_tag,input wire [255:0] rsp_data,
 input wire [4:0] read_head,output wire score_valid,input wire score_ready,output wire [47:0] score_tag,output wire [15:0] score_bits,
 output wire released,output wire [31:0] released_tag,output wire stale_response,output wire stream_idle
);
 wire ov,orr,og;wire [11:0] ol;wire [4:0] oh;wire [63:0] ks,km;wire [15:0] kl,dk;
 wire sv,sr;wire [47:0] st;wire [15:0] sb;   // scorer -> GS
 wire iv,ir;wire [47:0] it;wire [15:0] ib;   // GS -> router
 gqa4_frontend ctl(.clk(clk),.rst(rst),.enable(enable),.prefetch(1'b1),.n_local(n_local),.epoch(32'd1),
  .req_valid(req_valid),.req_ready(req_ready),.req_slot(req_slot),.req_beat(req_beat),.req_tag(req_tag),.req_kv(req_kv),.req_byte(req_byte),
  .rsp_valid(rsp_valid),.rsp_slot(rsp_slot),.rsp_beat(rsp_beat),.rsp_tag(rsp_tag),.rsp_data(rsp_data),
  .operand_valid(ov),.operand_ready(orr),.operand_local(ol),.operand_head(oh),.operand_group(og),.k_sign(ks),.k_mag(km),.k_low(kl),.k_delta(dk),.operand_tile_tag(),
  .released(released),.released_tag(released_tag),.stale_response(stale_response),.occupied(),.filled(),.allocation_done(),.stream_idle(stream_idle),
  .score_in_valid(iv),.score_in_ready(ir),.score_in_tag(it),.score_in_bits(ib),.read_head(read_head),.score_valid(score_valid),.score_ready(score_ready),.score_tag(score_tag),.score_bits(score_bits),.occupancy());
 wire [47:0] tag={3'd0,oh,5'd0,oh[4:2],14'd0,ol,bank_id};
 score7_top #(.CONTEXTS(2),.ARCH_B(0),.RAW_BF16(0)) scorer(.clk(clk),.rst(rst),.in_valid(ov),.first_group(!og),.last_group(og),.in_tag(tag),
  .q_sign(q_signs[(oh*2+og)*64+:64]),.q_mag(q_mags[(oh*2+og)*64+:64]),.q_low(q_lows[(oh*2+og)*32+:32]),.q_delta(q_deltas[(oh*2+og)*32+:32]),.normalization(norms[oh*32+:32]),
  .k_sign(ks),.k_mag(km),.k_low(kl),.k_delta(dk),.dim_mask(64'hffffffffffffffff),.in_ready(orr),.out_valid(sv),.out_tag(st),.score_bits(sb),.out_ready(sr),
  .count_valid(),.count_tag(),.count_rep(),.count_values(),.fp_issue_valid(),.fp_issue_tag(),.fp_issue_node(),.fp_complete_valid(),.fp_complete_tag(),.fp_complete_node(),.fp_complete_bits(),.empty(),.dependency_stall(),.context_stall(),.output_stall());
 gs_shared_score gs(.clk(clk),.rst(rst),.in_valid(sv),.in_ready(sr),.in_tag(st),.in_bits(sb),.out_valid(iv),.out_ready(ir),.out_tag(it),.out_bits(ib));
endmodule
