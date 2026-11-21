// Dense scorer with independent 16-byte/cycle Q and K SRAM read ports.
// Large lane counts assemble a wide core beat in one explicit staging register.
// This storage and its handshake are included in mapped synthesis.
module dense_score_port16 #(parameter LANES=16)(
 input wire clk,rst,in_valid,first_beat,last_beat,input wire [47:0] in_tag,
 input wire [127:0] q_bits,k_bits,input wire [15:0] normalization,
 output wire in_ready,out_valid,output wire [47:0] out_tag,
 output wire [15:0] score_bits,input wire out_ready,
 output wire [LANES-1:0] fp_issue,fp_complete,
 output wire [LANES*48-1:0] issue_tags,complete_tags,
 output wire [LANES*8-1:0] issue_nodes,complete_nodes,
 output wire [LANES*16-1:0] complete_bits,output wire empty);
 wire core_ready,core_empty;
 wire core_valid,core_first,core_last;
 wire [47:0] core_tag;
 wire [15:0] core_norm;
 wire [LANES*16-1:0] core_q,core_k;
 generate if(LANES<=8)begin:narrow
  assign in_ready=core_ready;
  assign core_valid=in_valid;assign core_first=first_beat;assign core_last=last_beat;
  assign core_tag=in_tag;assign core_norm=normalization;
  assign core_q=q_bits[LANES*16-1:0];assign core_k=k_bits[LANES*16-1:0];
  assign empty=core_empty;
 end else begin:assemble
  localparam CHUNKS=LANES/8;
  reg [$clog2(CHUNKS)-1:0] chunk;
  reg full,first_saved,last_saved;
  reg [47:0] tag_saved;
  reg [15:0] norm_saved;
  reg [LANES*16-1:0] q_saved,k_saved;
  assign in_ready=!full||core_ready;
  assign core_valid=full;assign core_first=first_saved;assign core_last=last_saved;
  assign core_tag=tag_saved;assign core_norm=norm_saved;
  assign core_q=q_saved;assign core_k=k_saved;
  assign empty=core_empty&&!full&&(chunk==0);
  always @(posedge clk)begin
   if(rst)begin chunk<=0;full<=0;first_saved<=0;last_saved<=0;end
   else begin
    if(full&&core_ready)full<=0;
    if(in_valid&&in_ready)begin
     q_saved[128*chunk+:128]<=q_bits;k_saved[128*chunk+:128]<=k_bits;
     if(chunk==0)begin first_saved<=first_beat;tag_saved<=in_tag;norm_saved<=normalization;end
     if(chunk==CHUNKS-1)begin chunk<=0;full<=1;last_saved<=last_beat;end
     else chunk<=chunk+1;
    end
   end
  end
 end endgenerate
 dense_score #(.LANES(LANES)) core(.clk(clk),.rst(rst),.in_valid(core_valid),
 .first_beat(core_first),.last_beat(core_last),.in_tag(core_tag),.normalization(core_norm),
 .q_bits(core_q),.k_bits(core_k),.in_ready(core_ready),.out_valid(out_valid),
 .out_tag(out_tag),.score_bits(score_bits),.out_ready(out_ready),
 .fp_issue(fp_issue),.fp_complete(fp_complete),.issue_tags(issue_tags),
 .complete_tags(complete_tags),.issue_nodes(issue_nodes),.complete_nodes(complete_nodes),
 .complete_bits(complete_bits),.empty(core_empty));
endmodule
