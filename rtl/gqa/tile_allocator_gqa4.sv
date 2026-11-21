// Eight-KV-head GQA schedule. One reserved tile at a time without prefetch;
// two reserved slots maximum with prefetch. Epoch is supplied at reset after
// upstream drain, or changed before restart so late responses cannot alias.
module tile_allocator_gqa4 #(parameter IW=12)(
 input wire clk,rst,enable,prefetch,any_occupied,input wire [IW-1:0] n_local,
 input wire [3:0] tile_tokens,input wire [31:0] epoch,
 output wire valid,input wire ready,output reg [IW-1:0] start,
 output wire [3:0] count,output reg [2:0] kv,output reg [31:0] tag,output reg done
);
 wire [IW-1:0] left=n_local-start;
 assign count=left<tile_tokens?left:tile_tokens;
 assign valid=enable&&!done&&n_local!=0&&(prefetch||!any_occupied);
 always @(posedge clk)begin
  if(rst)begin start<=0;kv<=0;tag<=epoch;done<=0;end
  else if(valid&&ready)begin
   tag<=tag+1;
   if(start+count==n_local)begin start<=0;if(kv==7)done<=1;else kv<=kv+1;end
   else start<=start+count;
  end
 end
endmodule
