// Four independent 64-bit tagged packet lanes; finite output and DMA staging.
module score_packet_fifo #(parameter DEPTH=16,AW=$clog2(DEPTH))(
 input wire clk,rst,input wire in_valid,output wire in_ready,input wire [63:0] in_data,
 output wire out_valid,input wire out_ready,output reg [63:0] out_data,output reg [AW:0] count);
 reg [63:0] data[0:DEPTH-1];reg [DEPTH-1:0] wp,rp;
 integer j,r;
 always @*begin out_data=0;for(r=0;r<DEPTH;r=r+1)out_data=out_data|(data[r]&{64{rp[r]}});end
 assign in_ready=count<DEPTH;assign out_valid=count!=0;
 wire push=in_valid&&in_ready,pop=out_valid&&out_ready;
 always @(posedge clk)begin
  if(rst)begin wp<=1;rp<=1;count<=0;end
  else begin
   if(push)begin for(j=0;j<DEPTH;j=j+1)if(wp[j])data[j]<=in_data;wp<={wp[DEPTH-2:0],wp[DEPTH-1]};end
   if(pop)rp<={rp[DEPTH-2:0],rp[DEPTH-1]};
   case({push,pop})2'b10:count<=count+1'b1;2'b01:count<=count-1'b1;default:count<=count;endcase
  end
 end
endmodule
module score_stream_interface(input wire clk,rst,input wire [3:0] in_valid,output wire [3:0] in_ready,
 input wire [255:0] in_data,output wire [3:0] out_valid,input wire [3:0] out_ready,
 output wire [255:0] out_data,output wire [31:0] lane_occupancy);
 genvar l;generate for(l=0;l<4;l=l+1)begin:lanes
  wire v,r;wire [63:0] d;wire [4:0] a;wire [2:0] b;
  score_packet_fifo #(16) holding(clk,rst,in_valid[l],in_ready[l],in_data[l*64+:64],v,r,d,a);
  score_packet_fifo #(4) dma(clk,rst,v,r,d,out_valid[l],out_ready[l],out_data[l*64+:64],b);
  assign lane_occupancy[l*8+:8]=a+b;
 end endgenerate
endmodule
