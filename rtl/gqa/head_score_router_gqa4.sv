// Thirty-two independent, bounded head FIFOs. One write and one selected read/edge.
// (head,token) travels in the 48-bit A2 tag; the selector requests its next token.
module head_score_router_gqa4 #(parameter HEADS=32, DEPTH=4, PW=$clog2(DEPTH), CW=$clog2(DEPTH+1))(
 input wire clk,rst,in_valid, output wire in_ready,
 input wire [47:0] in_tag,input wire [15:0] in_bits,
 input wire [4:0] read_head, output wire out_valid,
 output wire [47:0] out_tag,output wire [15:0] out_bits,input wire out_ready,
 output wire [HEADS*CW-1:0] occupancy
);
 reg [63:0] packets[0:HEADS*DEPTH-1];reg [PW-1:0] rd[0:HEADS-1],wr[0:HEADS-1];reg [CW-1:0] count[0:HEADS-1];
 wire [4:0] dest=in_tag[44:40];
 assign in_ready=count[dest]<DEPTH;
 assign out_valid=count[read_head]!=0;
 assign {out_tag,out_bits}=packets[read_head*DEPTH+rd[read_head]];
 genvar h;
 generate for(h=0;h<HEADS;h=h+1)begin:heads
  assign occupancy[h*CW+:CW]=count[h];
  always @(posedge clk)begin
   if(rst)begin rd[h]<=0;wr[h]<=0;count[h]<=0;end
   else begin
    case({in_valid&&in_ready&&dest==h,out_valid&&out_ready&&read_head==h})
     2'b10:count[h]<=count[h]+1;
     2'b01:count[h]<=count[h]-1;
     default:;
    endcase
    if(in_valid&&in_ready&&dest==h)begin
     packets[h*DEPTH+wr[h]]<={in_tag,in_bits};
     if(wr[h]==DEPTH-1)wr[h]<=0;else wr[h]<=wr[h]+1;
    end
    if(out_valid&&out_ready&&read_head==h)begin
     if(rd[h]==DEPTH-1)rd[h]<=0;else rd[h]<=rd[h]+1;
    end
   end
  end
 end endgenerate
endmodule
