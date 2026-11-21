// Four private preserved II=2 engines; finite shared tagged ingress/egress.
// Each score FIFO is registered. Per-engine frames do not overlap.
module selector_cluster #(parameter MAX_N=32,MAX_K=16,FIFO_DEPTH=4)(
 input wire clk,rst,
 input wire start_valid,input wire [1:0] start_engine,input wire [7:0] start_tag,
 input wire [15:0] cfg_n,cfg_k,output wire start_ready,
 input wire in_valid,input wire [1:0] in_engine,input wire [15:0] in_score,input wire in_keep,
 output wire in_ready,
 output wire out_valid,output wire [31:0] out_index,output wire [7:0] out_tag,
 output wire out_last,input wire out_ready,output wire [3:0] done,error,busy);
 localparam AW=$clog2(FIFO_DEPTH),CW=$clog2(FIFO_DEPTH+1);
 reg [16:0] fifo[0:4*FIFO_DEPTH-1];
 reg [AW-1:0] wr[0:3],rd[0:3];reg [CW-1:0] count[0:3];
 reg [7:0] tag[0:3];
 wire [3:0] core_in_ready,core_out_valid,core_out_last;
 wire [31:0] core_index[0:3];
 wire [3:0] push,pop;
 reg [1:0] rr,grant,held_grant;reg found,locked;
 integer i,offset,choice;
 assign start_ready=!busy[start_engine]&&(count[start_engine]==0);
 assign in_ready=busy[in_engine]&&(count[in_engine]<FIFO_DEPTH);
 always @* begin
   grant=held_grant;found=0;
   if(locked)found=core_out_valid[held_grant];
   else for(offset=0;offset<4;offset=offset+1)begin
     choice=(rr+offset)&3;
     if(!found&&core_out_valid[choice])begin grant=choice;found=1;end
   end
 end
 assign out_valid=found;
 assign out_index=core_index[grant];assign out_last=core_out_last[grant];assign out_tag=tag[grant];
 genvar e;
 generate for(e=0;e<4;e=e+1)begin: engines
   assign push[e]=in_valid&&in_ready&&(in_engine==e);
   assign pop[e]=(count[e]!=0)&&core_in_ready[e];
   radix_selection_pipeline #(.MAX_N(MAX_N),.MAX_K(MAX_K)) core(
     .clk(clk),.rst(rst),.start(start_valid&&start_ready&&(start_engine==e)),
     .cfg_n(cfg_n),.cfg_k(cfg_k),.reserve_current(1'b0),
     .in_valid(count[e]!=0),.in_keep(fifo[e*FIFO_DEPTH+rd[e]][16]),
     .in_score(fifo[e*FIFO_DEPTH+rd[e]][15:0]),.in_ready(core_in_ready[e]),
     .out_valid(core_out_valid[e]),.out_index(core_index[e]),.out_last(core_out_last[e]),
     .out_ready(found&&(grant==e)&&out_ready),.done(done[e]),.error(error[e]),.busy(busy[e]));
 end endgenerate
 always @(posedge clk)begin
   if(rst)begin
     rr<=0;locked<=0;held_grant<=0;
     for(i=0;i<4;i=i+1)begin wr[i]<=0;rd[i]<=0;count[i]<=0;tag[i]<=0;end
   end else begin
     if(out_valid&&!out_ready)begin locked<=1;held_grant<=grant;end
     if(out_valid&&out_ready)begin locked<=0;rr<=grant+1'b1;end
     for(i=0;i<4;i=i+1)begin
       if(start_valid&&start_ready&&start_engine==i)tag[i]<=start_tag;
       if(push[i])begin fifo[i*FIFO_DEPTH+wr[i]]<={in_keep,in_score};wr[i]<=wr[i]+1'b1;end
       if(pop[i])rd[i]<=rd[i]+1'b1;
       case({push[i],pop[i]})
         2'b10:count[i]<=count[i]+1'b1;
         2'b01:count[i]<=count[i]-1'b1;
         default:count[i]<=count[i];
       endcase
     end
   end
 end
endmodule
