// Fixed D128/G64 group DAG. A:9 nodes, B:11 nodes. No token context mux here.
// A nodes: p0 p1 p2 p3 a0 b0 a1 b1 group.
// B nodes: p0 p1 p2 p3 t0 t1 t2 t3 r0 r1 group.
module score7_group #(parameter ARCH_B=0,NODES=ARCH_B?11:9)(
 input wire clk,rst,load,input wire [15:0] in_ql,in_dq,in_kl,in_dk,
 input wire count_load,input wire [31:0] count_values,
 input wire issue,input wire wb,input wire [3:0] wb_op,input wire [15:0] wb_bits,
 output reg ready,output reg [3:0] op,output reg [15:0] a,b,c,
 output wire done,output wire [15:0] group_score);
 reg occupied,count_ready;
 reg [15:0] ql,dq,kl,dk;
 reg [31:0] saved_counts;
 reg [NODES-1:0] issued,completed;
 reg [15:0] value[0:NODES-1];
 wire [7:0] selected_count=op==4?saved_counts[31:24]:
  op==5?(ARCH_B?saved_counts[23:16]:saved_counts[15:8]):
  op==6?(ARCH_B?saved_counts[15:8]:saved_counts[23:16]):saved_counts[7:0];
 wire [15:0] count_operand;
 count_to_bf16 count_convert(selected_count,count_operand);
 wire [NODES-1:0] deps;
 assign deps[3:0]=4'b1111;
 wire [15:0] reduction0,reduction1;
 generate if(ARCH_B)begin:independent_deps
  assign deps[4]=completed[0]&&count_ready;
  assign deps[5]=completed[1]&&count_ready;
  assign deps[6]=completed[2]&&count_ready;
  assign deps[7]=completed[3]&&count_ready;
  assign deps[8]=completed[4]&&completed[5];
  assign deps[9]=completed[6]&&completed[7];
  assign deps[10]=completed[8]&&completed[9];
  assign reduction0=value[8];assign reduction1=value[9];
 end else begin:pairwise_deps
  assign deps[4]=completed[0]&&count_ready;
  assign deps[5]=completed[2]&&count_ready;
  assign deps[6]=completed[1]&&completed[4]&&count_ready;
  assign deps[7]=completed[3]&&completed[5]&&count_ready;
  assign deps[8]=completed[6]&&completed[7];
  assign reduction0=0;assign reduction1=0;
 end endgenerate
 integer i;
 always @*begin
  ready=0;op=0;
  // Highest ready node first shortens the remaining critical chain.
  for(i=0;i<NODES;i=i+1)if(occupied&&!issued[i]&&deps[i])begin ready=1;op=i;end
  a=0;b=0;c=0;
  case(op)
   0:begin a=dq;b=dk;end
   1:begin a=ql;b=dk;end
   2:begin a=dq;b=kl;end
   3:begin a=ql;b=kl;end
   4:begin a=value[0];b=count_operand;end
   5:begin a=ARCH_B?value[1]:value[2];b=count_operand;end
   6:begin a=ARCH_B?value[2]:value[1];b=count_operand;c=ARCH_B?0:value[4];end
   7:begin a=value[3];b=count_operand;c=ARCH_B?0:value[5];end
   8:begin a=16'h3f80;b=ARCH_B?value[4]:value[6];c=ARCH_B?value[5]:value[7];end
   9:begin a=16'h3f80;b=value[6];c=value[7];end
   10:begin a=16'h3f80;b=reduction0;c=reduction1;end
  endcase
  // MUL uses zero of product's sign, so negative zero is preserved.
  if(op<6||(ARCH_B&&op<8))c={a[15]^b[15],15'b0};
 end
 assign done=completed[NODES-1];assign group_score=value[NODES-1];
 always @(posedge clk)begin
  if(rst)begin occupied<=0;count_ready<=0;issued<=0;completed<=0;end
  else begin
   if(load)begin occupied<=1;count_ready<=0;issued<=0;completed<=0;
    ql<=in_ql;dq<=in_dq;kl<=in_kl;dk<=in_dk;end
   if(count_load)begin saved_counts<=count_values;count_ready<=1;end
   if(issue)issued[op]<=1;
   if(wb)begin value[wb_op]<=wb_bits;completed[wb_op]<=1;end
  end
 end
endmodule
