// HardFloat Release 1, expWidth=8, sigWidth=8. True 8x8 significand product.
// Three registered stages: preMul; multiply+postMul; BF16 rounding/unpack.
// II=1. RNE, gradual underflow, canonical RISC-V quiet NaN. No FP32 arithmetic.
module bf16_fma_pipeline(input wire clk,rst,in_valid,
 input wire [15:0] a,b,c, output wire out_valid,
 output wire [15:0] result,output wire [4:0] flags);
 wire [16:0] ar,br,cr;
 fNToRecFN #(8,8) ca(a,ar),cb(b,br),cc(c,cr);
 wire [7:0] ma,mb;wire [15:0] mc;wire [5:0] st;
 wire signed [9:0] ex;wire [3:0] di;wire [9:0] hi;
 mulAddRecFNToRaw_preMul #(8,8) pre(1'b1,2'b00,ar,br,cr,3'b000,ma,mb,mc,st,ex,di,hi);
 reg [7:0] ma1,mb1;reg [15:0] mc1;reg [5:0] st1;
 reg signed [9:0] ex1;reg [3:0] di1;reg [9:0] hi1;
 wire [15:0] product=ma1*mb1;
 wire [16:0] prod={1'b0,product}+{1'b0,mc1};
 wire inv,nan,inf,zero,sign;wire signed [9:0] er;wire [10:0] sr;
 mulAddRecFNToRaw_postMul #(8,8) post(st1,ex1,di1,hi1,prod,3'b000,inv,nan,inf,zero,sign,er,sr);
 reg inv2,nan2,inf2,zero2,sign2;reg signed [9:0] ex2;reg [10:0] sig2;
 wire [16:0] rounded;wire [4:0] fl;wire [15:0] ieee;
 roundRawFNToRecFN #(8,8,0) rounder(1'b1,inv2,1'b0,nan2,inf2,zero2,sign2,ex2,sig2,3'b000,rounded,fl);
 recFNToFN #(8,8) unpack(rounded,ieee);
 reg [15:0] res3;reg [4:0] flags3;reg [2:0] valid;
 assign out_valid=valid[2];assign result=res3;assign flags=flags3;
 always @(posedge clk)begin
  if(rst)valid<=0;
  else begin
   valid<={valid[1:0],in_valid};
   if(in_valid)begin ma1<=ma;mb1<=mb;mc1<=mc;st1<=st;ex1<=ex;di1<=di;hi1<=hi;end
   if(valid[0])begin inv2<=inv;nan2<=nan;inf2<=inf;zero2<=zero;sign2<=sign;ex2<=er;sig2<=sr;end
   if(valid[1])begin res3<=ieee;flags3<=fl;end
  end
 end
endmodule
