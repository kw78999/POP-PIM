// Exact BF16 x0.25 with RNE (bit-identical to bf16_fma_pipeline(x, 0.25, -0), incl. subnormals, +-0, Inf,
// canonical NaN 0x7FC0); verified exhaustively over all 2^16 inputs (tests/tb_quarter.cpp). Two register stages
// so the GS output timing matches a 3-stage FMA slot.
module bf16_quarter(input wire clk,input wire [15:0] x,output reg [15:0] y);
 wire s=x[15]; wire [7:0] e=x[14:7]; wire [6:0] m=x[6:0];
 wire [7:0] sig={e!=0,m};
 wire [1:0] sh=(e==0)?2'd2:(e==1)?2'd2:2'd1;       // E=max(e,1): result frac = sig >> (3-E)
 wire [7:0] q=sig>>sh; wire [1:0] rem=sh==2?sig[1:0]:{sig[0],1'b0};
 wire up=rem[1]&&(rem[0]||q[0]);                     // RNE: half and above; tie to even
 wire [7:0] r=q+up;                                  // 128 -> smallest normal via the exp field
 reg [15:0] y1;
 always @*begin
  if(e==8'hff) y1=(m!=0)?16'h7fc0:x;               // NaN -> canonical, Inf -> Inf
  else if(e==0&&m==0) y1=x;                         // +-0
  else if(e>=3) y1={s,e-8'd2,m};
  else y1={s,7'd0,r};
 end
 reg [15:0] y0; always @(posedge clk)begin y0<=y1;y<=y0;end
endmodule
// GS unit: GQA-shared score. Sits between the A2 scorer output and the bank's score router.
// The metadata stream emits a TILE(=4)-token tile head-major (h0 t0..t3, h1 t0..t3, ...), so per token it keeps
//   A = s0, then s0+s1 (BF16 RNE);  B = s2;  on s3: out = ((A + (B + s3)) * 0.25)   (every op BF16 RNE, as
// poppim/variants.py: gs_aggregate). x0.25 is an FMA with addend -0 so that -0 stays -0.
// One FMA serves s0+s1 and s2+s3 (they never arrive in the same cycle); a second FMA adds the pair sums and an exact
// x0.25 unit scales. Only (kv, local token, bank) = 21 bits of tag travel through the pipeline.
// ENTRIES in-flight tiles, keyed by (kv, token>>2). Output carries the leader head (kv*4) in the tag.
module gs_shared_score #(parameter ENTRIES=2, ODEPTH=4)(
 input wire clk,rst,
 input wire in_valid,output wire in_ready,input wire [47:0] in_tag,input wire [15:0] in_bits,
 output wire out_valid,input wire out_ready,output wire [47:0] out_tag,output wire [15:0] out_bits);
 localparam EW=ENTRIES<=1?1:$clog2(ENTRIES), OW=$clog2(ODEPTH+1)+1;
 wire [4:0] head=in_tag[44:40]; wire [1:0] ph=head[1:0]; wire [2:0] kv=head[4:2];
 wire [11:0] tok=in_tag[17:6]; wire [1:0] slot=tok[1:0]; wire [12:0] key={kv,tok[11:2]};
 reg [ENTRIES-1:0] valid; reg [12:0] keys[0:ENTRIES-1]; reg [3:0] seen0[0:ENTRIES-1],seen3[0:ENTRIES-1];
 reg [15:0] A[0:ENTRIES*4-1],B[0:ENTRIES*4-1]; reg [47:0] etag[0:ENTRIES-1];
 // lookup / allocate
 reg hit; reg [EW-1:0] hidx,fidx; reg free; integer e;
 always @*begin hit=0;hidx=0;free=0;fidx=0;
  for(e=ENTRIES-1;e>=0;e=e-1)begin
   if(valid[e]&&keys[e]==key)begin hit=1;hidx=e;end
   if(!valid[e])begin free=1;fidx=e;end
  end end
 wire [EW-1:0] idx=hit?hidx:fidx;
 // output credit: results in flight (<= 9 cycles of FMA latency) plus FIFO occupancy never exceed ODEPTH
 reg [OW-1:0] count; reg [3:0] inflight;
 wire credit=(count+inflight)<ODEPTH;
 assign in_ready=(ph==0)?(hit||free):(hit&&(ph!=3||credit));
 wire fire=in_valid&&in_ready;
 // FMA A: a*1.0+c
 wire [15:0] a_in=ph==1?A[idx*4+slot]:B[idx*4+slot];
 wire useA=fire&&(ph==1||ph==3);
 wire va,vb,vc; wire [15:0] ra,rb,rc;
 bf16_fma_pipeline fa(.clk(clk),.rst(rst),.in_valid(useA),.a(a_in),.b(16'h3F80),.c(in_bits),.out_valid(va),.result(ra),.flags());
 // side pipeline for FMA A: phase-3 flag, entry/slot, carried s01 and tag
 reg [2:0] p3; reg [EW+1:0] ws[0:2]; reg [15:0] s01[0:2]; reg [20:0] tg[0:2];
 // FMA B: s01*1.0 + s23
 bf16_fma_pipeline fb(.clk(clk),.rst(rst),.in_valid(va&&p3[2]),.a(s01[2]),.b(16'h3F80),.c(ra),.out_valid(vb),.result(rb),.flags());
 reg [20:0] tb[0:2],tc[0:1]; reg [1:0] vq;
 // x0.25 (exact, 2 stages)
 bf16_quarter fc(.clk(clk),.x(rb),.y(rc)); assign vc=vq[1];
 // output FIFO
 reg [36:0] fifo[0:ODEPTH-1]; reg [$clog2(ODEPTH)-1:0] wp,rp;
 assign out_valid=count!=0; wire [20:0] ft=fifo[rp][36:16]; assign out_bits=fifo[rp][15:0];
 assign out_tag={3'd0,ft[20:18],2'd0,5'd0,ft[20:18],14'd0,ft[17:0]};
 wire pop=out_valid&&out_ready;
 integer i;
 always @(posedge clk)begin
  if(rst)begin valid<=0;count<=0;inflight<=0;wp<=0;rp<=0;p3<=0;vq<=0;end
  else begin
   // FMA A side pipeline
   p3<={p3[1:0],useA&&ph==3}; vq<={vq[0],vb};
   ws[0]<={idx,slot};ws[1]<=ws[0];ws[2]<=ws[1];
   s01[0]<=A[idx*4+slot];s01[1]<=s01[0];s01[2]<=s01[1];
   tg[0]<={kv,in_tag[17:0]};tg[1]<=tg[0];tg[2]<=tg[1];
   tb[0]<=tg[2];tb[1]<=tb[0];tb[2]<=tb[1];tc[0]<=tb[2];tc[1]<=tc[0];
   if(va&&!p3[2])A[ws[2]]<=ra;                        // s0+s1 write-back
   if(fire)begin
    case(ph)
     2'd0:begin if(!hit)begin valid[idx]<=1;keys[idx]<=key;seen0[idx]<=0;seen3[idx]<=0;end
                seen0[idx][slot]<=1;A[idx*4+slot]<=in_bits;end
     2'd2:B[idx*4+slot]<=in_bits;
     2'd3:begin seen3[idx][slot]<=1;
                if((seen3[idx]|(4'b1<<slot))==seen0[idx])valid[idx]<=0;end   // tile retired after its last s3
     default:;
    endcase
   end
   inflight<=inflight+(fire&&ph==3)-vc;
   if(vc)begin fifo[wp]<={tc[1],rc};wp<=wp+1;end
   if(pop)rp<=rp+1;
   count<=count+vc-pop;
  end
 end
endmodule

