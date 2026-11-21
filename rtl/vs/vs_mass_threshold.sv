// VS: mass-threshold V-skip engine, one per KV head (GQA group of 4 query heads, shared selection list).
//
// Phase K (exact QK on the selected tokens): for each logit l of head h at selection rank r
//   X   = floor( fp32_fma(l, 2^19, C_h) )        C_h = (255 - 8*(anchor_h + M)) * 2^16   (round_min to int)
//   bin = X>>16 (1..255; <1 tail, >255 saturated), f = X & 0xffff
//   hist[h][bin] += (T1[f>>8] * T2[f&255]) >> 16  (~ e^{f/2^19} * 2^16, rounded down)    tail: c0[h]++   sat: keep_all[h]
//   bincode[r].byte[h] = clamp(bin, 0, 255)
// Scan (after the K barrier), per head: T = sum_b hist*E[b] + c0*E0P;  thr = T*KD;
//   b*_h = largest b with sum_{b'>=b} hist*E[b'] >= thr  (0 if none / keep_all)
// Phase V: token at rank r keeps its V row iff  OR_h (keep_all[h] | bincode[r].byte[h] >= b*_h).
// The kept set is a superset of the exact sorted-prefix set (bit-matching software reference:
// experiments/replay/make_vkeep_masks.py). SRAMs are external macros (CACTI): hist 1024x30b, bincode 2^RW x 32b.
`include "HardFloat_consts.vi"
module vs_mass_threshold #(parameter RW=13)(
 input wire clk,rst,
 input wire [127:0] c_coef,                 // FP32 C_h, h=0..3
 input wire [31:0] k_delta,e0p,              // FP32
 input wire clear,output wire busy,
 input wire lg_valid,output wire lg_ready,input wire [1:0] lg_head,input wire [RW-1:0] lg_rank,input wire [31:0] lg_logit,
 input wire scan_start,output reg scan_done,output reg [31:0] bstar,output reg [3:0] keep_all,
 input wire kq_valid,input wire [RW-1:0] kq_rank,output reg kq_out_valid,output reg kq_keep,
 output wire h_re,output wire [9:0] h_raddr,input wire [29:0] h_rdata,output wire h_we,output wire [9:0] h_waddr,output wire [29:0] h_wdata,
 output wire b_we,output wire [3:0] b_be,output wire [RW-1:0] b_waddr,output wire [31:0] b_wdata,
 output wire b_re,output wire [RW-1:0] b_raddr,input wire [31:0] b_rdata
);
`include "vs_tables.svh"
 // ---------------- state ----------------
 localparam IDLE=0,CLR=1,SA=2,ST=3,SK=4,SB=5,DONE=6;
 reg [2:0] st; reg [10:0] cnt; reg [1:0] sh; reg [8:0] sb;   // scan head / bin pointer
 reg [13:0] c0[0:3];
 assign busy=st!=IDLE;
 // ---------------- phase K update pipeline ----------------
 assign lg_ready=(st==IDLE);
 wire acc_in=lg_valid&&lg_ready;
 reg v1,v2,v3; reg [1:0] h1,h2,h3; reg [RW-1:0] r1,r2; reg [32:0] l1,c1; reg [32:0] x2;
 wire [32:0] lrec,crec,k19rec; wire [32:0] xrec;
 fNToRecFN #(8,24) cl(lg_logit,lrec),cc(c_coef[lg_head*32+:32],crec),ck(32'h49000000,k19rec);
 mulAddRecFN #(8,24) fx(.control(`flControl_tininessAfterRounding),.op(2'b00),.a(l1),.b(k19rec),.c(c1),
  .roundingMode(`round_near_even),.out(xrec),.exceptionFlags());
 wire [31:0] xi; wire [2:0] xflags;
 recFNToIN #(8,24,32) toi(.control(`flControl_tininessAfterRounding),.in(x2),.roundingMode(`round_min),.signedOut(1'b1),.out(xi),.intExceptionFlags(xflags));
 wire tail=!xflags[2]&&$signed(xi)<$signed(32'd65536);              // below bin 1 (overflow saturates to INT_MIN)
 wire sat=xflags[2]||(!tail&&xi>=32'd16777216);                     // >= 256*2^16, or NaN: keep the whole head
 wire [7:0] bin=xi[23:16]; wire [15:0] fr=xi[15:0];
 wire [33:0] prod=vs_t1(fr[15:8])*vs_t2(fr[7:0]);
 wire [17:0] inc=prod[33:16];
 reg [7:0] bin3; reg [17:0] inc3; reg norm3;
 // histogram read-modify-write with one-entry forwarding
 reg fw_v; reg [9:0] fw_a; reg [29:0] fw_d;
 wire [9:0] a3={h3,bin3};
 wire [29:0] base3=(fw_v&&fw_a==a3)?fw_d:h_rdata;
 wire [29:0] upd=base3+inc3;
 // ---------------- scan datapath ----------------
 reg acc_valid_q; reg [7:0] last_b;
 reg rdv1,rdv2; reg [7:0] rb1,rb2; reg [32:0] srec2; reg [32:0] acc,thr; reg hit;
 wire [32:0] sr_now,c0rec,e0rec,kdrec,eb_rec,fma_out; wire [31:0] eb=vs_eb(rb2);
 iNToRecFN #(30,8,24) sconv(.control(`flControl_tininessAfterRounding),.signedIn(1'b0),.in(h_rdata),.roundingMode(`round_near_even),.out(sr_now),.exceptionFlags());
 iNToRecFN #(30,8,24) cconv(.control(`flControl_tininessAfterRounding),.signedIn(1'b0),.in({16'd0,c0[sh]}),.roundingMode(`round_near_even),.out(c0rec),.exceptionFlags());
 fNToRecFN #(8,24) ce(e0p,e0rec),ckd(k_delta,kdrec),ceb(eb,eb_rec);
 reg [32:0] fa,fb,fc;
 always @*begin
  fa=srec2;fb=eb_rec;fc=acc;
  if(st==ST)begin fa=c0rec;fb=e0rec;end
  if(st==SK)begin fa=acc;fb=kdrec;fc=33'h0;end
 end
 mulAddRecFN #(8,24) fs(.control(`flControl_tininessAfterRounding),.op(2'b00),.a(fa),.b(fb),.c(fc),
  .roundingMode(`round_near_even),.out(fma_out),.exceptionFlags());
 wire ge_lt; compareRecFN #(8,24) cmp(.a(acc),.b(thr),.signaling(1'b0),.lt(ge_lt),.eq(),.gt(),.unordered(),.exceptionFlags());
 // ---------------- SRAM ports ----------------
 wire scan_rd=(st==SA&&sb<=9'd255&&sb>=9'd1)||(st==SB&&!hit&&sb>=9'd1&&sb<=9'd255);
 assign h_re=(v2&&!tail&&!sat)||scan_rd;
 assign h_raddr=scan_rd?{sh,sb[7:0]}:{h2,bin};
 assign h_we=(st==CLR)||(v3&&norm3);
 assign h_waddr=(st==CLR)?cnt[9:0]:a3;
 assign h_wdata=(st==CLR)?30'd0:upd;
 assign b_we=v2; assign b_waddr=r2; assign b_be=4'b1<<h2;
 assign b_wdata={4{tail?8'd0:sat?8'd255:bin}};
 assign b_re=kq_valid; assign b_raddr=kq_rank;
 integer k;
 always @(posedge clk)begin
  if(rst)begin st<=IDLE;v1<=0;v2<=0;v3<=0;fw_v<=0;scan_done<=0;keep_all<=0;kq_out_valid<=0;rdv1<=0;rdv2<=0;
   for(k=0;k<4;k=k+1)c0[k]<=0;end
  else begin
   // update pipeline
   v1<=acc_in; if(acc_in)begin l1<=lrec;c1<=crec;h1<=lg_head;r1<=lg_rank;end
   v2<=v1; x2<=xrec; h2<=h1; r2<=r1;
   v3<=v2; h3<=h2; bin3<=bin; inc3<=inc; norm3<=v2&&!tail&&!sat;
   if(v2&&tail)c0[h2]<=c0[h2]+1'b1;
   if(v2&&sat)keep_all[h2]<=1'b1;
   fw_v<=v3&&norm3; fw_a<=a3; fw_d<=upd;
   // keep query
   kq_out_valid<=kq_v1;
   // scan read pipeline: address at t, data at t+1 (convert, register), FMA at t+2
   rdv1<=scan_rd; rb1<=sb[7:0]; rdv2<=rdv1; rb2<=rb1; srec2<=sr_now;
   scan_done<=0;
   case(st)
    IDLE:begin
     if(clear)begin st<=CLR;cnt<=0;keep_all<=0;for(k=0;k<4;k=k+1)c0[k]<=0;end
     else if(scan_start&&!v1&&!v2&&!v3)begin st<=SA;sh<=0;sb<=1;acc<=0;hit<=0;end
    end
    CLR:begin cnt<=cnt+1'b1;if(cnt==11'd1023)st<=IDLE;end
    SA:begin   // pass A: total over bins 1..255
     if(sb<=9'd255)sb<=sb+1'b1;
     if(rdv2)acc<=fma_out;
     if(sb>9'd255&&!rdv1&&!rdv2)st<=ST;
    end
    ST:begin acc<=fma_out;st<=SK;end          // + c0*E0P
    SK:begin thr<=fma_out;acc<=0;sb<=9'd255;hit<=0;st<=SB;end
    SB:begin   // pass B: cumulative from the top bin
     if(!hit&&sb>=9'd1)sb<=sb-1'b1;
     if(rdv2&&!hit)begin acc<=fma_out;end
     // compare the cumulative value registered in the previous cycle (acc includes bins >= rb of last FMA)
     if(!hit&&acc_valid_q&&!ge_lt)begin hit<=1;bstar[sh*8+:8]<=keep_all[sh]?8'd0:last_b;end
     if((hit&&!rdv1&&!rdv2)||(!hit&&sb==0&&!rdv1&&!rdv2&&!acc_valid_q))begin
      if(!hit)bstar[sh*8+:8]<=8'd0;
      if(sh==2'd3)begin st<=DONE;end
      else begin sh<=sh+1'b1;sb<=1;acc<=0;hit<=0;st<=SA;end
     end
    end
    DONE:begin scan_done<=1;st<=IDLE;end
    default:st<=IDLE;
   endcase
  end
 end
 // acc_valid_q: acc was updated by a pass-B FMA in the previous cycle; last_b: that bin
 always @(posedge clk)begin
  if(rst)acc_valid_q<=0;
  else begin acc_valid_q<=(st==SB)&&rdv2&&!hit; last_b<=rb2; end
 end
 // keep decision: read at t, data at t+1, registered decision valid at t+2
 reg kq_v1; always @(posedge clk) kq_v1<=rst?1'b0:kq_valid;
 wire [3:0] kh;
 genvar g; generate for(g=0;g<4;g=g+1)begin:kd
  assign kh[g]=keep_all[g]||(b_rdata[g*8+:8]>=bstar[g*8+:8]);
 end endgenerate
 always @(posedge clk) kq_keep<=|kh;   // valid with kq_out_valid of the next cycle (data at t+1, decision at t+2)
endmodule
