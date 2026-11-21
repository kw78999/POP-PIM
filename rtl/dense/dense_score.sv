// Phase9 dense BF16 QK. Two score contexts, four recurrence chains per lane.
// Each lane reuses one unchanged Phase7 BF16 FMA for dot, tree reduction, norm.
// Input: ceil(128/LANES) BF16 Q/K beats, first/last and 48-bit token tag.
// Output: ordered BF16 score. Finite buffering, output backpressure, reset flush.
module dense_score #(parameter LANES=1,PARTS=4*LANES,REQS=128+PARTS)(
 input wire clk,rst,in_valid,first_beat,last_beat,input wire [47:0] in_tag,
 input wire [LANES*16-1:0] q_bits,k_bits,input wire [15:0] normalization,
 output wire in_ready,out_valid,output wire [47:0] out_tag,output wire [15:0] score_bits,input wire out_ready,
 output wire [LANES-1:0] fp_issue,fp_complete,
 output wire [LANES*48-1:0] issue_tags,complete_tags,
 output wire [LANES*8-1:0] issue_nodes,complete_nodes,
 output wire [LANES*16-1:0] complete_bits,output wire empty);
 reg [1:0] live,closed,result_ready,norm_issued;
 reg write_ctx,head;reg [7:0] input_beat;
 reg [47:0] tags[0:1];reg [15:0] norms[0:1],results[0:1];
 reg [8:0] dot_issued[0:1],dot_completed[0:1];
 reg [7:0] width[0:1];
 reg [2*LANES-1:0] red_issued[0:1],red_completed[0:1];
 reg [PARTS-1:0] used[0:1],partial_ready[0:1];
 reg [15:0] val[0:2*PARTS-1];
 reg buffer_valid,buffer_ctx,buffer_first;reg [7:0] buffer_beat;
 reg [LANES*16-1:0] buffer_q,buffer_k;reg [LANES-1:0] buffer_pending;
 wire input_context_ready=first_beat?!live[write_ctx]:(live[write_ctx]&&!closed[write_ctx]);
 wire data_ctx=buffer_ctx;
 wire [7:0] data_beat=buffer_beat;
 wire data_first=buffer_first;
 wire [LANES*16-1:0] data_q=buffer_q,data_k=buffer_k;
 wire data_available=buffer_valid;
 wire [LANES-1:0] dot_grant;
 wire buffer_drained=!(buffer_pending&~dot_grant);
 assign in_ready=input_context_ready&&(!buffer_valid||buffer_drained);
 wire accepted=in_valid&&in_ready;
 assign out_valid=live[head]&&result_ready[head];assign out_tag=tags[head];assign score_bits=results[head];assign empty=!(|live)&&!buffer_valid;
 wire [1:0] dot_done;
 assign dot_done[0]=closed[0]&&dot_completed[0]==128;
 assign dot_done[1]=closed[1]&&dot_completed[1]==128;
 wire [LANES-1:0] issue_ctx,return_ctx;
 wire [1:0] issue_kind[0:LANES-1],return_kind[0:LANES-1];
 wire [7:0] issue_index[0:LANES-1],return_index[0:LANES-1];
 wire [1:0] issue_slot[0:LANES-1],return_slot[0:LANES-1];
 genvar lane;
 generate for(lane=0;lane<LANES;lane=lane+1)begin:mac
  wire [15:0] dot_value[0:1],reduce_b[0:1],reduce_c[0:1];
  wire dot_used[0:1],dot_ready[0:1];
  genvar ct;
  for(ct=0;ct<2;ct=ct+1)begin:operands
   wire second_wave=red_issued[ct][lane];
   assign reduce_b[ct]=second_wave?val[ct*PARTS+2*(LANES+lane)]:val[ct*PARTS+2*lane];
   assign reduce_c[ct]=second_wave?val[ct*PARTS+2*(LANES+lane)+1]:val[ct*PARTS+2*lane+1];
   assign dot_value[ct]=data_beat[1:0]==0?val[ct*PARTS+lane]:data_beat[1:0]==1?val[ct*PARTS+LANES+lane]:data_beat[1:0]==2?val[ct*PARTS+2*LANES+lane]:val[ct*PARTS+3*LANES+lane];
   assign dot_used[ct]=used[ct][data_beat[1:0]*LANES+lane];
   assign dot_ready[ct]=partial_ready[ct][data_beat[1:0]*LANES+lane];
  end
  reg chosen,ctx;reg [1:0] kind,slot;reg [7:0] index;
  reg [15:0] a,b,c;integer off,cc,j,source,pi;
  always @*begin
   chosen=0;ctx=0;kind=0;slot=0;index=0;a=0;b=0;c=0;cc=0;j=0;source=0;pi=0;
   // Oldest ready reduction/norm first. Unused lanes still accept dot work.
   for(off=0;off<2;off=off+1)begin
    cc=head^off;
    if(!chosen&&live[cc]&&!result_ready[cc]&&dot_done[cc])begin
     if(width[cc]>1)begin
      j=lane;
      if(red_issued[cc][lane])j=LANES+lane;
      if(j<(width[cc]>>1)&&!red_issued[cc][j])begin
       chosen=1;ctx=cc;kind=1;index=j;slot=j>=LANES?1:0;
       a=16'h3f80;b=reduce_b[cc];c=reduce_c[cc];
      end
     end else if(width[cc]==1&&!norm_issued[cc]&&lane==0)begin
      chosen=1;ctx=cc;kind=2;index=0;
      a=val[cc*PARTS];b=norms[cc];c={a[15]^b[15],15'b0};
     end
    end
   end
   pi=(data_beat&3)*LANES+lane;
   if(!chosen&&data_available&&data_beat*LANES+lane<128&&
      buffer_pending[lane]&&
      (data_first||!dot_used[data_ctx]||dot_ready[data_ctx]))begin
    chosen=1;ctx=data_ctx;kind=0;index=data_beat*LANES+lane;slot=data_beat[1:0];
    a=data_q[16*lane+:16];b=data_k[16*lane+:16];
    c=(data_first||!dot_used[data_ctx])?16'b0:dot_value[data_ctx];
   end
  end
  assign fp_issue[lane]=chosen;assign issue_ctx[lane]=ctx;assign issue_kind[lane]=kind;assign issue_index[lane]=index;assign issue_slot[lane]=slot;
  assign dot_grant[lane]=chosen&&kind==0;
  assign issue_tags[48*lane+:48]=tags[ctx];
  assign issue_nodes[8*lane+:8]=kind==0?index:kind==1?128+PARTS-width[ctx]+index:REQS-1;
  wire fv;wire [15:0] fr;wire [4:0] flags;
  bf16_fma_pipeline fma(clk,rst,chosen,a,b,c,fv,fr,flags);
  reg [2:0] pc;reg [1:0] pk[0:2],ps[0:2];reg [7:0] px[0:2],pn[0:2];
  reg [47:0] pt[0:2];integer s;
  always @(posedge clk)begin
   if(rst)begin pc<=0;for(s=0;s<3;s=s+1)begin pk[s]<=0;ps[s]<=0;px[s]<=0;pn[s]<=0;pt[s]<=0;end end
   else begin
    pc<={pc[1:0],ctx};pk[0]<=kind;ps[0]<=slot;px[0]<=index;pn[0]<=issue_nodes[8*lane+:8];pt[0]<=issue_tags[48*lane+:48];
    for(s=1;s<3;s=s+1)begin pk[s]<=pk[s-1];ps[s]<=ps[s-1];px[s]<=px[s-1];pn[s]<=pn[s-1];pt[s]<=pt[s-1];end
   end
  end
  assign fp_complete[lane]=fv;assign return_ctx[lane]=pc[2];assign return_kind[lane]=pk[2];assign return_slot[lane]=ps[2];assign return_index[lane]=px[2];
  assign complete_tags[48*lane+:48]=pt[2];assign complete_nodes[8*lane+:8]=pn[2];assign complete_bits[16*lane+:16]=fr;
 end endgenerate
 // Static per-register write ports avoid a synthesized all-to-all memory mux.
 genvar vt,vp;
 generate for(vt=0;vt<2;vt=vt+1)begin:partial_context
  wire [2*LANES-1:0] completed_now;
  genvar rr;
  for(rr=0;rr<2*LANES;rr=rr+1)begin:completed_mask
   assign completed_now[rr]=red_completed[vt][rr]||
    (fp_complete[rr%LANES]&&return_ctx[rr%LANES]==vt&&return_kind[rr%LANES]==1&&return_index[rr%LANES]==rr);
  end
  wire advance=live[vt]&&dot_done[vt]&&width[vt]>1&&
   ((completed_now&((1<<(width[vt]>>1))-1))==((1<<(width[vt]>>1))-1));
  for(vp=0;vp<PARTS;vp=vp+1)begin:partial_register
   localparam UNIT=vp%LANES,SLOT=vp/LANES;
   always @(posedge clk)begin
    if(!rst)begin
     if(fp_complete[UNIT]&&return_ctx[UNIT]==vt&&
       ((return_kind[UNIT]==0&&return_slot[UNIT]==SLOT)||
        (SLOT<2&&return_kind[UNIT]==1&&return_slot[UNIT]==SLOT)))
      val[vt*PARTS+vp]<=complete_bits[16*UNIT+:16];
     if(((LANES&(LANES-1))!=0)&&advance&&width[vt][0]&&vp==(width[vt]>>1))
      val[vt*PARTS+vp]<=val[vt*PARTS+width[vt]-1];
    end
   end
  end
 end endgenerate
 integer t,l,p,di,dc;reg [2*LANES-1:0] ri,rc;reg [LANES-1:0] valid_input;
 always @*begin
  valid_input=0;for(l=0;l<LANES;l=l+1)valid_input[l]=input_beat*LANES+l<128;
 end
 always @(posedge clk)begin
  if(rst)begin
   live<=0;closed<=0;result_ready<=0;norm_issued<=0;write_ctx<=0;head<=0;input_beat<=0;buffer_valid<=0;buffer_pending<=0;
   for(t=0;t<2;t=t+1)begin used[t]<=0;partial_ready[t]<=0;dot_issued[t]<=0;dot_completed[t]<=0;width[t]<=PARTS;red_issued[t]<=0;red_completed[t]<=0;end
  end else begin
   if(out_valid&&out_ready)begin live[head]<=0;result_ready[head]<=0;head<=~head;end
   if(buffer_valid)begin
    buffer_pending<=buffer_pending&~dot_grant;
    if(buffer_drained)buffer_valid<=0;
   end
   if(accepted)begin
    if(first_beat)begin
     live[write_ctx]<=1;closed[write_ctx]<=0;result_ready[write_ctx]<=0;norm_issued[write_ctx]<=0;tags[write_ctx]<=in_tag;norms[write_ctx]<=normalization;
     used[write_ctx]<=0;partial_ready[write_ctx]<=0;dot_issued[write_ctx]<=0;dot_completed[write_ctx]<=0;width[write_ctx]<=PARTS;red_issued[write_ctx]<=0;red_completed[write_ctx]<=0;
    end
    if(last_beat)begin closed[write_ctx]<=1;input_beat<=0;write_ctx<=~write_ctx;end else input_beat<=input_beat+1;
    // A new accepted beat cannot share units already serving buffered data.
    buffer_pending<=valid_input;
    buffer_valid<=|valid_input;
    buffer_ctx<=write_ctx;buffer_beat<=input_beat;buffer_first<=first_beat;buffer_q<=q_bits;buffer_k<=k_bits;
   end
   for(t=0;t<2;t=t+1)begin
    di=0;dc=0;ri=red_issued[t];rc=red_completed[t];
    for(l=0;l<LANES;l=l+1)begin
     if(fp_issue[l]&&issue_ctx[l]==t)begin
      if(issue_kind[l]==0)begin
       di=di+1;p=issue_slot[l]*LANES+l;used[t][p]<=1;partial_ready[t][p]<=0;
      end else if(issue_kind[l]==1)ri[issue_index[l]]=1;
      else norm_issued[t]<=1;
     end
     if(fp_complete[l]&&return_ctx[l]==t)begin
      if(return_kind[l]==0)begin
       dc=dc+1;p=return_slot[l]*LANES+l;partial_ready[t][p]<=1;
      end else if(return_kind[l]==1)begin
       p=return_slot[l]*LANES+l;rc[return_index[l]]=1;
      end else begin results[t]<=complete_bits[16*l+:16];result_ready[t]<=1;end
     end
    end
    if(di)dot_issued[t]<=(accepted&&first_beat&&write_ctx==t)?di:dot_issued[t]+di;
    if(dc)dot_completed[t]<=dot_completed[t]+dc;
    if(live[t]&&dot_done[t]&&width[t]>1)begin
     red_issued[t]<=ri;red_completed[t]<=rc;
     if((rc&((1<<(width[t]>>1))-1))==((1<<(width[t]>>1))-1))begin
      
      width[t]<=(width[t]+1)>>1;red_issued[t]<=0;red_completed[t]<=0;
     end
    end
   end
  end
 end
endmodule
