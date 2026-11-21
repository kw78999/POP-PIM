// Phase 7 only: one shared BF16 unit, exactly 1 or 2 score contexts, two groups.
// Primary boundary Q FP32 / K FP16 -> BF16. RAW_BF16 selects BF16 metadata.
module score7_top #(parameter CONTEXTS=1,ARCH_B=0,RAW_BF16=0,
 parameter CW=CONTEXTS<=1?1:$clog2(CONTEXTS),GN=ARCH_B?11:9)(
 input wire clk,rst,in_valid,first_group,last_group,input wire [47:0] in_tag,
 input wire [63:0] q_sign,q_mag,k_sign,k_mag,dim_mask,
 input wire [31:0] q_low,q_delta,normalization,input wire [15:0] k_low,k_delta,
 output wire in_ready,out_valid,output wire [47:0] out_tag,
 output wire [15:0] score_bits,input wire out_ready,
 output wire count_valid,output wire [47:0] count_tag,output wire count_rep,
 output wire [31:0] count_values,
 output wire fp_issue_valid,output wire [47:0] fp_issue_tag,output wire [4:0] fp_issue_node,
 output wire fp_complete_valid,output wire [47:0] fp_complete_tag,output wire [4:0] fp_complete_node,
 output wire [15:0] fp_complete_bits,output wire empty,
 output wire dependency_stall,context_stall,output_stall);
 reg [CONTEXTS-1:0] live,closed,total_issued,total_done,norm_issued,result_ready;
 reg [47:0] tags[0:CONTEXTS-1];reg [15:0] norms[0:CONTEXTS-1],totals[0:CONTEXTS-1],results[0:CONTEXTS-1];
 reg [CW-1:0] write_ctx,head,count_ctx;reg write_rep,count_slot;
 wire [15:0] ql,dq,kl,dk,norm;
 generate if(RAW_BF16)begin:raw
  assign ql=q_low[15:0];assign dq=q_delta[15:0];assign kl=k_low;assign dk=k_delta;assign norm=normalization[15:0];
 end else begin:convert
  fp32_to_bf16 qlc(q_low,ql),dqc(q_delta,dq),nc(normalization,norm);
  fp16_to_bf16 klc(k_low,kl),dkc(k_delta,dk);
 end endgenerate
 wire cnt_ready,cnt_done;wire signed [7:0] cs,cq,ck,cqk;
 assign in_ready=cnt_ready&&(first_group?!live[write_ctx]:live[write_ctx]&&!closed[write_ctx]&&write_rep);
 wire accepted=in_valid&&in_ready;
 bitwise_count #(4) counts(clk,rst,accepted,q_sign,q_mag,k_sign,k_mag,dim_mask,cnt_ready,cnt_done,cs,cq,ck,cqk);
 assign count_valid=cnt_done;assign count_tag=tags[count_ctx];assign count_rep=count_slot;
 assign count_values={cqk,ck,cq,cs};
 assign out_valid=live[head]&&result_ready[head];assign out_tag=tags[head];assign score_bits=results[head];
 assign empty=!(|live);assign context_stall=in_valid&&cnt_ready&&first_group&&live[write_ctx];
 assign output_stall=out_valid&&!out_ready;
 wire [CONTEXTS*2-1:0] group_ready,group_done;
 wire [3:0] group_op[0:CONTEXTS*2-1];wire [15:0] ga[0:CONTEXTS*2-1],gb[0:CONTEXTS*2-1],gc[0:CONTEXTS*2-1],gs[0:CONTEXTS*2-1];
 reg chosen;reg [CW-1:0] issue_ctx;reg [4:0] issue_node;reg [15:0] fa,fb,fc;
 reg [CONTEXTS-1:0] grant;reg [CONTEXTS*2-1:0] group_grant;
 wire fv;wire [15:0] fr;wire [4:0] flags;
 reg [CW-1:0] pipe_ctx[0:2];reg [4:0] pipe_node[0:2];
 genvar g;
 generate for(g=0;g<CONTEXTS*2;g=g+1)begin:groups
  score7_group #(.ARCH_B(ARCH_B)) dag(.clk(clk),.rst(rst),
   .load(accepted&&write_ctx==g/2&&write_rep==g%2),.in_ql(ql),.in_dq(dq),.in_kl(kl),.in_dk(dk),
   .count_load(cnt_done&&count_ctx==g/2&&count_slot==g%2),.count_values(count_values),
   .issue(group_grant[g]),.wb(fv&&pipe_ctx[2]==g/2&&pipe_node[2]>=GN*(g%2)&&pipe_node[2]<GN*(g%2+1)),
   .wb_op(pipe_node[2]-GN*(g%2)),.wb_bits(fr),.ready(group_ready[g]),.op(group_op[g]),
   .a(ga[g]),.b(gb[g]),.c(gc[g]),.done(group_done[g]),.group_score(gs[g]));
 end endgenerate
 wire [CONTEXTS-1:0] local_ready;
 wire [15:0] la[0:CONTEXTS-1],lb[0:CONTEXTS-1],lc[0:CONTEXTS-1];
 wire [4:0] ln[0:CONTEXTS-1];
 generate for(g=0;g<CONTEXTS;g=g+1)begin:operand
  wire final_add=group_done[2*g]&&group_done[2*g+1]&&closed[g]&&!total_issued[g];
  wire final_norm=total_done[g]&&!norm_issued[g];
  wire choose0=group_ready[2*g];
  assign local_ready[g]=live[g]&&!result_ready[g]&&(final_add||final_norm||group_ready[2*g]||group_ready[2*g+1]);
  assign ln[g]=final_norm?2*GN+1:final_add?2*GN:choose0?group_op[2*g]:GN+group_op[2*g+1];
  assign la[g]=final_norm?totals[g]:final_add?16'h3f80:choose0?ga[2*g]:ga[2*g+1];
  assign lb[g]=final_norm?norms[g]:final_add?gs[2*g]:choose0?gb[2*g]:gb[2*g+1];
  assign lc[g]=final_norm?{totals[g][15]^norms[g][15],15'b0}:final_add?gs[2*g+1]:choose0?gc[2*g]:gc[2*g+1];
 end endgenerate
 integer offset,candidate,s;
 always @*begin
  chosen=0;issue_ctx=0;issue_node=0;fa=0;fb=0;fc=0;grant=0;group_grant=0;candidate=0;
  for(offset=0;offset<CONTEXTS;offset=offset+1)begin
   candidate=head+offset;if(candidate>=CONTEXTS)candidate=candidate-CONTEXTS;
   if(!chosen&&local_ready[candidate])begin chosen=1;issue_ctx=candidate;grant[candidate]=1;end
  end
  for(s=0;s<CONTEXTS;s=s+1)begin
   fa=fa|(la[s]&{16{grant[s]}});fb=fb|(lb[s]&{16{grant[s]}});fc=fc|(lc[s]&{16{grant[s]}});
   issue_node=issue_node|(ln[s]&{5{grant[s]}});
   group_grant[2*s]=grant[s]&&ln[s]<GN;group_grant[2*s+1]=grant[s]&&ln[s]>=GN&&ln[s]<2*GN;
  end
 end
 bf16_fma_pipeline fma(clk,rst,chosen,fa,fb,fc,fv,fr,flags);
 assign fp_issue_valid=chosen;assign fp_issue_tag=tags[issue_ctx];assign fp_issue_node=issue_node;
 assign fp_complete_valid=fv;assign fp_complete_tag=tags[pipe_ctx[2]];assign fp_complete_node=pipe_node[2];assign fp_complete_bits=fr;
 assign dependency_stall=(|live)&&!chosen&&!out_valid;
 integer i;
 always @(posedge clk)begin
  if(rst)begin
   live<=0;closed<=0;total_issued<=0;total_done<=0;norm_issued<=0;result_ready<=0;
   write_ctx<=0;head<=0;write_rep<=0;count_ctx<=0;count_slot<=0;
   for(i=0;i<3;i=i+1)begin pipe_ctx[i]<=0;pipe_node[i]<=0;end
  end else begin
   pipe_ctx[0]<=issue_ctx;pipe_node[0]<=issue_node;
   for(i=1;i<3;i=i+1)begin pipe_ctx[i]<=pipe_ctx[i-1];pipe_node[i]<=pipe_node[i-1];end
   if(out_valid&&out_ready)begin live[head]<=0;result_ready[head]<=0;if(head==CONTEXTS-1)head<=0;else head<=head+1;end
   if(accepted)begin
    count_ctx<=write_ctx;count_slot<=write_rep;
    if(first_group)begin
     live[write_ctx]<=1;closed[write_ctx]<=0;tags[write_ctx]<=in_tag;norms[write_ctx]<=norm;
     total_issued[write_ctx]<=0;total_done[write_ctx]<=0;norm_issued[write_ctx]<=0;result_ready[write_ctx]<=0;
    end
    if(last_group)begin closed[write_ctx]<=1;write_rep<=0;if(write_ctx==CONTEXTS-1)write_ctx<=0;else write_ctx<=write_ctx+1;end
    else write_rep<=1;
   end
   if(chosen)begin
    if(issue_node==2*GN)total_issued[issue_ctx]<=1;
    if(issue_node==2*GN+1)norm_issued[issue_ctx]<=1;
   end
   if(fv)begin
    if(pipe_node[2]==2*GN)begin totals[pipe_ctx[2]]<=fr;total_done[pipe_ctx[2]]<=1;end
    if(pipe_node[2]==2*GN+1)begin results[pipe_ctx[2]]<=fr;result_ready[pipe_ctx[2]]<=1;end
   end
  end
 end
endmodule
