// Bounded 4-pass radix prototype. Staging and candidates synthesize to flops.
// Candidate insertion ranking is serial; NOT the parallel merge-sort projection.
module topk_selector #(parameter MAX_N=32, MAX_K=8)(
    input wire clk,rst,start,
    input wire [15:0] cfg_n,cfg_k,input wire reserve_current,
    input wire in_valid,in_keep,input wire [15:0] in_score,
    output wire in_ready,
    output wire out_valid,output wire [31:0] out_index,
    output wire out_last,input wire out_ready,
    output reg done,error,output wire busy);
    localparam IDLE=0,LOAD=1,CHECK=2,CLEAR=3,SCAN=4,FIND=5,
               GATHER=6,BUBBLE=7,OUTPUT=8,CURRENT=9,FINISH=10;
    reg [3:0] state;
    reg [15:0] keys[0:MAX_N-1];
    reg keep[0:MAX_N-1];
    reg [15:0] candidate_key[0:MAX_K-1];
    reg [31:0] candidate_id[0:MAX_K-1];
    reg [15:0] hist[0:15];
    reg [15:0] n,k,load_pos,scan_pos,eligible,candidate_count,insert_pos,out_pos;
    reg [15:0] prefix,prefix_mask,boundary,rank_left,ties_left;
    reg [4:0] clear_pos;
    reg [3:0] digit;
    reg [3:0] shift_bits;
    reg reserved;
    wire [15:0] in_key;
    wire input_finite;
    wire better;
    wire [15:0] this_key=keys[scan_pos];
    wire this_keep=keep[scan_pos];
    wire [3:0] this_digit=(this_key>>shift_bits)&16'h000f;
    wire chosen=this_keep && ((this_key>boundary)||((this_key==boundary)&&(ties_left!=0)));
    fp16_ordered_key keygen(in_score,in_key,input_finite);
    score_comparator cmp(candidate_key[insert_pos],candidate_key[insert_pos-1'b1],
                         candidate_id[insert_pos],candidate_id[insert_pos-1'b1],better);
    assign busy=state!=IDLE;
    assign in_ready=state==LOAD;
    assign out_valid=(state==OUTPUT)||(state==CURRENT);
    assign out_index=state==CURRENT ? {16'b0,n-16'd1} : candidate_id[out_pos];
    assign out_last=(state==CURRENT)||((state==OUTPUT)&&(out_pos+1'b1==k)&&!reserved);
    always @(posedge clk) begin
        if(rst) begin
            state<=IDLE;done<=0;error<=0;
            n<=0;k<=0;reserved<=0;load_pos<=0;scan_pos<=0;eligible<=0;
            candidate_count<=0;insert_pos<=0;out_pos<=0;clear_pos<=0;
            prefix<=0;prefix_mask<=0;boundary<=0;rank_left<=0;ties_left<=0;digit<=0;shift_bits<=0;
        end else begin
            done<=0;
            case(state)
              IDLE: if(start) begin
                error<=0; n<=cfg_n;reserved<=reserve_current;
                k<=cfg_k-(reserve_current ? 16'd1:16'd0);
                load_pos<=0;eligible<=0;candidate_count<=0;out_pos<=0;
                prefix<=0;prefix_mask<=0;shift_bits<=12;
                if(cfg_n>MAX_N || cfg_k>MAX_K || cfg_k>cfg_n ||
                   (reserve_current && (cfg_n==0 || cfg_k==0))) begin
                    error<=1;state<=FINISH;
                end else if(cfg_n==0) state<=FINISH;
                else state<=LOAD;
              end
              LOAD: if(in_valid && in_ready) begin
                keys[load_pos]<=in_key;
                keep[load_pos]<=in_keep && !(reserved && load_pos==n-1'b1);
                if(in_keep && !(reserved && load_pos==n-1'b1)) eligible<=eligible+1'b1;
                if(!input_finite || (reserved && load_pos==n-1'b1 && !in_keep)) error<=1;
                if(load_pos+1'b1==n) state<=CHECK;
                else load_pos<=load_pos+1'b1;
              end
              CHECK: begin
                if(error || eligible<k) begin error<=1;state<=FINISH;end
                else if(k==0) state<=reserved ? CURRENT:FINISH;
                else begin rank_left<=k;clear_pos<=0;state<=CLEAR;end
              end
              CLEAR: begin
                hist[clear_pos]<=0;
                if(clear_pos==15) begin scan_pos<=0;state<=SCAN;end
                else clear_pos<=clear_pos+1'b1;
              end
              SCAN: begin
                if(this_keep && ((this_key & prefix_mask)==prefix))
                    hist[this_digit]<=hist[this_digit]+1'b1;
                if(scan_pos+1'b1==n) begin digit<=15;state<=FIND;end
                else scan_pos<=scan_pos+1'b1;
              end
              FIND: begin
                if(rank_left>hist[digit]) begin rank_left<=rank_left-hist[digit];digit<=digit-1'b1;end
                else begin
                    prefix<=prefix | ({12'b0,digit}<<shift_bits);
                    prefix_mask<=prefix_mask | (16'h000f<<shift_bits);
                    if(shift_bits==0) begin
                        boundary<=prefix|{12'b0,digit};ties_left<=rank_left;
                        scan_pos<=0;state<=GATHER;
                    end else begin shift_bits<=shift_bits-4;clear_pos<=0;state<=CLEAR;end
                end
              end
              GATHER: begin
                if(chosen) begin
                    candidate_key[candidate_count]<=this_key;
                    candidate_id[candidate_count]<={16'b0,scan_pos};
                    insert_pos<=candidate_count;candidate_count<=candidate_count+1'b1;
                    if(this_key==boundary) ties_left<=ties_left-1'b1;
                    state<=BUBBLE;
                end else if(scan_pos+1'b1==n) begin out_pos<=0;state<=OUTPUT;end
                else scan_pos<=scan_pos+1'b1;
              end
              BUBBLE: begin
                if(insert_pos!=0 && better) begin
                    candidate_key[insert_pos]<=candidate_key[insert_pos-1'b1];
                    candidate_key[insert_pos-1'b1]<=candidate_key[insert_pos];
                    candidate_id[insert_pos]<=candidate_id[insert_pos-1'b1];
                    candidate_id[insert_pos-1'b1]<=candidate_id[insert_pos];
                    insert_pos<=insert_pos-1'b1;
                end else if(scan_pos+1'b1==n) begin out_pos<=0;state<=OUTPUT;end
                else begin scan_pos<=scan_pos+1'b1;state<=GATHER;end
              end
              OUTPUT: if(out_ready) begin
                if(out_pos+1'b1==k) state<=reserved ? CURRENT:FINISH;
                else out_pos<=out_pos+1'b1;
              end
              CURRENT: if(out_ready) state<=FINISH;
              FINISH: begin done<=1;state<=IDLE;end
              default: state<=IDLE;
            endcase
        end
    end
endmodule
