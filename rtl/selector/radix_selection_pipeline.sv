// Bounded 4-pass radix prototype. Staging and candidates synthesize to flops.
// Registered array reads and bounded-width counters; initiation interval two.
// Deliberately serialized histogram updates avoid counter read-after-write hazards.
// One-hot replay address removes the shared binary read-address fanout seen in
// the initial N32 STA path. This is register storage, not an SRAM macro interface.
module radix_selection_pipeline #(parameter MAX_N=32, MAX_K=8)(
    input wire clk,rst,start,
    input wire [15:0] cfg_n,cfg_k,input wire reserve_current,
    input wire in_valid,in_keep,input wire [15:0] in_score,
    output wire in_ready,
    output wire out_valid,output wire [31:0] out_index,
    output wire out_last,input wire out_ready,
    output reg done,error,output wire busy);
    localparam CW=$clog2(MAX_N+1);
    localparam KW=$clog2(MAX_K+1);
    localparam KA=(MAX_K>1) ? $clog2(MAX_K):1;
    localparam IDLE=0,LOAD=1,CHECK=2,CLEAR=3,SCAN=4,FIND=5,
               GATHER=6,BUBBLE=7,OUTPUT=8,CURRENT=9,FINISH=10,SCAN_READ=11,GATHER_READ=12,FIND_READ=13;
    reg [3:0] state;
    reg [15:0] keys[0:MAX_N-1];
    reg keep[0:MAX_N-1];
    reg [CW-1:0] candidate_id[0:MAX_K-1];
    reg [CW-1:0] hist[0:15];
    reg [CW-1:0] hist_value;
    reg [CW-1:0] n,k,load_pos,scan_pos,eligible;
    reg [KW-1:0] candidate_count,out_pos;
    reg [15:0] prefix,prefix_mask,boundary;
    reg [CW-1:0] rank_left,ties_left;
    reg [4:0] clear_pos;
    reg [3:0] digit;
    reg [3:0] shift_bits;
    reg reserved;
    wire [15:0] in_key;
    wire input_finite;
    reg [15:0] this_key;
    reg this_keep;
    reg [MAX_N-1:0] scan_onehot;
    reg [15:0] replay_key;
    reg replay_keep;
    integer replay_i;
    always @* begin
        replay_key=0;
        replay_keep=0;
        for(replay_i=0;replay_i<MAX_N;replay_i=replay_i+1) begin
            replay_key=replay_key | (keys[replay_i] & {16{scan_onehot[replay_i]}});
            replay_keep=replay_keep | (keep[replay_i] & scan_onehot[replay_i]);
        end
    end
    wire [3:0] this_digit=(this_key>>shift_bits)&16'h000f;
    wire chosen=this_keep && ((this_key>boundary)||((this_key==boundary)&&(ties_left!=0)));
    wire [CW-1:0] current_index=n-1'b1;
    wire [CW-1:0] selected_index=candidate_id[out_pos[KA-1:0]];
    fp16_ordered_key keygen(in_score,in_key,input_finite);
    assign busy=state!=IDLE;
    assign in_ready=state==LOAD;
    assign out_valid=(state==OUTPUT)||(state==CURRENT);
    assign out_index=state==CURRENT ? {{(32-CW){1'b0}},current_index} :
                                      {{(32-CW){1'b0}},selected_index};
    assign out_last=(state==CURRENT)||((state==OUTPUT)&&(out_pos+1'b1==k)&&!reserved);
    always @(posedge clk) begin
        if(rst) begin
            state<=IDLE;done<=0;error<=0;
            n<=0;k<=0;reserved<=0;load_pos<=0;scan_pos<=0;eligible<=0;
            candidate_count<=0;out_pos<=0;clear_pos<=0;
            prefix<=0;prefix_mask<=0;boundary<=0;rank_left<=0;ties_left<=0;digit<=0;shift_bits<=0;
            scan_onehot<=0;
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
                if(clear_pos==15) begin scan_pos<=0;scan_onehot<=1;state<=SCAN_READ;end
                else clear_pos<=clear_pos+1'b1;
              end
              SCAN_READ: begin
                this_key<=replay_key;this_keep<=replay_keep;state<=SCAN;
              end
              GATHER_READ: begin
                this_key<=replay_key;this_keep<=replay_keep;state<=GATHER;
              end
              FIND_READ: begin hist_value<=hist[digit];state<=FIND;end
              SCAN: begin
                if(this_keep && ((this_key & prefix_mask)==prefix))
                    hist[this_digit]<=hist[this_digit]+1'b1;
                if(scan_pos+1'b1==n) begin digit<=15;state<=FIND_READ;end
                else begin scan_pos<=scan_pos+1'b1;scan_onehot<=scan_onehot<<1;state<=SCAN_READ;end
              end
              FIND: begin
                if(rank_left>hist_value) begin rank_left<=rank_left-hist_value;digit<=digit-1'b1;state<=FIND_READ;end
                else begin
                    prefix<=prefix | ({12'b0,digit}<<shift_bits);
                    prefix_mask<=prefix_mask | (16'h000f<<shift_bits);
                    if(shift_bits==0) begin
                        boundary<=prefix|{12'b0,digit};ties_left<=rank_left;
                        scan_pos<=0;scan_onehot<=1;state<=GATHER_READ;
                    end else begin shift_bits<=shift_bits-4;clear_pos<=0;state<=CLEAR;end
                end
              end
              GATHER: begin
                if(chosen) begin
                    candidate_id[candidate_count[KA-1:0]]<=scan_pos;
                    candidate_count<=candidate_count+1'b1;
                    if(this_key==boundary) ties_left<=ties_left-1'b1;
                end
                if(scan_pos+1'b1==n) begin out_pos<=0;state<=OUTPUT;end
                else begin scan_pos<=scan_pos+1'b1;scan_onehot<=scan_onehot<<1;state<=GATHER_READ;end
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
