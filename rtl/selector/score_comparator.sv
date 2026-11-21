module score_comparator(input wire [15:0] a_key,b_key,
                         input wire [31:0] a_index,b_index,
                         output wire a_better);
    assign a_better=(a_key>b_key)||((a_key==b_key)&&(a_index<b_index));
endmodule
