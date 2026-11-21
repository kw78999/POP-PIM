// Value-preserving order encoding, with +0/-0 canonicalized to the same key.
module fp16_ordered_key(input wire [15:0] score,
                        output wire [15:0] key,output wire finite);
    wire [15:0] canonical = (score[14:0] == 0) ? 16'h0000 : score;
    assign key = canonical[15] ? ~canonical : (canonical ^ 16'h8000);
    assign finite = score[14:10] != 5'b11111;
endmodule
