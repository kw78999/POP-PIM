// BF16 order key; zero tie canonicalization; nonfinite values rejected.
// This is a format adapter, not an additional arithmetic unit.
module bf16_sort_key(input wire [15:0] score_bits,output wire [15:0] key,output wire finite);
 wire [15:0] canonical=(score_bits[14:0]==0)?16'b0:score_bits;
 assign finite=score_bits[14:7]!=8'hff;
 assign key=canonical[15]?~canonical:(canonical^16'h8000);
endmodule
