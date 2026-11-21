// Format converters only; no binary32 arithmetic.
module fp32_to_bf16(input wire [31:0] a,output wire [15:0] b);
 wire [32:0] ar;wire [16:0] br;wire [4:0] flags;
 fNToRecFN #(8,24) enc(a,ar);
 recFNToRecFN #(8,24,8,8) cvt(1'b1,ar,3'b000,br,flags);
 recFNToFN #(8,8) dec(br,b);
endmodule
module fp16_to_bf16(input wire [15:0] a,output wire [15:0] b);
 wire [16:0] ar,br;wire [4:0] flags;
 fNToRecFN #(5,11) enc(a,ar);
 recFNToRecFN #(5,11,8,8) cvt(1'b1,ar,3'b000,br,flags);
 recFNToFN #(8,8) dec(br,b);
endmodule
module count_to_bf16(input wire [7:0] a,output wire [15:0] b);
 wire [16:0] br;wire [4:0] flags;
 iNToRecFN #(8,8,8) cvt(1'b1,1'b1,a,3'b000,br,flags);
 recFNToFN #(8,8) dec(br,b);
endmodule
