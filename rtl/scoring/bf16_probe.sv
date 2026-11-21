// Verification-only top: same BF16 unit plus boundary converters.
module bf16_probe(input wire clk,rst,in_valid,input wire [15:0] a,b,c,
 input wire [31:0] fp32_bits,input wire [15:0] fp16_bits,input wire [7:0] count,
 output wire out_valid,output wire [15:0] result,output wire [4:0] flags,
 output wire [15:0] converted32,converted16,converted_count,sort_key,output wire sort_finite);
 bf16_sort_key order_key(a,sort_key,sort_finite);
 bf16_fma_pipeline unit(clk,rst,in_valid,a,b,c,out_valid,result,flags);
 fp32_to_bf16 cv32(fp32_bits,converted32);fp16_to_bf16 cv16(fp16_bits,converted16);
 count_to_bf16 cvcount(count,converted_count);
endmodule
