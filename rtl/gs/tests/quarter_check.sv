module quarter_check(input wire clk,input wire [15:0] x,output wire [15:0] yq,output wire [15:0] yf);
 bf16_quarter q(.clk(clk),.x(x),.y(yq));
 bf16_fma_pipeline f(.clk(clk),.rst(1'b0),.in_valid(1'b1),.a(x),.b(16'h3E80),.c(16'h8000),.out_valid(),.result(yf),.flags());
endmodule
