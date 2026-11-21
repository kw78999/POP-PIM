// Single-update/cycle counter bank; no speculative RMW collision forwarding.
module radix_histogram #(parameter COUNTER_WIDTH=16)(
    input wire clk,rst,clear,valid,input wire [3:0] digit,read_digit,
    output wire [COUNTER_WIDTH-1:0] read_count);
    reg [COUNTER_WIDTH-1:0] count[0:15];
    integer i;
    always @(posedge clk) begin
        if (rst||clear) for(i=0;i<16;i=i+1) count[i]<=0;
        else if(valid) count[digit]<=count[digit]+1'b1;
    end
    assign read_count=count[read_digit];
endmodule
