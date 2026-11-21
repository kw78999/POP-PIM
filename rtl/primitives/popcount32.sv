module popcount32(input wire [31:0] bits,output wire [5:0] count);
 wire [1:0] l1[0:15];wire [2:0] l2[0:7];wire [3:0] l3[0:3];wire [4:0] l4[0:1];
 genvar i;
 generate
 for(i=0;i<16;i=i+1)begin:a assign l1[i]={1'b0,bits[2*i]}+bits[2*i+1];end
 for(i=0;i<8;i=i+1)begin:b assign l2[i]={1'b0,l1[2*i]}+l1[2*i+1];end
 for(i=0;i<4;i=i+1)begin:c assign l3[i]={1'b0,l2[2*i]}+l2[2*i+1];end
 for(i=0;i<2;i=i+1)begin:d assign l4[i]={1'b0,l3[2*i]}+l3[2*i+1];end
 endgenerate
 assign count={1'b0,l4[0]}+l4[1];
endmodule
