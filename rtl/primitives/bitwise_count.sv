// Sixteen population-count operations for one <=64-dimension representative.
// POP_LANES is the number of physical 32-bit population-count circuits.
module bitwise_count #(parameter POP_LANES=4)(
 input wire clk,rst,start,input wire [63:0] qs,qm,ks,km,mask,
 output wire ready,output reg done,
 output reg signed [7:0] cs,cq,ck,cqk);
 reg busy;reg [4:0] base;reg [31:0] words[0:15];reg [5:0] counts[0:15];
 wire [63:0] agree=~(qs^ks)&mask,mq=qm&mask,mk=km&mask,both=mq&mk;
 wire [5:0] lane[0:POP_LANES-1];genvar p;
 generate for(p=0;p<POP_LANES;p=p+1)begin:pc popcount32 pc(words[base+p],lane[p]);end endgenerate
 integer i,w;assign ready=!busy;
 reg reduce_pending;
 always @(posedge clk)begin
  if(rst)begin busy<=0;done<=0;reduce_pending<=0;base<=0;end
  else begin
   done<=0;
   if(start&&!busy)begin
    busy<=1;base<=0;reduce_pending<=0;
    for(w=0;w<2;w=w+1)begin
     words[w*8]<=agree[w*32+:32];words[w*8+1]<=mask[w*32+:32];
     words[w*8+2]<=(agree&mq)>>(w*32);words[w*8+3]<=mq>>(w*32);
     words[w*8+4]<=(agree&mk)>>(w*32);words[w*8+5]<=mk>>(w*32);
     words[w*8+6]<=(agree&both)>>(w*32);words[w*8+7]<=both>>(w*32);
    end
   end else if(busy)begin
    if(reduce_pending)begin
     cs<=((counts[0]+{2'b0,counts[8]})<<1)-(counts[1]+{2'b0,counts[9]});
     cq<=((counts[2]+{2'b0,counts[10]})<<1)-(counts[3]+{2'b0,counts[11]});
     ck<=((counts[4]+{2'b0,counts[12]})<<1)-(counts[5]+{2'b0,counts[13]});
     cqk<=((counts[6]+{2'b0,counts[14]})<<1)-(counts[7]+{2'b0,counts[15]});
     done<=1;busy<=0;reduce_pending<=0;
    end else begin
     for(i=0;i<POP_LANES;i=i+1)counts[base+i]<=lane[i];
     if(base+POP_LANES>=16)reduce_pending<=1;else base<=base+POP_LANES;
    end
   end
  end
 end
endmodule
