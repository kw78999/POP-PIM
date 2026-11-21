#include "Vquarter_check.h"
#include <cstdio>
int main(){Vquarter_check m;long bad=0;for(int x=0;x<65536;x++){m.x=x;for(int c=0;c<5;c++){m.clk=0;m.eval();m.clk=1;m.eval();}
 if(m.yq!=m.yf){if(bad<10)printf("x=%04x quarter=%04x fma=%04x\n",x,m.yq,m.yf);bad++;}}
 printf("{\"test\":\"bf16_quarter_exhaustive\",\"inputs\":65536,\"mismatch\":%ld}\n",bad);return bad!=0;}
