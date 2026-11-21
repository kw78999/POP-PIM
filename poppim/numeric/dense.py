"""Exact integer BF16 oracle. Each dense lane has 4 recurrence chains.
Dot products FMA into +0 initial accumulators. Balanced partial reduction, BF16 norm.
"""
import numpy as np
from .bf16 import fma_bits,add_bits,mul_bits,bf16_bits_from_fp32,widen
NORM=int(bf16_bits_from_fp32(np.float32(1/np.sqrt(128))))
def dense(q,k,lanes):
 parts=4*lanes;partial=[0]*parts;trace=[]
 for d in range(128):
  index=((d//lanes)%4)*lanes+d%lanes
  partial[index]=fma_bits(int(q[d]),int(k[d]),partial[index]);trace.append(partial[index])
 while len(partial)>1:
  new=[add_bits(partial[i],partial[i+1]) for i in range(0,len(partial)-1,2)];trace.extend(new)
  if len(partial)%2:new.append(partial[-1])
  partial=new
 score=mul_bits(partial[0],NORM);trace.append(score);assert len(trace)==128+parts
 return trace
