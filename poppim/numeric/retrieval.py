"""Independent exact-integer score DAG, unrelated to HardFloat implementation.
All inter-operation values are BF16 bits, including products and normalization.
"""
import numpy as np
from .bf16 import mul_bits,add_bits,fma_bits,bf16_bits_from_fp32,widen
GROUP=np.dtype([(n,'<u8') for n in ['qs','qm','ks','km','mask']]+[(n,'<u4') for n in ['ql','dq','norm']]+[(n,'<u2') for n in ['kl','dk']])
TEST_GROUP=np.dtype(GROUP.descr+[('counts','i1',(4,)),('expected','<u2',(11,))])
TEST_FRAME=np.dtype([('group',TEST_GROUP,(2,)),('final','<u2',(2,))])
META_FRAME=np.dtype([('group',GROUP,(2,)),('fp32','<u4'),('fp16','<u2')])
def counts(g):
 mask=int(g['mask']);m=~(int(g['qs'])^int(g['ks']))&mask
 return [2*(m&s).bit_count()-s.bit_count() for s in [mask,int(g['qm'])&mask,int(g['km'])&mask,int(g['qm'])&int(g['km'])&mask]]
def coefficient_bits(g):
 ql,dq,norm=[int(bf16_bits_from_fp32(np.array(g[x],dtype=np.uint32).view(np.float32))) for x in ['ql','dq','norm']]
 kl,dk=[int(bf16_bits_from_fp32(np.array(g[x],dtype=np.uint16).view(np.float16).astype(np.float32))) for x in ['kl','dk']]
 return ql,dq,kl,dk,norm
def group_ops(g,arch,raw=False):
 if raw:ql,dq,kl,dk,norm=map(lambda x:int(g[x]),['ql','dq','kl','dk','norm'])
 else:ql,dq,kl,dk,norm=coefficient_bits(g)
 cs,cq,ck,cqk=map(int,bf16_bits_from_fp32(counts(g)))
 p=[mul_bits(dq,dk),mul_bits(ql,dk),mul_bits(dq,kl),mul_bits(ql,kl)]
 if arch=='A':
  a0=mul_bits(p[0],cqk);b0=mul_bits(p[2],cq)
  a1=fma_bits(p[1],ck,a0);b1=fma_bits(p[3],cs,b0)
  return p+[a0,b0,a1,b1,add_bits(a1,b1)]
 if arch=='B':
  t=[mul_bits(a,b) for a,b in zip(p,[cqk,ck,cq,cs])]
  r=[add_bits(t[0],t[1]),add_bits(t[2],t[3])]
  return p+t+r+[add_bits(*r)]
 v=mul_bits(p[0],cqk)
 for a,b in zip(p[1:],[ck,cq,cs]):v=fma_bits(a,b,v)
 return v
def score(g,arch,raw=False):
 norm=int(g[0]['norm']) if raw else coefficient_bits(g[0])[-1]
 if arch=='S':
  total=0
  for x in g:total=add_bits(total,group_ops(x,'S',raw))
  return mul_bits(total,norm)
 a,b=[group_ops(x,arch,raw)[-1] for x in g];total=add_bits(a,b)
 return mul_bits(total,norm)
def fp32_score(g):
 from poppim.scoring.reference import fma32
 total=np.float32(0)
 for x in g:
  ql,dq=[np.array(x[n],dtype=np.uint32).view(np.float32) for n in ['ql','dq']]
  kl,dk=[np.array(x[n],dtype=np.uint16).view(np.float16).astype(np.float32) for n in ['kl','dk']]
  cs,cq,ck,cqk=map(np.float32,counts(x));p=[np.float32(dq*dk),np.float32(ql*dk),np.float32(dq*kl),np.float32(ql*kl)]
  v=np.float32(p[0]*cqk)
  for a,b in zip(p[1:],[ck,cq,cs]):v=fma32(a,b,v)
  total=np.float32(total+v)
 return np.float32(total*np.array(g[0]['norm'],dtype=np.uint32).view(np.float32))
def read_metadata(path):
 with open(path,'rb') as f:
  h,n=np.frombuffer(f.read(8),dtype='<u4');a=np.frombuffer(f.read(),dtype=META_FRAME).copy().reshape(int(h),int(n))
 return a
