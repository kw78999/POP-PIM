"""CPU indexed split attention with pre-applied RoPE and original token IDs.
FP32 operations follow the mathematical split/merge interface, not Triton trees.
"""
import numpy as np


def rope(x,positions):
    x=np.asarray(x,dtype=np.float64);d=x.shape[-1]
    angle=np.asarray(positions)[...,None]/(10000**(np.arange(d//2)*2/d))
    c=np.concatenate([np.cos(angle)]*2,axis=-1);s=np.concatenate([np.sin(angle)]*2,axis=-1)
    rotate=np.concatenate([-x[...,d//2:],x[...,:d//2]],axis=-1)
    return x*c+rotate*s


def indexed_attention(q,keys,values,indices,valid=None,acc_dtype=np.float32,output_dtype=np.float32,split_tokens=256,bias=None):
    """K/V include current token in final row. Slot mask follows its index slot.
    Optional token-indexed bias tests identity-dependent semantics; original
    PopKV decode uses RoPE before this interface and no such additive bias.
    """
    ids=np.asarray(indices,dtype=np.int64)
    keep=np.ones(len(ids),bool) if valid is None else np.asarray(valid,bool)
    if keep.shape!=ids.shape:raise ValueError('slot mask mismatch')
    if not keep.any():raise ValueError('no eligible attention tokens')
    q=np.asarray(q,dtype=acc_dtype); k=np.asarray(keys,dtype=acc_dtype);v=np.asarray(values,dtype=acc_dtype)
    partial=[]
    for begin in range(0,len(ids),split_tokens):
        use=ids[begin:begin+split_tokens][keep[begin:begin+split_tokens]]
        if not len(use):continue
        logits=np.sum(k[use]*q,axis=-1,dtype=acc_dtype)*acc_dtype(1/np.sqrt(q.shape[-1]))
        if bias is not None:logits=logits+np.asarray(bias,dtype=acc_dtype)[use]
        m=logits.max();p=np.exp(logits-m).astype(acc_dtype)
        partial.append((m,p.sum(dtype=acc_dtype),np.sum(p[:,None]*v[use],axis=0,dtype=acc_dtype)))
    m=max(t[0] for t in partial);den=acc_dtype(0);num=np.zeros(q.shape[-1],dtype=acc_dtype)
    for pm,ps,pv in partial:
        weight=acc_dtype(np.exp(pm-m));den+=ps*weight;num+=pv*weight
    return (num/den).astype(output_dtype)
