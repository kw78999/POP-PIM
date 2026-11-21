"""Independent cycle model of the preserved Phase 3 radix_selection_pipeline.

Signals are sampled before an edge; start's edge is outside step(). Every state,
including CLEAR/CHECK/FINISH and backpressure, is charged. Not a parallel engine.
"""
from collections import Counter
import numpy as np

STATES = ["IDLE","LOAD","CHECK","CLEAR","SCAN","FIND","GATHER","UNUSED",
          "OUTPUT","CURRENT","FINISH","SCAN_READ","GATHER_READ","FIND_READ"]


def key16(bits):
    bits=int(bits)
    canonical=0 if bits & 0x7fff == 0 else bits
    return ((~canonical)&65535) if canonical&32768 else canonical^32768


class RTLSelector:
    def __init__(self,n,k,reserve=False,max_n=None,max_k=None):
        self.n=n; self.budget=k; self.k=k-int(reserve); self.reserve=reserve
        self.error=(n>(max_n if max_n is not None else n) or
                    k>(max_k if max_k is not None else k) or k>n or
                    (reserve and (n==0 or k==0)))
        self.state=10 if self.error or n==0 else 1
        self.done=False
        self.keys=[0]*n; self.keep=[False]*n; self.hist=[0]*16; self.ids=[]
        self.load=self.scan=self.eligible=self.out=self.clear=0
        self.prefix=self.mask=self.boundary=self.rank=self.ties=0
        self.shift=12; self.digit=0; self.this_key=0; self.this_keep=False; self.hist_value=0
        self.stage_counts=Counter(); self.cycles=0; self.counter_updates=0

    def signals(self):
        valid=self.state in (8,9)
        index=self.n-1 if self.state==9 else self.ids[self.out] if self.state==8 else 0
        last=self.state==9 or (self.state==8 and self.out+1==self.k and not self.reserve)
        return dict(state=self.state,in_ready=self.state==1,out_valid=valid,
                    out_index=index,out_last=last,done=self.done,error=bool(self.error))

    def step(self,in_valid=False,in_bits=0,in_keep=True,out_ready=True):
        before=self.signals()
        if self.done:return before
        state=self.state;self.stage_counts[STATES[state]]+=1;self.cycles+=1
        if state==1 and in_valid:
            self.keys[self.load]=key16(in_bits)
            self.keep[self.load]=bool(in_keep and not(self.reserve and self.load==self.n-1))
            self.eligible+=self.keep[self.load]
            self.error |= (int(in_bits)&0x7c00)==0x7c00 or (self.reserve and self.load==self.n-1 and not in_keep)
            self.load+=1
            if self.load==self.n:self.state=2
        elif state==2:
            if self.error or self.eligible<self.k:self.error=True;self.state=10
            elif self.k==0:self.state=9 if self.reserve else 10
            else:self.rank=self.k;self.clear=0;self.state=3
        elif state==3:
            self.hist[self.clear]=0;self.clear+=1
            if self.clear==16:self.scan=0;self.state=11
        elif state in (11,12):
            self.this_key=self.keys[self.scan];self.this_keep=self.keep[self.scan]
            self.state=4 if state==11 else 6
        elif state==4:
            if self.this_keep and self.this_key&self.mask==self.prefix:
                self.hist[(self.this_key>>self.shift)&15]+=1;self.counter_updates+=1
            self.scan+=1
            if self.scan==self.n:self.digit=15;self.state=13
            else:self.state=11
        elif state==13:self.hist_value=self.hist[self.digit];self.state=5
        elif state==5:
            if self.rank>self.hist_value:
                self.rank-=self.hist_value;self.digit-=1;self.state=13
            else:
                self.prefix|=self.digit<<self.shift;self.mask|=15<<self.shift
                if self.shift==0:
                    self.boundary=self.prefix;self.ties=self.rank;self.scan=0;self.state=12
                else:self.shift-=4;self.clear=0;self.state=3
        elif state==6:
            if self.this_keep and (self.this_key>self.boundary or
                                   (self.this_key==self.boundary and self.ties>0)):
                self.ids.append(self.scan)
                if self.this_key==self.boundary:self.ties-=1
            self.scan+=1
            self.state=8 if self.scan==self.n else 12
        elif state==8 and out_ready:
            self.out+=1
            if self.out==self.k:self.state=9 if self.reserve else 10
        elif state==9 and out_ready:self.state=10
        elif state==10:self.done=True;self.state=0
        return before


def rtl_profile(scores,budget,valid=None,reserve=False):
    """Closed event counts for this specific FSM, cross-checked against step().

    Larger N keeps II=2; clock/SRAM physical implementation is not established.
    """
    from poppim.topk.radix_topk import radix_topk
    s=np.asarray(scores)
    ids,stats=radix_topk(s,budget,valid,reserve,output_mode="position",return_stats=True)
    n=len(s);k=budget-int(reserve)
    stages=Counter(LOAD=n,CHECK=int(n>0),FINISH=1)
    if k:
        boundary=stats["boundary_key"]
        finds=sum(16-((boundary>>shift)&15) for shift in (12,8,4,0))
        stages.update(CLEAR=64,SCAN_READ=4*n,SCAN=4*n,FIND_READ=finds,FIND=finds,
                      GATHER_READ=n,GATHER=n,OUTPUT=k)
    if reserve:stages["CURRENT"]=1
    return dict(cycles=sum(stages.values()),restart_interval_cycles=sum(stages.values())+1,
                stages=dict(stages),selected_indices=ids,counter_updates=stats["histogram_updates"],
                source="Analytical Estimate: RTL-FSM event counts; large-N timing unverified")
