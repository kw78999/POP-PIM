"""Explicit word-level SRAM service and logical/macro assumption costs."""
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class SRAMConfig:
    banks:int=8
    word_bits:int=32
    read_ports:int=1
    write_ports:int=1
    read_latency:int=2
    fifo_depth:int=16
    counter_banks:int=16
    counter_forwarding:bool=False
    ping_pong:bool=False
    bitcell_um2:float=.25
    peripheral_factor:float=1.3

    def __post_init__(self):
        if self.banks<1 or self.banks&(self.banks-1) or self.word_bits not in (16,32,64,128) or min(self.read_ports,self.write_ports,self.read_latency,self.fifo_depth,self.counter_banks)<1:raise ValueError('invalid SRAM config')

    def capacity(self,n=32768,k=4096,heads=16):
        score=math.ceil(16*n*heads/8);candidate=4*k*heads;keep=math.ceil(n*heads/8)
        counter=4*16*heads;fifo=2*self.fifo_depth*heads
        pingpong=score+keep if self.ping_pong else 0
        logical=score+candidate+keep+counter+fifo+pingpong
        words=math.ceil(logical*8/self.word_bits)
        rows=1<<(max(1,math.ceil(words/self.banks))-1).bit_length()
        physical=rows*self.banks*self.word_bits//8
        return dict(score_bytes=score,candidate_bytes=candidate,keep_bytes=keep,counter_bytes=counter,fifo_bytes=fifo,
             ping_pong_extra_bytes=pingpong,logical_bytes=logical,provisioned_macro_bytes=physical,rows_per_bank=rows,
             macro_area_estimate_um2=physical*8*self.bitcell_um2*self.peripheral_factor,
             macro_cost_class='Analytical Estimate',macro_cost_assumption='bitcell area and peripheral factor unverified; no compiled SRAM or PVT',
             bitcell_um2=self.bitcell_um2,peripheral_factor=self.peripheral_factor)

class WordSRAM:
    """Per-bank 1R1W reservations. Score 16b and index32b share a word-address space."""
    def __init__(self,config):
        self.config=config;self.read_free=[[] for _ in range(config.banks)];self.write_free=[[] for _ in range(config.banks)]
    def address(self,head,token,context=32768):
        byte=(head*context+token)*2;word=byte//(self.config.word_bits//8)
        return word%self.config.banks,word//self.config.banks,(byte%(self.config.word_bits//8))*8
