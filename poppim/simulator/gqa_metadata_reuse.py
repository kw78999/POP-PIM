"""GQA mapping/capacity oracle for the integrated per-bank full-KV metadata cache."""
from dataclasses import dataclass
import math
from poppim.scoring.reference import gqa_mapping

@dataclass(frozen=True)
class MetadataReuse:
    query_heads:int=4
    kv_heads:int=2
    bytes_per_token:int=40
    tile_tokens:int=4
    def mapping(self):return gqa_mapping(self.query_heads,self.kv_heads)
    def traffic(self,context,reuse=False):return context*(self.kv_heads if reuse else self.query_heads)*self.bytes_per_token
    def capacity(self,context,banks=64,word_bits=32):
        if context%banks or context//banks%self.tile_tokens:raise ValueError('four-token tiles required')
        words=math.ceil((context//banks)*self.bytes_per_token/(word_bits/8))
        rows=1<<(words-1).bit_length()
        return dict(logical_cache_bytes=context*self.bytes_per_token,
          provisioned_macro_bytes=rows*banks*(word_bits//8),valid_bits=context//self.tile_tokens,
          bank_tag_bits=banks*max(1,(self.kv_heads-1).bit_length()),
          cache_read_cycles=math.ceil(self.tile_tokens*self.bytes_per_token/(word_bits/8))+2,
          score_computations=context*self.query_heads,
          scope='one complete KV-head cache per bank, retained until all associated Q heads consume it')
