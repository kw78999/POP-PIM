#pragma once
#ifdef USE_GQA2
#include "GQA2.h"
using GQA=GQA2;
constexpr unsigned META_BUFFERS=2;
#else
#include "GQA1.h"
using GQA=GQA1;
constexpr unsigned META_BUFFERS=1;
#endif
#include <array>
#include <cstdint>
#include <stdexcept>
#include <deque>
#include <map>
#include <memory>
struct Response13 {unsigned slot,beat,tag,kv,byte;};
inline uint8_t metadata13(unsigned bank,unsigned kv,unsigned local_byte){return uint8_t((kv*512*40+local_byte)*13+bank*3);}
inline uint64_t metadata_word13(unsigned bank,unsigned kv,unsigned byte,unsigned count){uint64_t value=0;for(unsigned i=0;i<count;i++)value|=uint64_t(metadata13(bank,kv,byte+i))<<(8*i);return value;}
inline void fill13(GQA&d,unsigned bank,const Response13&r){d.rsp_slot=r.slot;d.rsp_beat=r.beat;d.rsp_tag=r.tag;for(int i=0;i<8;i++)d.rsp_data[i]=uint32_t(metadata_word13(bank,r.kv,r.byte+i*4,4));}
inline void reset13(GQA&d,unsigned n,unsigned epoch=100){d.clk=0;d.rst=1;d.enable=0;d.prefetch=1;d.n_local=n;d.epoch=epoch;d.req_ready=0;d.rsp_valid=0;d.operand_ready=0;d.score_in_valid=0;d.read_head=0;d.score_ready=0;d.eval();d.clk=1;d.eval();d.clk=0;d.eval();d.rst=0;d.enable=1;d.eval();}
inline void edge13(GQA&d){d.clk=1;d.eval();d.clk=0;d.eval();}
inline void verify_operand13(const GQA&d,unsigned bank){unsigned kv=d.operand_head/4,byte=d.operand_local*40+d.operand_group*20;if(d.k_sign!=metadata_word13(bank,kv,byte,8)||d.k_mag!=metadata_word13(bank,kv,byte+8,8)||d.k_low!=metadata_word13(bank,kv,byte+16,2)||d.k_delta!=metadata_word13(bank,kv,byte+18,2))throw std::runtime_error("RTL metadata payload/head corruption");}
