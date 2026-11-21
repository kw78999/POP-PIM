#pragma once
#include <array>
#include <cstdint>
namespace phase17diag {
inline std::array<bool,64> refresh_pending{};
inline uint64_t priority_compute_stalls=0,priority_timing_stalls=0,priority_peak=0;
inline uint64_t drain_wait_bank_ticks=0;
}
namespace phase20diag {
// 0 strict (retained), 1 work-conserving REFab, 2 PIM-aware single-bank refresh.
inline int policy=0;
inline bool drain_reads=false;
inline uint64_t wc_bypass_cycles=0,refsb=0,refsb_urgent=0,refsb_quiet=0,refresh_pre=0,urgent_bank_cycles=0;
inline int64_t max_debt=0,min_debt=0,postpone=0,pullin=0,headroom=0,violations=0;
inline uint64_t nREFI=0;
inline bool round_rule=true;
inline std::array<int64_t,64> final_debt{},issued{};
}
