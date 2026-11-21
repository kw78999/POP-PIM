// Independent scalar CPU cross-check of the source-defined arithmetic.
// No CUDA, no fast math. This is NOT a measurement of the CUDA kernel.
#include <cmath>
#include <cstdint>

extern "C" void scalar_scores(const uint32_t* qs, const uint32_t* qm,
    const float* ql, const float* dq, const uint32_t* ks, const uint32_t* km,
    const float* kl, const float* dk, int dim, int group, int tokens, float* out) {
    const int words = (dim+31)/32, reps = (dim+group-1)/group;
    for (int t=0; t<tokens; ++t) {
        float total = 0.f;
        for (int r=0; r<reps; ++r) {
            int cs=0,cq=0,ck=0,cqk=0;
            // Accumulate signed counts per dimension, independent of POPCOUNT.
            for (int d=r*group; d<dim && d<(r+1)*group; ++d) {
                const uint32_t bit = uint32_t(1) << (d%32);
                const int match = ((qs[d/32]&bit) == (ks[t*words+d/32]&bit)) ? 1 : -1;
                const bool mq = qm[d/32]&bit, mk = km[t*words+d/32]&bit;
                cs += match; cq += mq ? match : 0; ck += mk ? match : 0;
                cqk += mq && mk ? match : 0;
            }
            const int i=t*reps+r;
            const float tail = (dq[r]*dk[i])*float(cqk);
            const float g = std::fma(ql[r]*kl[i],float(cs),
                std::fma(dq[r]*kl[i],float(cq),std::fma(ql[r]*dk[i],float(ck),tail)));
            total = total+g;
        }
        out[t] = total*float(1.0/std::sqrt(double(dim)));
    }
}
