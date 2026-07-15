#pragma once

// DpBuffer — the DP table storage, split out of aligner.hpp so the leveled kernel
// TUs (which are compiled per ISA level) can see it without dragging in the whole
// Aligner template.  It is plain data: the only currency that crosses a level
// boundary (see simd_levels.hpp).

#include <vector>

// ── DpBuffer ──────────────────────────────────────────────────────────────────
//
// Holds all DP table vectors for one Aligner computation.
// Lives either inside the Aligner (own_buf_) or externally (e.g. per-thread).
// Vectors grow on demand and are never implicitly freed; call clear() to release.

struct DpBuffer {
    std::vector<double> H;                        // linear viterbi
    std::vector<double> VM, VX, VY;               // affine viterbi
    std::vector<double> F, B;                     // linear forward-backward
    std::vector<double> FM, FX, FY, BM, BX, BY;  // affine forward-backward

    // Used only by the Simd kernel.  `prof` is the query profile — the
    // substitution scores of every alphabet symbol against sequence B, laid out
    // contiguously in j so the DP row loop loads them with a vector load instead
    // of a gather (Full mode).  `subbuf` is the per-row equivalent for banded
    // mode, where a full-width profile would cost more than the banded DP itself.
    std::vector<double> prof, subbuf;

    // Used only by the striped affine kernel (leveled, in kernels_impl.inl).  It runs
    // on rolling striped rows and de-stripes each finished row into the row-major
    // VM/VX/VY above, so the traceback and hard_grad read the layout they expect and
    // the existing tests validate it unchanged.  srows = 6 rolling rows
    // {VM,VX,VY}×{prev,cur} in striped seg×W layout; sopenv = one striped openv row;
    // sprof = the query profile in striped order.  All O(n), not O(m·n).
    std::vector<double> srows, sopenv, sprof;

    void clear() noexcept {
        auto clr = [](std::vector<double>& v) noexcept { v.clear(); v.shrink_to_fit(); };
        clr(H);
        clr(VM); clr(VX); clr(VY);
        clr(F);  clr(B);
        clr(FM); clr(FX); clr(FY); clr(BM); clr(BX); clr(BY);
        clr(prof); clr(subbuf);
        clr(srows); clr(sopenv); clr(sprof);
    }
};
