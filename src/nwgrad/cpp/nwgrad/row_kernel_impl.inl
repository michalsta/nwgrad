// ── Row-wise banded leaf kernels — one instantiation per ISA level ────────────
//
// Included by level_common.inc *inside* the level namespace, immediately after
// kernels_impl.inl, so `KW` (the level's native double-lane count) is in scope.
// CMake compiles each level TU with that level's real -march, so the plain
// autovectorizable loops below become SSE2 / AVX2 / AVX-512 / NEON code with no
// #pragma target — which is the whole point of the per-TU scheme.
//
// These are the same bodies the old nwgrad_simd stamping emitted, moved here
// verbatim.  viterbi_affine_simd (aligner_simd.hpp) owns all the banded indexing
// and calls these through the LevelKernels function-pointer table; they see only
// contiguous row slices, never the band geometry.
//
// Bit-exactness survives every ISA width because the bodies contain only max, add
// and subtract on doubles — each IEEE-exact at any width, with no a*b+c for FMA to
// fuse.  The two X-state subtractions are kept LEFT-ASSOCIATED, `(v - go) - ge`,
// never folded to `v - (go + ge)`, which would round differently and break the
// tracebacks that re-derive the path by exact float equality.

#ifndef NWGRAD_LEVEL_NS
#  error "row_kernel_impl.inl is included from a level TU (level_common.inc); not standalone"
#endif

// Columns per interleaved block.  A property of the µarch, not the algorithm: a
// narrow, shallow core wants long vector runs; a wide out-of-order one wants short
// blocks, so block k+1's carry-free half issues underneath block k's serial VY
// latency chain.  Native lane count is the proxy — W≥4 means an AVX2+ (deep OOO)
// backend.  NWGRAD_ROW_BLOCK overrides it for re-tuning on new hardware.
#ifdef NWGRAD_ROW_BLOCK
inline constexpr int ROW_BLOCK = NWGRAD_ROW_BLOCK;
#else
inline constexpr int ROW_BLOCK = (KW >= 4) ? 64 : 256;
#endif

inline constexpr double ROW_NEG_INF = -std::numeric_limits<double>::infinity();

// The whole banded row in ONE leveled call: the interleaved block loop (carry-free
// VM/VX, then the serial VY carry, per block) and — for Local — the row's max3.
//
// This fuses what were three function-pointer calls per block (row_mx, row_y, row_m3)
// into one call per row.  Two reasons it matters, both from the banded measurement:
// the per-row indirect-call overhead dominates a narrow band (band 8 ≈ 1.06× before),
// and keeping mx/y/max in one function lets the compiler optimize across them (register
// allocation, scheduling) instead of at three opaque call boundaries.  Bit-exact:
// identical operations in identical order.  `blk` is the interleave block size; the two
// X subtractions stay left-associated, `(v-go)-ge`, never folded — the traceback
// re-derives the path by exact float equality and depends on it.  LOCAL is a compile-
// time literal (the two entries below); it folds the M-clamp-to-0 and the row max away.
// VM/VX read only row i-1 (carry-free); VY carries within the row (Farrar lazy-F in
// exact form — one `-ge_a` per step).  Returns max3 for Local (caller rescans for the
// argmax only if it beats the running best); Global ignores the return.
#define NWGRAD_BANDED_ROW_BODY(LOCAL)                                           \
    for (int base = lo; base <= hi; base += blk) {                             \
        const int end = (base + blk - 1 < hi) ? base + blk - 1 : hi;           \
        for (int j = base; j <= end; ++j) {          /* carry-free VM, VX */   \
            double d = vm_prev[j - 1];                                         \
            d = std::max(d, vx_prev[j - 1]);                                   \
            d = std::max(d, vy_prev[j - 1]);                                   \
            double mv = d + subrow[j];                                         \
            if (LOCAL) mv = std::max(mv, 0.0);                                 \
            double x = (vm_prev[j] - go_b) - ge_b;                             \
            x = std::max(x, vx_prev[j] - ge_b);                               \
            x = std::max(x, (vy_prev[j] - go_b) - ge_b);                       \
            vm_cur[j] = mv;                                                    \
            vx_cur[j] = x;                                                     \
        }                                                                     \
        for (int j = base; j <= end; ++j) {          /* serial VY carry */     \
            double open = (std::max(vm_cur[j - 1], vx_cur[j - 1]) - go_a) - ge_a; \
            vy_cur[j] = std::max(open, vy_cur[j - 1] - ge_a);                  \
        }                                                                     \
    }                                                                         \
    if (LOCAL) {                                                              \
        double best = ROW_NEG_INF;                                            \
        for (int j = lo; j <= hi; ++j)                                        \
            best = std::max(best,                                             \
                            std::max(vm_cur[j], std::max(vx_cur[j], vy_cur[j]))); \
        return best;                                                          \
    }                                                                         \
    return ROW_NEG_INF;

static double banded_row_global(
    double* __restrict vm_cur, double* __restrict vx_cur, double* __restrict vy_cur,
    const double* __restrict vm_prev, const double* __restrict vx_prev,
    const double* __restrict vy_prev, const double* __restrict subrow,
    int lo, int hi, int blk,
    double go_a, double ge_a, double go_b, double ge_b) noexcept {
    NWGRAD_BANDED_ROW_BODY(0)
}

static double banded_row_local(
    double* __restrict vm_cur, double* __restrict vx_cur, double* __restrict vy_cur,
    const double* __restrict vm_prev, const double* __restrict vx_prev,
    const double* __restrict vy_prev, const double* __restrict subrow,
    int lo, int hi, int blk,
    double go_a, double ge_a, double go_b, double ge_b) noexcept {
    NWGRAD_BANDED_ROW_BODY(1)
}

#undef NWGRAD_BANDED_ROW_BODY
