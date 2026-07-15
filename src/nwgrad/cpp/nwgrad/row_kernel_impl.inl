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

// Affine, carry-free half: VM and VX read only row i-1, so there is no loop-carried
// dependency and the whole row goes to the vector unit.  LOCAL is a 0/1 literal the
// compiler folds away.
#define NWGRAD_ROW_BODY_MX(LOCAL)                                               \
    for (int j = lo; j <= hi; ++j) {                                            \
        double d = vm_prev[j - 1];                                              \
        d = std::max(d, vx_prev[j - 1]);                                        \
        d = std::max(d, vy_prev[j - 1]);                                        \
        double mv = d + subrow[j];                                              \
        if (LOCAL) mv = std::max(mv, 0.0);                                      \
        double x = (vm_prev[j] - go_b) - ge_b;                                  \
        x = std::max(x, vx_prev[j] - ge_b);                                     \
        x = std::max(x, (vy_prev[j] - go_b) - ge_b);                            \
        vm_cur[j] = mv;                                                         \
        vx_cur[j] = x;                                                          \
    }

static void row_mx_global(
    double* __restrict vm_cur,       double* __restrict vx_cur,
    const double* __restrict vm_prev, const double* __restrict vx_prev,
    const double* __restrict vy_prev, const double* __restrict subrow,
    int lo, int hi, double go_b, double ge_b) noexcept {
    NWGRAD_ROW_BODY_MX(0)
}

static void row_mx_local(
    double* __restrict vm_cur,       double* __restrict vx_cur,
    const double* __restrict vm_prev, const double* __restrict vx_prev,
    const double* __restrict vy_prev, const double* __restrict subrow,
    int lo, int hi, double go_b, double ge_b) noexcept {
    NWGRAD_ROW_BODY_MX(1)
}

// Affine, the carry.  VY reads VM/VX of the *current* row at j-1 — already written
// by the mx pass and independent of VY — plus its own left neighbour.  That last
// term is the serial dependency no vectorizer can break, left deliberately scalar.
// Farrar's lazy-F in exact form: one `- ge_a` per step, the chain the scalar loop
// walks, hence a bit-identical fixpoint.
static void row_y(
    double* __restrict vy_cur,
    const double* __restrict vm_cur, const double* __restrict vx_cur,
    int lo, int hi, double go_a, double ge_a) noexcept {
    for (int j = lo; j <= hi; ++j) {
        double open = (std::max(vm_cur[j - 1], vx_cur[j - 1]) - go_a) - ge_a;
        vy_cur[j] = std::max(open, vy_cur[j - 1] - ge_a);
    }
}

// Row maximum of max3(VM,VX,VY): a pure reduction.  Local mode needs an argmax, but
// an argmax inside the DP loop blocks vectorization — so it is hoisted out: reduce
// here, and rescan for the index only when the row actually beats the running best.
static double row_m3(
    const double* __restrict a, const double* __restrict b,
    const double* __restrict c, int lo, int hi) noexcept {
    double best = ROW_NEG_INF;
    for (int j = lo; j <= hi; ++j)
        best = std::max(best, std::max(a[j], std::max(b[j], c[j])));
    return best;
}

#undef NWGRAD_ROW_BODY_MX
