#pragma once

// ── The Simd Viterbi kernel ───────────────────────────────────────────────────
//
// Included from the bottom of aligner.hpp, once Aligner is a complete type.  It
// is not a standalone header and must not be included directly.
//
//
// WHY THE TEXTBOOK SIMD ALIGNERS ARE NOT HERE
//
// Wozniak (antidiagonal), Rognes (query profile) and Farrar (striped + lazy-F)
// are all score-only, saturating int8/int16 algorithms.  They win by throwing the
// DP table away and packing 16-32 lanes into a register.  This library can do
// neither: the traceback walks VM/VX/VY and soft_grad exponentiates every cell of
// F/B, so the tables are the *product*, not scratch — and they are double because
// the whole premise is a *learned*, continuous substitution matrix.  Quantize to
// int16 and the gradient is gone.  A striped Farrar SW cannot produce a gradient
// at all.
//
// Exactly two of their ideas transplant, and they are what follows.
//
//
// 1. THE QUERY PROFILE (Rognes) — kills the gather.
//
// The scalar DP reads sub(i,j) = blk[a[i-1]*nalpha + b[j-1]]: an indexed gather.
// No vectorizer can vectorize a loop whose loads are gathers unless the target has
// one — and AArch64/NEON has *none at all*, while the shipped x86 wheels compile to
// the x86-64 baseline (SSE2: no gather either).  This is the single reason the
// scalar inner loop is emitted as scalar code on every platform we ship.
//
// The profile precomputes, for every alphabet symbol c, that symbol's scores
// against all of sequence B, laid out contiguously in j.  Row i then reads a plain
// contiguous slice — a vector load.
//
//
// 2. THE LAZY-F FIXUP (Farrar) — and here it is BIT-EXACT.
//
// The tracebacks in aligner.hpp do not store direction pointers.  They *re-derive*
// the path by exact floating-point equality against the stored table:
//
//     rat(buf.H, i, j) == rat(buf.H, i-1, j-1) + sub(i, j)
//
// So if this kernel reassociates arithmetic and shifts a single ULP, no traceback
// branch matches, the walk falls through to its final else, and it emits a gap
// where a match belongs.  Scores stay right while alignments and hard gradients
// quietly rot.  It does not fail loudly.  It fails silently, which is worse.
//
// The escape is that every transform below is bit-exact, not merely close:
//
//   - max is exact and associative in IEEE-754 (no NaN arises here).
//   - The carry-free passes compute the same single add/subtract per term as the
//     scalar code, in the same order.  In particular `(v - go) - ge` is kept as two
//     left-associated subtractions and is NOT folded into `v - (go + ge)`, which
//     would round differently.
//   - max(a-c, b-c) and max(a,b)-c agree bitwise: rounding is monotone, so the max
//     selects the same operand either way.
//   - The VY/H carry is propagated one `- ge` at a time — precisely the chain the
//     scalar loop walks — so its fixpoint is bit-identical, not approximate.
//
// That is the whole trick, and it is why no traceback needed rewriting and why the
// existing test suite is a valid verification of this file.  tests/cpp/test_simd_bitexact.cpp
// asserts the tables bit-for-bit; if it ever fails, this reasoning is wrong and the
// kernel is not safe to ship.

#ifndef NWGRAD_ALIGNER_HPP_INCLUDED
#  error "aligner_simd.hpp is included from aligner.hpp; do not include it directly"
#endif

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

namespace nwgrad_simd {

inline constexpr double NEG_INF = -std::numeric_limits<double>::infinity();

// ── Row kernels, stamped once per instruction set ─────────────────────────────
//
// The bodies live in macros, not in templates, so that each one can be *defined*
// inside a `#pragma GCC target(...)` region — which is the only way to get the
// compiler to emit AVX2 or AVX-512 for code in a header that must still load on a
// CPU that has neither.  The same restructured C++ is compiled three or four times;
// the compiler writes the vectors.  No intrinsic is hand-written anywhere.
//
// Selection is a function-pointer table, resolved once, NOT __attribute__((target_clones)).
// target_clones needs GNU ifunc, which musl and macOS do not have, and would break the
// musllinux and macOS wheels cibuildwheel already builds.  A plain table needs no ifunc
// and works on GCC and Clang across Linux, macOS and musl alike.  The indirect call
// costs a few cycles *per row*, against a row of hundreds of cells — it does not show up.
//
// On AArch64 none of this exists: NEON is mandatory baseline, so the kernels simply
// vectorize where they stand and the table has exactly one entry.  MSVC has no such
// pragma and likewise takes the single baseline entry.
//
// Bit-exactness survives the wider ISAs because the bodies contain only max, add and
// subtract on doubles — every one of which is IEEE-exact at any width.  There is no
// `a*b+c` anywhere, so there is nothing for FMA contraction to fuse and nothing for it
// to round differently.  test_simd_bitexact.cpp checks this against whatever ISA the
// running CPU selects, so CI on an AVX2 runner tests the AVX2 clone.

// The multi-ISA machinery needs `#pragma GCC target`, which is GCC/Clang on x86 only.
// Everywhere else — MSVC, and every AArch64 including Apple Silicon — the table has a
// single entry and the kernels simply vectorize at whatever the target's baseline is.
// On AArch64 that is NEON, which is mandatory, so Apple Silicon gets the full benefit
// with none of this scaffolding.
//
// -DNWGRAD_NO_MULTIVERSION forces that single-entry path on a machine that could have
// used the table.  It is not decoration: it is the only way to *compile and run* the
// MSVC/AArch64 code path on a Linux x86 box, and an untested fallback path is how a
// header-only library breaks on a platform nobody owns.
#if !defined(NWGRAD_NO_MULTIVERSION) && \
    defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#  define NWGRAD_X86_MULTIVERSION 1
#endif

// Columns per interleaved block; see the loop in viterbi_affine_simd.  The right value
// is a property of the *microarchitecture*, not of the algorithm, so it rides in the
// dispatch table alongside the kernels rather than being a compile-time constant — the
// same binary must serve both.  Measured with stress_batch (the real library binary; a
// microbenchmark gave code-layout artifacts larger than the effect):
//
//                        Opteron 6380 (SSE2)   i5-12500 (AVX2)
//     unblocked                 1.33x                1.28x
//     block =  64               1.17x                1.44x
//     block = 128               1.27x                1.33x
//     block = 256               1.36x                1.29x
//
// A narrow, shallow core wants long vector runs; a wide one wants short blocks, so that
// block k+1's carry-free half can issue underneath block k's serial VY latency chain.
// Guessing one number for both costs ~10% on whichever machine loses the coin toss.
// NWGRAD_ROW_BLOCK overrides it for re-tuning on new hardware.
#ifdef NWGRAD_ROW_BLOCK
inline constexpr int BLOCK_BASELINE = NWGRAD_ROW_BLOCK;
inline constexpr int BLOCK_WIDE     = NWGRAD_ROW_BLOCK;
#else
inline constexpr int BLOCK_BASELINE = 256;  // SSE2 / NEON / old cores
inline constexpr int BLOCK_WIDE     = 64;   // AVX2+ implies a deep out-of-order window
#endif

// Parameter lists, shared between the definitions and the function-pointer types.
#define NWGRAD_MX_ARGS                                                          \
    double* __restrict vm_cur,       double* __restrict vx_cur,                 \
    const double* __restrict vm_prev, const double* __restrict vx_prev,         \
    const double* __restrict vy_prev, const double* __restrict subrow,          \
    int lo, int hi, double go_b, double ge_b
#define NWGRAD_Y_ARGS                                                           \
    double* __restrict vy_cur,                                                  \
    const double* __restrict vm_cur, const double* __restrict vx_cur,           \
    int lo, int hi, double go_a, double ge_a
#define NWGRAD_M3_ARGS                                                          \
    const double* __restrict a, const double* __restrict b,                     \
    const double* __restrict c, int lo, int hi

// Affine, carry-free half: VM and VX read only row i-1, so they have no loop-carried
// dependency at all and the whole row goes to the vector unit.  LOCAL is a 0/1 literal
// the compiler folds away.
//
// The two subtractions in the X state are kept LEFT-ASSOCIATED — `(v - go_b) - ge_b`,
// never `v - (go_b + ge_b)`.  Folding them would round differently and break the
// bit-exactness the tracebacks depend on.
#define NWGRAD_BODY_MX(LOCAL)                                                   \
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

// Affine, the carry.  VY reads VM/VX of the *current* row at j-1 — already written
// above, and independent of VY — plus its own left neighbour.  That last term is the
// serial dependency no vectorizer can break, and it is deliberately left scalar.
//
// This is Farrar's lazy-F in exact form: one `- ge_a` per propagation step, the same
// chain the scalar loop walks, hence a bit-identical fixpoint rather than a close one.
#define NWGRAD_BODY_Y                                                           \
    for (int j = lo; j <= hi; ++j) {                                            \
        double open = (std::max(vm_cur[j - 1], vx_cur[j - 1]) - go_a) - ge_a;   \
        vy_cur[j] = std::max(open, vy_cur[j - 1] - ge_a);                       \
    }

// Row maximum of max3(VM,VX,VY): a pure reduction.  Local mode needs an *argmax*, but
// an argmax inside the DP loop blocks vectorization, so it is hoisted out — reduce
// first, and rescan for the index only when the row actually beats the running best.
#define NWGRAD_BODY_M3                                                          \
    double best = NEG_INF;                                                      \
    for (int j = lo; j <= hi; ++j)                                              \
        best = std::max(best, std::max(a[j], std::max(b[j], c[j])));            \
    return best;

#define NWGRAD_STAMP(NS)                                                        \
    namespace NS {                                                              \
        inline void row_mx_global(NWGRAD_MX_ARGS) noexcept { NWGRAD_BODY_MX(0) } \
        inline void row_mx_local (NWGRAD_MX_ARGS) noexcept { NWGRAD_BODY_MX(1) } \
        inline void row_y        (NWGRAD_Y_ARGS)  noexcept { NWGRAD_BODY_Y }    \
        inline double row_m3     (NWGRAD_M3_ARGS) noexcept { NWGRAD_BODY_M3 }   \
    }

// Whatever -march the translation unit was compiled with.  On AArch64 this is NEON;
// on shipped x86 wheels it is the x86-64 baseline, i.e. SSE2.
NWGRAD_STAMP(isa_baseline)

#ifdef NWGRAD_X86_MULTIVERSION
#  pragma GCC push_options
#  pragma GCC target("avx")
     NWGRAD_STAMP(isa_avx)
#  pragma GCC pop_options

#  pragma GCC push_options
#  pragma GCC target("avx2,fma")
     NWGRAD_STAMP(isa_avx2)
#  pragma GCC pop_options

#  pragma GCC push_options
#  pragma GCC target("avx512f,avx512dq,avx512vl,avx512bw")
     NWGRAD_STAMP(isa_avx512)
#  pragma GCC pop_options
#endif

// ── The dispatch table ────────────────────────────────────────────────────────

struct RowKernels {
    void   (*mx_global)(NWGRAD_MX_ARGS);
    void   (*mx_local) (NWGRAD_MX_ARGS);
    void   (*y)        (NWGRAD_Y_ARGS);
    double (*m3)       (NWGRAD_M3_ARGS);
    const char* isa;
    int    block;      // columns per interleaved block; see BLOCK_* above
};

#define NWGRAD_TABLE(NS, NAME, BLK) \
    RowKernels{ &NS::row_mx_global, &NS::row_mx_local, &NS::row_y, &NS::row_m3, NAME, BLK }

// NWGRAD_ISA overrides the CPU probe.  This is not a debugging afterthought: without
// it the AVX-512 path would be unreachable on a machine that lacks it and therefore
// untestable, and — more usefully — it is what lets a benchmark compare ISA levels on
// one machine instead of guessing.  An unsupported request falls back rather than
// crashing with SIGILL.
inline RowKernels select_row_kernels() {
#ifdef NWGRAD_X86_MULTIVERSION
    __builtin_cpu_init();
    const char* want = std::getenv("NWGRAD_ISA");
    const bool has_avx    = __builtin_cpu_supports("avx");
    const bool has_avx2   = __builtin_cpu_supports("avx2");
    const bool has_avx512 = __builtin_cpu_supports("avx512f")
                         && __builtin_cpu_supports("avx512vl")
                         && __builtin_cpu_supports("avx512dq");

    if (want) {
        std::string w(want);
        if (w == "baseline")                  return NWGRAD_TABLE(isa_baseline, "baseline", BLOCK_BASELINE);
        if (w == "avx"    && has_avx)         return NWGRAD_TABLE(isa_avx,      "avx",      BLOCK_BASELINE);
        if (w == "avx2"   && has_avx2)        return NWGRAD_TABLE(isa_avx2,     "avx2",     BLOCK_WIDE);
        if (w == "avx512" && has_avx512)      return NWGRAD_TABLE(isa_avx512,   "avx512",   BLOCK_WIDE);
        // Asked for something this CPU cannot run: fall through to the probe.
    }

    if (has_avx512) return NWGRAD_TABLE(isa_avx512, "avx512", BLOCK_WIDE);
    if (has_avx2)   return NWGRAD_TABLE(isa_avx2,   "avx2",   BLOCK_WIDE);
    // Deliberately NOT selecting plain AVX by default.  It is a measured regression on
    // Bulldozer/Piledriver, whose FP unit cracks every 256-bit op into two 128-bit
    // halves — so the wider registers buy nothing and cost extra uops.  AVX without
    // AVX2 essentially means that family.  Reachable via NWGRAD_ISA=avx if you want to
    // measure it; not chosen for you.
#endif
    return NWGRAD_TABLE(isa_baseline, "baseline", BLOCK_BASELINE);
}

// Resolved once, on first use.  A function-local static is thread-safe since C++11,
// which matters: the batch workers all reach this concurrently.
inline const RowKernels& row_kernels() {
    static const RowKernels k = select_row_kernels();
    return k;
}

inline const char* active_isa() { return row_kernels().isa; }

}  // namespace nwgrad_simd


// ── Query profile ─────────────────────────────────────────────────────────────
//
// prof[c*(n+1) + j] = score of symbol c against b[j-1], for j in [1, n].
//
// Indexed by the DP's own 1-based j so the row kernels can read subrow[j] directly
// alongside vm_cur[j].  Slot j=0 is padding and never read — it exists so that
// `subrow` is a genuine pointer into the buffer rather than `prof_row - 1`, which
// would form a pointer before the start of the allocation and is undefined behaviour
// even if it happens to work.
//
// Rebuilt on every simd call rather than cached: the cost is O(nalpha * n) at ~2 ops
// per entry against a DP of O(m * n) at ~15 ops per cell — about 2.7/m, so ~5% at
// m=50 and ~1% at m=200 — and it buys immunity to a whole class of staleness bugs
// when the matrix changes under a SeqPair between re-alignments.
template<GapModel GM, AlignMode AM, AlignBand AB>
void Aligner<GM, AM, AB>::build_profile(DpBuffer& buf) const {
    const size_t w = static_cast<size_t>(n_) + 1;
    const size_t need = static_cast<size_t>(nalpha_) * w;
    if (buf.prof.size() < need) buf.prof.resize(need);

    for (int c = 0; c < nalpha_; ++c) {
        const double* blk_row = blk_ + static_cast<size_t>(c) * static_cast<size_t>(nalpha_);
        double* dst = buf.prof.data() + static_cast<size_t>(c) * w;
        dst[0] = 0.0;  // padding; never read
        for (int j = 1; j <= n_; ++j)
            dst[j] = blk_row[b_idx_[static_cast<size_t>(j) - 1]];
    }
}

// Substitution scores for row i, contiguous in j.
//
// Full mode hands back a slice of the profile.  Banded mode gathers the row's short
// span instead: a full-width profile costs O(nalpha * n) against a banded DP of only
// O(m * band), so for a narrow band it would outcost the thing it is accelerating.
template<GapModel GM, AlignMode AM, AlignBand AB>
const double* Aligner<GM, AM, AB>::subrow(DpBuffer& buf, int i, int lo, int hi) const {
    if constexpr (AB == AlignBand::Full) {
        (void)lo; (void)hi;
        const size_t w = static_cast<size_t>(n_) + 1;
        return buf.prof.data() +
               static_cast<size_t>(a_idx_[static_cast<size_t>(i) - 1]) * w;
    } else {
        const size_t w = static_cast<size_t>(n_) + 1;
        if (buf.subbuf.size() < w) buf.subbuf.resize(w);
        const double* blk_row =
            blk_ + static_cast<size_t>(a_idx_[static_cast<size_t>(i) - 1]) *
                       static_cast<size_t>(nalpha_);
        double* dst = buf.subbuf.data();
        for (int j = lo; j <= hi; ++j)
            dst[j] = blk_row[b_idx_[static_cast<size_t>(j) - 1]];
        return dst;
    }
}


// ═════════════════════════════════════════════════════════════════════════════
// Linear gap model — WHY THERE IS NO SIMD KERNEL FOR IT
// ═════════════════════════════════════════════════════════════════════════════
//
// This is not an omission.  A vectorized linear kernel was written, measured, and
// thrown away, because it was SLOWER than the scalar one it replaced.
//
// The linear recurrence collapses to a single carry:
//
//     H[i][j] = max( t[j], H[i][j-1] - ge_a )        t[j] carry-free from row i-1
//
// and that carry is a pure *latency* chain — one subtract, one max, roughly six
// cycles, serially, per cell.  No vector width in existence shortens it.  All SIMD
// can do is compute t[] faster, and t[] is only two of the three terms.  Measured on
// this box (Piledriver, 2 doubles/vector), scalar linear already runs at ~9.7
// cycles/cell against that ~6-cycle floor: there is almost nothing left to win, and
// the extra pass over the row costs more than the vectorization saves.
//
//     linear/global  len=200   scalar  9.7 cyc/cell   simd 10.6 cyc/cell   0.92x
//     linear/global  len= 50   scalar 10.5 cyc/cell   simd 14.3 cyc/cell   0.73x
//
// Blocking the two halves so the carry-free work could issue underneath the carry's
// latency recovered some of it (0.82x -> 0.92x at len=200) and still did not break
// even.  The affine model escapes this only because it has ~4x more work per cell to
// hide behind the same chain — see below, where the same idea wins 2.3x.
//
// So there is no linear kernel here at all, and run_viterbi() routes the linear model
// straight to viterbi_linear() regardless of the kernel field.  This is what a runtime
// kernel selector is *for*: a compile-time template parameter would have forced us
// either to ship a kernel we know to be slower, or to make the configuration fail to
// compile.  DpKernel::Simd stays a legal request for a linear aligner — it simply
// returns the fastest linear kernel that exists, which is the scalar one.
// ═════════════════════════════════════════════════════════════════════════════
// Viterbi (Simd) — Affine gap model
// ═════════════════════════════════════════════════════════════════════════════
//
// The affine model vectorizes *better* than the linear one, which is not obvious.
// Two of its three tables (VM, VX) read only row i-1 and so are entirely carry-free;
// only VY carries.  That is roughly three quarters of the per-cell work handed to the
// vector unit, against only two thirds in the linear case.

template<GapModel GM, AlignMode AM, AlignBand AB>
void Aligner<GM, AM, AB>::viterbi_affine_simd(DpBuffer& buf) {
    using namespace nwgrad_simd;

    band_fill(buf.VM, NEG_INF);
    band_fill(buf.VX, NEG_INF);
    band_fill(buf.VY, NEG_INF);

    if constexpr (AM == AlignMode::Global) {
        at(buf.VM, 0, 0) = 0.0;
        const int bi = border_rows(), bj = border_cols();
        for (int i = 1; i <= bi; ++i)
            at(buf.VX, i, 0) = -(params_->gap_open_b + i * params_->gap_extend_b);
        for (int j = 1; j <= bj; ++j)
            at(buf.VY, 0, j) = -(params_->gap_open_a + j * params_->gap_extend_a);
    } else {
        for (int i = 0; i <= m_; ++i) at(buf.VM, i, 0) = 0.0;
        for (int j = 0; j <= n_; ++j) at(buf.VM, 0, j) = 0.0;
    }

    if constexpr (AB == AlignBand::Full) build_profile(buf);

    const double go_a = params_->gap_open_a,   ge_a = params_->gap_extend_a;
    const double go_b = params_->gap_open_b,   ge_b = params_->gap_extend_b;

    best_i_ = 0; best_j_ = 0; best_tbl_ = TBTable::M;
    double best_local = 0.0;

    // Resolved once per DP, not once per row: an indirect call per row is nothing
    // against a row of hundreds of cells.
    const RowKernels& K = row_kernels();

    for (int i = 1; i <= m_; ++i) {
        const int lo = jlo(i), hi = jhi(i);
        if (lo > hi) continue;

        const double* sr = subrow(buf, i, lo, hi);

        const size_t off_cur  = static_cast<size_t>(i)     * stride_;
        const size_t off_prev = static_cast<size_t>(i - 1) * stride_;
        double* vm_cur = buf.VM.data() + off_cur;
        double* vx_cur = buf.VX.data() + off_cur;
        double* vy_cur = buf.VY.data() + off_cur;
        const double* vm_prev = buf.VM.data() + off_prev;
        const double* vx_prev = buf.VX.data() + off_prev;
        const double* vy_prev = buf.VY.data() + off_prev;

        // Interleaved in blocks, NOT two full-row passes.
        //
        // The VY carry is a serial latency chain.  Run as one whole-row pass it is
        // *exposed*: nothing is left to issue alongside it.  The scalar kernel never
        // has that problem — its fused loop lets the out-of-order engine compute
        // iteration j+1's VM/VX underneath iteration j's VY chain.  Splitting the row
        // in two throws that overlap away, and on a wide core it costs more than the
        // vectorization wins.
        //
        // Blocking gives it back: block k+1's carry-free half reads only row i-1, so it
        // is independent of block k's carry and can issue underneath it, while each half
        // is still long enough to vectorize.  Bit-exactness is untouched — same
        // operations, same order; VY[base] reads VM/VX[base-1] from the block just done.
        const int blk = K.block;
        for (int base = lo; base <= hi; base += blk) {
            const int end = std::min(base + blk - 1, hi);
            if constexpr (AM == AlignMode::Local)
                K.mx_local (vm_cur, vx_cur, vm_prev, vx_prev, vy_prev, sr, base, end, go_b, ge_b);
            else
                K.mx_global(vm_cur, vx_cur, vm_prev, vx_prev, vy_prev, sr, base, end, go_b, ge_b);
            K.y(vy_cur, vm_cur, vx_cur, base, end, go_a, ge_a);
        }

        if constexpr (AM == AlignMode::Local) {
            if (K.m3(vm_cur, vx_cur, vy_cur, lo, hi) > best_local) {
                for (int j = lo; j <= hi; ++j) {
                    const double mv = vm_cur[j], xv = vx_cur[j], yv = vy_cur[j];
                    const double best_here = std::max(mv, std::max(xv, yv));
                    if (best_here > best_local) {
                        best_local = best_here; best_i_ = i; best_j_ = j;
                        if      (mv >= xv && mv >= yv) best_tbl_ = TBTable::M;
                        else if (xv >= yv)             best_tbl_ = TBTable::X;
                        else                            best_tbl_ = TBTable::Y;
                    }
                }
            }
        }
    }

    if constexpr (AM == AlignMode::Global) {
        double vm = rat(buf.VM, m_, n_), vx = rat(buf.VX, m_, n_), vy = rat(buf.VY, m_, n_);
        viterbi_score_ = std::max({vm, vx, vy});
        best_i_ = m_; best_j_ = n_;
        if      (vm >= vx && vm >= vy) best_tbl_ = TBTable::M;
        else if (vx >= vy)             best_tbl_ = TBTable::X;
        else                            best_tbl_ = TBTable::Y;
    } else {
        viterbi_score_ = best_local;
    }
}
