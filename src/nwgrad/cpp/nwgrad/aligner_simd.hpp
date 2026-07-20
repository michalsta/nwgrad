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
//
//
// HOW THE ISA IS CHOSEN (there is now only one mechanism)
//
// The vectorized code is compiled once per instruction-set *level*, each level in
// its own translation unit with that level's real -march flag (see simd_levels.hpp
// and the level_*.cpp TUs).  This file holds no #pragma GCC target machinery and no
// second ISA probe: the striped Full kernel and the row-wise GuideBanded leaf
// kernels both come from one LevelKernels table, selected by the aligner's backend (the
// global default when "auto", overridable via set_isa_level / NWGRAD_ISA).  A header-only consumer who links
// no level TU gets nullptr kernels and falls back to the scalar path (run_viterbi).

#ifndef NWGRAD_ALIGNER_HPP_INCLUDED
#  error "aligner_simd.hpp is included from aligner.hpp; do not include it directly"
#endif

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include "simd_levels.hpp"

namespace nwgrad_simd {
inline constexpr double NEG_INF = -std::numeric_limits<double>::infinity();
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
template<GapModel GM, AlignMode AM, AlignBand AB, class T>
void Aligner<GM, AM, AB, T>::build_profile(DpBuffer& buf) const {
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
template<GapModel GM, AlignMode AM, AlignBand AB, class T>
const double* Aligner<GM, AM, AB, T>::subrow(DpBuffer& buf, int i, int lo, int hi) const {
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
// compile.  A simd backend stays a legal request for a linear aligner — it simply
// returns the fastest linear kernel that exists, which is the scalar one.
// ═════════════════════════════════════════════════════════════════════════════
// Viterbi (Simd) — Affine gap model, GuideBanded (and Full fallback)
// ═════════════════════════════════════════════════════════════════════════════
//
// The affine model vectorizes *better* than the linear one, which is not obvious.
// Two of its three tables (VM, VX) read only row i-1 and so are entirely carry-free;
// only VY carries.  That is roughly three quarters of the per-cell work handed to the
// vector unit, against only two thirds in the linear case.
//
// This function owns the banded indexing (band_fill / at / rat / jlo / jhi / stride_);
// the vectorized inner loops are the leaf kernels in the LevelKernels table, compiled
// per ISA level.  Full-band Global+Local is served by the faster striped kernel
// (run_dispatched_affine); this path is taken for GuideBanded, and as the Full
// fallback should the striped kernel ever be absent.
template<GapModel GM, AlignMode AM, AlignBand AB, class T>
void Aligner<GM, AM, AB, T>::viterbi_affine_simd(DpBuffer& buf, const LevelKernels& K) {
    using nwgrad_simd::NEG_INF;

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
        // One leveled call runs the whole row (the interleaved block loop is inside the
        // kernel now), instead of ~3 function-pointer calls per block — the per-row
        // indirect-call overhead was what made a narrow band ≈ scalar.
        const int blk = K.row_block;
        if constexpr (AM == AlignMode::Local) {
            const double rowmax = K.banded_row_local(
                vm_cur, vx_cur, vy_cur, vm_prev, vx_prev, vy_prev, sr, lo, hi, blk,
                go_a, ge_a, go_b, ge_b);
            if (rowmax > best_local) {
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
        } else {
            K.banded_row_global(
                vm_cur, vx_cur, vy_cur, vm_prev, vx_prev, vy_prev, sr, lo, hi, blk,
                go_a, ge_a, go_b, ge_b);
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
