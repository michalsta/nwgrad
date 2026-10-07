#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "align_params.hpp"

enum class GapModel  { Linear, Affine };
enum class AlignMode { Global, Local  };
enum class AlignBand { Full,   GuideBanded };

// How the traceback recovers a cell's predecessor.
//
//   Scores    retain VM/VX/VY for the whole DP and RE-DERIVE the argmax by repeating
//             the forward pass's comparison chain.  12 B/cell (float32).
//   Pointers  record the predecessor as one byte per cell per state during the fill
//             and simply FOLLOW it; scores then need only two rolling rows.
//             3 B/cell — and measured 1.4-2.2x faster, because the footprint is what
//             drives the page-fault cost that dominates the memory-bound regime.
//
// The two are bit-identical — same score, alignment, guide_j and gradient, ties
// included — because Pointers records exactly the argmax Scores would re-derive.
// Affine Full only; the linear and banded paths always keep their score tables.
//
//   Hirschberg  divide and conquer in O(n) memory: sweep to the middle row from both
//               ends keeping only rolling rows, join them to find where the optimal
//               path crosses, recurse on the two halves.  Nothing above the base case
//               is ever materialized, so the footprint is a few rows rather than a
//               table — ~0.9 MB for a 36,000 aa self-pair against 3.9 GB for Pointers.
//               It pays ~2x the cell work for that (each level re-sweeps its half).
//
// *** Hirschberg is NOT bit-exact with the other two, by construction. ***
//
// The join picks a midpoint argmax, and that choice cannot reproduce the backward-greedy
// M>X>Y tie-break the other two share, because the tie-break depends on the rows below
// the split — which Hirschberg has already discarded.  Where the optimum is unique the
// answers agree exactly; where alignments tie, it returns a DIFFERENT optimal path, hence
// a valid but different subgradient (path score == Viterbi score, always).  BUT: a pair
// no longer than hb_cutoff never splits, and is run AS Pointers (see run_viterbi), so it
// IS bit-exact — divergence is confined to pairs longer than hb_cutoff.  Affine Full Global
// only; explicitly asking for it on Local / linear / banded THROWS rather than silently
// running a different algorithm than requested.
//
//   Default    a per-problem SENTINEL, never stored on an Aligner: it resolves at compile
//              time to a Hirschberg mode for affine+global+full (the case Hirschberg
//              implements) and to Pointers for everything else.  This is what "traceback
//              defaults to Hirschberg" means without breaking Local/linear/banded, which
//              have no Hirschberg variant.  It resolves further on the SCALAR TYPE —
//              HirschbergPmax at T=float, Hirschberg at T=double — so the two precisions
//              do not merely differ in width, they run different carries; see kDefaultTb
//              for why.  SeqPair/SeqPairBatch leave the Aligner's own default in place
//              when handed this.
//
//   HirschbergPmax
//              Hirschberg with the VY carry computed by the CLOSED-FORM PREFIX MAX
//              (VY[c] = prefixmax_k(open[k] + k*ge_a) - (c-1)*ge_a) instead of the serial
//              `- ge_a` chain + lazy-F.  It is the one mode in this library that can
//              return a SUBOPTIMAL path, because the ramp k*ge_a is added and then
//              subtracted again and the round-trip costs a rounding proportional to the
//              column index.  Read that sentence before selecting it: no other mode here
//              trades optimality, and this one buys speed with it.  It is nonetheless the
//              DEFAULT AT T=float (see kDefaultTb) — at that precision the ramp is
//              two orders of magnitude below float32's own rounding, so the trade is
//              already paid for; at T=double it is strictly opt-in.  Everything else
//              about it is Hirschberg — same recursion, same
//              exact base case below hb_cutoff, same exact row 0 — so pairs that never
//              split are unaffected.  It remains bit-identical scalar-vs-simd across
//              every ISA level (the closed form depends only on the absolute column
//              index, not on the vector width).  See hb_kernel_impl.inl.
enum class TracebackMode { Scores, Pointers, Hirschberg, HirschbergPmax, Default };

// How compute_forward_back() evaluates log Z and the expected counts.
//   Scaled (default): probability space with exact power-of-two row rescaling and the
//     gradient fused into the backward pass — no transcendental call per cell.  Throws
//     std::domain_error when a pair's dynamic range does not fit a double (see
//     run_fwdbwd and soft_kernel_impl.inl); it never returns a silently degraded result.
//   ScaledOrLog: Scaled, falling back per problem to Log where Scaled would throw.
//   Log: the original log-space recurrences (lse per cell).  Unlimited range, slow.
// The soft path is NOT bit-exact across modes, ISAs or compilers; it is tested with
// tolerances.  The Viterbi/hard path is unaffected by any of this.
enum class SoftImpl { Scaled, ScaledOrLog, Log };

// Reassociating reductions for the scaled soft path only (needs -fopenmp-simd, which
// the build sets; without it the pragma is ignored and the loop stays scalar).
#define NWGRAD_PRAGMA_(x) _Pragma(#x)
#define NWGRAD_SOFT_SIMD_SUM(...) NWGRAD_PRAGMA_(omp simd reduction(+:__VA_ARGS__))

// Both Hirschberg modes share the whole divide-and-conquer driver — the recursion, the
// join, the base case, the memory story — and differ only in which carry the SWEEP uses.
// Everything structural therefore asks this rather than naming one of them.
inline constexpr bool is_hirschberg(TracebackMode t) noexcept {
    return t == TracebackMode::Hirschberg || t == TracebackMode::HirschbergPmax;
}

// Which Viterbi backend fills the DP tables is a runtime field (backend_, below), one of
// the unified vocabulary in simd_levels.hpp: "scalar_fallback" (the original plain
// kernel), "auto" (defer to the global default = best simd), or a named simd level
// ("sse2"/"avx2"/"avx512"/"neon").  Every simd level is *bit-exact* with the scalar one —
// identical tables to the last bit — which is what lets the exact-float-equality
// tracebacks keep working unchanged; so the backend is a speed knob, never a correctness
// one.  It selects the Viterbi (hence hard_grad) path only; forward-backward and soft_grad
// are the same shared code either way.  Runtime, not a template parameter: the branch is
// taken once per compute_viterbi() and amortized over m*n cells.  Backends are encoded as
// ints (kBackendAuto / kBackendScalar / a SimdLevel index) so an aligner can store one.

// ── Utility: convert a pair of aligned strings to a guide_j vector ────────────
//
// a_aligned, b_aligned: equal-length strings with '-' for gaps.
// Returns a vector of length (m+1) where m = non-gap chars in a_aligned.
// guide_j[i] = the column j in the DP table after consuming i characters of a.
// Pass the result to Aligner::set_problem when using AlignBand::GuideBanded.
//
// When no explicit guide is given to set_problem, a proportional guide
// guide_j[i] = round(i * n / m) is used, tracking the main diagonal of the
// rectangular DP matrix.

inline std::vector<int> guide_j_from_aligned(std::string_view a_aligned,
                                              std::string_view b_aligned) {
    if (a_aligned.size() != b_aligned.size())
        throw std::invalid_argument("aligned strings must have equal length");
    std::vector<int> gj;
    int j = 0;
    gj.push_back(j);
    for (size_t c = 0; c < a_aligned.size(); ++c) {
        bool a_gap = (a_aligned[c] == '-');
        bool b_gap = (b_aligned[c] == '-');
        if (a_gap && b_gap)
            throw std::invalid_argument("aligned strings have a column with two gaps");
        if (!a_gap && !b_gap) { ++j; gj.push_back(j); }  // match/mismatch
        else if (a_gap)         ++j;                       // gap in a: j advances
        else                    gj.push_back(j);           // gap in b: i advances
    }
    return gj;
}

// ── log-sum-exp helpers ───────────────────────────────────────────────────────

static inline double lse2(double a, double b) noexcept {
    if (a == -std::numeric_limits<double>::infinity()) return b;
    if (b == -std::numeric_limits<double>::infinity()) return a;
    if (a > b) return a + std::log1p(std::exp(b - a));
    return b + std::log1p(std::exp(a - b));
}

static inline double lse3(double a, double b, double c) noexcept {
    return lse2(lse2(a, b), c);
}

// DpBuffer (the DP table storage) now lives in its own header so the leveled kernel
// TUs can include it without the whole Aligner.
#include "dp_buffer.hpp"

// Runtime per-ISA-level dispatch table for the leveled affine kernels.
#include "simd_levels.hpp"

// The per-pair scaled soft kernels at the baseline ISA (scalar_fallback, header-only
// builds, levels without their own copy).  The level TUs compile the same file again
// with their -march; see soft_kernel_impl.inl.
namespace nwgrad_soft_base {
#include "soft_kernel_impl.inl"
}

// ── Aligner ───────────────────────────────────────────────────────────────────
//
// Pipeline API (call in order):
//   1. set_problem(seq_a, seq_b, params[, band[, guide_j]])
//   2a. compute_viterbi()       — max-based DP; enables score(), alignment(), hard_grad()
//   2b. compute_forward_back()  — lse forward+backward; enables log_z(), soft_grad()
//   3. score() / log_z() / alignment() / hard_grad() / soft_grad()
//
// Gap model and alignment mode are fixed at compile time via template parameters.
// AlignParams carries asymmetric gap penalties: gap_{open,extend}_{a,b} where
// suffix _a applies to gaps in sequence A (Y/VY state, consuming B) and
// suffix _b applies to gaps in sequence B (X/VX state, consuming A).
//
// Each of 2a/2b/3 has an overload that accepts a DpBuffer& to use instead of
// the internal own_buf_.  These overloads do NOT touch viterbi_done_/fwdbwd_done_
// and are intended for callers that manage their own buffer lifetime (e.g. a
// per-thread DpBuffer in a batch worker).  The scalar results (score, log_z,
// best_i/j/tbl) are always written to the Aligner regardless of which buffer is used.
//
// AlignBand::Full (default): full (m+1)×(n+1) DP, no restriction.
//
// AlignBand::GuideBanded: band of half-width `band` centered on a reference alignment.
//   guide_j[i] = the column the guide visits after consuming i chars of a.
//   If guide_j is omitted (or empty), a proportional diagonal guide
//   guide_j[i] = round(i * n / m) is used.  Unlike a literal guide_j[i] = i, this
//   tracks the main diagonal of the rectangular DP, so it stays centred for
//   unequal-length sequences without requiring band >= |m-n|.
//   For a custom guide, compute it with guide_j_from_aligned(a_aligned, b_aligned);
//   the endpoint (m, n) is always in-band as long as the guide is a complete alignment.
//
// In GuideBanded mode the full (m+1)×(n+1) table is still allocated (peak memory is
// unchanged), but the per-call init fill and the DP/backward loops are all restricted
// to the band, so the work per re-alignment is O(m·band) rather than O(m·n).  Only the
// band region (plus a one-column margin and adjacent-row overlap, see band_row_span)
// is initialised to NEG_INF; band-edge reads therefore always land on an initialised
// cell and cannot corrupt max/lse operations with stale data from a previous problem.

template<GapModel GM, AlignMode AM, AlignBand AB = AlignBand::Full, class T = double>
struct Aligner {

    // Viterbi-precision buffer type.  T is the scalar precision of the Viterbi /
    // hard-gradient DP: double by default, or float32 for ~2x SIMD lanes and half
    // the table footprint.  This member alias deliberately *shadows* the global
    // ::DpBuffer (= DpBufferT<double>) throughout the class body, so every DpBuffer&
    // signature below already means "the buffer at this Aligner's precision" with no
    // per-site edit.  For T=double it resolves to exactly the old type, which is what
    // keeps the double build byte-for-byte unchanged.  The forward-backward / soft
    // tables inside DpBufferT<T> are double regardless of T.
    using DpBuffer = DpBufferT<T>;

    // ── Public pipeline API — own internal buffer ─────────────────────────────

    // Convenience: take characters, validate and encode them against the params'
    // alphabet into aligner-owned buffers.  An out-of-alphabet character throws.
    //
    // Callers that align the same sequences repeatedly should encode once and
    // use the span overload below — SeqPair does exactly that, so a banded
    // re-alignment under a new matrix costs no re-encoding.
    void set_problem(std::string_view a, std::string_view b,
                     const AlignParams& params,
                     int band = 0,
                     std::vector<int> guide_j = {}) {
        const Alphabet& alpha = params.matrix.alphabet();
        a_own_ = alpha.encode(a);
        b_own_ = alpha.encode(b);
        set_problem(std::span<const uint8_t>(a_own_),
                    std::span<const uint8_t>(b_own_),
                    params, band, std::move(guide_j));
    }

    // Sequences arrive already encoded to alphabet indices — the DP never sees a
    // char.  Encoding (and validation) happens once, at the boundary, in SeqPair
    // / BatchAligner.  The spans must outlive the DP calls that follow.
    void set_problem(std::span<const uint8_t> a, std::span<const uint8_t> b,
                     const AlignParams& params,
                     int band = 0,
                     std::vector<int> guide_j = {}) {
        // Invalidate first: a guide rejected below must not leave the previous
        // problem's results readable against this problem's sequences.
        problem_set_      = false;
        viterbi_done_     = false;
        fwdbwd_done_      = false;
        any_viterbi_done_ = false;
        any_fwdbwd_done_  = false;
        fwdbwd_is_newest_ = false;
        a_idx_  = a;
        b_idx_  = b;
        params_ = &params;
        blk_    = params.matrix.data();
        nalpha_ = params.matrix.size();
        // The Viterbi-precision copy of the substitution block.  T=double aliases the
        // master (no copy); float32 converts once per problem into blkT_storage_.
        if constexpr (std::is_same_v<T, double>) {
            blkT_ = blk_;
        } else {
            const std::size_t nn = static_cast<std::size_t>(nalpha_) * nalpha_;
            blkT_storage_.resize(nn);
            for (std::size_t k = 0; k < nn; ++k)
                blkT_storage_[k] = static_cast<T>(blk_[k]);
            blkT_ = blkT_storage_.data();
        }
        band_   = band;
        m_      = static_cast<int>(a.size());
        n_      = static_cast<int>(b.size());
        stride_ = static_cast<size_t>(n_ + 1);
        sz_     = static_cast<size_t>(m_ + 1) * stride_;
        if constexpr (AB == AlignBand::GuideBanded) {
            if (band_ < 0)
                throw std::invalid_argument(
                    "nwgrad: band must be >= 0, got " + std::to_string(band_));
            if (guide_j.empty()) {
                guide_j_.resize(static_cast<size_t>(m_ + 1));
                if (m_ > 0)
                    for (int i = 0; i <= m_; ++i)
                        guide_j_[i] = static_cast<int>(
                            std::llround(static_cast<double>(i) * n_ / m_));
                else
                    guide_j_[0] = 0;
            } else {
                // The band helpers index guide_j_[0..m] unchecked, so a supplied guide
                // must describe THIS problem: one entry per row, each a column of B,
                // never moving left.  guide_j_from_aligned() checks only the aligned
                // strings' own syntax, not that they spell this a and b.
                validate_guide(guide_j);
                guide_j_ = std::move(guide_j);
            }
        }
        problem_set_      = true;
    }

    // Allocate an empty buffer shell. Must be called once before compute_viterbi()
    // / compute_forward_back() with own buffer. Buffers grow implicitly as needed.
    void alloc_buf() {
        own_buf_allocated_ = true;
    }

    // Select the DP kernel.  Scalar (the default) and Simd write bit-identical
    // tables; Simd is the vectorized one.  Affects compute_viterbi() only —
    // compute_forward_back() is shared and ignores this.
    // Set the Viterbi backend: kBackendAuto (defer to the global default), kBackendScalar
    // (scalar fallback), or a SimdLevel index.  simd_levels.hpp::parse_backend() turns the
    // string vocabulary ("scalar_fallback"/"auto"/"sse2"/"avx2"/...) into this int.
    void set_kernel(int backend) noexcept { backend_ = backend; }
    // How the traceback recovers predecessors — see TracebackMode.  Defaults to
    // Pointers (smaller and faster); set Scores when you need VM/VX/VY to survive
    // the fill, e.g. to compare tables across kernels.
    // Default resolves to this Aligner's compile-time kDefaultTb (Hirschberg for the
    // affine+global+full case, Pointers elsewhere) — so a caller can pass the sentinel
    // through without knowing the problem type.
    void set_traceback(TracebackMode t) noexcept {
        tb_auto_ = (t == TracebackMode::Default);
        tb_ = tb_auto_ ? kDefaultTb : t;
    }
    TracebackMode traceback() const noexcept { return tb_; }
    // Rows per block at which the Hirschberg recursion stops splitting and solves
    // outright — see hb_cutoff_.  Unlike set_traceback this is safe to change between
    // runs: it alters how the DP divides, not what it retains.  It is a THREE-way knob
    // (base-case memory, per-block overhead, tie-break fidelity), not just a speed one:
    // at cutoff >= m no split happens at all and the result is the Pointers fill, hence
    // bit-exact — the memory win is what pays for the divergence.
    void set_hb_cutoff(int rows) {
        if (rows < 1)
            throw std::invalid_argument("nwgrad: hb_cutoff must be >= 1");
        hb_cutoff_ = rows;
    }
    int hb_cutoff() const noexcept { return hb_cutoff_; }
    int  kernel() const noexcept { return backend_; }
    // Which simd fill the Full band uses at double precision: the striped kernel (false,
    // the default) or the row-wise one the guide-banded path uses (true).  Both write
    // tables bit-identical to the scalar fill, so this is a speed knob only.  Row-wise
    // pays no lazy-F fixpoint, which wins on short pairs (miRNA x site, ~22 x 50:
    // 1.5-1.8x, more with cheap gaps); striped wins on long ones.  Row-wise keeps
    // VM/VX/VY (24 B/cell) even under traceback Pointers, which it cannot record.
    void set_rowwise_full(bool on) noexcept { rowwise_full_ = on; }
    bool rowwise_full() const noexcept { return rowwise_full_; }

    void compute_viterbi() {
        check_problem();
        check_own_buf_allocated();
        ensure_viterbi_ptruf(own_buf_);  // Grow if needed
        run_viterbi(own_buf_);
        viterbi_done_ = any_viterbi_done_ = true;
        fwdbwd_is_newest_ = false;
    }

    void compute_forward_back() {
        check_problem();
        check_own_buf_allocated();
        run_fwdbwd(own_buf_);
        fwdbwd_done_ = any_fwdbwd_done_ = true;
        fwdbwd_is_newest_ = true;
    }

    // Returns the score from whichever DP ran most recently (Viterbi score or log Z),
    // regardless of which buffer was used.
    double score() const {
        if (fwdbwd_is_newest_) return log_z_;
        if (any_viterbi_done_) return viterbi_score_;
        throw std::logic_error(
            "nwgrad: call compute_viterbi() or compute_forward_back() before score()");
    }

    double log_z() const {
        if (!any_fwdbwd_done_)
            throw std::logic_error("nwgrad: call compute_forward_back() before log_z()");
        return log_z_;
    }

    std::vector<std::pair<int,int>> alignment() const {
        check_viterbi();
        if (hirschberg_) return alignment_hb();
        if constexpr (GM == GapModel::Linear) return traceback_linear(own_buf_);
        else                                   return traceback_affine(own_buf_);
    }

    std::vector<int> guide_j_from_viterbi() const {
        check_viterbi();
        return guide_j_from_viterbi(own_buf_);
    }

    // Reconstruct the gapped aligned sequences (a, b) from the Viterbi traceback.
    // Each string uses '-' for gaps and the two are of equal length.  Works for
    // both global (full-length) and local (matched sub-region only) alignment.
    std::pair<std::string, std::string> aligned() const {
        check_viterbi();
        return aligned(own_buf_);
    }

    void hard_grad(AlignParams& grad) const {
        check_viterbi();
        hard_grad(own_buf_, grad);
    }

    void soft_grad(AlignParams& grad) const {
        check_fwdbwd();
        soft_grad(own_buf_, grad);
    }

    // Release own DP table memory and reset computed-state flags.
    // The buffer shell remains allocated; set_problem() must be called for next DP run.
    void free_dp() noexcept {
        own_buf_.clear();
        problem_set_     = false;
        viterbi_done_    = false;
        fwdbwd_done_     = false;
        any_viterbi_done_ = false;
        any_fwdbwd_done_ = false;
        fwdbwd_is_newest_ = false;
    }

    // ── Extended API — caller-supplied DpBuffer ───────────────────────────────
    //
    // set_problem() must have been called first.
    // viterbi_done_ / fwdbwd_done_ are NOT touched; callers manage their own state.
    // Scalar results (viterbi_score_, log_z_, best_i_, best_j_, best_tbl_) are
    // always written to the Aligner so score() / log_z() work after these calls.

    void compute_viterbi(DpBuffer& buf) {
        check_problem();
        ensure_viterbi_ptruf(buf);  // Grow if needed (allows implicit growth from size 0)
        run_viterbi(buf);
        any_viterbi_done_ = true;
        fwdbwd_is_newest_ = false;
    }

    // soft_guide="posterior": the next forward-backward also records, per row of A, the
    // column of greatest posterior mass; posterior_guide() turns those rows into a guide
    // (non-decreasing, in [0, n]) — a band centre without running a Viterbi.
    void set_posterior_guide(bool on) noexcept { want_post_ = on; }
    static std::vector<int> monotone_guide(const int* rows, int m, int n) {
        std::vector<int> gj(static_cast<size_t>(m) + 1);
        int run = 0;
        for (int i = 0; i <= m; ++i) {
            run = std::max(run, std::min(std::max(rows[i], 0), n));
            gj[static_cast<size_t>(i)] = run;
        }
        return gj;
    }
    std::vector<int> posterior_guide() const { return monotone_guide(post_rows_.data(), m_, n_); }

    // For a GuideBanded inter-pair fill (InterJobT::blo): this problem's rows as lane
    // `lane` of W — per row i (0..M) the computed columns [jlo, jhi] (empty past m) into
    // blo/bhi[i*W + lane], the initialised span (band_row_span, clipped to columns >= 1)
    // into slo/shi[i] — min/max-merged with the other lanes' — and the Global border
    // extents.  set_problem() first.  Full: nothing to do.
    void banded_lane_rows(int W, int lane, int M, int* blo, int* bhi, int* slo, int* shi,
                          int& bri, int& brj) const {
        if constexpr (AB == AlignBand::GuideBanded) {
            for (int i = 1; i <= M; ++i) {
                const size_t k = static_cast<size_t>(i) * W + lane;
                if (i > m_) { blo[k] = 1; bhi[k] = 0; continue; }
                blo[k] = jlo(i); bhi[k] = jhi(i);
                int lo, hi; band_row_span(i, lo, hi);
                slo[i] = std::min(slo[i], std::max(1, lo));
                shi[i] = std::max(shi[i], hi);
            }
            bri = border_rows(); brj = border_cols();
        }
    }

    // In place of compute_viterbi(buf): adopt lane `lane` of an inter-pair fill
    // (InterJob, inter_kernel_impl.inl) that left W pairs' tables interleaved in
    // buf.VM/VX/VY, this aligner's pair being that lane.  The tables are bit-identical
    // to this pair's own affine Full fill, so score(), the traceback and hard_grad()
    // behave exactly as after compute_viterbi(buf).  Local passes the lane's best cell
    // as the kernel found it; Global reads the score at (m, n).  set_problem() first.
    // `stride`: the group's row length (n+1 of its longest B) when B lengths are ragged
    // (InterJobT::nb); 0 = this problem's own n+1.
    void adopt_interleaved(const DpBuffer& buf, int W, int lane,
                           double local_best, int best_i, int best_j, size_t stride = 0) {
        check_problem();
        inter_stride_ = stride ? stride : stride_;
        tables_striped_ = false;
        pointers_       = false;
        hirschberg_     = false;
        inter_w_ = W; inter_lane_ = lane;
        if constexpr (GM == GapModel::Linear) {
            // Linear: H in buf.H's slot of the interleaved fill (InterJob.linear).
            if constexpr (AM == AlignMode::Local) {
                viterbi_score_ = static_cast<T>(local_best);
                best_i_ = best_i; best_j_ = best_j;
            } else {
                viterbi_score_ = rat(buf.H, m_, n_);
            }
        } else if constexpr (AM == AlignMode::Local) {
            viterbi_score_ = static_cast<T>(local_best);
            best_i_ = best_i; best_j_ = best_j; best_tbl_ = TBTable::M;
            if (best_i > 0) {
                const T mv = rat(buf.VM, best_i, best_j), xv = rat(buf.VX, best_i, best_j),
                        yv = rat(buf.VY, best_i, best_j);
                if      (mv >= xv && mv >= yv) best_tbl_ = TBTable::M;
                else if (xv >= yv)             best_tbl_ = TBTable::X;
                else                            best_tbl_ = TBTable::Y;
            }
        } else {
            const T vm = rat(buf.VM, m_, n_), vx = rat(buf.VX, m_, n_), vy = rat(buf.VY, m_, n_);
            viterbi_score_ = std::max({vm, vx, vy});
            best_i_ = m_; best_j_ = n_;
            if      (vm >= vx && vm >= vy) best_tbl_ = TBTable::M;
            else if (vx >= vy)             best_tbl_ = TBTable::X;
            else                            best_tbl_ = TBTable::Y;
        }
        any_viterbi_done_ = true;
        fwdbwd_is_newest_ = false;
    }

    void compute_forward_back(DpBuffer& buf) {
        check_problem();
        run_fwdbwd(buf);
        any_fwdbwd_done_ = true;
        fwdbwd_is_newest_ = true;
    }

    std::vector<int> guide_j_from_viterbi(const DpBuffer& buf) const {
        // Hirschberg holds the path itself, so there is nothing to walk back — replay it.
        if (hirschberg_) return guide_j_affine_hb();   // the move list is gap-model free
        if (pointers_) {
            if constexpr (GM == GapModel::Affine) return guide_j_affine_ptr(buf);
        }
        if constexpr (GM == GapModel::Linear) return guide_j_linear(buf);
        else                                   return guide_j_affine(buf);
    }

    // The cell the alignment ends at, (i, j) in the DP: (m, n) for Global, the best cell
    // for Local.  Every traceback walks back from it, so with the aligned strings it
    // fixes where the alignment starts too (SeqPair::coordinates).
    std::pair<int, int> alignment_end() const {
        if constexpr (AM == AlignMode::Global) return {m_, n_};   // not every fill records it
        else                                   return {best_i_, best_j_};
    }

    std::pair<std::string, std::string> aligned(const DpBuffer& buf) const {
        std::string a, b;
        if (hirschberg_) { aligned_hb(a, b); return {std::move(a), std::move(b)}; }
        if constexpr (GM == GapModel::Linear) aligned_linear(buf, a, b);
        else                                   aligned_affine(buf, a, b);
        return {std::move(a), std::move(b)};
    }

    // guide_j_from_viterbi(buf) and hard_grad(buf, grad) in one traceback walk where
    // the path is read from score tables (affine, not Pointers or Hirschberg); two walks
    // otherwise.  Same guide, same gradient either way.
    void hard_grad_and_guide(const DpBuffer& buf, AlignParams& grad, std::vector<int>& gj) const {
        if constexpr (GM == GapModel::Affine) {
            if (!hirschberg_ && !pointers_) { hard_grad_affine(buf, grad, &gj); return; }
        }
        gj = guide_j_from_viterbi(buf);
        hard_grad(buf, grad);
    }

    void hard_grad(const DpBuffer& buf, AlignParams& grad) const {
        if constexpr (GM == GapModel::Linear)
            if (hirschberg_) { hard_grad_linear_hb(grad); return; }
        if constexpr (GM == GapModel::Affine) {
            if (hirschberg_) { hard_grad_affine_hb(grad); return; }
            if (pointers_)   { hard_grad_affine_ptr(buf, grad); return; }
        }
        if constexpr (GM == GapModel::Linear) hard_grad_linear(buf, grad);
        else                                   hard_grad_affine(buf, grad);
    }

    // Introspection / testing: copy a DP table into canonical (m+1)×(n+1) row-major
    // order, reading through the current layout (de-stripes when the simd Full kernel
    // left it striped).  Used by the bit-exactness test to compare across layouts.
    template <class V>
    std::vector<typename V::value_type> to_row_major(const V& t) const {
        using E = typename V::value_type;
        const size_t rm_stride = static_cast<size_t>(n_) + 1;
        // Pointers leaves the score tables unallocated; say so rather than read
        // past the end of an empty vector.
        if (t.size() < cell_index(m_, n_) + 1)
            throw std::logic_error(
                "nwgrad: DP table is empty — TracebackMode::Pointers retains predecessor "
                "codes, not score tables; use set_traceback(TracebackMode::Scores) to "
                "inspect VM/VX/VY");
        std::vector<E> out(static_cast<size_t>(m_ + 1) * rm_stride);
        for (int i = 0; i <= m_; ++i)
            for (int j = 0; j <= n_; ++j)
                out[static_cast<size_t>(i) * rm_stride + j] = rat(t, i, j);
        return out;
    }

    void soft_grad(const DpBuffer& buf, AlignParams& grad) const {
        if (soft_counts_ready_) { add_scaled_counts(grad); return; }
        if constexpr (GM == GapModel::Linear) soft_grad_linear(buf, grad);
        else                                   soft_grad_affine(buf, grad);
    }

    // See SoftImpl.  Takes effect at the next compute_forward_back().
    void set_soft_impl(SoftImpl s) noexcept { soft_impl_ = s; }
    SoftImpl soft_impl() const noexcept { return soft_impl_; }
    // Whether the most recent compute_forward_back() ran the scaled path (false: log).
    bool soft_scaled_used() const noexcept { return soft_scaled_used_; }

    // Soft temperature T > 0 (default 1).  The soft path then evaluates the smoothed
    // score T·log Z(θ/T) — log_z() returns that — and its gradient with respect to θ,
    // which is the expected counts under θ/T.  T → 0 recovers the Viterbi score and
    // (on tie-free inputs) the hard counts; T = 1 is the plain log Z.
    void set_soft_temperature(double t) {
        if (!(t > 0.0 && t <= std::numeric_limits<double>::max()))
            throw std::invalid_argument("nwgrad: soft temperature must be finite and > 0, got " +
                                        std::to_string(t));
        soft_temp_ = t;
    }
    double soft_temperature() const noexcept { return soft_temp_; }

private:
    static constexpr double NEG_INF = -std::numeric_limits<double>::infinity();

    // ── Problem state ─────────────────────────────────────────────────────────
    std::span<const uint8_t> a_idx_, b_idx_;   // alphabet indices, not characters
    // Backing store for the string_view overload of set_problem(); empty when
    // the caller supplied encoded spans directly.
    std::vector<uint8_t> a_own_, b_own_;
    const AlignParams*  params_     = nullptr;
    const double*       blk_        = nullptr; // params_->matrix.data(), cached (double master)
    // The substitution block in the Viterbi precision T.  For T=double it aliases blk_
    // (zero copy); for float32 it points into blkT_storage_.  subT() reads it; sub()
    // keeps reading the double master, so the soft path is untouched by T.
    const T*            blkT_       = nullptr;
    std::vector<T>      blkT_storage_;
    int                 nalpha_     = 0;       // params_->matrix.size(), cached
    int                 band_       = 0;
    int                 m_ = 0, n_ = 0;
    size_t              stride_ = 0, sz_ = 0;
    std::vector<int>    guide_j_;  // used only when AB == GuideBanded

    // ── Striped table layout ──────────────────────────────────────────────────
    // The leveled simd Viterbi (affine Full) writes VM/VX/VY in Farrar striped layout
    // to skip a per-row de-stripe copy.  When tables_striped_ is set, rat/at index those
    // tables striped (cell_index); every row-major fill clears it.  striped_seg_ and
    // striped_w_ come straight from the kernel (ViterbiJob.seg / .width); a striped row
    // is striped_seg_*striped_w_ + 1 doubles, slot 0 being column 0.
    bool   tables_striped_ = false;
    bool   rowwise_full_   = false;   // see set_rowwise_full()
    // Set by adopt_interleaved(): VM/VX/VY hold inter_w_ pairs' tables interleaved per
    // cell and this aligner's pair is lane inter_lane_.  0 = not interleaved; every fill
    // of this aligner's own clears it (run_viterbi).
    bool             want_post_ = false;   // soft_guide="posterior": fill post_rows_
    std::vector<int> post_rows_;
    int    inter_w_ = 0, inter_lane_ = 0;
    size_t inter_stride_ = 0;   // the interleaved tables' row length (>= stride_ if ragged)
    // Pointers mode: DM/DX/DY hold predecessor codes, VM/VX/VY are NOT retained.
    bool        pointers_ = false;
    // Hirschberg mode: no tables at all survive the fill.  The recursion recovers the
    // path directly, so hops_ IS the result — every consumer (gradient, alignment,
    // guide_j) replays it instead of walking a table.  One byte per step, so a path
    // is O(m+n) where a table would be O(m·n).
    bool        hirschberg_ = false;
    std::vector<unsigned char> hops_;   // 0=M diagonal, 1=X gap-in-b, 2=Y gap-in-a
    // Local Hirschberg only: the 0-based cell where the recovered local path STARTS.
    // hops_ is a path within the sub-rectangle A[hb_start_i_..ie) x B[hb_start_j_..je),
    // so every consumer replays it from this origin rather than (0, 0).  For Global (and
    // any non-local mode) both are 0, so the consumers are byte-for-byte unchanged there.
    int         hb_start_i_ = 0, hb_start_j_ = 0;
    // Rows per block at which the recursion stops splitting and runs the Pointers fill
    // instead.  1 would be pure Hirschberg (minimum memory, maximum re-sweeping); a
    // block of `cutoff` rows costs cutoff*n direction bytes and saves a level of
    // recursion.  512 from a fleet sweep AFTER hb_base was vectorized (sse2/avx2/avx512/
    // neon × len 500..8000): with a vectorized base case the optimum jumped from 32 to
    // ~512, because a bigger base case is now cheap and it means fewer 2x-work splits.
    // 512 sits in the sweet spot on every host — pairs <= 512 don't split, so they run
    // the exact Pointers fill at full speed (0.99-1.04x pointers, bit-exact); longer
    // pairs split and win 1.4-2.7x at high thread counts by staying out of the memory
    // wall.  A much larger cutoff (2048) re-enters that wall on long pairs.
    int         hb_cutoff_ = 512;
    // The Aligner's OWN default, resolved at compile time from the template case: the
    // affine+global+full specialization is the one Hirschberg implements, so it defaults
    // there; every other case keeps Pointers (which the linear/banded paths ignore
    // anyway).  A default-constructed Aligner therefore never lands on a mode that would
    // throw.  The TracebackMode::Default sentinel maps to exactly this.
    //
    // The Hirschberg case defaults FURTHER, on the scalar type: at T=float the carry is
    // the closed-form prefix max, at T=double the exact serial chain.  That split is a
    // deliberate judgment about what each precision is FOR.  float32 is the throughput
    // mode — it has already accepted ~8.4e-3 of its own rounding against a double oracle,
    // and the ramp's worst measured contribution on top of that is 4.9e-4, two orders of
    // magnitude below the error the caller has already agreed to pay, in exchange for
    // 1.5-3.9x on homologous data.  T=double is chosen BY people who want exactness, so
    // it keeps the carry that has it: there the ramp would be the largest error term in
    // the computation (~9.1e-13 against the exact mode's ~1.6e-12) rather than a rounding
    // lost in the noise.  Either default is overridable per problem.
    //
    // LOCAL affine Full at T=float defaults to HirschbergPmax too (2026-10-05), since its
    // endpoint scans took the prefix-max carry: measured (nighthaven, AVX2, float32,
    // 1000-3000 aa, 1 / 12 threads) 1.6x / 1.5-2.3x Pointers on 30%-mutated homologues
    // and 2.8-3.6x / 3.7-5x on unrelated pairs, identical totals; pairs <= hb_cutoff run
    // AS Pointers.  Local double stays Pointers (exact Local Hirschberg loses there; pmax
    // at double is opt-in by the rule above).  Linear and banded resolve to Pointers.
    static constexpr TracebackMode kDefaultTb =
        (GM == GapModel::Affine && AB == AlignBand::Full && AM == AlignMode::Global)
            ? (std::is_same_v<T, float> ? TracebackMode::HirschbergPmax
                                        : TracebackMode::Hirschberg)
        : (GM == GapModel::Affine && AB == AlignBand::Full && std::is_same_v<T, float>)
            ? TracebackMode::HirschbergPmax
            : TracebackMode::Pointers;
    TracebackMode tb_ = kDefaultTb;
    bool          tb_auto_ = true;   // tb_ came from "auto" (see linear_ptr_fill)
    int    striped_seg_    = 0;
    int    striped_w_      = 1;

    int backend_ = kBackendAuto;  // which Viterbi backend fills the tables (default: auto)

    bool problem_set_       = false;
    bool own_buf_allocated_ = false; // own buffer shell has been allocated
    bool viterbi_done_      = false; // own_buf_ has valid viterbi data
    bool fwdbwd_done_       = false; // own_buf_ has valid fwdbwd data
    bool any_viterbi_done_  = false; // viterbi scalar results are valid (either buffer)
    bool any_fwdbwd_done_   = false; // fwdbwd scalar results are valid (either buffer)
    bool fwdbwd_is_newest_  = false; // true if forward-back ran more recently than viterbi

    // ── Scalar results (written by every compute_viterbi / compute_forward_back)
    double viterbi_score_ = 0.0;
    double log_z_         = 0.0;
    int    best_i_ = 0, best_j_ = 0;

    // ── Scaled soft path state ────────────────────────────────────────────────
    SoftImpl soft_impl_ = SoftImpl::Scaled;
    bool     soft_scaled_used_ = false;
    bool     soft_counts_ready_ = false;   // scnt_/sg_* hold this run's gradient
    double   soft_temp_ = 1.0;
    // exp(substitution block), and the expected counts the scaled backward pass
    // accumulated (matrix: +count; gaps: -count, i.e. already the gradient's sign).
    std::vector<double> es_, scnt_, srow_, srs_;
    std::vector<int> sband_;   // per-row band ranges handed to the soft kernel
    double sg_go_a_ = 0, sg_ge_a_ = 0, sg_go_b_ = 0, sg_ge_b_ = 0;
    enum class TBTable { M, X, Y };
    mutable TBTable best_tbl_ = TBTable::M;

    // ── Own DP buffer (used by the no-arg public API) ─────────────────────────
    DpBuffer own_buf_;

    // ── Precondition checks ───────────────────────────────────────────────────
    void check_problem() const {
        if (!problem_set_)
            throw std::logic_error("nwgrad: call set_problem() first");
    }
    void check_viterbi() const {
        check_problem();
        if (!viterbi_done_)
            throw std::logic_error("nwgrad: call compute_viterbi() first");
    }
    void check_fwdbwd() const {
        check_problem();
        if (!fwdbwd_done_)
            throw std::logic_error("nwgrad: call compute_forward_back() first");
    }

    // Throw if own buffer shell has never been allocated.
    void check_own_buf_allocated() const {
        if (!own_buf_allocated_)
            throw std::logic_error(
                "nwgrad: own DP buffer not allocated; call alloc_buf() before compute_viterbi() / compute_forward_back()");
    }

    // Throw if external viterbi buffer has never been allocated (size is 0).
    // After initial allocation, buffers may grow implicitly as needed.
    void check_external_viterbi_ptruf(const DpBuffer& buf) const {
        bool allocated;
        if constexpr (GM == GapModel::Linear)
            allocated = (buf.H.size() > 0);
        else
            allocated = (buf.VM.size() > 0);
        if (!allocated) {
            std::string gap_model_str = (GM == GapModel::Linear) ? "linear" : "affine";
            throw std::logic_error(
                "nwgrad: external viterbi DP buffer not allocated; provide a pre-allocated buffer or allocate with buf.H.resize(sz) before compute_viterbi(buf)");
        }
    }

    // Throw if external forward-backward buffer has never been allocated (size is 0).
    // After initial allocation, buffers may grow implicitly as needed.
    void check_external_fwdbwd_buf(const DpBuffer& buf) const {
        bool allocated;
        if constexpr (GM == GapModel::Linear)
            allocated = (buf.F.size() > 0);
        else
            allocated = (buf.FM.size() > 0);
        if (!allocated) {
            std::string gap_model_str = (GM == GapModel::Linear) ? "linear" : "affine";
            throw std::logic_error(
                "nwgrad: external forward-backward DP buffer not allocated; provide a pre-allocated buffer or allocate with buf.F.resize(sz) before compute_forward_back(buf)");
        }
    }

    // Whether this run actually selects rowwise score tables. The preference
    // alone is insufficient: splitting Hirschberg, float32 and scalar kernels
    // ignore it. Share the predicate with dispatch so those paths keep their
    // smaller memory bounds even when rowwise was requested.
    bool uses_rowwise_full() const {
        if constexpr (GM == GapModel::Affine && AB == AlignBand::Full &&
                      std::is_same_v<T, double>) {
            if (!rowwise_full_ || (is_hirschberg(tb_) && m_ > hb_cutoff_)) return false;
            const int backend = (backend_ == kBackendAuto) ? global_default_backend() : backend_;
            return backend >= 0 && level_kernels(backend).banded_row_local;
        }
        return false;
    }

    // Grow external buffer to fit current problem (thread-owned path).
    // Linear Full: whether the fill keeps direction bytes (viterbi_linear_ptr) rather than
    // H.  Explicit "pointers" (and Hirschberg's short pairs) always do.  Under "auto" —
    // which resolves to pointers for linear — only a pair whose H table exceeds 2 MiB
    // does: the pointer fill is ~1.35x slower per thread, but past L2 the H table's
    // traffic dominates.  Measured (nighthaven, AVX2, double, Global, 12 threads): H
    // wins 1.6x up to L=350 (1 MB), ties at L=500 (2 MB), loses 2.4x at 700 and 3.5x
    // at 1000.  Same path either way, so this is a speed choice only.
    static constexpr size_t kLinearHTableBytes = size_t(2) << 20;
    bool linear_ptr_fill() const noexcept {
        if constexpr (GM != GapModel::Linear || AB != AlignBand::Full) return false;
        else {
            if (is_hirschberg(tb_)) return true;
            if (tb_ != TracebackMode::Pointers) return false;
            return !tb_auto_ || sz_ * sizeof(T) > kLinearHTableBytes;
        }
    }

    void ensure_viterbi_ptruf(DpBuffer& buf) const {
        if constexpr (GM == GapModel::Linear) {
            // The pointer / Hirschberg fills keep no H (they allocate their own).
            if (linear_ptr_fill()) return;
            if (buf.H.size() < sz_) buf.H.resize(sz_);
        } else {
            // Pointers mode never touches VM/VX/VY — it keeps two rolling rows and byte
            // codes — so allocating them here would hand back the entire footprint
            // saving before the kernel ever runs.  Measured: skipping this is the
            // difference between B costing MORE memory than A and costing ~4x less.
            // Hirschberg allocates even less: its scratch is O(n) rows sized on demand
            // inside the recursion, and the whole point is that no O(m*n) table exists.
            if constexpr (AB == AlignBand::Full)
                if ((tb_ == TracebackMode::Pointers || is_hirschberg(tb_)) && !uses_rowwise_full()) return;
            for (auto* v : {&buf.VM, &buf.VX, &buf.VY}) if (v->size() < sz_) v->resize(sz_);
        }
    }

    void ensure_fwdbwd_buf(DpBuffer& buf) const {
        if constexpr (GM == GapModel::Linear) {
            // F and B checked separately: the scaled path grows F alone.
            if (buf.F.size() < sz_) buf.F.resize(sz_);
            if (buf.B.size() < sz_) buf.B.resize(sz_);
        } else {
            if (buf.FM.size() < sz_ || buf.BM.size() < sz_) {
                buf.FM.resize(sz_); buf.FX.resize(sz_); buf.FY.resize(sz_);
                buf.BM.resize(sz_); buf.BX.resize(sz_); buf.BY.resize(sz_);
            }
        }
    }

    // ── Cell accessors ────────────────────────────────────────────────────────
    // Every table access in the DP, the tracebacks and the gradient goes through these,
    // so switching VM/VX/VY between row-major and striped is a change to cell_index alone.
    // Striped layout MUST match the kernel in kernels_impl.inl: row size striped_seg_*
    // striped_w_ + 1, slot 0 = column 0, column j (1..n) at 1 + ((j-1)%seg)*W + (j-1)/seg.
    __attribute__((always_inline)) size_t cell_index(int i, int j) const noexcept {
        if (inter_w_) {
            // Inter-pair fill (adopt_interleaved): W pairs' tables interleaved per cell.
            return (static_cast<size_t>(i) * inter_stride_ + static_cast<size_t>(j)) *
                   static_cast<size_t>(inter_w_) + static_cast<size_t>(inter_lane_);
        }
        if (tables_striped_) {
            // rowsz = (seg+1)*W: slot 0 is column 0, slots [W, W+seg*W) are the striped
            // columns 1..n (started at W so every W-wide access lands on an aligned
            // address, given the 64-byte-aligned allocator), slots 1..W-1 pad.  Column j
            // is at W + ((j-1)%seg)*W + (j-1)/seg.  Must match kernels_impl.inl exactly.
            const size_t W = static_cast<size_t>(striped_w_);
            const size_t rowsz = (static_cast<size_t>(striped_seg_) + 1) * W;
            if (j == 0) return static_cast<size_t>(i) * rowsz;
            const int jj = j - 1;
            return static_cast<size_t>(i) * rowsz + W +
                   static_cast<size_t>(jj % striped_seg_) * W +
                   static_cast<size_t>(jj / striped_seg_);
        }
        return static_cast<size_t>(i) * stride_ + static_cast<size_t>(j);
    }
    // Element-generic so they serve both the T-typed Viterbi tables (VM/VX/VY/H) and the
    // always-double soft tables (F/B/FM..BY) from one definition; the layout switch in
    // cell_index() is identical for either element type.
    template <class V> typename V::value_type& at(V& t, int i, int j) const {
        return t[cell_index(i, j)];
    }
    template <class V> typename V::value_type rat(const V& t, int i, int j) const {
        return t[cell_index(i, j)];
    }
    // The forward-backward tables (F/B, FM..BY) are ALWAYS row-major, whatever layout
    // the Viterbi fill left VM/VX/VY or DM/DX/DY in.  They get their own accessors so
    // the soft path never has to touch tables_striped_ / pointers_: those describe the
    // retained Viterbi state, which a traceback after compute_forward_back() still
    // reads (SeqPair's soft mode runs Viterbi, then forward-backward, then aligned()).
    // Clearing them here once made Pointers walk unallocated score tables and a
    // striped Scores walk read its tables row-major.
    double& sat(DVec& t, int i, int j) const noexcept {
        return t[static_cast<size_t>(i) * stride_ + static_cast<size_t>(j)];
    }
    double srat(const DVec& t, int i, int j) const noexcept {
        return t[static_cast<size_t>(i) * stride_ + static_cast<size_t>(j)];
    }

    // ── Substitution lookup (the DP hot path) ─────────────────────────────────
    // Offset of the (a[i-1], b[j-1]) cell in an n_alpha × n_alpha block.  DP
    // coordinates are 1-based, so i-1 / j-1 index the sequences.
    size_t sub_off(int i, int j) const noexcept {
        return static_cast<size_t>(a_idx_[static_cast<size_t>(i) - 1]) *
                   static_cast<size_t>(nalpha_) +
               static_cast<size_t>(b_idx_[static_cast<size_t>(j) - 1]);
    }

    // Substitution score for a[i-1] against b[j-1] (double master; used by the soft path).
    double sub(int i, int j) const noexcept { return blk_[sub_off(i, j)]; }

    // Substitution score in the Viterbi precision T (reads the T-converted block).  The
    // Viterbi fill and its tracebacks use this so the traceback float-equality
    // re-derivation happens in the SAME precision the forward pass rounded in — which is
    // load-bearing for bit-exactness when T=float.  For T=double it equals sub().
    T subT(int i, int j) const noexcept { return blkT_[sub_off(i, j)]; }

    // The characters behind those indices — used only to render alignments.
    char sym_a(int i) const noexcept {
        return params_->matrix.alphabet().symbol_at(
            static_cast<int>(a_idx_[static_cast<size_t>(i) - 1]));
    }
    char sym_b(int j) const noexcept {
        return params_->matrix.alphabet().symbol_at(
            static_cast<int>(b_idx_[static_cast<size_t>(j) - 1]));
    }

    // The gradient's matrix block, checked to be over the same alphabet as the
    // params.  Checked once per grad call, not once per cell.
    double* grad_block(AlignParams& grad) const {
        if (&grad.matrix.alphabet() != &params_->matrix.alphabet())
            throw std::invalid_argument(
                "nwgrad: gradient alphabet \"" + grad.matrix.alphabet().symbols() +
                "\" does not match params alphabet \"" +
                params_->matrix.alphabet().symbols() + "\"");
        return grad.matrix.data();
    }

    // ── Band helpers ──────────────────────────────────────────────────────────
    void validate_guide(const std::vector<int>& gj) const {
        if (gj.size() != static_cast<size_t>(m_) + 1)
            throw std::invalid_argument(
                "nwgrad: guide_j has " + std::to_string(gj.size()) +
                " entries, but sequence A of length " + std::to_string(m_) +
                " needs " + std::to_string(m_ + 1) +
                " (do the aligned strings spell the input sequences?)");
        for (size_t i = 0; i < gj.size(); ++i) {
            if (gj[i] < 0 || gj[i] > n_)
                throw std::invalid_argument(
                    "nwgrad: guide_j[" + std::to_string(i) + "] = " +
                    std::to_string(gj[i]) + " is outside [0, " +
                    std::to_string(n_) + "] for sequence B of length " +
                    std::to_string(n_));
            if (i > 0 && gj[i] < gj[i - 1])
                throw std::invalid_argument(
                    "nwgrad: guide_j decreases at row " + std::to_string(i) +
                    " (" + std::to_string(gj[i - 1]) + " -> " +
                    std::to_string(gj[i]) + ")");
        }
    }

    int jlo(int i) const noexcept {
        if constexpr (AB == AlignBand::Full) return 1;
        else                                  return std::max(1, guide_j_[i] - band_);
    }
    // Global: the last row's band always reaches n.  Every global path ends at (m, n),
    // but a guide may stop short of it — trailing gaps in A consume B residues without
    // appending an entry (guide_j_from_aligned) — and a band that misses (m, n) left the
    // score and the traceback's start reading a cell no fill had written: 0.0 instead of
    // 10.0 for the optimal path itself (band 0, "ACGTACGTAC-------" vs a 17-mer).
    int jhi(int i) const noexcept {
        if constexpr (AB == AlignBand::Full) return n_;
        else {
            if constexpr (AM == AlignMode::Global) if (i == m_) return n_;
            int hi = guide_j_[i] + band_;
            if (i < m_) hi = std::max(hi, guide_j_[i + 1] - 1 + band_);
            return std::min(n_, hi);
        }
    }
    int jlo0(int i) const noexcept {
        if constexpr (AB == AlignBand::Full) return 0;
        else                                  return std::max(0, guide_j_[i] - band_);
    }
    int jhi0(int i) const noexcept {
        if constexpr (AB == AlignBand::Full) return n_;
        else {
            if constexpr (AM == AlignMode::Global) if (i == m_) return n_;   // see jhi
            int hi = guide_j_[i] + band_;
            if (i < m_) hi = std::max(hi, guide_j_[i + 1] - 1 + band_);
            return std::min(n_, hi);
        }
    }

    int border_rows() const noexcept {
        if constexpr (AB == AlignBand::Full) return m_;
        else {
            int bi = 0;
            while (bi < m_ && guide_j_[bi + 1] <= band_) ++bi;
            return bi;
        }
    }
    int border_cols() const noexcept {
        if constexpr (AB == AlignBand::Full) return n_;
        else                                  return jhi(0);
    }

    // Inclusive column window [lo, hi] that must be initialised for row i in
    // GuideBanded mode.  It is the union of row i's band with its neighbours'
    // bands, widened by one column on each side.  The neighbour union covers the
    // cells row i±1 read from / write into row i (the recurrence reads row i-1,
    // the backward pass writes into row i-1); the one-column margin covers the
    // NEG_INF guard cell just left of each band edge, so every band-edge read
    // lands on an initialised cell rather than stale data from a previous problem.
    void band_row_span(int i, int& lo, int& hi) const noexcept {
        lo = jlo0(i); hi = jhi0(i);
        if (i > 0)  { lo = std::min(lo, jlo0(i - 1)); hi = std::max(hi, jhi0(i - 1)); }
        if (i < m_) { lo = std::min(lo, jlo0(i + 1)); hi = std::max(hi, jhi0(i + 1)); }
        lo = std::max(0,  lo - 1);
        hi = std::min(n_, hi + 1);
    }

    // GuideBanded: NEG_INF-fill only the band region (O(m·band)).  Full: no-op —
    // the linear forward tables need no pre-fill (every in-band cell is written).
    template <class V> void banded_fill(V& vec) const {
        if constexpr (AB == AlignBand::GuideBanded) {
            for (int i = 0; i <= m_; ++i) {
                int lo, hi; band_row_span(i, lo, hi);
                auto* row = vec.data() + static_cast<size_t>(i) * stride_;
                std::fill(row + lo, row + hi + 1, NEG_INF);
            }
        }
    }

    // Fill `vec` with `value`.  Full: the whole (m+1)×(n+1) table.  GuideBanded:
    // only the band region (O(m·band)).  Element-generic; `value` is passed as double
    // and std::fill converts it to the table's element type (NEG_INF stays -inf).
    template <class V> void band_fill(V& vec, double value) const {
        if constexpr (AB == AlignBand::Full) {
            std::fill(vec.begin(), vec.begin() + static_cast<ptrdiff_t>(sz_),
                      static_cast<typename V::value_type>(value));
        } else {
            for (int i = 0; i <= m_; ++i) {
                int lo, hi; band_row_span(i, lo, hi);
                auto* row = vec.data() + static_cast<size_t>(i) * stride_;
                std::fill(row + lo, row + hi + 1,
                          static_cast<typename V::value_type>(value));
            }
        }
    }

    // ── guide_j extraction helpers ────────────────────────────────────────────

    static void fill_guide_gaps(std::vector<int>& gj) {
        int n = static_cast<int>(gj.size());
        int lo = 0;
        while (lo < n) {
            if (gj[lo] < 0) { ++lo; continue; }
            int hi = lo + 1;
            while (hi < n && gj[hi] < 0) ++hi;
            if (hi == n) {
                for (int k = lo + 1; k < n; ++k) gj[k] = gj[lo];
                break;
            }
            for (int k = lo + 1; k < hi; ++k)
                gj[k] = gj[lo] + (gj[hi] - gj[lo]) * (k - lo) / (hi - lo);
            lo = hi;
        }
    }

    // The four linear walkers (guide_j_linear, traceback_linear, aligned_linear,
    // hard_grad_linear) re-derive each step by exact float equality, like the affine
    // ones — but unlike them they had no border guard.  Column 0 is seeded closed-form,
    // H[i][0] = -i*ge_b, which is NOT bit-equal to the stepwise H[i-1][0] - ge_b when
    // ge_b is not representable (0.1: some i differ by an ULP).  The gap-in-B test then
    // failed on the border, the walk fell through to --j, and read H[i][-1] — off the
    // row: out of bounds in a release build, an assertion under _GLIBCXX_ASSERTIONS.
    // On column 0 only an upward move remains, so `j == 0` now takes it outright, as the
    // affine walkers' "col 0: only upward moves remain" does.  (Row 0 was already safe:
    // both earlier branches need i > 0, so the else takes --j.)
    std::vector<int> guide_j_linear(const DpBuffer& buf) const {
        std::vector<int> gj(static_cast<size_t>(m_ + 1), -1);
        gj[0] = 0;
        gj[static_cast<size_t>(m_)] = n_;
        const int i0 = (AM == AlignMode::Global) ? m_ : best_i_;
        gj[static_cast<size_t>(i0)] = (AM == AlignMode::Global) ? n_ : best_j_;
        walk_linear(buf, [&](int k, int i, int j) {
            if (k == 0)      gj[static_cast<size_t>(i - 1)] = j - 1;
            else if (k == 1) gj[static_cast<size_t>(i - 1)] = j;   // gap in a: no update
        });
        fill_guide_gaps(gj);
        return gj;
    }

    // The one walk behind the four linear walkers: emit(k, i, j) for each move out of
    // (i, j) — k = 0 diagonal, 1 up (gap in B), 2 left (gap in A) — from the optimum
    // back to the start.  The move is read from the recorded code (pointers_, see
    // viterbi_linear_ptr) or re-derived by exact equality on H, in the same order the
    // fill's max chose it, so both give the same walk, ties included.
    template <class Emit>
    void walk_linear(const DpBuffer& buf, Emit&& emit) const {
        int i = (AM == AlignMode::Global) ? m_ : best_i_;
        int j = (AM == AlignMode::Global) ? n_ : best_j_;
        while (true) {
            if (i == 0 && j == 0) break;
            int k;
            if (pointers_) {
                k = dcode(buf.DM, i, j);
                if (k == 3) break;
            } else {
                if constexpr (AM == AlignMode::Local)
                    if (rat(buf.H, i, j) <= 0.0) break;
                if (i > 0 && j > 0 &&
                    rat(buf.H, i, j) == rat(buf.H, i-1, j-1) + subT(i, j))
                    k = 0;
                else if (i > 0 && (j == 0 || rat(buf.H, i, j) == rat(buf.H, i-1, j) -
                                                 static_cast<T>(params_->gap_extend_b)))
                    k = 1;
                else
                    k = 2;
            }
            emit(k, i, j);
            if (k == 0)      { --i; --j; }
            else if (k == 1) --i;
            else             --j;
        }
    }

    // ── variant-B tracebacks: read the recorded predecessor, do not re-derive ──
    //
    // These mirror guide_j_affine()/hard_grad_affine() step for step; the only
    // change is where the predecessor comes from.  The forward pass recorded it
    // with the same `>=` M>X>Y chain those functions run, so the walk is identical
    // — including at ties, which is the property Hirschberg could not offer.
    // Directions live in whatever layout the fill used -- row-major from the scalar
    // B fill, striped from the leveled B kernel -- so index them through the same
    // cell_index() the score tables use.  One copy of the striping arithmetic, not two.
    unsigned char dcode(const BVec& d, int i, int j) const noexcept {
        return d[cell_index(i, j)];
    }

    std::vector<int> guide_j_affine_ptr(const DpBuffer& buf) const {
        std::vector<int> gj(static_cast<size_t>(m_ + 1), -1);
        gj[0] = 0;
        gj[static_cast<size_t>(m_)] = n_;
        int i = best_i_, j = best_j_;
        TBTable tbl = best_tbl_;
        gj[static_cast<size_t>(i)] = j;
        while (true) {
            if (i == 0 && j == 0) break;
            if constexpr (AM == AlignMode::Local)
                if (tbl == TBTable::M && dcode(buf.DM, i, j) == 3) break;
            // Same border guard as the score-table walks — see the note there.
            if      (i == 0) tbl = TBTable::Y;
            else if (j == 0) tbl = TBTable::X;
            unsigned char c;
            if (tbl == TBTable::M) {
                c = dcode(buf.DM, i, j); --i; --j; gj[static_cast<size_t>(i)] = j;
            } else if (tbl == TBTable::X) {
                c = dcode(buf.DX, i, j); --i; gj[static_cast<size_t>(i)] = j;
            } else {
                c = dcode(buf.DY, i, j); --j;
            }
            tbl = (c == 1) ? TBTable::X : (c == 2 ? TBTable::Y : TBTable::M);
        }
        fill_guide_gaps(gj);
        return gj;
    }

    void hard_grad_affine_ptr(const DpBuffer& buf, AlignParams& grad) const {
        double* gblk = grad_block(grad);
        int i = best_i_, j = best_j_;
        TBTable tbl = best_tbl_;
        while (true) {
            if (i == 0 && j == 0) break;
            if constexpr (AM == AlignMode::Local)
                if (tbl == TBTable::M && dcode(buf.DM, i, j) == 3) break;
            // Same border guard as the score-table walks — see the note there.
            if      (i == 0) tbl = TBTable::Y;
            else if (j == 0) tbl = TBTable::X;
            unsigned char c;
            if (tbl == TBTable::M) {
                gblk[sub_off(i, j)] += 1.0;
                c = dcode(buf.DM, i, j); --i; --j;
                tbl = (c == 1) ? TBTable::X : (c == 2 ? TBTable::Y : TBTable::M);
            } else if (tbl == TBTable::X) {
                grad.gap_extend_b -= 1.0;
                c = dcode(buf.DX, i, j); --i;
                const TBTable prev = (c == 1) ? TBTable::X : (c == 2 ? TBTable::Y : TBTable::M);
                if (prev != TBTable::X) grad.gap_open_b -= 1.0;
                tbl = prev;
            } else {
                grad.gap_extend_a -= 1.0;
                c = dcode(buf.DY, i, j); --j;
                const TBTable prev = (c == 1) ? TBTable::X : (c == 2 ? TBTable::Y : TBTable::M);
                if (prev != TBTable::Y) grad.gap_open_a -= 1.0;
                tbl = prev;
            }
        }
    }

    std::vector<int> guide_j_affine(const DpBuffer& buf) const {
        std::vector<int> gj(static_cast<size_t>(m_ + 1), -1);
        gj[0] = 0;
        gj[static_cast<size_t>(m_)] = n_;

        int i = best_i_, j = best_j_;
        TBTable tbl = best_tbl_;
        gj[static_cast<size_t>(i)] = j;
        const T go_a = static_cast<T>(params_->gap_open_a);
        const T ge_a = static_cast<T>(params_->gap_extend_a);
        const T go_b = static_cast<T>(params_->gap_open_b);
        const T ge_b = static_cast<T>(params_->gap_extend_b);

        while (true) {
            if (i == 0 && j == 0) break;
            if constexpr (AM == AlignMode::Local)
                if (tbl == TBTable::M && rat(buf.VM, i, j) <= 0.0) break;
            // GuideBanded with a band narrower than the path needs: the band can fail
            // to admit any route back to the origin, leaving every predecessor at
            // -inf.  The M>=X>=Y tie-break then picks M, and M steps DIAGONALLY — so
            // i or j goes negative, cell_index() turns that into a huge size_t, and
            // the walk reads off the end of the table.  That was a segfault, not a
            // suboptimal score (realign_banded: "A"*22 vs "C"*87 at band<=32).
            // At a border only one move is legal; forcing it is what a well-formed
            // DP would have chosen anyway, so valid alignments are unaffected.
            if      (i == 0) tbl = TBTable::Y;   // row 0: only leftward moves remain
            else if (j == 0) tbl = TBTable::X;   // col 0: only upward moves remain

            if (tbl == TBTable::M) {
                T vm = rat(buf.VM,i-1,j-1), vx = rat(buf.VX,i-1,j-1), vy = rat(buf.VY,i-1,j-1);
                --i; --j;
                gj[static_cast<size_t>(i)] = j;
                if      (vm >= vx && vm >= vy) tbl = TBTable::M;
                else if (vx >= vy)             tbl = TBTable::X;
                else                            tbl = TBTable::Y;
            } else if (tbl == TBTable::X) {
                // X state: gap in B (advance i), uses gap_b params
                T fm = rat(buf.VM,i-1,j) - go_b - ge_b;
                T fx = rat(buf.VX,i-1,j)        - ge_b;
                T fy = rat(buf.VY,i-1,j) - go_b - ge_b;
                --i;
                gj[static_cast<size_t>(i)] = j;
                if      (fm >= fx && fm >= fy) tbl = TBTable::M;
                else if (fx >= fy)             tbl = TBTable::X;
                else                            tbl = TBTable::Y;
            } else {
                // Y state: gap in A (advance j), uses gap_a params
                T fm = rat(buf.VM,i,j-1) - go_a - ge_a;
                T fx = rat(buf.VX,i,j-1) - go_a - ge_a;
                T fy = rat(buf.VY,i,j-1)        - ge_a;
                --j;  // gap in a: i stays, no gj update
                if      (fm >= fx && fm >= fy) tbl = TBTable::M;
                else if (fx >= fy)             tbl = TBTable::X;
                else                            tbl = TBTable::Y;
            }
        }
        fill_guide_gaps(gj);
        return gj;
    }

    // ── Backend dispatch ──────────────────────────────────────────────────────
    //
    // The one place in the library where the scalar/simd choice exists.  The branch is
    // taken once per DP, not once per cell.  Every simd level writes tables bit-identical
    // to the scalar one, so nothing downstream can tell them apart.  The linear model has
    // no PER-PAIR simd kernel — deliberately.  Its recurrence collapses to a single carry
    // that is a pure latency chain along the row (~6 cycles/cell), and the scalar loop
    // already runs at ~9 cycles/cell against that floor; a row-wise vectorized linear
    // kernel was written, measured at 0.90x, and deleted.  So any simd backend is a legal
    // request for a linear aligner and runs the scalar fill here.  (Batches of short pairs
    // vectorize linear ACROSS pairs instead — SeqPairBatch fill="interpair", one pair per
    // lane, bit-identical — which sidesteps the chain.)
    void run_viterbi(DpBuffer& buf) {
        // Default to row-major; only the striped Full kernel (run_dispatched_affine) flips
        // this back on.  Every other fill here writes VM/VX/VY row-major.
        tables_striped_ = false;
        pointers_     = false;
        hirschberg_   = false;
        inter_w_      = 0;
        // Hirschberg: divide and conquer, no tables above the base case.  It is a
        // DIFFERENT algorithm with a different tie-break, not a faster spelling of the
        // same one, so an unsupported combination throws instead of falling back — a
        // silent substitution would make a benchmark of it meaningless and a gradient
        // from it unattributable.
        if (is_hirschberg(tb_)) {
            if constexpr (AB != AlignBand::Full)
                throw std::logic_error(
                    "nwgrad: traceback=\"hirschberg\" is implemented for full DP only; a "
                    "guide band already bounds memory, which is the only thing Hirschberg buys");
            else if (m_ > hb_cutoff_) {
                // Linear: one exact sweep for both "hirschberg" and "hirschberg_pmax" —
                // the prefix-max carry exists to remove affine's lazy-F fixpoint, and the
                // scalar linear sweep has none.
                if constexpr (GM == GapModel::Linear) { viterbi_linear_hirschberg(buf); return; }
                else if constexpr (AM == AlignMode::Local) { viterbi_affine_hirschberg_local(buf); return; }
                else { viterbi_affine_hirschberg(buf); return; }
            }
            // A pair no longer than hb_cutoff never splits, so it is run AS Pointers —
            // the same fill, table and traceback, hence bit-exact with traceback=
            // "pointers" by construction.  It used to run the Hirschberg base case
            // instead, which is the Pointers fill in spirit but not in arithmetic: its
            // borders are seeded and carried differently, so with a non-representable
            // gap_extend (0.1) it could settle a float tie on a path one ULP worse, and
            // it replayed the score from the path rather than reading the table — 27/41
            // short pairs differed from Pointers.  Same memory (the base case kept
            // a Pointers-sized table too) and the same speed (measured 0.99-1.04x).
        }
        // Variant B: direction pointers + rolling rows, 3 B/cell instead of 12/24.
        // Full affine only; the banded and linear paths keep their score tables.
        // (the pointer kernels are chosen after the backend is resolved, below: every
        // simd level has its own, and only the scalar path falls back here.)
        if constexpr (GM == GapModel::Linear) {
            // No simd per-pair linear kernel: every backend is scalar here.  The pointer
            // fill keeps direction bytes, not H, where linear_ptr_fill() says so.
            if (linear_ptr_fill()) { viterbi_linear_ptr(buf); return; }
            viterbi_linear(buf);
            return;
        }
        // Resolve this aligner's backend: auto -> the global default, then scalar or a level.
        const int backend = (backend_ == kBackendAuto) ? global_default_backend() : backend_;
        // Pointers is Full-affine only; banded/linear always keep score tables.
        // (a Hirschberg mode reaches here only for a short pair — see above)
        const bool use_ptr = (tb_ == TracebackMode::Pointers || is_hirschberg(tb_)) &&
                             (AB == AlignBand::Full);
        if (backend == kBackendScalar) {
            if (use_ptr) viterbi_affine_ptr(buf); else viterbi_affine(buf);
            return;
        }

        // A simd level: its kernels.  If that level's TU was not linked (header-only
        // single-level build), every pointer is null and we fall through to scalar.
        const LevelKernels& K = level_kernels(backend);
        if constexpr (AB == AlignBand::Full) {
            // Opt-in row-wise fill (set_rowwise_full): score tables, row-major, read back
            // by the Scores traceback whatever tb_ says — the paths are identical.
            if constexpr (std::is_same_v<T, double>) {
                if (uses_rowwise_full()) { viterbi_affine_simd(buf, K); return; }
            }
            // The striped Full kernel exists at both precisions (viterbi / viterbi_f).
            if constexpr (std::is_same_v<T, double>) {
                if (use_ptr && K.viterbi_ptr) { run_dispatched_affine(buf, K, true);  return; }
                if (K.viterbi)             { run_dispatched_affine(buf, K, false); return; }
            } else {
                if (use_ptr && K.viterbi_ptr_f) { run_dispatched_affine(buf, K, true);  return; }
                if (K.viterbi_f)             { run_dispatched_affine(buf, K, false); return; }
            }
            if (use_ptr) { viterbi_affine_ptr(buf); return; }   // level lacks the ptr kernel
        }
        // The row-wise banded kernel is double-only for now; float banded falls back to the
        // (T-templated) scalar fill.  Gated so viterbi_affine_simd — whose body is
        // double-typed — is never instantiated for T=float.
        if constexpr (std::is_same_v<T, double>) {
            if (K.banded_row_local) { viterbi_affine_simd(buf, K); return; }
        }
        viterbi_affine(buf);
    }

    // Build a plain ViterbiJob from Aligner state, run the dispatched (leveled) kernel,
    // and copy its scalar results back.  The kernel fills buf.VM/VX/VY in STRIPED layout
    // (table_layout == 1) and reports seg/width; rat/at then read them striped.
    void run_dispatched_affine(DpBuffer& buf, const LevelKernels& K, bool use_ptr = false) {
        ViterbiJob<T> job{};
        job.a = a_idx_.data(); job.m = m_;
        job.b = b_idx_.data(); job.n = n_;
        // blkT_ is the substitution block already in the Viterbi precision T (= blk_ when
        // T is double); the penalties convert to T on assignment.
        job.blk = blkT_;       job.nalpha = nalpha_;
        job.go_a = static_cast<T>(params_->gap_open_a);   job.ge_a = static_cast<T>(params_->gap_extend_a);
        job.go_b = static_cast<T>(params_->gap_open_b);   job.ge_b = static_cast<T>(params_->gap_extend_b);
        job.align_mode = (AM == AlignMode::Local) ? 1 : 0;
        job.align_band = 0; job.band = 0;
        job.guide_j = nullptr; job.guide_len = 0;
        job.buf = &buf;
        if constexpr (std::is_same_v<T, double>) { if (use_ptr) K.viterbi_ptr(job); else K.viterbi(job); }
        else                                     { if (use_ptr) K.viterbi_ptr_f(job); else K.viterbi_f(job); }
        pointers_ = use_ptr;   // Pointers leaves codes, not score tables
        // Adopt the layout the kernel produced (striped for the Full striped kernel).
        if (job.table_layout == 1) {
            tables_striped_ = true;
            striped_seg_ = job.seg; striped_w_ = job.width;
        } else {
            tables_striped_ = false;
        }
        viterbi_score_ = job.score;
        best_i_ = job.best_i; best_j_ = job.best_j;
        best_tbl_ = (job.best_tbl == 0) ? TBTable::M
                  : (job.best_tbl == 1) ? TBTable::X : TBTable::Y;
    }

    // ── Simd kernel — defined out-of-line in aligner_simd.hpp ─────────────────
    //
    // Declared here, defined there, and that header is included at the bottom of
    // this one so both are always visible at the point of instantiation.  The
    // vectorization machinery (query profile, lazy-F) stays out of this file
    // entirely; K supplies the row-wise leaf kernels for the active ISA level.
    void viterbi_affine_simd(DpBuffer& buf, const LevelKernels& K);

    // Substitution scores for row i, contiguous in j over [lo, hi] — a vector
    // load, not a gather.  Full mode serves a slice of the prebuilt query
    // profile; banded mode gathers the row's short span into buf.subbuf, since a
    // full-width profile would outcost the banded DP.  Defined in aligner_simd.hpp.
    void build_profile(DpBuffer& buf) const;
    const double* subrow(DpBuffer& buf, int i, int lo, int hi) const;

    // ═════════════════════════════════════════════════════════════════════════
    // Viterbi — Linear gap model
    // ═════════════════════════════════════════════════════════════════════════

    // In precision T throughout, like the affine fill: the walkers below re-derive
    // each step by exact equality against this table, so they must round where it
    // rounded.  It used to add in double (sub(), the raw double gap costs) and store
    // into a float table at T=float; the walkers' double recomputation then rarely
    // matched the stored float for a non-representable cost, every test fell through
    // to "gap in A", and the returned path was not the one scored (-18.7 vs -7.9).
    // At T=double every cast here is the identity: results are unchanged bit for bit.
    void viterbi_linear(DpBuffer& buf) {
        banded_fill(buf.H);

        if constexpr (AM == AlignMode::Global) {
            at(buf.H, 0, 0) = 0.0;
            const int bi = border_rows(), bj = border_cols();
            for (int i = 1; i <= bi; ++i) at(buf.H, i, 0) = -static_cast<T>(i) * static_cast<T>(params_->gap_extend_b);
            for (int j = 1; j <= bj; ++j) at(buf.H, 0, j) = -static_cast<T>(j) * static_cast<T>(params_->gap_extend_a);
        } else {
            for (int i = 0; i <= m_; ++i) at(buf.H, i, 0) = 0.0;
            for (int j = 0; j <= n_; ++j) at(buf.H, 0, j) = 0.0;
        }

        best_i_ = 0; best_j_ = 0;
        double best_local = 0.0;
        for (int i = 1; i <= m_; ++i) {
            for (int j = jlo(i); j <= jhi(i); ++j) {
                T v = std::max({
                    rat(buf.H, i-1, j-1) + subT(i, j),
                    rat(buf.H, i-1, j)   - static_cast<T>(params_->gap_extend_b),  // gap in B (advance i)
                    rat(buf.H, i,   j-1) - static_cast<T>(params_->gap_extend_a),  // gap in A (advance j)
                });
                if constexpr (AM == AlignMode::Local) {
                    v = std::max(v, static_cast<T>(0));
                    if (v > best_local) { best_local = v; best_i_ = i; best_j_ = j; }
                }
                at(buf.H, i, j) = v;
            }
        }

        viterbi_score_ = (AM == AlignMode::Global) ? rat(buf.H, m_, n_) : best_local;
    }

    // TracebackMode::Pointers for linear gaps (Full band): viterbi_linear's fill, in the
    // same precision and the same order, retaining one direction byte per cell (buf.DM,
    // row-major) and two rolling rows instead of H — 1 B/cell against 4 (float32) / 8
    // (double), and no O(m*n) score table at all.  Codes: 0 diagonal, 1 up (gap in B),
    // 2 left (gap in A), 3 stop.  Bit-exact with the H walk, ties included: the max below
    // keeps the FIRST largest of (diag+s, up-ge_b, left-ge_a), exactly as std::max over
    // the initializer list does, and the H walk tests the same three in the same order by
    // equality — so both pick the first largest.  Local: stop where the clamped value is
    // <= 0 (the H walk's stop test); borders stop (Local), go up / left (Global).
    void viterbi_linear_ptr(DpBuffer& buf) {
        const size_t st = stride_;
        const size_t dsz = static_cast<size_t>(m_ + 1) * st;
        if (buf.DM.size() < dsz) buf.DM.resize(dsz);
        if (buf.rM.size() < st) { buf.rM.resize(st); buf.qM.resize(st); }
        T* prev = buf.qM.data(); T* cur = buf.rM.data();
        unsigned char* D = buf.DM.data();
        const T ge_a = static_cast<T>(params_->gap_extend_a);
        const T ge_b = static_cast<T>(params_->gap_extend_b);
        constexpr bool local = AM == AlignMode::Local;
        prev[0] = static_cast<T>(0); D[0] = 3;
        for (int j = 1; j <= n_; ++j) {
            prev[j] = local ? static_cast<T>(0) : -static_cast<T>(j) * ge_a;
            D[j] = local ? 3 : 2;
        }
        int bi = 0, bj = 0;   // locals: best_i_/best_j_ are members (see below)
        const int n = n_;     // ditto: a member bound is reloaded after every byte store
        double best_local = 0.0;
        const unsigned char* __restrict bcode = b_idx_.data();
        for (int i = 1; i <= m_; ++i) {
            unsigned char* __restrict d = D + static_cast<size_t>(i) * st;
            // Locals, not members: the byte stores below may alias anything, so a member
            // read in the loop (subT's blkT_/a_idx_/b_idx_) would be reloaded every cell.
            const T* __restrict prow = blkT_ + static_cast<size_t>(a_idx_[static_cast<size_t>(i) - 1]) * nalpha_;
            const T* __restrict pv = prev;
            T* __restrict cv = cur;
            cv[0] = local ? static_cast<T>(0) : -static_cast<T>(i) * ge_b;
            d[0] = local ? 3 : 1;
            // Two passes, as gcc -O3 splits viterbi_linear's own loop: the carry-free
            // max(diag+s, up-ge_b) and its code vectorize across the row; only
            // max(t, left-ge_a) stays on the serial chain.  One fused loop (the code on
            // the carry) ran 2.6x slower than the H fill; this, ~1.6x (a third pass
            // reading the codes off the values was slower still: 2x).  The values are the
            // same std::max calls in the same order, so bit-exact.
            for (int j = 1; j <= n; ++j) {
                const T d0 = pv[j - 1] + prow[bcode[j - 1]];
                const T u = pv[j] - ge_b;
                cv[j] = std::max(d0, u);
                d[j] = static_cast<unsigned char>(d0 < u);
            }
            T left = cv[0];
            for (int j = 1; j <= n; ++j) {
                const T t = cv[j], l = left - ge_a;
                T v = std::max(t, l);
                const int tl = t < l;
                unsigned char c = static_cast<unsigned char>(d[j] + tl * (2 - d[j]));
                if constexpr (local) {
                    v = std::max(v, static_cast<T>(0));
                    c = (v <= static_cast<T>(0)) ? static_cast<unsigned char>(3) : c;
                    if (v > best_local) { best_local = v; bi = i; bj = j; }
                }
                cv[j] = v; d[j] = c;
                left = v;
            }
            std::swap(prev, cur);
        }
        best_i_ = bi; best_j_ = bj;
        viterbi_score_ = local ? best_local : prev[n_];
        pointers_ = true;
    }

    std::vector<std::pair<int,int>> traceback_linear(const DpBuffer& buf) const {
        std::vector<std::pair<int,int>> path;
        walk_linear(buf, [&](int k, int i, int j) { if (k == 0) path.emplace_back(i - 1, j - 1); });
        std::reverse(path.begin(), path.end());
        return path;
    }

    void aligned_linear(const DpBuffer& buf, std::string& a, std::string& b) const {
        walk_linear(buf, [&](int k, int i, int j) {
            if (k == 0)      { a.push_back(sym_a(i)); b.push_back(sym_b(j)); }
            else if (k == 1) { a.push_back(sym_a(i)); b.push_back('-'); }   // gap in B
            else             { a.push_back('-'); b.push_back(sym_b(j)); }   // gap in A
        });
        std::reverse(a.begin(), a.end());
        std::reverse(b.begin(), b.end());
    }

    void hard_grad_linear(const DpBuffer& buf, AlignParams& grad) const {
        double* gblk = grad_block(grad);
        walk_linear(buf, [&](int k, int i, int j) {
            if (k == 0)      gblk[sub_off(i, j)] += 1.0;
            else if (k == 1) grad.gap_extend_b -= 1.0;   // the score subtracts this penalty
            else             grad.gap_extend_a -= 1.0;
        });
    }

    // ═════════════════════════════════════════════════════════════════════════
    // Viterbi — Affine gap model
    // ═════════════════════════════════════════════════════════════════════════

    // ── TracebackMode::Pointers: affine Viterbi, scalar reference ─────────────
    //
    // Computes the same DP as viterbi_affine() but retains, instead of three score
    // tables, one byte per cell per state naming the predecessor the forward pass
    // chose.  Scores live in two rolling rows, so the retained footprint is
    // 3 B/cell against 12 (float32) / 24 (double).
    //
    // BIT-EXACTNESS is the whole contract, and it holds for a specific reason: the
    // codes below are produced by the *identical* comparison chain the traceback
    // would otherwise re-derive from the tables — `>=` in M, X, Y order.  Change
    // one `>=` to `>` here and B silently returns a different (still valid)
    // subgradient at ties, which is exactly the defect that disqualified Hirschberg.
    //
    // DM additionally carries code 3 = "VM <= 0 at this cell", which is how Local's
    // `rat(VM,i,j) <= 0` stop survives having no VM table to consult.
    //
    // Borders are recorded too, not just the recurrence interior: the traceback
    // walks column 0 and row 0 (X at (1,0) resolves to M, at (i>1,0) to X), and
    // those cells are border-initialised, never visited by the main loop.
    void viterbi_affine_ptr(DpBuffer& buf) {
        const T go_a = static_cast<T>(params_->gap_open_a);
        const T ge_a = static_cast<T>(params_->gap_extend_a);
        const T go_b = static_cast<T>(params_->gap_open_b);
        const T ge_b = static_cast<T>(params_->gap_extend_b);

        const std::size_t stride = stride_;   // row-major, matches cell_index()
        const std::size_t dsz    = static_cast<std::size_t>(m_ + 1) * stride;
        // Per vector, not keyed on DM / rM: viterbi_linear_ptr grows only DM, rM and qM.
        for (auto* d : {&buf.DM, &buf.DX, &buf.DY}) if (d->size() < dsz) d->resize(dsz);
        for (auto* r : {&buf.rM, &buf.rX, &buf.rY, &buf.qM, &buf.qX, &buf.qY})
            if (r->size() < stride) r->resize(stride);
        T* pM = buf.qM.data(); T* pX = buf.qX.data(); T* pY = buf.qY.data();  // row i-1
        T* cM = buf.rM.data(); T* cX = buf.rX.data(); T* cY = buf.rY.data();  // row i
        unsigned char* dM = buf.DM.data();
        unsigned char* dX = buf.DX.data();
        unsigned char* dY = buf.DY.data();

        // Same three-way argmax the traceback runs, in one place so the two cannot drift.
        auto pick = [](T vm, T vx, T vy) -> unsigned char {
            if (vm >= vx && vm >= vy) return 0;
            return (vx >= vy) ? 1 : 2;
        };

        // ── row 0 ────────────────────────────────────────────────────────────
        for (std::size_t j = 0; j <= static_cast<std::size_t>(n_); ++j) {
            if constexpr (AM == AlignMode::Local) { pM[j] = 0; pX[j] = NEG_INF; pY[j] = NEG_INF; }
            else {
                pM[j] = (j == 0) ? static_cast<T>(0) : NEG_INF;
                pX[j] = NEG_INF;
                pY[j] = (j == 0) ? NEG_INF : -(go_a + static_cast<T>(j) * ge_a);
            }
            dM[j] = (AM == AlignMode::Local || j == 0) ? 3 : 3;   // row 0 is a start
            dX[j] = 0;
            // Y along row 0 extends leftward: (0,1) opens from M(0,0), the rest extend Y.
            dY[j] = (j <= 1) ? 0 : 2;
        }
        if constexpr (AM == AlignMode::Global) dM[0] = 3;

        best_i_ = 0; best_j_ = 0; best_tbl_ = TBTable::M;
        T best_local = static_cast<T>(0);

        for (int i = 1; i <= m_; ++i) {
            const std::size_t rb = static_cast<std::size_t>(i) * stride;
            // column 0 of this row
            if constexpr (AM == AlignMode::Local) { cM[0] = 0; cX[0] = NEG_INF; }
            else { cM[0] = NEG_INF; cX[0] = -(go_b + static_cast<T>(i) * ge_b); }
            cY[0] = NEG_INF;
            dM[rb] = 3;
            dX[rb] = (i <= 1) ? 0 : 1;    // (1,0) opens from M(0,0); deeper rows extend X
            dY[rb] = 0;

            for (int j = 1; j <= n_; ++j) {
                const std::size_t k = static_cast<std::size_t>(j);
                const T dgM = pM[k-1], dgX = pX[k-1], dgY = pY[k-1];
                T m_val = std::max({dgM, dgX, dgY}) + subT(i, j);
                unsigned char km = pick(dgM, dgX, dgY);

                const T ux = pM[k] - go_b - ge_b, vx = pX[k] - ge_b, wx = pY[k] - go_b - ge_b;
                const T x_val = std::max({ux, vx, wx});
                const unsigned char kx = pick(ux, vx, wx);

                const T uy = cM[k-1] - go_a - ge_a, vy = cX[k-1] - go_a - ge_a, wy = cY[k-1] - ge_a;
                const T y_val = std::max({uy, vy, wy});
                const unsigned char ky = pick(uy, vy, wy);

                if constexpr (AM == AlignMode::Local) {
                    m_val = std::max(m_val, static_cast<T>(0));
                    const T best_here = std::max({m_val, x_val, y_val});
                    if (best_here > best_local) {
                        best_local = best_here; best_i_ = i; best_j_ = j;
                        if      (m_val >= x_val && m_val >= y_val) best_tbl_ = TBTable::M;
                        else if (x_val >= y_val)                    best_tbl_ = TBTable::X;
                        else                                         best_tbl_ = TBTable::Y;
                    }
                    if (m_val <= static_cast<T>(0)) km = 3;   // traceback stops here
                }

                cM[k] = m_val; cX[k] = x_val; cY[k] = y_val;
                dM[rb + k] = km; dX[rb + k] = kx; dY[rb + k] = ky;
            }
            std::swap(pM, cM); std::swap(pX, cX); std::swap(pY, cY);
        }

        if constexpr (AM == AlignMode::Global) {
            const std::size_t k = static_cast<std::size_t>(n_);
            const T vm = pM[k], vx = pX[k], vy = pY[k];       // pM is row m_ after the swap
            viterbi_score_ = std::max({vm, vx, vy});
            best_i_ = m_; best_j_ = n_;
            if      (vm >= vx && vm >= vy) best_tbl_ = TBTable::M;
            else if (vx >= vy)             best_tbl_ = TBTable::X;
            else                            best_tbl_ = TBTable::Y;
        } else {
            viterbi_score_ = best_local;
        }
        pointers_ = true;
    }

    // ── Hirschberg / Myers-Miller: linear space, no retained tables ───────────
    //
    // Everything below works on a SUB-RECTANGLE of the DP: rows a[i0..i1), columns
    // b[j0..j1), in the sequences' own 0-based coordinates.  Two boundary flags carry
    // the affine state across a cut, and they are the whole reason this is Myers-Miller
    // rather than plain Hirschberg:
    //
    //   in_x   the path is ALREADY inside a gap-in-b run when it enters this block, so
    //          the block's first X move extends (-ge_b) instead of opening (-go_b-ge_b).
    //   out_x  the path is still inside such a run when it leaves.
    //
    // Only X can straddle a row cut.  A Y move stays within its row, and the cut is
    // between rows, so a Y run is wholly on one side of it — which is why there is no
    // in_y/out_y.  Getting this wrong does not crash; it silently charges one extra
    // gap open per crossing, so the score comes out slightly low.  The score assertion
    // in the tests is what actually pins it.
    //
    // A block is entered in one of two states (in_x), so the DP's origin is seeded at
    // whichever of M/X is legal: M[0][0]=0 for a fresh arrival, X[0][0]=0 to continue a
    // run.  That single seed is the entire mechanism.

    // Grow a scratch row vector to hold `n` elements.
    static void hb_fit(typename DpBufferT<T>::TVec& v, std::size_t n) {
        if (v.size() < n) v.resize(n);
    }

    // The leveled striped sweep for this aligner's backend, or null to go scalar.
    // Resolved per sweep rather than cached: it is one predicated load against an
    // O(H*ncols) sweep, and caching it would have to be invalidated on set_kernel().
    // True when this aligner's sweep uses the closed-form prefix-max carry rather than
    // the exact serial one.  Only the SWEEP differs: the recursion, the join, the base
    // case and row 0 are shared, which is why this is a predicate and not a second driver.
    bool hb_pmax() const noexcept { return tb_ == TracebackMode::HirschbergPmax; }

    const LevelKernels* hb_level() const noexcept {
        const int backend = (backend_ == kBackendAuto) ? global_default_backend() : backend_;
        if (backend == kBackendScalar) return nullptr;
        const LevelKernels& K = level_kernels(backend);
        const bool dbl = std::is_same_v<T, double>;
        const bool have = hb_pmax() ? (dbl ? K.hb_sweep_pmax != nullptr : K.hb_sweep_pmax_f != nullptr)
                                    : (dbl ? K.hb_sweep      != nullptr : K.hb_sweep_f      != nullptr);
        return have ? &K : nullptr;
    }

    // Run one half-sweep through the leveled kernel.  Forward and reverse differ only in
    // the walk direction, so they are the same call with the steps negated.
    void hb_run_level(const LevelKernels& K, DpBuffer& buf, int a_start, int a_step,
                      int b_start, int b_step, int H, int ncols, bool in_x,
                      T* oM, T* oX, T* oY) const {
        HbJob<T> job{};
        job.a = a_idx_.data(); job.b = b_idx_.data();
        job.blk = blkT_; job.nalpha = nalpha_;
        job.go_a = static_cast<T>(params_->gap_open_a);
        job.ge_a = static_cast<T>(params_->gap_extend_a);
        job.go_b = static_cast<T>(params_->gap_open_b);
        job.ge_b = static_cast<T>(params_->gap_extend_b);
        job.a_start = a_start; job.a_step = a_step;
        job.b_start = b_start; job.b_step = b_step;
        job.H = H; job.ncols = ncols; job.in_x = in_x ? 1 : 0;
        job.buf = &buf;
        job.outM = oM; job.outX = oX; job.outY = oY;
        if constexpr (std::is_same_v<T, double>) {
            if (hb_pmax()) K.hb_sweep_pmax(job); else K.hb_sweep(job);
        } else {
            if (hb_pmax()) K.hb_sweep_pmax_f(job); else K.hb_sweep_f(job);
        }
    }

    // Forward sweep of rows (i0, i1], keeping two rolling rows.  On return the three
    // out-pointers hold row i1 of the block, indexed by column offset c = j - j0.
    // Costs (i1-i0)*(j1-j0) cells and O(j1-j0) memory: nothing is retained.
    //
    // Pmax = true swaps the VY carry for the closed-form prefix max — see the long note
    // in hb_kernel_impl.inl.  `rmp` then holds the gap ramp k*ge_a for k = 0..(j1-j0),
    // precomputed by the caller.  It is a LOAD and not a multiply on purpose: an
    // adjacent multiply would let the compiler contract `g + k*ge_a` into an FMA, which
    // rounds once instead of twice and would make this reference disagree with the simd
    // kernel on any -mfma level.  Row 0 stays exact in both variants.
    template <bool Pmax = false>
    void hb_fwd(int i0, int i1, int j0, int j1, bool in_x,
                T* pM, T* pX, T* pY, T* cM, T* cX, T* cY,
                T*& oM, T*& oX, T*& oY, const T* rmp = nullptr) const {
        const T go_a = static_cast<T>(params_->gap_open_a);
        const T ge_a = static_cast<T>(params_->gap_extend_a);
        const T go_b = static_cast<T>(params_->gap_open_b);
        const T ge_b = static_cast<T>(params_->gap_extend_b);
        const int W = j1 - j0;

        // Row i0: the block's origin, then Y moves rightward along it.
        pM[0] = in_x ? static_cast<T>(NEG_INF) : static_cast<T>(0);
        pX[0] = in_x ? static_cast<T>(0)       : static_cast<T>(NEG_INF);
        pY[0] = static_cast<T>(NEG_INF);
        for (int c = 1; c <= W; ++c) {
            pM[c] = static_cast<T>(NEG_INF);
            pX[c] = static_cast<T>(NEG_INF);
            pY[c] = std::max({pM[c-1] - go_a - ge_a, pX[c-1] - go_a - ge_a, pY[c-1] - ge_a});
        }

        for (int r = i0 + 1; r <= i1; ++r) {
            cM[0] = static_cast<T>(NEG_INF);
            cX[0] = std::max({pM[0] - go_b - ge_b, pX[0] - ge_b, pY[0] - go_b - ge_b});
            cY[0] = static_cast<T>(NEG_INF);
            // The running prefix max over Q[k] = g[k] + k*ge_a, seeded with the column-0
            // border folded in as k = -1.  Unused (and cold) when Pmax is false.
            T P = cY[0] - ge_a;
            for (int c = 1; c <= W; ++c) {
                // subT is 1-based over the full sequences; row r consumes a[r-1].
                cM[c] = std::max({pM[c-1], pX[c-1], pY[c-1]}) + subT(r, j0 + c);
                cX[c] = std::max({pM[c] - go_b - ge_b, pX[c] - ge_b, pY[c] - go_b - ge_b});
                if constexpr (Pmax) {
                    // g[k] for k = c-1.  Max-then-subtract mirrors the kernel's `ov`
                    // exactly (subtracting a common value is order-preserving, so this is
                    // bit-identical to the three-way form the exact branch uses).
                    const T g = (std::max(cM[c-1], cX[c-1]) - go_a) - ge_a;
                    P = std::max(g + rmp[c-1], P);
                    cY[c] = P - rmp[c-1];
                } else {
                    cY[c] = std::max({cM[c-1] - go_a - ge_a, cX[c-1] - go_a - ge_a, cY[c-1] - ge_a});
                }
            }
            std::swap(pM, cM); std::swap(pX, cX); std::swap(pY, cY);
        }
        oM = pM; oX = pX; oY = pY;      // after the last swap, p holds row i1
    }

    // Reverse sweep of rows [i0, i1), the mirror image of hb_fwd: it aligns the SUFFIX
    // a[r..i1) with b[.. j1), so on return oS[c] is the best score of aligning
    // a[i0..i1) with b[(j0+c)..j1) given the block's FIRST forward move has type S.
    // out_x seeds the far end exactly as in_x seeds the near one.
    template <bool Pmax = false>
    void hb_rev(int i0, int i1, int j0, int j1, bool out_x,
                T* pM, T* pX, T* pY, T* cM, T* cX, T* cY,
                T*& oM, T*& oX, T*& oY, const T* rmp = nullptr) const {
        const T go_a = static_cast<T>(params_->gap_open_a);
        const T ge_a = static_cast<T>(params_->gap_extend_a);
        const T go_b = static_cast<T>(params_->gap_open_b);
        const T ge_b = static_cast<T>(params_->gap_extend_b);
        const int W = j1 - j0;

        // Reversed column offset d = j1 - j, so d grows leftward from the block's end.
        pM[0] = out_x ? static_cast<T>(NEG_INF) : static_cast<T>(0);
        pX[0] = out_x ? static_cast<T>(0)       : static_cast<T>(NEG_INF);
        pY[0] = static_cast<T>(NEG_INF);
        for (int d = 1; d <= W; ++d) {
            pM[d] = static_cast<T>(NEG_INF);
            pX[d] = static_cast<T>(NEG_INF);
            pY[d] = std::max({pM[d-1] - go_a - ge_a, pX[d-1] - go_a - ge_a, pY[d-1] - ge_a});
        }

        for (int r = i1 - 1; r >= i0; --r) {
            cM[0] = static_cast<T>(NEG_INF);
            cX[0] = std::max({pM[0] - go_b - ge_b, pX[0] - ge_b, pY[0] - go_b - ge_b});
            cY[0] = static_cast<T>(NEG_INF);
            T P = cY[0] - ge_a;
            for (int d = 1; d <= W; ++d) {
                // Mirrored diagonal: this step pairs a[r] with b[j1-d], i.e. subT(r+1, j1-d+1).
                cM[d] = std::max({pM[d-1], pX[d-1], pY[d-1]}) + subT(r + 1, j1 - d + 1);
                cX[d] = std::max({pM[d] - go_b - ge_b, pX[d] - ge_b, pY[d] - go_b - ge_b});
                if constexpr (Pmax) {
                    const T g = (std::max(cM[d-1], cX[d-1]) - go_a) - ge_a;
                    P = std::max(g + rmp[d-1], P);
                    cY[d] = P - rmp[d-1];
                } else {
                    cY[d] = std::max({cM[d-1] - go_a - ge_a, cX[d-1] - go_a - ge_a, cY[d-1] - ge_a});
                }
            }
            std::swap(pM, cM); std::swap(pX, cX); std::swap(pY, cY);
        }
        oM = pM; oX = pX; oY = pY;
    }

    // Base case: a block short enough to solve outright.  This is the Pointers fill,
    // restricted to the sub-rectangle and seeded by in_x/out_x, recording one direction
    // byte per cell per state and walking them back.  Appends the block's moves to hops_
    // in forward order.  Memory is 3*(rows+1)*(cols+1) bytes, bounded by hb_cutoff_.
    // The leveled base-case kernel for this backend, or null to go scalar.  Registered
    // together with hb_sweep, but checked separately so a partially-populated level table
    // cannot dispatch one without the other.
    const LevelKernels* hb_base_level() const noexcept {
        const int backend = (backend_ == kBackendAuto) ? global_default_backend() : backend_;
        if (backend == kBackendScalar) return nullptr;
        const LevelKernels& K = level_kernels(backend);
        const bool have = std::is_same_v<T, double> ? (K.hb_base != nullptr)
                                                    : (K.hb_base_f != nullptr);
        return have ? &K : nullptr;
    }

    // Vectorized base case: the striped kernel fills buf.hbD (3 direction planes, striped)
    // and returns seg/width + the final-cell scores; the walk-back below reads them
    // striped.  Bit-identical to the scalar fill, so which one runs is a speed choice.
    void hb_base_striped_run(const LevelKernels& K, DpBuffer& buf,
                             int i0, int i1, int j0, int j1, bool in_x, bool out_x) {
        const int H = i1 - i0, NC = j1 - j0;
        HbBaseJob<T> job{};
        job.a = a_idx_.data(); job.b = b_idx_.data();
        job.blk = blkT_; job.nalpha = nalpha_;
        job.go_a = static_cast<T>(params_->gap_open_a);
        job.ge_a = static_cast<T>(params_->gap_extend_a);
        job.go_b = static_cast<T>(params_->gap_open_b);
        job.ge_b = static_cast<T>(params_->gap_extend_b);
        job.a_start = i0; job.a_step = 1;      // row r=1..H consumes a[i0 + (r-1)]
        job.b_start = j0; job.b_step = 1;      // col c=1..NC pairs   b[j0 + (c-1)]
        job.H = H; job.ncols = NC; job.in_x = in_x ? 1 : 0; job.buf = &buf;
        if constexpr (std::is_same_v<T, double>) K.hb_base(job);
        else                                     K.hb_base_f(job);

        // Striped walk-back over buf.hbD.  Column c lives at slot (c==0 ? 0 : W+scol(c)).
        const int seg = job.seg, Wl = job.width;
        const std::size_t rowsz = (std::size_t)(seg + 1) * Wl;
        const std::size_t off   = (std::size_t)Wl;
        const std::size_t plane = (std::size_t)(H + 1) * rowsz;
        const unsigned char* dM = buf.hbD.data();
        const unsigned char* dX = dM + plane;
        const unsigned char* dY = dX + plane;
        auto idx = [&](int r, int c) -> std::size_t {
            return (std::size_t)r * rowsz + (c == 0 ? 0 : off + (std::size_t)((c - 1) % seg) * Wl + (c - 1) / seg);
        };
        int r = H, c = NC;
        TBTable tbl;
        if (out_x) tbl = TBTable::X;
        else tbl = (job.fM >= job.fX && job.fM >= job.fY) ? TBTable::M
                 : ((job.fX >= job.fY) ? TBTable::X : TBTable::Y);
        const std::size_t first = hops_.size();
        while (r > 0 || c > 0) {
            const std::size_t k = idx(r, c);
            unsigned char code;
            if (tbl == TBTable::M)      { code = dM[k]; hops_.push_back(0); --r; --c; }
            else if (tbl == TBTable::X) { code = dX[k]; hops_.push_back(1); --r; }
            else                        { code = dY[k]; hops_.push_back(2); --c; }
            tbl = (code == 1) ? TBTable::X : (code == 2 ? TBTable::Y : TBTable::M);
        }
        std::reverse(hops_.begin() + static_cast<std::ptrdiff_t>(first), hops_.end());
    }

    void hb_base(DpBuffer& buf, int i0, int i1, int j0, int j1,
                 bool in_x, bool out_x) {
        const T go_a = static_cast<T>(params_->gap_open_a);
        const T ge_a = static_cast<T>(params_->gap_extend_a);
        const T go_b = static_cast<T>(params_->gap_open_b);
        const T ge_b = static_cast<T>(params_->gap_extend_b);
        const int H = i1 - i0, W = j1 - j0;
        if (H == 0 && W == 0) return;
        if (const LevelKernels* K = hb_base_level()) {
            hb_base_striped_run(*K, buf, i0, i1, j0, j1, in_x, out_x);
            return;
        }

        const std::size_t stride = static_cast<std::size_t>(W) + 1;
        const std::size_t plane  = static_cast<std::size_t>(H + 1) * stride;
        hb_fit_b(buf.hbD, 3 * plane);
        unsigned char* dM = buf.hbD.data();
        unsigned char* dX = dM + plane;
        unsigned char* dY = dX + plane;

        // Score rows: two rolling is not enough here because Y reads the current row,
        // but M/X read only the row above — the same shape as the Pointers fill.
        hb_fit(buf.hfa, stride); hb_fit(buf.hfb, stride); hb_fit(buf.hfc, stride);
        hb_fit(buf.hfd, stride); hb_fit(buf.hfe, stride); hb_fit(buf.hff, stride);
        T* pM = buf.hfa.data(); T* pX = buf.hfb.data(); T* pY = buf.hfc.data();
        T* cM = buf.hfd.data(); T* cX = buf.hfe.data(); T* cY = buf.hff.data();

        auto pick = [](T vm, T vx, T vy) -> unsigned char {
            if (vm >= vx && vm >= vy) return 0;
            return (vx >= vy) ? 1 : 2;
        };

        pM[0] = in_x ? static_cast<T>(NEG_INF) : static_cast<T>(0);
        pX[0] = in_x ? static_cast<T>(0)       : static_cast<T>(NEG_INF);
        pY[0] = static_cast<T>(NEG_INF);
        dM[0] = 3; dX[0] = 3; dY[0] = 3;
        for (int c = 1; c <= W; ++c) {
            pM[c] = static_cast<T>(NEG_INF);
            pX[c] = static_cast<T>(NEG_INF);
            const T uy = pM[c-1] - go_a - ge_a, vy = pX[c-1] - go_a - ge_a, wy = pY[c-1] - ge_a;
            pY[c] = std::max({uy, vy, wy});
            dM[c] = 3; dX[c] = 3; dY[static_cast<std::size_t>(c)] = pick(uy, vy, wy);
        }

        for (int r = 1; r <= H; ++r) {
            const std::size_t rb = static_cast<std::size_t>(r) * stride;
            const T ux0 = pM[0] - go_b - ge_b, vx0 = pX[0] - ge_b, wx0 = pY[0] - go_b - ge_b;
            cM[0] = static_cast<T>(NEG_INF);
            cX[0] = std::max({ux0, vx0, wx0});
            cY[0] = static_cast<T>(NEG_INF);
            dM[rb] = 3; dX[rb] = pick(ux0, vx0, wx0); dY[rb] = 3;
            for (int c = 1; c <= W; ++c) {
                const T dgM = pM[c-1], dgX = pX[c-1], dgY = pY[c-1];
                cM[c] = std::max({dgM, dgX, dgY}) + subT(i0 + r, j0 + c);
                const T ux = pM[c] - go_b - ge_b, vx = pX[c] - ge_b, wx = pY[c] - go_b - ge_b;
                cX[c] = std::max({ux, vx, wx});
                const T uy = cM[c-1] - go_a - ge_a, vy = cX[c-1] - go_a - ge_a, wy = cY[c-1] - ge_a;
                cY[c] = std::max({uy, vy, wy});
                dM[rb + c] = pick(dgM, dgX, dgY);
                dX[rb + c] = pick(ux, vx, wx);
                dY[rb + c] = pick(uy, vy, wy);
            }
            std::swap(pM, cM); std::swap(pX, cX); std::swap(pY, cY);
        }

        // Where does the block end?  out_x forces X; otherwise take the best of the three.
        int r = H, c = W;
        TBTable tbl;
        if (out_x) tbl = TBTable::X;
        else {
            const T vm = pM[W], vx = pX[W], vy = pY[W];
            tbl = (vm >= vx && vm >= vy) ? TBTable::M : ((vx >= vy) ? TBTable::X : TBTable::Y);
        }

        // Walk back, emitting moves; reverse at the end since we produce them backwards.
        const std::size_t first = hops_.size();
        while (r > 0 || c > 0) {
            const std::size_t k = static_cast<std::size_t>(r) * stride + static_cast<std::size_t>(c);
            unsigned char code;
            if (tbl == TBTable::M)      { code = dM[k]; hops_.push_back(0); --r; --c; }
            else if (tbl == TBTable::X) { code = dX[k]; hops_.push_back(1); --r; }
            else                        { code = dY[k]; hops_.push_back(2); --c; }
            tbl = (code == 1) ? TBTable::X : (code == 2 ? TBTable::Y : TBTable::M);
        }
        std::reverse(hops_.begin() + static_cast<std::ptrdiff_t>(first), hops_.end());
    }

    static void hb_fit_b(BVec& v, std::size_t n) { if (v.size() < n) v.resize(n); }

    // One half-sweep, simd if this backend has a kernel and scalar otherwise, writing its
    // final row contiguously into (oM,oX,oY).  The two are bit-identical by construction
    // — same operations, same order, same left-association — so which one runs is a
    // speed decision and never a correctness one, exactly as with the Viterbi backends.
    void hb_half(DpBuffer& buf, bool reverse, int i0, int i1, int j0, int j1,
                 bool flag, T* oM, T* oX, T* oY) {
        const int H = i1 - i0, NC = j1 - j0;
        if (const LevelKernels* K = hb_level()) {
            if (reverse) hb_run_level(*K, buf, i1 - 1, -1, j1 - 1, -1, H, NC, flag, oM, oX, oY);
            else         hb_run_level(*K, buf, i0,     +1, j0,     +1, H, NC, flag, oM, oX, oY);
            return;
        }
        T *sM, *sX, *sY;
        if (hb_pmax()) {
            // The ramp, contiguous: rmp[k] = k*ge_a.  Same scalar multiply of the same
            // exactly-representable integer the striped kernel uses for the same absolute
            // column k, so the two agree to the bit — and built here, outside the row
            // loop, so the carry loop below contains no multiply to contract into an FMA.
            const T ge_a = static_cast<T>(params_->gap_extend_a);
            hb_fit(buf.hramp, static_cast<std::size_t>(NC) + 1);
            T* rmp = buf.hramp.data();
            for (int k = 0; k <= NC; ++k) rmp[k] = static_cast<T>(k) * ge_a;
            if (reverse)
                hb_rev<true>(i0, i1, j0, j1, flag,
                             buf.hfa.data(), buf.hfb.data(), buf.hfc.data(),
                             buf.hfd.data(), buf.hfe.data(), buf.hff.data(), sM, sX, sY, rmp);
            else
                hb_fwd<true>(i0, i1, j0, j1, flag,
                             buf.hfa.data(), buf.hfb.data(), buf.hfc.data(),
                             buf.hfd.data(), buf.hfe.data(), buf.hff.data(), sM, sX, sY, rmp);
        } else if (reverse)
            hb_rev(i0, i1, j0, j1, flag,
                   buf.hfa.data(), buf.hfb.data(), buf.hfc.data(),
                   buf.hfd.data(), buf.hfe.data(), buf.hff.data(), sM, sX, sY);
        else
            hb_fwd(i0, i1, j0, j1, flag,
                   buf.hfa.data(), buf.hfb.data(), buf.hfc.data(),
                   buf.hfd.data(), buf.hfe.data(), buf.hff.data(), sM, sX, sY);
        std::copy(sM, sM + NC + 1, oM);
        std::copy(sX, sX + NC + 1, oX);
        std::copy(sY, sY + NC + 1, oY);
    }

    // The recursion.  Splits the block's rows in half, sweeps to the split row from both
    // ends, and joins the two halves to find the column the optimal path crosses at.
    void hb_solve(DpBuffer& buf, int i0, int i1, int j0, int j1, bool in_x, bool out_x) {
        const int H = i1 - i0, W = j1 - j0;
        if (H <= hb_cutoff_) { hb_base(buf, i0, i1, j0, j1, in_x, out_x); return; }

        const int p = i0 + H / 2;
        const std::size_t stride = static_cast<std::size_t>(W) + 1;
        hb_fit(buf.hfa, stride); hb_fit(buf.hfb, stride); hb_fit(buf.hfc, stride);
        hb_fit(buf.hfd, stride); hb_fit(buf.hfe, stride); hb_fit(buf.hff, stride);
        hb_fit(buf.hra, stride); hb_fit(buf.hrb, stride); hb_fit(buf.hrc, stride);
        hb_fit(buf.hrd, stride); hb_fit(buf.hre, stride); hb_fit(buf.hrf, stride);
        hb_fit(buf.hsM, stride); hb_fit(buf.hsX, stride); hb_fit(buf.hsY, stride);

        // Both halves land in their own contiguous output rows — hs* for the forward,
        // hr{a,b,c} for the reverse — so the two never contend for scratch and the
        // simd and scalar paths deliver their results in the same place.
        hb_half(buf, false, i0, p, j0, j1, in_x,
                buf.hsM.data(), buf.hsX.data(), buf.hsY.data());
        const T* FM = buf.hsM.data(); const T* FX = buf.hsX.data(); const T* FY = buf.hsY.data();

        hb_half(buf, true, p, i1, j0, j1, out_x,
                buf.hra.data(), buf.hrb.data(), buf.hrc.data());
        const T* rM = buf.hra.data(); const T* rX = buf.hrb.data(); const T* rY = buf.hrc.data();

        // Join.  For each crossing column c the prefix ends in some state and the suffix
        // begins in some state; every combination is a legal path, so the plain sum over
        // the best of each is admissible.  The one case that is NOT a plain sum is an X
        // run straddling the cut: both halves then charge the open, so one -go_b is
        // refunded.  That refund is strictly positive, so taking the max over both forms
        // picks it automatically wherever it applies.
        const T go_b = static_cast<T>(params_->gap_open_b);
        T best = static_cast<T>(NEG_INF);
        int cstar = 0;
        bool span = false;
        for (int c = 0; c <= W; ++c) {
            const int d = W - c;                       // reverse sweep is indexed leftward
            const T f = std::max({FM[c], FX[c], FY[c]});
            const T g = std::max({rM[d], rX[d], rY[d]});
            const T plain = f + g;
            if (plain > best) { best = plain; cstar = c; span = false; }
            if (FX[c] > static_cast<T>(NEG_INF) && rX[d] > static_cast<T>(NEG_INF)) {
                const T joined = FX[c] + rX[d] + go_b;
                if (joined > best) { best = joined; cstar = c; span = true; }
            }
        }

        const int jstar = j0 + cstar;
        hb_solve(buf, i0, p, j0, jstar, in_x, span);
        hb_solve(buf, p, i1, jstar, j1, span, out_x);
    }

    // Entry point: recover the whole path, then read the score back off it.  The score is
    // recomputed from the path rather than taken from the join, which is a genuine check
    // and not bookkeeping — if the affine boundary handling were wrong the two would
    // disagree, and the tests compare this score against the other tracebacks'.
    void viterbi_affine_hirschberg(DpBuffer& buf) {
        hops_.clear();
        hops_.reserve(static_cast<std::size_t>(m_ + n_));
        hb_solve(buf, 0, m_, 0, n_, false, false);

        // Replay for the score.  Also fixes best_i_/best_j_/best_tbl_, which the shared
        // accessors still expect even though no table backs them here.
        //
        // Accumulate in T, not double.  The DP's own score is the path's prefix sum with
        // a rounding at every step; replaying in double instead rounds nowhere and lands
        // 1e-3 away under T=float — small, but enough to make "does Hirschberg agree with
        // pointers" untestable by exact comparison, which is the one question worth asking
        // of it.  Same precision, same order, same operations: identical paths now give
        // identical scores to the bit, so any difference that remains is a real one.
        const T go_a = static_cast<T>(params_->gap_open_a);
        const T ge_a = static_cast<T>(params_->gap_extend_a);
        const T go_b = static_cast<T>(params_->gap_open_b);
        const T ge_b = static_cast<T>(params_->gap_extend_b);
        T s = static_cast<T>(0);
        int i = 0, j = 0;
        unsigned char prev = 3;                     // 3 = no previous move
        for (unsigned char op : hops_) {
            if (op == 0) { ++i; ++j; s += subT(i, j); }
            else if (op == 1) {
                if (prev != 1) s -= go_b;
                s -= ge_b; ++i;
            } else {
                if (prev != 2) s -= go_a;
                s -= ge_a; ++j;
            }
            prev = op;
        }
        viterbi_score_ = static_cast<double>(s);
        best_i_ = m_; best_j_ = n_;
        best_tbl_ = hops_.empty() ? TBTable::M
                  : (hops_.back() == 1 ? TBTable::X
                     : (hops_.back() == 2 ? TBTable::Y : TBTable::M));
        hirschberg_ = true;
    }

    // ── Local (Smith-Waterman) linear space: endpoint scans + a global box ─────
    //
    // The leveled endpoint-scan kernel for this backend, or null to go scalar — the
    // twin of hb_level()/hb_base_level().
    const LevelKernels* hb_scan_level() const noexcept {
        const int backend = (backend_ == kBackendAuto) ? global_default_backend() : backend_;
        if (backend == kBackendScalar) return nullptr;
        const LevelKernels& K = level_kernels(backend);
        const bool have = std::is_same_v<T, double> ? (K.hb_scan != nullptr)
                                                    : (K.hb_scan_f != nullptr);
        return have ? &K : nullptr;
    }

    // Run one endpoint scan over the sub-rectangle rows [i0, i1), cols [j0, j1): simd if
    // this backend has the kernel, scalar otherwise.  `local` picks clamped-SW (forward
    // end cell) vs unclamped global-suffix (reverse start cell); `reverse` negates the
    // walk.  Reports the best cell in BLOCK-1-based coords (row 1..H, col 1..NC) and its
    // score.  The two paths are bit-identical by construction — same recurrence, same
    // topmost/leftmost tie-break — so which runs is a speed choice, exactly as elsewhere.
    void hb_scan(DpBuffer& buf, bool local, bool reverse, int i0, int i1, int j0, int j1,
                 T& best, int& bi, int& bj) {
        const int H = i1 - i0, NC = j1 - j0;
        if (const LevelKernels* K = hb_scan_level()) {
            HbScanJob<T> job{};
            job.a = a_idx_.data(); job.b = b_idx_.data();
            job.blk = blkT_; job.nalpha = nalpha_;
            job.go_a = static_cast<T>(params_->gap_open_a);
            job.ge_a = static_cast<T>(params_->gap_extend_a);
            job.go_b = static_cast<T>(params_->gap_open_b);
            job.ge_b = static_cast<T>(params_->gap_extend_b);
            if (reverse) { job.a_start = i1 - 1; job.a_step = -1; job.b_start = j1 - 1; job.b_step = -1; }
            else         { job.a_start = i0;     job.a_step = +1; job.b_start = j0;     job.b_step = +1; }
            job.H = H; job.ncols = NC; job.local = local ? 1 : 0;
            job.pmax = hb_pmax() ? 1 : 0;
            job.buf = &buf;
            if constexpr (std::is_same_v<T, double>) K->hb_scan(job);
            else                                     K->hb_scan_f(job);
            best = job.best; bi = job.best_i; bj = job.best_j;
            return;
        }
        hb_scan_scalar(buf, local, reverse, i0, i1, j0, j1, best, bi, bj);
    }

    // Scalar reference endpoint scan.  Rolling rows only; the borders and the M-clamp
    // switch on `local`, the walk direction on `reverse`.  The argmax is topmost-row /
    // leftmost-column, strict > — the SAME tie-break the simd kernel uses, so the two
    // agree to the bit (the property NWGRAD_ISA relies on).
    void hb_scan_scalar(DpBuffer& buf, bool local, bool reverse, int i0, int i1, int j0, int j1,
                        T& best, int& bi, int& bj) {
        const int H = i1 - i0, NC = j1 - j0;
        best = static_cast<T>(0); bi = 0; bj = 0;
        if (H == 0 || NC == 0) return;
        const T go_a = static_cast<T>(params_->gap_open_a);
        const T ge_a = static_cast<T>(params_->gap_extend_a);
        const T go_b = static_cast<T>(params_->gap_open_b);
        const T ge_b = static_cast<T>(params_->gap_extend_b);
        const int a_start = reverse ? i1 - 1 : i0, a_step = reverse ? -1 : +1;
        const int b_start = reverse ? j1 - 1 : j0, b_step = reverse ? -1 : +1;

        const std::size_t stride = static_cast<std::size_t>(NC) + 1;
        hb_fit(buf.hfa, stride); hb_fit(buf.hfb, stride); hb_fit(buf.hfc, stride);
        hb_fit(buf.hfd, stride); hb_fit(buf.hfe, stride); hb_fit(buf.hff, stride);
        T* pM = buf.hfa.data(); T* pX = buf.hfb.data(); T* pY = buf.hfc.data();
        T* cM = buf.hfd.data(); T* cX = buf.hfe.data(); T* cY = buf.hff.data();
        // Pmax: the closed-form carry, as hb_scan_impl<..., true> (the ramp loaded, k*ge_a).
        const bool pmax = hb_pmax();
        if (pmax) hb_fit(buf.hramp, stride);
        T* rmp = buf.hramp.data();
        if (pmax) for (int k = 0; k < NC; ++k) rmp[k] = static_cast<T>(k) * ge_a;

        // Row 0.  Local: M = 0 for every column (fresh start anywhere).  Global: the
        // Y-gap-open series, so the reverse pass computes the suffixes' global alignment.
        pM[0] = static_cast<T>(0); pX[0] = static_cast<T>(NEG_INF); pY[0] = static_cast<T>(NEG_INF);
        for (int c = 1; c <= NC; ++c) {
            if (local) { pM[c] = static_cast<T>(0); pX[c] = static_cast<T>(NEG_INF); pY[c] = static_cast<T>(NEG_INF); }
            else {
                pM[c] = static_cast<T>(NEG_INF); pX[c] = static_cast<T>(NEG_INF);
                pY[c] = std::max((std::max(pM[c-1], pX[c-1]) - go_a) - ge_a, pY[c-1] - ge_a);
            }
        }

        T gbest = static_cast<T>(0); int gi = 0, gj = 0;
        for (int t = 0; t < H; ++t) {
            const int arow = a_start + t * a_step;                 // 0-based sequence row
            cM[0] = local ? static_cast<T>(0) : static_cast<T>(NEG_INF);
            cX[0] = std::max(std::max((pM[0] - go_b) - ge_b, pX[0] - ge_b), (pY[0] - go_b) - ge_b);
            cY[0] = static_cast<T>(NEG_INF);
            T P = cY[0] - ge_a;   // Pmax: the running prefix max, the border as its seed
            for (int c = 1; c <= NC; ++c) {
                const int bcol = b_start + (c - 1) * b_step;       // 0-based sequence col
                T m = std::max({pM[c-1], pX[c-1], pY[c-1]}) + subT(arow + 1, bcol + 1);
                if (local) m = std::max(m, static_cast<T>(0));
                const T x = std::max(std::max((pM[c] - go_b) - ge_b, pX[c] - ge_b), (pY[c] - go_b) - ge_b);
                T y;
                if (pmax) {
                    const T g = (std::max(cM[c-1], cX[c-1]) - go_a) - ge_a;
                    P = std::max(g + rmp[c-1], P);
                    y = P - rmp[c-1];
                } else {
                    y = std::max(std::max((cM[c-1] - go_a) - ge_a, (cX[c-1] - go_a) - ge_a), cY[c-1] - ge_a);
                }
                cM[c] = m; cX[c] = x; cY[c] = y;
                const T here = std::max({m, x, y});
                if (here > gbest) { gbest = here; gi = t + 1; gj = c; }
            }
            std::swap(pM, cM); std::swap(pX, cX); std::swap(pY, cY);
        }
        best = gbest; bi = gi; bj = gj;
    }

    // Entry: linear-space Smith-Waterman.  Forward clamped scan -> end cell (ie, je) and
    // score S; reverse global-suffix scan over the prefix rectangle -> start (is, js);
    // then hb_solve GLOBALLY aligns the box A[is..ie) x B[js..je), which scores exactly S.
    // Everything is O(n) memory.  Below hb_cutoff the box never splits and the fill is the
    // exact Pointers one, so short local pairs are bit-exact; longer ones return a valid
    // (equal-score) but possibly different optimal path at ties, like Global Hirschberg.
    void viterbi_affine_hirschberg_local(DpBuffer& buf) {
        hops_.clear();
        hb_start_i_ = 0; hb_start_j_ = 0;

        T s_fwd; int ie, je;
        hb_scan(buf, /*local=*/true, /*reverse=*/false, 0, m_, 0, n_, s_fwd, ie, je);

        // No positive-scoring local alignment: the empty alignment (score 0) is optimal.
        if (s_fwd <= static_cast<T>(0) || ie <= 0 || je <= 0) {
            viterbi_score_ = 0.0;
            best_i_ = 0; best_j_ = 0; best_tbl_ = TBTable::M;
            hirschberg_ = true;
            return;
        }

        T s_rev; int p, q;
        hb_scan(buf, /*local=*/false, /*reverse=*/true, 0, ie, 0, je, s_rev, p, q);
        const int is = ie - p, js = je - q;

        hb_solve(buf, is, ie, js, je, false, false);
        hb_start_i_ = is; hb_start_j_ = js;

        // Replay for the score, seeded at the local start (subT is 1-based).  Same order
        // and precision as the DP, so the path's score is exact — the value the tests pin
        // against SW Pointers.
        const T go_a = static_cast<T>(params_->gap_open_a);
        const T ge_a = static_cast<T>(params_->gap_extend_a);
        const T go_b = static_cast<T>(params_->gap_open_b);
        const T ge_b = static_cast<T>(params_->gap_extend_b);
        T s = static_cast<T>(0);
        int i = is, j = js;
        unsigned char prev = 3;
        for (unsigned char op : hops_) {
            if (op == 0) { ++i; ++j; s += subT(i, j); }
            else if (op == 1) { if (prev != 1) s -= go_b; s -= ge_b; ++i; }
            else              { if (prev != 2) s -= go_a; s -= ge_a; ++j; }
            prev = op;
        }
        viterbi_score_ = static_cast<double>(s);
        best_i_ = ie; best_j_ = je;
        best_tbl_ = hops_.empty() ? TBTable::M
                  : (hops_.back() == 1 ? TBTable::X
                     : (hops_.back() == 2 ? TBTable::Y : TBTable::M));
        hirschberg_ = true;
    }

    // ── Linear gaps in linear space ─────────────────────────────────────────────
    //
    // Hirschberg for linear gaps: no gap state crosses a row cut (a gap move costs ge
    // whatever precedes it), so the join is a plain argmax of forward + reverse scores —
    // none of the affine path's boundary flags or open refunds.  Scalar sweeps (linear
    // has no simd per-pair kernel), O(n) rows; the base case is the Pointers fill on the
    // sub-rectangle, walked back into hops_.  Local: the affine path's endpoint reduction
    // — a clamped forward scan for the end (ie, je) and its score, an unclamped global-
    // suffix scan for the start, then the global solve of the box.  Like affine
    // Hirschberg it may pick a different optimal path at ties above hb_cutoff, never a
    // worse one (the score is replayed from the path); a pair <= hb_cutoff runs AS
    // Pointers (run_viterbi), bit-exact.

    // Forward sweep from (i0, j0), Global borders, to row i1: out[c] = best score of
    // A[i0..i1) x B[j0..j0+c).  Reverse (rev): from (i1, j1) back to row i0, out[d] =
    // best score of A[i0..i1) x B[j1-d..j1).
    void lhb_sweep(DpBuffer& buf, bool rev, int i0, int i1, int j0, int j1, T* out) {
        const int W = j1 - j0;
        hb_fit(buf.hfa, static_cast<std::size_t>(W) + 1);
        T* row = buf.hfa.data();
        const T ge_a = static_cast<T>(params_->gap_extend_a);
        const T ge_b = static_cast<T>(params_->gap_extend_b);
        row[0] = static_cast<T>(0);
        for (int c = 1; c <= W; ++c) row[c] = row[c - 1] - ge_a;
        for (int r = 1; r <= i1 - i0; ++r) {
            const int i = rev ? i1 - r + 1 : i0 + r;   // 1-based DP row of A[i-1]
            T diag = row[0];
            row[0] = row[0] - ge_b;
            for (int c = 1; c <= W; ++c) {
                const int j = rev ? j1 - c + 1 : j0 + c;
                T v = diag + subT(i, j);
                const T u = row[c] - ge_b, l = row[c - 1] - ge_a;
                if (v < u) v = u;
                if (v < l) v = l;
                diag = row[c];
                row[c] = v;
            }
        }
        std::copy(row, row + W + 1, out);
    }

    // Base case: Pointers on the sub-rectangle, Global borders, walked back from
    // (i1, j1) to (i0, j0); the moves appended to hops_ in forward order.
    void lhb_base(DpBuffer& buf, int i0, int i1, int j0, int j1) {
        const int H = i1 - i0, W = j1 - j0;
        const std::size_t st = static_cast<std::size_t>(W) + 1;
        if (buf.DM.size() < (static_cast<std::size_t>(H) + 1) * st)
            buf.DM.resize((static_cast<std::size_t>(H) + 1) * st);
        hb_fit(buf.hfa, st); hb_fit(buf.hfb, st);
        T* prev = buf.hfa.data(); T* cur = buf.hfb.data();
        unsigned char* D = buf.DM.data();
        const T ge_a = static_cast<T>(params_->gap_extend_a);
        const T ge_b = static_cast<T>(params_->gap_extend_b);
        prev[0] = static_cast<T>(0);
        for (int c = 1; c <= W; ++c) { prev[c] = prev[c - 1] - ge_a; D[c] = 2; }
        for (int r = 1; r <= H; ++r) {
            unsigned char* d = D + static_cast<std::size_t>(r) * st;
            cur[0] = prev[0] - ge_b; d[0] = 1;
            for (int c = 1; c <= W; ++c) {
                T v = prev[c - 1] + subT(i0 + r, j0 + c);
                unsigned char k = 0;
                const T u = prev[c] - ge_b, l = cur[c - 1] - ge_a;
                if (v < u) { v = u; k = 1; }
                if (v < l) { v = l; k = 2; }
                cur[c] = v; d[c] = k;
            }
            std::swap(prev, cur);
        }
        const std::size_t at = hops_.size();
        int r = H, c = W;
        while (r > 0 || c > 0) {
            const unsigned char k = D[static_cast<std::size_t>(r) * st + c];
            hops_.push_back(k);
            if (k == 0) { --r; --c; } else if (k == 1) --r; else --c;
        }
        std::reverse(hops_.begin() + static_cast<std::ptrdiff_t>(at), hops_.end());
    }

    void lhb_solve(DpBuffer& buf, int i0, int i1, int j0, int j1) {
        const int H = i1 - i0, W = j1 - j0;
        if (H <= hb_cutoff_) { lhb_base(buf, i0, i1, j0, j1); return; }
        const int p = i0 + H / 2;
        hb_fit(buf.hsM, static_cast<std::size_t>(W) + 1);
        hb_fit(buf.hra, static_cast<std::size_t>(W) + 1);
        lhb_sweep(buf, false, i0, p, j0, j1, buf.hsM.data());
        lhb_sweep(buf, true, p, i1, j0, j1, buf.hra.data());
        const T* F = buf.hsM.data(); const T* R = buf.hra.data();
        T best = static_cast<T>(NEG_INF);
        int cstar = 0;
        for (int c = 0; c <= W; ++c) {
            const T v = F[c] + R[W - c];
            if (v > best) { best = v; cstar = c; }
        }
        lhb_solve(buf, i0, p, j0, j0 + cstar);
        lhb_solve(buf, p, i1, j0 + cstar, j1);
    }

    void viterbi_linear_hirschberg(DpBuffer& buf) {
        hops_.clear();
        hb_start_i_ = 0; hb_start_j_ = 0;
        int i_end = m_, j_end = n_;
        const T ge_a = static_cast<T>(params_->gap_extend_a);
        const T ge_b = static_cast<T>(params_->gap_extend_b);
        if constexpr (AM == AlignMode::Local) {
            // Forward clamped scan: the best cell, first row then first column (strict >,
            // as viterbi_linear chooses it).
            hb_fit(buf.hfa, static_cast<std::size_t>(n_) + 1);
            T* row = buf.hfa.data();
            std::fill(row, row + n_ + 1, static_cast<T>(0));
            T best = static_cast<T>(0); int ie = 0, je = 0;
            for (int i = 1; i <= m_; ++i) {
                T diag = row[0];
                for (int j = 1; j <= n_; ++j) {
                    T v = diag + subT(i, j);
                    const T u = row[j] - ge_b, l = row[j - 1] - ge_a;
                    if (v < u) v = u;
                    if (v < l) v = l;
                    v = std::max(v, static_cast<T>(0));
                    diag = row[j];
                    row[j] = v;
                    if (v > best) { best = v; ie = i; je = j; }
                }
            }
            if (!(best > static_cast<T>(0))) {
                viterbi_score_ = 0.0; best_i_ = 0; best_j_ = 0; hirschberg_ = true; return;
            }
            // Unclamped global-suffix scan over [0, ie) x [0, je): the start maximizing
            // the box's global score (any argmax scores the local optimum).
            hb_fit(buf.hfb, static_cast<std::size_t>(je) + 1);
            T* g = buf.hfb.data();                       // g[q] = score from (p, q) to (ie, je)
            g[je] = static_cast<T>(0);
            for (int q = je - 1; q >= 0; --q) g[q] = g[q + 1] - ge_a;
            T sbest = g[0]; int is = ie, js = 0;
            for (int q = 0; q <= je; ++q) if (g[q] > sbest) { sbest = g[q]; js = q; }
            for (int pi = ie - 1; pi >= 0; --pi) {
                T diag = g[je];
                g[je] = g[je] - ge_b;
                if (g[je] > sbest) { sbest = g[je]; is = pi; js = je; }
                for (int q = je - 1; q >= 0; --q) {
                    T v = diag + subT(pi + 1, q + 1);
                    const T u = g[q] - ge_b, l = g[q + 1] - ge_a;
                    if (v < u) v = u;
                    if (v < l) v = l;
                    diag = g[q];
                    g[q] = v;
                    if (v > sbest) { sbest = v; is = pi; js = q; }
                }
            }
            hb_start_i_ = is; hb_start_j_ = js;
            i_end = ie; j_end = je;
            lhb_solve(buf, is, ie, js, je);
        } else {
            lhb_solve(buf, 0, m_, 0, n_);
        }
        // Replay for the score, in T and in path order (the tests compare it exactly).
        T sc = static_cast<T>(0);
        int i = hb_start_i_, j = hb_start_j_;
        for (unsigned char op : hops_) {
            if (op == 0)      { ++i; ++j; sc += subT(i, j); }
            else if (op == 1) { ++i; sc -= ge_b; }
            else              { ++j; sc -= ge_a; }
        }
        viterbi_score_ = static_cast<double>(sc);
        best_i_ = i_end; best_j_ = j_end;
        hirschberg_ = true;
    }

    void hard_grad_linear_hb(AlignParams& grad) const {
        double* gblk = grad_block(grad);
        int i = hb_start_i_, j = hb_start_j_;
        for (unsigned char op : hops_) {
            if (op == 0)      { ++i; ++j; gblk[sub_off(i, j)] += 1.0; }
            else if (op == 1) { ++i; grad.gap_extend_b -= 1.0; }
            else              { ++j; grad.gap_extend_a -= 1.0; }
        }
    }

    // ── Consumers of a Hirschberg path ────────────────────────────────────────
    // Each mirrors its table-walking twin, but replays hops_ forward instead of
    // walking predecessors backward.  Same emissions, same gradient convention.

    void hard_grad_affine_hb(AlignParams& grad) const {
        double* gblk = grad_block(grad);
        int i = hb_start_i_, j = hb_start_j_;   // local: the path starts at (is, js), not the origin
        unsigned char prev = 3;
        for (unsigned char op : hops_) {
            if (op == 0) { ++i; ++j; gblk[sub_off(i, j)] += 1.0; }
            else if (op == 1) {
                if (prev != 1) grad.gap_open_b -= 1.0;
                grad.gap_extend_b -= 1.0; ++i;
            } else {
                if (prev != 2) grad.gap_open_a -= 1.0;
                grad.gap_extend_a -= 1.0; ++j;
            }
            prev = op;
        }
    }

    std::vector<std::pair<int,int>> alignment_hb() const {
        // Emit the 0-based sequence position of each matched pair BEFORE advancing —
        // the same convention traceback_affine uses (it emits i-1,j-1 from 1-based DP
        // coords), so alignment() agrees across tracebacks and with pairs_from_gapped().
        std::vector<std::pair<int,int>> path;
        int i = hb_start_i_, j = hb_start_j_;
        for (unsigned char op : hops_) {
            if (op == 0)      { path.emplace_back(i, j); ++i; ++j; }
            else if (op == 1) { ++i; }
            else              { ++j; }
        }
        return path;
    }

    void aligned_hb(std::string& a, std::string& b) const {
        a.clear(); b.clear();
        int i = hb_start_i_, j = hb_start_j_;
        for (unsigned char op : hops_) {
            if (op == 0)      { ++i; ++j; a.push_back(sym_a(i)); b.push_back(sym_b(j)); }
            else if (op == 1) { ++i; a.push_back(sym_a(i)); b.push_back('-'); }
            else              { ++j; a.push_back('-');      b.push_back(sym_b(j)); }
        }
    }

    std::vector<int> guide_j_affine_hb() const {
        std::vector<int> gj(static_cast<std::size_t>(m_ + 1), -1);
        gj[0] = 0;
        // Local: the path covers only rows [is, ie); seed the start anchor and let the
        // rows outside it interpolate (fill_guide_gaps), exactly as guide_j_affine does
        // from its local traceback.  For Global (is=js=0) this is the original walk.
        int i = hb_start_i_, j = hb_start_j_;
        gj[static_cast<std::size_t>(i)] = j;
        for (unsigned char op : hops_) {
            if (op == 0)      { ++i; ++j; gj[static_cast<std::size_t>(i)] = j; }
            else if (op == 1) { ++i;      gj[static_cast<std::size_t>(i)] = j; }
            else              { ++j; }
        }
        gj[static_cast<std::size_t>(m_)] = n_;
        fill_guide_gaps(gj);
        return gj;
    }

    void viterbi_affine(DpBuffer& buf) {
        band_fill(buf.VM, NEG_INF);
        band_fill(buf.VX, NEG_INF);
        band_fill(buf.VY, NEG_INF);

        // Gap penalties in the Viterbi precision T.  Converting once, up front, is what
        // makes this scalar fill bit-exact with the T-typed simd kernel: both then do
        // (v - go) - ge in T with the same left association, so their VM/VX/VY tables
        // agree to the last bit (which the argmax tracebacks below depend on).
        const T go_a = static_cast<T>(params_->gap_open_a);
        const T ge_a = static_cast<T>(params_->gap_extend_a);
        const T go_b = static_cast<T>(params_->gap_open_b);
        const T ge_b = static_cast<T>(params_->gap_extend_b);

        if constexpr (AM == AlignMode::Global) {
            at(buf.VM, 0, 0) = static_cast<T>(0);
            const int bi = border_rows(), bj = border_cols();
            // VX along column 0: all gaps in B (consuming A), uses gap_b params
            for (int i = 1; i <= bi; ++i)
                at(buf.VX, i, 0) = -(go_b + static_cast<T>(i) * ge_b);
            // VY along row 0: all gaps in A (consuming B), uses gap_a params
            for (int j = 1; j <= bj; ++j)
                at(buf.VY, 0, j) = -(go_a + static_cast<T>(j) * ge_a);
        } else {
            for (int i = 0; i <= m_; ++i) at(buf.VM, i, 0) = static_cast<T>(0);
            for (int j = 0; j <= n_; ++j) at(buf.VM, 0, j) = static_cast<T>(0);
        }

        best_i_ = 0; best_j_ = 0; best_tbl_ = TBTable::M;
        T best_local = static_cast<T>(0);

        for (int i = 1; i <= m_; ++i) {
            for (int j = jlo(i); j <= jhi(i); ++j) {
                T diag  = std::max({rat(buf.VM,i-1,j-1), rat(buf.VX,i-1,j-1), rat(buf.VY,i-1,j-1)});
                T m_val = diag + subT(i, j);
                // X state: gap in B (advance i), uses gap_b params
                T x_val = std::max({
                    rat(buf.VM,i-1,j) - go_b - ge_b,
                    rat(buf.VX,i-1,j)        - ge_b,
                    rat(buf.VY,i-1,j) - go_b - ge_b});
                // Y state: gap in A (advance j), uses gap_a params
                T y_val = std::max({
                    rat(buf.VM,i,j-1) - go_a - ge_a,
                    rat(buf.VX,i,j-1) - go_a - ge_a,
                    rat(buf.VY,i,j-1)        - ge_a});

                if constexpr (AM == AlignMode::Local) {
                    m_val = std::max(m_val, static_cast<T>(0));
                    T best_here = std::max({m_val, x_val, y_val});
                    if (best_here > best_local) {
                        best_local = best_here; best_i_ = i; best_j_ = j;
                        if      (m_val >= x_val && m_val >= y_val) best_tbl_ = TBTable::M;
                        else if (x_val >= y_val)                    best_tbl_ = TBTable::X;
                        else                                         best_tbl_ = TBTable::Y;
                    }
                }

                at(buf.VM, i, j) = m_val; at(buf.VX, i, j) = x_val; at(buf.VY, i, j) = y_val;
            }
        }

        if constexpr (AM == AlignMode::Global) {
            T vm = rat(buf.VM, m_, n_), vx = rat(buf.VX, m_, n_), vy = rat(buf.VY, m_, n_);
            viterbi_score_ = std::max({vm, vx, vy});
            best_i_ = m_; best_j_ = n_;
            if      (vm >= vx && vm >= vy) best_tbl_ = TBTable::M;
            else if (vx >= vy)             best_tbl_ = TBTable::X;
            else                            best_tbl_ = TBTable::Y;
        } else {
            viterbi_score_ = best_local;
        }
    }

    template<typename EmitFn>
    void traceback_affine_impl(const DpBuffer& buf, EmitFn&& emit) const {
        int i = best_i_, j = best_j_;
        TBTable tbl = best_tbl_;
        // Variant B retains predecessor codes, not score tables — reading VM/VX/VY
        // below would walk off an unallocated vector.  (It did: align_full() +
        // aligned() segfaulted before this branch existed.)
        if (pointers_) {
            while (true) {
                if (i == 0 && j == 0) break;
                if constexpr (AM == AlignMode::Local)
                    if (tbl == TBTable::M && dcode(buf.DM, i, j) == 3) break;
                // Same border guard as the score-table walks — see the note there.
                if      (i == 0) tbl = TBTable::Y;
                else if (j == 0) tbl = TBTable::X;
                unsigned char c;
                if (tbl == TBTable::M) {
                    emit(i - 1, j - 1);
                    c = dcode(buf.DM, i, j); --i; --j;
                } else if (tbl == TBTable::X) {
                    c = dcode(buf.DX, i, j); --i;
                } else {
                    c = dcode(buf.DY, i, j); --j;
                }
                tbl = (c == 1) ? TBTable::X : (c == 2 ? TBTable::Y : TBTable::M);
            }
            return;
        }
        // Same T-precision penalties as the forward pass, so the predecessor argmax
        // re-derived here reproduces the branch the fill took, bit for bit.
        const T go_a = static_cast<T>(params_->gap_open_a);
        const T ge_a = static_cast<T>(params_->gap_extend_a);
        const T go_b = static_cast<T>(params_->gap_open_b);
        const T ge_b = static_cast<T>(params_->gap_extend_b);

        while (true) {
            if (i == 0 && j == 0) break;
            if constexpr (AM == AlignMode::Local)
                if (tbl == TBTable::M && rat(buf.VM, i, j) <= 0.0) break;
            // GuideBanded with a band narrower than the path needs: the band can fail
            // to admit any route back to the origin, leaving every predecessor at
            // -inf.  The M>=X>=Y tie-break then picks M, and M steps DIAGONALLY — so
            // i or j goes negative, cell_index() turns that into a huge size_t, and
            // the walk reads off the end of the table.  That was a segfault, not a
            // suboptimal score (realign_banded: "A"*22 vs "C"*87 at band<=32).
            // At a border only one move is legal; forcing it is what a well-formed
            // DP would have chosen anyway, so valid alignments are unaffected.
            if      (i == 0) tbl = TBTable::Y;   // row 0: only leftward moves remain
            else if (j == 0) tbl = TBTable::X;   // col 0: only upward moves remain

            if (tbl == TBTable::M) {
                emit(i-1, j-1);
                T vm = rat(buf.VM,i-1,j-1), vx = rat(buf.VX,i-1,j-1), vy = rat(buf.VY,i-1,j-1);
                --i; --j;
                if      (vm >= vx && vm >= vy) tbl = TBTable::M;
                else if (vx >= vy)             tbl = TBTable::X;
                else                            tbl = TBTable::Y;
            } else if (tbl == TBTable::X) {
                T fm = rat(buf.VM,i-1,j) - go_b - ge_b;
                T fx = rat(buf.VX,i-1,j)        - ge_b;
                T fy = rat(buf.VY,i-1,j) - go_b - ge_b;
                --i;
                if      (fm >= fx && fm >= fy) tbl = TBTable::M;
                else if (fx >= fy)             tbl = TBTable::X;
                else                            tbl = TBTable::Y;
            } else {
                T fm = rat(buf.VM,i,j-1) - go_a - ge_a;
                T fx = rat(buf.VX,i,j-1) - go_a - ge_a;
                T fy = rat(buf.VY,i,j-1)        - ge_a;
                --j;
                if      (fm >= fx && fm >= fy) tbl = TBTable::M;
                else if (fx >= fy)             tbl = TBTable::X;
                else                            tbl = TBTable::Y;
            }
        }
    }

    std::vector<std::pair<int,int>> traceback_affine(const DpBuffer& buf) const {
        std::vector<std::pair<int,int>> path;
        traceback_affine_impl(buf, [&](int i, int j) { path.emplace_back(i, j); });
        std::reverse(path.begin(), path.end());
        return path;
    }

    void aligned_affine(const DpBuffer& buf, std::string& a, std::string& b) const {
        int i = best_i_, j = best_j_;
        TBTable tbl = best_tbl_;
        // Pointers: no score tables to read.  (This is the FOURTH copy of this walk
        // in the file — guide_j, hard_grad, traceback_impl and here — and missing it
        // was a segfault, not a wrong answer, because B leaves VM/VX/VY unallocated.)
        if (pointers_) {
            while (true) {
                if (i == 0 && j == 0) break;
                if constexpr (AM == AlignMode::Local)
                    if (tbl == TBTable::M && dcode(buf.DM, i, j) == 3) break;
                // Same border guard as the score-table walks — see the note there.
                if      (i == 0) tbl = TBTable::Y;
                else if (j == 0) tbl = TBTable::X;
                unsigned char c;
                if (tbl == TBTable::M) {
                    a.push_back(sym_a(i)); b.push_back(sym_b(j));
                    c = dcode(buf.DM, i, j); --i; --j;
                } else if (tbl == TBTable::X) {
                    a.push_back(sym_a(i)); b.push_back('-');
                    c = dcode(buf.DX, i, j); --i;
                } else {
                    a.push_back('-'); b.push_back(sym_b(j));
                    c = dcode(buf.DY, i, j); --j;
                }
                tbl = (c == 1) ? TBTable::X : (c == 2 ? TBTable::Y : TBTable::M);
            }
            std::reverse(a.begin(), a.end());
            std::reverse(b.begin(), b.end());
            return;
        }
        const T go_a = static_cast<T>(params_->gap_open_a);
        const T ge_a = static_cast<T>(params_->gap_extend_a);
        const T go_b = static_cast<T>(params_->gap_open_b);
        const T ge_b = static_cast<T>(params_->gap_extend_b);

        while (true) {
            if (i == 0 && j == 0) break;
            if constexpr (AM == AlignMode::Local)
                if (tbl == TBTable::M && rat(buf.VM, i, j) <= 0.0) break;
            // GuideBanded with a band narrower than the path needs: the band can fail
            // to admit any route back to the origin, leaving every predecessor at
            // -inf.  The M>=X>=Y tie-break then picks M, and M steps DIAGONALLY — so
            // i or j goes negative, cell_index() turns that into a huge size_t, and
            // the walk reads off the end of the table.  That was a segfault, not a
            // suboptimal score (realign_banded: "A"*22 vs "C"*87 at band<=32).
            // At a border only one move is legal; forcing it is what a well-formed
            // DP would have chosen anyway, so valid alignments are unaffected.
            if      (i == 0) tbl = TBTable::Y;   // row 0: only leftward moves remain
            else if (j == 0) tbl = TBTable::X;   // col 0: only upward moves remain

            if (tbl == TBTable::M) {
                a.push_back(sym_a(i)); b.push_back(sym_b(j));
                T vm = rat(buf.VM,i-1,j-1), vx = rat(buf.VX,i-1,j-1), vy = rat(buf.VY,i-1,j-1);
                --i; --j;
                if      (vm >= vx && vm >= vy) tbl = TBTable::M;
                else if (vx >= vy)             tbl = TBTable::X;
                else                            tbl = TBTable::Y;
            } else if (tbl == TBTable::X) {
                a.push_back(sym_a(i)); b.push_back('-');  // gap in B
                T fm = rat(buf.VM,i-1,j) - go_b - ge_b;
                T fx = rat(buf.VX,i-1,j)        - ge_b;
                T fy = rat(buf.VY,i-1,j) - go_b - ge_b;
                --i;
                if      (fm >= fx && fm >= fy) tbl = TBTable::M;
                else if (fx >= fy)             tbl = TBTable::X;
                else                            tbl = TBTable::Y;
            } else {
                a.push_back('-'); b.push_back(sym_b(j));  // gap in A
                T fm = rat(buf.VM,i,j-1) - go_a - ge_a;
                T fx = rat(buf.VX,i,j-1) - go_a - ge_a;
                T fy = rat(buf.VY,i,j-1)        - ge_a;
                --j;
                if      (fm >= fx && fm >= fy) tbl = TBTable::M;
                else if (fx >= fy)             tbl = TBTable::X;
                else                            tbl = TBTable::Y;
            }
        }
        std::reverse(a.begin(), a.end());
        std::reverse(b.begin(), b.end());
    }

    // gj != nullptr: also record the guide path, exactly as guide_j_affine() would —
    // the two walks take the same steps — so one walk serves both.
    void hard_grad_affine(const DpBuffer& buf, AlignParams& grad,
                          std::vector<int>* gj = nullptr) const {
        double* gblk = grad_block(grad);
        int i = best_i_, j = best_j_;
        TBTable tbl = best_tbl_;
        if (gj) {
            gj->assign(static_cast<size_t>(m_ + 1), -1);
            (*gj)[0] = 0;
            (*gj)[static_cast<size_t>(m_)] = n_;
            (*gj)[static_cast<size_t>(i)] = j;
        }
        // T-precision penalties for the predecessor argmax; the gradient counts
        // themselves accumulate into the double grad block, exact for integer counts.
        const T go_a = static_cast<T>(params_->gap_open_a);
        const T ge_a = static_cast<T>(params_->gap_extend_a);
        const T go_b = static_cast<T>(params_->gap_open_b);
        const T ge_b = static_cast<T>(params_->gap_extend_b);

        while (true) {
            if (i == 0 && j == 0) break;
            if constexpr (AM == AlignMode::Local)
                if (tbl == TBTable::M && rat(buf.VM, i, j) <= 0.0) break;
            if constexpr (AM == AlignMode::Global) {
                // On row 0 or column 0 one move remains all the way to the origin: the
                // path ends in a single leading gap run, which is one open and i or j
                // extends — exactly what stepping through the Full DP's border values
                // counts.  Counted directly because a guide band need not cover the
                // border: guide_j[0] is where the old path LEFT row 0, so after a long
                // leading gap the band misses the origin, and stepping read cells never
                // initialised for this pair (stale data from the thread's previous one:
                // banded_grad gap-open counts that changed with the thread count).
                if (i == 0) {
                    grad.gap_extend_a -= static_cast<double>(j);
                    grad.gap_open_a   -= 1.0;   // (moves along row 0 add no guide entry)
                    break;
                }
                if (j == 0) {
                    grad.gap_extend_b -= static_cast<double>(i);
                    grad.gap_open_b   -= 1.0;
                    if (gj) for (int r = i - 1; r >= 0; --r) (*gj)[static_cast<size_t>(r)] = 0;  // as the X steps would
                    break;
                }
            }
            // GuideBanded with a band narrower than the path needs: the band can fail
            // to admit any route back to the origin, leaving every predecessor at
            // -inf.  The M>=X>=Y tie-break then picks M, and M steps DIAGONALLY — so
            // i or j goes negative, cell_index() turns that into a huge size_t, and
            // the walk reads off the end of the table.  That was a segfault, not a
            // suboptimal score (realign_banded: "A"*22 vs "C"*87 at band<=32).
            // At a border only one move is legal; forcing it is what a well-formed
            // DP would have chosen anyway, so valid alignments are unaffected.
            if      (i == 0) tbl = TBTable::Y;   // row 0: only leftward moves remain
            else if (j == 0) tbl = TBTable::X;   // col 0: only upward moves remain

            if (tbl == TBTable::M) {
                gblk[sub_off(i, j)] += 1.0;
                T vm = rat(buf.VM,i-1,j-1), vx = rat(buf.VX,i-1,j-1), vy = rat(buf.VY,i-1,j-1);
                --i; --j;
                if (gj) (*gj)[static_cast<size_t>(i)] = j;
                if      (vm >= vx && vm >= vy) tbl = TBTable::M;
                else if (vx >= vy)             tbl = TBTable::X;
                else                            tbl = TBTable::Y;
            } else if (tbl == TBTable::X) {
                // X state: gap in B (advance i), uses gap_b params
                grad.gap_extend_b -= 1.0;   // the score subtracts this penalty
                T fm = rat(buf.VM,i-1,j) - go_b - ge_b;
                T fx = rat(buf.VX,i-1,j)        - ge_b;
                T fy = rat(buf.VY,i-1,j) - go_b - ge_b;
                --i;
                if (gj) (*gj)[static_cast<size_t>(i)] = j;
                TBTable prev;
                if      (fm >= fx && fm >= fy) prev = TBTable::M;
                else if (fx >= fy)             prev = TBTable::X;
                else                            prev = TBTable::Y;
                if (prev != TBTable::X) grad.gap_open_b -= 1.0;  // gap opening
                tbl = prev;
            } else {
                // Y state: gap in A (advance j), uses gap_a params
                grad.gap_extend_a -= 1.0;
                T fm = rat(buf.VM,i,j-1) - go_a - ge_a;
                T fx = rat(buf.VX,i,j-1) - go_a - ge_a;
                T fy = rat(buf.VY,i,j-1)        - ge_a;
                --j;
                TBTable prev;
                if      (fm >= fx && fm >= fy) prev = TBTable::M;
                else if (fx >= fy)             prev = TBTable::X;
                else                            prev = TBTable::Y;
                if (prev != TBTable::Y) grad.gap_open_a -= 1.0;  // gap opening
                tbl = prev;
            }
        }
        if (gj) fill_guide_gaps(*gj);
    }

    // ═════════════════════════════════════════════════════════════════════════
    // Forward-backward — Linear gap model
    // ═════════════════════════════════════════════════════════════════════════

    void fwdbwd_linear(DpBuffer& buf) {
        // ── Forward ──
        banded_fill(buf.F);

        if constexpr (AM == AlignMode::Global) {
            sat(buf.F, 0, 0) = 0.0;
            const int bi = border_rows(), bj = border_cols();
            for (int i = 1; i <= bi; ++i)
                sat(buf.F, i, 0) = -static_cast<double>(i) * params_->gap_extend_b;
            for (int j = 1; j <= bj; ++j)
                sat(buf.F, 0, j) = -static_cast<double>(j) * params_->gap_extend_a;
        } else {
            for (int i = 0; i <= m_; ++i) sat(buf.F, i, 0) = 0.0;
            for (int j = 0; j <= n_; ++j) sat(buf.F, 0, j) = 0.0;
        }

        for (int i = 1; i <= m_; ++i) {
            for (int j = jlo(i); j <= jhi(i); ++j) {
                double v = lse3(
                    srat(buf.F, i-1, j-1) + sub(i, j),
                    srat(buf.F, i-1, j)   - params_->gap_extend_b,
                    srat(buf.F, i,   j-1) - params_->gap_extend_a
                );
                if constexpr (AM == AlignMode::Local) v = lse2(v, 0.0);
                sat(buf.F, i, j) = v;
            }
        }

        if constexpr (AM == AlignMode::Global) {
            log_z_ = srat(buf.F, m_, n_);
        } else {
            log_z_ = NEG_INF;
            for (int i = 0; i <= m_; ++i)
                for (int j = jlo0(i); j <= jhi0(i); ++j)
                    log_z_ = lse2(log_z_, srat(buf.F, i, j));
        }

        // ── Backward ──
        if constexpr (AM == AlignMode::Global) {
            band_fill(buf.B, NEG_INF);
            sat(buf.B, m_, n_) = 0.0;
        } else {
            band_fill(buf.B, 0.0);
        }

        for (int i = m_; i >= 0; --i) {
            for (int j = jhi0(i); j >= jlo0(i); --j) {
                double bval = srat(buf.B, i, j);
                if (bval == NEG_INF) continue;
                if (i > 0 && j > 0)
                    sat(buf.B,i-1,j-1) = lse2(srat(buf.B,i-1,j-1),
                                              bval + sub(i, j));
                if (i > 0)
                    sat(buf.B,i-1,j)   = lse2(srat(buf.B,i-1,j),   bval - params_->gap_extend_b);
                if (j > 0)
                    sat(buf.B,i,j-1)   = lse2(srat(buf.B,i,j-1),   bval - params_->gap_extend_a);
            }
        }
    }

    // Local borders are constants (a free start), not gap moves: the forward pass sets
    // F(i,0) = F(0,j) = 0 rather than deriving them from a neighbour minus a gap cost,
    // so a gap transition INTO a border cell does not exist and must not be counted,
    // even though the backward pass leaves finite values there.  Global borders really
    // are charged gap runs, so there the border targets stay in.
    static constexpr int kGapTargetMin = (AM == AlignMode::Local) ? 1 : 0;

    void soft_grad_linear(const DpBuffer& buf, AlignParams& grad) const {
        double* gblk = grad_block(grad);
        // Matrix gradient: match steps (i-1,j-1) → (i,j)
        for (int i = 1; i <= m_; ++i) {
            for (int j = jlo(i); j <= jhi(i); ++j) {
                double bval = srat(buf.B, i, j);
                if (bval == NEG_INF) continue;
                double log_p = srat(buf.F, i-1, j-1)
                               + sub(i, j)
                               + bval - log_z_;
                gblk[sub_off(i, j)] += std::exp(log_p);
            }
        }

        // gap_extend_b: B-gap steps (i-1,j) → (i,j), cost = -gap_extend_b
        for (int i = 1; i <= m_; ++i) {
            for (int j = std::max(kGapTargetMin, jlo0(i)); j <= jhi0(i); ++j) {
                double bval = srat(buf.B, i, j);
                if (bval == NEG_INF) continue;
                double fval = srat(buf.F, i-1, j);
                if (fval == NEG_INF) continue;
                grad.gap_extend_b -= std::exp(fval - params_->gap_extend_b + bval - log_z_);
            }
        }

        // gap_extend_a: A-gap steps (i,j-1) → (i,j), cost = -gap_extend_a
        for (int i = kGapTargetMin; i <= m_; ++i) {
            for (int j = std::max(1, jlo0(i)); j <= jhi0(i); ++j) {
                double bval = srat(buf.B, i, j);
                if (bval == NEG_INF) continue;
                double fval = srat(buf.F, i, j-1);
                if (fval == NEG_INF) continue;
                grad.gap_extend_a -= std::exp(fval - params_->gap_extend_a + bval - log_z_);
            }
        }
    }

    // ═════════════════════════════════════════════════════════════════════════
    // Forward-backward — Affine gap model
    // ═════════════════════════════════════════════════════════════════════════

    void fwdbwd_affine(DpBuffer& buf) {
        // ── Forward ──
        band_fill(buf.FM, NEG_INF);
        band_fill(buf.FX, NEG_INF);
        band_fill(buf.FY, NEG_INF);

        if constexpr (AM == AlignMode::Global) {
            sat(buf.FM, 0, 0) = 0.0;
            const int bi = border_rows(), bj = border_cols();
            for (int i = 1; i <= bi; ++i)
                sat(buf.FX, i, 0) = -(params_->gap_open_b + static_cast<double>(i) * params_->gap_extend_b);
            for (int j = 1; j <= bj; ++j)
                sat(buf.FY, 0, j) = -(params_->gap_open_a + static_cast<double>(j) * params_->gap_extend_a);
        } else {
            for (int i = 0; i <= m_; ++i) sat(buf.FM, i, 0) = 0.0;
            for (int j = 0; j <= n_; ++j) sat(buf.FM, 0, j) = 0.0;
        }

        for (int i = 1; i <= m_; ++i) {
            for (int j = jlo(i); j <= jhi(i); ++j) {
                double diag  = lse3(srat(buf.FM,i-1,j-1), srat(buf.FX,i-1,j-1), srat(buf.FY,i-1,j-1));
                double m_val = diag + sub(i, j);
                double x_val = lse3(
                    srat(buf.FM,i-1,j) - params_->gap_open_b - params_->gap_extend_b,
                    srat(buf.FX,i-1,j)                       - params_->gap_extend_b,
                    srat(buf.FY,i-1,j) - params_->gap_open_b - params_->gap_extend_b);
                double y_val = lse3(
                    srat(buf.FM,i,j-1) - params_->gap_open_a - params_->gap_extend_a,
                    srat(buf.FX,i,j-1) - params_->gap_open_a - params_->gap_extend_a,
                    srat(buf.FY,i,j-1)                       - params_->gap_extend_a);
                if constexpr (AM == AlignMode::Local) m_val = lse2(m_val, 0.0);
                sat(buf.FM, i, j) = m_val; sat(buf.FX, i, j) = x_val; sat(buf.FY, i, j) = y_val;
            }
        }

        if constexpr (AM == AlignMode::Global) {
            log_z_ = lse3(srat(buf.FM,m_,n_), srat(buf.FX,m_,n_), srat(buf.FY,m_,n_));
        } else {
            log_z_ = NEG_INF;
            for (int i = 0; i <= m_; ++i)
                for (int j = jlo0(i); j <= jhi0(i); ++j)
                    log_z_ = lse2(log_z_, lse3(srat(buf.FM,i,j), srat(buf.FX,i,j), srat(buf.FY,i,j)));
        }

        // ── Backward ──
        if constexpr (AM == AlignMode::Global) {
            band_fill(buf.BM, NEG_INF);
            band_fill(buf.BX, NEG_INF);
            band_fill(buf.BY, NEG_INF);
            sat(buf.BM,m_,n_) = 0.0; sat(buf.BX,m_,n_) = 0.0; sat(buf.BY,m_,n_) = 0.0;
        } else {
            band_fill(buf.BM, 0.0);
            band_fill(buf.BX, 0.0);
            band_fill(buf.BY, 0.0);
        }

        for (int i = m_; i >= 0; --i) {
            for (int j = jhi0(i); j >= jlo0(i); --j) {
                double bm = srat(buf.BM,i,j), bx = srat(buf.BX,i,j), by = srat(buf.BY,i,j);

                if (bm != NEG_INF && i > 0 && j > 0) {
                    double contrib = bm + sub(i, j);
                    sat(buf.BM,i-1,j-1) = lse2(srat(buf.BM,i-1,j-1), contrib);
                    sat(buf.BX,i-1,j-1) = lse2(srat(buf.BX,i-1,j-1), contrib);
                    sat(buf.BY,i-1,j-1) = lse2(srat(buf.BY,i-1,j-1), contrib);
                }
                // X state (gap in B) contribution to predecessors at (i-1,j)
                if (bx != NEG_INF && i > 0) {
                    sat(buf.BM,i-1,j) = lse2(srat(buf.BM,i-1,j),
                                             bx - params_->gap_open_b - params_->gap_extend_b);
                    sat(buf.BX,i-1,j) = lse2(srat(buf.BX,i-1,j), bx - params_->gap_extend_b);
                    sat(buf.BY,i-1,j) = lse2(srat(buf.BY,i-1,j),
                                             bx - params_->gap_open_b - params_->gap_extend_b);
                }
                // Y state (gap in A) contribution to predecessors at (i,j-1)
                if (by != NEG_INF && j > 0) {
                    sat(buf.BM,i,j-1) = lse2(srat(buf.BM,i,j-1),
                                             by - params_->gap_open_a - params_->gap_extend_a);
                    sat(buf.BX,i,j-1) = lse2(srat(buf.BX,i,j-1),
                                             by - params_->gap_open_a - params_->gap_extend_a);
                    sat(buf.BY,i,j-1) = lse2(srat(buf.BY,i,j-1), by - params_->gap_extend_a);
                }
            }
        }
    }

    void soft_grad_affine(const DpBuffer& buf, AlignParams& grad) const {
        double* gblk = grad_block(grad);
        // Matrix gradient: match steps
        for (int i = 1; i <= m_; ++i) {
            for (int j = jlo(i); j <= jhi(i); ++j) {
                double bm = srat(buf.BM, i, j);
                if (bm == NEG_INF) continue;
                double pred  = lse3(srat(buf.FM,i-1,j-1), srat(buf.FX,i-1,j-1), srat(buf.FY,i-1,j-1));
                double log_p = pred + sub(i, j) + bm - log_z_;
                gblk[sub_off(i, j)] += std::exp(log_p);
            }
        }

        // gap_extend_b: expected # of X-state steps (gap in B)
        for (int i = 1; i <= m_; ++i) {
            for (int j = jlo0(i); j <= jhi0(i); ++j) {
                double fx = srat(buf.FX, i, j), bx = srat(buf.BX, i, j);
                if (fx == NEG_INF || bx == NEG_INF) continue;
                grad.gap_extend_b -= std::exp(fx + bx - log_z_);
            }
        }

        // gap_extend_a: expected # of Y-state steps (gap in A)
        for (int i = 0; i <= m_; ++i) {
            for (int j = std::max(1, jlo0(i)); j <= jhi0(i); ++j) {
                double fy = srat(buf.FY, i, j), by = srat(buf.BY, i, j);
                if (fy == NEG_INF || by == NEG_INF) continue;
                grad.gap_extend_a -= std::exp(fy + by - log_z_);
            }
        }

        // gap_open_b: expected # of B-gap openings (M→X or Y→X transitions).  Local
        // border X cells are unreachable (FX = -inf there), so no opening lands on one.
        for (int i = 1; i <= m_; ++i) {
            for (int j = std::max(kGapTargetMin, jlo0(i)); j <= jhi0(i); ++j) {
                double bx = srat(buf.BX, i, j);
                if (bx == NEG_INF) continue;
                double fm = srat(buf.FM, i-1, j), fy_prev = srat(buf.FY, i-1, j);
                double log_open = lse2(fm, fy_prev)
                                  - params_->gap_open_b - params_->gap_extend_b
                                  + bx - log_z_;
                if (log_open > -700) grad.gap_open_b -= std::exp(log_open);
            }
        }

        // gap_open_a: expected # of A-gap openings (M→Y or X→Y transitions)
        for (int i = kGapTargetMin; i <= m_; ++i) {
            for (int j = std::max(1, jlo0(i)); j <= jhi0(i); ++j) {
                double by = srat(buf.BY, i, j);
                if (by == NEG_INF) continue;
                double fm = srat(buf.FM, i, j-1), fx_prev = srat(buf.FX, i, j-1);
                double log_open = lse2(fm, fx_prev)
                                  - params_->gap_open_a - params_->gap_extend_a
                                  + by - log_z_;
                if (log_open > -700) grad.gap_open_a -= std::exp(log_open);
            }
        }
    }

    // ═════════════════════════════════════════════════════════════════════════
    // Forward-backward — scaled probability space (SoftImpl::Scaled)
    // ═════════════════════════════════════════════════════════════════════════
    //
    // The same recurrences as fwdbwd_linear/fwdbwd_affine with exp() applied: every
    // log-space lse becomes a +, every +score a ×exp(score), so a cell costs only
    // multiply-adds.  exp() runs |Σ|² + 4 times per problem, log() once.
    //
    // Range.  Each forward row i is stored times 2^-S[i] (S[i] cumulative; the factor
    // is a power of two, so rescaling is exact; the row max lands in [1, 2)), each
    // backward row times 2^-T[i], and Z = 2^ze · zr.  Posteriors are then
    // F̂·B̂ · 2^(S+T-ze)/zr, an ldexp — no exp.
    //
    // All terms are non-negative, so normal values carry full relative precision, and
    // the only error is MASS LOST to the bottom of the double range: a term that went
    // subnormal or to 0 (a deep cell, a weight like exp(-100), a local free start
    // 2^-S).  Any lost term is below DBL_MIN in its row's scaled units (times 2^-k if
    // the rescale scaled the row UP), and mass lost at a cell moves any posterior —
    // and Z — by at most its own posterior, F_lost·B/Z = lost · B̂ · 2^(S+T-ze)/zr.
    // With B̂ (or F̂) at most 2 after rescaling, each row's possible damage is bounded
    // by quantities already in hand, so the run sums that bound and fails (returns
    // false) when it exceeds 2^-45 (kSpLossTol), or when a weight or a row overflows.  The
    // bound is rigorous, not a heuristic: a pair passes exactly when nothing that was
    // lost could matter at the tolerance.  What fails in practice: a row whose forward
    // and backward mass sit ~1000 binary orders apart (extreme score ranges, very low
    // temperature), or a step score beyond ~+709 (exp overflows).  SoftImpl decides
    // whether that throws or falls back to the log path.

    void run_fwdbwd(DpBuffer& buf) {
        soft_counts_ready_ = false;
        if (soft_impl_ != SoftImpl::Log) {
            bool ok;
            ok = fwdbwd_scaled(buf);
            if (ok) {
                soft_scaled_used_ = soft_counts_ready_ = true;
                log_z_ *= soft_temp_;
                return;
            }
            if (soft_impl_ == SoftImpl::Scaled)
                throw std::domain_error(
                    "nwgrad: soft path: this pair's dynamic range does not fit the scaled "
                    "probability-space forward-backward (lengths " + std::to_string(m_) +
                    " x " + std::to_string(n_) + "; mass lost to underflow could move the "
                    "result by more than 2^-45, or a step score exceeds ~709 so exp() "
                    "overflows — typically a very low temperature or extreme scores). "
                    "Use soft_impl=\"scaled_or_log\" (falls back per pair) or \"log\".");
        }
        soft_scaled_used_ = false;
        ensure_fwdbwd_buf(buf);
        if (soft_temp_ == 1.0) {
            if constexpr (GM == GapModel::Linear) fwdbwd_linear(buf);
            else                                   fwdbwd_affine(buf);
            if (want_post_) posterior_rows_log(buf);
            return;   // soft_grad() sweeps the tables lazily, as it always has
        }
        // T != 1: run the log path on θ/T, and sweep the gradient NOW, while params_ and
        // log_z_ still describe θ/T; the counts are kept like the scaled path's.
        const AlignParams* orig = params_;
        const double* orig_blk = blk_;
        const AlignParams tp = (*orig) * (1.0 / soft_temp_);
        params_ = &tp;
        blk_ = tp.matrix.data();
        try {
            AlignParams g = AlignParams::zeros_like(tp);
            if constexpr (GM == GapModel::Linear) { fwdbwd_linear(buf); soft_grad_linear(buf, g); }
            else                                   { fwdbwd_affine(buf); soft_grad_affine(buf, g); }
            if (want_post_) posterior_rows_log(buf);
            const size_t nn = static_cast<size_t>(nalpha_) * nalpha_;
            scnt_.assign(g.matrix.data(), g.matrix.data() + nn);
            sg_go_a_ = g.gap_open_a; sg_ge_a_ = g.gap_extend_a;
            sg_go_b_ = g.gap_open_b; sg_ge_b_ = g.gap_extend_b;
        } catch (...) { params_ = orig; blk_ = orig_blk; throw; }
        params_ = orig;
        blk_ = orig_blk;
        log_z_ *= soft_temp_;
        soft_counts_ready_ = true;
    }

    void add_scaled_counts(AlignParams& grad) const {
        double* gblk = grad_block(grad);
        const size_t nn = static_cast<size_t>(nalpha_) * nalpha_;
        for (size_t k = 0; k < nn; ++k) gblk[k] += scnt_[k];
        grad.gap_open_a   += sg_go_a_;
        grad.gap_extend_a += sg_ge_a_;
        grad.gap_open_b   += sg_go_b_;
        grad.gap_extend_b += sg_ge_b_;
    }

    // Weights only need to be finite: a weight that underflows is lost mass, which
    // the per-row bound accounts for.
    static bool weight_bad(double w) noexcept {
        return !(w <= std::numeric_limits<double>::max());
    }
    // exp of the substitution block; false if any weight is out of range.
    bool prepare_scaled() {
        const size_t nn = static_cast<size_t>(nalpha_) * nalpha_;
        es_.resize(nn);
        scnt_.assign(nn, 0.0);
        srow_.resize(static_cast<size_t>(nalpha_));
        sg_go_a_ = sg_ge_a_ = sg_go_b_ = sg_ge_b_ = 0.0;
        bool bad = false;
        const double it = 1.0 / soft_temp_;
        for (size_t k = 0; k < nn; ++k) { es_[k] = std::exp(blk_[k] * it); bad |= weight_bad(es_[k]); }
        return !bad;
    }

    // Query profile: prof[a·w + j] = exp(score(a, b[j-1]) / T) for j = 1..n, 0 at j = 0
    // and j = n+1, so a row reads its substitution weights contiguously.
    const double* build_profile(DpBuffer& buf, size_t w) {
        const size_t na = static_cast<size_t>(nalpha_);
        buf.sqp.assign(na * w, 0.0);
        double* P = buf.sqp.data();
        for (size_t a = 0; a < na; ++a) {
            const double* er = es_.data() + a * na;
            double* pr = P + a * w;
            for (int j = 1; j <= n_; ++j) pr[j] = er[b_idx_[static_cast<size_t>(j) - 1]];
        }
        return P;
    }

    // soft_guide="posterior" on the log path: row i's column of greatest EXIT mass (see
    // SoftPairJob::gpost), from the full log tables (first on ties, over the band) — as
    // the scaled kernels.  Each row is taken relative to its largest posterior.
    void posterior_rows_log(const DpBuffer& buf) {
        post_rows_.assign(static_cast<size_t>(m_) + 1, 0);
        std::vector<double> lp, ly;
        const double lea = -params_->gap_extend_a;   // log ea (params_ is theta/T here)
        for (int i = 0; i <= m_; ++i) {
            const int lo = jlo0(i), hi = jhi0(i);
            if (lo > hi) continue;
            lp.assign(static_cast<size_t>(hi - lo + 2), NEG_INF);
            ly.assign(static_cast<size_t>(hi - lo + 2), NEG_INF);
            double rmax = NEG_INF;
            for (int j = lo; j <= hi; ++j) {
                double v, y;
                if constexpr (GM == GapModel::Linear) {
                    v = srat(buf.F, i, j) + srat(buf.B, i, j);
                    y = (j < hi) ? srat(buf.F, i, j) + lea + srat(buf.B, i, j + 1) : NEG_INF;
                } else {
                    v = lse3(srat(buf.FM, i, j) + srat(buf.BM, i, j),
                             srat(buf.FX, i, j) + srat(buf.BX, i, j),
                             srat(buf.FY, i, j) + srat(buf.BY, i, j));
                    y = (j < hi) ? srat(buf.FY, i, j + 1) + srat(buf.BY, i, j + 1) : NEG_INF;
                }
                lp[static_cast<size_t>(j - lo)] = v; ly[static_cast<size_t>(j - lo)] = y;
                rmax = std::max(rmax, v);
            }
            double bv = -1.0; int bj = lo;
            for (int j = lo; j <= hi; ++j) {
                const double e = std::exp(lp[static_cast<size_t>(j - lo)] - rmax) -
                                 std::exp(ly[static_cast<size_t>(j - lo)] - rmax);
                if (e > bv) { bv = e; bj = j; }
            }
            post_rows_[static_cast<size_t>(i)] = bj;
        }
    }

    // Both gap models: build the SoftPairJob and run the per-pair kernel of this
    // aligner's backend (soft_kernel_impl.inl; the baseline copy when the backend is
    // scalar_fallback or its level has none).  False: out of range (see run_fwdbwd).
    bool fwdbwd_scaled(DpBuffer& buf) {
        if (!prepare_scaled()) return false;
        constexpr bool lin = (GM == GapModel::Linear);
        const double it = 1.0 / soft_temp_;
        SoftPairJob J{};
        if (want_post_) { post_rows_.assign(static_cast<size_t>(m_) + 1, 0); J.gpost = post_rows_.data(); }
        J.ea = std::exp(-params_->gap_extend_a * it);
        J.eb = std::exp(-params_->gap_extend_b * it);
        if (!lin) {
            J.oa = std::exp(-(params_->gap_open_a + params_->gap_extend_a) * it);
            J.ob = std::exp(-(params_->gap_open_b + params_->gap_extend_b) * it);
            if (weight_bad(J.oa) || weight_bad(J.ob)) return false;
        }
        if (weight_bad(J.ea) || weight_bad(J.eb)) return false;
        if constexpr (lin) { if (buf.F.size() < sz_) buf.F.resize(sz_); }
        else for (auto* v : {&buf.FM, &buf.FX, &buf.FY}) if (v->size() < sz_) v->resize(sz_);
        const size_t w = static_cast<size_t>(n_) + 2;
        for (DVec* r : {&buf.sb0, &buf.sb1, &buf.sb2, &buf.sb3, &buf.sb4, &buf.sb5})
            if (r->size() < w) r->resize(w);
        const size_t mm = static_cast<size_t>(m_) + 1;
        sband_.resize(6 * mm);
        int* bl = sband_.data();
        for (int i = 0; i <= m_; ++i) {
            bl[i] = jlo(i); bl[mm + i] = jhi(i); bl[2 * mm + i] = jlo0(i); bl[3 * mm + i] = jhi0(i);
            int lo, hi;
            if constexpr (AB == AlignBand::Full) { lo = 0; hi = n_; } else band_row_span(i, lo, hi);
            bl[4 * mm + i] = lo; bl[5 * mm + i] = hi;
        }
        buf.sexp.resize(mm);
        srs_.resize(2 * mm);
        J.m = m_; J.n = n_; J.nalpha = nalpha_;
        J.a = a_idx_.data(); J.b = b_idx_.data();
        J.P = build_profile(buf, w);
        J.local = (AM == AlignMode::Local) ? 1 : 0;
        J.full = (AB == AlignBand::Full) ? 1 : 0;
        J.bi = border_rows(); J.bj = border_cols();
        J.jlo = bl; J.jhi = bl + mm; J.jlo0 = bl + 2 * mm; J.jhi0 = bl + 3 * mm;
        J.slo = bl + 4 * mm; J.shi = bl + 5 * mm;
        J.stride = stride_;
        if constexpr (lin) J.FM = buf.F.data();
        else { J.FM = buf.FM.data(); J.FX = buf.FX.data(); J.FY = buf.FY.data(); }
        J.r0 = buf.sb0.data(); J.r1 = buf.sb1.data(); J.r2 = buf.sb2.data();
        J.r3 = buf.sb3.data(); J.r4 = buf.sb4.data(); J.r5 = buf.sb5.data();
        J.S = buf.sexp.data(); J.rowsum = srs_.data(); J.fmax = srs_.data() + mm;
        J.scnt = scnt_.data(); J.srow = srow_.data();
        soft_pair_fn fn = lin ? &nwgrad_soft_base::soft_pair_linear : &nwgrad_soft_base::soft_pair_affine;
        const int be = (backend_ == kBackendAuto) ? global_default_backend() : backend_;
        if (be >= 0) {
            const LevelKernels& K = level_kernels(be);
            soft_pair_fn f = lin ? K.soft_pair_linear : K.soft_pair_affine;
            if (f) fn = f;
        }
        fn(J);
        if (!J.ok) return false;
        log_z_ = J.log_z;
        sg_go_a_ = J.g_oa; sg_ge_a_ = J.g_ea; sg_go_b_ = J.g_ob; sg_ge_b_ = J.g_eb;
        return true;
    }
};

// The inter-pair soft pass's weights (InterSoftJob es/oa/ea/ob/eb, linear) for `P` at
// temperature T: exp(score / T), exp(-(go+ge)/T), exp(-ge/T).  False when one overflows
// — the pairs then run their own path (which raises or falls back per soft_impl).
// Shared by SeqPairBatch and BatchAligner so both batch the same weights.
inline bool inter_soft_weights(const AlignParams& P, double T, bool lin,
                               std::vector<double>& es, InterSoftJob& sj) {
    constexpr double big = std::numeric_limits<double>::max();
    const double it = 1.0 / T;
    const size_t nn = static_cast<size_t>(P.matrix.size()) * P.matrix.size();
    es.resize(nn);
    bool fin = true;
    for (size_t k = 0; k < nn; ++k) { es[k] = std::exp(P.matrix.data()[k] * it); fin &= es[k] <= big; }
    sj.oa = std::exp(-(P.gap_open_a + P.gap_extend_a) * it);
    sj.ea = std::exp(-P.gap_extend_a * it);
    sj.ob = std::exp(-(P.gap_open_b + P.gap_extend_b) * it);
    sj.eb = std::exp(-P.gap_extend_b * it);
    sj.linear = lin ? 1 : 0;
    sj.es = es.data(); sj.nalpha = P.matrix.size();
    return fin && sj.ea <= big && sj.eb <= big && (lin || (sj.oa <= big && sj.ob <= big));
}

// The Simd kernel's out-of-line definitions.  Included last, once Aligner is a
// complete type, so every instantiation sees both kernels.
#define NWGRAD_ALIGNER_HPP_INCLUDED 1
#include "aligner_simd.hpp"
