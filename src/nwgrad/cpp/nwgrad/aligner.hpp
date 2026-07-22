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
            if (guide_j.empty()) {
                guide_j_.resize(static_cast<size_t>(m_ + 1));
                if (m_ > 0)
                    for (int i = 0; i <= m_; ++i)
                        guide_j_[i] = static_cast<int>(
                            std::llround(static_cast<double>(i) * n_ / m_));
                else
                    guide_j_[0] = 0;
            } else {
                guide_j_ = std::move(guide_j);
            }
        }
        problem_set_      = true;
        viterbi_done_     = false;
        fwdbwd_done_      = false;
        any_viterbi_done_ = false;
        any_fwdbwd_done_  = false;
        fwdbwd_is_newest_ = false;
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
    int  kernel() const noexcept { return backend_; }

    void compute_viterbi() {
        check_problem();
        check_own_buf_allocated();
        ensure_viterbi_buf(own_buf_);  // Grow if needed
        run_viterbi(own_buf_);
        viterbi_done_ = any_viterbi_done_ = true;
        fwdbwd_is_newest_ = false;
    }

    void compute_forward_back() {
        check_problem();
        check_own_buf_allocated();
        ensure_fwdbwd_buf(own_buf_);  // Grow if needed
        tables_striped_ = false; // F/B are always row-major, even after a striped Viterbi
        if constexpr (GM == GapModel::Linear) fwdbwd_linear(own_buf_);
        else                                   fwdbwd_affine(own_buf_);
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
        ensure_viterbi_buf(buf);  // Grow if needed (allows implicit growth from size 0)
        run_viterbi(buf);
        any_viterbi_done_ = true;
        fwdbwd_is_newest_ = false;
    }

    void compute_forward_back(DpBuffer& buf) {
        check_problem();
        ensure_fwdbwd_buf(buf);  // Grow if needed (allows implicit growth from size 0)
        tables_striped_ = false; // F/B are always row-major, even after a striped Viterbi
        if constexpr (GM == GapModel::Linear) fwdbwd_linear(buf);
        else                                   fwdbwd_affine(buf);
        any_fwdbwd_done_ = true;
        fwdbwd_is_newest_ = true;
    }

    std::vector<int> guide_j_from_viterbi(const DpBuffer& buf) const {
        if constexpr (GM == GapModel::Linear) return guide_j_linear(buf);
        else                                   return guide_j_affine(buf);
    }

    std::pair<std::string, std::string> aligned(const DpBuffer& buf) const {
        std::string a, b;
        if constexpr (GM == GapModel::Linear) aligned_linear(buf, a, b);
        else                                   aligned_affine(buf, a, b);
        return {std::move(a), std::move(b)};
    }

    void hard_grad(const DpBuffer& buf, AlignParams& grad) const {
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
        std::vector<E> out(static_cast<size_t>(m_ + 1) * rm_stride);
        for (int i = 0; i <= m_; ++i)
            for (int j = 0; j <= n_; ++j)
                out[static_cast<size_t>(i) * rm_stride + j] = rat(t, i, j);
        return out;
    }

    void soft_grad(const DpBuffer& buf, AlignParams& grad) const {
        if constexpr (GM == GapModel::Linear) soft_grad_linear(buf, grad);
        else                                   soft_grad_affine(buf, grad);
    }

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
    void check_external_viterbi_buf(const DpBuffer& buf) const {
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

    // Grow external buffer to fit current problem (thread-owned path).
    void ensure_viterbi_buf(DpBuffer& buf) const {
        if constexpr (GM == GapModel::Linear) {
            if (buf.H.size() < sz_) buf.H.resize(sz_);
        } else {
            if (buf.VM.size() < sz_) { buf.VM.resize(sz_); buf.VX.resize(sz_); buf.VY.resize(sz_); }
        }
    }

    void ensure_fwdbwd_buf(DpBuffer& buf) const {
        if constexpr (GM == GapModel::Linear) {
            if (buf.F.size() < sz_) { buf.F.resize(sz_); buf.B.resize(sz_); }
        } else {
            if (buf.FM.size() < sz_) {
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
    size_t cell_index(int i, int j) const noexcept {
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
    int jlo(int i) const noexcept {
        if constexpr (AB == AlignBand::Full) return 1;
        else                                  return std::max(1, guide_j_[i] - band_);
    }
    int jhi(int i) const noexcept {
        if constexpr (AB == AlignBand::Full) return n_;
        else {
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

    std::vector<int> guide_j_linear(const DpBuffer& buf) const {
        std::vector<int> gj(static_cast<size_t>(m_ + 1), -1);
        gj[0] = 0;
        gj[static_cast<size_t>(m_)] = n_;

        int i = (AM == AlignMode::Global) ? m_ : best_i_;
        int j = (AM == AlignMode::Global) ? n_ : best_j_;
        gj[static_cast<size_t>(i)] = j;

        while (true) {
            if constexpr (AM == AlignMode::Global) { if (i == 0 && j == 0) break; }
            else                                    { if (rat(buf.H, i, j) <= 0.0) break; }
            if (i > 0 && j > 0 &&
                rat(buf.H, i, j) == rat(buf.H, i-1, j-1) + sub(i, j))
            {
                --i; --j;
                gj[static_cast<size_t>(i)] = j;
            } else if (i > 0 && rat(buf.H, i, j) == rat(buf.H, i-1, j) - params_->gap_extend_b) {
                --i;
                gj[static_cast<size_t>(i)] = j;
            } else {
                --j;  // gap in a: i stays, no gj update
            }
        }
        fill_guide_gaps(gj);
        return gj;
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
    // no simd kernel — deliberately.  Its recurrence collapses to a single carry that is a
    // pure latency chain (~6 cycles/cell, unbreakable by any vector width), and the scalar
    // loop already runs at ~9 cycles/cell against that floor.  A vectorized linear kernel
    // was written, measured at 0.90x, and deleted.  So any simd backend is a legal request
    // for a linear aligner; it simply runs the fastest linear kernel there is, the scalar.
    void run_viterbi(DpBuffer& buf) {
        // Default to row-major; only the striped Full kernel (run_dispatched_affine) flips
        // this back on.  Every other fill here writes VM/VX/VY row-major.
        tables_striped_ = false;
        if constexpr (GM == GapModel::Linear) {
            viterbi_linear(buf);   // no simd linear kernel: every backend is scalar here
            return;
        }
        // Resolve this aligner's backend: auto -> the global default, then scalar or a level.
        const int backend = (backend_ == kBackendAuto) ? global_default_backend() : backend_;
        if (backend == kBackendScalar) { viterbi_affine(buf); return; }

        // A simd level: its kernels.  If that level's TU was not linked (header-only
        // single-level build), every pointer is null and we fall through to scalar.
        const LevelKernels& K = level_kernels(backend);
        if constexpr (AB == AlignBand::Full) {
            // The striped Full kernel exists at both precisions (viterbi / viterbi_f).
            if constexpr (std::is_same_v<T, double>) {
                if (K.viterbi)   { run_dispatched_affine(buf, K); return; }
            } else {
                if (K.viterbi_f) { run_dispatched_affine(buf, K); return; }
            }
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
    void run_dispatched_affine(DpBuffer& buf, const LevelKernels& K) {
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
        if constexpr (std::is_same_v<T, double>) K.viterbi(job);
        else                                     K.viterbi_f(job);
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

    void viterbi_linear(DpBuffer& buf) {
        banded_fill(buf.H);

        if constexpr (AM == AlignMode::Global) {
            at(buf.H, 0, 0) = 0.0;
            const int bi = border_rows(), bj = border_cols();
            for (int i = 1; i <= bi; ++i) at(buf.H, i, 0) = -i * params_->gap_extend_b;
            for (int j = 1; j <= bj; ++j) at(buf.H, 0, j) = -j * params_->gap_extend_a;
        } else {
            for (int i = 0; i <= m_; ++i) at(buf.H, i, 0) = 0.0;
            for (int j = 0; j <= n_; ++j) at(buf.H, 0, j) = 0.0;
        }

        best_i_ = 0; best_j_ = 0;
        double best_local = 0.0;
        for (int i = 1; i <= m_; ++i) {
            for (int j = jlo(i); j <= jhi(i); ++j) {
                double v = std::max({
                    rat(buf.H, i-1, j-1) + sub(i, j),
                    rat(buf.H, i-1, j)   - params_->gap_extend_b,  // gap in B (advance i)
                    rat(buf.H, i,   j-1) - params_->gap_extend_a,  // gap in A (advance j)
                });
                if constexpr (AM == AlignMode::Local) {
                    v = std::max(v, 0.0);
                    if (v > best_local) { best_local = v; best_i_ = i; best_j_ = j; }
                }
                at(buf.H, i, j) = v;
            }
        }

        viterbi_score_ = (AM == AlignMode::Global) ? rat(buf.H, m_, n_) : best_local;
    }

    std::vector<std::pair<int,int>> traceback_linear(const DpBuffer& buf) const {
        std::vector<std::pair<int,int>> path;
        int i = (AM == AlignMode::Global) ? m_ : best_i_;
        int j = (AM == AlignMode::Global) ? n_ : best_j_;

        while (true) {
            if (i == 0 && j == 0) break;
            if constexpr (AM == AlignMode::Local)
                if (rat(buf.H, i, j) <= 0.0) break;
            if (i > 0 && j > 0 &&
                rat(buf.H, i, j) == rat(buf.H, i-1, j-1) + sub(i, j))
            {
                path.emplace_back(i-1, j-1);
                --i; --j;
            } else if (i > 0 && rat(buf.H, i, j) == rat(buf.H, i-1, j) - params_->gap_extend_b) {
                --i;  // gap in B
            } else {
                --j;  // gap in A
            }
        }
        std::reverse(path.begin(), path.end());
        return path;
    }

    void aligned_linear(const DpBuffer& buf, std::string& a, std::string& b) const {
        int i = (AM == AlignMode::Global) ? m_ : best_i_;
        int j = (AM == AlignMode::Global) ? n_ : best_j_;

        while (true) {
            if (i == 0 && j == 0) break;
            if constexpr (AM == AlignMode::Local)
                if (rat(buf.H, i, j) <= 0.0) break;
            if (i > 0 && j > 0 &&
                rat(buf.H, i, j) == rat(buf.H, i-1, j-1) + sub(i, j))
            {
                a.push_back(sym_a(i)); b.push_back(sym_b(j));
                --i; --j;
            } else if (i > 0 && rat(buf.H, i, j) == rat(buf.H, i-1, j) - params_->gap_extend_b) {
                a.push_back(sym_a(i)); b.push_back('-');  // gap in B
                --i;
            } else {
                a.push_back('-'); b.push_back(sym_b(j));  // gap in A
                --j;
            }
        }
        std::reverse(a.begin(), a.end());
        std::reverse(b.begin(), b.end());
    }

    void hard_grad_linear(const DpBuffer& buf, AlignParams& grad) const {
        double* gblk = grad_block(grad);
        int i = (AM == AlignMode::Global) ? m_ : best_i_;
        int j = (AM == AlignMode::Global) ? n_ : best_j_;

        while (true) {
            if (i == 0 && j == 0) break;
            if constexpr (AM == AlignMode::Local)
                if (rat(buf.H, i, j) <= 0.0) break;
            if (i > 0 && j > 0 &&
                rat(buf.H, i, j) == rat(buf.H, i-1, j-1) + sub(i, j))
            {
                gblk[sub_off(i, j)] += 1.0;
                --i; --j;
            } else if (i > 0 && rat(buf.H, i, j) == rat(buf.H, i-1, j) - params_->gap_extend_b) {
                grad.gap_extend_b -= 1.0;   // the score subtracts this penalty
                --i;
            } else {
                grad.gap_extend_a -= 1.0;
                --j;
            }
        }
    }

    // ═════════════════════════════════════════════════════════════════════════
    // Viterbi — Affine gap model
    // ═════════════════════════════════════════════════════════════════════════

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

    void hard_grad_affine(const DpBuffer& buf, AlignParams& grad) const {
        double* gblk = grad_block(grad);
        int i = best_i_, j = best_j_;
        TBTable tbl = best_tbl_;
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
    }

    // ═════════════════════════════════════════════════════════════════════════
    // Forward-backward — Linear gap model
    // ═════════════════════════════════════════════════════════════════════════

    void fwdbwd_linear(DpBuffer& buf) {
        // ── Forward ──
        banded_fill(buf.F);

        if constexpr (AM == AlignMode::Global) {
            at(buf.F, 0, 0) = 0.0;
            const int bi = border_rows(), bj = border_cols();
            for (int i = 1; i <= bi; ++i)
                at(buf.F, i, 0) = -static_cast<double>(i) * params_->gap_extend_b;
            for (int j = 1; j <= bj; ++j)
                at(buf.F, 0, j) = -static_cast<double>(j) * params_->gap_extend_a;
        } else {
            for (int i = 0; i <= m_; ++i) at(buf.F, i, 0) = 0.0;
            for (int j = 0; j <= n_; ++j) at(buf.F, 0, j) = 0.0;
        }

        for (int i = 1; i <= m_; ++i) {
            for (int j = jlo(i); j <= jhi(i); ++j) {
                double v = lse3(
                    rat(buf.F, i-1, j-1) + sub(i, j),
                    rat(buf.F, i-1, j)   - params_->gap_extend_b,
                    rat(buf.F, i,   j-1) - params_->gap_extend_a
                );
                if constexpr (AM == AlignMode::Local) v = lse2(v, 0.0);
                at(buf.F, i, j) = v;
            }
        }

        if constexpr (AM == AlignMode::Global) {
            log_z_ = rat(buf.F, m_, n_);
        } else {
            log_z_ = NEG_INF;
            for (int i = 0; i <= m_; ++i)
                for (int j = jlo0(i); j <= jhi0(i); ++j)
                    log_z_ = lse2(log_z_, rat(buf.F, i, j));
        }

        // ── Backward ──
        if constexpr (AM == AlignMode::Global) {
            band_fill(buf.B, NEG_INF);
            at(buf.B, m_, n_) = 0.0;
        } else {
            band_fill(buf.B, 0.0);
        }

        for (int i = m_; i >= 0; --i) {
            for (int j = jhi0(i); j >= jlo0(i); --j) {
                double bval = rat(buf.B, i, j);
                if (bval == NEG_INF) continue;
                if (i > 0 && j > 0)
                    at(buf.B,i-1,j-1) = lse2(rat(buf.B,i-1,j-1),
                                              bval + sub(i, j));
                if (i > 0)
                    at(buf.B,i-1,j)   = lse2(rat(buf.B,i-1,j),   bval - params_->gap_extend_b);
                if (j > 0)
                    at(buf.B,i,j-1)   = lse2(rat(buf.B,i,j-1),   bval - params_->gap_extend_a);
            }
        }
    }

    void soft_grad_linear(const DpBuffer& buf, AlignParams& grad) const {
        double* gblk = grad_block(grad);
        // Matrix gradient: match steps (i-1,j-1) → (i,j)
        for (int i = 1; i <= m_; ++i) {
            for (int j = jlo(i); j <= jhi(i); ++j) {
                double bval = rat(buf.B, i, j);
                if (bval == NEG_INF) continue;
                double log_p = rat(buf.F, i-1, j-1)
                               + sub(i, j)
                               + bval - log_z_;
                gblk[sub_off(i, j)] += std::exp(log_p);
            }
        }

        // gap_extend_b: B-gap steps (i-1,j) → (i,j), cost = -gap_extend_b
        for (int i = 1; i <= m_; ++i) {
            for (int j = jlo0(i); j <= jhi0(i); ++j) {
                double bval = rat(buf.B, i, j);
                if (bval == NEG_INF) continue;
                double fval = rat(buf.F, i-1, j);
                if (fval == NEG_INF) continue;
                grad.gap_extend_b -= std::exp(fval - params_->gap_extend_b + bval - log_z_);
            }
        }

        // gap_extend_a: A-gap steps (i,j-1) → (i,j), cost = -gap_extend_a
        for (int i = 0; i <= m_; ++i) {
            for (int j = std::max(1, jlo0(i)); j <= jhi0(i); ++j) {
                double bval = rat(buf.B, i, j);
                if (bval == NEG_INF) continue;
                double fval = rat(buf.F, i, j-1);
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
            at(buf.FM, 0, 0) = 0.0;
            const int bi = border_rows(), bj = border_cols();
            for (int i = 1; i <= bi; ++i)
                at(buf.FX, i, 0) = -(params_->gap_open_b + static_cast<double>(i) * params_->gap_extend_b);
            for (int j = 1; j <= bj; ++j)
                at(buf.FY, 0, j) = -(params_->gap_open_a + static_cast<double>(j) * params_->gap_extend_a);
        } else {
            for (int i = 0; i <= m_; ++i) at(buf.FM, i, 0) = 0.0;
            for (int j = 0; j <= n_; ++j) at(buf.FM, 0, j) = 0.0;
        }

        for (int i = 1; i <= m_; ++i) {
            for (int j = jlo(i); j <= jhi(i); ++j) {
                double diag  = lse3(rat(buf.FM,i-1,j-1), rat(buf.FX,i-1,j-1), rat(buf.FY,i-1,j-1));
                double m_val = diag + sub(i, j);
                double x_val = lse3(
                    rat(buf.FM,i-1,j) - params_->gap_open_b - params_->gap_extend_b,
                    rat(buf.FX,i-1,j)                       - params_->gap_extend_b,
                    rat(buf.FY,i-1,j) - params_->gap_open_b - params_->gap_extend_b);
                double y_val = lse3(
                    rat(buf.FM,i,j-1) - params_->gap_open_a - params_->gap_extend_a,
                    rat(buf.FX,i,j-1) - params_->gap_open_a - params_->gap_extend_a,
                    rat(buf.FY,i,j-1)                       - params_->gap_extend_a);
                if constexpr (AM == AlignMode::Local) m_val = lse2(m_val, 0.0);
                at(buf.FM, i, j) = m_val; at(buf.FX, i, j) = x_val; at(buf.FY, i, j) = y_val;
            }
        }

        if constexpr (AM == AlignMode::Global) {
            log_z_ = lse3(rat(buf.FM,m_,n_), rat(buf.FX,m_,n_), rat(buf.FY,m_,n_));
        } else {
            log_z_ = NEG_INF;
            for (int i = 0; i <= m_; ++i)
                for (int j = jlo0(i); j <= jhi0(i); ++j)
                    log_z_ = lse2(log_z_, lse3(rat(buf.FM,i,j), rat(buf.FX,i,j), rat(buf.FY,i,j)));
        }

        // ── Backward ──
        if constexpr (AM == AlignMode::Global) {
            band_fill(buf.BM, NEG_INF);
            band_fill(buf.BX, NEG_INF);
            band_fill(buf.BY, NEG_INF);
            at(buf.BM,m_,n_) = 0.0; at(buf.BX,m_,n_) = 0.0; at(buf.BY,m_,n_) = 0.0;
        } else {
            band_fill(buf.BM, 0.0);
            band_fill(buf.BX, 0.0);
            band_fill(buf.BY, 0.0);
        }

        for (int i = m_; i >= 0; --i) {
            for (int j = jhi0(i); j >= jlo0(i); --j) {
                double bm = rat(buf.BM,i,j), bx = rat(buf.BX,i,j), by = rat(buf.BY,i,j);

                if (bm != NEG_INF && i > 0 && j > 0) {
                    double contrib = bm + sub(i, j);
                    at(buf.BM,i-1,j-1) = lse2(rat(buf.BM,i-1,j-1), contrib);
                    at(buf.BX,i-1,j-1) = lse2(rat(buf.BX,i-1,j-1), contrib);
                    at(buf.BY,i-1,j-1) = lse2(rat(buf.BY,i-1,j-1), contrib);
                }
                // X state (gap in B) contribution to predecessors at (i-1,j)
                if (bx != NEG_INF && i > 0) {
                    at(buf.BM,i-1,j) = lse2(rat(buf.BM,i-1,j),
                                             bx - params_->gap_open_b - params_->gap_extend_b);
                    at(buf.BX,i-1,j) = lse2(rat(buf.BX,i-1,j), bx - params_->gap_extend_b);
                    at(buf.BY,i-1,j) = lse2(rat(buf.BY,i-1,j),
                                             bx - params_->gap_open_b - params_->gap_extend_b);
                }
                // Y state (gap in A) contribution to predecessors at (i,j-1)
                if (by != NEG_INF && j > 0) {
                    at(buf.BM,i,j-1) = lse2(rat(buf.BM,i,j-1),
                                             by - params_->gap_open_a - params_->gap_extend_a);
                    at(buf.BX,i,j-1) = lse2(rat(buf.BX,i,j-1),
                                             by - params_->gap_open_a - params_->gap_extend_a);
                    at(buf.BY,i,j-1) = lse2(rat(buf.BY,i,j-1), by - params_->gap_extend_a);
                }
            }
        }
    }

    void soft_grad_affine(const DpBuffer& buf, AlignParams& grad) const {
        double* gblk = grad_block(grad);
        // Matrix gradient: match steps
        for (int i = 1; i <= m_; ++i) {
            for (int j = jlo(i); j <= jhi(i); ++j) {
                double bm = rat(buf.BM, i, j);
                if (bm == NEG_INF) continue;
                double pred  = lse3(rat(buf.FM,i-1,j-1), rat(buf.FX,i-1,j-1), rat(buf.FY,i-1,j-1));
                double log_p = pred + sub(i, j) + bm - log_z_;
                gblk[sub_off(i, j)] += std::exp(log_p);
            }
        }

        // gap_extend_b: expected # of X-state steps (gap in B)
        for (int i = 1; i <= m_; ++i) {
            for (int j = jlo0(i); j <= jhi0(i); ++j) {
                double fx = rat(buf.FX, i, j), bx = rat(buf.BX, i, j);
                if (fx == NEG_INF || bx == NEG_INF) continue;
                grad.gap_extend_b -= std::exp(fx + bx - log_z_);
            }
        }

        // gap_extend_a: expected # of Y-state steps (gap in A)
        for (int i = 0; i <= m_; ++i) {
            for (int j = std::max(1, jlo0(i)); j <= jhi0(i); ++j) {
                double fy = rat(buf.FY, i, j), by = rat(buf.BY, i, j);
                if (fy == NEG_INF || by == NEG_INF) continue;
                grad.gap_extend_a -= std::exp(fy + by - log_z_);
            }
        }

        // gap_open_b: expected # of B-gap openings (M→X or Y→X transitions)
        for (int i = 1; i <= m_; ++i) {
            for (int j = jlo0(i); j <= jhi0(i); ++j) {
                double bx = rat(buf.BX, i, j);
                if (bx == NEG_INF) continue;
                double fm = rat(buf.FM, i-1, j), fy_prev = rat(buf.FY, i-1, j);
                double log_open = lse2(fm, fy_prev)
                                  - params_->gap_open_b - params_->gap_extend_b
                                  + bx - log_z_;
                if (log_open > -700) grad.gap_open_b -= std::exp(log_open);
            }
        }

        // gap_open_a: expected # of A-gap openings (M→Y or X→Y transitions)
        for (int i = 0; i <= m_; ++i) {
            for (int j = std::max(1, jlo0(i)); j <= jhi0(i); ++j) {
                double by = rat(buf.BY, i, j);
                if (by == NEG_INF) continue;
                double fm = rat(buf.FM, i, j-1), fx_prev = rat(buf.FX, i, j-1);
                double log_open = lse2(fm, fx_prev)
                                  - params_->gap_open_a - params_->gap_extend_a
                                  + by - log_z_;
                if (log_open > -700) grad.gap_open_a -= std::exp(log_open);
            }
        }
    }
};

// The Simd kernel's out-of-line definitions.  Included last, once Aligner is a
// complete type, so every instantiation sees both kernels.
#define NWGRAD_ALIGNER_HPP_INCLUDED 1
#include "aligner_simd.hpp"
