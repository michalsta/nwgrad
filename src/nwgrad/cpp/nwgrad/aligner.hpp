#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include "align_params.hpp"

enum class GapModel  { Linear, Affine };
enum class AlignMode { Global, Local  };
enum class AlignBand { Full,   GuideBanded };

// ── Utility: convert a pair of aligned strings to a guide_j vector ────────────
//
// a_aligned, b_aligned: equal-length strings with '-' for gaps.
// Returns a vector of length (m+1) where m = non-gap chars in a_aligned.
// guide_j[i] = the column j in the DP table after consuming i characters of a.
// Pass the result to Aligner::set_problem when using AlignBand::GuideBanded.
//
// When no explicit guide is given to set_problem, the trivial alignment
// guide_j[i] = i is used, which is equivalent to the old diagonal banded DP.

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

    void clear() noexcept {
        auto clr = [](std::vector<double>& v) noexcept { v.clear(); v.shrink_to_fit(); };
        clr(H);
        clr(VM); clr(VX); clr(VY);
        clr(F);  clr(B);
        clr(FM); clr(FX); clr(FY); clr(BM); clr(BX); clr(BY);
    }
};

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
//   If guide_j is omitted (or empty), the trivial diagonal guide guide_j[i] = i is
//   used — identical in effect to the old diagonal-only banded DP, but works correctly
//   for unequal-length sequences when band >= |m-n|.
//   For a custom guide, compute it with guide_j_from_aligned(a_aligned, b_aligned);
//   the endpoint (m, n) is always in-band as long as the guide is a complete alignment.
//
// In GuideBanded mode the full (m+1)×(n+1) table is still allocated; only the fill
// and backward loops are restricted.  Out-of-band cells are initialised to NEG_INF
// so they cannot corrupt max/lse operations along band edges.

template<GapModel GM, AlignMode AM, AlignBand AB = AlignBand::Full>
struct Aligner {

    // ── Public pipeline API — own internal buffer ─────────────────────────────

    void set_problem(std::string_view a, std::string_view b,
                     const AlignParams& params,
                     int band = 0,
                     std::vector<int> guide_j = {}) {
        seq_a_  = a;
        seq_b_  = b;
        params_ = &params;
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
    }

    // Pre-allocate the own DP buffer for sequences of length m×n.
    // Must be called before compute_viterbi() / compute_forward_back() with own buffer.
    // with_fwdbwd=false skips forward-backward tables (sufficient for hard/no-grad modes).
    void alloc_own_buf(int m, int n, bool with_fwdbwd = true) {
        size_t sz = static_cast<size_t>(m + 1) * static_cast<size_t>(n + 1);
        if constexpr (GM == GapModel::Linear) {
            own_buf_.H.resize(sz);
            if (with_fwdbwd) { own_buf_.F.resize(sz); own_buf_.B.resize(sz); }
        } else {
            own_buf_.VM.resize(sz); own_buf_.VX.resize(sz); own_buf_.VY.resize(sz);
            if (with_fwdbwd) {
                own_buf_.FM.resize(sz); own_buf_.FX.resize(sz); own_buf_.FY.resize(sz);
                own_buf_.BM.resize(sz); own_buf_.BX.resize(sz); own_buf_.BY.resize(sz);
            }
        }
    }

    void compute_viterbi() {
        check_problem();
        ensure_viterbi_buf(own_buf_);
        if constexpr (GM == GapModel::Linear) viterbi_linear(own_buf_);
        else                                   viterbi_affine(own_buf_);
        viterbi_done_ = any_viterbi_done_ = true;
    }

    void compute_forward_back() {
        check_problem();
        ensure_fwdbwd_buf(own_buf_);
        if constexpr (GM == GapModel::Linear) fwdbwd_linear(own_buf_);
        else                                   fwdbwd_affine(own_buf_);
        fwdbwd_done_ = any_fwdbwd_done_ = true;
    }

    // Returns the most recent viterbi score or log Z, regardless of which buffer was used.
    double score() const {
        if (any_viterbi_done_) return viterbi_score_;
        if (any_fwdbwd_done_)  return log_z_;
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

    void hard_grad(AlignParams& grad) const {
        check_viterbi();
        hard_grad(own_buf_, grad);
    }

    void soft_grad(AlignParams& grad) const {
        check_fwdbwd();
        soft_grad(own_buf_, grad);
    }

    // Release own DP table memory and reset computed-state flags.
    // set_problem() must be called again before the next DP run.
    void free_dp() noexcept {
        own_buf_.clear();
        problem_set_      = false;
        viterbi_done_     = false;
        fwdbwd_done_      = false;
        any_viterbi_done_ = false;
        any_fwdbwd_done_  = false;
    }

    // ── Extended API — caller-supplied DpBuffer ───────────────────────────────
    //
    // set_problem() must have been called first.
    // viterbi_done_ / fwdbwd_done_ are NOT touched; callers manage their own state.
    // Scalar results (viterbi_score_, log_z_, best_i_, best_j_, best_tbl_) are
    // always written to the Aligner so score() / log_z() work after these calls.

    void compute_viterbi(DpBuffer& buf) {
        check_problem();
        ensure_viterbi_buf(buf);
        if constexpr (GM == GapModel::Linear) viterbi_linear(buf);
        else                                   viterbi_affine(buf);
        any_viterbi_done_ = true;
    }

    void compute_forward_back(DpBuffer& buf) {
        check_problem();
        ensure_fwdbwd_buf(buf);
        if constexpr (GM == GapModel::Linear) fwdbwd_linear(buf);
        else                                   fwdbwd_affine(buf);
        any_fwdbwd_done_ = true;
    }

    std::vector<int> guide_j_from_viterbi(const DpBuffer& buf) const {
        if constexpr (GM == GapModel::Linear) return guide_j_linear(buf);
        else                                   return guide_j_affine(buf);
    }

    void hard_grad(const DpBuffer& buf, AlignParams& grad) const {
        if constexpr (GM == GapModel::Linear) hard_grad_linear(buf, grad);
        else                                   hard_grad_affine(buf, grad);
    }

    void soft_grad(const DpBuffer& buf, AlignParams& grad) const {
        if constexpr (GM == GapModel::Linear) soft_grad_linear(buf, grad);
        else                                   soft_grad_affine(buf, grad);
    }

private:
    static constexpr double NEG_INF = -std::numeric_limits<double>::infinity();

    // ── Problem state ─────────────────────────────────────────────────────────
    std::string_view    seq_a_, seq_b_;
    const AlignParams*  params_     = nullptr;
    int                 band_       = 0;
    int                 m_ = 0, n_ = 0;
    size_t              stride_ = 0, sz_ = 0;
    std::vector<int>    guide_j_;  // used only when AB == GuideBanded

    bool problem_set_      = false;
    bool viterbi_done_     = false;  // own_buf_ has valid viterbi data
    bool fwdbwd_done_      = false;  // own_buf_ has valid fwdbwd data
    bool any_viterbi_done_ = false;  // viterbi scalar results are valid (either buffer)
    bool any_fwdbwd_done_  = false;  // fwdbwd scalar results are valid (either buffer)

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

    // Throw if own_buf_ viterbi tables are too small (not pre-allocated).
    void check_own_viterbi_buf() const {
        bool ok;
        if constexpr (GM == GapModel::Linear)
            ok = (own_buf_.H.size() >= sz_);
        else
            ok = (own_buf_.VM.size() >= sz_);
        if (!ok)
            throw std::logic_error(
                "nwgrad: own DP buffer not allocated; call alloc_dp() first");
    }

    // Throw if own_buf_ fwdbwd tables are too small (not pre-allocated).
    void check_own_fwdbwd_buf() const {
        bool ok;
        if constexpr (GM == GapModel::Linear)
            ok = (own_buf_.F.size() >= sz_);
        else
            ok = (own_buf_.FM.size() >= sz_);
        if (!ok)
            throw std::logic_error(
                "nwgrad: own DP buffer not allocated; call alloc_dp() first");
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
    double& at(std::vector<double>& t, int i, int j) const {
        return t[static_cast<size_t>(i) * stride_ + static_cast<size_t>(j)];
    }
    double rat(const std::vector<double>& t, int i, int j) const {
        return t[static_cast<size_t>(i) * stride_ + static_cast<size_t>(j)];
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

    // Fill vec[0..sz_) with NEG_INF for GuideBanded (no-op for Full).
    void banded_fill(std::vector<double>& vec) const {
        if constexpr (AB == AlignBand::GuideBanded)
            std::fill(vec.begin(), vec.begin() + static_cast<ptrdiff_t>(sz_), NEG_INF);
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
                rat(buf.H, i, j) == rat(buf.H, i-1, j-1) + params_->matrix.score(seq_a_[i-1], seq_b_[j-1]))
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

        while (true) {
            if (i == 0 && j == 0) break;
            if constexpr (AM == AlignMode::Local)
                if (tbl == TBTable::M && rat(buf.VM, i, j) <= 0.0) break;

            if (tbl == TBTable::M) {
                double vm = rat(buf.VM,i-1,j-1), vx = rat(buf.VX,i-1,j-1), vy = rat(buf.VY,i-1,j-1);
                --i; --j;
                gj[static_cast<size_t>(i)] = j;
                if      (vm >= vx && vm >= vy) tbl = TBTable::M;
                else if (vx >= vy)             tbl = TBTable::X;
                else                            tbl = TBTable::Y;
            } else if (tbl == TBTable::X) {
                // X state: gap in B (advance i), uses gap_b params
                double fm = rat(buf.VM,i-1,j) - params_->gap_open_b - params_->gap_extend_b;
                double fx = rat(buf.VX,i-1,j) - params_->gap_extend_b;
                double fy = rat(buf.VY,i-1,j) - params_->gap_open_b - params_->gap_extend_b;
                --i;
                gj[static_cast<size_t>(i)] = j;
                if      (fm >= fx && fm >= fy) tbl = TBTable::M;
                else if (fx >= fy)             tbl = TBTable::X;
                else                            tbl = TBTable::Y;
            } else {
                // Y state: gap in A (advance j), uses gap_a params
                double fm = rat(buf.VM,i,j-1) - params_->gap_open_a - params_->gap_extend_a;
                double fx = rat(buf.VX,i,j-1) - params_->gap_open_a - params_->gap_extend_a;
                double fy = rat(buf.VY,i,j-1) - params_->gap_extend_a;
                --j;  // gap in a: i stays, no gj update
                if      (fm >= fx && fm >= fy) tbl = TBTable::M;
                else if (fx >= fy)             tbl = TBTable::X;
                else                            tbl = TBTable::Y;
            }
        }
        fill_guide_gaps(gj);
        return gj;
    }

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
                    rat(buf.H, i-1, j-1) + params_->matrix.score(seq_a_[i-1], seq_b_[j-1]),
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
                rat(buf.H, i, j) == rat(buf.H, i-1, j-1) + params_->matrix.score(seq_a_[i-1], seq_b_[j-1]))
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

    void hard_grad_linear(const DpBuffer& buf, AlignParams& grad) const {
        int i = (AM == AlignMode::Global) ? m_ : best_i_;
        int j = (AM == AlignMode::Global) ? n_ : best_j_;

        while (true) {
            if (i == 0 && j == 0) break;
            if constexpr (AM == AlignMode::Local)
                if (rat(buf.H, i, j) <= 0.0) break;
            if (i > 0 && j > 0 &&
                rat(buf.H, i, j) == rat(buf.H, i-1, j-1) + params_->matrix.score(seq_a_[i-1], seq_b_[j-1]))
            {
                grad.matrix.mat[static_cast<unsigned char>(seq_a_[i-1])]
                               [static_cast<unsigned char>(seq_b_[j-1])] += 1.0;
                --i; --j;
            } else if (i > 0 && rat(buf.H, i, j) == rat(buf.H, i-1, j) - params_->gap_extend_b) {
                grad.gap_extend_b += 1.0;
                --i;
            } else {
                grad.gap_extend_a += 1.0;
                --j;
            }
        }
    }

    // ═════════════════════════════════════════════════════════════════════════
    // Viterbi — Affine gap model
    // ═════════════════════════════════════════════════════════════════════════

    void viterbi_affine(DpBuffer& buf) {
        std::fill(buf.VM.begin(), buf.VM.begin() + static_cast<ptrdiff_t>(sz_), NEG_INF);
        std::fill(buf.VX.begin(), buf.VX.begin() + static_cast<ptrdiff_t>(sz_), NEG_INF);
        std::fill(buf.VY.begin(), buf.VY.begin() + static_cast<ptrdiff_t>(sz_), NEG_INF);

        if constexpr (AM == AlignMode::Global) {
            at(buf.VM, 0, 0) = 0.0;
            const int bi = border_rows(), bj = border_cols();
            // VX along column 0: all gaps in B (consuming A), uses gap_b params
            for (int i = 1; i <= bi; ++i)
                at(buf.VX, i, 0) = -(params_->gap_open_b + i * params_->gap_extend_b);
            // VY along row 0: all gaps in A (consuming B), uses gap_a params
            for (int j = 1; j <= bj; ++j)
                at(buf.VY, 0, j) = -(params_->gap_open_a + j * params_->gap_extend_a);
        } else {
            for (int i = 0; i <= m_; ++i) at(buf.VM, i, 0) = 0.0;
            for (int j = 0; j <= n_; ++j) at(buf.VM, 0, j) = 0.0;
        }

        best_i_ = 0; best_j_ = 0; best_tbl_ = TBTable::M;
        double best_local = 0.0;

        for (int i = 1; i <= m_; ++i) {
            for (int j = jlo(i); j <= jhi(i); ++j) {
                double diag  = std::max({rat(buf.VM,i-1,j-1), rat(buf.VX,i-1,j-1), rat(buf.VY,i-1,j-1)});
                double m_val = diag + params_->matrix.score(seq_a_[i-1], seq_b_[j-1]);
                // X state: gap in B (advance i), uses gap_b params
                double x_val = std::max({
                    rat(buf.VM,i-1,j) - params_->gap_open_b - params_->gap_extend_b,
                    rat(buf.VX,i-1,j)                       - params_->gap_extend_b,
                    rat(buf.VY,i-1,j) - params_->gap_open_b - params_->gap_extend_b});
                // Y state: gap in A (advance j), uses gap_a params
                double y_val = std::max({
                    rat(buf.VM,i,j-1) - params_->gap_open_a - params_->gap_extend_a,
                    rat(buf.VX,i,j-1) - params_->gap_open_a - params_->gap_extend_a,
                    rat(buf.VY,i,j-1)                       - params_->gap_extend_a});

                if constexpr (AM == AlignMode::Local) {
                    m_val = std::max(m_val, 0.0);
                    double best_here = std::max({m_val, x_val, y_val});
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

    template<typename EmitFn>
    void traceback_affine_impl(const DpBuffer& buf, EmitFn&& emit) const {
        int i = best_i_, j = best_j_;
        TBTable tbl = best_tbl_;

        while (true) {
            if (i == 0 && j == 0) break;
            if constexpr (AM == AlignMode::Local)
                if (tbl == TBTable::M && rat(buf.VM, i, j) <= 0.0) break;

            if (tbl == TBTable::M) {
                emit(i-1, j-1);
                double vm = rat(buf.VM,i-1,j-1), vx = rat(buf.VX,i-1,j-1), vy = rat(buf.VY,i-1,j-1);
                --i; --j;
                if      (vm >= vx && vm >= vy) tbl = TBTable::M;
                else if (vx >= vy)             tbl = TBTable::X;
                else                            tbl = TBTable::Y;
            } else if (tbl == TBTable::X) {
                double fm = rat(buf.VM,i-1,j) - params_->gap_open_b - params_->gap_extend_b;
                double fx = rat(buf.VX,i-1,j) - params_->gap_extend_b;
                double fy = rat(buf.VY,i-1,j) - params_->gap_open_b - params_->gap_extend_b;
                --i;
                if      (fm >= fx && fm >= fy) tbl = TBTable::M;
                else if (fx >= fy)             tbl = TBTable::X;
                else                            tbl = TBTable::Y;
            } else {
                double fm = rat(buf.VM,i,j-1) - params_->gap_open_a - params_->gap_extend_a;
                double fx = rat(buf.VX,i,j-1) - params_->gap_open_a - params_->gap_extend_a;
                double fy = rat(buf.VY,i,j-1) - params_->gap_extend_a;
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

    void hard_grad_affine(const DpBuffer& buf, AlignParams& grad) const {
        int i = best_i_, j = best_j_;
        TBTable tbl = best_tbl_;

        while (true) {
            if (i == 0 && j == 0) break;
            if constexpr (AM == AlignMode::Local)
                if (tbl == TBTable::M && rat(buf.VM, i, j) <= 0.0) break;

            if (tbl == TBTable::M) {
                grad.matrix.mat[static_cast<unsigned char>(seq_a_[i-1])]
                               [static_cast<unsigned char>(seq_b_[j-1])] += 1.0;
                double vm = rat(buf.VM,i-1,j-1), vx = rat(buf.VX,i-1,j-1), vy = rat(buf.VY,i-1,j-1);
                --i; --j;
                if      (vm >= vx && vm >= vy) tbl = TBTable::M;
                else if (vx >= vy)             tbl = TBTable::X;
                else                            tbl = TBTable::Y;
            } else if (tbl == TBTable::X) {
                // X state: gap in B (advance i), uses gap_b params
                grad.gap_extend_b += 1.0;
                double fm = rat(buf.VM,i-1,j) - params_->gap_open_b - params_->gap_extend_b;
                double fx = rat(buf.VX,i-1,j)                       - params_->gap_extend_b;
                double fy = rat(buf.VY,i-1,j) - params_->gap_open_b - params_->gap_extend_b;
                --i;
                TBTable prev;
                if      (fm >= fx && fm >= fy) prev = TBTable::M;
                else if (fx >= fy)             prev = TBTable::X;
                else                            prev = TBTable::Y;
                if (prev != TBTable::X) grad.gap_open_b += 1.0;  // gap opening
                tbl = prev;
            } else {
                // Y state: gap in A (advance j), uses gap_a params
                grad.gap_extend_a += 1.0;
                double fm = rat(buf.VM,i,j-1) - params_->gap_open_a - params_->gap_extend_a;
                double fx = rat(buf.VX,i,j-1) - params_->gap_open_a - params_->gap_extend_a;
                double fy = rat(buf.VY,i,j-1)                       - params_->gap_extend_a;
                --j;
                TBTable prev;
                if      (fm >= fx && fm >= fy) prev = TBTable::M;
                else if (fx >= fy)             prev = TBTable::X;
                else                            prev = TBTable::Y;
                if (prev != TBTable::Y) grad.gap_open_a += 1.0;  // gap opening
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
                    rat(buf.F, i-1, j-1) + params_->matrix.score(seq_a_[i-1], seq_b_[j-1]),
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
            std::fill(buf.B.begin(), buf.B.begin() + static_cast<ptrdiff_t>(sz_), NEG_INF);
            at(buf.B, m_, n_) = 0.0;
        } else {
            std::fill(buf.B.begin(), buf.B.begin() + static_cast<ptrdiff_t>(sz_), 0.0);
        }

        for (int i = m_; i >= 0; --i) {
            for (int j = jhi0(i); j >= jlo0(i); --j) {
                double bval = rat(buf.B, i, j);
                if (bval == NEG_INF) continue;
                if (i > 0 && j > 0)
                    at(buf.B,i-1,j-1) = lse2(rat(buf.B,i-1,j-1),
                                              bval + params_->matrix.score(seq_a_[i-1], seq_b_[j-1]));
                if (i > 0)
                    at(buf.B,i-1,j)   = lse2(rat(buf.B,i-1,j),   bval - params_->gap_extend_b);
                if (j > 0)
                    at(buf.B,i,j-1)   = lse2(rat(buf.B,i,j-1),   bval - params_->gap_extend_a);
            }
        }
    }

    void soft_grad_linear(const DpBuffer& buf, AlignParams& grad) const {
        // Matrix gradient: match steps (i-1,j-1) → (i,j)
        for (int i = 1; i <= m_; ++i) {
            for (int j = jlo(i); j <= jhi(i); ++j) {
                double bval = rat(buf.B, i, j);
                if (bval == NEG_INF) continue;
                double log_p = rat(buf.F, i-1, j-1)
                               + params_->matrix.score(seq_a_[i-1], seq_b_[j-1])
                               + bval - log_z_;
                grad.matrix.mat[static_cast<unsigned char>(seq_a_[i-1])]
                               [static_cast<unsigned char>(seq_b_[j-1])] += std::exp(log_p);
            }
        }

        // gap_extend_b: B-gap steps (i-1,j) → (i,j), cost = -gap_extend_b
        for (int i = 1; i <= m_; ++i) {
            for (int j = jlo0(i); j <= jhi0(i); ++j) {
                double bval = rat(buf.B, i, j);
                if (bval == NEG_INF) continue;
                double fval = rat(buf.F, i-1, j);
                if (fval == NEG_INF) continue;
                grad.gap_extend_b += std::exp(fval - params_->gap_extend_b + bval - log_z_);
            }
        }

        // gap_extend_a: A-gap steps (i,j-1) → (i,j), cost = -gap_extend_a
        for (int i = 0; i <= m_; ++i) {
            for (int j = std::max(1, jlo0(i)); j <= jhi0(i); ++j) {
                double bval = rat(buf.B, i, j);
                if (bval == NEG_INF) continue;
                double fval = rat(buf.F, i, j-1);
                if (fval == NEG_INF) continue;
                grad.gap_extend_a += std::exp(fval - params_->gap_extend_a + bval - log_z_);
            }
        }
    }

    // ═════════════════════════════════════════════════════════════════════════
    // Forward-backward — Affine gap model
    // ═════════════════════════════════════════════════════════════════════════

    void fwdbwd_affine(DpBuffer& buf) {
        // ── Forward ──
        std::fill(buf.FM.begin(), buf.FM.begin()+static_cast<ptrdiff_t>(sz_), NEG_INF);
        std::fill(buf.FX.begin(), buf.FX.begin()+static_cast<ptrdiff_t>(sz_), NEG_INF);
        std::fill(buf.FY.begin(), buf.FY.begin()+static_cast<ptrdiff_t>(sz_), NEG_INF);

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
                double m_val = diag + params_->matrix.score(seq_a_[i-1], seq_b_[j-1]);
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
            std::fill(buf.BM.begin(), buf.BM.begin()+static_cast<ptrdiff_t>(sz_), NEG_INF);
            std::fill(buf.BX.begin(), buf.BX.begin()+static_cast<ptrdiff_t>(sz_), NEG_INF);
            std::fill(buf.BY.begin(), buf.BY.begin()+static_cast<ptrdiff_t>(sz_), NEG_INF);
            at(buf.BM,m_,n_) = 0.0; at(buf.BX,m_,n_) = 0.0; at(buf.BY,m_,n_) = 0.0;
        } else {
            std::fill(buf.BM.begin(), buf.BM.begin()+static_cast<ptrdiff_t>(sz_), 0.0);
            std::fill(buf.BX.begin(), buf.BX.begin()+static_cast<ptrdiff_t>(sz_), 0.0);
            std::fill(buf.BY.begin(), buf.BY.begin()+static_cast<ptrdiff_t>(sz_), 0.0);
        }

        for (int i = m_; i >= 0; --i) {
            for (int j = jhi0(i); j >= jlo0(i); --j) {
                double bm = rat(buf.BM,i,j), bx = rat(buf.BX,i,j), by = rat(buf.BY,i,j);

                if (bm != NEG_INF && i > 0 && j > 0) {
                    double contrib = bm + params_->matrix.score(seq_a_[i-1], seq_b_[j-1]);
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
        // Matrix gradient: match steps
        for (int i = 1; i <= m_; ++i) {
            for (int j = jlo(i); j <= jhi(i); ++j) {
                double bm = rat(buf.BM, i, j);
                if (bm == NEG_INF) continue;
                double pred  = lse3(rat(buf.FM,i-1,j-1), rat(buf.FX,i-1,j-1), rat(buf.FY,i-1,j-1));
                double log_p = pred + params_->matrix.score(seq_a_[i-1], seq_b_[j-1]) + bm - log_z_;
                grad.matrix.mat[static_cast<unsigned char>(seq_a_[i-1])]
                               [static_cast<unsigned char>(seq_b_[j-1])] += std::exp(log_p);
            }
        }

        // gap_extend_b: expected # of X-state steps (gap in B)
        for (int i = 1; i <= m_; ++i) {
            for (int j = jlo0(i); j <= jhi0(i); ++j) {
                double fx = rat(buf.FX, i, j), bx = rat(buf.BX, i, j);
                if (fx == NEG_INF || bx == NEG_INF) continue;
                grad.gap_extend_b += std::exp(fx + bx - log_z_);
            }
        }

        // gap_extend_a: expected # of Y-state steps (gap in A)
        for (int i = 0; i <= m_; ++i) {
            for (int j = std::max(1, jlo0(i)); j <= jhi0(i); ++j) {
                double fy = rat(buf.FY, i, j), by = rat(buf.BY, i, j);
                if (fy == NEG_INF || by == NEG_INF) continue;
                grad.gap_extend_a += std::exp(fy + by - log_z_);
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
                if (log_open > -700) grad.gap_open_b += std::exp(log_open);
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
                if (log_open > -700) grad.gap_open_a += std::exp(log_open);
            }
        }
    }
};
