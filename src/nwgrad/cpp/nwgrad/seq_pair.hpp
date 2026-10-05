#pragma once

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "align_params.hpp"
#include "aligner.hpp"

// ── Internal per-(GapModel, AlignMode) state ─────────────────────────────────
// Holds one Full and one GuideBanded aligner, reusing their DP buffers across
// repeated align_full() / realign_banded() calls on the same SeqPair.

template<GapModel GM, AlignMode AM, class T = double>
struct SeqPairState {
    Aligner<GM, AM, AlignBand::Full,        T> full_al;
    Aligner<GM, AM, AlignBand::GuideBanded, T> band_al;
};

enum class GradMode { None, Hard, Soft };

// ── SeqPair ───────────────────────────────────────────────────────────────────
//
// Persistent sequence-pair object that caches alignment and gradient state.
//
// Lifecycle:
//   1. Construct with fixed sequences, gap params, gap model, align mode, grad mode.
//   2. Set (or swap) the substitution matrix via set_matrix().
//   3. Run align_full() for the first alignment.
//   4. Optionally call realign_banded(bw) to re-align cheaply around the
//      current path under the current matrix.
//   5. Call compute_grad() to populate the gradient cache (not available for GradMode::None).
//
// GradMode::Hard  — Viterbi DP; score = max-path score; grad = subgradient (pair counts)
// GradMode::Soft  — Viterbi for guide_j + forward-backward; score = log Z; grad = expected counts
// GradMode::None  — Viterbi only; compute_grad() throws
//
// Invariants:
//   - score_valid => path_valid
//   - grad_valid  => score_valid
//   - set_params() clears score_valid, grad_valid and dp_valid (the retained tables
//     and traceback belong to the old params); path_valid stays (guide still usable)
//   - align_full() / realign_banded() set path_valid + score_valid, clear grad_valid
//   - compute_grad() requires score_valid and grad_mode != None
//   - SubstMatrix pointed to by matrix_ must outlive this object

template<class T = double>
struct SeqPairT {
    // Viterbi precision T (float32 default in the Python surface, double via the named
    // *Double variant).  Shadows the global ::DpBuffer so DpBuffer& signatures below mean
    // this precision's buffer with no per-site edit; the gradient/score stay double.
    using DpBuffer = DpBufferT<T>;

    using StateVar = std::variant<
        SeqPairState<GapModel::Linear, AlignMode::Global, T>,
        SeqPairState<GapModel::Linear, AlignMode::Local,  T>,
        SeqPairState<GapModel::Affine, AlignMode::Global, T>,
        SeqPairState<GapModel::Affine, AlignMode::Local,  T>
    >;

    // Sequences are validated and encoded to alphabet indices here, once, using
    // the params' alphabet.  An out-of-alphabet character throws.  Everything
    // downstream — the DP, the gradient — works on indices only.
    // `kernel` is the Viterbi backend (an int: kBackendAuto/kBackendScalar or a SimdLevel
    // index — see simd_levels.hpp).  Every simd level is bit-exact with the scalar one, so
    // it is purely a speed knob — and it is a runtime argument, not a template parameter,
    // so StateVar above stays at four arms instead of eight.  See the note in aligner.hpp.
    SeqPairT(std::string_view a, std::string_view b,
            const AlignParams& params,
            GapModel gm, AlignMode am,
            GradMode grad_mode = GradMode::Hard,
            int kernel = kBackendAuto,
            TracebackMode tb = TracebackMode::Default)
        : a_idx_(params.matrix.alphabet().encode(a)),
          b_idx_(params.matrix.alphabet().encode(b)),
          params_(&params),
          grad_mode_(grad_mode),
          kernel_(kernel),
          gm_(gm), am_(am),
          grad_(params.matrix.alphabet())
    {
        if      (gm == GapModel::Linear && am == AlignMode::Global)
            state_.template emplace<SeqPairState<GapModel::Linear, AlignMode::Global, T>>();
        else if (gm == GapModel::Linear && am == AlignMode::Local)
            state_.template emplace<SeqPairState<GapModel::Linear, AlignMode::Local, T>>();
        else if (gm == GapModel::Affine && am == AlignMode::Global)
            state_.template emplace<SeqPairState<GapModel::Affine, AlignMode::Global, T>>();
        else
            state_.template emplace<SeqPairState<GapModel::Affine, AlignMode::Local, T>>();

        std::visit([kernel, tb](auto& st) {
            st.full_al.set_kernel(kernel);
            st.band_al.set_kernel(kernel);
            // Only the Full aligner can use pointers; the banded one needs its score
            // tables and its footprint is O(m*band) already.
            st.full_al.set_traceback(tb);
            st.band_al.set_traceback(TracebackMode::Scores);
        }, state_);
    }

    int kernel() const noexcept { return kernel_; }

    // The RESOLVED mode the full aligner will actually run — set_traceback mapped the
    // Default sentinel to this pair's compile-time default already, so a caller who
    // passed "auto" sees "hirschberg" here on an affine+global+full pair, "pointers"
    // otherwise.  More useful than echoing back "auto".
    TracebackMode traceback() const noexcept {
        return std::visit([](const auto& st) { return st.full_al.traceback(); }, state_);
    }

    // Hirschberg base-case size, in rows.  Only the full aligner ever runs Hirschberg,
    // so only it is told.  Settable (unlike traceback) because it changes how the DP
    // divides, not what it retains — no allocation decision depends on it.
    void set_hb_cutoff(int rows) {
        std::visit([rows](auto& st) { st.full_al.set_hb_cutoff(rows); }, state_);
        hb_cutoff_ = rows;
    }
    int hb_cutoff() const noexcept { return hb_cutoff_; }

    // Full-band simd fill at double precision: striped (default) or row-wise — see
    // Aligner::set_rowwise_full.  Bit-identical results, so settable at any time.
    void set_rowwise_full(bool on) {
        std::visit([on](auto& st) { st.full_al.set_rowwise_full(on); }, state_);
        rowwise_full_ = on;
    }
    bool rowwise_full() const noexcept { return rowwise_full_; }

    // How the soft path evaluates forward-backward — see SoftImpl.  Not bit-exact
    // across choices (tolerance-tested), so it invalidates nothing already cached.
    void set_soft_impl(SoftImpl s) {
        std::visit([s](auto& st) { st.full_al.set_soft_impl(s); st.band_al.set_soft_impl(s); }, state_);
        soft_impl_ = s;
    }
    SoftImpl soft_impl() const noexcept { return soft_impl_; }

    // Swap alignment parameters.  Invalidates score, gradient and the retained DP
    // tables (aligned() throws until the next align); the guide path stays.
    // realign_banded() remains callable after this — it will re-score the
    // existing guide path under the new params.
    // The new params must outlive this SeqPair.
    //
    // The new params must be over the same alphabet: the sequences were encoded
    // to indices under the old one, and those indices would silently mean
    // different residues under a different alphabet.
    void set_params(const AlignParams& params) {
        if (&params.matrix.alphabet() != &params_->matrix.alphabet())
            throw std::invalid_argument(
                "nwgrad: set_params() cannot change the alphabet (\"" +
                params_->matrix.alphabet().symbols() + "\" -> \"" +
                params.matrix.alphabet().symbols() +
                "\"); construct a new SeqPair instead");
        params_ = &params;
        score_valid_ = false;
        grad_valid_  = false;
        // The aligners' retained tables, and the params pointer they took at
        // set_problem(), belong to the OLD params, which the caller may now free
        // (the Python binding drops its reference).  A traceback re-derives each
        // step from those params, so aligned() is invalid until the next DP.
        // guide_j_ is plain data and survives, which is what realign_banded() needs.
        dp_valid_    = false;
    }

    // Allocate own DP buffer shells. Must be called before align_full() / realign_banded() / compute_grad().
    // score_and_grad_with_dp() uses caller-supplied buffers and never needs this.
    // Buffers grow implicitly as needed.
    void alloc_dp() {
        std::visit([](auto& st) {
            st.full_al.alloc_buf();
            st.band_al.alloc_buf();
        }, state_);
    }

    // Full DP alignment.  Sets path_valid + score_valid; clears grad_valid.
    // Hard/None: computes Viterbi only.  Soft: computes Viterbi (for guide_j)
    // then forward-backward; score returns log Z.
    void align_full() {
        std::visit([&](auto& st) {
            st.full_al.set_problem(a_idx_, b_idx_, *params_);
            run_dp(st.full_al);
        }, state_);
        last_banded_  = false;
        path_valid_   = true;
        score_valid_  = true;
        grad_valid_   = false;
        dp_valid_     = true;
    }

    // Banded DP around the current guide path under the current matrix.
    // Requires path_valid (align_full() must have been called at least once).
    // Sets score_valid; clears grad_valid.
    // Note: if the optimal path under the current matrix lies outside the band,
    // this silently returns a suboptimal score (no detection).
    void realign_banded(int bandwidth) {
        if (!path_valid_)
            throw std::logic_error(
                "nwgrad: call align_full() before realign_banded()");
        std::visit([&](auto& st) {
            st.band_al.set_problem(a_idx_, b_idx_, *params_, bandwidth, guide_j_);
            run_dp(st.band_al);
        }, state_);
        last_banded_  = true;
        score_valid_  = true;
        grad_valid_   = false;
        dp_valid_     = true;
    }

    // Full align + grad using a caller-supplied DpBuffer (e.g. thread-owned).
    // Results (score, guide_j, grad) are saved to this SeqPair's cached fields.
    // The pair's own DP buffers are never touched; dp_valid_ stays false.
    // Grad is computed only when grad_mode_ != None.
    //
    // There is deliberately NO bandwidth parameter, and no fused "full DP then
    // banded DP" entry point.  One existed and was removed: it ran the full DP,
    // took the optimal path as a guide, and then re-ran a banded DP *around that
    // same path*.  The optimum is inside that band by construction, so the second
    // DP maximises over a subset that provably contains the maximiser and cannot
    // return anything different.  Measured across 300 pairs it was bit-identical
    // at every bandwidth tested including 1 -- same scores, same gradients -- and
    // cost +7% at band=32 and +26% at band=128 for that identical answer.  The
    // only state it changed was a last_banded_ flag.
    //
    // Banding is worth something only when the guide comes from a DIFFERENT
    // matrix than the one being scored, which is the align-once / re-align-often
    // training loop.  That is banded_grad_with_dp() below, and it is a separate
    // path because it is a separate workload with its own cost shape.
    void score_and_grad_with_dp(DpBuffer& buf) {
        std::visit([&](auto& st) {
            st.full_al.set_problem(a_idx_, b_idx_, *params_);
            last_banded_ = false;
            if (grad_mode_ == GradMode::Hard) {
                // As run_dp_with_buf + grad_with_buf, with the guide taken from the
                // gradient's own traceback walk where the tables allow it.
                st.full_al.compute_viterbi(buf);
                score_ = st.full_al.score();
                grad_.zero();
                st.full_al.hard_grad_and_guide(buf, grad_, guide_j_);
                return;
            }
            run_dp_with_buf(st.full_al, buf);
            if (grad_mode_ != GradMode::None) {
                grad_.zero();
                grad_with_buf(st.full_al, buf);
            }
        }, state_);
        path_valid_  = true;
        score_valid_ = true;
        grad_valid_  = (grad_mode_ != GradMode::None);
        dp_valid_    = false;
    }

    // Banded align + grad around the CACHED guide path, using a caller-supplied
    // DpBuffer.  The counterpart to score_and_grad_with_dp() for the re-alignment
    // half of a training loop: align once with the full DP, then call this after
    // each matrix update.  No full DP is run, which is the entire point.
    //
    // Requires a guide: path_valid_ must already be true.  set_params() preserves
    // path_valid_ precisely so this survives a matrix update.
    //
    // Like SeqPair::realign_banded(), this returns a SUBOPTIMAL score without
    // complaint if the new matrix's optimal path has wandered outside the band.
    // That is the trade being made, not a defect -- but it is real, and it is why
    // this is not a drop-in substitute for the full DP.
    void banded_grad_with_dp(DpBuffer& buf, int bandwidth) {
        if (!path_valid_)
            throw std::logic_error(
                "nwgrad: banded_grad_with_dp() needs a guide path; run the full "
                "score_and_grad_with_dp() (or align_full()) at least once first");
        std::visit([&](auto& st) {
            st.band_al.set_problem(a_idx_, b_idx_, *params_, bandwidth, guide_j_);
            run_dp_with_buf(st.band_al, buf);
            last_banded_ = true;
            if (grad_mode_ != GradMode::None) {
                grad_.zero();
                grad_with_buf(st.band_al, buf);
            }
        }, state_);
        path_valid_  = true;
        score_valid_ = true;
        grad_valid_  = (grad_mode_ != GradMode::None);
        dp_valid_    = false;
    }

    // Release the DP table memory of both full and banded aligners.
    // Cached score, gradient, and guide_j remain valid.
    // compute_grad() will throw until the next align_full() / realign_banded() call.
    void drop_dp() noexcept {
        std::visit([](auto& st) {
            st.full_al.free_dp();
            st.band_al.free_dp();
        }, state_);
        dp_valid_ = false;
    }

    // Compute and cache the gradient from the current alignment.
    // Requires score_valid and dp_valid.  Throws if grad_mode == None.
    void compute_grad() {
        if (!score_valid_)
            throw std::logic_error(
                "nwgrad: call align_full() or realign_banded() first");
        if (!dp_valid_)
            throw std::logic_error(
                "nwgrad: DP tables have been dropped; call align_full() or realign_banded() first");
        if (grad_mode_ == GradMode::None)
            throw std::logic_error(
                "nwgrad: grad_mode is None; construct with Hard or Soft to enable gradients");
        grad_.zero();
        std::visit([&](auto& st) {
            if (grad_mode_ == GradMode::Hard) {
                if (last_banded_) st.band_al.hard_grad(grad_);
                else              st.full_al.hard_grad(grad_);
            } else {
                if (last_banded_) st.band_al.soft_grad(grad_);
                else              st.full_al.soft_grad(grad_);
            }
        }, state_);
        grad_valid_ = true;
    }

    // Convenience: align then compute the gradient in one call, returning both.
    // Allocates the DP buffers if needed.  Throws if grad_mode == None.
    std::pair<double, AlignParams> score_and_grad() {
        alloc_dp();
        align_full();
        compute_grad();
        return {score_, grad_};
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    double score() const {
        if (!score_valid_)
            throw std::logic_error(
                "nwgrad: score not valid; call align_full() or realign_banded() first");
        return score_;
    }

    const AlignParams& grad() const {
        if (!grad_valid_)
            throw std::logic_error(
                "nwgrad: gradient not computed; call compute_grad() first");
        return grad_;
    }

    // The current alignment as a pair of gapped strings (a, b).
    // Requires the DP tables to be present (i.e. align_full() / realign_banded()
    // was called and drop_dp() has not been called since).  For GradMode::Soft
    // this returns the Viterbi (maximum-score) alignment.
    std::pair<std::string, std::string> aligned() const {
        if (!path_valid_)
            throw std::logic_error(
                "nwgrad: alignment not computed; call align_full() first");
        if (!dp_valid_)
            throw std::logic_error(
                "nwgrad: DP tables have been dropped; call align_full() or realign_banded() first");
        std::pair<std::string, std::string> out;
        std::visit([&](auto& st) {
            if (last_banded_) out = st.band_al.aligned();
            else              out = st.full_al.aligned();
        }, state_);
        return out;
    }

    // The alignment as Biopython-style coordinates: two rows (positions in A and in B)
    // with a column at the start, at every change between aligned, gap-in-B and
    // gap-in-A columns, and at the end — what Bio.Align.Alignment(sequences,
    // coordinates) takes.  Local alignments start where the path starts, not at 0.
    // Same availability as aligned().
    std::pair<std::vector<int64_t>, std::vector<int64_t>> coordinates() const {
        const auto [sa, sb] = aligned();
        std::pair<int, int> end;
        std::visit([&](auto& st) {
            end = last_banded_ ? st.band_al.alignment_end() : st.full_al.alignment_end();
        }, state_);
        int64_t i = end.first, j = end.second;
        for (char c : sa) if (c != '-') --i;
        for (char c : sb) if (c != '-') --j;
        std::vector<int64_t> ci{i}, cj{j};
        int prev = -1;
        for (size_t k = 0; k < sa.size(); ++k) {
            const int t = (sa[k] == '-') ? 2 : (sb[k] == '-') ? 1 : 0;
            if (prev != -1 && t != prev) { ci.push_back(i); cj.push_back(j); }
            if (sa[k] != '-') ++i;
            if (sb[k] != '-') ++j;
            prev = t;
        }
        ci.push_back(i); cj.push_back(j);
        return {std::move(ci), std::move(cj)};
    }

    // The current alignment as a guide_j vector (length m+1).
    // Valid when path_valid().
    const std::vector<int>& guide_j() const {
        if (!path_valid_)
            throw std::logic_error(
                "nwgrad: alignment not computed; call align_full() first");
        return guide_j_;
    }

    bool path_valid()  const noexcept { return path_valid_;  }
    bool score_valid() const noexcept { return score_valid_; }
    // score_and_grad_with_dp() for a pair whose affine Full DP was filled as lane `lane`
    // of an inter-pair fill (InterJob) into buf's interleaved tables: same results,
    // bit for bit — only the fill was shared.  Hard or no gradient only.
    void score_and_grad_interleaved(DpBuffer& buf, int W, int lane,
                                    double local_best, int best_i, int best_j) {
        std::visit([&](auto& st) {
            st.full_al.set_problem(a_idx_, b_idx_, *params_);
            st.full_al.adopt_interleaved(buf, W, lane, local_best, best_i, best_j);
            score_ = st.full_al.score();
            last_banded_ = false;
            if (grad_mode_ == GradMode::Hard) {
                grad_.zero();
                st.full_al.hard_grad_and_guide(buf, grad_, guide_j_);   // one walk for both
            } else {
                guide_j_ = st.full_al.guide_j_from_viterbi(buf);
            }
        }, state_);
        path_valid_  = true;
        score_valid_ = true;
        grad_valid_  = (grad_mode_ != GradMode::None);
        dp_valid_    = false;
    }

    // For the inter-pair scheduler: the problem as the fill needs it.
    const std::vector<uint8_t>& a_codes() const noexcept { return a_idx_; }
    const std::vector<uint8_t>& b_codes() const noexcept { return b_idx_; }
    const AlignParams* params_ptr() const noexcept { return params_; }
    GapModel gap_model() const noexcept { return gm_; }
    AlignMode align_mode() const noexcept { return am_; }

    bool grad_valid()  const noexcept { return grad_valid_;  }
    bool dp_valid()    const noexcept { return dp_valid_;    }

    GradMode grad_mode() const noexcept { return grad_mode_; }

    // Encoded lengths.  Exposed so a scheduler can cost a pair without decoding
    // it: the full-square DP is (len_a+1)*(len_b+1) cells.
    size_t len_a() const noexcept { return a_idx_.size(); }
    size_t len_b() const noexcept { return b_idx_.size(); }

    // Decoded on demand: only the encoded indices are stored.
    std::string seq_a() const { return params_->matrix.alphabet().decode(a_idx_); }
    std::string seq_b() const { return params_->matrix.alphabet().decode(b_idx_); }

    const Alphabet& alphabet() const noexcept { return params_->matrix.alphabet(); }

private:
    std::vector<uint8_t> a_idx_, b_idx_;   // alphabet indices, not characters
    const AlignParams*   params_;
    GradMode             grad_mode_;
    int                  kernel_;   // Viterbi backend, forwarded to each aligner
    GapModel             gm_;
    AlignMode            am_;
    int                  hb_cutoff_ = 512;   // mirrors Aligner's default (fleet-swept)
    bool                 rowwise_full_ = false;
    SoftImpl             soft_impl_ = SoftImpl::Scaled;

    bool             path_valid_   = false;
    bool             score_valid_  = false;
    bool             last_banded_  = false;
    bool             dp_valid_     = false;
    double           score_        = 0.0;
    std::vector<int> guide_j_;

    bool        grad_valid_ = false;
    AlignParams grad_;

    StateVar state_;

    // Run DP on `al` using its own buffer; update score_ / guide_j_.
    template<typename Al>
    void run_dp(Al& al) {
        al.compute_viterbi();
        guide_j_ = al.guide_j_from_viterbi();
        if (grad_mode_ == GradMode::Soft) {
            al.compute_forward_back();
            score_ = al.log_z();
        } else {
            score_ = al.score();
        }
    }

    // Run DP on `al` using external `buf`; update score_ / guide_j_.
    template<typename Al>
    void run_dp_with_buf(Al& al, DpBuffer& buf) {
        al.compute_viterbi(buf);
        guide_j_ = al.guide_j_from_viterbi(buf);
        if (grad_mode_ == GradMode::Soft) {
            al.compute_forward_back(buf);
            score_ = al.log_z();
        } else {
            score_ = al.score();
        }
    }

    // Accumulate grad from `al` using external `buf` into grad_.
    template<typename Al>
    void grad_with_buf(Al& al, DpBuffer& buf) {
        if (grad_mode_ == GradMode::Hard) al.hard_grad(buf, grad_);
        else                               al.soft_grad(buf, grad_);
    }
};

// The default (double) alias keeps every existing C++ user and test unchanged; the
// Python surface binds SeqPairT<float> as `SeqPair` and SeqPairT<double> as `SeqPairDouble`.
using SeqPair = SeqPairT<double>;
