#pragma once

// ── BatchEngine<T, GapModel, AlignMode> ─────────────────────────────────────
//
// A batch of sequence pairs of ONE problem type, with every pair's state held in flat
// arrays the batch owns.  The engine underneath SeqPairBatch, SeqPair and the streaming
// align(); see SeqPairBatchT (seq_pair_batch.hpp) for the runtime-typed façade.
//
// Why flat arrays.  A batch used to hold one SeqPair object per pair, and each object
// held two complete Aligners (Full and GuideBanded, 1776 B each): 3768 B inline and
// ~4.1 KB resident per pair, ~10 GB for 2.5 M miRNA x site pairs — of which a pair needs
// ~350 B (its codes, score, guide, gradient and flags).  Batch-wide walks (set_params,
// scores, weighted_grad) were one cache miss per pair, chasing pointers to scattered
// heap objects.  Here a pair is an index: its codes live in one array, its guide in
// another, its gradient in a third, and the Aligners exist once per WORKER thread,
// reused across pairs (set_problem() resets every per-problem field; the old
// BatchAligner already ran one Aligner per thread this way).
//
// Why one problem type.  The type is a template, so no per-pair std::variant and no
// per-pair branch on gap model or mode — and a worker's DpBuffer only ever serves one
// Aligner type.  (Mixing linear and affine pairs on one buffer is what exposed the
// DX/DY overrun fixed in e2cac6b; that fix stays, since `auto` still picks different
// fills per pair within a type.)
//
// Every per-pair operation below is a port of the SeqPair method of the same name, in
// the same order of Aligner calls: the per-pair results are bit-identical to the old
// classes, which tests/cpp/test_batch_engine.cpp checks against them directly.
//
// Params are per add_many() call: a SEGMENT of consecutive pairs shares one
// AlignParams (and one Viterbi backend).  set_params() points every segment at the new
// params.  Params are held by pointer; the caller keeps them alive (the Python binding
// pins them).

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "align_params.hpp"
#include "aligner.hpp"
#include "batch_result.hpp"
#include "grad_mode.hpp"
#include "parallel.hpp"
#include "simd_levels.hpp"

template<class T, GapModel GM, AlignMode AM>
class BatchEngine {
public:
    using DpBuffer = DpBufferT<T>;
    using FullAl   = Aligner<GM, AM, AlignBand::Full, T>;
    using BandAl   = Aligner<GM, AM, AlignBand::GuideBanded, T>;
    static constexpr GapModel  gap_model  = GM;
    static constexpr AlignMode align_mode = AM;

    // ── Construction ─────────────────────────────────────────────────────────

    BatchEngine(int n_threads, GradMode gd, TracebackMode tb = TracebackMode::Default)
        : n_threads_(n_threads > 0 ? n_threads : default_thread_count()),
          grad_mode_(gd), tb_(tb) {
        FullAl probe;   // what "auto" resolves to for this problem type
        probe.set_traceback(tb_);
        tb_resolved_ = probe.traceback();
    }

    BatchEngine(const BatchEngine&)            = delete;
    BatchEngine& operator=(const BatchEngine&) = delete;
    BatchEngine(BatchEngine&&)                 = default;
    BatchEngine& operator=(BatchEngine&&)      = default;

    // ── Settings ─────────────────────────────────────────────────────────────
    // Batch-wide.  The ones that change a result invalidate the cached results they
    // change; the rest are speed knobs (bit-identical results).

    int n_threads() const noexcept { return n_threads_; }
    void set_n_threads(int n) { n_threads_ = n > 0 ? n : default_thread_count(); }
    GradMode grad_mode() const noexcept { return grad_mode_; }
    // The traceback as constructed ("auto" stays Default) and as this type resolves it.
    TracebackMode traceback() const noexcept { return tb_; }
    TracebackMode traceback_resolved() const noexcept { return tb_resolved_; }

    // Hirschberg base-case rows (see Aligner::set_hb_cutoff).  Speed/memory, and — above
    // the cutoff — which of several optimal paths; it applies to every pair from the
    // next alignment on.
    int hb_cutoff() const noexcept { return hb_cutoff_; }
    void set_hb_cutoff(int rows) {
        if (rows < 1) throw std::invalid_argument("nwgrad: hb_cutoff must be >= 1");
        hb_cutoff_ = rows;
    }

    // Full-band fill: striped / row-wise (Aligner::set_rowwise_full), and whether
    // score_and_grad() / banded_grad() fill W pairs per vector (inter-pair).  Speed only.
    bool rowwise_full = false;
    bool inter_fill   = true;

    // Soft path.  set_soft_impl changes results within tolerance only and invalidates
    // nothing (as SeqPair::set_soft_impl); a new temperature changes the soft score, so
    // it invalidates soft results.
    SoftImpl soft_impl() const noexcept { return soft_impl_; }
    void set_soft_impl(SoftImpl s) noexcept { soft_impl_ = s; }
    double soft_temperature() const noexcept { return soft_temp_; }
    void set_soft_temperature(double t) {
        if (!(t > 0.0 && t <= std::numeric_limits<double>::max()))
            throw std::invalid_argument("nwgrad: soft temperature must be finite and > 0, got " +
                                        std::to_string(t));
        soft_temp_ = t;
        if (grad_mode_ == GradMode::Soft)
            for (auto& f : flags_) f &= static_cast<uint8_t>(~(kScore | kGrad | kHeld));
    }
    // Soft guide policy (see SeqPair::set_soft_guide_lazy / _posterior): at most one set.
    bool soft_guide_lazy() const noexcept { return soft_lazy_; }
    bool soft_guide_posterior() const noexcept { return soft_post_; }
    void set_soft_guide(bool lazy, bool posterior) noexcept {
        soft_lazy_ = lazy && !posterior;
        soft_post_ = posterior;
    }

    // Scheduling (score_and_grad / banded_grad): see run_sorted_phase_ and
    // banded_grad_lpt_ for the measurements behind each.
    bool   sorted_schedule = false;
    double reserve_frac    = 0.0;
    double long_cost_ratio = 1.0;
    double weight_lo       = 500.0;
    double weight_hi       = 1700.0;
    struct PhaseProfile {
        double chunk_s = 0.0, reserve_s = 0.0;
        double chunk_cells = 0.0, reserve_cells = 0.0;
        long   chunk_tasks = 0, reserve_tasks = 0;
        double finish_s = 0.0;
    };
    bool profile = false;
    std::vector<PhaseProfile> profile_out;

    // ── Adding pairs ─────────────────────────────────────────────────────────

    // Append N pairs as one segment under `params` (which must outlive the batch's use
    // of it) and Viterbi backend `kernel`.  Sequences are validated and encoded here, in
    // parallel, straight into the batch's arrays; an out-of-alphabet character throws
    // (naming it and its position) and leaves the batch exactly as it was.
    void add_many(const std::vector<std::string_view>& seqs_a,
                  const std::vector<std::string_view>& seqs_b,
                  const AlignParams& params, int kernel = kBackendAuto) {
        if (seqs_a.size() != seqs_b.size())
            throw std::invalid_argument(
                "nwgrad: add_many() needs seqs_a and seqs_b of equal length (got " +
                std::to_string(seqs_a.size()) + " and " + std::to_string(seqs_b.size()) + ")");
        const Alphabet& alpha = params.matrix.alphabet();
        if (alpha_ && &alpha != alpha_)
            throw std::invalid_argument(
                "nwgrad: SeqPair over alphabet \"" + alpha.symbols() +
                "\" cannot join a batch over alphabet \"" + alpha_->symbols() + "\"");
        const size_t N = seqs_a.size();
        if (N == 0) return;
        if (N > std::numeric_limits<uint32_t>::max() - size())
            throw std::length_error("nwgrad: batch too large");

        const size_t N0 = size();
        const size_t ca0 = ca_.size(), cb0 = cb_.size();
        // Offsets first (serial: a prefix sum), then encode in parallel into place.
        for (size_t i = 0; i < N; ++i) {
            oa_.push_back(oa_.back() + seqs_a[i].size());
            ob_.push_back(ob_.back() + seqs_b[i].size());
        }
        try {
            ca_.resize(oa_.back());
            cb_.resize(ob_.back());
            parallel_for_(N, [&](size_t k) {
                alpha.encode_into(seqs_a[k], ca_.data() + oa_[N0 + k]);
                alpha.encode_into(seqs_b[k], cb_.data() + ob_[N0 + k]);
            });
        } catch (...) {
            oa_.resize(N0 + 1); ob_.resize(N0 + 1);
            ca_.resize(ca0);    cb_.resize(cb0);
            throw;
        }
        alpha_ = &alpha;
        nn_ = static_cast<size_t>(alpha.size()) * static_cast<size_t>(alpha.size());
        segs_.push_back({&params, kernel, N0});
        seg_.resize(N0 + N, static_cast<uint32_t>(segs_.size() - 1));
        score_.resize(N0 + N, 0.0);
        flags_.resize(N0 + N, 0);
        guide_.resize(oa_.back() + N0 + N, 0);   // pair i: m+1 entries at oa_[i] + i
        if (grad_mode_ != GradMode::None) grad_.resize((N0 + N) * gstride(), 0.0);
        ++generation_;
    }

    size_t size() const noexcept { return score_.size(); }
    bool empty() const noexcept { return score_.empty(); }
    const Alphabet& alphabet() const {
        if (!alpha_) throw std::logic_error("nwgrad: empty batch has no alphabet");
        return *alpha_;
    }

    // ── Per-pair accessors ───────────────────────────────────────────────────

    size_t len_a(size_t i) const noexcept { return oa_[i + 1] - oa_[i]; }
    size_t len_b(size_t i) const noexcept { return ob_[i + 1] - ob_[i]; }
    std::span<const uint8_t> codes_a(size_t i) const noexcept { return {ca_.data() + oa_[i], len_a(i)}; }
    std::span<const uint8_t> codes_b(size_t i) const noexcept { return {cb_.data() + ob_[i], len_b(i)}; }
    std::string seq_a(size_t i) const { return decode_(codes_a(i)); }
    std::string seq_b(size_t i) const { return decode_(codes_b(i)); }
    const AlignParams& params(size_t i) const noexcept { return *segs_[seg_[i]].params; }
    int kernel(size_t i) const noexcept { return segs_[seg_[i]].kernel; }

    bool path_valid(size_t i)  const noexcept { return flags_[i] & kPath; }
    bool score_valid(size_t i) const noexcept { return flags_[i] & kScore; }
    bool grad_valid(size_t i)  const noexcept { return flags_[i] & kGrad; }
    // A stored alignment path (aligned(), coordinates()) — kept only on request.
    bool path_stored(size_t i) const noexcept { return flags_[i] & kStored; }
    bool guide_pending(size_t i) const noexcept { return flags_[i] & kPending; }

    double score(size_t i) const {
        if (!score_valid(i))
            throw std::logic_error(
                "nwgrad: score not valid; call align_full() or realign_banded() first");
        return score_[i];
    }
    // Pair i's gradient into `out` (same alphabet).
    void grad(size_t i, AlignParams& out) const {
        if (!grad_valid(i))
            throw std::logic_error("nwgrad: gradient not computed; call compute_grad() first");
        const double* g = grad_.data() + i * gstride();
        std::memcpy(out.matrix.data(), g, nn_ * sizeof(double));
        out.gap_open_a = g[nn_]; out.gap_extend_a = g[nn_ + 1];
        out.gap_open_b = g[nn_ + 2]; out.gap_extend_b = g[nn_ + 3];
    }
    AlignParams grad(size_t i) const { AlignParams g(alphabet()); grad(i, g); return g; }

    // The cached guide (column after each row of A), resolving a pending lazy guide.
    std::vector<int> guide_j(size_t i) {
        if (!path_valid(i))
            throw std::logic_error("nwgrad: alignment not computed; call align_full() first");
        if (guide_pending(i)) resolve_guide_(i, solo_work_());
        const int* g = guide_ptr_(i);
        return {g, g + len_a(i) + 1};
    }

    // The cached guide as stored: no lazy resolution (for grouping, and for tests).
    std::vector<int> guide_raw(size_t i) const {
        const int* g = guide_ptr_(i);
        return {g, g + len_a(i) + 1};
    }

    // The stored alignment as gapped strings, and as Biopython-style coordinates (see
    // SeqPair::coordinates for the format).
    std::pair<std::string, std::string> aligned(size_t i) const {
        check_stored_(i);
        const auto& ops = path_[i];
        auto [i0, j0] = path_start_(i);
        std::string a, b;
        const auto ca = codes_a(i), cb = codes_b(i);
        size_t ia = static_cast<size_t>(i0), jb = static_cast<size_t>(j0);
        for (uint32_t op : ops) {
            const uint32_t t = op >> 30, len = op & 0x3fffffffu;
            for (uint32_t k = 0; k < len; ++k) {
                a.push_back(t == 2 ? '-' : alpha_->symbol_at(ca[ia++]));
                b.push_back(t == 1 ? '-' : alpha_->symbol_at(cb[jb++]));
            }
        }
        return {std::move(a), std::move(b)};
    }
    std::pair<std::vector<int64_t>, std::vector<int64_t>> coordinates(size_t i) const {
        check_stored_(i);
        auto [i0, j0] = path_start_(i);
        int64_t x = i0, y = j0;
        std::vector<int64_t> ci{x}, cj{y};
        const auto& ops = path_[i];
        for (size_t k = 0; k < ops.size(); ++k) {
            const uint32_t t = ops[k] >> 30, len = ops[k] & 0x3fffffffu;
            if (t != 2) x += len;
            if (t != 1) y += len;
            if (k + 1 < ops.size()) { ci.push_back(x); cj.push_back(y); }
        }
        ci.push_back(x); cj.push_back(y);
        return {std::move(ci), std::move(cj)};
    }

    // ── Batch operations ─────────────────────────────────────────────────────

    // Point every segment at `params` (same alphabet).  Clears every cached score,
    // gradient and stored path; guides stay (banded_grad() bands around them).
    void set_params(const AlignParams& params) {
        if (empty()) return;
        if (&params.matrix.alphabet() != alpha_)
            throw std::invalid_argument(
                "nwgrad: set_params() cannot change the alphabet (\"" + alpha_->symbols() +
                "\" -> \"" + params.matrix.alphabet().symbols() + "\"); construct new pairs instead");
        for (auto& s : segs_) s.params = &params;
        for (auto& f : flags_) f &= static_cast<uint8_t>(~(kScore | kGrad | kStored | kHeld));
    }
    // set_params for one pair's segment only (a single-pair batch: SeqPair).
    void set_params_segment(size_t seg, const AlignParams& params) {
        if (&params.matrix.alphabet() != alpha_)
            throw std::invalid_argument(
                "nwgrad: set_params() cannot change the alphabet (\"" + alpha_->symbols() +
                "\" -> \"" + params.matrix.alphabet().symbols() + "\"); construct a new SeqPair instead");
        segs_[seg].params = &params;
        const size_t end = seg + 1 < segs_.size() ? segs_[seg + 1].begin : size();
        for (size_t i = segs_[seg].begin; i < end; ++i)
            flags_[i] &= static_cast<uint8_t>(~(kScore | kGrad | kStored | kHeld));
    }

    // Full DP on every pair: score, guide and (unless grad_mode None) gradient, cached
    // per pair.  keep_paths: also store each pair's alignment path (aligned(),
    // coordinates()); it forces the Viterbi a lazy or posterior soft guide would skip.
    // Returns the sum of scores, in pair order (bit-reproducible whatever the schedule).
    //
    // hold_grads: leave each gradient computed but NOT valid until compute_grad() — the
    // state the old align_full() / realign_banded() left (they computed no gradient;
    // compute_grad() derived it from the retained tables).  The deprecated wrappers use it.
    double score_and_grad(bool keep_paths = false, bool hold_grads = false) {
        const size_t N = size();
        if (N == 0) return 0.0;
        prepare_paths_(keep_paths);
        ++full_calls_;   // guides may have moved: the banded grouping is stale
        if (inter_fill)           score_and_grad_inter_(keep_paths);
        else if (sorted_schedule) score_and_grad_sorted_(keep_paths);
        else                      score_and_grad_dynamic_(keep_paths);
        if (hold_grads) hold_all_();
        return sum_scores_();
    }

    // Banded re-align + gradient around every pair's cached guide (the training-loop
    // step after set_params()).  Throws if any pair has no guide yet.
    double banded_grad(int bandwidth, bool keep_paths = false, bool hold_grads = false) {
        const size_t N = size();
        if (N == 0) return 0.0;
        if (bandwidth <= 0)
            throw std::invalid_argument(
                "nwgrad: banded_grad() needs bandwidth > 0 (got " + std::to_string(bandwidth) +
                "); use score_and_grad() for full DP");
        for (size_t i = 0; i < N; ++i)
            if (!path_valid(i))
                throw std::logic_error(
                    "nwgrad: banded_grad_with_dp() needs a guide path; run the full "
                    "score_and_grad_with_dp() (or align_full()) at least once first");
        prepare_paths_(keep_paths);
        if (inter_fill && !keep_paths) banded_grad_inter_(bandwidth);
        else if (sorted_schedule)      banded_grad_lpt_(bandwidth, keep_paths);
        else                           banded_grad_dynamic_(bandwidth, keep_paths);
        if (hold_grads) hold_all_();
        return sum_scores_();
    }

    // Free the stored paths.  Scores, released gradients and guides stay; a held
    // gradient goes with the path (as the old drop_dp() made compute_grad() throw).
    void drop_paths() noexcept {
        std::vector<std::vector<uint32_t>>().swap(path_);
        std::vector<std::array<int, 2>>().swap(path_end_);
        for (auto& f : flags_) f &= static_cast<uint8_t>(~(kStored | kHeld));
    }
    void drop_path(size_t i) noexcept {
        if (i < path_.size()) { std::vector<uint32_t>().swap(path_[i]); }
        flags_[i] &= static_cast<uint8_t>(~(kStored | kHeld));
    }

    // One pair, on the calling thread (SeqPair's own operations).
    void score_and_grad_one(size_t i, bool keep_path, bool hold_grad = false) {
        prepare_paths_(keep_path, /*all=*/false);
        run_full_(i, solo_work_(), keep_path);
        if (hold_grad) hold_(i);
    }
    void banded_one(size_t i, int bandwidth, bool keep_path, bool hold_grad = false) {
        if (!path_valid(i))
            throw std::logic_error("nwgrad: call align_full() before realign_banded()");
        prepare_paths_(keep_path, /*all=*/false);
        run_banded_(i, solo_work_(), bandwidth, keep_path);
        if (hold_grad) hold_(i);
    }
    // SeqPair::compute_grad: release pair i's held gradient, with the old preconditions
    // and messages, in the old order.
    void compute_grad_one(size_t i) {
        if (grad_valid(i)) return;
        if (!score_valid(i))
            throw std::logic_error("nwgrad: call align_full() or realign_banded() first");
        if (!(flags_[i] & kHeld) && !(grad_mode_ == GradMode::None && path_stored(i)))
            throw std::logic_error(
                "nwgrad: DP tables have been dropped; call align_full() or realign_banded() first");
        if (grad_mode_ == GradMode::None)
            throw std::logic_error(
                "nwgrad: grad_mode is None; construct with Hard or Soft to enable gradients");
        flags_[i] = static_cast<uint8_t>((flags_[i] & ~kHeld) | kGrad);
    }

    // The cached scores, in pair order.  Throws if any pair has none.
    std::vector<double> scores() const {
        for (size_t i = 0; i < size(); ++i)
            if (!score_valid(i))
                throw std::logic_error(
                    "nwgrad: score not valid; call align_full() or realign_banded() first");
        return score_;
    }

    // sum_i w_i * grad_i over the cached gradients, in fixed blocks of
    // WEIGHTED_GRAD_BLOCK pairs (each in pair order, the blocks in parallel, then the
    // block sums in block order): bit-reproducible, independent of n_threads, and the
    // same arithmetic, element for element, as the old per-object sum.  No product is
    // formed next to the add that consumes it (scale_into_; see hb_kernel_impl.inl's ramp).
    static constexpr size_t WEIGHTED_GRAD_BLOCK = 4096;
    AlignParams weighted_grad(const double* weights, size_t n) const {
        const Alphabet& alpha = alphabet();   // throws if empty
        if (n != size())
            throw std::invalid_argument(
                "nwgrad: weighted_grad() needs one weight per pair (got " + std::to_string(n) +
                " weights for " + std::to_string(size()) + " pairs)");
        check_all_grads_();
        return blocked_sum_(alpha, weights);
    }
    // The summed gradient (weights all 1: w*g == g exactly, so this is the plain blocked
    // sum).  Unlike the old per-thread completion-order merge, reproducible bit for bit.
    // Releases held gradients first (compute_grad_one, in pair order: the first pair
    // that cannot raises its own message).
    AlignParams compute_grad() {
        const Alphabet& alpha = alphabet();
        for (size_t i = 0; i < size(); ++i) compute_grad_one(i);
        return blocked_sum_(alpha, nullptr);
    }
    // The cached gradients as arrays: matrices[i*n*n ..] (row-major, alphabet order) and
    // gaps[i*4 ..] (gap_open_a, gap_extend_a, gap_open_b, gap_extend_b).
    void grads_into(double* matrices, double* gaps) const {
        alphabet();
        check_all_grads_();
        for (size_t i = 0; i < size(); ++i) {
            const double* g = grad_.data() + i * gstride();
            std::memcpy(matrices + i * nn_, g, nn_ * sizeof(double));
            std::memcpy(gaps + i * 4, g + nn_, 4 * sizeof(double));
        }
    }

    // ── Streaming align (what BatchAligner was) ─────────────────────────────
    //
    // Align `problems` under `params` and return each score plus the gradient SUMMED over
    // them, storing nothing per pair: sequences are encoded per worker as they are
    // reached, so memory stays O(threads), not O(N).  The batch's own pairs are not
    // touched; its type, grad mode, threads, soft settings, traceback and fill apply.
    // Hard: Viterbi + path counts; Soft: forward-backward only (no Viterbi); None:
    // Viterbi score.  band > 0 bands every problem (an empty guide_j gets the diagonal);
    // band == 0 with guides bands at width 0 — all problems or none must carry one.
    // A port of BatchAlignerT::align: same per-problem calls, same grouping.  The
    // gradient is merged per thread in completion order (hard counts: exact; soft: the
    // last bits may vary between runs).
    BatchResult align_stream(const std::vector<ProblemInstance>& problems,
                             const AlignParams& params, int band = 0,
                             int kernel = kBackendAuto) const {
        const size_t N = problems.size();
        BatchResult result(params.matrix.alphabet());
        result.scores.resize(N, 0.0);
        if (N == 0) return result;
        if (alpha_ && &params.matrix.alphabet() != alpha_)
            throw std::invalid_argument(
                "nwgrad: align() params over alphabet \"" + params.matrix.alphabet().symbols() +
                "\" in a batch over alphabet \"" + alpha_->symbols() + "\"");
        if (band < 0)
            throw std::invalid_argument("nwgrad: band must be >= 0, got " + std::to_string(band));
        // AlignBand is a compile-time parameter: one call is either all Full or all
        // GuideBanded.  A band-0 call where only SOME problems carry a guide would drop
        // the guides silently — reject it on the caller's thread.
        if (band == 0) {
            bool any_guided = false, any_unguided = false;
            for (const auto& p : problems) (p.guide_j.empty() ? any_unguided : any_guided) = true;
            if (any_guided && any_unguided)
                throw std::invalid_argument(
                    "nwgrad: mixed batch with band == 0 — some problems carry a "
                    "guide_j and others don't. All problems in one align() call "
                    "must either all supply a guide_j or none of them; split "
                    "into separate align() calls, or set band > 0 so an unguided "
                    "problem gets an automatic diagonal guide instead.");
        }
        const bool banded = band > 0 || !problems[0].guide_j.empty();
        if (inter_fill) {
            const int be = stream_inter_backend_(params, band, kernel);
            if (be >= 0) {
                if (banded) stream_inter_<AlignBand::GuideBanded>(problems, params, band, kernel, result, be);
                else        stream_inter_<AlignBand::Full>(problems, params, band, kernel, result, be);
                return result;
            }
        }
        std::atomic<size_t> work_idx{0};
        std::mutex grad_mutex;
        auto worker = [&]() {
            AlignParams local_grad = AlignParams::zeros_like(params);
            if (banded) stream_loop_<AlignBand::GuideBanded>(problems, params, band, kernel, work_idx, result.scores, local_grad);
            else        stream_loop_<AlignBand::Full>(problems, params, band, kernel, work_idx, result.scores, local_grad);
            if (grad_mode_ != GradMode::None) {
                std::lock_guard<std::mutex> lock(grad_mutex);
                result.grad += local_grad;
            }
        };
        run_workers_guarded(std::min<int>(n_threads_, static_cast<int>(std::min<size_t>(N, 1u << 30))), worker);
        return result;
    }

private:
    // Validate and encode one sequence, naming the PAIR and which sequence on failure —
    // the aligner's own message could only name a position in an anonymous string.
    static void stream_encode_(const Alphabet& alpha, std::string_view s, std::vector<uint8_t>& out,
                               size_t pair_idx, char which) {
        out.clear();
        out.reserve(s.size());
        for (size_t k = 0; k < s.size(); ++k) {
            const int i = alpha.index_of(s[k]);
            if (i < 0) {
                std::string msg = "nwgrad: pair " + std::to_string(pair_idx) + ", sequence ";
                msg += which;
                msg += ": character '";
                msg += s[k];
                msg += "' at position " + std::to_string(k) + " is not in alphabet \"" +
                       alpha.symbols() + "\"";
                throw std::invalid_argument(msg);
            }
            out.push_back(static_cast<uint8_t>(i));
        }
    }
    template<AlignBand AB>
    void configure_stream_(Aligner<GM, AM, AB, T>& al, int kernel) const {
        al.set_kernel(kernel);
        al.set_soft_impl(soft_impl_);
        al.set_soft_temperature(soft_temp_);
        if constexpr (AB == AlignBand::Full) {
            al.set_traceback(tb_);
            al.set_hb_cutoff(hb_cutoff_);
            al.set_rowwise_full(rowwise_full);
        }
    }
    template<AlignBand AB>
    void stream_loop_(const std::vector<ProblemInstance>& problems, const AlignParams& params,
                      int band, int kernel, std::atomic<size_t>& work_idx,
                      std::vector<double>& scores, AlignParams& local_grad) const {
        Aligner<GM, AM, AB, T> al;
        configure_stream_<AB>(al, kernel);
        DpBuffer buf;
        const Alphabet& alpha = params.matrix.alphabet();
        std::vector<uint8_t> a_enc, b_enc;
        const size_t N = problems.size();
        for (size_t idx; (idx = work_idx.fetch_add(1, std::memory_order_relaxed)) < N; ) {
            const auto& p = problems[idx];
            stream_encode_(alpha, p.seq_a, a_enc, idx, 'a');
            stream_encode_(alpha, p.seq_b, b_enc, idx, 'b');
            al.set_problem(a_enc, b_enc, params, band, p.guide_j);
            if (grad_mode_ == GradMode::Hard) {
                al.compute_viterbi(buf); scores[idx] = al.score(); al.hard_grad(buf, local_grad);
            } else if (grad_mode_ == GradMode::Soft) {
                al.compute_forward_back(buf); scores[idx] = al.log_z(); al.soft_grad(buf, local_grad);
            } else {
                al.compute_viterbi(buf); scores[idx] = al.score();
            }
        }
    }
    // The level a shared pass would run on, or -1 (every problem its own path): the
    // scalar backend, soft_impl "log" or a banded soft call, linear Global Viterbi below
    // 4 lanes or over 8 letters, linear banded.
    int stream_inter_backend_(const AlignParams& params, int band, int kernel) const {
        const int backend = (kernel == kBackendAuto) ? global_default_backend() : kernel;
        if (backend < 0) return -1;
        const LevelKernels& K = level_kernels(backend);
        if (grad_mode_ == GradMode::Soft)
            return (K.inter_soft && K.inter_w > 0 && soft_impl_ != SoftImpl::Log && band == 0) ? backend : -1;
        if (inter_w_(K) <= 0) return -1;
        if (GM == GapModel::Linear &&
            (band > 0 || (AM == AlignMode::Global && (inter_w_(K) < 4 || params.matrix.size() > 8))))
            return -1;
        return backend;
    }
    // Problems grouped W at a time, sorted by len B then len A (counting sorts: each call
    // brings new problems, nothing is cached).  The rest run their own path in the same
    // pass: empty sequences, and Full pairs the traceback would split (Hirschberg past
    // the cutoff).  A guided band-0 call is GuideBanded and stays so.
    template<AlignBand AB>
    void stream_inter_(const std::vector<ProblemInstance>& problems, const AlignParams& params,
                       int band, int kernel, BatchResult& result, int backend) const {
        const size_t N = problems.size();
        const LevelKernels& K = level_kernels(backend);
        const bool soft = grad_mode_ == GradMode::Soft;
        const int W = soft ? K.inter_w : inter_w_(K);
        const bool hb_case = !soft && AB == AlignBand::Full && is_hirschberg(tb_resolved_);
        std::vector<size_t> other, elig;
        size_t maxa = 0, maxb = 0;
        for (size_t i = 0; i < N; ++i) {
            const size_t la = problems[i].seq_a.size(), lb = problems[i].seq_b.size();
            if (la == 0 || lb == 0 || (hb_case && la > static_cast<size_t>(hb_cutoff_)) ||
                !inter_pair_fits(la, lb, GM == GapModel::Affine, soft, inter_w_(K), sizeof(T), K.inter_w)) {
                other.push_back(i); continue;
            }
            elig.push_back(i);
            maxa = std::max(maxa, la); maxb = std::max(maxb, lb);
        }
        auto csort = [&](auto key, size_t kmax) {   // LSD counting sort: len A, then len B
            std::vector<size_t> cnt(kmax + 2, 0), out(elig.size());
            for (size_t i : elig) ++cnt[key(i) + 1];
            for (size_t k = 1; k < cnt.size(); ++k) cnt[k] += cnt[k - 1];
            for (size_t i : elig) out[cnt[key(i)]++] = i;
            elig.swap(out);
        };
        csort([&](size_t i) { return problems[i].seq_a.size(); }, maxa);
        csort([&](size_t i) { return problems[i].seq_b.size(); }, maxb);
        std::vector<std::pair<size_t, size_t>> groups;
        for (size_t s = 0; s < elig.size();) {
            const size_t lb = problems[elig[s]].seq_b.size();
            size_t e = s + 1;
            while (e < elig.size() && e - s < static_cast<size_t>(W) &&
                   problems[elig[e]].seq_b.size() * 4 <= lb * 5 + 16) ++e;
            groups.emplace_back(s, e);
            s = e;
        }
        std::atomic<size_t> idx{0};
        std::mutex grad_mutex;
        auto worker = [&]() {
            AlignParams local_grad = AlignParams::zeros_like(params);
            stream_inter_loop_<AB>(problems, params, band, kernel, elig, groups, other, idx,
                                   result.scores, local_grad, K, W);
            if (grad_mode_ != GradMode::None) {
                std::lock_guard<std::mutex> lock(grad_mutex);
                result.grad += local_grad;
            }
        };
        const size_t tasks = groups.size() + other.size();
        run_workers_guarded(std::min<int>(n_threads_, static_cast<int>(std::max<size_t>(tasks, 1))), worker);
    }
    template<AlignBand AB>
    void stream_inter_loop_(const std::vector<ProblemInstance>& problems, const AlignParams& params,
                            int band, int kernel, const std::vector<size_t>& elig,
                            const std::vector<std::pair<size_t, size_t>>& groups,
                            const std::vector<size_t>& other, std::atomic<size_t>& idx,
                            std::vector<double>& scores, AlignParams& local_grad,
                            const LevelKernels& K, int W) const {
        using Al = Aligner<GM, AM, AB, T>;
        std::vector<Al> al(W);
        for (Al& x : al) configure_stream_<AB>(x, kernel);
        DpBuffer buf;
        const Alphabet& alpha = params.matrix.alphabet();
        std::vector<std::vector<uint8_t>> ae(W), be(W);
        std::vector<const unsigned char*> a(W), b(W);
        std::vector<int> m(W), bi(W), bj(W), blo, bhi, ulo, uhi, bri(W), brj(W), nb(W);
        std::vector<T> best(W), blkT;
        constexpr bool lin = GM == GapModel::Linear;
        const bool soft = grad_mode_ == GradMode::Soft;
        std::vector<double> es, slogz, scnt, sgap;
        DVec sscr;
        std::vector<int> siscr, sok;
        InterSoftJob sj{};
        const bool soft_ok = soft && inter_soft_weights(params, soft_temp_, lin, es, sj);
        const size_t nn = static_cast<size_t>(params.matrix.size()) * params.matrix.size();
        if (!soft) {
            blkT.resize(nn);
            for (size_t k = 0; k < nn; ++k) blkT[k] = static_cast<T>(params.matrix.data()[k]);
        }
        auto own = [&](size_t i) {   // one problem on its own path: stream_loop_'s body
            Al& x = al[0];
            const auto& p = problems[i];
            stream_encode_(alpha, p.seq_a, ae[0], i, 'a');
            stream_encode_(alpha, p.seq_b, be[0], i, 'b');
            x.set_problem(ae[0], be[0], params, band, p.guide_j);
            if (grad_mode_ == GradMode::Hard) {
                x.compute_viterbi(buf); scores[i] = x.score(); x.hard_grad(buf, local_grad);
            } else if (grad_mode_ == GradMode::Soft) {
                x.compute_forward_back(buf); scores[i] = x.log_z(); x.soft_grad(buf, local_grad);
            } else {
                x.compute_viterbi(buf); scores[i] = x.score();
            }
        };
        const size_t G = groups.size(), tasks = G + other.size();
        for (size_t t; (t = idx.fetch_add(1, std::memory_order_relaxed)) < tasks; ) {
            if (t >= G) { own(other[t - G]); continue; }
            const auto [s, e] = groups[t];
            const size_t real = e - s;
            if (soft && !soft_ok) { for (size_t k = s; k < e; ++k) own(elig[k]); continue; }
            int M = 0, n = 0;
            bool ragged = false;
            for (int l = 0; l < W; ++l) {
                const size_t lr = std::min<size_t>(l, real - 1);   // spare lanes repeat the last
                if (static_cast<size_t>(l) == lr) {
                    const size_t i = elig[s + lr];
                    stream_encode_(alpha, problems[i].seq_a, ae[l], i, 'a');
                    stream_encode_(alpha, problems[i].seq_b, be[l], i, 'b');
                }
                a[l] = ae[lr].data(); b[l] = be[lr].data();
                m[l] = static_cast<int>(ae[lr].size());
                nb[l] = static_cast<int>(be[lr].size());
                M = std::max(M, m[l]);
                n = std::max(n, nb[l]);
                ragged |= nb[l] != nb[0];
            }
            const size_t gstride = ragged ? static_cast<size_t>(n) + 1 : 0;
            if (soft) {
                const size_t need = inter_soft_scratch(n, M, W);
                if (sscr.size() < need) sscr.resize(need);
                siscr.resize(static_cast<size_t>(M + 1) * W);
                slogz.resize(W); scnt.resize(W * nn); sgap.resize(W * 4); sok.resize(W);
                sj.a = a.data(); sj.m = m.data(); sj.b = b.data(); sj.n = n; sj.M = M;
                sj.nb = ragged ? nb.data() : nullptr;
                sj.align_mode = (AM == AlignMode::Local) ? 1 : 0;
                sj.scratch = sscr.data(); sj.iscratch = siscr.data();
                sj.logz = slogz.data(); sj.counts = scnt.data(); sj.gaps = sgap.data();
                sj.ok = sok.data();
                K.inter_soft(sj);
                for (size_t l = 0; l < real; ++l) {
                    const size_t i = elig[s + l];
                    if (!sok[l]) { own(i); continue; }
                    scores[i] = slogz[l] * soft_temp_;
                    double* g = local_grad.matrix.data();
                    for (size_t k = 0; k < nn; ++k) g[k] += scnt[l * nn + k];
                    local_grad.gap_open_a += sgap[l * 4 + 0]; local_grad.gap_extend_a += sgap[l * 4 + 1];
                    local_grad.gap_open_b += sgap[l * 4 + 2]; local_grad.gap_extend_b += sgap[l * 4 + 3];
                }
                continue;
            }
            if constexpr (AB == AlignBand::GuideBanded) {
                blo.assign(static_cast<size_t>(M + 1) * W, 1);
                bhi.assign(static_cast<size_t>(M + 1) * W, 0);
                ulo.assign(M + 1, n + 1); uhi.assign(M + 1, 0);
                std::fill(bri.begin(), bri.end(), 0); std::fill(brj.begin(), brj.end(), 0);
            }
            for (size_t l = 0; l < real; ++l) {
                al[l].set_problem(ae[l], be[l], params, band, problems[elig[s + l]].guide_j);
                if constexpr (AB == AlignBand::GuideBanded)
                    al[l].banded_lane_rows(W, static_cast<int>(l), M, blo.data(), bhi.data(),
                                           ulo.data(), uhi.data(), bri[l], brj[l]);
            }
            const size_t sz = static_cast<size_t>(M + 1) * (n + 1) * W;
            InterJobT<T> job{};
            job.a = a.data(); job.m = m.data(); job.b = b.data(); job.n = n; job.M = M;
            job.align_mode = (AM == AlignMode::Local) ? 1 : 0;
            if (ragged) job.nb = nb.data();
            job.blk = blkT.data(); job.nalpha = params.matrix.size();
            job.go_a = static_cast<T>(params.gap_open_a); job.ge_a = static_cast<T>(params.gap_extend_a);
            job.go_b = static_cast<T>(params.gap_open_b); job.ge_b = static_cast<T>(params.gap_extend_b);
            job.linear = lin ? 1 : 0;
            if constexpr (lin) { if (buf.H.size() < sz) buf.H.resize(sz); job.VM = buf.H.data(); }
            else {
                for (auto* v : {&buf.VM, &buf.VX, &buf.VY}) if (v->size() < sz) v->resize(sz);
                job.VM = buf.VM.data(); job.VX = buf.VX.data(); job.VY = buf.VY.data();
            }
            job.best = best.data(); job.best_i = bi.data(); job.best_j = bj.data();
            if constexpr (AB == AlignBand::GuideBanded) {
                job.blo = blo.data(); job.bhi = bhi.data();
                job.ulo = ulo.data(); job.uhi = uhi.data();
                job.bri = bri.data(); job.brj = brj.data();
            }
            if constexpr (std::is_same_v<T, double>) K.inter_fill(job);
            else                                     K.inter_fill_f(job);
            for (size_t l = 0; l < real; ++l) {
                const size_t i = elig[s + l];
                al[l].adopt_interleaved(buf, W, static_cast<int>(l), best[l], bi[l], bj[l], gstride);
                scores[i] = al[l].score();
                if (grad_mode_ == GradMode::Hard) al[l].hard_grad(buf, local_grad);
            }
        }
    }

public:
private:
    // ── Per-pair state ───────────────────────────────────────────────────────
    enum : uint8_t {
        kPath = 1,         // a guide exists (banded_grad may run)
        kScore = 2,        // score_ is current
        kGrad = 4,         // grad_ is current
        kPending = 8,      // lazy soft guide not computed yet
        kLastBanded = 16,  // the cached results came from a banded DP
        kStored = 32,      // path_ holds the current alignment
        kHeld = 64,        // grad_ holds a gradient compute_grad() has not released yet
    };
    struct Segment { const AlignParams* params; int kernel; size_t begin; };

    int n_threads_;
    GradMode grad_mode_;
    TracebackMode tb_, tb_resolved_ = TracebackMode::Pointers;
    int hb_cutoff_ = 512;
    SoftImpl soft_impl_ = SoftImpl::Scaled;
    double soft_temp_ = 1.0;
    bool soft_lazy_ = false, soft_post_ = false;

    const Alphabet* alpha_ = nullptr;
    size_t nn_ = 0;
    std::vector<uint8_t> ca_, cb_;          // all pairs' codes, A and B
    std::vector<size_t>  oa_{0}, ob_{0};    // pair i: [oa_[i], oa_[i+1])
    std::vector<int>     guide_;            // pair i: m+1 entries at oa_[i] + i
    std::vector<double>  score_;
    std::vector<uint8_t> flags_;
    std::vector<uint32_t> seg_;
    std::vector<Segment> segs_;
    std::vector<double>  grad_;             // pair i: gstride() doubles (matrix, 4 gaps)
    // Stored paths (keep_paths): run-length ops, (type << 30) | length, type 0 = both
    // advance, 1 = gap in B (A advances), 2 = gap in A; and the cell the path ends at.
    std::vector<std::vector<uint32_t>> path_;
    std::vector<std::array<int, 2>>    path_end_;

    size_t gstride() const noexcept { return nn_ + 4; }
    int* guide_ptr_(size_t i) noexcept { return guide_.data() + oa_[i] + i; }
    const int* guide_ptr_(size_t i) const noexcept { return guide_.data() + oa_[i] + i; }
    double* grad_ptr_(size_t i) noexcept { return grad_.data() + i * gstride(); }

    std::string decode_(std::span<const uint8_t> c) const {
        std::string s(c.size(), '\0');
        for (size_t k = 0; k < c.size(); ++k) s[k] = alpha_->symbol_at(c[k]);
        return s;
    }
    void set_flag_(size_t i, uint8_t f, bool on) noexcept {
        flags_[i] = on ? static_cast<uint8_t>(flags_[i] | f) : static_cast<uint8_t>(flags_[i] & ~f);
    }

    void check_stored_(size_t i) const {
        if (!path_valid(i))
            throw std::logic_error("nwgrad: alignment not computed; call align_full() first");
        if (!path_stored(i))
            throw std::logic_error(
                "nwgrad: DP tables have been dropped; call align_full() or realign_banded() first");
    }
    std::pair<int, int> path_start_(size_t i) const {
        int x = path_end_[i][0], y = path_end_[i][1];
        for (uint32_t op : path_[i]) {
            const uint32_t t = op >> 30, len = op & 0x3fffffffu;
            if (t != 2) x -= static_cast<int>(len);
            if (t != 1) y -= static_cast<int>(len);
        }
        return {x, y};
    }
    void check_all_grads_() const {
        for (size_t i = 0; i < size(); ++i)
            if (!grad_valid(i))
                throw std::logic_error("nwgrad: gradient not computed; call compute_grad() first");
    }
    void prepare_paths_(bool keep, bool all = true) {
        if (!keep) {
            if (all) for (auto& f : flags_) f &= static_cast<uint8_t>(~kStored);
            return;
        }
        if (path_.size() != size()) { path_.resize(size()); path_end_.resize(size()); }
    }
    double sum_scores_() const {
        // Pair-index order, never completion order: the total is reproducible.
        return std::accumulate(score_.begin(), score_.end(), 0.0);
    }

    // dst[k] = s * src[k].  Out of line on purpose (see weighted_grad).
    [[gnu::noinline]] static void scale_into_(double* dst, const double* src, size_t n, double s) {
        for (size_t k = 0; k < n; ++k) dst[k] = src[k] * s;
    }
    AlignParams blocked_sum_(const Alphabet& alpha, const double* weights) const {
        const size_t n = size(), gs = gstride();
        const size_t nblocks = (n + WEIGHTED_GRAD_BLOCK - 1) / WEIGHTED_GRAD_BLOCK;
        std::vector<double> partial(nblocks * gs, 0.0);
        std::atomic<size_t> next{0};
        auto worker = [&]() {
            std::vector<double> scaled(gs);
            for (size_t b; (b = next.fetch_add(1, std::memory_order_relaxed)) < nblocks; ) {
                double* acc = partial.data() + b * gs;
                const size_t end = std::min(n, (b + 1) * WEIGHTED_GRAD_BLOCK);
                for (size_t i = b * WEIGHTED_GRAD_BLOCK; i < end; ++i) {
                    const double* g = grad_.data() + i * gs;
                    if (weights) {
                        scale_into_(scaled.data(), g, gs, weights[i]);
                        for (size_t k = 0; k < gs; ++k) acc[k] += scaled[k];
                    } else {
                        for (size_t k = 0; k < gs; ++k) acc[k] += g[k];
                    }
                }
            }
        };
        if (nblocks > 0)
            run_workers_guarded(std::min<int>(n_threads_, static_cast<int>(nblocks)), worker);
        std::vector<double> tot(gs, 0.0);
        for (size_t b = 0; b < nblocks; ++b)
            for (size_t k = 0; k < gs; ++k) tot[k] += partial[b * gs + k];
        AlignParams out(alpha);
        std::memcpy(out.matrix.data(), tot.data(), nn_ * sizeof(double));
        out.gap_open_a = tot[nn_]; out.gap_extend_a = tot[nn_ + 1];
        out.gap_open_b = tot[nn_ + 2]; out.gap_extend_b = tot[nn_ + 3];
        return out;
    }

    // ── A worker's state: one buffer, one Aligner per band, gradient scratch ─────
    struct Work {
        DpBuffer buf;
        FullAl full;
        BandAl band;
        AlignParams g;
        std::vector<int> gj;
        explicit Work(const BatchEngine& e) : g(e.alphabet()) { configure(e); }
        // The batch's settings, (re)applied: a reused Work follows setting changes.
        void configure(const BatchEngine& e) {
            full.set_traceback(e.tb_);
            full.set_hb_cutoff(e.hb_cutoff_);
            full.set_rowwise_full(e.rowwise_full);
            full.set_soft_impl(e.soft_impl_);
            full.set_soft_temperature(e.soft_temp_);
            // Only the Full aligner can use pointers; the banded one needs its score
            // tables and its footprint is O(m*band) already.
            band.set_traceback(TracebackMode::Scores);
            band.set_soft_impl(e.soft_impl_);
            band.set_soft_temperature(e.soft_temp_);
        }
    };
    // The calling thread's Work for single-pair operations (SeqPair's align_full() and
    // friends), kept between calls: a fresh one per call re-allocated the DP tables
    // every time, which the pre-0.6 SeqPair (pair-owned tables) did not — measured
    // 4-10 % per repeated call.  Same thread-safety as SeqPair had: one caller at a time.
    std::unique_ptr<Work> solo_;
    Work& solo_work_() {
        if (!solo_) solo_ = std::make_unique<Work>(*this);
        else        solo_->configure(*this);
        return *solo_;
    }

    template<class Al>
    void set_problem_(Al& al, size_t i) const {
        al.set_kernel(kernel(i));
        al.set_problem(codes_a(i), codes_b(i), params(i));
    }
    void store_grad_(size_t i, const AlignParams& g) {
        double* d = grad_ptr_(i);
        std::memcpy(d, g.matrix.data(), nn_ * sizeof(double));
        d[nn_] = g.gap_open_a; d[nn_ + 1] = g.gap_extend_a;
        d[nn_ + 2] = g.gap_open_b; d[nn_ + 3] = g.gap_extend_b;
    }
    void store_guide_(size_t i, const std::vector<int>& gj) {
        if (gj.size() != len_a(i) + 1)
            throw std::logic_error("nwgrad: internal: guide of the wrong length");
        std::copy(gj.begin(), gj.end(), guide_ptr_(i));
    }
    template<class Al>
    void store_path_(size_t i, const Al& al, const DpBuffer& buf) {
        const auto [sa, sb] = al.aligned(buf);
        const auto end = al.alignment_end();
        auto& ops = path_[i];
        ops.clear();
        uint32_t t_prev = 3, len = 0;
        for (size_t k = 0; k < sa.size(); ++k) {
            const uint32_t t = (sa[k] == '-') ? 2u : (sb[k] == '-') ? 1u : 0u;
            if (t != t_prev && len) { ops.push_back((t_prev << 30) | len); len = 0; }
            t_prev = t; ++len;
        }
        if (len) ops.push_back((t_prev << 30) | len);
        path_end_[i] = {end.first, end.second};
        flags_[i] |= kStored;
    }
    // Hard or soft gradient of the DP `al` just ran, into pair i.
    template<class Al>
    void grad_with_buf_(size_t i, Al& al, Work& w) {
        w.g.zero();
        if (grad_mode_ == GradMode::Hard) al.hard_grad(w.buf, w.g);
        else                              al.soft_grad(w.buf, w.g);
        store_grad_(i, w.g);
    }
    // SeqPair::run_dp_with_buf: Viterbi, guide, then forward-backward for a soft pair.
    template<class Al>
    void run_dp_with_buf_(size_t i, Al& al, Work& w, bool keep) {
        al.compute_viterbi(w.buf);
        store_guide_(i, al.guide_j_from_viterbi(w.buf));
        if (keep) store_path_(i, al, w.buf);
        if (grad_mode_ == GradMode::Soft) {
            al.compute_forward_back(w.buf);
            score_[i] = al.log_z();
        } else {
            score_[i] = al.score();
        }
    }
    void hold_(size_t i) noexcept {
        if (flags_[i] & kGrad) flags_[i] = static_cast<uint8_t>((flags_[i] & ~kGrad) | kHeld);
    }
    void hold_all_() noexcept { for (size_t i = 0; i < size(); ++i) hold_(i); }
    void finish_(size_t i, bool banded, bool pending) {
        uint8_t f = flags_[i] & kStored;
        f |= kPath | kScore;
        if (grad_mode_ != GradMode::None) f |= kGrad;
        if (banded)  f |= kLastBanded;
        if (pending) f |= kPending;
        flags_[i] = f;
    }

    // SeqPair::score_and_grad_with_dp.
    void run_full_(size_t i, Work& w, bool keep) {
        flags_[i] &= static_cast<uint8_t>(~kStored);
        auto& al = w.full;
        set_problem_(al, i);
        const bool lazy = soft_lazy_ && !keep, post = soft_post_ && !keep;
        if (grad_mode_ == GradMode::Hard) {
            al.compute_viterbi(w.buf);
            score_[i] = al.score();
            w.g.zero();
            al.hard_grad_and_guide(w.buf, w.g, w.gj);
            store_grad_(i, w.g);
            store_guide_(i, w.gj);
            if (keep) store_path_(i, al, w.buf);
            finish_(i, false, false);
            return;
        }
        if (grad_mode_ == GradMode::Soft && (lazy || post)) {
            al.set_posterior_guide(post);
            al.compute_forward_back(w.buf);
            al.set_posterior_guide(false);
            score_[i] = al.log_z();
            grad_with_buf_(i, al, w);
            if (post) store_guide_(i, al.posterior_guide());
            finish_(i, false, lazy);
            return;
        }
        run_dp_with_buf_(i, al, w, keep);
        if (grad_mode_ != GradMode::None) grad_with_buf_(i, al, w);
        finish_(i, false, false);
    }

    // SeqPair::resolve_guide: the pending lazy guide, under the params current now.
    void resolve_guide_(size_t i, Work& w) {
        if (!guide_pending(i)) return;
        set_problem_(w.full, i);
        w.full.compute_viterbi(w.buf);
        store_guide_(i, w.full.guide_j_from_viterbi(w.buf));
        flags_[i] &= static_cast<uint8_t>(~(kPending | kStored));
    }

    // SeqPair::banded_grad_with_dp.
    void run_banded_(size_t i, Work& w, int bandwidth, bool keep) {
        resolve_guide_(i, w);
        flags_[i] &= static_cast<uint8_t>(~kStored);
        auto& al = w.band;
        al.set_kernel(kernel(i));
        const int* g = guide_ptr_(i);
        al.set_problem(codes_a(i), codes_b(i), params(i), bandwidth,
                       std::vector<int>(g, g + len_a(i) + 1));
        if (grad_mode_ == GradMode::Hard) {   // one traceback walk for guide and gradient
            al.compute_viterbi(w.buf);
            score_[i] = al.score();
            w.g.zero();
            al.hard_grad_and_guide(w.buf, w.g, w.gj);
            store_grad_(i, w.g);
            store_guide_(i, w.gj);
            if (keep) store_path_(i, al, w.buf);
            finish_(i, true, false);
            return;
        }
        run_dp_with_buf_(i, al, w, keep);
        if (grad_mode_ != GradMode::None) grad_with_buf_(i, al, w);
        finish_(i, true, false);
    }

    // A soft lane's result from the inter-pair soft pass (SeqPair::SoftLane).
    struct SoftLane { double logz; const double* counts; const double* gaps;
                      const int* gpost = nullptr; int gstride = 0; };
    void apply_soft_lane_(size_t i, const SoftLane& soft) {
        score_[i] = soft.logz * soft_temp_;
        double* d = grad_ptr_(i);
        for (size_t k = 0; k < nn_; ++k) d[k] = soft.counts[k];
        for (size_t k = 0; k < 4; ++k) d[nn_ + k] = soft.gaps[k];
        if (soft_post_ && soft.gpost) {
            const int m = static_cast<int>(len_a(i)), n = static_cast<int>(len_b(i));
            std::vector<int> rows(static_cast<size_t>(m) + 1);
            for (int r = 0; r <= m; ++r) rows[static_cast<size_t>(r)] = soft.gpost[static_cast<size_t>(r) * soft.gstride];
            store_guide_(i, FullAl::monotone_guide(rows.data(), m, n));
        }
    }

    // SeqPair::score_and_grad_interleaved: pair i is lane `lane` of an inter-pair fill.
    void run_interleaved_(size_t i, Work& w, int W, int lane, double best, int bi, int bj,
                          const SoftLane* soft, size_t stride, bool keep) {
        flags_[i] &= static_cast<uint8_t>(~kStored);
        auto& al = w.full;
        set_problem_(al, i);
        al.adopt_interleaved(w.buf, W, lane, best, bi, bj, stride);
        score_[i] = al.score();
        if (grad_mode_ == GradMode::Hard) {
            w.g.zero();
            al.hard_grad_and_guide(w.buf, w.g, w.gj);   // one walk for both
            store_grad_(i, w.g);
            store_guide_(i, w.gj);
        } else {
            store_guide_(i, al.guide_j_from_viterbi(w.buf));
            if (grad_mode_ == GradMode::Soft && soft) {
                apply_soft_lane_(i, *soft);
            } else if (grad_mode_ == GradMode::Soft) {
                if (keep) store_path_(i, al, w.buf);   // before forward-backward reuses buf
                al.compute_forward_back(w.buf);
                score_[i] = al.log_z();
                grad_with_buf_(i, al, w);
                finish_(i, false, false);
                return;
            }
        }
        if (keep) store_path_(i, al, w.buf);
        finish_(i, false, false);
    }

    // SeqPair::score_and_grad_with_soft_lane: a soft pair whose forward-backward ran in
    // the inter-pair soft pass, with no shared Viterbi fill.  soft == nullptr: own path.
    void run_soft_lane_(size_t i, Work& w, const SoftLane* soft, bool keep) {
        if (!soft || grad_mode_ != GradMode::Soft) { run_full_(i, w, keep); return; }
        flags_[i] &= static_cast<uint8_t>(~kStored);
        const bool lazy = soft_lazy_ && !keep, post = soft_post_ && !keep;
        bool pending = false;
        if (lazy) pending = true;
        else if (post && soft->gpost) pending = false;   // apply_soft_lane_ sets the guide
        else {
            set_problem_(w.full, i);
            w.full.compute_viterbi(w.buf);
            store_guide_(i, w.full.guide_j_from_viterbi(w.buf));
            if (keep) store_path_(i, w.full, w.buf);
        }
        apply_soft_lane_(i, *soft);
        finish_(i, false, pending);
    }

    // SeqPair::banded_lane_setup / banded_grad_interleaved.
    void banded_lane_setup_(size_t i, Work& w, int bandwidth, int W, int lane, int M,
                            int* blo, int* bhi, int* slo, int* shi, int& bri, int& brj) {
        const int* g = guide_ptr_(i);
        w.band.set_kernel(kernel(i));
        w.band.set_problem(codes_a(i), codes_b(i), params(i), bandwidth,
                           std::vector<int>(g, g + len_a(i) + 1));
        w.band.banded_lane_rows(W, lane, M, blo, bhi, slo, shi, bri, brj);
    }
    void banded_interleaved_(size_t i, Work& w, int bandwidth, int W, int lane,
                             double best, int bi, int bj) {
        // The lane's problem was set by banded_lane_setup_ on a fresh BandAl per lane?
        // No: one BandAl per worker, so set it again (set_problem is cheap and the
        // banded tables live in the shared buffer, untouched by set_problem).
        const int* g = guide_ptr_(i);
        auto& al = w.band;
        al.set_kernel(kernel(i));
        al.set_problem(codes_a(i), codes_b(i), params(i), bandwidth,
                       std::vector<int>(g, g + len_a(i) + 1));
        flags_[i] &= static_cast<uint8_t>(~kStored);
        al.adopt_interleaved(w.buf, W, lane, best, bi, bj);
        score_[i] = al.score();
        if (grad_mode_ == GradMode::Hard) {
            w.g.zero();
            al.hard_grad_and_guide(w.buf, w.g, w.gj);
            store_grad_(i, w.g);
            store_guide_(i, w.gj);
        } else {
            store_guide_(i, al.guide_j_from_viterbi(w.buf));
        }
        finish_(i, true, false);
    }

    // ── Inter-pair plan ──────────────────────────────────────────────────────

    static int inter_w_(const LevelKernels& K) {
        if constexpr (std::is_same_v<T, double>) return K.inter_fill ? K.inter_w : 0;
        else                                     return K.inter_fill_f ? K.inter_w_f : 0;
    }
    // Linear Global's own fill is cheap: the shared one loses below 4 lanes (DNA, W=2:
    // 0.76x on SSE2) and for alphabets over 8 (protein, the per-row gather: 1.10-1.22x
    // at 12 threads on AVX2).  Soft pairs still take the soft pass and skip the fill.
    bool linear_fill_ok_(const LevelKernels& K) const {
        return !(GM == GapModel::Linear && AM == AlignMode::Global &&
                 (inter_w_(K) < 4 || alpha_->size() > 8));
    }
    // The vector backend an inter-pair fill of pair i runs on, or -1 (own fill).
    int inter_backend_(size_t i) const {
        if (len_a(i) == 0 || len_b(i) == 0) return -1;
        const int backend = (kernel(i) == kBackendAuto) ? global_default_backend() : kernel(i);
        if (backend < 0) return -1;
        const LevelKernels& K = level_kernels(backend);
        if (inter_w_(K) <= 0) return -1;
        if (!inter_pair_fits(len_a(i), len_b(i), GM == GapModel::Affine,
                             grad_mode_ == GradMode::Soft, inter_w_(K), sizeof(T), K.inter_w))
            return -1;
        if (!linear_fill_ok_(K) && grad_mode_ != GradMode::Soft) return -1;
        // Hirschberg pairs past the cutoff split: no shared fill.
        if (is_hirschberg(tb_resolved_) && len_a(i) > static_cast<size_t>(hb_cutoff_)) return -1;
        return backend;
    }

public:
    // Ragged B: one group may mix B lengths (InterJobT::nb) — every lane is padded to the
    // longest, so a group admits a pair only while that padding stays small.
    static bool ragged_ok(size_t shortest, size_t len_b) {
        return len_b * 4 <= shortest * 5 + 16;   // <= 1.25x + 4 columns
    }
private:
    struct InterPlan {
        size_t generation = static_cast<size_t>(-1);
        int default_backend = -1000, hb_cutoff = -1;
        std::vector<size_t> elig, other;
        std::vector<int> backend;
        std::vector<std::pair<size_t, size_t>> groups;   // [start, end) into elig
    };
    InterPlan plan_;
    size_t generation_ = 0;   // bumped by add_many(); keys plan_

    // Built once and reused until pairs are added, the default ISA changes or the
    // Hirschberg cutoff moves: re-sorting 2.5M pairs every training iteration was ~0.9 s
    // of serial work per call.  Groups: (backend, len B) runs of len-A-sorted pairs, cut
    // every W and wherever len B outgrows the group's shortest past ragged_ok().
    void ensure_plan_() {
        const int def_backend = global_default_backend();
        if (plan_.generation == generation_ && plan_.default_backend == def_backend &&
            plan_.hb_cutoff == hb_cutoff_)
            return;
        const size_t N = size();
        InterPlan P;
        P.generation = generation_; P.default_backend = def_backend; P.hb_cutoff = hb_cutoff_;
        P.backend.assign(N, -1);
        struct Key { int backend; size_t len_b, len_a, i; };
        std::vector<Key> keys;
        keys.reserve(N);
        for (size_t i = 0; i < N; ++i) {
            P.backend[i] = inter_backend_(i);
            if (P.backend[i] < 0) { P.other.push_back(i); continue; }
            keys.push_back({P.backend[i], len_b(i), len_a(i), i});
        }
        std::sort(keys.begin(), keys.end(), [](const Key& x, const Key& y) {
            if (x.backend != y.backend) return x.backend < y.backend;
            if (x.len_b != y.len_b) return x.len_b < y.len_b;
            if (x.len_a != y.len_a) return x.len_a < y.len_a;
            return x.i < y.i;
        });
        P.elig.reserve(keys.size());
        for (const Key& k : keys) P.elig.push_back(k.i);
        for (size_t s = 0; s < keys.size();) {
            const int W = inter_w_(level_kernels(keys[s].backend));
            size_t e = s + 1;
            while (e < keys.size() && e - s < static_cast<size_t>(W) &&
                   keys[e].backend == keys[s].backend && ragged_ok(keys[s].len_b, keys[e].len_b))
                ++e;
            P.groups.emplace_back(s, e);
            s = e;
        }
        plan_ = std::move(P);
    }

    // fill = "interpair": each group one InterJob, then each lane's pair adopts its
    // tables for score, path and gradient; the other pairs run their own fill.  One
    // atomic counter over all tasks.  (SeqPairBatch::score_and_grad_inter_.)
    void score_and_grad_inter_(bool keep) {
        ensure_plan_();
        const auto& elig = plan_.elig;
        const auto& other = plan_.other;
        const auto& backend = plan_.backend;
        const auto& groups = plan_.groups;
        const size_t G = groups.size(), tasks = G + other.size();
        const bool soft = grad_mode_ == GradMode::Soft;
        // A lazy-guide soft group needs no Viterbi at all, nor a posterior-guide one.
        const bool no_viterbi = soft && (soft_lazy_ || soft_post_) && !keep;
        const bool want_post = soft && soft_post_ && !keep;
        std::atomic<size_t> idx{0};
        auto worker = [&]() {
            Work w(*this);
            std::vector<const unsigned char*> a, b;
            std::vector<int> m, bi, bj, nb;
            std::vector<T> best, blkT;
            DVec sscr;
            std::vector<int> siscr, sok, sgp;
            std::vector<double> ses, slogz, scnt, sgap;
            while (true) {
                const size_t t = idx.fetch_add(1, std::memory_order_relaxed);
                if (t >= tasks) break;
                if (t >= G) { run_full_(other[t - G], w, keep); continue; }
                const auto [s, e] = groups[t];
                const size_t real = e - s;
                const size_t i0 = elig[s];
                bool same_params = true;
                for (size_t l = 1; l < real; ++l)
                    same_params &= &params(elig[s + l]) == &params(i0);
                if (!same_params) {   // pairs given their own params: no shared fill
                    for (size_t l = 0; l < real; ++l) run_full_(elig[s + l], w, keep);
                    continue;
                }
                const LevelKernels& K = level_kernels(backend[i0]);
                const int W = inter_w_(K);
                a.assign(W, nullptr); b.assign(W, nullptr); m.assign(W, 0);
                bi.assign(W, 0); bj.assign(W, 0); best.assign(W, T(0));
                int M = 0, n = 0;
                bool ragged = false;
                nb.assign(W, 0);
                for (int l = 0; l < W; ++l) {
                    // Short group: the spare lanes repeat the last pair; results dropped.
                    const size_t i = elig[s + std::min<size_t>(l, real - 1)];
                    a[l] = codes_a(i).data(); b[l] = codes_b(i).data();
                    m[l] = static_cast<int>(len_a(i));
                    nb[l] = static_cast<int>(len_b(i));
                    M = std::max(M, m[l]);
                    n = std::max(n, nb[l]);
                    ragged |= nb[l] != nb[0];
                }
                const size_t gstride_ = ragged ? static_cast<size_t>(n) + 1 : 0;
                const AlignParams& P = params(i0);
                constexpr bool lin = GM == GapModel::Linear;
                const bool skip_fill = no_viterbi || !linear_fill_ok_(K);
                InterJobT<T> job{};
                job.a = a.data(); job.m = m.data(); job.b = b.data(); job.n = n; job.M = M;
                job.align_mode = (AM == AlignMode::Local) ? 1 : 0;
                if (ragged) job.nb = nb.data();
                auto& buf = w.buf;
                if (!skip_fill) {
                    const size_t sz = static_cast<size_t>(M + 1) * (n + 1) * W;
                    // Linear's one table rides in the H slot (adopt_interleaved reads buf.H).
                    if constexpr (lin) { if (buf.H.size() < sz) buf.H.resize(sz); }
                    else for (auto* v : {&buf.VM, &buf.VX, &buf.VY}) if (v->size() < sz) v->resize(sz);
                    // In T, as the pair's own fill rounds them (blkT_, the cast penalties).
                    const size_t nbk = static_cast<size_t>(P.matrix.size()) * P.matrix.size();
                    blkT.resize(nbk);
                    for (size_t k = 0; k < nbk; ++k) blkT[k] = static_cast<T>(P.matrix.data()[k]);
                    job.blk = blkT.data(); job.nalpha = P.matrix.size();
                    job.go_a = static_cast<T>(P.gap_open_a); job.ge_a = static_cast<T>(P.gap_extend_a);
                    job.go_b = static_cast<T>(P.gap_open_b); job.ge_b = static_cast<T>(P.gap_extend_b);
                    job.linear = lin ? 1 : 0;
                    if constexpr (lin) job.VM = buf.H.data();
                    else { job.VM = buf.VM.data(); job.VX = buf.VX.data(); job.VY = buf.VY.data(); }
                    job.best = best.data(); job.best_i = bi.data(); job.best_j = bj.data();
                    if constexpr (std::is_same_v<T, double>) K.inter_fill(job);
                    else                                     K.inter_fill_f(job);
                }
                // Soft lanes share the forward-backward too (one soft_impl, not "log",
                // one temperature — batch-wide here), when the weights fit.
                bool soft_group = K.inter_soft != nullptr && soft && soft_impl_ != SoftImpl::Log;
                const size_t nnk = nn_;
                if (soft_group) {
                    InterSoftJob sj{};
                    const bool fin = inter_soft_weights(P, soft_temp_, lin, ses, sj);
                    soft_group = fin;
                    if (fin) {
                        // The soft pass is double, K.inter_w lanes: a float32 group runs
                        // it in W / SW chunks.
                        const int SW = K.inter_w;
                        slogz.resize(W); scnt.resize(W * nnk); sgap.resize(W * 4); sok.assign(W, 0);
                        if (want_post) sgp.assign(static_cast<size_t>(M + 1) * W, 0);
                        sj.n = n;
                        sj.align_mode = job.align_mode;
                        for (int c = 0; c < W && static_cast<size_t>(c) < real; c += SW) {
                            int Mc = 0;
                            for (int l = c; l < c + SW; ++l) Mc = std::max(Mc, m[l]);
                            const size_t need = inter_soft_scratch(n, Mc, SW);
                            if (sscr.size() < need) sscr.resize(need);
                            siscr.resize(static_cast<size_t>(Mc + 1) * SW);
                            sj.a = a.data() + c; sj.m = m.data() + c; sj.b = b.data() + c; sj.M = Mc;
                            sj.nb = ragged ? nb.data() + c : nullptr;
                            sj.scratch = sscr.data(); sj.iscratch = siscr.data();
                            sj.logz = slogz.data() + c; sj.counts = scnt.data() + c * nnk;
                            sj.gaps = sgap.data() + c * 4; sj.ok = sok.data() + c;
                            sj.gpost = want_post ? sgp.data() + static_cast<size_t>(c) * (M + 1) : nullptr;
                            K.inter_soft(sj);
                        }
                    }
                }
                for (size_t l = 0; l < real; ++l) {
                    const size_t i = elig[s + l];
                    SoftLane lane{};
                    const bool use = soft_group && sok[l];
                    if (use) {
                        lane = {slogz[l], scnt.data() + l * nnk, sgap.data() + l * 4};
                        if (want_post) {
                            const int SW = K.inter_w, c = static_cast<int>(l) / SW * SW;
                            lane.gpost = sgp.data() + static_cast<size_t>(c) * (M + 1) + (l - c);
                            lane.gstride = SW;
                        }
                    }
                    if (skip_fill) run_soft_lane_(i, w, use ? &lane : nullptr, keep);
                    else run_interleaved_(i, w, W, static_cast<int>(l), best[l], bi[l], bj[l],
                                          use ? &lane : nullptr, gstride_, keep);
                }
            }
        };
        run_workers_(tasks, worker);
    }

    // The banded grouping: the plan's eligible pairs re-sorted, within each run of equal
    // (backend, len B), by where the guide runs (its column at the middle row of A, then
    // len A) — lanes whose bands overlap make the union each row computes narrow.  Speed
    // only.  Rebuilt when the plan is, or after a score_and_grad() (which recomputes
    // every guide).  Hard affine only: soft and linear pairs run their own banded path
    // in PAIR order (walking them permuted measured 5-10 % slower).
    struct BandGroups {
        size_t plan_gen = static_cast<size_t>(-1), full_calls = static_cast<size_t>(-1);
        int default_backend = -1000, hb_cutoff = -1;
        std::vector<size_t> elig, other;
        std::vector<std::pair<size_t, size_t>> groups;
    };
    BandGroups band_;
    size_t full_calls_ = 0;

    void ensure_band_groups_() {
        ensure_plan_();
        if (band_.plan_gen == plan_.generation && band_.full_calls == full_calls_ &&
            band_.default_backend == plan_.default_backend && band_.hb_cutoff == plan_.hb_cutoff)
            return;
        BandGroups Bg;
        Bg.plan_gen = plan_.generation; Bg.full_calls = full_calls_;
        Bg.default_backend = plan_.default_backend; Bg.hb_cutoff = plan_.hb_cutoff;
        Bg.other = plan_.other;
        const bool shared = GM == GapModel::Affine && grad_mode_ != GradMode::Soft;
        std::vector<size_t> E;
        if (shared) E = plan_.elig;
        else Bg.other.insert(Bg.other.end(), plan_.elig.begin(), plan_.elig.end());
        std::sort(Bg.other.begin(), Bg.other.end());
        Bg.elig = E;
        struct Key { int mid, len_a; size_t i; };
        std::vector<Key> keys;
        for (size_t s = 0; s < E.size();) {
            const int be = plan_.backend[E[s]];
            size_t e = s + 1;
            while (e < E.size() && plan_.backend[E[e]] == be && len_b(E[e]) == len_b(E[s])) ++e;
            keys.clear();
            for (size_t k = s; k < e; ++k) {
                const size_t i = E[k];
                const int la = static_cast<int>(len_a(i));
                keys.push_back({guide_ptr_(i)[la / 2], la, i});
            }
            std::sort(keys.begin(), keys.end(), [](const Key& x, const Key& y) {
                if (x.mid != y.mid) return x.mid < y.mid;
                if (x.len_a != y.len_a) return x.len_a < y.len_a;
                return x.i < y.i;
            });
            for (size_t k = s; k < e; ++k) Bg.elig[k] = keys[k - s].i;
            const int W = inter_w_(level_kernels(be));
            for (size_t g = s; g < e; g += W)
                Bg.groups.emplace_back(g, std::min(e, g + static_cast<size_t>(W)));
            s = e;
        }
        band_ = std::move(Bg);
    }

    // banded_grad() under fill = "interpair": each group a GuideBanded InterJob, every
    // lane around its own guide, its tables bit-identical to its own banded fill.
    // (SeqPairBatch::banded_grad_inter_.)
    void banded_grad_inter_(int bandwidth) {
        ensure_band_groups_();
        const auto& elig = band_.elig;
        const auto& other = band_.other;
        const auto& backend = plan_.backend;
        const auto& groups = band_.groups;
        const size_t G = groups.size(), tasks = G + other.size();
        std::atomic<size_t> idx{0};
        auto worker = [&]() {
            Work w(*this);
            std::vector<const unsigned char*> a, b;
            std::vector<int> m, bi, bj, blo, bhi, ulo, uhi, bri, brj;
            std::vector<T> best, blkT;
            while (true) {
                const size_t t = idx.fetch_add(1, std::memory_order_relaxed);
                if (t >= tasks) break;
                if (t >= G) { run_banded_(other[t - G], w, bandwidth, false); continue; }
                const auto [s, e] = groups[t];
                const size_t real = e - s;
                const size_t i0 = elig[s];
                bool shared = true;
                for (size_t l = 0; l < real && shared; ++l) shared = &params(elig[s + l]) == &params(i0);
                if (!shared) {
                    for (size_t l = 0; l < real; ++l) run_banded_(elig[s + l], w, bandwidth, false);
                    continue;
                }
                const LevelKernels& K = level_kernels(backend[i0]);
                const int W = inter_w_(K);
                int M = 0;
                for (size_t l = 0; l < real; ++l)
                    M = std::max(M, static_cast<int>(len_a(elig[s + l])));
                const int n = static_cast<int>(len_b(i0));
                a.assign(W, nullptr); b.assign(W, nullptr); m.assign(W, 0);
                bi.assign(W, 0); bj.assign(W, 0); best.assign(W, T(0));
                bri.assign(W, 0); brj.assign(W, 0);
                blo.assign(static_cast<size_t>(M + 1) * W, 1);
                bhi.assign(static_cast<size_t>(M + 1) * W, 0);
                ulo.assign(M + 1, n + 1); uhi.assign(M + 1, 0);
                for (int l = 0; l < W; ++l) {
                    // Spare lanes: the last pair's codes, no rows (m = 0), empty bands.
                    const size_t i = elig[s + std::min<size_t>(l, real - 1)];
                    a[l] = codes_a(i).data(); b[l] = codes_b(i).data();
                    if (static_cast<size_t>(l) >= real) continue;
                    resolve_guide_(i, w);
                    m[l] = static_cast<int>(len_a(i));
                    banded_lane_setup_(i, w, bandwidth, W, l, M, blo.data(), bhi.data(),
                                       ulo.data(), uhi.data(), bri[l], brj[l]);
                }
                const AlignParams& P = params(i0);
                auto& buf = w.buf;
                const size_t sz = static_cast<size_t>(M + 1) * (n + 1) * W;
                for (auto* v : {&buf.VM, &buf.VX, &buf.VY}) if (v->size() < sz) v->resize(sz);
                InterJobT<T> job{};
                job.a = a.data(); job.m = m.data(); job.b = b.data(); job.n = n; job.M = M;
                job.align_mode = (AM == AlignMode::Local) ? 1 : 0;
                const size_t nbk = static_cast<size_t>(P.matrix.size()) * P.matrix.size();
                blkT.resize(nbk);
                for (size_t k = 0; k < nbk; ++k) blkT[k] = static_cast<T>(P.matrix.data()[k]);
                job.blk = blkT.data(); job.nalpha = P.matrix.size();
                job.go_a = static_cast<T>(P.gap_open_a); job.ge_a = static_cast<T>(P.gap_extend_a);
                job.go_b = static_cast<T>(P.gap_open_b); job.ge_b = static_cast<T>(P.gap_extend_b);
                job.linear = 0;
                job.VM = buf.VM.data(); job.VX = buf.VX.data(); job.VY = buf.VY.data();
                job.best = best.data(); job.best_i = bi.data(); job.best_j = bj.data();
                job.blo = blo.data(); job.bhi = bhi.data();
                job.ulo = ulo.data(); job.uhi = uhi.data();
                job.bri = bri.data(); job.brj = brj.data();
                if constexpr (std::is_same_v<T, double>) K.inter_fill(job);
                else                                     K.inter_fill_f(job);
                for (size_t l = 0; l < real; ++l)
                    banded_interleaved_(elig[s + l], w, bandwidth, W, static_cast<int>(l),
                                        best[l], bi[l], bj[l]);
            }
        };
        if constexpr (GM == GapModel::Affine) run_workers_(tasks, worker);
        else banded_grad_dynamic_(bandwidth, false);
    }

    // ── Own-fill schedulers (SeqPairBatch's, unchanged in substance) ─────────

    void score_and_grad_dynamic_(bool keep) {
        const size_t N = size();
        std::atomic<size_t> idx{0};
        auto worker = [&]() {
            Work w(*this);
            for (size_t i; (i = idx.fetch_add(1, std::memory_order_relaxed)) < N; )
                run_full_(i, w, keep);
        };
        run_workers_(N, worker);
    }

    double length_weight_(double cells) const noexcept {
        if (long_cost_ratio <= 1.0 || weight_hi <= weight_lo) return 1.0;
        const double eff = std::sqrt(cells);
        const double t = std::clamp((eff - weight_lo) / (weight_hi - weight_lo), 0.0, 1.0);
        return 1.0 + (long_cost_ratio - 1.0) * t;
    }

    // Length-sorted, equal-work chunks, one per thread, plus a reserve of the smallest
    // tasks: bounds the DP memory high-water mark at sum-over-chunks rather than
    // n_threads x the global maximum (human proteome, 16 threads: 4.9 vs 30.6 GB).  See
    // the measurements recorded at SeqPairBatch's long_cost_ratio and reserve_frac.
    template<typename Fn>
    void run_sorted_phase_(const std::vector<double>& cost, int nthr, Fn&& task) {
        const size_t N = cost.size();
        std::vector<size_t> order(N);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return cost[a] < cost[b]; });
        const double total = std::accumulate(cost.begin(), cost.end(), 0.0);
        size_t r = 0;
        {
            const double want = total * std::clamp(reserve_frac, 0.0, 0.5);
            double acc = 0.0;
            while (r < N && acc < want && r + static_cast<size_t>(nthr) < N) acc += cost[order[r++]];
        }
        double chunk_total = 0.0;
        for (size_t i = r; i < N; ++i) chunk_total += cost[order[i]];
        std::vector<size_t> bound(static_cast<size_t>(nthr) + 1, r);
        {
            double acc = 0.0;
            size_t k = 1;
            for (size_t i = r; i < N && k < static_cast<size_t>(nthr); ++i) {
                acc += cost[order[i]];
                if (acc >= chunk_total * static_cast<double>(k) / static_cast<double>(nthr))
                    bound[k++] = i + 1;
            }
            while (k <= static_cast<size_t>(nthr)) bound[k++] = N;
        }
        bound[static_cast<size_t>(nthr)] = N;
        std::atomic<int>    next_worker{0};
        std::atomic<size_t> reserve_idx{0};
        using clk = std::chrono::steady_clock;
        const bool prof = profile;
        if (prof) profile_out.assign(static_cast<size_t>(nthr), PhaseProfile{});
        const auto t_start = clk::now();
        auto since = [](clk::time_point a, clk::time_point b) {
            return std::chrono::duration<double>(b - a).count();
        };
        auto worker = [&]() {
            const int k = next_worker.fetch_add(1, std::memory_order_relaxed);
            Work w(*this);
            PhaseProfile pp;
            const auto t0 = clk::now();
            if (k < nthr) {
                // Descending within the chunk: the buffer reaches its high-water mark on
                // the first task and never reallocates afterwards.
                for (size_t p = bound[static_cast<size_t>(k) + 1]; p > bound[static_cast<size_t>(k)]; --p) {
                    task(order[p - 1], w);
                    if (prof) { pp.chunk_cells += cost[order[p - 1]]; ++pp.chunk_tasks; }
                }
            }
            const auto t1 = clk::now();
            while (true) {   // finished early: drain the reserve, largest first
                const size_t s = reserve_idx.fetch_add(1, std::memory_order_relaxed);
                if (s >= r) break;
                task(order[r - 1 - s], w);
                if (prof) { pp.reserve_cells += cost[order[r - 1 - s]]; ++pp.reserve_tasks; }
            }
            // Chunks no worker claimed (a failed thread launch): never a short sum.
            for (int k2; (k2 = next_worker.fetch_add(1, std::memory_order_relaxed)) < nthr; ) {
                for (size_t p = bound[static_cast<size_t>(k2) + 1]; p > bound[static_cast<size_t>(k2)]; --p) {
                    task(order[p - 1], w);
                    if (prof) { pp.chunk_cells += cost[order[p - 1]]; ++pp.chunk_tasks; }
                }
            }
            const auto t2 = clk::now();
            if (prof && k >= 0 && static_cast<size_t>(k) < profile_out.size()) {
                pp.chunk_s = since(t0, t1); pp.reserve_s = since(t1, t2); pp.finish_s = since(t_start, t2);
                profile_out[static_cast<size_t>(k)] = pp;
            }
        };
        run_workers_(N, worker);
    }

    void score_and_grad_sorted_(bool keep) {
        const size_t N = size();
        const int nthr = std::min<int>(n_threads_, static_cast<int>(N));
        std::vector<double> cost(N);
        for (size_t i = 0; i < N; ++i) {
            const double cells = static_cast<double>(len_a(i) + 1) * static_cast<double>(len_b(i) + 1);
            cost[i] = cells * length_weight_(cells);
        }
        run_sorted_phase_(cost, nthr, [&](size_t i, Work& w) { run_full_(i, w, keep); });
    }

    void banded_grad_dynamic_(int bandwidth, bool keep) {
        const size_t N = size();
        std::atomic<size_t> idx{0};
        auto worker = [&]() {
            Work w(*this);
            for (size_t i; (i = idx.fetch_add(1, std::memory_order_relaxed)) < N; )
                run_banded_(i, w, bandwidth, keep);
        };
        run_workers_(N, worker);
    }

    // Longest-processing-time-first over a plain atomic counter: banded work is linear
    // in length and fits every thread at once, so the chunked scheduler would be
    // overhead (see SeqPairBatch::banded_grad_lpt_).
    void banded_grad_lpt_(int bandwidth, bool keep) {
        const size_t N = size();
        const double bw = 2.0 * static_cast<double>(bandwidth) + 1.0;
        std::vector<double> cost(N);
        for (size_t i = 0; i < N; ++i)
            cost[i] = static_cast<double>(len_a(i) + 1) * std::min(static_cast<double>(len_b(i) + 1), bw);
        std::vector<size_t> order(N);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return cost[a] > cost[b]; });
        std::atomic<size_t> idx{0};
        auto worker = [&]() {
            Work w(*this);
            for (size_t k; (k = idx.fetch_add(1, std::memory_order_relaxed)) < N; )
                run_banded_(order[k], w, bandwidth, keep);
        };
        run_workers_(N, worker);
    }

    template<typename Fn>
    void parallel_for_(size_t N, Fn&& fn) {
        std::atomic<size_t> idx{0};
        auto worker = [&]() {
            for (size_t i; (i = idx.fetch_add(1, std::memory_order_relaxed)) < N; ) fn(i);
        };
        run_workers_(N, worker);
    }
    template<typename Worker>
    void run_workers_(size_t N, Worker& worker) {
        if (N == 0) return;
        run_workers_guarded(std::min<int>(n_threads_, static_cast<int>(std::min<size_t>(N, 1u << 30))), worker);
    }
};
