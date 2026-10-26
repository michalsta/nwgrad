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
// classes; tests/cpp/test_batch_engine.cpp checks them against raw Aligners.
//
// The pairs are given once, at construction, under one AlignParams and one Viterbi
// backend; set_params() replaces the params for every pair.  Params are owned copies,
// so callers can update or release their originals without changing cached results.

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
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

// Every setting of a batch, its default and its validation, in one place. The engine
// owns the only copy. The setters here validate and normalize;
// BatchEngine's setters add what the engine must do on a change (soft_temperature
// invalidates soft results, n_threads resolves 0).
struct BatchSettings {
    int  n_threads = 0;                                  // 0: physical cores (resolved)
    TracebackMode traceback = TracebackMode::Default;    // fixed at construction
    int  hb_cutoff = 512;                                // Hirschberg base-case rows
    bool rowwise_full = false;                           // fill "rowwise" (own fills)
    bool inter_fill = true;                              // fill "interpair"
    SoftImpl soft_impl = SoftImpl::Scaled;
    double soft_temperature = 1.0;
    bool soft_guide_lazy = false, soft_guide_posterior = false;   // at most one set
    // Scheduling (own fills): see BatchEngine::run_sorted_phase_ / banded_grad_lpt_.
    bool   sorted_schedule = false;
    double reserve_frac = 0.0;
    double long_cost_ratio = 1.0;
    double weight_lo = 500.0;
    double weight_hi = 1700.0;
    bool   profile = false;
    // How a tracked batch stores its hard gradients: false, each pair's path (2 bits a
    // move, replayed on demand); true, its sparse field counts (faster to sum, ~10x the
    // memory).  See BatchEngine, "Tracked hard gradients".
    bool   track_counts = false;

    void set_hb_cutoff(int rows) {
        if (rows < 1) throw std::invalid_argument("nwgrad: hb_cutoff must be >= 1");
        hb_cutoff = rows;
    }
    void set_soft_temperature(double t) {
        if (!(t > 0.0 && t <= std::numeric_limits<double>::max()))
            throw std::invalid_argument("nwgrad: soft temperature must be finite and > 0, got " +
                                        std::to_string(t));
        soft_temperature = t;
    }
    void set_soft_guide(bool lazy, bool posterior) noexcept {
        soft_guide_lazy = lazy && !posterior;
        soft_guide_posterior = posterior;
    }
    void set_fill(bool rowwise, bool inter) noexcept { rowwise_full = rowwise; inter_fill = inter; }
    void set_n_threads(int n) {
        if (n < 0)
            throw std::invalid_argument("nwgrad: n_threads must be >= 0 (0: the default thread count), got " +
                                        std::to_string(n));
        n_threads = n > 0 ? n : default_thread_count();
    }
};

template<class T, GapModel GM, AlignMode AM>
class BatchEngine {
public:
    using DpBuffer = DpBufferT<T>;
    using FullAl   = Aligner<GM, AM, AlignBand::Full, T>;
    using BandAl   = Aligner<GM, AM, AlignBand::GuideBanded, T>;
    static constexpr GapModel  gap_model  = GM;
    static constexpr AlignMode align_mode = AM;

    // ── Construction ─────────────────────────────────────────────────────────

    // No pairs: an engine for streaming align_stream() only.
    BatchEngine(BatchSettings s, GradMode gd) : s_(std::move(s)), grad_mode_(gd) {
        if constexpr (nw_is_int<T>)
            if (gd == GradMode::Soft)
                throw std::invalid_argument(
                    std::string("nwgrad: grad_mode \"soft\" (forward-backward) needs a floating-point "
                                "scalar type; this batch's is ") + nw_type_name<T>());
        s_.set_n_threads(s_.n_threads);
        FullAl probe;   // what "auto" resolves to for this problem type
        probe.set_traceback(s_.traceback);
        tb_resolved_ = probe.traceback();
    }
    BatchEngine(int n_threads, GradMode gd, TracebackMode tb = TracebackMode::Default)
        : BatchEngine(BatchSettings{n_threads, tb}, gd) {}

    // The N pairs (seqs_a[i], seqs_b[i]) under a copy of `params` (the batch owns its
    // params: a later in-place change to the caller's object does not reach it —
    // set_params() does) and Viterbi backend `kernel`.  Sequences are validated and
    // encoded here, in parallel, straight into the batch's arrays; an out-of-alphabet
    // character throws, naming it and its position.
    //
    // Tracks (see aligner.hpp, "Tracks"): `tracks` — one per pair, each the parameter set
    // of every residue of A — and `gap_tracks` — one per pair, the set of every DP row
    // 0..m for gaps in A — are optional (null: all pairs default; an EMPTY entry: that
    // pair defaults).  Defaults are as Aligner::set_problem's.  The batch is TRACKED when
    // its params have more than one set or any track is given; every pair then carries
    // both tracks (zeros / derived where not given), and set_params() must keep K.  A
    // tracked batch stores a hard gradient as the pair's PATH (2 bits per step) or, with
    // track_counts, its sparse field counts — not as K dense sets — and sums it on demand.
    BatchEngine(BatchSettings s, GradMode gd, const std::vector<std::string_view>& seqs_a,
                const std::vector<std::string_view>& seqs_b, const AlignParams& params,
                int kernel = kBackendAuto, const PairTracks& tracks = {},
                const PairTracks& gap_tracks = {})
        : BatchEngine(std::move(s), gd) {
        if (seqs_a.size() != seqs_b.size())
            throw std::invalid_argument(
                "nwgrad: seqs_a and seqs_b need equal length (got " +
                std::to_string(seqs_a.size()) + " and " + std::to_string(seqs_b.size()) + ")");
        const size_t N = seqs_a.size();
        if (tracks && tracks.size() != N)
            throw std::invalid_argument("nwgrad: tracks needs one entry per pair (got " +
                                        std::to_string(tracks.size()) + " for " + std::to_string(N) + " pairs)");
        if (gap_tracks && gap_tracks.size() != N)
            throw std::invalid_argument("nwgrad: gap_tracks needs one entry per pair (got " +
                                        std::to_string(gap_tracks.size()) + " for " + std::to_string(N) + " pairs)");
        if (N == 0) return;
        if (N > std::numeric_limits<uint32_t>::max())
            throw std::length_error("nwgrad: batch too large");
        const int K = params.num_sets();
        check_tracks_(tracks, [&](size_t k) { return seqs_a[k].size(); }, K, "track");
        check_tracks_(gap_tracks, [&](size_t k) { return seqs_a[k].size() + 1; }, K, "gap_track");

        const Alphabet& alpha = params.matrix.alphabet();
        // Offsets first (serial: a prefix sum), then encode in parallel into place.
        oa_.reserve(N + 1); ob_.reserve(N + 1);
        for (size_t i = 0; i < N; ++i) {
            oa_.push_back(oa_.back() + seqs_a[i].size());
            ob_.push_back(ob_.back() + seqs_b[i].size());
        }
        ca_.resize(oa_.back());
        cb_.resize(ob_.back());
        parallel_for_(N, [&](size_t k) {
            alpha.encode_into(seqs_a[k], ca_.data() + oa_[k]);
            alpha.encode_into(seqs_b[k], cb_.data() + ob_[k]);
        });
        alpha_ = &alpha;
        nn_ = static_cast<size_t>(alpha.size()) * static_cast<size_t>(alpha.size());
        nsets_ = K;
        tracked_ = K > 1 || tracks || gap_tracks;
        params_ = own_params_(params);
        kernel_ = kernel;
        score_.assign(N, 0.0);
        flags_.assign(N, 0);
        guide_.assign(oa_.back() + N, 0);   // pair i: m+1 entries at oa_[i] + i
        if (tracked_) {
            tk_.assign(oa_.back(), 0);         // pair i: m entries at oa_[i]
            gk_.assign(oa_.back() + N, 0);     // pair i: m+1 entries at oa_[i] + i
            parallel_for_(N, [&](size_t i) {
                const size_t m = len_a(i);
                uint16_t* t = tk_.data() + oa_[i];
                uint16_t* g = gk_.data() + oa_[i] + i;
                const auto tr = tracks ? tracks[i] : std::span<const uint16_t>();
                const auto gt = gap_tracks ? gap_tracks[i] : std::span<const uint16_t>();
                if (!tr.empty()) std::copy(tr.begin(), tr.end(), t);
                if (!gt.empty()) {
                    std::copy(gt.begin(), gt.end(), g);
                } else {
                    g[0] = m ? t[0] : 0;
                    for (size_t r = 1; r <= m; ++r) g[r] = t[r - 1];
                }
            });
            size_grad_store_();
        } else if (grad_mode_ != GradMode::None) {
            grad_.assign(N * gstride(), 0.0);
        }
    }

    BatchEngine(const BatchEngine&)            = delete;
    BatchEngine& operator=(const BatchEngine&) = delete;
    BatchEngine(BatchEngine&&)                 = default;
    BatchEngine& operator=(BatchEngine&&)      = default;

    // ── Settings ─────────────────────────────────────────────────────────────
    // Read through settings(); change through the setters below.  The ones that change a
    // result invalidate what they change; the rest are speed knobs (bit-identical
    // results).  hb_cutoff applies to every pair from the next alignment on.

    const BatchSettings& settings() const noexcept { return s_; }
    GradMode grad_mode() const noexcept { return grad_mode_; }
    int n_threads() const noexcept { return s_.n_threads; }
    // The traceback as constructed ("auto" stays Default) and as this type resolves it.
    TracebackMode traceback() const noexcept { return s_.traceback; }
    TracebackMode traceback_resolved() const noexcept { return tb_resolved_; }

    void set_n_threads(int n) { s_.set_n_threads(n); }
    void set_hb_cutoff(int rows) { s_.set_hb_cutoff(rows); }
    void set_fill(bool rowwise, bool inter) noexcept { s_.set_fill(rowwise, inter); }
    // soft_impl changes results within tolerance only and invalidates nothing (as
    // SeqPair::set_soft_impl); a new temperature changes the soft score.
    void set_soft_impl(SoftImpl v) noexcept { s_.soft_impl = v; }
    void set_soft_temperature(double t) {
        s_.set_soft_temperature(t);
        if (grad_mode_ == GradMode::Soft)
            for (auto& f : flags_) f &= static_cast<uint8_t>(~(kScore | kGrad));
    }
    void set_soft_guide(bool lazy, bool posterior) noexcept { s_.set_soft_guide(lazy, posterior); }
    void set_sorted_schedule(bool v) noexcept { s_.sorted_schedule = v; }
    void set_reserve_frac(double v) noexcept { s_.reserve_frac = v; }
    void set_long_cost_ratio(double v) noexcept { s_.long_cost_ratio = v; }
    void set_weight_lo(double v) noexcept { s_.weight_lo = v; }
    void set_weight_hi(double v) noexcept { s_.weight_hi = v; }
    void set_profile(bool v) noexcept { s_.profile = v; }
    // Paths or counts for a tracked batch's hard gradients: a change drops the stored ones
    // (scores and guides stay; the next score_and_grad / banded_grad stores them anew).
    void set_track_counts(bool v) {
        if (v == s_.track_counts) return;
        s_.track_counts = v;
        if (!tracked_ || grad_mode_ != GradMode::Hard) return;
        for (auto& f : flags_) f &= static_cast<uint8_t>(~kGrad);
        reset_hard_store_();
        size_grad_store_();
    }

    struct PhaseProfile {
        double chunk_s = 0.0, reserve_s = 0.0;
        double chunk_cells = 0.0, reserve_cells = 0.0;
        long   chunk_tasks = 0, reserve_tasks = 0;
        double finish_s = 0.0;
    };
    std::vector<PhaseProfile> profile_out;   // per worker, after a profiled sorted run

    // Whether the batch is tracked, and its parameter set count K (0 when empty).
    bool tracked() const noexcept { return tracked_; }
    int num_sets() const noexcept { return nsets_; }
    // Pair i's track (len_a entries) and gap track (len_a + 1), tracked batches only.
    std::span<const uint16_t> track(size_t i) const noexcept {
        return tracked_ ? std::span<const uint16_t>(tk_.data() + oa_[i], len_a(i)) : std::span<const uint16_t>();
    }
    std::span<const uint16_t> gap_track(size_t i) const noexcept {
        return tracked_ ? std::span<const uint16_t>(gk_.data() + oa_[i] + i, len_a(i) + 1)
                        : std::span<const uint16_t>();
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
    // The batch's params (non-empty batches) and Viterbi backend.
    const AlignParams& params() const noexcept { return params_->p; }
    int kernel() const noexcept { return kernel_; }

    bool path_valid(size_t i)  const noexcept { return flags_[i] & kPath; }
    bool score_valid(size_t i) const noexcept { return flags_[i] & kScore; }
    bool grad_valid(size_t i)  const noexcept { return flags_[i] & kGrad; }
    // A stored alignment path (aligned(), coordinates()) — kept only on request.
    bool path_stored(size_t i) const noexcept { return flags_[i] & kStored; }
    bool guide_pending(size_t i) const noexcept { return flags_[i] & kPending; }

    double score(size_t i) const {
        if (!score_valid(i))
            throw std::logic_error(
                "nwgrad: score not valid; compute an alignment first");
        return score_[i];
    }
    // Pair i's gradient into `out` (same alphabet; K sets when tracked).
    void grad(size_t i, AlignParams& out) const {
        if (!grad_valid(i))
            throw std::logic_error("nwgrad: gradient not computed; call score_and_grad() or banded_grad() first");
        if (tracked_) {
            std::vector<double> flat(kstride_(), 0.0);
            pair_flat_grad_(i, flat.data());
            flat_to_params_(flat.data(), out);
            return;
        }
        const double* g = grad_.data() + i * gstride();
        std::memcpy(out.matrix.data(), g, nn_ * sizeof(double));
        out.gap_open_a = g[nn_]; out.gap_extend_a = g[nn_ + 1];
        out.gap_open_b = g[nn_ + 2]; out.gap_extend_b = g[nn_ + 3];
    }
    AlignParams grad(size_t i) const { AlignParams g(alphabet(), nsets_); grad(i, g); return g; }

    // The cached guide (column after each row of A), resolving a pending lazy guide.
    std::vector<int> guide_j(size_t i) {
        if (!path_valid(i))
            throw std::logic_error("nwgrad: alignment not computed; call align() first");
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

    // Replace the params with a copy of `params` (same alphabet, same K).  Clears every
    // cached score, gradient and stored path; guides stay (banded_grad() bands around them).
    void set_params(const AlignParams& params) {
        if (empty()) return;
        if (&params.matrix.alphabet() != alpha_)
            throw std::invalid_argument(
                "nwgrad: set_params() cannot change the alphabet (\"" + alpha_->symbols() +
                "\" -> \"" + params.matrix.alphabet().symbols() + "\"); construct new pairs instead");
        check_sets_(params);
        params_ = own_params_(params);
        for (auto& f : flags_) f &= static_cast<uint8_t>(~(kScore | kGrad | kStored));
    }

    // Full DP on every pair: score, guide and (unless grad_mode None) gradient, cached
    // per pair.  keep_paths: also store each pair's alignment path (aligned(),
    // coordinates()); it forces the Viterbi a lazy or posterior soft guide would skip.
    // Returns the sum of scores, in pair order (bit-reproducible whatever the schedule).
    //
    double score_and_grad(bool keep_paths = false) {
        const size_t N = size();
        if (N == 0) return 0.0;
        prepare_paths_(keep_paths);
        ++full_calls_;   // guides may have moved: the banded grouping is stale
        tc_begin_full_();
        if (grad_mode_ == GradMode::None && !keep_paths) score_only_();
        else if (s_.inter_fill)      score_and_grad_inter_(keep_paths);
        else if (s_.sorted_schedule) score_and_grad_sorted_(keep_paths);
        else                      score_and_grad_dynamic_(keep_paths);
        return sum_scores_();
    }

    // Banded re-align + gradient around every pair's cached guide (the training-loop
    // step after set_params()).  Throws if any pair has no guide yet.
    double banded_grad(int bandwidth, bool keep_paths = false) {
        const size_t N = size();
        if (N == 0) return 0.0;
        if (bandwidth <= 0)
            throw std::invalid_argument(
                "nwgrad: banded_grad() needs bandwidth > 0 (got " + std::to_string(bandwidth) +
                "); use score_and_grad() for full DP");
        for (size_t i = 0; i < N; ++i)
            if (!path_valid(i))
                throw std::logic_error(
                    "nwgrad: banded_grad() needs a guide path; run score_and_grad() first");
        prepare_paths_(keep_paths);
        tc_begin_full_();
        if (s_.inter_fill && !keep_paths) banded_grad_inter_(bandwidth);
        else if (s_.sorted_schedule)      banded_grad_lpt_(bandwidth, keep_paths);
        else                           banded_grad_dynamic_(bandwidth, keep_paths);
        return sum_scores_();
    }

    // Free the stored paths.  Scores, published gradients and guides stay.
    // Both also free the single-pair DP buffer (solo_), which is only a cache; drop_paths()
    // the workers' pooled buffers too.
    void drop_paths() noexcept {
        std::vector<std::vector<uint32_t>>().swap(path_);
        std::vector<std::array<int, 2>>().swap(path_end_);
        for (auto& f : flags_) f &= static_cast<uint8_t>(~kStored);
        solo_.reset();
        drop_pool_();
    }
    void drop_path(size_t i) noexcept {
        if (i < path_.size()) { std::vector<uint32_t>().swap(path_[i]); }
        flags_[i] &= static_cast<uint8_t>(~kStored);
        solo_.reset();
    }

    // One pair, on the calling thread (SeqPair's own operations).
    void score_and_grad_one(size_t i, bool keep_path) {
        prepare_paths_(keep_path, /*all=*/false);
        run_full_(i, solo_work_(), keep_path);
    }
    void banded_one(size_t i, int bandwidth, bool keep_path) {
        if (!path_valid(i))
            throw std::logic_error("nwgrad: call align() or score_and_grad() before banded_align()");
        prepare_paths_(keep_path, /*all=*/false);
        run_banded_(i, solo_work_(), bandwidth, keep_path);
    }
    // The cached scores, in pair order.  Throws if any pair has none.
    std::vector<double> scores() const {
        for (size_t i = 0; i < size(); ++i)
            if (!score_valid(i))
                throw std::logic_error(
                    "nwgrad: score not valid; compute an alignment first");
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
    AlignParams compute_grad() const {
        const Alphabet& alpha = alphabet();   // throws if empty
        check_all_grads_();
        return blocked_sum_(alpha, nullptr);
    }
    // The cached gradients as arrays: matrices[i*n*n ..] (row-major, alphabet order) and
    // gaps[i*4 ..] (gap_open_a, gap_extend_a, gap_open_b, gap_extend_b).  Tracked: K sets
    // per pair, matrices[(i*K + k)*n*n ..] and gaps[(i*K + k)*4 ..] — zeros included.
    void grads_into(double* matrices, double* gaps) const {
        alphabet();
        check_all_grads_();
        if (tracked_) {
            const size_t K = static_cast<size_t>(nsets_);
            parallel_for_c_(size(), [&](size_t i) {
                double* mi = matrices + i * K * nn_;
                double* gi = gaps + i * K * 4;
                std::fill(mi, mi + K * nn_, 0.0);
                std::fill(gi, gi + K * 4, 0.0);
                Touched& t = touched_tls_();
                pair_grad_(i, t);
                for (size_t q = 0; q < t.n; ++q) {
                    const size_t k = t.keys[q];
                    if (k < K * nn_) mi[k] = t.vals[q]; else gi[k - K * nn_] = t.vals[q];
                }
            });
            return;
        }
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
                             int kernel = kBackendAuto, const StreamTracks& st = {}) const {
        const size_t N = problems.size();
        BatchResult result(params);
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
        // Tracks: checked here, on the caller's thread, naming the problem.
        const int K = params.num_sets();
        if (st.tracks && st.tracks.size() != N)
            throw std::invalid_argument("nwgrad: align() needs one track per problem");
        if (st.gap_tracks && st.gap_tracks.size() != N)
            throw std::invalid_argument("nwgrad: align() needs one gap_track per problem");
        check_tracks_(st.tracks, [&](size_t i) { return problems[i].seq_a.size(); }, K, "track");
        check_tracks_(st.gap_tracks, [&](size_t i) { return problems[i].seq_a.size() + 1; }, K, "gap_track");
        if (s_.inter_fill) {
            const int be = stream_inter_backend_(params, kernel, banded);
            if (be >= 0) {
                if (banded) stream_inter_<AlignBand::GuideBanded>(problems, params, band, kernel, result, be, st);
                else        stream_inter_<AlignBand::Full>(problems, params, band, kernel, result, be, st);
                return result;
            }
        }
        std::atomic<size_t> work_idx{0};
        std::mutex grad_mutex;
        auto worker = [&]() {
            AlignParams local_grad = AlignParams::zeros_like(params);
            if (banded) stream_loop_<AlignBand::GuideBanded>(problems, params, band, kernel, work_idx, result.scores, local_grad, st);
            else        stream_loop_<AlignBand::Full>(problems, params, band, kernel, work_idx, result.scores, local_grad, st);
            if (grad_mode_ != GradMode::None) {
                std::lock_guard<std::mutex> lock(grad_mutex);
                result.grad += local_grad;
            }
        };
        run_workers_guarded(std::min<int>(s_.n_threads, static_cast<int>(std::min<size_t>(N, 1u << 30))), worker);
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
        al.set_soft_impl(s_.soft_impl);
        al.set_soft_temperature(s_.soft_temperature);
        if constexpr (AB == AlignBand::Full) {
            al.set_traceback(s_.traceback);
            al.set_hb_cutoff(s_.hb_cutoff);
            al.set_rowwise_full(s_.rowwise_full);
        }
    }
    template<AlignBand AB>
    void stream_loop_(const std::vector<ProblemInstance>& problems, const AlignParams& params,
                      int band, int kernel, std::atomic<size_t>& work_idx,
                      std::vector<double>& scores, AlignParams& local_grad, const StreamTracks& st) const {
        Aligner<GM, AM, AB, T> al;
        configure_stream_<AB>(al, kernel);
        DpBuffer buf;
        const Alphabet& alpha = params.matrix.alphabet();
        std::vector<uint8_t> a_enc, b_enc;
        const size_t N = problems.size();
        for (size_t idx; (idx = work_idx.fetch_add(1, std::memory_order_relaxed)) < N; )
            stream_one_(al, problems, idx, alpha, params, band, buf, a_enc, b_enc, scores, local_grad, st);
    }
    // A problem's effective tracks: as given, or the defaults (all zeros; derived).
    static void effective_tracks_(std::span<const uint16_t> tk, std::span<const uint16_t> gk, size_t m,
                                  std::vector<uint16_t>& t, std::vector<uint16_t>& g) {
        if (tk.empty()) t.assign(m, 0); else t.assign(tk.begin(), tk.end());
        if (gk.empty()) {
            g.resize(m + 1);
            g[0] = m ? t[0] : 0;
            for (size_t r = 1; r <= m; ++r) g[r] = t[r - 1];
        } else {
            g.assign(gk.begin(), gk.end());
        }
    }
    // Tracked soft lane rows (InterSoftJob::rs / rw) of one problem: its stacked rows and
    // the exp'd penalties of each row at temperature T.  False when a weight overflows.
    static bool soft_rows_of_(std::span<const uint8_t> ca, std::span<const uint16_t> tk,
                              std::span<const uint16_t> gk, const AlignParams& P, double temp,
                              std::vector<uint32_t>& rs, std::vector<double>& rw) {
        constexpr bool lin = GM == GapModel::Linear;
        const double it = 1.0 / temp, big = std::numeric_limits<double>::max();
        const uint32_t na = static_cast<uint32_t>(P.matrix.size());
        const size_t m = ca.size();
        rs.resize(m);
        for (size_t r = 0; r < m; ++r) rs[r] = static_cast<uint32_t>(tk[r]) * na + ca[r];
        rw.assign(4 * (m + 1), 0.0);
        bool ok = true;
        for (size_t r = 0; r <= m; ++r) {
            const int g = gk[r];
            rw[4 * r + 1] = std::exp(-P.gap(g, AlignParams::kExtendA) * it);
            if (!lin) rw[4 * r] = std::exp(-(P.gap(g, AlignParams::kOpenA) + P.gap(g, AlignParams::kExtendA)) * it);
            if (r > 0) {
                const int t = tk[r - 1];
                rw[4 * r + 3] = std::exp(-P.gap(t, AlignParams::kExtendB) * it);
                if (!lin) rw[4 * r + 2] = std::exp(-(P.gap(t, AlignParams::kOpenB) + P.gap(t, AlignParams::kExtendB)) * it);
            }
            for (int f = 0; f < 4; ++f) ok &= rw[4 * r + static_cast<size_t>(f)] <= big;
        }
        return ok;
    }
    // A per-row soft record (counts m x n, gaps 4 x (m+1)) added into a K-set gradient.
    static void add_rows_(AlignParams& g, std::span<const uint8_t> ca, std::span<const uint16_t> tk,
                          std::span<const uint16_t> gk, const double* cnt, const double* gaps) {
        const size_t na = static_cast<size_t>(g.matrix.size()), m = ca.size();
        double* blk = g.matrix.data();
        for (size_t r = 0; r < m; ++r) {
            double* d = blk + (static_cast<size_t>(tk[r]) * na + ca[r]) * na;
            for (size_t c = 0; c < na; ++c) d[c] += cnt[r * na + c];
        }
        for (size_t r = 0; r <= m; ++r) {
            g.gap_ref(gk[r], AlignParams::kOpenA) += gaps[4 * r];
            g.gap_ref(gk[r], AlignParams::kExtendA) += gaps[4 * r + 1];
            if (r > 0) {
                g.gap_ref(tk[r - 1], AlignParams::kOpenB) += gaps[4 * r + 2];
                g.gap_ref(tk[r - 1], AlignParams::kExtendB) += gaps[4 * r + 3];
            }
        }
    }
    // One streaming problem on its own path: encode, set up, fill, score, gradient.
    template<class Al>
    void stream_one_(Al& al, const std::vector<ProblemInstance>& problems, size_t i,
                     const Alphabet& alpha, const AlignParams& params, int band, DpBuffer& buf,
                     std::vector<uint8_t>& a_enc, std::vector<uint8_t>& b_enc,
                     std::vector<double>& scores, AlignParams& local_grad, const StreamTracks& st) const {
        const auto& p = problems[i];
        stream_encode_(alpha, p.seq_a, a_enc, i, 'a');
        stream_encode_(alpha, p.seq_b, b_enc, i, 'b');
        al.set_problem(a_enc, b_enc, params, band, p.guide_j, st.track(i), st.gap_track(i));
        if (grad_mode_ == GradMode::Hard) {
            al.compute_viterbi(buf); scores[i] = al.score(); al.hard_grad(buf, local_grad);
        } else if (grad_mode_ == GradMode::Soft) {
            al.compute_forward_back(buf); scores[i] = al.log_z(); al.soft_grad(buf, local_grad);
        } else {
            scores[i] = al.compute_score(buf);   // no table, no traceback (exact fill's score)
        }
    }
    // The level a shared pass would run on, or -1 (every problem its own path): the
    // scalar backend, soft_impl "log" or a banded soft call, linear Global Viterbi below
    // 4 lanes or over 8 letters, linear banded.  "Banded" includes band == 0 WITH guides
    // (GuideBanded at width 0): the shared soft pass is Full-only, and it used to be
    // admitted there — ignoring every guide and returning the unbanded log Z, so
    // fill="interpair" and fill="striped" disagreed (inherited from BatchAligner).
    int stream_inter_backend_(const AlignParams& params, int kernel, bool banded) const {
        const int backend = (kernel == kBackendAuto) ? global_default_backend() : kernel;
        if (backend < 0) return -1;
        const LevelKernels& K = level_kernels(backend);
        if (grad_mode_ == GradMode::Soft)
            return (K.inter_soft && K.d.inter_w > 0 && s_.soft_impl != SoftImpl::Log && !banded) ? backend : -1;
        if (inter_w_(K) <= 0) return -1;
        if (GM == GapModel::Linear &&
            (banded || (AM == AlignMode::Global && (inter_w_(K) < 4 || params.matrix.size() > 8))))
            return -1;
        return backend;
    }
    // Problems grouped W at a time, sorted by len B then len A (counting sorts: each call
    // brings new problems, nothing is cached).  The rest run their own path in the same
    // pass: empty sequences, and Full pairs the traceback would split (Hirschberg past
    // the cutoff).  A guided band-0 call is GuideBanded and stays so.
    template<AlignBand AB>
    void stream_inter_(const std::vector<ProblemInstance>& problems, const AlignParams& params,
                       int band, int kernel, BatchResult& result, int backend, const StreamTracks& st) const {
        const size_t N = problems.size();
        const LevelKernels& K = level_kernels(backend);
        const bool soft = grad_mode_ == GradMode::Soft;
        const int W = soft ? K.d.inter_w : inter_w_(K);
        const bool hb_case = !soft && AB == AlignBand::Full && is_hirschberg(tb_resolved_);
        std::vector<size_t> other, elig;
        size_t maxa = 0, maxb = 0;
        for (size_t i = 0; i < N; ++i) {
            const size_t la = problems[i].seq_a.size(), lb = problems[i].seq_b.size();
            if (la == 0 || lb == 0 || (hb_case && hb_splits(tb_resolved_, s_.traceback == TracebackMode::Default ? FullAl::kAutoPointerCells : 0, s_.hb_cutoff, la, lb)) ||
                !inter_pair_fits(la, lb, GM == GapModel::Affine, soft, inter_w_(K), sizeof(T), K.d.inter_w)) {
                other.push_back(i); continue;
            }
            // the integer linear gather gate of inter_eligible_ (hard fills only; score-only
            // groups are gated by inter_score_ok in the loop)
            if constexpr (GM == GapModel::Linear)
                if (grad_mode_ == GradMode::Hard && params.matrix.size() > 8 &&
                    (la + 1) * (lb + 1) > inter_linear_gather_cap<T, AM == AlignMode::Global>()) {
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
        // Tracked lanes (per-lane rows into the kernels): params with sets, or any track.
        const bool trk = params.num_sets() > 1 || st.any();
        std::atomic<size_t> idx{0};
        std::mutex grad_mutex;
        auto worker = [&]() {
            AlignParams local_grad = AlignParams::zeros_like(params);
            stream_inter_loop_<AB>(problems, params, band, kernel, elig, groups, other, idx,
                                   result.scores, local_grad, K, W, trk, st);
            if (grad_mode_ != GradMode::None) {
                std::lock_guard<std::mutex> lock(grad_mutex);
                result.grad += local_grad;
            }
        };
        const size_t tasks = groups.size() + other.size();
        run_workers_guarded(std::min<int>(s_.n_threads, static_cast<int>(std::max<size_t>(tasks, 1))), worker);
    }
    template<AlignBand AB>
    void stream_inter_loop_(const std::vector<ProblemInstance>& problems, const AlignParams& params,
                            int band, int kernel, const std::vector<size_t>& elig,
                            const std::vector<std::pair<size_t, size_t>>& groups,
                            const std::vector<size_t>& other, std::atomic<size_t>& idx,
                            std::vector<double>& scores, AlignParams& local_grad,
                            const LevelKernels& K, int W, bool trk, const StreamTracks& st) const {
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
        const bool soft_ok = soft && inter_soft_weights(params, s_.soft_temperature, lin, es, sj);
        const size_t nn = static_cast<size_t>(params.matrix.size()) * params.matrix.size();
        const size_t na = static_cast<size_t>(params.matrix.size());
        if (!soft) cast_block_(params, blkT);   // params are fixed for the call: once per worker
        // Tracked lanes: each lane's effective tracks, its TrackJob (hard / none) or soft rows.
        std::vector<std::vector<uint16_t>> lt(W), lg(W);
        std::vector<TrackJob<T>> tj(W);
        std::vector<std::vector<uint32_t>> lrs(W);
        std::vector<std::vector<double>> lrw(W);
        std::vector<const uint32_t*> lrsp(W);
        std::vector<const double*> lrwp(W);
        auto own = [&](size_t i) {
            stream_one_(al[0], problems, i, alpha, params, band, buf, ae[0], be[0], scores, local_grad, st);
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
                    if (trk) effective_tracks_(st.track(i), st.gap_track(i), ae[l].size(), lt[l], lg[l]);
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
                if (trk) {
                    bool fin = true;
                    for (int l = 0; l < W; ++l) {
                        const size_t lr = std::min<size_t>(l, real - 1);
                        if (static_cast<size_t>(l) == lr)
                            fin &= soft_rows_of_(ae[l], lt[l], lg[l], params, s_.soft_temperature, lrs[l], lrw[l]);
                        lrsp[l] = lrs[lr].data(); lrwp[l] = lrw[lr].data();
                    }
                    if (!fin) { for (size_t k = s; k < e; ++k) own(elig[k]); continue; }
                    sj.rs = lrsp.data(); sj.rw = lrwp.data();
                }
                const size_t ccnt = trk ? static_cast<size_t>(M) * na : nn;
                const size_t cgap = trk ? 4 * (static_cast<size_t>(M) + 1) : 4;
                const size_t need = inter_soft_scratch(n, M, W);
                if (sscr.size() < need) sscr.resize(need);
                siscr.resize(static_cast<size_t>(M + 1) * W);
                slogz.resize(W); scnt.resize(W * ccnt); sgap.resize(W * cgap); sok.resize(W);
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
                    scores[i] = slogz[l] * s_.soft_temperature;
                    if (trk) {
                        add_rows_(local_grad, ae[l], lt[l], lg[l], scnt.data() + l * ccnt, sgap.data() + l * cgap);
                        continue;
                    }
                    double* g = local_grad.matrix.data();
                    for (size_t k = 0; k < nn; ++k) g[k] += scnt[l * nn + k];
                    local_grad.gap_open_a += sgap[l * 4 + 0]; local_grad.gap_extend_a += sgap[l * 4 + 1];
                    local_grad.gap_open_b += sgap[l * 4 + 2]; local_grad.gap_extend_b += sgap[l * 4 + 3];
                }
                continue;
            }
            if constexpr (AB == AlignBand::Full) {
                if (grad_mode_ == GradMode::None && has_inter_score_(K)) {
                    if (!inter_score_ok(na, M, n)) {   // the gather: short pairs only
                        for (size_t l = 0; l < real; ++l) own(elig[s + l]);
                        continue;
                    }
                    // Score only: one row per table, no adopt / traceback.
                    InterJobT<T> job{};
                    job.a = a.data(); job.m = m.data(); job.b = b.data(); job.n = n; job.M = M;
                    job.align_mode = (AM == AlignMode::Local) ? 1 : 0;
                    if (ragged) job.nb = nb.data();
                    set_inter_params_(job, params, blkT);
                    if (trk) {
                        for (int l = 0; l < W; ++l) {
                            const size_t lr = std::min<size_t>(l, real - 1);
                            if (static_cast<size_t>(l) == lr) {
                                al[l].set_problem(ae[l], be[l], params, band, {}, lt[l], lg[l]);
                                al[l].track_job(tj[l]);
                            } else {
                                tj[l] = tj[lr];
                            }
                        }
                        job.tr = tj.data();
                    }
                    wire_inter_tables_(job, buf, static_cast<size_t>(n + 1) * W);
                    job.best = best.data();
                    run_inter_score_(K, job);
                    for (size_t l = 0; l < real; ++l) scores[elig[s + l]] = static_cast<double>(best[l]);
                    continue;
                }
            }
            if constexpr (AB == AlignBand::GuideBanded) {
                blo.assign(static_cast<size_t>(M + 1) * W, 1);
                bhi.assign(static_cast<size_t>(M + 1) * W, 0);
                ulo.assign(M + 1, n + 1); uhi.assign(M + 1, 0);
                std::fill(bri.begin(), bri.end(), 0); std::fill(brj.begin(), brj.end(), 0);
            }
            for (size_t l = 0; l < real; ++l) {
                if (trk) {
                    al[l].set_problem(ae[l], be[l], params, band, problems[elig[s + l]].guide_j, lt[l], lg[l]);
                    al[l].track_job(tj[l]);
                } else {
                    al[l].set_problem(ae[l], be[l], params, band, problems[elig[s + l]].guide_j);
                }
                if constexpr (AB == AlignBand::GuideBanded)
                    al[l].banded_lane_rows(W, static_cast<int>(l), M, blo.data(), bhi.data(),
                                           ulo.data(), uhi.data(), bri[l], brj[l]);
            }
            InterJobT<T> job{};
            job.a = a.data(); job.m = m.data(); job.b = b.data(); job.n = n; job.M = M;
            job.align_mode = (AM == AlignMode::Local) ? 1 : 0;
            if (ragged) job.nb = nb.data();
            if (trk) {
                for (size_t l = real; l < static_cast<size_t>(W); ++l) tj[l] = tj[real - 1];
                job.tr = tj.data();
            }
            set_inter_params_(job, params, blkT);
            wire_inter_tables_(job, buf, static_cast<size_t>(M + 1) * (n + 1) * W);
            job.best = best.data(); job.best_i = bi.data(); job.best_j = bj.data();
            if constexpr (AB == AlignBand::GuideBanded) {
                job.blo = blo.data(); job.bhi = bhi.data();
                job.ulo = ulo.data(); job.uhi = uhi.data();
                job.bri = bri.data(); job.brj = brj.data();
            }
            run_inter_fill_(K, job);
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
    };
    // The batch's own copy of its params, with what a tracked batch prepares from it
    // once: its gap fields cast to T, 4 per set (setup_tracks_ reads them), and (float32,
    // K > 1) its whole matrix stack cast to T (the Aligner would otherwise cast K matrices
    // per pair).  Shared: the pending guides' params (gparams_) may be the current ones.
    // An integer T prepares every params ONCE here (nw_int_prepare: checked casts of
    // the whole stack and every set's gaps, and step_max), tracked or not, so a
    // non-integral parameter raises at construction / set_params() and no pair repeats
    // the checks.
    struct OwnedParams {
        AlignParams p;
        std::vector<T> stack, gaps;
        T step_max = T(-1);   // integer T: the largest step (nw_int_prepare); -1 = not prepared
        const T* stack_or_null() const noexcept { return stack.empty() ? nullptr : stack.data(); }
    };
    std::shared_ptr<const OwnedParams> own_params_(const AlignParams& p) const {
        auto o = std::make_shared<OwnedParams>(OwnedParams{p, {}, {}, T(-1)});
        if constexpr (nw_is_int<T>) {
            nw_int_prepare<T>(p, p.num_sets(), o->stack, o->gaps, o->step_max);
            return o;
        }
        if (!tracked_) return o;   // untracked (and standalone SeqPair): no allocation
        const std::vector<double> all = p.all_gaps();
        o->gaps.assign(all.begin(), all.end());
        if constexpr (!std::is_same_v<T, double>) {
            if (p.num_sets() > 1) {
                const size_t nk = static_cast<size_t>(p.matrix.size()) * p.matrix.size() * p.num_sets();
                o->stack.assign(p.matrix.data(), p.matrix.data() + nk);
            }
        }
        return o;
    }

    BatchSettings s_;
    GradMode grad_mode_;
    TracebackMode tb_resolved_ = TracebackMode::Pointers;

    const Alphabet* alpha_ = nullptr;
    size_t nn_ = 0;
    std::vector<uint8_t> ca_, cb_;          // all pairs' codes, A and B
    std::vector<size_t>  oa_{0}, ob_{0};    // pair i: [oa_[i], oa_[i+1])
    std::vector<int>     guide_;            // pair i: m+1 entries at oa_[i] + i
    std::vector<double>  score_;
    std::vector<uint8_t> flags_;
    std::shared_ptr<const OwnedParams> params_;
    // The params the PENDING guides belong to: null for the lazy soft guide ("under the
    // params current then"); set by score_only_() to the params it scored, so a deferred
    // guide is the one the eager path would have stored.
    std::shared_ptr<const OwnedParams> gparams_;
    int kernel_ = kBackendAuto;
    std::vector<double>  grad_;             // pair i: gstride() doubles (matrix, 4 gaps); untracked
    // ── Tracks ──
    bool tracked_ = false;
    int  nsets_ = 0;                        // K of the params
    std::vector<uint16_t> tk_;              // tracked: pair i's track at oa_[i]
    std::vector<uint16_t> gk_;              // tracked: pair i's gap track at oa_[i] + i
    // Tracked hard gradients, in one of two stores (settings().track_counts): the summed
    // values are the same, the costs differ.
    // Paths (the default): pair i's moves, 2 bits each (0 diagonal, 1 gap in B, 2 gap in
    // A), in tp_[tpo_[i] ..], at most len_a + len_b of them (a path has no more steps),
    // the path at its END (written from the back: Aligner::hard_path_packed); tps_[i] =
    // {i0, j0, steps, off}: the start cell, the move count and the first move's index.
    // 8 bytes of padding at the end (tpath_grad_'s window).  Replayed per sum.
    std::vector<uint8_t>  tp_;
    std::vector<size_t>   tpo_{0};
    std::vector<std::array<int32_t, 4>> tps_;
    // Counts: pair i's touched fields of the flat K-set layout and their counts (gap
    // fields stand for -count), exactly as many as it has (~10-40 a miRNA pair), counted
    // in the walk (Aligner::hard_fields).  Each WRITER (a worker of one fill call, or the
    // single-pair path's Work) appends its pairs' entries to its own buffer,
    // tcw_[writer]; tcp_[i] says where pair i's are.  A whole-batch fill (score_and_grad,
    // banded_grad) starts a new generation: it recomputes every pair, so every buffer is
    // reused from empty (tcw_used_ counts the writers handed out).  Single-pair
    // recomputes append to their writer and leave the old entries dead; tc_compact_
    // drops them when they outgrow the live ones.
    // An entry is one word, key << 12 | count, while count < 4095 and key < 2^20 (K *
    // (n*n + 4) fields: 2600 sets of a 20-letter alphabet), else three: kTcEscape, key,
    // count (a one-word entry never equals kTcEscape).  Read in order, never indexed.
    static constexpr uint32_t kTcEscape = 0xFFFFFFFFu;
    static constexpr size_t kTcAhead = 16;   // pairs prefetched ahead in the sum (8-32: the same)
    // n, off: in words; esc: whether any entry is escaped (else n entries, one word each)
    struct TRef { uint32_t w = 0, n = 0, gen = 0, esc = 0; uint64_t off = 0; };
    std::deque<std::vector<uint32_t>> tcw_;    // (a deque: a buffer never moves)
    std::vector<TRef>     tcp_;
    size_t   tcw_used_ = 0;
    uint32_t tc_gen_ = 1;
    size_t   tc_dead_ = 0;   // dead entries in the single-pair writer's buffer
    // Tracked soft gradients: pair i's per-row record (Aligner::soft_record_counts / _gaps)
    // at sr_[sro_[i] ..]: len_a*n counts, then 4*(len_a+1) gap fields.
    std::vector<double>   sr_;
    std::vector<size_t>   sro_{0};
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
            throw std::logic_error("nwgrad: alignment not computed; call align() first");
        if (!path_stored(i))
            throw std::logic_error(
                "nwgrad: stored path unavailable; use align(keep_paths=True) or banded_align(keep_paths=True)");
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
        if (grad_mode_ == GradMode::None)
            throw std::logic_error("nwgrad: grad_mode is None; construct with Hard or Soft to enable gradients");
        for (size_t i = 0; i < size(); ++i)
            if (!grad_valid(i))
                throw std::logic_error("nwgrad: gradient not computed; call score_and_grad() or banded_grad() first");
    }
    void check_sets_(const AlignParams& params) const {
        if (params.num_sets() != nsets_)
            throw std::invalid_argument(
                "nwgrad: set_params() cannot change the number of parameter sets (" +
                std::to_string(nsets_) + " -> " + std::to_string(params.num_sets()) + ")");
    }
    // Every pair's track of `t` (want(k) entries each, or empty: the default) against
    // K sets: the lengths per pair (a flat buffer's are its offsets'), the values in one
    // pass over all of them; the per-pair search only names an offender.
    template <class Want>
    static void check_tracks_(const PairTracks& t, Want&& want, int K, const char* what) {
        if (!t) return;
        for (size_t k = 0; k < t.size(); ++k) {
            const auto r = t[k];
            if (!r.empty() && r.size() != want(k)) check_track_(r, want(k), K, k, what);
        }
        if (t.max_entry() < K) return;
        for (size_t k = 0; k < t.size(); ++k)
            if (!t[k].empty()) check_track_(t[k], want(k), K, k, what);
    }
    static void check_track_(std::span<const uint16_t> t, size_t want, int K, size_t pair,
                             const char* what) {
        if (t.size() != want)
            throw std::invalid_argument(
                "nwgrad: pair " + std::to_string(pair) + ": " + what + " has " + std::to_string(t.size()) +
                " entries, expected " + std::to_string(want) +
                (std::string(what) == "track" ? " (one set per residue of A)" : " (one per DP row 0..len(A))"));
        for (size_t r = 0; r < t.size(); ++r)
            if (t[r] >= K)
                throw std::invalid_argument(
                    "nwgrad: pair " + std::to_string(pair) + ": " + what + "[" + std::to_string(r) + "] = " +
                    std::to_string(t[r]) + " is not a parameter set (params has " + std::to_string(K) + ")");
    }
    // Both hard stores emptied (their memory freed).  A new generation: no counts entry,
    // nor a Work's writer, from before survives.
    void reset_hard_store_() {
        std::vector<uint8_t>().swap(tp_);
        tpo_.assign(1, 0);
        std::vector<std::array<int32_t, 4>>().swap(tps_);
        tcw_.clear();
        std::vector<TRef>().swap(tcp_);
        ++tc_gen_;
        tcw_used_ = 0;
        tc_dead_ = 0;
    }
    // A tracked batch's gradient store, sized for every pair: path or count slots (hard)
    // or record slots (soft).  Called on an empty store (construction, a reset).
    void size_grad_store_() {
        const size_t N = size();
        if (grad_mode_ == GradMode::Soft) {
            const size_t na = static_cast<size_t>(alpha_->size());
            sro_.reserve(N + 1);
            for (size_t i = 0; i < N; ++i)
                sro_.push_back(sro_.back() + len_a(i) * na + 4 * (len_a(i) + 1));
            sr_.assign(sro_.back(), 0.0);
            return;
        }
        if (grad_mode_ != GradMode::Hard) return;
        if (s_.track_counts) {
            tcp_.resize(N);
            return;
        }
        tpo_.reserve(N + 1);
        for (size_t i = 0; i < N; ++i)
            tpo_.push_back(tpo_.back() + (len_a(i) + len_b(i) + 3) / 4);
        tp_.assign(tpo_.back() + 8, 0);   // + 8: tpath_grad_ reads the moves 8 bytes at a time
        tps_.assign(N, {0, 0, 0, 0});
    }
    void store_srec_(size_t i, const std::vector<double>& cnt, const std::vector<double>& gaps) {
        double* d = sr_.data() + sro_[i];
        std::copy(cnt.begin(), cnt.end(), d);
        std::copy(gaps.begin(), gaps.end(), d + cnt.size());
    }
    // Pair i's path (and with gj its guide), packed into its slot.
    template <class Al>
    void store_tpath_(size_t i, const Al& al, const DpBuffer& buf, std::vector<int>* gj) {
        uint8_t* d = tp_.data() + tpo_[i];
        const int cap = static_cast<int>(len_a(i) + len_b(i));
        std::memset(d, 0, (static_cast<size_t>(cap) + 3) / 4);
        int i0 = 0, j0 = 0;
        const int off = al.hard_path_packed(buf, d, cap, i0, j0, gj);
        tps_[i] = {i0, j0, cap - off, off};
    }
    struct Work;   // (below: a worker's state)
    // A whole-batch hard fill begins (it recomputes every pair): a new generation, every
    // writer buffer reused from empty, and no pair's gradient valid until it is stored.
    void tc_begin_full_() {
        if (!tracked_ || grad_mode_ != GradMode::Hard || !s_.track_counts) return;
        ++tc_gen_;
        tcw_used_ = 0;
        tc_dead_ = 0;
        for (auto& f : flags_) f &= static_cast<uint8_t>(~kGrad);
    }
    static std::mutex& tc_mutex_() { static std::mutex m; return m; }
    // Work w's buffer for this generation (handed out under a lock: workers start
    // concurrently).
    std::vector<uint32_t>& tc_writer_(Work& w) {
        if (w.tcbuf && w.tcgen == tc_gen_) return *w.tcbuf;
        std::lock_guard<std::mutex> lk(tc_mutex_());
        const size_t idx = tcw_used_++;
        if (idx == tcw_.size()) tcw_.emplace_back();
        tcw_[idx].clear();
        w.tcbuf = &tcw_[idx];
        w.tcw = static_cast<uint32_t>(idx);
        w.tcgen = tc_gen_;
        return *w.tcbuf;
    }
    // Pair i's hard gradient (and with gj its guide) from the DP `al` just ran: counted
    // field by field along the traceback walk, then appended to w's buffer.
    template <class Al>
    void store_tcounts_(size_t i, const Al& al, const DpBuffer& buf, std::vector<int>* gj, Work& w) {
        Touched& t = w.tc;
        t.begin(kstride_(), 2 * (len_a(i) + len_b(i)) + 2);
        uint32_t* keys = t.keys.data();
        uint32_t* slot = t.slot.data();
        size_t nk = 0;
        al.hard_fields(buf, [&](uint32_t k) { if (slot[k]++ == 0) keys[nk++] = k; }, gj);
        std::vector<uint32_t>& out = tc_writer_(w);
        const uint64_t off = out.size();
        uint32_t esc = 0;
        for (size_t q = 0; q < nk; ++q) {
            const uint32_t k = keys[q], c = slot[k];
            slot[k] = 0;
            if (c < 4095u && k < (1u << 20)) out.push_back(k << 12 | c);
            else { out.push_back(kTcEscape); out.push_back(k); out.push_back(c); esc = 1; }
        }
        TRef& r = tcp_[i];
        const bool solo = &w == solo_.get();
        if (solo && r.gen == tc_gen_ && r.w == w.tcw) tc_dead_ += r.n;   // overwritten in place
        r = {w.tcw, static_cast<uint32_t>(out.size() - off), tc_gen_, esc, off};
        if (solo && tc_dead_ > 4096 && tc_dead_ * 2 > out.size()) tc_compact_(w);
    }
    // Drop the dead entries of w's buffer (the single-pair writer's: only it overwrites).
    void tc_compact_(Work& w) {
        std::vector<uint32_t>& b = *w.tcbuf;
        std::vector<uint32_t> nb;
        nb.reserve(b.size() - tc_dead_);
        for (TRef& r : tcp_) {
            if (r.gen != tc_gen_ || r.w != w.tcw) continue;
            const uint64_t off = nb.size();
            nb.insert(nb.end(), b.begin() + static_cast<std::ptrdiff_t>(r.off),
                      b.begin() + static_cast<std::ptrdiff_t>(r.off + r.n));
            r.off = off;
        }
        b.swap(nb);
        tc_dead_ = 0;
    }
    // Pair i's hard gradient (and with gj its guide) from the DP `al` just ran, into the
    // batch's store.
    template <class Al>
    void store_hard_(size_t i, const Al& al, const DpBuffer& buf, std::vector<int>* gj, Work& w) {
        if (s_.track_counts) store_tcounts_(i, al, buf, gj, w);
        else                 store_tpath_(i, al, buf, gj);
    }
    // One pair's gradient, sparse: the first n entries of keys / vals are the fields it
    // touches (in first-touch order) and their values.  `slot` has one entry per field of
    // the flat K-set layout and is left all-zero again after each pair; while a pair is
    // gathered it holds a hard path's or walk's COUNT per field (tpath_grad_ / end_counts,
    // store_tcounts_), or a soft block's position + 1 at the block's first field
    // (add_block / end_blocks).
    struct Touched {
        std::vector<uint32_t> keys, slot;
        std::vector<double> vals;
        size_t n = 0;
        void begin(size_t nfields, size_t maxkeys) {
            if (slot.size() < nfields) slot.assign(nfields, 0);
            if (keys.size() < maxkeys) { keys.resize(maxkeys); vals.resize(maxkeys); }
            n = 0;
        }
        // The counts as values, negative from field `neg` on (the gap fields: the sign
        // bit set, no branch — the two kinds interleave unpredictably).
        void end_counts(uint32_t neg) {
            for (size_t q = 0; q < n; ++q) {
                const uint32_t k = keys[q];
                const double c = static_cast<double>(slot[k]);
                vals[q] = std::bit_cast<double>(std::bit_cast<uint64_t>(c) |
                                                (static_cast<uint64_t>(k >= neg) << 63));
                slot[k] = 0;
            }
        }
        // `len` values onto the fields k0.. (the layout's blocks are disjoint, and a block
        // always comes with the same k0 and len).
        void add_block(uint32_t k0, const double* v, uint32_t len) {
            uint32_t& sl = slot[k0];
            if (sl) {
                double* d = vals.data() + (sl - 1);
                for (uint32_t q = 0; q < len; ++q) d[q] += v[q];
                return;
            }
            sl = static_cast<uint32_t>(n + 1);
            for (uint32_t q = 0; q < len; ++q) { keys[n + q] = k0 + q; vals[n + q] = v[q]; }
            n += len;
        }
        void end_blocks() { for (size_t q = 0; q < n; ++q) slot[keys[q]] = 0; }
    };
    // Pair i's hard gradient from its stored path, sparse into `t`: every field of the
    // flat K-set layout it touches — the matrix stack first (set*n*n + a*n + b), then the
    // gaps (K*n*n + 4*set + field) — with the walk gradient's counts (gap fields negative).
    void tpath_grad_(size_t i, Touched& t) const {
        const auto ca = codes_a(i), cb = codes_b(i);
        const auto tk = track(i), gk = gap_track(i);
        const size_t na = static_cast<size_t>(alpha_->size());
        const size_t gbase = static_cast<size_t>(nsets_) * nn_;
        const uint8_t* d = tp_.data() + tpo_[i];
        size_t r = static_cast<size_t>(tps_[i][0]), c = static_cast<size_t>(tps_[i][1]);
        const int steps = tps_[i][2], off = tps_[i][3];
        t.begin(kstride_(), 2 * static_cast<size_t>(steps) + 1);
        uint32_t* keys = t.keys.data();
        uint32_t* slot = t.slot.data();
        size_t n = 0;
        auto one = [&](uint32_t key, uint32_t cnt) { keys[n] = key; n += slot[key] == 0; slot[key] += cnt; };
        // By runs of one move (up to 28, from a 64-bit window): a diagonal or B-gap run
        // walks its rows (each row its own set), a run of gaps in A stays in row r (one
        // key).  A gap run's first step also opens (the open field is extend - 1); a run
        // split at 28 continues with prev == mv, so it does not open twice.
        int prev = 3;
        static_assert(std::endian::native == std::endian::little, "the window reads move k at bit 2k");
        for (int k = off, end = off + steps; k < end; ) {
            uint64_t wd;
            std::memcpy(&wd, d + (k >> 2), sizeof wd);
            wd >>= 2 * (k & 3);
            const int mv = static_cast<int>(wd & 3);
            const uint64_t x = (wd ^ (0x5555555555555555ull * static_cast<uint64_t>(mv))) & ((1ull << 56) - 1);
            const int run = std::min(x ? std::countr_zero(x) >> 1 : 28, end - k);
            if (mv == 0) {
                for (int q = 0; q < run; ++q)
                    one(static_cast<uint32_t>(static_cast<size_t>(tk[r + q]) * nn_ + ca[r + q] * na + cb[c + q]), 1);
                r += run; c += run;
            } else if (mv == 1) {
                const auto eb = [&](size_t row) {
                    return static_cast<uint32_t>(gbase + 4 * static_cast<size_t>(tk[row]) + AlignParams::kExtendB);
                };
                if (GM == GapModel::Affine && prev != 1) one(eb(r) - 1, 1);
                for (int q = 0; q < run; ++q) one(eb(r + q), 1);
                r += run;
            } else {
                const uint32_t ea = static_cast<uint32_t>(gbase + 4 * static_cast<size_t>(gk[r]) + AlignParams::kExtendA);
                if (GM == GapModel::Affine && prev != 2) one(ea - 1, 1);
                one(ea, static_cast<uint32_t>(run));
                c += run;
            }
            prev = mv;
            k += run;
        }
        t.n = n;
        t.end_counts(static_cast<uint32_t>(gbase));
    }
    // Pair i's stored hard gradient, sparse into `t`: its fields of the flat K-set layout
    // — the matrix stack first (set*n*n + a*n + b), then the gaps (K*n*n + 4*set +
    // field) — with the walk gradient's counts (gap fields negative).
    void tcount_grad_(size_t i, Touched& t) const {
        const TRef& r = tcp_[i];
        t.begin(kstride_(), r.n);   // (entries <= words)
        const uint32_t gbase = static_cast<uint32_t>(static_cast<size_t>(nsets_) * nn_);
        const uint32_t* d = tcw_[r.w].data() + r.off;
        // The value: the count, its sign bit set on the gap fields (no branch: the two
        // kinds interleave unpredictably).
        auto put = [&](size_t n, uint32_t k, uint32_t c) {
            t.keys[n] = k;
            t.vals[n] = std::bit_cast<double>(std::bit_cast<uint64_t>(static_cast<double>(c)) |
                                              (static_cast<uint64_t>(k >= gbase) << 63));
        };
        if (!r.esc) {   // every entry one word
            for (size_t q = 0; q < r.n; ++q) put(q, d[q] >> 12, d[q] & 0xFFFu);
            t.n = r.n;
            return;
        }
        size_t n = 0;
        for (size_t q = 0; q < r.n; ++n) {
            if (d[q] != kTcEscape) { put(n, d[q] >> 12, d[q] & 0xFFFu); q += 1; }
            else                   { put(n, d[q + 1], d[q + 2]); q += 3; }
        }
        t.n = n;
    }
    // The flat K-set gradient layout of tpath_grad_, tcount_grad_ and srec_grad_, to and
    // from an AlignParams.
    size_t kstride_() const noexcept { return static_cast<size_t>(nsets_) * (nn_ + 4); }
    void flat_to_params_(const double* flat, AlignParams& out) const {
        std::memcpy(out.matrix.data(), flat, static_cast<size_t>(nsets_) * nn_ * sizeof(double));
        const double* g = flat + static_cast<size_t>(nsets_) * nn_;
        for (int k = 0; k < nsets_; ++k)
            for (int f = 0; f < 4; ++f) out.gap_ref(k, f) = g[4 * static_cast<size_t>(k) + static_cast<size_t>(f)];
    }

    void prepare_paths_(bool keep, bool all = true) {
        if (!keep) {
            if (all) {   // every path is about to be stale: free them, don't just flag them
                for (auto& f : flags_) f &= static_cast<uint8_t>(~kStored);
                std::vector<std::vector<uint32_t>>().swap(path_);
                std::vector<std::array<int, 2>>().swap(path_end_);
            }
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
    // dst row r = w[r] * src row r, for cnt rows of gs.  Out of line for the same reason.
    static constexpr size_t SUM_CHUNK = 64;
    [[gnu::noinline]] static void scale_rows_(double* dst, const double* src, size_t gs,
                                              const double* w, size_t cnt) {
        for (size_t r = 0; r < cnt; ++r) {
            const double s = w[r];
            for (size_t k = 0; k < gs; ++k) dst[r * gs + k] = src[r * gs + k] * s;
        }
    }
    // acc[k] += src row r [k] for r = 0..cnt-1, in row order.  GS > 0: gs == GS, summed in
    // a local array the compiler can keep in registers (acc may alias nothing it reads,
    // but a pointer cannot say so).
    template <size_t GS>
    static void add_rows_(double* acc, const double* src, size_t gs, size_t cnt) {
        if constexpr (GS > 0) {
            double a[GS];
            for (size_t k = 0; k < GS; ++k) a[k] = acc[k];
            for (size_t r = 0; r < cnt; ++r)
                for (size_t k = 0; k < GS; ++k) a[k] += src[r * GS + k];
            for (size_t k = 0; k < GS; ++k) acc[k] = a[k];
        } else {
            for (size_t r = 0; r < cnt; ++r)
                for (size_t k = 0; k < gs; ++k) acc[k] += src[r * gs + k];
        }
    }
    AlignParams blocked_sum_(const Alphabet& alpha, const double* weights) const {
        if (tracked_) return blocked_sum_tracked_(alpha, weights);
        const size_t n = size(), gs = gstride();
        const size_t nblocks = (n + WEIGHTED_GRAD_BLOCK - 1) / WEIGHTED_GRAD_BLOCK;
        std::vector<double> partial(nblocks * gs, 0.0);
        std::atomic<size_t> next{0};
        // Products for SUM_CHUNK pairs per out-of-line call, then the adds; at the DNA/RNA
        // gradient sizes the adds run on a fixed-size local accumulator (registers, a
        // compile-time trip).  Each acc[k] adds the same products in the same pair order.
        // One call per pair over a 20-double loop was ~0.14 ms per 20k pairs; gcc PGO got
        // 0.10 by unrolling that loop for the trip it saw.
        auto worker = [&]() {
            std::vector<double> scaled(SUM_CHUNK * gs);
            for (size_t b; (b = next.fetch_add(1, std::memory_order_relaxed)) < nblocks; ) {
                double* acc = partial.data() + b * gs;
                const size_t end = std::min(n, (b + 1) * WEIGHTED_GRAD_BLOCK);
                for (size_t i = b * WEIGHTED_GRAD_BLOCK; i < end; i += SUM_CHUNK) {
                    const size_t cnt = std::min(SUM_CHUNK, end - i);
                    const double* g = grad_.data() + i * gs;
                    if (weights) {
                        scale_rows_(scaled.data(), g, gs, weights + i, cnt);
                        g = scaled.data();
                    }
                    if (gs == 20)      add_rows_<20>(acc, g, gs, cnt);
                    else if (gs == 29) add_rows_<29>(acc, g, gs, cnt);
                    else               add_rows_<0>(acc, g, gs, cnt);
                }
            }
        };
        if (nblocks > 0)
            run_workers_guarded(std::min<int>(s_.n_threads, static_cast<int>(nblocks)), worker);
        std::vector<double> tot(gs, 0.0);
        for (size_t b = 0; b < nblocks; ++b)
            for (size_t k = 0; k < gs; ++k) tot[k] += partial[b * gs + k];
        AlignParams out(alpha);
        std::memcpy(out.matrix.data(), tot.data(), nn_ * sizeof(double));
        out.gap_open_a = tot[nn_]; out.gap_extend_a = tot[nn_ + 1];
        out.gap_open_b = tot[nn_ + 2]; out.gap_extend_b = tot[nn_ + 3];
        return out;
    }

    // Tracked: as blocked_sum_, from each pair's stored gradient (pair_grad_): per pair,
    // each field it touches gets weight * count, formed out of line, added once — exactly
    // what the dense sum adds there; the fields it does not touch would get w * 0, which
    // leaves a sum that started at +0 unchanged.  So with one set this is bit-identical
    // to the untracked sum, and with K the same arithmetic per field.
    AlignParams blocked_sum_tracked_(const Alphabet& alpha, const double* weights) const {
        const size_t n = size(), gs = kstride_();
        const size_t nblocks = (n + WEIGHTED_GRAD_BLOCK - 1) / WEIGHTED_GRAD_BLOCK;
        std::vector<double> partial(nblocks * gs, 0.0);
        std::atomic<size_t> next{0};
        const bool counts = grad_mode_ != GradMode::Soft && s_.track_counts;
        auto worker = [&]() {
            std::vector<double> scaled;
            Touched t;
            for (size_t b; (b = next.fetch_add(1, std::memory_order_relaxed)) < nblocks; ) {
                double* acc = partial.data() + b * gs;
                const size_t end = std::min(n, (b + 1) * WEIGHTED_GRAD_BLOCK);
                for (size_t i = b * WEIGHTED_GRAD_BLOCK; i < end; ++i) {
                    if (counts && i + kTcAhead < end) {   // the pairs' entries are scattered over the writers
                        const TRef& r = tcp_[i + kTcAhead];
                        const uint32_t* d = tcw_[r.w].data() + r.off;
                        for (uint32_t q = 0; q < r.n; q += 16) __builtin_prefetch(d + q);
                    }
                    pair_grad_(i, t);
                    const size_t nk = t.n;
                    if (weights) {
                        if (scaled.size() < nk) scaled.resize(nk);
                        scale_into_(scaled.data(), t.vals.data(), nk, weights[i]);
                        for (size_t q = 0; q < nk; ++q) acc[t.keys[q]] += scaled[q];
                    } else {
                        for (size_t q = 0; q < nk; ++q) acc[t.keys[q]] += t.vals[q];
                    }
                }
            }
        };
        if (nblocks > 0)
            run_workers_guarded(std::min<int>(s_.n_threads, static_cast<int>(nblocks)), worker);
        std::vector<double> tot(gs, 0.0);
        for (size_t b = 0; b < nblocks; ++b)
            for (size_t k = 0; k < gs; ++k) tot[k] += partial[b * gs + k];
        AlignParams out(alpha, nsets_);
        flat_to_params_(tot.data(), out);
        return out;
    }
    // Pair i's stored soft record, scattered into the flat K-set layout (kstride_'s):
    // row r's counts to its set's block, its gap fields to its sets — summed in row order,
    // each touched field emitted once.
    void srec_grad_(size_t i, Touched& t) const {
        const auto ca = codes_a(i);
        const auto tk = track(i), gk = gap_track(i);
        const size_t na = static_cast<size_t>(alpha_->size()), m = len_a(i);
        const size_t gbase = static_cast<size_t>(nsets_) * nn_;
        t.begin(kstride_(), m * na + 4 * (m + 1));
        const double* rec = sr_.data() + sro_[i];
        const uint32_t nau = static_cast<uint32_t>(na);
        for (size_t r = 0; r < m; ++r)   // row r's counts: its set's row of a[r]
            t.add_block(static_cast<uint32_t>((static_cast<size_t>(tk[r]) * na + ca[r]) * na), rec + r * na, nau);
        const double* g = rec + m * na;
        for (size_t r = 0; r <= m; ++r) {   // (open, extend) pairs: A's of row r, B's into it
            t.add_block(static_cast<uint32_t>(gbase + 4 * static_cast<size_t>(gk[r]) + AlignParams::kOpenA), g + 4 * r, 2);
            if (r > 0)
                t.add_block(static_cast<uint32_t>(gbase + 4 * static_cast<size_t>(tk[r - 1]) + AlignParams::kOpenB),
                            g + 4 * r + 2, 2);
        }
        t.end_blocks();
    }
    // Pair i's stored tracked gradient, sparse into `t` (keys / vals).
    void pair_grad_(size_t i, Touched& t) const {
        if (grad_mode_ == GradMode::Soft) srec_grad_(i, t);
        else if (s_.track_counts)         tcount_grad_(i, t);
        else                              tpath_grad_(i, t);
    }
    void pair_flat_grad_(size_t i, double* flat) const {
        Touched& t = touched_tls_();
        pair_grad_(i, t);
        for (size_t q = 0; q < t.n; ++q) flat[t.keys[q]] = t.vals[q];
    }
    static Touched& touched_tls_() { static thread_local Touched t; return t; }

    // ── A worker's state: one buffer, one Aligner per band, gradient scratch ─────
    struct Work {
        DpBuffer buf;
        FullAl full;
        BandAl band;
        AlignParams g;
        std::vector<int> gj;
        std::vector<FullAl> lanes;           // tracked inter-pair: one aligner per lane
        std::vector<BandAl> blanes;
        std::vector<TrackJob<T>> tjobs;
        std::vector<std::vector<uint32_t>> srs;   // tracked inter-pair soft: per lane, rows
        std::vector<std::vector<double>> srw;     // ... and weights (InterSoftJob::rs / rw)
        std::vector<const uint32_t*> srsp;
        std::vector<const double*> srwp;
        // ... and each set's four weights, of params *swp: valid for one call (inter-pair
        // groups run on the call's own Works, never on the persistent solo_).
        const AlignParams* swp = nullptr;
        std::vector<double> sw;
        // tracked hard, counts: this Work's writer buffer (BatchEngine::tc_writer_) and its scratch
        std::vector<uint32_t>* tcbuf = nullptr;
        uint32_t tcw = 0, tcgen = 0;
        Touched tc;
        explicit Work(const BatchEngine& e) : g(e.alphabet(), e.nsets_) { configure(e); }
        // The batch's settings, (re)applied: a reused Work follows setting changes.
        void configure(const BatchEngine& e) {
            full.set_traceback(e.s_.traceback);
            full.set_hb_cutoff(e.s_.hb_cutoff);
            full.set_rowwise_full(e.s_.rowwise_full);
            full.set_soft_impl(e.s_.soft_impl);
            full.set_soft_temperature(e.s_.soft_temperature);
            // Only the Full aligner can use pointers; the banded one needs its score
            // tables and its footprint is O(m*band) already.
            band.set_traceback(TracebackMode::Scores);
            band.set_soft_impl(e.s_.soft_impl);
            band.set_soft_temperature(e.s_.soft_temperature);
        }
        // State valid for one call only, reset when a pooled Work starts a new one: swp is
        // compared by ADDRESS (a new call's params may sit where the last one's did).
        void new_call() noexcept { swp = nullptr; tcbuf = nullptr; }
    };
    // The workers' Works, kept between calls and leased one per worker per call, as
    // solo_ is for single pairs: a fresh Work per call first-touched its DP buffers (the
    // Pointers tables, the Grid lines) in every worker at once, and those page faults
    // serialize — nighthaven, 12 threads, L3000: Pointers and Grid 1.x-2x the time of a
    // warm buffer (AGENTS.md, "Grid traceback").  At most n_threads Works are pooled (the
    // same footprint a call holds anyway, now kept until drop_paths() or the batch dies).
    // (the mutex through a pointer: SeqPairBatchT moves its engine, and std::mutex cannot)
    mutable std::unique_ptr<std::mutex> pool_mu_ = std::make_unique<std::mutex>();
    mutable std::vector<std::unique_ptr<Work>> pool_;
    struct WorkLease {
        const BatchEngine& e;
        std::unique_ptr<Work> w;
        explicit WorkLease(const BatchEngine& eng) : e(eng) {
            {
                std::lock_guard<std::mutex> lock(*e.pool_mu_);
                if (!e.pool_.empty()) { w = std::move(e.pool_.back()); e.pool_.pop_back(); }
            }
            if (w) { w->configure(e); w->new_call(); }
            else   w = std::make_unique<Work>(e);
        }
        ~WorkLease() {
            try {
                std::lock_guard<std::mutex> lock(*e.pool_mu_);
                e.pool_.push_back(std::move(w));
            } catch (...) {}   // a failed push frees the Work; the next call builds one
        }
        WorkLease(const WorkLease&) = delete;
        WorkLease& operator=(const WorkLease&) = delete;
        Work& operator*() const noexcept { return *w; }
    };
    void drop_pool_() const noexcept {
        if (!pool_mu_) return;   // moved from
        std::lock_guard<std::mutex> lock(*pool_mu_);
        pool_.clear();
    }
    // The calling thread's Work for single-pair operations (SeqPair's align() and
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
        al.set_kernel(kernel_);
        problem_(al, i, *params_, 0, {});
    }
    // Pair i on `al` under params `o` (band, guide), with its tracks when tracked.
    template<class Al>
    [[gnu::always_inline]] inline void problem_(Al& al, size_t i, const OwnedParams& o, int band,
                                                std::vector<int>&& guide) const {
        if (tracked_) al.set_problem(codes_a(i), codes_b(i), o.p, band, std::move(guide),
                                     track(i), gap_track(i), o.stack_or_null(), o.gaps.data(), o.step_max);
        else if constexpr (nw_is_int<T>)   // the prepared stack, gaps and step_max (no tracks)
            al.set_problem(codes_a(i), codes_b(i), o.p, band, std::move(guide), {}, {},
                           o.stack_or_null(), o.gaps.data(), o.step_max);
        else          al.set_problem(codes_a(i), codes_b(i), o.p, band, std::move(guide));
    }
    // The hard gradient and guide of the DP `al` just ran, into pair i, in one walk —
    // the untracked sequence spelled out at each site (the code it always was).
#define NWGRAD_HARD_GRAD_GUIDE(i, al, w)                                   \
    do {                                                                    \
        if (tracked_) { hard_grad_guide_(i, al, w); }                       \
        else {                                                              \
            (w).g.zero();                                                   \
            (al).hard_grad_and_guide((w).buf, (w).g, (w).gj);               \
            store_grad_(i, (w).g);                                          \
            store_guide_(i, (w).gj);                                        \
        }                                                                   \
    } while (0)
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
    // The hard gradient and the guide of the DP `al` just ran, into pair i, in one walk:
    // tracked, into the batch's store (store_hard_); otherwise the dense gradient.
    template<class Al>
    void hard_grad_guide_(size_t i, Al& al, Work& w) {   // tracked
        store_hard_(i, al, w.buf, &w.gj, w);
        store_guide_(i, w.gj);
    }
    // Hard or soft gradient of the DP `al` just ran, into pair i.
    template<class Al>
    void grad_with_buf_(size_t i, Al& al, Work& w) {
        if (tracked_) {
            if (grad_mode_ == GradMode::Hard) {
                store_hard_(i, al, w.buf, nullptr, w);
                return;
            }
            store_srec_(i, al.soft_record_counts(), al.soft_record_gaps());
            return;
        }
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
    // Modern fills publish all computed results together.
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
        // Publish score/gradient validity only after the whole fill succeeds. A
        // soft fill can throw after its Viterbi has already stored a new path.
        flags_[i] &= static_cast<uint8_t>(~(kScore | kGrad | kStored));
        auto& al = w.full;
        set_problem_(al, i);
        const bool lazy = s_.soft_guide_lazy && !keep, post = s_.soft_guide_posterior && !keep;
        if (grad_mode_ == GradMode::Hard) {
            al.compute_viterbi(w.buf);
            score_[i] = al.score();
            NWGRAD_HARD_GRAD_GUIDE(i, al, w);
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

    // SeqPair::resolve_guide: the pending guide, under the params it belongs to (gparams_;
    // a lazy soft guide: the params current now).
    void resolve_guide_(size_t i, Work& w) {
        if (!guide_pending(i)) return;
        w.full.set_kernel(kernel_);
        problem_(w.full, i, gparams_ ? *gparams_ : *params_, 0, {});
        w.full.compute_viterbi(w.buf);
        store_guide_(i, w.full.guide_j_from_viterbi(w.buf));
        flags_[i] &= static_cast<uint8_t>(~(kPending | kStored));
    }

    // SeqPair::banded_grad_with_dp.
    void run_banded_(size_t i, Work& w, int bandwidth, bool keep) {
        resolve_guide_(i, w);
        flags_[i] &= static_cast<uint8_t>(~(kScore | kGrad | kStored));
        auto& al = w.band;
        al.set_kernel(kernel_);
        const int* g = guide_ptr_(i);
        problem_(al, i, *params_, bandwidth, std::vector<int>(g, g + len_a(i) + 1));
        if (grad_mode_ == GradMode::Hard) {   // one traceback walk for guide and gradient
            al.compute_viterbi(w.buf);
            score_[i] = al.score();
            NWGRAD_HARD_GRAD_GUIDE(i, al, w);
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
        score_[i] = soft.logz * s_.soft_temperature;
        if (tracked_) {   // the lane's per-row record: len_a rows of counts, len_a+1 of gaps
            const size_t m = len_a(i), na = static_cast<size_t>(alpha_->size());
            double* d = sr_.data() + sro_[i];
            std::copy(soft.counts, soft.counts + m * na, d);
            std::copy(soft.gaps, soft.gaps + 4 * (m + 1), d + m * na);
        } else {
        double* d = grad_ptr_(i);
        for (size_t k = 0; k < nn_; ++k) d[k] = soft.counts[k];
        for (size_t k = 0; k < 4; ++k) d[nn_ + k] = soft.gaps[k];
        }
        if (s_.soft_guide_posterior && soft.gpost) {
            const int m = static_cast<int>(len_a(i)), n = static_cast<int>(len_b(i));
            std::vector<int> rows(static_cast<size_t>(m) + 1);
            for (int r = 0; r <= m; ++r) rows[static_cast<size_t>(r)] = soft.gpost[static_cast<size_t>(r) * soft.gstride];
            store_guide_(i, FullAl::monotone_guide(rows.data(), m, n));
        }
    }

    // SeqPair::score_and_grad_interleaved: pair i is lane `lane` of an inter-pair fill.
    // `pre`: the lane's aligner, already set up with pair i (tracked lanes); else w.full.
    void run_interleaved_(size_t i, Work& w, int W, int lane, double best, int bi, int bj,
                          const SoftLane* soft, size_t stride, bool keep, FullAl* pre = nullptr) {
        flags_[i] &= static_cast<uint8_t>(~kStored);
        auto& al = pre ? *pre : w.full;
        if (!pre) set_problem_(al, i);
        al.adopt_interleaved(w.buf, W, lane, best, bi, bj, stride);
        score_[i] = al.score();
        if (grad_mode_ == GradMode::Hard) {
            NWGRAD_HARD_GRAD_GUIDE(i, al, w);   // one walk for both
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
        const bool lazy = s_.soft_guide_lazy && !keep, post = s_.soft_guide_posterior && !keep;
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
        w.band.set_kernel(kernel_);
        problem_(w.band, i, *params_, bandwidth, std::vector<int>(g, g + len_a(i) + 1));
        w.band.banded_lane_rows(W, lane, M, blo, bhi, slo, shi, bri, brj);
    }
    void banded_interleaved_(size_t i, Work& w, int bandwidth, int W, int lane,
                             double best, int bi, int bj, BandAl* pre = nullptr) {
        // The lane's problem was set by banded_lane_setup_ on a fresh BandAl per lane?
        // No: one BandAl per worker, so set it again (set_problem is cheap and the
        // banded tables live in the shared buffer, untouched by set_problem).  Tracked
        // lanes have their own (`pre`), set up before the fill.
        auto& al = pre ? *pre : w.band;
        if (!pre) {
            const int* g = guide_ptr_(i);
            al.set_kernel(kernel_);
            problem_(al, i, *params_, bandwidth, std::vector<int>(g, g + len_a(i) + 1));
        }
        flags_[i] &= static_cast<uint8_t>(~kStored);
        al.adopt_interleaved(w.buf, W, lane, best, bi, bj);
        score_[i] = al.score();
        if (grad_mode_ == GradMode::Hard) {
            NWGRAD_HARD_GRAD_GUIDE(i, al, w);
        } else {
            store_guide_(i, al.guide_j_from_viterbi(w.buf));
        }
        finish_(i, true, false);
    }

    // ── Inter-pair plan ──────────────────────────────────────────────────────

    static int inter_w_(const LevelKernels& K) {
        const TypedKernels<T>& KT = K.template of<T>();
        return KT.inter_fill ? KT.inter_w : 0;
    }

    // The pieces of an InterJob every inter-pair fill shares (score_and_grad_inter_,
    // banded_grad_inter_, stream_inter_loop_), so the three cannot drift apart: the
    // kernels' inputs must be exactly what each pair's own fill would use.
    //   cast_block_:        the substitution block in T, as the own fill rounds it (blkT_)
    //   set_inter_params_:  block, alphabet size, the four penalties in T, gap model
    //   wire_inter_tables_: size the shared tables (H for linear, VM/VX/VY for affine) to
    //                       `sz` cells per table and point the job at them
    //   run_inter_fill_:    the kernel of this precision
    // (The whole matrix stack: a tracked lane reads its stacked row.)
    static void cast_block_(const AlignParams& P, std::vector<T>& blkT) {
        const size_t nbk = static_cast<size_t>(P.matrix.size()) * P.matrix.size() *
                           static_cast<size_t>(P.num_sets());
        blkT.resize(nbk);
        for (size_t k = 0; k < nbk; ++k) {
            if constexpr (nw_is_int<T>) blkT[k] = nw_int_checked<T>(P.matrix.data()[k], "substitution score");
            else                        blkT[k] = static_cast<T>(P.matrix.data()[k]);
        }
    }
    // Tracked inter-pair lanes: lane l's aligner set up with its pair (the group's last
    // pair repeated in spare lanes) and its rows in w.tjobs[l], before the fill, so the
    // fill reads every lane's own rows and the lane adopts its tables without a second
    // set_problem().  Untracked groups need none of this (they share the scalars).
    template <class Al>
    void tracked_lanes_(Work& w, std::vector<Al>& lanes, int W, size_t real,
                        const std::vector<size_t>& order, size_t s, int bandwidth = 0) {
        if (lanes.size() < static_cast<size_t>(W)) lanes.resize(static_cast<size_t>(W));
        w.tjobs.assign(static_cast<size_t>(W), TrackJob<T>{});
        for (int l = 0; l < W; ++l) {
            if (static_cast<size_t>(l) >= real) { w.tjobs[static_cast<size_t>(l)] = w.tjobs[real - 1]; continue; }
            const size_t i = order[s + static_cast<size_t>(l)];
            Al& al = lanes[static_cast<size_t>(l)];
            configure_lane_(al);
            al.set_kernel(kernel_);
            if constexpr (std::is_same_v<Al, BandAl>) {
                const int* g = guide_ptr_(i);
                problem_(al, i, *params_, bandwidth, std::vector<int>(g, g + len_a(i) + 1));
            } else {
                problem_(al, i, *params_, 0, {});
            }
            al.track_job(w.tjobs[static_cast<size_t>(l)]);
        }
    }
    // Tracked inter-pair soft: lane l's rows (its stacked rows) and weights (exp of each
    // row's penalties at the batch temperature), the group's last pair in spare lanes.
    // False when a weight overflows (the group then runs its pairs' own paths).
    bool soft_lanes_(Work& w, int W, size_t real, const std::vector<size_t>& order, size_t s,
                     const AlignParams& P) const {
        w.srs.resize(static_cast<size_t>(W)); w.srw.resize(static_cast<size_t>(W));
        w.srsp.assign(static_cast<size_t>(W), nullptr); w.srwp.assign(static_cast<size_t>(W), nullptr);
        const double it = 1.0 / s_.soft_temperature, big = std::numeric_limits<double>::max();
        const uint32_t na = static_cast<uint32_t>(alpha_->size());
        constexpr bool lin = GM == GapModel::Linear;
        if (w.swp != &P) {   // each set's four weights, once per params and call
            const int K = P.num_sets();
            w.sw.assign(4 * static_cast<size_t>(K), 0.0);
            for (int k = 0; k < K; ++k) {
                double* sk = w.sw.data() + 4 * static_cast<size_t>(k);
                sk[1] = std::exp(-P.gap(k, AlignParams::kExtendA) * it);
                if (!lin) sk[0] = std::exp(-(P.gap(k, AlignParams::kOpenA) + P.gap(k, AlignParams::kExtendA)) * it);
                sk[3] = std::exp(-P.gap(k, AlignParams::kExtendB) * it);
                if (!lin) sk[2] = std::exp(-(P.gap(k, AlignParams::kOpenB) + P.gap(k, AlignParams::kExtendB)) * it);
            }
            w.swp = &P;
        }
        bool ok = true;
        for (int l = 0; l < W; ++l) {
            const size_t ll = std::min<size_t>(static_cast<size_t>(l), real - 1);
            const size_t i = order[s + ll], m = len_a(i);
            auto& rs = w.srs[static_cast<size_t>(l)];
            auto& rw = w.srw[static_cast<size_t>(l)];
            const auto ca = codes_a(i);
            const auto tk = track(i), gk = gap_track(i);
            rs.resize(m);
            for (size_t r = 0; r < m; ++r) rs[r] = static_cast<uint32_t>(tk[r]) * na + ca[r];
            rw.resize(4 * (m + 1));
            const double* sw = w.sw.data();
            for (size_t r = 0; r <= m; ++r) {
                const size_t g = 4 * static_cast<size_t>(gk[r]);
                rw[4 * r] = sw[g]; rw[4 * r + 1] = sw[g + 1];
                if (r > 0) {
                    const size_t t = 4 * static_cast<size_t>(tk[r - 1]);
                    rw[4 * r + 2] = sw[t + 2]; rw[4 * r + 3] = sw[t + 3];
                } else {
                    rw[2] = rw[3] = 0.0;
                }
                for (int f = 0; f < 4; ++f) ok &= rw[4 * r + static_cast<size_t>(f)] <= big;
            }
            w.srsp[static_cast<size_t>(l)] = rs.data();
            w.srwp[static_cast<size_t>(l)] = rw.data();
        }
        return ok;
    }
    void configure_lane_(FullAl& al) const {
        al.set_traceback(s_.traceback);
        al.set_hb_cutoff(s_.hb_cutoff);
        al.set_rowwise_full(s_.rowwise_full);
        al.set_soft_impl(s_.soft_impl);
        al.set_soft_temperature(s_.soft_temperature);
    }
    void configure_lane_(BandAl& al) const {
        al.set_traceback(TracebackMode::Scores);
        al.set_soft_impl(s_.soft_impl);
        al.set_soft_temperature(s_.soft_temperature);
    }
    static void set_inter_params_(InterJobT<T>& job, const AlignParams& P, const std::vector<T>& blkT) {
        job.blk = blkT.data(); job.nalpha = P.matrix.size();
        job.go_a = static_cast<T>(P.gap_open_a); job.ge_a = static_cast<T>(P.gap_extend_a);
        job.go_b = static_cast<T>(P.gap_open_b); job.ge_b = static_cast<T>(P.gap_extend_b);
        job.linear = GM == GapModel::Linear ? 1 : 0;
    }
    // cast_block_ + set_inter_params_ for the batch's params, except that a float32
    // tracked group reads their once-cast stack (OwnedParams) instead of casting K
    // matrices again.
    void block_for_(InterJobT<T>& job, std::vector<T>& blkT) const {
        const T* pp = params_->stack_or_null();
        if (!pp) cast_block_(params_->p, blkT);
        set_inter_params_(job, params_->p, blkT);
        if (pp) job.blk = pp;
    }
    static void wire_inter_tables_(InterJobT<T>& job, DpBuffer& buf, size_t sz) {
        if constexpr (GM == GapModel::Linear) {
            // Linear's one table rides in the H slot (adopt_interleaved reads buf.H).
            if (buf.H.size() < sz) buf.H.resize(sz);
            job.VM = buf.H.data();
        } else {
            for (auto* v : {&buf.VM, &buf.VX, &buf.VY}) if (v->size() < sz) v->resize(sz);
            job.VM = buf.VM.data(); job.VX = buf.VX.data(); job.VY = buf.VY.data();
        }
    }
    static void run_inter_fill_(const LevelKernels& K, InterJobT<T>& job) {
        K.template of<T>().inter_fill(job);
    }
    // Score only (grad_mode None, Full): the same lanes on ONE row per table, each lane's
    // score out in job.best; false when the level has no such kernel.
    static bool has_inter_score_(const LevelKernels& K) {
        return K.template of<T>().inter_score != nullptr;
    }
    static void run_inter_score_(const LevelKernels& K, InterJobT<T>& job) {
        K.template of<T>().inter_score(job);
    }
    // Linear Global takes the shared fill only from 4 lanes: below that (DNA, W=2) the own
    // fill wins (0.76x on SSE2).  Alphabets over 8 used to be excluded too, on a 1.10-1.22x
    // measured against the old baseline-compiled own fill; re-measured 2026-10-23 against the
    // leveled striped linear kernel (nighthaven AVX2, protein, inter / own): hard L20-80
    // 0.54-0.85 at 1 thread and 0.67-0.87 at 12 for every type, score-only 0.09-0.78 — so the
    // exclusion is gone, and the per-pair caps in inter_eligible_ take the bands that lose.
    bool linear_fill_ok_(const LevelKernels& K) const {
        return !(GM == GapModel::Linear && AM == AlignMode::Global && inter_w_(K) < 4);
    }
    // The vector backend the batch's inter-pair fills run on, or -1 (every pair its own
    // fill).
    int inter_backend_() const {
        const int backend = (kernel_ == kBackendAuto) ? global_default_backend() : kernel_;
        if (backend < 0) return -1;
        const LevelKernels& K = level_kernels(backend);
        if (inter_w_(K) <= 0) return -1;
        if (!linear_fill_ok_(K) && grad_mode_ != GradMode::Soft) return -1;
        return backend;
    }
    // Whether pair i joins an inter-pair fill on `backend` (else: its own fill).
    bool inter_eligible_(size_t i, const LevelKernels& K) const {
        if (len_a(i) == 0 || len_b(i) == 0) return false;
        if (!inter_pair_fits(len_a(i), len_b(i), GM == GapModel::Affine,
                             grad_mode_ == GradMode::Soft, inter_w_(K), sizeof(T), K.d.inter_w))
            return false;
        // Linear, integer T, alphabet over 8: the inter-pair fill gathers per lane-cell,
        // and the leveled striped linear kernel now beats it past ~L120 (inter / own, Local
        // protein, hard, 1 thread: L100 0.85-0.99, L150 1.12-1.22, L200 1.14-1.33; nighthaven
        // AVX2 and spot NEON agree).  Floating T keeps the shared fill: on AVX2 it still wins
        // through L200 (0.63-0.84), and the NEON loss there (1.08-1.18 from L150) does not
        // justify costing the primary target.
        // (floating T: capped on AArch64 only, inter_linear_gather_cap — simd_levels.hpp)
        if constexpr (GM == GapModel::Linear)
            if (alpha_->size() > 8 && grad_mode_ != GradMode::Soft &&
                (len_a(i) + 1) * (len_b(i) + 1) > inter_linear_gather_cap<T, AM == AlignMode::Global>())
                return false;
        // Hirschberg pairs past the cutoff split: no shared fill.
        return !hb_splits(tb_resolved_, s_.traceback == TracebackMode::Default ? FullAl::kAutoPointerCells : 0, s_.hb_cutoff, len_a(i), len_b(i));
    }

public:
    // Ragged B: one group may mix B lengths (InterJobT::nb) — every lane is padded to the
    // longest, so a group admits a pair only while that padding stays small.
    static bool ragged_ok(size_t shortest, size_t len_b) {
        return len_b * 4 <= shortest * 5 + 16;   // <= 1.25x + 4 columns
    }
    // The inter-pair plan as the next "interpair" call runs it, rebuilt if stale:
    // {pairs in shared groups, pairs on their own fill, groups}.  For nwgrad._tuning
    // sweeps, to see where the gates routed the pairs.  (The score-only gather gate,
    // inter_score_ok, acts per group inside the call and is not reflected here.)
    std::array<size_t, 3> inter_plan_counts() {
        ensure_plan_();
        return {plan_.elig.size(), plan_.other.size(), plan_.groups.size()};
    }
private:
    struct InterPlan {
        int default_backend = -1000, hb_cutoff = -1;   // what it was built for
        unsigned gate_epoch = ~0u;                      // ... and under which gate overrides
        int backend = -1;                               // every group's vector backend
        std::vector<size_t> elig, other;
        std::vector<std::pair<size_t, size_t>> groups;   // [start, end) into elig
    };
    InterPlan plan_;

    // Built once and reused until the default ISA changes, the Hirschberg cutoff moves or
    // a fill gate is overridden (nwgrad._tuning):
    // re-sorting 2.5M pairs every training iteration was ~0.9 s of serial work per call.
    // Groups: len B runs of len-A-sorted pairs, cut every W and wherever len B outgrows
    // the group's shortest past ragged_ok().
    void ensure_plan_() {
        const int def_backend = global_default_backend();
        const unsigned epoch = gate_epoch();
        if (plan_.default_backend == def_backend && plan_.hb_cutoff == s_.hb_cutoff &&
            plan_.gate_epoch == epoch) return;
        const size_t N = size();
        InterPlan P;
        P.default_backend = def_backend; P.hb_cutoff = s_.hb_cutoff; P.gate_epoch = epoch;
        P.backend = inter_backend_();
        if (P.backend < 0) {
            P.other.resize(N);
            std::iota(P.other.begin(), P.other.end(), size_t{0});
            plan_ = std::move(P);
            return;
        }
        const LevelKernels& K = level_kernels(P.backend);
        struct Key { size_t len_b, len_a, i; };
        std::vector<Key> keys;
        keys.reserve(N);
        for (size_t i = 0; i < N; ++i) {
            if (!inter_eligible_(i, K)) { P.other.push_back(i); continue; }
            keys.push_back({len_b(i), len_a(i), i});
        }
        std::sort(keys.begin(), keys.end(), [](const Key& x, const Key& y) {
            if (x.len_b != y.len_b) return x.len_b < y.len_b;
            if (x.len_a != y.len_a) return x.len_a < y.len_a;
            return x.i < y.i;
        });
        P.elig.reserve(keys.size());
        for (const Key& k : keys) P.elig.push_back(k.i);
        const size_t W = static_cast<size_t>(inter_w_(K));
        for (size_t s = 0; s < keys.size();) {
            size_t e = s + 1;
            while (e < keys.size() && e - s < W && ragged_ok(keys[s].len_b, keys[e].len_b)) ++e;
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
        const auto& groups = plan_.groups;
        const size_t G = groups.size(), tasks = G + other.size();
        const bool soft = grad_mode_ == GradMode::Soft;
        // A lazy-guide soft group needs no Viterbi at all, nor a posterior-guide one.
        const bool no_viterbi = soft && (s_.soft_guide_lazy || s_.soft_guide_posterior) && !keep;
        const bool want_post = soft && s_.soft_guide_posterior && !keep;
        std::atomic<size_t> idx{0};
        auto worker = [&]() {
            WorkLease lease(*this);
            Work& w = *lease;
            std::vector<const unsigned char*> a, b;
            std::vector<int> m, bi, bj, nb;
            std::vector<T> best, blkT;
            DVec sscr;
            std::vector<int> siscr, sok, sgp;
            std::vector<double> ses, slogz, scnt, sgap;
            std::vector<int> cM_;   // tracked soft: each lane's chunk M (its row stride)
            while (true) {
                const size_t t = idx.fetch_add(1, std::memory_order_relaxed);
                if (t >= tasks) break;
                if (t >= G) { run_full_(other[t - G], w, keep); continue; }
                const auto [s, e] = groups[t];
                const size_t real = e - s;
                const LevelKernels& K = level_kernels(plan_.backend);
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
                const AlignParams& P = params();
                constexpr bool lin = GM == GapModel::Linear;
                const bool skip_fill = no_viterbi || !linear_fill_ok_(K);
                InterJobT<T> job{};
                job.a = a.data(); job.m = m.data(); job.b = b.data(); job.n = n; job.M = M;
                job.align_mode = (AM == AlignMode::Local) ? 1 : 0;
                if (ragged) job.nb = nb.data();
                auto& buf = w.buf;
                const bool tl = tracked_ && !skip_fill;   // tracked lanes, set up before the fill
                if (tl) {
                    tracked_lanes_(w, w.lanes, W, real, elig, s);
                    job.tr = w.tjobs.data();
                }
                if (!skip_fill) {
                    block_for_(job, blkT);
                    wire_inter_tables_(job, buf, static_cast<size_t>(M + 1) * (n + 1) * W);
                    job.best = best.data(); job.best_i = bi.data(); job.best_j = bj.data();
                    run_inter_fill_(K, job);
                }
                // Soft lanes share the forward-backward too (one soft_impl, not "log",
                // one temperature — batch-wide here), when the weights fit.
                bool soft_group = K.inter_soft != nullptr && soft && s_.soft_impl != SoftImpl::Log;
                const size_t nnk = nn_;
                if (soft_group) {
                    InterSoftJob sj{};
                    bool fin = inter_soft_weights(P, s_.soft_temperature, lin, ses, sj);
                    if (fin && tracked_) fin = soft_lanes_(w, W, real, elig, s, P);
                    soft_group = fin;
                    // Tracked: outputs per lane and row; chunk c's at c*M rows (its own M
                    // is the stride inside it).
                    const size_t ccnt = tracked_ ? static_cast<size_t>(M) * alpha_->size() : nnk;
                    const size_t cgap = tracked_ ? 4 * (static_cast<size_t>(M) + 1) : 4;
                    if (fin) {
                        // The soft pass is double, K.inter_w lanes: a float32 group runs
                        // it in W / SW chunks.
                        const int SW = K.d.inter_w;
                        slogz.resize(W); scnt.resize(W * ccnt); sgap.resize(W * cgap); sok.assign(W, 0);
                        cM_.assign(static_cast<size_t>(W), 0);
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
                            sj.logz = slogz.data() + c; sj.counts = scnt.data() + c * ccnt;
                            sj.gaps = sgap.data() + c * cgap; sj.ok = sok.data() + c;
                            if (tracked_) {
                                sj.rs = w.srsp.data() + c; sj.rw = w.srwp.data() + c;
                                for (int l = c; l < c + SW; ++l) cM_[static_cast<size_t>(l)] = Mc;
                            }
                            sj.gpost = want_post ? sgp.data() + static_cast<size_t>(c) * (M + 1) : nullptr;
                            K.inter_soft(sj);
                        }
                    }
                }
                for (size_t l = 0; l < real; ++l) {
                    const size_t i = elig[s + l];
                    SoftLane lane{};
                    const bool use = soft_group && sok[l];
                    if (use && tracked_) {
                        const int SW = K.d.inter_w, c = static_cast<int>(l) / SW * SW, Mc = cM_[l];
                        const size_t na = alpha_->size();
                        lane = {slogz[l],
                                scnt.data() + static_cast<size_t>(c) * static_cast<size_t>(M) * na +
                                    (l - static_cast<size_t>(c)) * static_cast<size_t>(Mc) * na,
                                sgap.data() + static_cast<size_t>(c) * 4 * (static_cast<size_t>(M) + 1) +
                                    (l - static_cast<size_t>(c)) * 4 * (static_cast<size_t>(Mc) + 1)};
                        if (want_post) {
                            lane.gpost = sgp.data() + static_cast<size_t>(c) * (M + 1) + (l - c);
                            lane.gstride = SW;
                        }
                    } else if (use) {
                        lane = {slogz[l], scnt.data() + l * nnk, sgap.data() + l * 4};
                        if (want_post) {
                            const int SW = K.d.inter_w, c = static_cast<int>(l) / SW * SW;
                            lane.gpost = sgp.data() + static_cast<size_t>(c) * (M + 1) + (l - c);
                            lane.gstride = SW;
                        }
                    }
                    if (skip_fill) run_soft_lane_(i, w, use ? &lane : nullptr, keep);
                    else run_interleaved_(i, w, W, static_cast<int>(l), best[l], bi[l], bj[l],
                                          use ? &lane : nullptr, gstride_, keep,
                                          tl ? &w.lanes[l] : nullptr);
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
        size_t full_calls = static_cast<size_t>(-1);
        int default_backend = -1000, hb_cutoff = -1;
        std::vector<size_t> elig, other;
        std::vector<std::pair<size_t, size_t>> groups;
    };
    BandGroups band_;
    size_t full_calls_ = 0;

    void ensure_band_groups_() {
        ensure_plan_();
        if (band_.full_calls == full_calls_ && band_.default_backend == plan_.default_backend &&
            band_.hb_cutoff == plan_.hb_cutoff)
            return;
        BandGroups Bg;
        Bg.full_calls = full_calls_;
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
        const size_t W = E.empty() ? 0 : static_cast<size_t>(inter_w_(level_kernels(plan_.backend)));
        for (size_t s = 0; s < E.size();) {
            size_t e = s + 1;
            while (e < E.size() && len_b(E[e]) == len_b(E[s])) ++e;
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
            for (size_t g = s; g < e; g += W) Bg.groups.emplace_back(g, std::min(e, g + W));
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
        const auto& groups = band_.groups;
        const size_t G = groups.size(), tasks = G + other.size();
        std::atomic<size_t> idx{0};
        auto worker = [&]() {
            WorkLease lease(*this);
            Work& w = *lease;
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
                const LevelKernels& K = level_kernels(plan_.backend);
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
                    if (!tracked_)
                        banded_lane_setup_(i, w, bandwidth, W, l, M, blo.data(), bhi.data(),
                                           ulo.data(), uhi.data(), bri[l], brj[l]);
                }
                if (tracked_) {
                    tracked_lanes_(w, w.blanes, W, real, elig, s, bandwidth);
                    for (size_t l = 0; l < real; ++l)
                        w.blanes[l].banded_lane_rows(W, static_cast<int>(l), M, blo.data(), bhi.data(),
                                                     ulo.data(), uhi.data(), bri[l], brj[l]);
                }
                InterJobT<T> job{};
                job.a = a.data(); job.m = m.data(); job.b = b.data(); job.n = n; job.M = M;
                job.align_mode = (AM == AlignMode::Local) ? 1 : 0;
                if (tracked_) job.tr = w.tjobs.data();
                block_for_(job, blkT);   // affine: this path runs for affine only
                wire_inter_tables_(job, w.buf, static_cast<size_t>(M + 1) * (n + 1) * W);
                job.best = best.data(); job.best_i = bi.data(); job.best_j = bj.data();
                job.blo = blo.data(); job.bhi = bhi.data();
                job.ulo = ulo.data(); job.uhi = uhi.data();
                job.bri = bri.data(); job.brj = brj.data();
                run_inter_fill_(K, job);
                for (size_t l = 0; l < real; ++l)
                    banded_interleaved_(elig[s + l], w, bandwidth, W, static_cast<int>(l),
                                        best[l], bi[l], bj[l], tracked_ ? &w.blanes[l] : nullptr);
            }
        };
        if constexpr (GM == GapModel::Affine) run_workers_(tasks, worker);
        else banded_grad_dynamic_(bandwidth, false);
    }

    // ── Own-fill schedulers (SeqPairBatch's, unchanged in substance) ─────────

    // grad_mode None without stored paths: scores only — compute_score per pair, the
    // inter-pair score kernel for the plan's groups — and no Viterbi table or traceback.
    // The guides banded_grad() bands around become PENDING, resolved on first use under
    // the params scored here (gparams_), so they are exactly the guides the eager
    // path would have stored.  Scores are the exact optimum (the Pointers fill's score), as
    // `auto`'s Grid reports too (a float32 hirschberg_pmax pair may report a hair below).
    void score_only_() {
        gparams_ = params_;
        const size_t N = size();
        const bool inter = s_.inter_fill;
        if (inter) ensure_plan_();
        const size_t G = inter ? plan_.groups.size() : 0;
        const size_t tasks = inter ? G + plan_.other.size() : N;
        std::atomic<size_t> idx{0};
        auto worker = [&]() {
            WorkLease lease(*this);
            Work& w = *lease;
            std::vector<const unsigned char*> a, b;
            std::vector<int> m, nb;
            std::vector<T> best, blkT;
            auto own = [&](size_t i) {
                flags_[i] &= static_cast<uint8_t>(~kStored);
                set_problem_(w.full, i);
                score_[i] = w.full.compute_score(w.buf);
                finish_(i, false, true);
            };
            for (size_t t; (t = idx.fetch_add(1, std::memory_order_relaxed)) < tasks; ) {
                if (!inter) { own(t); continue; }
                if (t >= G) { own(plan_.other[t - G]); continue; }
                const auto [s, e] = plan_.groups[t];
                const size_t real = e - s;
                const LevelKernels& K = level_kernels(plan_.backend);
                if (!has_inter_score_(K) || !linear_fill_ok_(K)) {
                    for (size_t l = 0; l < real; ++l) own(plan_.elig[s + l]);
                    continue;
                }
                const int W = inter_w_(K);
                a.assign(W, nullptr); b.assign(W, nullptr); m.assign(W, 0); nb.assign(W, 0);
                best.assign(W, T(0));
                int M = 0, n = 0;
                bool ragged = false;
                for (int l = 0; l < W; ++l) {
                    const size_t i = plan_.elig[s + std::min<size_t>(l, real - 1)];
                    a[l] = codes_a(i).data(); b[l] = codes_b(i).data();
                    m[l] = static_cast<int>(len_a(i));
                    nb[l] = static_cast<int>(len_b(i));
                    M = std::max(M, m[l]);
                    n = std::max(n, nb[l]);
                    ragged |= nb[l] != nb[0];
                }
                if (!inter_score_ok(alpha_->size(), M, n)) {   // the gather: short pairs only
                    for (size_t l = 0; l < real; ++l) own(plan_.elig[s + l]);
                    continue;
                }
                InterJobT<T> job{};
                job.a = a.data(); job.m = m.data(); job.b = b.data(); job.n = n; job.M = M;
                job.align_mode = (AM == AlignMode::Local) ? 1 : 0;
                if (ragged) job.nb = nb.data();
                if (tracked_) {
                    tracked_lanes_(w, w.lanes, W, real, plan_.elig, s);
                    job.tr = w.tjobs.data();
                }
                block_for_(job, blkT);
                wire_inter_tables_(job, w.buf, static_cast<size_t>(n + 1) * W);
                job.best = best.data();
                run_inter_score_(K, job);
                for (size_t l = 0; l < real; ++l) {
                    const size_t i = plan_.elig[s + l];
                    flags_[i] &= static_cast<uint8_t>(~kStored);
                    score_[i] = static_cast<double>(best[l]);
                    finish_(i, false, true);
                }
            }
        };
        run_workers_(tasks, worker);
    }

    void score_and_grad_dynamic_(bool keep) {
        const size_t N = size();
        std::atomic<size_t> idx{0};
        auto worker = [&]() {
            WorkLease lease(*this);
            Work& w = *lease;
            for (size_t i; (i = idx.fetch_add(1, std::memory_order_relaxed)) < N; )
                run_full_(i, w, keep);
        };
        run_workers_(N, worker);
    }

    double length_weight_(double cells) const noexcept {
        if (s_.long_cost_ratio <= 1.0 || s_.weight_hi <= s_.weight_lo) return 1.0;
        const double eff = std::sqrt(cells);
        const double t = std::clamp((eff - s_.weight_lo) / (s_.weight_hi - s_.weight_lo), 0.0, 1.0);
        return 1.0 + (s_.long_cost_ratio - 1.0) * t;
    }

    // Length-sorted, equal-work chunks, one per thread, plus a reserve of the smallest
    // tasks: bounds the DP memory high-water mark at sum-over-chunks rather than
    // n_threads x the global maximum (human proteome, 16 threads: 4.9 vs 30.6 GB).  See
    // the measurements recorded at SeqPairBatch's s_.long_cost_ratio and s_.reserve_frac.
    template<typename Fn>
    void run_sorted_phase_(const std::vector<double>& cost, int nthr, Fn&& task) {
        const size_t N = cost.size();
        std::vector<size_t> order(N);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return cost[a] < cost[b]; });
        const double total = std::accumulate(cost.begin(), cost.end(), 0.0);
        size_t r = 0;
        {
            const double want = total * std::clamp(s_.reserve_frac, 0.0, 0.5);
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
        const bool prof = s_.profile;
        if (prof) profile_out.assign(static_cast<size_t>(nthr), PhaseProfile{});
        const auto t_start = clk::now();
        auto since = [](clk::time_point a, clk::time_point b) {
            return std::chrono::duration<double>(b - a).count();
        };
        auto worker = [&]() {
            const int k = next_worker.fetch_add(1, std::memory_order_relaxed);
            WorkLease lease(*this);
            Work& w = *lease;
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
        const int nthr = std::min<int>(s_.n_threads, static_cast<int>(N));
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
            WorkLease lease(*this);
            Work& w = *lease;
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
            WorkLease lease(*this);
            Work& w = *lease;
            for (size_t k; (k = idx.fetch_add(1, std::memory_order_relaxed)) < N; )
                run_banded_(order[k], w, bandwidth, keep);
        };
        run_workers_(N, worker);
    }

    template<typename Fn>
    void parallel_for_c_(size_t N, Fn&& fn) const {
        std::atomic<size_t> idx{0};
        auto worker = [&]() {
            for (size_t i; (i = idx.fetch_add(1, std::memory_order_relaxed)) < N; ) fn(i);
        };
        if (N == 0) return;
        run_workers_guarded(std::min<int>(s_.n_threads, static_cast<int>(std::min<size_t>(N, 1u << 30))), worker);
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
        run_workers_guarded(std::min<int>(s_.n_threads, static_cast<int>(std::min<size_t>(N, 1u << 30))), worker);
    }
};
#undef NWGRAD_HARD_GRAD_GUIDE
