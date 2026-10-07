#pragma once
// FROZEN ORACLE — not part of the library.
//
// The batch classes as they were before the 0.6 restructure (SeqPair with two Aligners
// per pair, SeqPairBatch over SeqPair objects, BatchAligner), kept verbatim inside
// namespace legacy so tests/cpp/test_batch_engine.cpp can keep proving that the
// restructured BatchEngine / SeqPairBatch / align_stream return bit-identical results.
// They share the CURRENT Aligner, so the comparison isolates the batch layer.  Delete
// this directory together with that test once the equivalence no longer needs proving.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "align_params.hpp"
#include "parallel.hpp"
#include "legacy_seq_pair.hpp"

// SeqPairBatch holds non-owning pointers to SeqPair objects added via add().
// On the Python side, nb::keep_alive ensures each added SeqPair outlives the batch.
//
// Pairs created by add_many() are the exception: the batch owns those outright
// (see owned_), because they never exist as Python objects at all.

namespace legacy {

template<class T = double>
struct SeqPairBatchT {
    // Viterbi precision T; SeqPair/DpBuffer shadow the global aliases so the members and
    // worker buffers below are at this precision.  Python binds SeqPairBatchT<float> as
    // `SeqPairBatch` (accepting `SeqPair` = SeqPairT<float>) and <double> as the *Double.
    using SeqPair  = SeqPairT<T>;
    using DpBuffer = DpBufferT<T>;

    std::vector<SeqPair*> pairs;
    int n_threads;

    // Work-scheduling policy for score_and_grad().  See score_and_grad_sorted_().
    // false = one atomic counter, tasks in insertion order (the original).
    // true  = length-sorted, equal-work chunks with per-thread size affinity.
    // Both produce identical results; this trades memory and makespan, not answers.
    bool   sorted_schedule = false;
    // Fraction of total work held back from the chunks as filler for threads that
    // finish early.  Drawn from the SMALLEST tasks, which is what makes it safe:
    // any thread can run one inside the buffer its own chunk already forced it to
    // allocate, so soaking up imbalance never grows the memory high-water mark
    // much (it does grow it some -- reserving the cheap tasks pushes every chunk
    // boundary up: on the human proteome at 16 threads, 2.67 GB at 0.05 against
    // 4.18 GB at 0.50, both still far under dynamic's 15.30 GB).
    //
    // DEFAULT 0.0 -- NO RESERVE -- because it measured best, which was not the
    // expectation.  Fleet sweep, 300 timed arms: 4 hosts (sse2/avx2/avx512/neon),
    // 5 length distributions (lognormal sigma 0..1.2, fixed mean, so only the tail
    // shape varies), 6 reserve fractions, 2-3 thread counts each.
    //
    //   tailed work (40 groups) mean ratio vs dynamic:
    //     rf=0.00 0.770 | 0.05 0.780 | 0.10 0.781 | 0.20 0.796 | 0.35 0.824 | 0.50 0.871
    //   flat work (10 groups): 1.015 .. 1.067 -- rf barely registers, AND the hosts
    //     disagree on which end is better (skynet 1.05->1.50 as rf rises, solace
    //     1.03->0.95), so there is nothing portable there to tune toward.
    //   real proteome (n=45594, conc 0.616, avx2, 8 threads): dynamic 47.17s,
    //     rf=0.00 0.768x, rf=0.05 0.864x, rf=0.20 0.791x.
    //
    // An ADAPTIVE rf -- keyed on the work concentration the scheduler already
    // computes -- was designed and then rejected on this data: its mean regret
    // against the per-case optimum was +0.032, WORSE than the constant 0.00's
    // +0.020.  It would have been fitting noise.  The knob stays because a caller
    // with a workload unlike anything swept here may want it; the default is off.
    double reserve_frac = 0.0;

    // Traceback strategy for pairs this batch constructs (add_many).  Fixed at
    // construction: it decides what the DP retains, so flipping it mid-flight would
    // only be meaningful between calls and invited stale-state bugs.  Pairs added by
    // add() keep whatever they were built with.
    TracebackMode traceback() const noexcept { return tb_; }

    // ── Cost-model weighting: cells are NOT fungible ─────────────────────────
    //
    // The plain square cost assumes every cell costs the same.  It does not.
    // Measured per-chunk throughput (nighthaven avx2, 6 threads, equal-CELL
    // chunks, so any spread here is pure model error):
    //
    //   chunk   16..771 aa  599.6 Mcell/s      chunk 1660..2308  206.7
    //   chunk  771..1188    339.5              chunk 2309..4243  202.3
    //   chunk 1189..1660    222.7              chunk 4359..8829  194.2
    //
    // A 3.09x spread, which left 22% of the machine idle and the LONG-sequence
    // thread straggling by 6.7 s.  weight() corrects for that: a task's cost is
    // its cells times a factor rising from 1 to `long_cost_ratio` as the effective
    // length crosses [weight_lo, weight_hi] -- the cache cliff.
    //
    // DEFAULT 1.0 -- WEIGHTING OFF -- because correcting the imbalance measured
    // SLOWER, which was not the expectation.  Sweep on nighthaven (avx2, 6 threads,
    // 11.6 Gcells, equal-cell chunks as the 1.0 baseline):
    //
    //   ratio  wall    finish spread  idle   straggler
    //   1.0    9.85s   6.63s          22%    t5 (4359..8829 aa)   1.000x
    //   2.0   10.31s   4.53s          10%    t5                   1.046x
    //   3.5   10.41s   1.59s           5%    t5                   1.057x
    //   4.5   10.25s   0.75s           3%    t1 (1160..1613 aa)   1.041x
    //   6.0   10.45s   0.58s           3%    t1                   1.061x
    //
    // It does what it says: at ratio >= 4.5 the straggler moves off the long chunk
    // onto a short one and idle collapses from 22% to 3%.  And every setting is
    // slower.  Busy time rises from 46.6 to ~59.7 thread-seconds for identical work
    // -- balancing does not reclaim idle capacity, it converts idle into contention,
    // because threads that finish early stop competing for memory and let the
    // memory-bound thread have the bus to itself.  The "22% idle" was never waste.
    //
    // So this is a knob for someone whose workload behaves unlike anything measured
    // here, not a default.  Set > 1.0 to shift the straggler toward the short chunks;
    // expect to pay for it.
    //
    // weight() is MONOTONIC in length, so the sort order is untouched and each
    // thread still owns a contiguous size band -- the property the whole memory
    // bound rests on.  Only the partition boundaries move.
    double long_cost_ratio = 1.0;    // 1.0 = no weighting (see above)
    double weight_lo = 500.0;        // below this, cache-resident: weight 1
    double weight_hi = 1700.0;       // above this, saturated: weight long_cost_ratio

    double length_weight(double cells) const noexcept {
        if (long_cost_ratio <= 1.0 || weight_hi <= weight_lo) return 1.0;
        const double eff = std::sqrt(cells);          // geometric-mean length
        const double t = std::clamp((eff - weight_lo) / (weight_hi - weight_lo),
                                    0.0, 1.0);
        return 1.0 + (long_cost_ratio - 1.0) * t;
    }

    // ── Per-thread schedule profiling (opt-in, off by default) ───────────────
    //
    // Exists to answer one specific question that timings alone could not: when
    // reserve_frac rises, does the CHUNK phase itself get slower -- because pulling
    // the cheap short sequences into the reserve leaves every thread grinding
    // uniformly long, memory-bound work at the same moment -- or is the reserve
    // merely relocating work?  Answering that needs cells AND seconds per phase,
    // so a per-phase Mcell/s can be computed; wall time alone cannot distinguish
    // "moved work" from "same work, run slower".
    //
    // Off by default and read only after the fact: two clock reads per worker per
    // phase, nothing inside the task loop.
    struct PhaseProfile {
        double chunk_s = 0.0,  reserve_s = 0.0;      // busy seconds in each phase
        double chunk_cells = 0.0, reserve_cells = 0.0;
        long   chunk_tasks = 0, reserve_tasks = 0;
        double finish_s = 0.0;                        // finish time from schedule start
    };
    bool profile = false;
    std::vector<PhaseProfile> profile_out;

    explicit SeqPairBatchT(int nt = 0, TracebackMode tb = TracebackMode::Default)
        : n_threads(nt > 0 ? nt : default_threads()), tb_(tb) {}

    // Non-copyable.  It never was, meaningfully — a copy would duplicate the raw
    // pointers in `pairs` and alias every borrowed SeqPair.  But it has to be
    // *said*, not merely true: std::vector<unique_ptr<T>> still advertises a copy
    // constructor (declared, ill-formed only if instantiated), so
    // is_copy_constructible_v<SeqPairBatch> stayed true and nanobind emitted a
    // copy thunk for it, which failed to compile inside the STL.  Moves are fine:
    // unique_ptr keeps every SeqPair at a fixed address, so `pairs` stays valid.
    SeqPairBatchT(const SeqPairBatchT&)            = delete;
    SeqPairBatchT& operator=(const SeqPairBatchT&) = delete;
    SeqPairBatchT(SeqPairBatchT&&)                 = default;
    SeqPairBatchT& operator=(SeqPairBatchT&&)      = default;

    // Every pair in a batch must share an alphabet: their gradients are summed,
    // and summing across alphabets is meaningless.  Checked here, on the
    // caller's thread, at the point of the mistake — rather than deep inside a
    // worker where the diagnostic would be useless.
    void add(SeqPair* sp) {
        if (!pairs.empty() && &sp->alphabet() != &pairs.front()->alphabet())
            throw std::invalid_argument(
                "nwgrad: SeqPair over alphabet \"" + sp->alphabet().symbols() +
                "\" cannot join a batch over alphabet \"" +
                pairs.front()->alphabet().symbols() + "\"");
        pairs.push_back(sp);
        unique_checked_ = false;  // verified lazily, at the next dispatch
        ++generation_;
    }

    // Bulk-construct N pairs in C++, in parallel, and append them.  The batch
    // owns them; they are reachable exactly like added ones, via operator[].
    //
    // This exists because constructing the pairs one at a time from Python is
    // the dominant cost of a large batch and almost none of it is alignment.
    // Measured on 2.5M pairs (22x50 nt): ~5.0s of C++ construction, ~6.6s of
    // nanobind wrapper objects, ~2.7s of add()'s keep-alive bookkeeping, and
    // ~3.3s of cyclic GC re-walking 2.5M live containers — against 6.4s for the
    // DP those pairs exist to feed.  Here the pairs never become Python objects,
    // so the last three costs do not arise and the first is threaded.
    //
    // `params` must outlive the batch, exactly as for a SeqPair built by hand.
    void add_many(const std::vector<std::string_view>& seqs_a,
                  const std::vector<std::string_view>& seqs_b,
                  const AlignParams& params,
                  GapModel gm, AlignMode am, GradMode gd,
                  int kernel = kBackendAuto) {
        if (seqs_a.size() != seqs_b.size())
            throw std::invalid_argument(
                "nwgrad: add_many() needs seqs_a and seqs_b of equal length (got " +
                std::to_string(seqs_a.size()) + " and " +
                std::to_string(seqs_b.size()) + ")");

        // Same eager, caller-thread alphabet check add() makes: a worker must
        // never be the one to discover a mismatch.
        if (!pairs.empty() &&
            &params.matrix.alphabet() != &pairs.front()->alphabet())
            throw std::invalid_argument(
                "nwgrad: SeqPair over alphabet \"" +
                params.matrix.alphabet().symbols() +
                "\" cannot join a batch over alphabet \"" +
                pairs.front()->alphabet().symbols() + "\"");

        const size_t N = seqs_a.size();
        if (N == 0) return;

        // Construct into a staging vector first.  An out-of-alphabet character
        // throws inside a worker; run_workers_guarded rethrows it on this
        // thread, `staged` unwinds, and the batch is left exactly as it was.
        // Writing it straight into `pairs` would leave a half-filled batch
        // behind a raised exception.
        std::vector<std::unique_ptr<SeqPair>> staged(N);
        std::atomic<size_t> idx{0};
        auto worker = [&]() {
            while (true) {
                size_t i = idx.fetch_add(1, std::memory_order_relaxed);
                if (i >= N) break;
                staged[i] = std::make_unique<SeqPair>(seqs_a[i], seqs_b[i],
                                                      params, gm, am, gd, kernel, tb_);
                // Harmless unless the pair's resolved traceback is Hirschberg (it only
                // reads hb_cutoff then), so applied unconditionally — tb_ may be the
                // Default sentinel, which resolves to Hirschberg per pair, not here.
                staged[i]->set_hb_cutoff(hb_cutoff);
                staged[i]->set_rowwise_full(rowwise_full);
                staged[i]->set_soft_impl(soft_impl);
                staged[i]->set_soft_temperature(soft_temperature);
                staged[i]->set_soft_guide_lazy(soft_guide_lazy);
                if (soft_guide_posterior) staged[i]->set_soft_guide_posterior(true);
            }
        };
        run_workers(N, worker);

        // unique_ptr keeps each SeqPair at a fixed address, so growing owned_
        // never invalidates the raw pointers in `pairs`.
        owned_.reserve(owned_.size() + N);
        pairs.reserve(pairs.size() + N);
        for (auto& up : staged) {
            pairs.push_back(up.get());
            owned_.push_back(std::move(up));
        }
        ++generation_;
    }

    size_t size() const noexcept { return pairs.size(); }

    // The alphabet shared by every pair in the batch.  Throws if empty.
    const Alphabet& alphabet() const {
        if (pairs.empty())
            throw std::logic_error("nwgrad: empty batch has no alphabet");
        return pairs.front()->alphabet();
    }

    SeqPair& operator[](size_t i) { return *pairs[i]; }
    const SeqPair& operator[](size_t i) const { return *pairs[i]; }

    // ── Batch operations ──────────────────────────────────────────────────────

    // Pre-allocate own DP tables on all pairs in parallel.
    // Must be called before align_full() / realign_banded() / compute_grad().
    // score_and_grad() uses thread-owned buffers and never requires this.
    void alloc_dp() {
        parallel_for(pairs.size(), [&](size_t i) { pairs[i]->alloc_dp(); });
    }

    // Set params on all pairs (O(1) per pair — no threading needed).
    // Clears score_valid and grad_valid on every pair; path_valid is preserved.
    // Point every pair at `params`, in parallel: each pair only stores the pointer
    // and invalidates its cached results, and the pairs are independent.  The
    // alphabet is checked once, before any pair changes, so a mismatch leaves the
    // whole batch as it was.  `params` must outlive the batch's use of it (the
    // Python binding keeps it alive).
    void set_params(const AlignParams& params) {
        if (pairs.empty()) return;
        if (&params.matrix.alphabet() != &alphabet())
            throw std::invalid_argument(
                "nwgrad: set_params() cannot change the alphabet (\"" +
                alphabet().symbols() + "\" -> \"" + params.matrix.alphabet().symbols() +
                "\"); construct new pairs instead");
        const size_t n = pairs.size();
        const size_t nblocks = (n + SCORES_BLOCK - 1) / SCORES_BLOCK;
        std::atomic<size_t> next{0};
        auto worker = [&]() {
            for (size_t b; (b = next.fetch_add(1, std::memory_order_relaxed)) < nblocks; ) {
                const size_t end = std::min(n, (b + 1) * SCORES_BLOCK);
                for (size_t i = b * SCORES_BLOCK; i < end; ++i) pairs[i]->set_params(params);
            }
        };
        run_workers(nblocks, worker);
    }

    // Drop DP tables on all pairs in parallel to free O(mn) memory per pair.
    // Cached scores, gradients, and guide_j vectors are preserved.
    void drop_dp() {
        parallel_for(pairs.size(), [&](size_t i) { pairs[i]->drop_dp(); });
    }

    // Full DP alignment of all pairs in parallel.
    // Returns sum of scores.
    double align_full() {
        const size_t N = pairs.size();
        std::vector<double> scores(N, 0.0);
        parallel_for(N, [&](size_t i) {
            pairs[i]->align_full();
            scores[i] = pairs[i]->score();
        });
        return std::accumulate(scores.begin(), scores.end(), 0.0);
    }

    // Banded realignment of all pairs in parallel around their current paths.
    // Returns sum of scores.
    double realign_banded(int bandwidth) {
        const size_t N = pairs.size();
        std::vector<double> scores(N, 0.0);
        parallel_for(N, [&](size_t i) {
            pairs[i]->realign_banded(bandwidth);
            scores[i] = pairs[i]->score();
        });
        return std::accumulate(scores.begin(), scores.end(), 0.0);
    }

    // Compute gradient on all pairs in parallel.
    // Returns the summed AlignParams gradient over all pairs.
    // If a pair already has grad_valid() == true (e.g. after score_and_grad_with_dp),
    // its cached gradient is used directly without rerunning the DP.
    // Throws on an empty batch: the sum has no alphabet, and a zero gradient
    // labelled with a guessed one would be a silent wrong answer.
    AlignParams compute_grad() {
        const Alphabet& alpha = alphabet();   // throws if empty
        const size_t N = pairs.size();
        std::atomic<size_t> idx{0};
        std::mutex grad_mutex;
        AlignParams grad_out(alpha);

        auto worker = [&]() {
            AlignParams local(alpha);
            while (true) {
                size_t i = idx.fetch_add(1, std::memory_order_relaxed);
                if (i >= N) break;
                if (!pairs[i]->grad_valid())
                    pairs[i]->compute_grad();
                local += pairs[i]->grad();
            }
            std::lock_guard<std::mutex> lock(grad_mutex);
            grad_out += local;
        };

        run_workers(N, worker);
        return grad_out;
    }

    // The pairs' cached scores, in pair order.  Runs no alignment: call
    // score_and_grad(), align_full(), realign_banded() or banded_grad() first.
    // Throws if any pair has no valid score.
    std::vector<double> scores() const {
        const size_t n = pairs.size();
        std::vector<double> out(n);
        const size_t nblocks = (n + SCORES_BLOCK - 1) / SCORES_BLOCK;
        std::atomic<size_t> next{0};
        auto worker = [&]() {
            for (size_t b; (b = next.fetch_add(1, std::memory_order_relaxed)) < nblocks; ) {
                const size_t end = std::min(n, (b + 1) * SCORES_BLOCK);
                for (size_t i = b * SCORES_BLOCK; i < end; ++i) out[i] = pairs[i]->score();
            }
        };
        if (nblocks > 0)
            run_workers_guarded(std::min<int>(n_threads, static_cast<int>(nblocks)), worker);
        return out;
    }

    // Pairs per worker task in scores() and set_params().  Only a scheduling
    // granularity: every pair is handled alone, so results do not depend on it.
    static constexpr size_t SCORES_BLOCK = 4096;

    // sum_i weights[i] * grad_i over the pairs' CACHED gradients.  Runs no
    // alignment, so a caller whose weights depend on the scores (a logistic
    // likelihood, say) runs score_and_grad(), derives the weights from
    // scores(), then calls this.
    //
    // Summed in fixed blocks of WEIGHTED_GRAD_BLOCK pairs: each block in pair
    // order, the blocks in parallel, then the block sums in block order.  The
    // blocks depend only on the pair count, so the result is bit-reproducible
    // and independent of n_threads (and identical to a plain pair-order sum for
    // batches of at most one block).  The time is not the multiply-adds but
    // fetching each pair's gradient from its own heap objects; parallel workers
    // keep many of those memory accesses in flight.
    //
    // No product is ever formed next to the add that consumes it: scale_into()
    // stores w_i * grad_i out of line, and the sum loads it.  A multiply adjacent to
    // an add is what compilers contract into an FMA, which rounds once instead of
    // twice -- on FMA targets (AArch64, by default) the sum would then differ in the
    // last bits from targets without FMA.  Same rule as the precomputed gap ramp in
    // hb_kernel_impl.inl.
    //
    // Throws on an empty batch (the sum has no alphabet), on a weight count other
    // than size(), and on a pair without a valid gradient.  Only reads the pairs,
    // so unlike the alignment operations it does not need each pair to appear once.
    AlignParams weighted_grad(const double* weights, size_t n) const {
        const Alphabet& alpha = alphabet();   // throws if empty
        if (n != pairs.size())
            throw std::invalid_argument(
                "nwgrad: weighted_grad() needs one weight per pair (got " +
                std::to_string(n) + " weights for " +
                std::to_string(pairs.size()) + " pairs)");
        const size_t nblocks = (n + WEIGHTED_GRAD_BLOCK - 1) / WEIGHTED_GRAD_BLOCK;
        std::vector<AlignParams> partial(nblocks, AlignParams(alpha));
        std::atomic<size_t> next{0};
        auto worker = [&]() {
            AlignParams scaled(alpha);
            for (size_t b; (b = next.fetch_add(1, std::memory_order_relaxed)) < nblocks; ) {
                const size_t end = std::min(n, (b + 1) * WEIGHTED_GRAD_BLOCK);
                for (size_t i = b * WEIGHTED_GRAD_BLOCK; i < end; ++i) {
                    scale_into(scaled, pairs[i]->grad(), weights[i]);
                    partial[b] += scaled;
                }
            }
        };
        if (nblocks > 0)
            run_workers_guarded(std::min<int>(n_threads, static_cast<int>(nblocks)), worker);
        AlignParams out(alpha);
        for (const AlignParams& p : partial) out += p;
        return out;
    }

    // Pairs per block in weighted_grad().  Part of the result's definition: a
    // different block size sums in a different order and can change the last bits.
    static constexpr size_t WEIGHTED_GRAD_BLOCK = 4096;

    // dst = s * src.  Out of line on purpose; see weighted_grad().
    [[gnu::noinline]] static void scale_into(AlignParams& dst, const AlignParams& src, double s) {
        dst = src;
        dst *= s;
    }

    // The pairs' CACHED gradients, copied out in pair order: pair i's matrix to
    // matrices[i*n*n ..] (row-major, n = alphabet().size(), rows and columns in
    // alphabet order) and its gap fields to gaps[i*4 ..] as gap_open_a,
    // gap_extend_a, gap_open_b, gap_extend_b.  The caller sizes both buffers for
    // size() pairs.  Runs no alignment, so per-pair gradients reach the caller as
    // two arrays rather than one AlignParams object per pair.
    //
    // Throws on an empty batch (no alphabet, so no matrix size) and on a pair
    // without a valid gradient.
    void grads_into(double* matrices, double* gaps) const {
        const size_t nn = static_cast<size_t>(alphabet().size()) *
                          static_cast<size_t>(alphabet().size());   // throws if empty
        for (size_t i = 0; i < pairs.size(); ++i) {
            const AlignParams& g = pairs[i]->grad();
            g.matrix.to_array(matrices + i * nn);
            double* gp = gaps + i * 4;
            gp[0] = g.gap_open_a;
            gp[1] = g.gap_extend_a;
            gp[2] = g.gap_open_b;
            gp[3] = g.gap_extend_b;
        }
    }

    // Full-pipeline batch operation using per-thread DpBuffers.
    // For each pair: runs the full DP, computes grad,
    // stores score + guide_j + grad into the SeqPair.  The pairs' own DP tables
    // are never allocated; dp_valid() remains false after this call.
    // Returns sum of scores.
    double score_and_grad() {
        const size_t N = pairs.size();
        if (N == 0) return 0.0;
        std::vector<double> scores(N, 0.0);
        ++full_calls_;   // guides may have moved: the banded grouping is stale

        if (inter_fill)           score_and_grad_inter_(scores);
        else if (sorted_schedule) score_and_grad_sorted_(scores);
        else                      score_and_grad_dynamic_(scores);

        // Summed in pair-index order, never in completion order, so the total is
        // reproducible bit-for-bit no matter which schedule ran or how the threads
        // interleaved.  Changing the schedule must not change the answer.
        return std::accumulate(scores.begin(), scores.end(), 0.0);
    }

    // Banded re-align + grad on every pair, around each pair's CACHED guide path,
    // using per-thread DpBuffers.  This is the re-alignment half of a training
    // loop: score_and_grad() once to establish guides, then set_params() +
    // banded_grad(bw) after each matrix update.  No full DP is run.
    // Returns sum of scores.  Throws if any pair has no guide yet.
    double banded_grad(int bandwidth) {
        const size_t N = pairs.size();
        if (N == 0) return 0.0;
        if (bandwidth <= 0)
            throw std::invalid_argument(
                "nwgrad: banded_grad() needs bandwidth > 0 (got " +
                std::to_string(bandwidth) + "); use score_and_grad() for full DP");
        std::vector<double> scores(N, 0.0);

        if (inter_fill)           banded_grad_inter_(scores, bandwidth);
        else if (sorted_schedule) banded_grad_lpt_(scores, bandwidth);
        else                      banded_grad_dynamic_(scores, bandwidth);

        return std::accumulate(scores.begin(), scores.end(), 0.0);
    }

private:
    // The vector backend an inter-pair fill of this pair would run on, or -1 when the
    // pair must take its own fill: an alphabet over 8 letters or the scalar backend.
    // float32 runs the float kernel (inter_fill_f), twice the lanes in the same register.  Soft pairs qualify: the shared fill is their guide Viterbi
    // (bit-identical to their own), and forward-backward then runs per pair. The mutable Hirschberg cutoff is
    // handled separately when building and validating the cached plan.
    // The inter-pair fill and its lane count at this batch's precision.
    static int inter_w_(const LevelKernels& K) {
        if constexpr (std::is_same_v<T, double>) return K.inter_fill ? K.inter_w : 0;
        else                                     return K.inter_fill_f ? K.inter_w_f : 0;
    }

    // Linear Global's own fill is cheap: the shared one loses below 4 lanes (DNA, W=2:
    // 0.76x on SSE2) and for alphabets over 8 (protein, the per-row gather: 1.10-1.22x
    // at 12 threads on AVX2).  Soft pairs still take the soft pass and skip the fill.
    static bool linear_fill_ok_(const SeqPair& p, const LevelKernels& K) {
        return !(p.gap_model() == GapModel::Linear && p.align_mode() == AlignMode::Global &&
                 (inter_w_(K) < 4 || p.params_ptr()->matrix.size() > 8));
    }

    int inter_backend_(const SeqPair& p) const {
        {
            if (p.len_a() == 0 || p.len_b() == 0) return -1;
            const int backend = (p.kernel() == kBackendAuto) ? global_default_backend() : p.kernel();
            if (backend < 0) return -1;
            const LevelKernels& K = level_kernels(backend);
            if (inter_w_(K) <= 0) return -1;
            if (!inter_pair_fits(p.len_a(), p.len_b(), p.gap_model() == GapModel::Affine,
                                 p.grad_mode() == GradMode::Soft, inter_w_(K), sizeof(T),
                                 K.inter_w))
                return -1;
            // The linear inter-pair VITERBI fill, Global, at 2 lanes is a measured LOSS
            // (skynet sse2: 0.76x the scalar fill, which is already cheap there); Local
            // wins at every width (1.41x sse2, 3.23x avx2) and Global from 4 lanes (1.41x
            // avx2).  NEON (W=2) unmeasured, so held to the same rule.  Soft pairs still
            // qualify — their soft pass wins — and skip the fill (linear_fill_ok_).
            if (!linear_fill_ok_(p, K) && p.grad_mode() != GradMode::Soft) return -1;
            return backend;
        }
    }

    // fill = "interpair": the qualifying pairs grouped W at a time by (params, backend,
    // alignment mode, length of B), sorted by length of A so a group's rows are nearly all used; each
    // group is one InterJob, then each lane's pair adopts its tables for score, path and
    // gradient.  The other pairs run their own fill.  One atomic counter over all tasks.
    void score_and_grad_inter_(std::vector<double>& scores) {
        // Reuse the grouping until pairs are added, the default ISA changes, or a
        // Hirschberg cutoff crosses the pair's length. Only potentially eligible
        // Hirschberg pairs need cutoff checks; a Pointers batch has none. Re-sorting
        // 2.5M pairs every training iteration was ~0.9 s of serial work per call.
        // Parameter identity is checked per group at run time instead.
        const int def_backend = global_default_backend();
        if (plan_.generation != generation_ || plan_.default_backend != def_backend ||
            plan_.n_pairs != pairs.size() || !inter_plan_cutoffs_match_())
            build_inter_plan_(def_backend);
        const auto& elig = plan_.elig;
        const auto& other = plan_.other;
        const auto& backend = plan_.backend;
        const auto& groups = plan_.groups;
        const size_t G = groups.size(), tasks = G + other.size();
        std::atomic<size_t> idx{0};
        auto worker = [&]() {
            DpBuffer buf;
            std::vector<const unsigned char*> a, b;
            std::vector<int> m, bi, bj, nb;
            std::vector<T> best, blkT;
            // Soft groups: the inter-pair soft pass's scratch and per-lane results.
            DVec sscr;
            std::vector<int> siscr, sok, sgp;
            std::vector<double> ses, slogz, scnt, sgap;
            while (true) {
                const size_t t = idx.fetch_add(1, std::memory_order_relaxed);
                if (t >= tasks) break;
                if (t >= G) {
                    const size_t i = other[t - G];
                    pairs[i]->score_and_grad_with_dp(buf);
                    scores[i] = pairs[i]->score();
                    continue;
                }
                const auto [s, e] = groups[t];
                const size_t real = e - s;
                const SeqPair& p0 = *pairs[elig[s]];
                bool same_params = true;
                for (size_t l = 1; l < real; ++l)
                    same_params &= pairs[elig[s + l]]->params_ptr() == p0.params_ptr();
                if (!same_params) {   // pairs given their own params: no shared fill
                    for (size_t l = 0; l < real; ++l) {
                        const size_t i = elig[s + l];
                        pairs[i]->score_and_grad_with_dp(buf);
                        scores[i] = pairs[i]->score();
                    }
                    continue;
                }
                const LevelKernels& K = level_kernels(backend[elig[s]]);
                const int W = inter_w_(K);
                a.assign(W, nullptr); b.assign(W, nullptr); m.assign(W, 0);
                bi.assign(W, 0); bj.assign(W, 0); best.assign(W, T(0));
                int M = 0, n = 0;
                bool ragged = false;
                nb.assign(W, 0);
                for (int l = 0; l < W; ++l) {
                    // Short group: the spare lanes repeat the last pair; their results are dropped.
                    const SeqPair& p = *pairs[elig[s + std::min<size_t>(l, real - 1)]];
                    a[l] = p.a_codes().data(); b[l] = p.b_codes().data();
                    m[l] = static_cast<int>(p.len_a());
                    nb[l] = static_cast<int>(p.len_b());
                    M = std::max(M, m[l]);
                    n = std::max(n, nb[l]);
                    ragged |= nb[l] != nb[0];
                }
                const size_t gstride = ragged ? static_cast<size_t>(n) + 1 : 0;
                const AlignParams& P = *p0.params_ptr();
                const bool lin = p0.gap_model() == GapModel::Linear;
                // A lazy-guide soft group needs no Viterbi at all: skip the shared fill.
                // So does a posterior-guide group (its guide comes from the soft pass).
                auto no_viterbi = [](const SeqPair& p) {
                    return p.grad_mode() == GradMode::Soft &&
                           (p.soft_guide_lazy() || p.soft_guide_posterior());
                };
                bool lazy = no_viterbi(p0);
                for (size_t l = 1; l < real && lazy; ++l) lazy = no_viterbi(*pairs[elig[s + l]]);
                bool want_post = false;
                for (size_t l = 0; l < real; ++l) want_post |= pairs[elig[s + l]]->soft_guide_posterior();
                const bool skip_fill = lazy || !linear_fill_ok_(p0, K);
                InterJobT<T> job{};
                job.a = a.data(); job.m = m.data(); job.b = b.data(); job.n = n; job.M = M;
                job.align_mode = (p0.align_mode() == AlignMode::Local) ? 1 : 0;
                if (ragged) job.nb = nb.data();
                if (!skip_fill) {
                    const size_t sz = static_cast<size_t>(M + 1) * (n + 1) * W;
                    // Linear's one table rides in the H slot (adopt_interleaved reads buf.H).
                    if (lin) { if (buf.H.size() < sz) buf.H.resize(sz); }
                    else if (buf.VM.size() < sz) { buf.VM.resize(sz); buf.VX.resize(sz); buf.VY.resize(sz); }
                    // In T, as the pair's own fill rounds them (blkT_, the cast penalties).
                    const size_t nb = static_cast<size_t>(P.matrix.size()) * P.matrix.size();
                    blkT.resize(nb);
                    for (size_t k = 0; k < nb; ++k) blkT[k] = static_cast<T>(P.matrix.data()[k]);
                    job.blk = blkT.data(); job.nalpha = P.matrix.size();
                    job.go_a = static_cast<T>(P.gap_open_a); job.ge_a = static_cast<T>(P.gap_extend_a);
                    job.go_b = static_cast<T>(P.gap_open_b); job.ge_b = static_cast<T>(P.gap_extend_b);
                    job.linear = lin ? 1 : 0;
                    if (lin) job.VM = buf.H.data();
                    else { job.VM = buf.VM.data(); job.VX = buf.VX.data(); job.VY = buf.VY.data(); }
                    job.best = best.data(); job.best_i = bi.data(); job.best_j = bj.data();
                    if constexpr (std::is_same_v<T, double>) K.inter_fill(job);
                    else                                     K.inter_fill_f(job);
                }
                // Soft lanes share the forward-backward too when every real lane is soft
                // with one soft_impl (not "log") and one temperature, and the weights fit.
                bool soft_group = K.inter_soft != nullptr && p0.grad_mode() == GradMode::Soft &&
                                  p0.soft_impl() != SoftImpl::Log;
                for (size_t l = 1; l < real && soft_group; ++l) {
                    const SeqPair& p = *pairs[elig[s + l]];
                    soft_group = p.grad_mode() == GradMode::Soft && p.soft_impl() == p0.soft_impl() &&
                                 p.soft_temperature() == p0.soft_temperature();
                }
                const int na = P.matrix.size();
                const size_t nn = static_cast<size_t>(na) * na;
                if (soft_group) {
                    InterSoftJob sj{};
                    const bool fin = inter_soft_weights(P, p0.soft_temperature(), lin, ses, sj);
                    soft_group = fin;
                    if (fin) {
                        // The soft pass is double (the forward-backward is double at any
                        // T), K.inter_w lanes: a float32 group runs it in W / SW chunks.
                        const int SW = K.inter_w;
                        slogz.resize(W); scnt.resize(W * nn); sgap.resize(W * 4); sok.assign(W, 0);
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
                            sj.logz = slogz.data() + c; sj.counts = scnt.data() + c * nn;
                            sj.gaps = sgap.data() + c * 4; sj.ok = sok.data() + c;
                            // chunk-local layout: row i, lane l at [i * SW + l]
                            sj.gpost = want_post ? sgp.data() + static_cast<size_t>(c) * (M + 1) : nullptr;
                            K.inter_soft(sj);
                        }
                    }
                }
                for (size_t l = 0; l < real; ++l) {
                    const size_t i = elig[s + l];
                    typename SeqPair::SoftLane lane{};
                    const bool use = soft_group && sok[l];
                    if (use) {
                        lane = {slogz[l], scnt.data() + l * nn, sgap.data() + l * 4};
                        if (want_post) {
                            const int SW = K.inter_w, c = static_cast<int>(l) / SW * SW;
                            lane.gpost = sgp.data() + static_cast<size_t>(c) * (M + 1) + (l - c);
                            lane.gstride = SW;
                        }
                    }
                    if (skip_fill) pairs[i]->score_and_grad_with_soft_lane(buf, use ? &lane : nullptr);
                    else     pairs[i]->score_and_grad_interleaved(buf, W, static_cast<int>(l),
                                                                  best[l], bi[l], bj[l],
                                                                  use ? &lane : nullptr, gstride);
                    scores[i] = pairs[i]->score();
                }
            }
        };
        run_workers(tasks, worker);
    }

    // banded_grad() under fill = "interpair": the same groups as score_and_grad(), each
    // a GuideBanded InterJob — every lane around its own guide, its tables bit-identical
    // to its own banded fill (see InterJobT::blo).  Hard affine pairs only: a group with
    // a soft lane (banded forward-backward), mixed params or linear gaps runs each pair's
    // own banded path, as do the pairs outside the plan.  Measured (nighthaven AVX2,
    // Manakov 100k, bw 2-8, 1 thread): affine 0.45-0.89x the own path's time, linear
    // 0.60-1.27x — hence affine only.  At 12 threads affine is ~1.0x (per-pair setup and
    // the traceback walk, not the fill, are most of a banded step on 22 x 50 pairs).
    void banded_grad_inter_(std::vector<double>& scores, int bandwidth) {
        const int def_backend = global_default_backend();
        if (plan_.generation != generation_ || plan_.default_backend != def_backend ||
            plan_.n_pairs != pairs.size() || !inter_plan_cutoffs_match_())
            build_inter_plan_(def_backend);
        if (band_.plan_gen != plan_.generation || band_.plan_n != plan_.n_pairs ||
            band_.full_calls != full_calls_ || band_.default_backend != def_backend)
            build_band_groups_();
        const auto& elig = band_.elig;
        const auto& other = band_.other;
        const auto& backend = plan_.backend;
        const auto& groups = band_.groups;
        const size_t G = groups.size(), tasks = G + other.size();
        std::atomic<size_t> idx{0};
        auto worker = [&]() {
            DpBuffer buf;
            std::vector<const unsigned char*> a, b;
            std::vector<int> m, bi, bj, blo, bhi, ulo, uhi, bri, brj;
            std::vector<T> best, blkT;
            auto own = [&](size_t i) {
                pairs[i]->banded_grad_with_dp(buf, bandwidth);
                scores[i] = pairs[i]->score();
            };
            while (true) {
                const size_t t = idx.fetch_add(1, std::memory_order_relaxed);
                if (t >= tasks) break;
                if (t >= G) { own(other[t - G]); continue; }
                const auto [s, e] = groups[t];
                const size_t real = e - s;
                const SeqPair& p0 = *pairs[elig[s]];
                const LevelKernels& K = level_kernels(backend[elig[s]]);
                // Affine only: linear's own banded fill is already cheap and the shared
                // one measured 0.83-1.27x of it on AVX2 (a loss at narrow bands).
                bool shared = p0.gap_model() == GapModel::Affine;
                for (size_t l = 0; l < real && shared; ++l) {
                    const SeqPair& p = *pairs[elig[s + l]];
                    shared = p.params_ptr() == p0.params_ptr() && p.grad_mode() != GradMode::Soft;
                }
                if (!shared) { for (size_t l = 0; l < real; ++l) own(elig[s + l]); continue; }
                const int W = inter_w_(K);
                int M = 0;
                for (size_t l = 0; l < real; ++l)
                    M = std::max(M, static_cast<int>(pairs[elig[s + l]]->len_a()));
                const int n = static_cast<int>(p0.len_b());
                a.assign(W, nullptr); b.assign(W, nullptr); m.assign(W, 0);
                bi.assign(W, 0); bj.assign(W, 0); best.assign(W, T(0));
                bri.assign(W, 0); brj.assign(W, 0);
                blo.assign(static_cast<size_t>(M + 1) * W, 1);
                bhi.assign(static_cast<size_t>(M + 1) * W, 0);
                ulo.assign(M + 1, n + 1); uhi.assign(M + 1, 0);
                for (int l = 0; l < W; ++l) {
                    // Spare lanes: the last pair's codes, no rows (m = 0), empty bands.
                    SeqPair& p = *pairs[elig[s + std::min<size_t>(l, real - 1)]];
                    a[l] = p.a_codes().data(); b[l] = p.b_codes().data();
                    if (static_cast<size_t>(l) >= real) continue;
                    m[l] = static_cast<int>(p.len_a());
                    p.banded_lane_setup(bandwidth, W, l, M, blo.data(), bhi.data(),
                                        ulo.data(), uhi.data(), bri[l], brj[l]);
                }
                const AlignParams& P = *p0.params_ptr();
                const bool lin = p0.gap_model() == GapModel::Linear;
                const size_t sz = static_cast<size_t>(M + 1) * (n + 1) * W;
                if (lin) { if (buf.H.size() < sz) buf.H.resize(sz); }
                else if (buf.VM.size() < sz) { buf.VM.resize(sz); buf.VX.resize(sz); buf.VY.resize(sz); }
                InterJobT<T> job{};
                job.a = a.data(); job.m = m.data(); job.b = b.data(); job.n = n; job.M = M;
                job.align_mode = (p0.align_mode() == AlignMode::Local) ? 1 : 0;
                const size_t nb = static_cast<size_t>(P.matrix.size()) * P.matrix.size();
                blkT.resize(nb);
                for (size_t k = 0; k < nb; ++k) blkT[k] = static_cast<T>(P.matrix.data()[k]);
                job.blk = blkT.data(); job.nalpha = P.matrix.size();
                job.go_a = static_cast<T>(P.gap_open_a); job.ge_a = static_cast<T>(P.gap_extend_a);
                job.go_b = static_cast<T>(P.gap_open_b); job.ge_b = static_cast<T>(P.gap_extend_b);
                job.linear = lin ? 1 : 0;
                if (lin) job.VM = buf.H.data();
                else { job.VM = buf.VM.data(); job.VX = buf.VX.data(); job.VY = buf.VY.data(); }
                job.best = best.data(); job.best_i = bi.data(); job.best_j = bj.data();
                job.blo = blo.data(); job.bhi = bhi.data();
                job.ulo = ulo.data(); job.uhi = uhi.data();
                job.bri = bri.data(); job.brj = brj.data();
                if constexpr (std::is_same_v<T, double>) K.inter_fill(job);
                else                                     K.inter_fill_f(job);
                for (size_t l = 0; l < real; ++l) {
                    const size_t i = elig[s + l];
                    pairs[i]->banded_grad_interleaved(buf, W, static_cast<int>(l),
                                                      best[l], bi[l], bj[l]);
                    scores[i] = pairs[i]->score();
                }
            }
        };
        run_workers(tasks, worker);
    }

    // The banded grouping: the plan's eligible pairs, within each run of equal (backend,
    // gap model, mode, len B), re-sorted by where the guide runs (its column at the
    // middle row of A, then len A) — lanes whose bands overlap make the union each row
    // computes narrow (Manakov local, W=4: 79 vector cells per pair against 169 sorted by
    // len A).  Speed only: any grouping is bit-exact.  Rebuilt when the plan is, or after
    // a score_and_grad() (which recomputes every guide); banded_grad() steps reuse it.
    struct BandGroups {
        size_t plan_gen = static_cast<size_t>(-1), plan_n = 0, full_calls = static_cast<size_t>(-1);
        int default_backend = -1000;
        std::vector<size_t> elig, other;
        std::vector<std::pair<size_t, size_t>> groups;
    };
    BandGroups band_;
    size_t full_calls_ = 0;

    void build_band_groups_() {
        BandGroups Bg;
        Bg.plan_gen = plan_.generation; Bg.plan_n = plan_.n_pairs;
        Bg.full_calls = full_calls_; Bg.default_backend = plan_.default_backend;
        // Only hard affine pairs can share a banded fill; the rest (linear, soft) run
        // their own, in PAIR order with the plan's ineligible ones — leaving them in the
        // guide-sorted groups walked the pair objects in a permuted order, 5-10 % slower
        // than the plain per-pair path (measured, linear, AVX2).
        std::vector<size_t> E;
        E.reserve(plan_.elig.size());
        Bg.other = plan_.other;
        for (size_t i : plan_.elig) {
            const SeqPair& p = *pairs[i];
            if (p.gap_model() == GapModel::Affine && p.grad_mode() != GradMode::Soft) E.push_back(i);
            else Bg.other.push_back(i);
        }
        std::sort(Bg.other.begin(), Bg.other.end());
        Bg.elig = E;
        struct Key { int mid, len_a; size_t i; };
        std::vector<Key> keys;
        for (size_t s = 0; s < E.size();) {
            const SeqPair& p0 = *pairs[E[s]];
            const int be = plan_.backend[E[s]];
            size_t e = s + 1;
            while (e < E.size()) {
                const SeqPair& p = *pairs[E[e]];
                if (plan_.backend[E[e]] != be || p.gap_model() != p0.gap_model() ||
                    p.align_mode() != p0.align_mode() || p.len_b() != p0.len_b()) break;
                ++e;
            }
            keys.clear();
            for (size_t k = s; k < e; ++k) {
                const SeqPair& p = *pairs[E[k]];
                const auto& g = p.guide_j_raw();
                const int la = static_cast<int>(p.len_a());
                keys.push_back({g.size() > static_cast<size_t>(la / 2) ? g[la / 2] : 0, la, E[k]});
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

public:
    // Ragged B: one group may mix B lengths (InterJobT::nb) — every lane is padded to the
    // longest, so a group admits a pair only while that padding stays small.
    static bool ragged_ok_(size_t shortest, size_t len_b) {
        return len_b * 4 <= shortest * 5 + 16;   // <= 1.25x + 4 columns
    }
private:

    struct InterPlan {
        size_t generation = static_cast<size_t>(-1);
        int default_backend = -1000;
        size_t n_pairs = 0;
        std::vector<size_t> elig, other;
        std::vector<int> backend;
        std::vector<std::pair<size_t, size_t>> groups;   // [start, end) into elig
        // Pair index and whether its cutoff admitted the shared fill. Track both
        // eligible and splitting pairs, so lowering OR raising the cutoff works.
        std::vector<std::pair<size_t, bool>> hirschberg_cutoffs;
    };
    InterPlan plan_;
    size_t generation_ = 0;   // bumped by add() / add_many(); keys plan_

    bool inter_plan_cutoffs_match_() const {
        for (const auto& [i, eligible] : plan_.hirschberg_cutoffs) {
            const SeqPair& p = *pairs[i];
            if ((p.len_a() <= static_cast<size_t>(p.hb_cutoff())) != eligible) return false;
        }
        return true;
    }

    void build_inter_plan_(int def_backend) {
        const size_t N = pairs.size();
        InterPlan P;
        P.generation = generation_; P.default_backend = def_backend; P.n_pairs = N;
        P.backend.assign(N, -1);
        // Sort keys gathered once, so the sort does not chase pair pointers.
        struct Key { int backend, gm, mode; size_t len_b, len_a, i; };
        std::vector<Key> keys;
        keys.reserve(N);
        for (size_t i = 0; i < N; ++i) {
            const SeqPair& p = *pairs[i];
            P.backend[i] = inter_backend_(p);
            if (P.backend[i] >= 0 && is_hirschberg(p.traceback())) {
                const bool eligible = p.len_a() <= static_cast<size_t>(p.hb_cutoff());
                P.hirschberg_cutoffs.emplace_back(i, eligible);
                if (!eligible) P.backend[i] = -1;
            }
            if (P.backend[i] < 0) { P.other.push_back(i); continue; }
            keys.push_back({P.backend[i], static_cast<int>(p.gap_model()),
                            static_cast<int>(p.align_mode()), p.len_b(), p.len_a(), i});
        }
        std::sort(keys.begin(), keys.end(), [](const Key& x, const Key& y) {
            if (x.backend != y.backend) return x.backend < y.backend;
            if (x.gm != y.gm) return x.gm < y.gm;
            if (x.mode != y.mode) return x.mode < y.mode;
            if (x.len_b != y.len_b) return x.len_b < y.len_b;
            if (x.len_a != y.len_a) return x.len_a < y.len_a;
            return x.i < y.i;
        });
        P.elig.reserve(keys.size());
        for (const Key& k : keys) P.elig.push_back(k.i);
        // Groups: runs of equal (backend, gap model, mode), cut every W — and wherever
        // len B outgrows the group's first (shortest) by more than ragged_ok_ allows.
        for (size_t s = 0; s < keys.size();) {
            const int W = inter_w_(level_kernels(keys[s].backend));
            size_t e = s + 1;
            while (e < keys.size() && e - s < static_cast<size_t>(W) &&
                   keys[e].backend == keys[s].backend && keys[e].gm == keys[s].gm &&
                   keys[e].mode == keys[s].mode &&
                   ragged_ok_(keys[s].len_b, keys[e].len_b))
                ++e;
            P.groups.emplace_back(s, e);
            s = e;
        }
        plan_ = std::move(P);
    }

    // The original: one atomic counter, tasks in insertion order.
    void score_and_grad_dynamic_(std::vector<double>& scores) {
        const size_t N = pairs.size();
        std::atomic<size_t> idx{0};
        auto worker = [&]() {
            DpBuffer buf;
            while (true) {
                size_t i = idx.fetch_add(1, std::memory_order_relaxed);
                if (i >= N) break;
                pairs[i]->score_and_grad_with_dp(buf);
                scores[i] = pairs[i]->score();
            }
        };
        run_workers(N, worker);
    }

    // Length-sorted, equal-work chunks, one chunk per thread, plus a shared
    // reserve of the smallest tasks.
    //
    // The problem this solves is MEMORY, and it is not a load-balancing problem.
    // Under the dynamic schedule every thread eventually draws a long sequence, so
    // every thread's DpBuffer grows to the global maximum and stays there: the
    // high-water mark is n_threads * 24 * (Lmax+1)^2 regardless of how few long
    // sequences there are.  On the human proteome at 16 threads that is 30.6 GB,
    // and at 60 threads 114.7 GB -- which is why the thread count was a memory
    // decision rather than a throughput one.
    //
    // Sorting alone does not fix it, in either direction: descending order hands
    // the T largest tasks out first, ascending order converges every thread on the
    // giants at the end.  Bounding the footprint requires *affinity* between a
    // thread and a size class, which is what the chunking below is for.  Memory
    // becomes sum over chunks of 24*(chunk max +1)^2 -- 4.9 GB and 16.1 GB for the
    // two cases above -- because equal-WORK chunks drawn from a long tail contain
    // very few long sequences (the top 1% of the proteome by length holds 25% of
    // all cells).
    //
    // The cost model here is deliberately the plain square, (m+1)*(n+1) cells,
    // with NO correction for the measured per-cell cost curve.  That curve is real
    // -- a cell in a 1000 aa alignment costs ~6.5x a cell in a 300 aa one, because
    // the tables stop fitting cache and the kernel stalls (IPC 2.41 -> 0.38 at
    // constant instruction count) -- so these chunks are equal in cells but NOT in
    // time, and the long-sequence chunks overrun. The reserve absorbs some of it by
    // construction, since the chunk that finishes early is the small-sequence one
    // and the reserve is exactly what it can consume. Correcting the model properly
    // needs a startup calibration, because the curve depends on thread count and
    // host, not on the task alone.
    // One scheduled pass: sort by `cost`, hold back a reserve of the cheapest
    // tasks, cut equal-work chunks, and run task(index, buf) with per-thread
    // buffers.  Factored out because banded work needs TWO passes over the same
    // pairs under two different cost models (see below).
    template<typename Fn>
    void run_sorted_phase_(const std::vector<double>& cost, int nthr, Fn&& task) {
        const size_t N = cost.size();

        std::vector<size_t> order(N);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
                  [&](size_t a, size_t b) { return cost[a] < cost[b]; });

        const double total = std::accumulate(cost.begin(), cost.end(), 0.0);

        // Reserve: the cheapest tasks, taken off the front, until reserve_frac of
        // the total work is held back.  Capped so it can never swallow a chunk.
        size_t r = 0;
        {
            const double want = total * std::clamp(reserve_frac, 0.0, 0.5);
            double acc = 0.0;
            while (r < N && acc < want && r + static_cast<size_t>(nthr) < N)
                acc += cost[order[r++]];
        }

        double chunk_total = 0.0;
        for (size_t i = r; i < N; ++i) chunk_total += cost[order[i]];

        // Equal-work contiguous chunks over the remaining sorted order.
        std::vector<size_t> bound(static_cast<size_t>(nthr) + 1, r);
        {
            double acc = 0.0;
            size_t k = 1;
            for (size_t i = r; i < N && k < static_cast<size_t>(nthr); ++i) {
                acc += cost[order[i]];
                if (acc >= chunk_total * static_cast<double>(k) /
                           static_cast<double>(nthr))
                    bound[k++] = i + 1;
            }
            while (k <= static_cast<size_t>(nthr)) bound[k++] = N;
        }
        bound[static_cast<size_t>(nthr)] = N;

        std::atomic<int>    next_worker{0};
        std::atomic<size_t> reserve_idx{0};

        using clk = std::chrono::steady_clock;
        const bool prof = profile;
        if (prof) { profile_out.assign(static_cast<size_t>(nthr), PhaseProfile{}); }
        const auto t_start = clk::now();
        auto since = [&](clk::time_point a, clk::time_point b) {
            return std::chrono::duration<double>(b - a).count();
        };

        auto worker = [&]() {
            const int k = next_worker.fetch_add(1, std::memory_order_relaxed);
            DpBuffer buf;
            PhaseProfile pp;

            const auto t0 = clk::now();
            if (k < nthr) {
                // Descending within the chunk: the buffer reaches its high-water
                // mark on the first task and never reallocates afterwards.
                for (size_t p = bound[static_cast<size_t>(k) + 1];
                     p > bound[static_cast<size_t>(k)]; --p) {
                    task(order[p - 1], buf);
                    if (prof) { pp.chunk_cells += cost[order[p - 1]]; ++pp.chunk_tasks; }
                }
            }
            const auto t1 = clk::now();
            // Finished early -> drain the reserve.  Largest reserve task first,
            // so the last thing any thread picks up is the cheapest work there is.
            while (true) {
                size_t s = reserve_idx.fetch_add(1, std::memory_order_relaxed);
                if (s >= r) break;
                task(order[r - 1 - s], buf);
                if (prof) { pp.reserve_cells += cost[order[r - 1 - s]]; ++pp.reserve_tasks; }
            }
            // Sweep up chunks no worker claimed.  Normally there are none: nthr
            // workers start and each claims its own chunk first.  But a thread
            // launch can fail (run_workers_guarded then carries on with fewer
            // workers), and chunk k would otherwise go unprocessed — a silently
            // short sum.  A late-starting worker whose chunk was taken here gets
            // k >= nthr, skips its chunk and finds the reserve drained: correct
            // either way.
            for (int k2; (k2 = next_worker.fetch_add(1, std::memory_order_relaxed)) < nthr; ) {
                for (size_t p = bound[static_cast<size_t>(k2) + 1];
                     p > bound[static_cast<size_t>(k2)]; --p) {
                    task(order[p - 1], buf);
                    if (prof) { pp.chunk_cells += cost[order[p - 1]]; ++pp.chunk_tasks; }
                }
            }
            const auto t2 = clk::now();
            if (prof && k >= 0 && static_cast<size_t>(k) < profile_out.size()) {
                pp.chunk_s   = since(t0, t1);
                pp.reserve_s = since(t1, t2);
                pp.finish_s  = since(t_start, t2);
                profile_out[static_cast<size_t>(k)] = pp;
            }
        };

        run_workers(N, worker);
    }

    void score_and_grad_sorted_(std::vector<double>& scores) {
        const size_t N = pairs.size();
        const int    nthr = std::min<int>(n_threads, static_cast<int>(N));

        std::vector<double> cost(N);
        for (size_t i = 0; i < N; ++i) {
            const double cells = static_cast<double>(pairs[i]->len_a() + 1) *
                                 static_cast<double>(pairs[i]->len_b() + 1);
            cost[i] = cells * length_weight(cells);
        }

        run_sorted_phase_(cost, nthr, [&](size_t i, DpBuffer& buf) {
            pairs[i]->score_and_grad_with_dp(buf);
            scores[i] = pairs[i]->score();
        });
    }

    void banded_grad_dynamic_(std::vector<double>& scores, int bandwidth) {
        const size_t N = pairs.size();
        std::atomic<size_t> idx{0};
        auto worker = [&]() {
            DpBuffer buf;
            while (true) {
                size_t i = idx.fetch_add(1, std::memory_order_relaxed);
                if (i >= N) break;
                pairs[i]->banded_grad_with_dp(buf, bandwidth);
                scores[i] = pairs[i]->score();
            }
        };
        run_workers(N, worker);
    }

    // The banded scheduler.  DELIBERATELY NOT the chunked, memory-affine one
    // score_and_grad() uses -- banded work has a different shape in both of the
    // dimensions that motivated that design, so copying it would be cargo cult.
    //
    //   Memory: a banded task needs (m+1)*(2*bandwidth+1) cells, not (m+1)(n+1).
    //   The longest human protein at bandwidth 32 is ~7 MB of tables against
    //   ~1 GB for its full DP.  Every thread can hold the worst case at once, so
    //   there is nothing for per-thread size affinity to save, and the chunking
    //   that buys it -- and the reserve that patches the imbalance chunking
    //   creates -- would be pure overhead.
    //
    //   Balance: cost is LINEAR in length here, not quadratic, so the spread
    //   between the cheapest and dearest task is far narrower and a plain atomic
    //   counter already balances well.  The one thing it does badly is drawing a
    //   long task last, with no work left to overlap it.
    //
    // So: longest-processing-time-first (LPT) over a plain atomic counter.  Sort
    // descending, dispatch dynamically.  That is the classic fix for exactly that
    // tail (makespan <= 4/3 optimal), and it costs one sort, no barrier, no
    // chunks, no reserve.  It is strictly less machinery than the full-DP
    // scheduler, not more, because the problem is genuinely easier.
    void banded_grad_lpt_(std::vector<double>& scores, int bandwidth) {
        const size_t N = pairs.size();
        const double bw = 2.0 * static_cast<double>(bandwidth) + 1.0;

        std::vector<double> cost(N);
        for (size_t i = 0; i < N; ++i)
            cost[i] = static_cast<double>(pairs[i]->len_a() + 1) *
                      std::min(static_cast<double>(pairs[i]->len_b() + 1), bw);

        std::vector<size_t> order(N);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
                  [&](size_t a, size_t b) { return cost[a] > cost[b]; });

        std::atomic<size_t> idx{0};
        auto worker = [&]() {
            DpBuffer buf;
            while (true) {
                size_t k = idx.fetch_add(1, std::memory_order_relaxed);
                if (k >= N) break;
                const size_t i = order[k];
                pairs[i]->banded_grad_with_dp(buf, bandwidth);
                scores[i] = pairs[i]->score();
            }
        };
        run_workers(N, worker);
    }

    // Pairs built by add_many(), owned by the batch.  `pairs` holds raw pointers
    // into these; unique_ptr keeps the addresses stable as the vector grows.
    std::vector<std::unique_ptr<SeqPair>> owned_;

    // Physical cores, not logical -- see parallel.hpp::physical_cores() for the
    // measurements.  hardware_concurrency() costs up to 1.44x on an SMT host.
    TracebackMode tb_ = TracebackMode::Default;

    // False after an add() until check_unique_() has verified that no SeqPair
    // repeats.  True for an empty batch and after add_many(), which cannot repeat.
    bool unique_checked_ = true;

public:
    // Hirschberg base-case size in rows, applied to pairs built by add_many().  Ignored
    // unless traceback resolves to Hirschberg.  512 from a fleet sweep AFTER hb_base was
    // vectorized; see the Aligner field for the mechanism (pairs <= cutoff run the exact
    // pointers fill; longer ones split).
    int hb_cutoff = 512;

    // Full-band fill for pairs built by add_many(): striped (false) or row-wise (true).
    // See Aligner::set_rowwise_full.  Setting it through the Python `fill` property
    // also applies it to the pairs already in the batch.
    bool rowwise_full = false;

    // Soft-path evaluation for pairs built by add_many() — see SoftImpl.  The Python
    // `soft_impl` property also applies it to the pairs already in the batch.
    SoftImpl soft_impl = SoftImpl::Scaled;
    // Soft temperature for pairs built by add_many() — see Aligner::set_soft_temperature.
    double soft_temperature = 1.0;
    // Soft guide policy for pairs built by add_many() — see SeqPair::set_soft_guide_lazy.
    bool soft_guide_lazy = false;
    bool soft_guide_posterior = false;   // see SeqPair::set_soft_guide_posterior

    // score_and_grad() and banded_grad() fill W pairs at once, one per vector lane
    // (InterJobT), wherever a pair qualifies (inter_backend_); the rest run their own
    // fill.  Bit-identical results either way.  The DEFAULT since 2026-10-05: measured
    // 0.2-0.5x striped's time on short pairs (AVX2), the first call's plan included;
    // the size cap keeps long pairs on their own path.  Python: fill = "interpair".
    bool inter_fill = true;

private:

    static int default_threads() noexcept { return default_thread_count(); }

    // Dispatch a lambda(size_t index) over [0, N) with per-index atomics.
    template<typename Fn>
    void parallel_for(size_t N, Fn&& fn) {
        std::atomic<size_t> idx{0};
        auto worker = [&]() {
            while (true) {
                size_t i = idx.fetch_add(1, std::memory_order_relaxed);
                if (i >= N) break;
                fn(i);
            }
        };
        run_workers(N, worker);
    }

    // Every SeqPair in the batch must be distinct.  Workers align pairs[i] and
    // pairs[j] concurrently; if both are one object, two threads write its DP tables
    // and cached state at once — measured: wrong totals (some above the optimum),
    // segfaults and hangs, with no error raised.  Even on one thread a duplicate
    // double-counts the pair in every sum.  So it is rejected, but LAZILY: add()
    // only marks the batch unchecked, and the next dispatch sorts a copy of the
    // pointers once and caches the verdict until the next add().  O(N log N) once
    // per batch change (~150 ms at 2.5M pairs) and nothing per training step, where
    // an eager hash set would hold ~40 B per pair for the batch's whole life.
    // add_many() pairs are freshly allocated and cannot repeat each other or an
    // existing pair, so add_many() leaves the verdict alone; re-adding one of them
    // with add() is caught like any other duplicate.  (`pairs` is public: code that
    // pushes into it directly bypasses this check, as it bypasses add()'s.)
    void check_unique_() {
        if (unique_checked_) return;
        std::vector<const SeqPair*> sorted(pairs.begin(), pairs.end());
        std::sort(sorted.begin(), sorted.end());
        auto dup = std::adjacent_find(sorted.begin(), sorted.end());
        if (dup != sorted.end()) {
            std::vector<size_t> where;
            for (size_t i = 0; i < pairs.size() && where.size() < 2; ++i)
                if (pairs[i] == *dup) where.push_back(i);
            throw std::invalid_argument(
                "nwgrad: the same SeqPair is in this batch more than once (at "
                "indices " + std::to_string(where[0]) + " and " +
                std::to_string(where[1]) + "); workers would align it "
                "concurrently.  Add each pair once.");
        }
        unique_checked_ = true;
    }

    template<typename Worker>
    void run_workers(size_t N, Worker& worker) {
        check_unique_();
        if (N == 0) return;
        int actual = std::min<int>(n_threads, static_cast<int>(N));
        run_workers_guarded(actual, worker);
    }
};

// Default (double) alias; Python binds SeqPairBatchT<float> as `SeqPairBatch`.
using SeqPairBatch = SeqPairBatchT<double>;

} // namespace legacy
