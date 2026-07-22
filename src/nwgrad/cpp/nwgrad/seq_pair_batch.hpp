#pragma once

#include <algorithm>
#include <atomic>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <vector>

#include "align_params.hpp"
#include "parallel.hpp"
#include "seq_pair.hpp"

// SeqPairBatch holds non-owning pointers to SeqPair objects added via add().
// On the Python side, nb::keep_alive ensures each added SeqPair outlives the batch.
//
// Pairs created by add_many() are the exception: the batch owns those outright
// (see owned_), because they never exist as Python objects at all.
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

    explicit SeqPairBatchT(int nt = 0)
        : n_threads(nt > 0 ? nt : default_threads()) {}

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
                                                      params, gm, am, gd, kernel);
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
    void set_params(const AlignParams& params) {
        for (auto& sp : pairs) sp->set_params(params);
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

    // Full-pipeline batch operation using per-thread DpBuffers.
    // For each pair: runs the full DP, computes grad,
    // stores score + guide_j + grad into the SeqPair.  The pairs' own DP tables
    // are never allocated; dp_valid() remains false after this call.
    // Returns sum of scores.
    double score_and_grad() {
        const size_t N = pairs.size();
        if (N == 0) return 0.0;
        std::vector<double> scores(N, 0.0);

        if (sorted_schedule) score_and_grad_sorted_(scores);
        else                 score_and_grad_dynamic_(scores);

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

        if (sorted_schedule) banded_grad_lpt_(scores, bandwidth);
        else                 banded_grad_dynamic_(scores, bandwidth);

        return std::accumulate(scores.begin(), scores.end(), 0.0);
    }

private:
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

        auto worker = [&]() {
            const int k = next_worker.fetch_add(1, std::memory_order_relaxed);
            DpBuffer buf;
            if (k < nthr) {
                // Descending within the chunk: the buffer reaches its high-water
                // mark on the first task and never reallocates afterwards.
                for (size_t p = bound[static_cast<size_t>(k) + 1];
                     p > bound[static_cast<size_t>(k)]; --p)
                    task(order[p - 1], buf);
            }
            // Finished early -> drain the reserve.  Largest reserve task first,
            // so the last thing any thread picks up is the cheapest work there is.
            while (true) {
                size_t s = reserve_idx.fetch_add(1, std::memory_order_relaxed);
                if (s >= r) break;
                task(order[r - 1 - s], buf);
            }
        };

        run_workers(N, worker);
    }

    void score_and_grad_sorted_(std::vector<double>& scores) {
        const size_t N = pairs.size();
        const int    nthr = std::min<int>(n_threads, static_cast<int>(N));

        std::vector<double> cost(N);
        for (size_t i = 0; i < N; ++i)
            cost[i] = static_cast<double>(pairs[i]->len_a() + 1) *
                      static_cast<double>(pairs[i]->len_b() + 1);

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

    template<typename Worker>
    void run_workers(size_t N, Worker& worker) {
        if (N == 0) return;
        int actual = std::min<int>(n_threads, static_cast<int>(N));
        run_workers_guarded(actual, worker);
    }
};

// Default (double) alias; Python binds SeqPairBatchT<float> as `SeqPairBatch`.
using SeqPairBatch = SeqPairBatchT<double>;
