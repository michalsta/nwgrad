#pragma once

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
struct SeqPairBatch {
    std::vector<SeqPair*> pairs;
    int n_threads;

    explicit SeqPairBatch(int nt = 0)
        : n_threads(nt > 0 ? nt : default_threads()) {}

    // Non-copyable.  It never was, meaningfully — a copy would duplicate the raw
    // pointers in `pairs` and alias every borrowed SeqPair.  But it has to be
    // *said*, not merely true: std::vector<unique_ptr<T>> still advertises a copy
    // constructor (declared, ill-formed only if instantiated), so
    // is_copy_constructible_v<SeqPairBatch> stayed true and nanobind emitted a
    // copy thunk for it, which failed to compile inside the STL.  Moves are fine:
    // unique_ptr keeps every SeqPair at a fixed address, so `pairs` stays valid.
    SeqPairBatch(const SeqPairBatch&)            = delete;
    SeqPairBatch& operator=(const SeqPairBatch&) = delete;
    SeqPairBatch(SeqPairBatch&&)                 = default;
    SeqPairBatch& operator=(SeqPairBatch&&)      = default;

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
                  GapModel gm, AlignMode am, GradMode gd) {
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
                                                      params, gm, am, gd);
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
    // For each pair: runs full DP (+ banded DP if bandwidth > 0), computes grad,
    // stores score + guide_j + grad into the SeqPair.  The pairs' own DP tables
    // are never allocated; dp_valid() remains false after this call.
    // Returns sum of scores.
    double score_and_grad(int bandwidth = 0) {
        const size_t N = pairs.size();
        std::vector<double> scores(N, 0.0);
        std::atomic<size_t> idx{0};

        auto worker = [&]() {
            DpBuffer buf;
            while (true) {
                size_t i = idx.fetch_add(1, std::memory_order_relaxed);
                if (i >= N) break;
                pairs[i]->score_and_grad_with_dp(buf, bandwidth);
                scores[i] = pairs[i]->score();
            }
        };

        run_workers(N, worker);
        return std::accumulate(scores.begin(), scores.end(), 0.0);
    }

private:
    // Pairs built by add_many(), owned by the batch.  `pairs` holds raw pointers
    // into these; unique_ptr keeps the addresses stable as the vector grows.
    std::vector<std::unique_ptr<SeqPair>> owned_;

    static int default_threads() noexcept {
        unsigned int hw = std::thread::hardware_concurrency();
        return hw > 0 ? static_cast<int>(hw) : 1;
    }

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
