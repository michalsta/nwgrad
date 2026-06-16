#pragma once

#include <atomic>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <vector>

#include "align_params.hpp"
#include "seq_pair.hpp"

// SeqPairBatch holds non-owning pointers to SeqPair objects.
// On the Python side, nb::keep_alive ensures each added SeqPair outlives the batch.
struct SeqPairBatch {
    std::vector<SeqPair*> pairs;
    int n_threads;

    explicit SeqPairBatch(int nt = 0)
        : n_threads(nt > 0 ? nt : default_threads()) {}

    void add(SeqPair* sp) {
        pairs.push_back(sp);
    }

    size_t size() const noexcept { return pairs.size(); }

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
    AlignParams compute_grad() {
        const size_t N = pairs.size();
        std::atomic<size_t> idx{0};
        std::mutex grad_mutex;
        AlignParams grad_out{};

        auto worker = [&]() {
            AlignParams local{};
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
        if (N > 0) grad_out.matrix.order_ = pairs[0]->grad().matrix.order_;
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
        std::vector<std::thread> threads;
        threads.reserve(static_cast<size_t>(actual - 1));
        for (int t = 0; t < actual - 1; ++t)
            threads.emplace_back(worker);
        worker();
        for (auto& t : threads) t.join();
    }
};
