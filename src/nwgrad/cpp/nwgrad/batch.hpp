#pragma once

#include <atomic>
#include <mutex>
#include <string_view>
#include <thread>
#include <vector>

#include "aligner.hpp"

struct ProblemInstance {
    std::string_view seq_a;
    std::string_view seq_b;
    std::vector<int> guide_j;  // empty → trivial diagonal guide (band around main diagonal)
};

struct BatchResult {
    std::vector<double> scores;
    double grad[256][256]{};
};

struct BatchAligner {
    BlosumMatrix matrix;
    double gap_open;
    double gap_extend;
    int    band;       // 0 = full DP; > 0 = banded with this half-width
    GapModel  gap_model;
    AlignMode align_mode;
    enum class GradMode { None, Hard, Soft };
    GradMode grad_mode;
    int n_threads;

    BatchAligner(BlosumMatrix mat, double go, double ge, int band,
                 GapModel gm, AlignMode am, GradMode gd, int nt)
        : matrix(std::move(mat)), gap_open(go), gap_extend(ge), band(band),
          gap_model(gm), align_mode(am), grad_mode(gd), n_threads(nt) {}

    BatchResult align(const std::vector<ProblemInstance>& problems) const {
        const size_t N = problems.size();
        BatchResult result;
        result.scores.resize(N, 0.0);

        if (N == 0) return result;

        std::atomic<size_t> work_idx{0};
        std::mutex grad_mutex;

        auto worker = [&]() {
            double local_grad[256][256]{};
            dispatch_worker(problems, N, work_idx, result.scores, local_grad);
            if (grad_mode != GradMode::None) {
                std::lock_guard<std::mutex> lock(grad_mutex);
                for (int r = 0; r < 256; ++r)
                    for (int c = 0; c < 256; ++c)
                        result.grad[r][c] += local_grad[r][c];
            }
        };

        int actual_threads = std::min<int>(n_threads, static_cast<int>(N));
        std::vector<std::thread> threads;
        threads.reserve(static_cast<size_t>(actual_threads - 1));
        for (int t = 0; t < actual_threads - 1; ++t)
            threads.emplace_back(worker);
        worker();
        for (auto& t : threads) t.join();

        return result;
    }

private:
    template<GapModel GM, AlignMode AM, AlignBand AB>
    void work_loop(
        const std::vector<ProblemInstance>& problems,
        size_t N,
        std::atomic<size_t>& work_idx,
        std::vector<double>& scores,
        double local_grad[256][256]) const
    {
        Aligner<GM, AM, AB> al;
        DpBuffer buf;  // reused across iterations; grows to the largest pair seen
        while (true) {
            size_t idx = work_idx.fetch_add(1, std::memory_order_relaxed);
            if (idx >= N) break;
            const auto& p = problems[idx];
            al.set_problem(p.seq_a, p.seq_b, matrix, gap_extend, gap_open, band, p.guide_j);

            if (grad_mode == GradMode::Hard) {
                al.compute_viterbi(buf);
                scores[idx] = al.score();
                al.hard_grad(buf, local_grad);
            } else if (grad_mode == GradMode::Soft) {
                al.compute_forward_back(buf);
                scores[idx] = al.log_z();
                al.soft_grad(buf, local_grad);
            } else {
                al.compute_viterbi(buf);
                scores[idx] = al.score();
            }
        }
    }

    void dispatch_worker(
        const std::vector<ProblemInstance>& problems,
        size_t N,
        std::atomic<size_t>& work_idx,
        std::vector<double>& scores,
        double local_grad[256][256]) const
    {
        const bool banded = (band > 0) || (!problems.empty() && !problems[0].guide_j.empty());
#define DISPATCH(GM, AM) \
        if (gap_model == GapModel::GM && align_mode == AlignMode::AM) { \
            if (banded) work_loop<GapModel::GM, AlignMode::AM, AlignBand::GuideBanded>(problems, N, work_idx, scores, local_grad); \
            else        work_loop<GapModel::GM, AlignMode::AM, AlignBand::Full>       (problems, N, work_idx, scores, local_grad); \
            return; \
        }
        DISPATCH(Linear, Global)
        DISPATCH(Linear, Local)
        DISPATCH(Affine, Global)
        DISPATCH(Affine, Local)
#undef DISPATCH
    }
};
