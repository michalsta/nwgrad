#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "align_params.hpp"
#include "aligner.hpp"
#include "parallel.hpp"

struct ProblemInstance {
    std::string_view seq_a;
    std::string_view seq_b;
    std::vector<int> guide_j;  // empty → trivial diagonal guide (band around main diagonal)
};

struct BatchResult {
    std::vector<double> scores;
    AlignParams grad;

    explicit BatchResult(const Alphabet& alpha) : grad(alpha) {}
};

template<class T = double>
struct BatchAlignerT {
    // Viterbi precision T (float32 default in Python, double via BatchAlignerDouble).
    // Shadows the global ::DpBuffer; AlignParams/scores stay double.
    using DpBuffer = DpBufferT<T>;

    AlignParams params;
    int    band;       // 0 = full DP; > 0 = banded with this half-width
    GapModel  gap_model;
    AlignMode align_mode;
    enum class GradMode { None, Hard, Soft };
    GradMode grad_mode;
    int n_threads;
    // Which Viterbi fills the DP tables.  A runtime field, exactly like the three
    // above — the two kernels are bit-exact, so this never changes a result, only a
    // duration.  Keeping it off the template parameter list is what leaves the
    // DISPATCH macro below at four arms rather than eight.
    DpKernel kernel;

    BatchAlignerT(AlignParams p, int band,
                 GapModel gm, AlignMode am, GradMode gd, int nt,
                 DpKernel k = DpKernel::Scalar)
        : params(std::move(p)), band(band),
          gap_model(gm), align_mode(am), grad_mode(gd), n_threads(nt), kernel(k) {}

    BatchResult align(const std::vector<ProblemInstance>& problems) const {
        const size_t N = problems.size();
        BatchResult result(params.matrix.alphabet());
        result.scores.resize(N, 0.0);

        if (N == 0) return result;

        // dispatch_worker picks one AlignBand for the whole batch by looking at
        // problems[0].guide_j — AlignBand is a compile-time Aligner parameter,
        // so one worker loop can't switch between Full and GuideBanded per
        // problem. That is fine when band > 0 (every problem gets banded, with
        // an auto diagonal guide filling in for any empty guide_j) or when no
        // problem carries a guide_j at all. It silently drops every guide past
        // problems[0] when band == 0 and only *some* problems carry one — the
        // dropped problems still get the correct (full-DP) score, just not the
        // banding they were given, with no diagnostic. Reject that case here,
        // on the caller's thread, before any work is dispatched.
        if (band == 0) {
            bool any_guided = false, any_unguided = false;
            for (const auto& p : problems) {
                if (p.guide_j.empty()) any_unguided = true;
                else                   any_guided   = true;
            }
            if (any_guided && any_unguided)
                throw std::invalid_argument(
                    "nwgrad: mixed batch with band == 0 — some problems carry a "
                    "guide_j and others don't. All problems in one align() call "
                    "must either all supply a guide_j or none of them; split "
                    "into separate align() calls, or set band > 0 so an unguided "
                    "problem gets an automatic diagonal guide instead.");
        }

        std::atomic<size_t> work_idx{0};
        std::mutex grad_mutex;

        auto worker = [&]() {
            AlignParams local_grad = AlignParams::zeros_like(params);
            dispatch_worker(problems, N, work_idx, result.scores, local_grad);
            if (grad_mode != GradMode::None) {
                std::lock_guard<std::mutex> lock(grad_mutex);
                result.grad += local_grad;
            }
        };

        int actual_threads = std::min<int>(n_threads, static_cast<int>(N));
        run_workers_guarded(actual_threads, worker);

        return result;
    }

private:
    // Validate and encode one sequence into `out`.  An out-of-alphabet character
    // throws, naming the pair and which of the two sequences it was in — the
    // aligner's own message could only name a position in an anonymous string.
    // The throw happens on a worker thread and is rethrown to the caller by
    // run_workers_guarded().
    static void encode_into(const Alphabet& alpha, std::string_view s,
                            std::vector<uint8_t>& out, size_t pair_idx, char which) {
        out.clear();
        out.reserve(s.size());
        for (size_t k = 0; k < s.size(); ++k) {
            int i = alpha.index_of(s[k]);
            if (i < 0) {
                std::string msg = "nwgrad: pair ";
                msg += std::to_string(pair_idx);
                msg += ", sequence ";
                msg += which;
                msg += ": character '";
                msg += s[k];
                msg += "' at position " + std::to_string(k) +
                       " is not in alphabet \"" + alpha.symbols() + "\"";
                throw std::invalid_argument(msg);
            }
            out.push_back(static_cast<uint8_t>(i));
        }
    }

    template<GapModel GM, AlignMode AM, AlignBand AB>
    void work_loop(
        const std::vector<ProblemInstance>& problems,
        size_t N,
        std::atomic<size_t>& work_idx,
        std::vector<double>& scores,
        AlignParams& local_grad) const
    {
        Aligner<GM, AM, AB, T> al;
        al.set_kernel(kernel);
        DpBuffer buf;  // reused across iterations; grows to the largest pair seen
        const Alphabet& alpha = params.matrix.alphabet();
        // Encoding buffers, reused across iterations: the batch never
        // materialises all N encoded sequences at once.
        std::vector<uint8_t> a_enc, b_enc;
        while (true) {
            size_t idx = work_idx.fetch_add(1, std::memory_order_relaxed);
            if (idx >= N) break;
            const auto& p = problems[idx];
            encode_into(alpha, p.seq_a, a_enc, idx, 'a');
            encode_into(alpha, p.seq_b, b_enc, idx, 'b');
            al.set_problem(a_enc, b_enc, params, band, p.guide_j);

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
        AlignParams& local_grad) const
    {
        // Safe to decide from problems[0] alone: align() has already rejected any
        // batch that mixes guided and unguided problems under band == 0, so every
        // problem here agrees with problems[0] on whether it carries a guide_j.
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

// Default (double) alias; Python binds BatchAlignerT<float> as `BatchAligner`.
using BatchAligner = BatchAlignerT<double>;
