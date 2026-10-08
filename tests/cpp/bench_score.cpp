// Score-only throughput: compute_score() against the full fill it replaces.
//
//   bench_score <threads> [reps] [lens...]
//
// Arms, all in one process (the repo's rule: separately compiled binaries carry code
// layout artifacts larger than the effects measured here):
//   fill_auto   compute_viterbi() + score() under traceback "auto" — what the score-only
//               convenience functions ran before compute_score existed
//   fill_ptr    the same under traceback "pointers" (the exact fill, 3 B/cell)
//   score       compute_score() on the default simd level
//   score_sc    compute_score() on the scalar backend (the rolling-row reference)
// NWGRAD_SCORE_VARIANT=twopass selects the unfused kernel for `score` (run the binary
// twice).  Pairs: random protein (BLOSUM-like real matrix) and DNA, homologous (30 %
// mutated), affine and linear, Global and Local, double and float32.  Mcell/s, best of reps.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "align_params.hpp"
#include "aligner.hpp"
#include "simd_levels.hpp"

namespace {

struct Problem { std::string a, b; };

std::vector<Problem> make_pairs(const Alphabet& A, int len, int npairs, std::mt19937_64& rng) {
    std::vector<Problem> out;
    const int na = A.size();
    auto res = [&] { return A.symbols()[rng() % na]; };
    for (int k = 0; k < npairs; ++k) {
        std::string a; for (int x = 0; x < len; ++x) a.push_back(res());
        std::string b;
        for (char c : a) {
            const int r = static_cast<int>(rng() % 100);
            if (r < 20) b.push_back(res());          // substitution
            else if (r < 25) continue;               // deletion
            else if (r < 30) { b.push_back(c); b.push_back(res()); }   // insertion
            else b.push_back(c);
        }
        out.push_back({a, b});
    }
    return out;
}

AlignParams make_params(const Alphabet& A, std::mt19937_64& rng) {
    AlignParams p(A);
    std::normal_distribution<double> nd(0.0, 1.0);
    for (int x = 0; x < A.size(); ++x)
        for (int y = 0; y < A.size(); ++y) p.matrix.at(x, y) = x == y ? 2.0 + 0.3 * nd(rng) : -0.7 + 0.5 * nd(rng);
    p.gap_open_a = p.gap_open_b = 3.3;
    p.gap_extend_a = p.gap_extend_b = 0.3;
    return p;
}

template <class F>
double run_threads(int threads, size_t n, F&& f) {
    std::atomic<size_t> next{0};
    auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> ts;
    for (int t = 0; t < threads; ++t)
        ts.emplace_back([&] { f(next, n); });
    for (auto& t : ts) t.join();
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

double sink = 0;

template <GapModel GM, AlignMode AM, class T>
void bench_cfg(const char* name, const std::vector<Problem>& P, const AlignParams& p,
               int threads, int reps) {
    double cells = 0;
    for (const auto& q : P) cells += double(q.a.size()) * double(q.b.size());
    auto arm = [&](int mode) {   // 0 fill_auto, 1 fill_ptr, 2 score, 3 score_sc
        double best = 1e30;
        for (int r = 0; r < reps; ++r) {
            std::atomic<long> chk{0};
            const double t = run_threads(threads, P.size(), [&](std::atomic<size_t>& next, size_t n) {
                Aligner<GM, AM, AlignBand::Full, T> al;
                if (mode == 1) al.set_traceback(TracebackMode::Pointers);
                if (mode == 3) al.set_kernel(kBackendScalar);
                DpBufferT<T> buf;
                double local = 0;
                for (size_t k; (k = next.fetch_add(1)) < n;) {
                    al.set_problem(P[k].a, P[k].b, p);
                    if (mode <= 1) { al.compute_viterbi(buf); local += al.score(); }
                    else           local += al.compute_score(buf);
                }
                chk += static_cast<long>(local);
            });
            sink += chk.load();
            best = std::min(best, t);
        }
        return cells / best / 1e6;
    };
    const double a0 = arm(0), a1 = arm(1), a2 = arm(2), a3 = arm(3);
    std::printf("%-26s %8.0f %8.0f %8.0f %8.0f   %5.2fx %5.2fx\n", name, a0, a1, a2, a3,
                a2 / a0, a2 / a1);
    std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
    const int threads = argc > 1 ? std::atoi(argv[1]) : 1;
    const int reps = argc > 2 ? std::atoi(argv[2]) : 3;
    std::vector<int> lens;
    for (int k = 3; k < argc; ++k) lens.push_back(std::atoi(argv[k]));
    if (lens.empty()) lens = {50, 200, 1000, 3000};
    std::mt19937_64 rng(7);
    std::printf("isa=%d threads=%d variant=%s\n", (int)global_default_backend(), threads,
                std::getenv("NWGRAD_SCORE_VARIANT") ? std::getenv("NWGRAD_SCORE_VARIANT") : "fused");
    std::printf("%-26s %8s %8s %8s %8s   %6s %6s   (Mcell/s)\n", "config", "fill_auto",
                "fill_ptr", "score", "score_sc", "s/auto", "s/ptr");
    for (int len : lens) {
        // enough cells per arm to time (~4e8 cells at one thread, more pairs at more)
        const double target = 2e8 * std::max(1, threads);
        const int np = std::max(threads * 4, static_cast<int>(target / (double(len) * len)));
        for (int alpha = 0; alpha < 2; ++alpha) {
            const Alphabet& A = alpha ? Alphabet::dna() : Alphabet::protein();
            const auto P = make_pairs(A, len, np, rng);
            const AlignParams p = make_params(A, rng);
            char nm[64];
            auto name = [&](const char* k) {
                std::snprintf(nm, sizeof nm, "%s %s L%d", alpha ? "dna" : "prot", k, len);
                return nm;
            };
            bench_cfg<GapModel::Affine, AlignMode::Global, double>(name("aff glob f64"), P, p, threads, reps);
            bench_cfg<GapModel::Affine, AlignMode::Local,  double>(name("aff loc  f64"), P, p, threads, reps);
            bench_cfg<GapModel::Affine, AlignMode::Global, float >(name("aff glob f32"), P, p, threads, reps);
            bench_cfg<GapModel::Affine, AlignMode::Local,  float >(name("aff loc  f32"), P, p, threads, reps);
            bench_cfg<GapModel::Linear, AlignMode::Global, double>(name("lin glob f64"), P, p, threads, reps);
            bench_cfg<GapModel::Linear, AlignMode::Local,  float >(name("lin loc  f32"), P, p, threads, reps);
        }
    }
    std::fprintf(stderr, "sink %g\n", sink);
}
