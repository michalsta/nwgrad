// bench_f32.cpp — head-to-head double vs float32 striped Viterbi kernel.
//
// The whole reason float32 exists here: 2x the SIMD lanes and half the DP-table
// footprint.  This measures both regimes it should help — ALU-bound (1 thread, short
// sequences) and memory-bound (many threads, long sequences) — for the affine Global
// Full path, the striped kernel, forward-only and with the hard gradient.
//
// It links the per-ISA level TUs, so kernel=Simd dispatches the real striped kernel
// (viterbi for double, viterbi_f for float), not the header-only scalar fallback.
// Sequences are pre-encoded once and buffers reused, so what is timed is the kernel.
//
//   ./bench_f32 --n 2000 --seq-len 200 --threads 1 60
//   ./bench_f32 --n 400  --seq-len 200 700 1500 --threads 1 60 --grad hard

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "aligner.hpp"
#include "align_params.hpp"
#include "simd_levels.hpp"

// ── BLOSUM62 (canonical AA order ACDEFGHIKLMNPQRSTVWY) ───────────────────────
static const double BLOSUM62[20][20] = {
    { 4,  0, -2, -1, -2,  0, -2, -1, -1, -1, -1, -2, -1, -1, -1,  1,  0,  0, -3, -2},
    { 0,  9, -3, -4, -2, -3, -3, -1, -3, -1, -1, -3, -3, -3, -3, -1, -1, -1, -2, -2},
    {-2, -3,  6,  2, -3, -1, -1, -3, -1, -4, -3,  1, -1,  0, -2,  0, -1, -3, -4, -3},
    {-1, -4,  2,  5, -3, -2,  0, -3,  1, -3, -2,  0, -1,  2,  0,  0, -1, -2, -3, -2},
    {-2, -2, -3, -3,  6, -3, -1,  0, -3,  0,  0, -3, -4, -3, -3, -2, -2, -1,  1,  3},
    { 0, -3, -1, -2, -3,  6, -2, -4, -2, -4, -3,  0, -2, -2, -2,  0, -2, -3, -2, -3},
    {-2, -3, -1,  0, -1, -2,  8, -3, -1, -3, -2,  1, -2,  0,  0, -1, -2, -3, -2,  2},
    {-1, -1, -3, -3,  0, -4, -3,  4, -3,  2,  1, -3, -3, -3, -3, -2, -1,  3, -3, -1},
    {-1, -3, -1,  1, -3, -2, -1, -3,  5, -2, -1,  0, -1,  1,  2,  0, -1, -2, -3, -2},
    {-1, -1, -4, -3,  0, -4, -3,  2, -2,  4,  2, -3, -3, -2, -2, -2, -1,  1, -2, -1},
    {-1, -1, -3, -2,  0, -3, -2,  1, -1,  2,  5, -2, -2,  0, -1, -1, -1,  1, -1, -1},
    {-2, -3,  1,  0, -3,  0,  1, -3,  0, -3, -2,  6, -2,  0,  0,  1,  0, -3, -4, -2},
    {-1, -3, -1, -1, -4, -2, -2, -3, -1, -3, -2, -2,  7, -1, -2, -1, -1, -2, -4, -3},
    {-1, -3,  0,  2, -3, -2,  0, -3,  1, -2,  0,  0, -1,  5,  1,  0, -1, -2, -2, -1},
    {-1, -3, -2,  0, -3, -2,  0, -3,  2, -2, -1,  0, -2,  1,  5, -1, -1, -3, -3, -2},
    { 1, -1,  0,  0, -2,  0, -1, -2,  0, -2, -1,  1, -1,  0, -1,  4,  1, -2, -3, -2},
    { 0, -1, -1, -1, -2, -2, -2, -1, -1, -1, -1,  0, -1, -1, -1,  1,  5,  0, -2, -2},
    { 0, -1, -3, -2, -1, -3, -3,  3, -2,  1,  1, -3, -2, -2, -3, -2,  0,  4, -3, -1},
    {-3, -2, -4, -3,  1, -2, -2, -3, -3, -2, -1, -4, -4, -2, -3, -3, -2, -3, 11,  2},
    {-2, -2, -3, -2,  3, -3,  2, -1, -2, -1, -1, -2, -3, -1, -2, -2, -2, -1,  2,  7},
};

static AlignParams make_params(double open, double ext) {
    double flat[400];
    for (int i = 0; i < 20; ++i)
        for (int j = 0; j < 20; ++j) flat[i * 20 + j] = BLOSUM62[i][j];
    AlignParams p{SubstMatrix(flat)};
    p.gap_open_a = open; p.gap_extend_a = ext;
    p.gap_open_b = open; p.gap_extend_b = ext;
    return p;
}

using Clock = std::chrono::steady_clock;

// One timed pass: nthreads workers, each owning a reused DpBufferT<T>, dividing the
// pre-encoded pairs via an atomic counter.  Returns seconds for the whole pass.
template <class T>
static double run_pass(bool grad,
                       const std::vector<std::vector<uint8_t>>& A,
                       const std::vector<std::vector<uint8_t>>& B,
                       const AlignParams& p, int nthreads) {
    std::atomic<size_t> next{0};
    const size_t N = A.size();
    auto worker = [&] {
        Aligner<GapModel::Affine, AlignMode::Global, AlignBand::Full, T> al;
        DpBufferT<T> buf;
        AlignParams grad_acc = AlignParams::zeros_like(p);
        for (;;) {
            size_t i = next.fetch_add(1, std::memory_order_relaxed);
            if (i >= N) break;
            al.set_problem(std::span<const uint8_t>(A[i]), std::span<const uint8_t>(B[i]), p);
            al.set_kernel(DpKernel::Simd);
            al.compute_viterbi(buf);
            if (grad) al.hard_grad(buf, grad_acc);
        }
    };
    auto t0 = Clock::now();
    std::vector<std::thread> ts;
    for (int t = 0; t < nthreads; ++t) ts.emplace_back(worker);
    for (auto& t : ts) t.join();
    auto t1 = Clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

int main(int argc, char** argv) {
    int n = 2000;
    std::vector<int> lens = {200};
    std::vector<int> threads = {1, 60};
    bool grad = false;

    for (int i = 1; i < argc; ++i) {
        auto nums = [&](std::vector<int>& out) {
            out.clear();
            while (i + 1 < argc && std::isdigit((unsigned char)argv[i + 1][0])) out.push_back(std::atoi(argv[++i]));
        };
        if      (!std::strcmp(argv[i], "--n"))       n = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--seq-len")) nums(lens);
        else if (!std::strcmp(argv[i], "--threads")) nums(threads);
        else if (!std::strcmp(argv[i], "--grad"))    grad = (std::string(argv[++i]) == "hard");
        else { std::fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
    }

    AlignParams p = make_params(11.0, 1.0);
    const Alphabet& alpha = p.matrix.alphabet();
    std::mt19937_64 rng(20260717);
    std::uniform_int_distribution<int> aa(0, 19);
    static constexpr const char* ORDER = "ACDEFGHIKLMNPQRSTVWY";

    std::printf("bench_f32  isa=%s  n=%d  grad=%s\n\n",
                get_isa_level().c_str(), n, grad ? "hard" : "none");
    std::printf("%5s %4s %8s %11s %11s %8s\n",
                "len", "thr", "cells/pr", "f64 Mcell/s", "f32 Mcell/s", "f32/f64");

    for (int L : lens) {
        // Fresh random pairs for this length; pre-encode once.
        std::vector<std::vector<uint8_t>> A(n), B(n);
        for (int k = 0; k < n; ++k) {
            std::string sa(L, 'A'), sb(L, 'A');
            for (char& c : sa) c = ORDER[aa(rng)];
            for (char& c : sb) c = ORDER[aa(rng)];
            A[k] = alpha.encode(sa); B[k] = alpha.encode(sb);
        }
        const double cells = double(n) * double(L) * double(L);

        for (int th : threads) {
            // warmup each precision once (page-in, branch predictors, buffer growth)
            run_pass<double>(grad, A, B, p, th);
            run_pass<float >(grad, A, B, p, th);
            double td = run_pass<double>(grad, A, B, p, th);
            double tf = run_pass<float >(grad, A, B, p, th);
            double md = cells / td / 1e6, mf = cells / tf / 1e6;
            std::printf("%5d %4d %8.0f %11.1f %11.1f %8.2fx\n",
                        L, th, double(L) * L, md, mf, mf / md);
        }
    }
    return 0;
}
