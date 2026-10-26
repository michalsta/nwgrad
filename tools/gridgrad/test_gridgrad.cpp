// Bit-exact check: GridGrad vs Aligner<Affine, Global> with TracebackMode::Pointers (scalar).
#include "nwgrad/aligner.hpp"
#include "gridgrad.hpp"
#include <cstdio>
#include <cstring>
#include <random>
#include <string>

static const char* AA = "ACDEFGHIKLMNPQRSTVWY";

template <class T>
static int check(std::mt19937& rng, int trial) {
    std::uniform_int_distribution<int> L(0, 300), aa(0, 19), coin(0, 99);
    const int ma = L(rng);
    std::string A, B;
    for (int i = 0; i < ma; ++i) A += AA[aa(rng)];
    // B: mutated A (ties galore with integer scores) or random
    const bool rel = coin(rng) < 70;
    if (rel) { for (char c : A) { int r = coin(rng); if (r < 5) continue; if (r < 10) B += AA[aa(rng)]; B += (coin(rng) < 30 ? AA[aa(rng)] : c); } }
    else { const int nb = L(rng); for (int i = 0; i < nb; ++i) B += AA[aa(rng)]; }

    std::array<double, 400> src{};
    const int scheme = trial % 3;
    std::uniform_int_distribution<int> sc(-4, 6);
    std::uniform_real_distribution<double> sr(-4, 6);
    for (int i = 0; i < 20; ++i) for (int j = 0; j < 20; ++j)
        src[i * 20 + j] = scheme == 2 ? sr(rng) : (scheme == 0 ? (i == j ? 2.0 : -1.0) : sc(rng));
    AlignParams p(SubstMatrix(src.data()));
    if (scheme == 2) { p.gap_open_a = 3.1; p.gap_extend_a = 0.1; p.gap_open_b = 2.7; p.gap_extend_b = 0.3; }
    else { p.gap_open_a = 3; p.gap_extend_a = 1; p.gap_open_b = 2; p.gap_extend_b = 1; }

    Aligner<GapModel::Affine, AlignMode::Global, AlignBand::Full, T> al;
    al.alloc_buf();
    al.set_kernel(kBackendScalar);
    al.set_traceback(TracebackMode::Pointers);
    al.set_problem(A, B, p);
    al.compute_viterbi();
    AlignParams g = p * 0.0;
    al.hard_grad(g);

    const Alphabet& alpha = p.matrix.alphabet();
    auto ea = alpha.encode(A), eb = alpha.encode(B);
    std::vector<T> blk(400);
    for (int k = 0; k < 400; ++k) blk[k] = static_cast<T>(src[k]);
    std::uniform_int_distribution<int> grid(1, 40);
    GridGrad<T> gg{ea.data(), eb.data(), (int)ea.size(), (int)eb.size(), 20, blk.data(),
                   (T)p.gap_open_a, (T)p.gap_extend_a, (T)p.gap_open_b, (T)p.gap_extend_b,
                   grid(rng), grid(rng)};
    gg.run();
    bool ok = gg.score == al.score() && gg.g_go_a == g.gap_open_a && gg.g_ge_a == g.gap_extend_a &&
              gg.g_go_b == g.gap_open_b && gg.g_ge_b == g.gap_extend_b;
    for (int k = 0; k < 400; ++k) ok = ok && gg.gmat[k] == g.matrix.data()[k];
    if (!ok) std::printf("MISMATCH T=%s trial %d m=%d n=%d R=%d C=%d score %.9g vs %.9g\n",
                         sizeof(T) == 4 ? "f32" : "f64", trial, gg.m, gg.n, gg.R, gg.C, gg.score, al.score());
    return ok ? 0 : 1;
}

int main(int argc, char** argv) {
    const int N = argc > 1 ? std::atoi(argv[1]) : 2000;
    std::mt19937 rng(1);
    int bad = 0;
    for (int t = 0; t < N; ++t) { bad += check<float>(rng, t); bad += check<double>(rng, t); }
    std::printf("%d mismatches in %d cases\n", bad, 2 * N);
    return bad != 0;
}
