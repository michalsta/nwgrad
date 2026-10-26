// ── TracebackMode::Grid: Pointers' result, bit for bit ───────────────────────────
//
// Grid's contract is identity with traceback=Pointers — score (every bit), path, guide,
// aligned strings and hard gradient, ties included — on every backend, both floating
// precisions.  Pairs above hb_cutoff take the grid (a small cutoff here, so the recursion
// and its block borders are exercised at modest lengths); the rest, and every case Grid
// does not implement (Local, linear, tracks, negative opens, integer T, scalar backend),
// run as Pointers, which this checks too.  Random problems mix ties (small-integer
// matrices and gaps), non-representable extends (0.1) and planted homology (long gap
// runs).  NWGRAD_GRID_FUZZ=<n> scales the iteration count.

#include "catch.hpp"

#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "align_params.hpp"
#include "aligner.hpp"
#include "simd_levels.hpp"

namespace {

std::vector<int> grid_backends() {
    std::vector<int> v{kBackendScalar};
    for (SimdLevel l : detect_available_levels()) v.push_back(static_cast<int>(l));
    return v;
}

long grid_iters(long dflt) {
    const char* e = std::getenv("NWGRAD_GRID_FUZZ");
    return e ? std::atol(e) : dflt;
}

bool same_bits(double x, double y) {
    if (x == 0.0 && y == 0.0) return true;
    return std::memcmp(&x, &y, sizeof x) == 0;
}

const char* kAA = "ACDEFGHIKLMNPQRSTVWY";

template <GapModel GM, AlignMode AM, class T>
void check_grid(const std::string& a, const std::string& b, const AlignParams& p, int kern,
                int cutoff, long& n_grid) {
    auto run = [&](TracebackMode tb, Aligner<GM, AM, AlignBand::Full, T>& al) {
        al.alloc_buf();
        al.set_kernel(kern);
        al.set_traceback(tb);
        al.set_hb_cutoff(cutoff);
        al.set_problem(a, b, p);
        al.compute_viterbi();
    };
    Aligner<GM, AM, AlignBand::Full, T> ref, got;
    run(TracebackMode::Pointers, ref);
    run(TracebackMode::Grid, got);
    REQUIRE(same_bits(got.score(), ref.score()));
    REQUIRE(got.alignment() == ref.alignment());
    REQUIRE(got.guide_j_from_viterbi() == ref.guide_j_from_viterbi());
    REQUIRE(got.aligned() == ref.aligned());
    AlignParams gr = p * 0.0, gg = p * 0.0;
    ref.hard_grad(gr);
    got.hard_grad(gg);
    REQUIRE(gg.gap_open_a == gr.gap_open_a);
    REQUIRE(gg.gap_extend_a == gr.gap_extend_a);
    REQUIRE(gg.gap_open_b == gr.gap_open_b);
    REQUIRE(gg.gap_extend_b == gr.gap_extend_b);
    for (std::size_t k = 0; k < 400; ++k) REQUIRE(gg.matrix.data()[k] == gr.matrix.data()[k]);
    if (static_cast<int>(a.size()) > cutoff) ++n_grid;
}

AlignParams random_params(std::mt19937& rng, int scheme) {
    std::array<double, 400> src{};
    std::uniform_int_distribution<int> sc(-4, 6);
    std::uniform_real_distribution<double> sr(-4, 6);
    for (int i = 0; i < 20; ++i)
        for (int j = 0; j < 20; ++j)
            src[i * 20 + j] = scheme == 2 ? sr(rng) : (scheme == 0 ? (i == j ? 2.0 : -1.0) : sc(rng));
    AlignParams p(SubstMatrix(src.data()));
    if (scheme == 2) { p.gap_open_a = 3.1; p.gap_extend_a = 0.1; p.gap_open_b = 2.7; p.gap_extend_b = 0.3; }
    else if (scheme == 1) { p.gap_open_a = 0; p.gap_extend_a = 1; p.gap_open_b = 2; p.gap_extend_b = 0; }
    else { p.gap_open_a = 3; p.gap_extend_a = 1; p.gap_open_b = 2; p.gap_extend_b = 1; }
    return p;
}

std::pair<std::string, std::string> random_pair(std::mt19937& rng, int maxlen) {
    std::uniform_int_distribution<int> L(0, maxlen), aa(0, 19), coin(0, 99);
    std::string A, B;
    const int ma = L(rng);
    for (int i = 0; i < ma; ++i) A += kAA[aa(rng)];
    if (coin(rng) < 70) {   // planted homology: substitutions and indels
        for (char c : A) {
            const int r = coin(rng);
            if (r < 5) continue;
            if (r < 10) B += kAA[aa(rng)];
            B += coin(rng) < 30 ? kAA[aa(rng)] : c;
        }
    } else {
        const int nb = L(rng);
        for (int i = 0; i < nb; ++i) B += kAA[aa(rng)];
    }
    return {A, B};
}

}  // namespace

TEST_CASE("Grid traceback is bit-identical to Pointers (affine Global)", "[grid][traceback]") {
    std::mt19937 rng(20261026);
    const long iters = grid_iters(60);
    long n_grid = 0;
    std::uniform_int_distribution<int> cut(1, 40);
    for (long t = 0; t < iters; ++t) {
        const AlignParams p = random_params(rng, static_cast<int>(t % 3));
        auto [A, B] = random_pair(rng, t % 10 == 0 ? 1200 : 400);
        const int cutoff = cut(rng);
        for (int kern : grid_backends()) {
            check_grid<GapModel::Affine, AlignMode::Global, double>(A, B, p, kern, cutoff, n_grid);
            check_grid<GapModel::Affine, AlignMode::Global, float>(A, B, p, kern, cutoff, n_grid);
        }
    }
    REQUIRE(n_grid > 0);
}

TEST_CASE("Grid traceback runs as Pointers where it is not implemented", "[grid][traceback]") {
    std::mt19937 rng(7);
    long n_grid = 0;
    for (int t = 0; t < 12; ++t) {
        AlignParams p = random_params(rng, t % 3);
        auto [A, B] = random_pair(rng, 300);
        for (int kern : grid_backends()) {
            check_grid<GapModel::Affine, AlignMode::Local, double>(A, B, p, kern, 8, n_grid);
            check_grid<GapModel::Linear, AlignMode::Global, float>(A, B, p, kern, 8, n_grid);
            AlignParams q = p;   // a negative open: the D/X form does not apply
            q.gap_open_a = -0.5;
            check_grid<GapModel::Affine, AlignMode::Global, double>(A, B, q, kern, 8, n_grid);
        }
    }
}
