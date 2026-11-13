// ── TracebackMode::Grid: Pointers' result, bit for bit ───────────────────────────
//
// Grid's contract is identity with traceback=Pointers — score (every bit), path, guide,
// aligned strings and hard gradient, ties included — on every backend, both floating
// precisions.  Pairs above hb_cutoff take the grid (a small cutoff here, so the recursion
// and its block borders are exercised at modest lengths); the rest, and every case Grid
// does not implement (tracks, negative opens, linear Local with a negative extend, scalar
// backend), run as Pointers, which this checks too.  Linear gaps since 2026-11-07 (below).  Integer T (every mode, int16 in both regimes) since 2026-11-05.  Random problems mix ties (small-integer
// matrices and gaps), non-representable extends (0.1) and planted homology (long gap
// runs).  NWGRAD_GRID_FUZZ=<n> scales the iteration count.

#include "catch.hpp"

#include <array>
#include <cstdint>
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

// SimdLevel::Avx512 exists on x86 builds only (the ARM64 job has no such enumerator).
bool is_avx512(int kern) {
#if !(defined(__aarch64__) || defined(_M_ARM64))
    return kern == static_cast<int>(SimdLevel::Avx512);
#else
    (void)kern;
    return false;
#endif
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
    if (got.ran_grid()) ++n_grid;
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

TEST_CASE("Grid traceback is bit-identical to Pointers (affine Glocal)", "[grid][traceback]") {
    // Row 0 a start at every column, the end at row m's first best column (from the top
    // level's last row, or the base block when the problem is one); the walk stops on row 0.
    std::mt19937 rng(20261104);
    const long iters = grid_iters(60);
    long n_grid = 0;
    std::uniform_int_distribution<int> cut(1, 40);
    for (long t = 0; t < iters; ++t) {
        const AlignParams p = random_params(rng, static_cast<int>(t % 3));
        auto [A, B] = random_pair(rng, t % 10 == 0 ? 1200 : 400);
        if (t % 7 == 3) B = B.substr(0, B.size() / 8);   // short B: the end column near 0
        const int cutoff = cut(rng);
        for (int kern : grid_backends()) {
            check_grid<GapModel::Affine, AlignMode::Glocal, double>(A, B, p, kern, cutoff, n_grid);
            check_grid<GapModel::Affine, AlignMode::Glocal, float>(A, B, p, kern, cutoff, n_grid);
        }
    }
    REQUIRE(n_grid > 0);
}

TEST_CASE("Grid traceback is bit-identical to Pointers (affine Local)", "[grid][traceback]") {
    // The end cell from the exact forward scan, M clamped at 0 in the grid kernel and the
    // base fill, the walk stopped at a start (code 3); unrelated pairs often score <= 0
    // (the empty alignment) or end near a border.
    std::mt19937 rng(20261105);
    const long iters = grid_iters(60);
    long n_grid = 0;
    std::uniform_int_distribution<int> cut(1, 40);
    for (long t = 0; t < iters; ++t) {
        const AlignParams p = random_params(rng, static_cast<int>(t % 3));
        auto [A, B] = random_pair(rng, t % 10 == 0 ? 1200 : 400);
        if (t % 5 == 2) {   // homology in the middle of unrelated flanks
            std::uniform_int_distribution<int> aa(0, 19), L(0, 150);
            std::string f1, f2, f3, f4;
            for (int k = L(rng); k > 0; --k) f1 += kAA[aa(rng)];
            for (int k = L(rng); k > 0; --k) f2 += kAA[aa(rng)];
            for (int k = L(rng); k > 0; --k) f3 += kAA[aa(rng)];
            for (int k = L(rng); k > 0; --k) f4 += kAA[aa(rng)];
            A = f1 + A + f2; B = f3 + B + f4;
        }
        const int cutoff = cut(rng);
        for (int kern : grid_backends()) {
            check_grid<GapModel::Affine, AlignMode::Local, double>(A, B, p, kern, cutoff, n_grid);
            check_grid<GapModel::Affine, AlignMode::Local, float>(A, B, p, kern, cutoff, n_grid);
        }
    }
    REQUIRE(n_grid > 0);
}

// Integral parameters (the integer types take no other); scheme 3: BLOSUM-like range with
// large opens, which puts int16 past the quarter rule (the saturating regime) at ~L300.
AlignParams random_int_params(std::mt19937& rng, int scheme) {
    std::array<double, 400> src{};
    std::uniform_int_distribution<int> sc(-4, 6), sb(-4, 11);
    for (int i = 0; i < 20; ++i)
        for (int j = 0; j < 20; ++j)
            src[i * 20 + j] = scheme == 0 ? (i == j ? 2.0 : -1.0) : (scheme == 3 ? sb(rng) : sc(rng));
    AlignParams p(SubstMatrix(src.data()));
    if (scheme == 1)      { p.gap_open_a = 0; p.gap_extend_a = 1; p.gap_open_b = 2; p.gap_extend_b = 0; }
    else if (scheme == 3) { p.gap_open_a = 11; p.gap_extend_a = 1; p.gap_open_b = 10; p.gap_extend_b = 2; }
    else                  { p.gap_open_a = 3; p.gap_extend_a = 1; p.gap_open_b = 2; p.gap_extend_b = 1; }
    return p;
}

// Integer T (2026-11-05): the fused prefix-max forward (int16 / int32 from 4 lanes), the exact
// lazy-F below 4 lanes (int64 at 128 bits), int16 in both regimes; every mode.
template <AlignMode AM>
void grid_int_round(std::uint32_t seed, long iters) {
    std::mt19937 rng(seed);
    long ng[3] = {0, 0, 0};   // int32, int64, int16
    std::uniform_int_distribution<int> cut(1, 40);
    for (long t = 0; t < iters; ++t) {
        auto [A, B] = random_pair(rng, t % 10 == 0 ? 1000 : 400);
        const int cutoff = cut(rng);
        const int scheme = static_cast<int>(t % 4);
        const AlignParams p = random_int_params(rng, scheme == 2 ? 0 : scheme);
        for (int kern : grid_backends()) {
            check_grid<GapModel::Affine, AM, int32_t>(A, B, p, kern, cutoff, ng[0]);
            check_grid<GapModel::Affine, AM, int64_t>(A, B, p, kern, cutoff, ng[1]);
            if (A.size() + B.size() < 1400 || scheme == 0)   // within int16's bound
                check_grid<GapModel::Affine, AM, int16_t>(A, B, p, kern, cutoff, ng[2]);
        }
    }
    for (long c : ng) REQUIRE(c > 0);   // each type took the grid on some pair
}

TEST_CASE("Grid traceback is bit-identical to Pointers (integer T, every mode)", "[grid][traceback]") {
    grid_int_round<AlignMode::Global>(20261107, grid_iters(30));
    grid_int_round<AlignMode::Local>(20261108, grid_iters(30));
    grid_int_round<AlignMode::Glocal>(20261109, grid_iters(20));
}

TEST_CASE("Grid traceback runs as Pointers where it is not implemented", "[grid][traceback]") {
    std::mt19937 rng(7);
    long n_grid = 0;
    for (int t = 0; t < 12; ++t) {
        AlignParams p = random_params(rng, t % 3);
        auto [A, B] = random_pair(rng, 300);
        for (int kern : grid_backends()) {
            AlignParams r = p;   // linear Local with a gaining gap: degenerate, not gridded
            r.gap_extend_a = -0.5;
            check_grid<GapModel::Linear, AlignMode::Local, double>(A, B, r, kern, 8, n_grid);
            AlignParams ri = random_int_params(rng, 0);
            ri.gap_extend_b = -1;
            check_grid<GapModel::Linear, AlignMode::Local, int32_t>(A, B, ri, kern, 8, n_grid);
            AlignParams q = p;   // a negative open: the D/X form does not apply
            q.gap_open_a = -0.5;
            check_grid<GapModel::Affine, AlignMode::Global, double>(A, B, q, kern, 8, n_grid);
        }
    }
    REQUIRE(n_grid == 0);
}

TEST_CASE("auto takes Grid for Glocal, and for Local above the pointer budget", "[grid][traceback]") {
    // kDefaultTb (2026-11-05): Glocal past hb_cutoff (512 rows) runs Grid on every simd level (the
    // scalar backend falls back, pmax / Pointers); Local runs Pointers within
    // kAutoPointerCells and Grid above it.  Every run is Pointers' result.
    std::mt19937 rng(20261106);
    const AlignParams p = random_params(rng, 0);
    auto run = [&](auto& al, TracebackMode tb, int kern, const std::string& a, const std::string& b) {
        al.alloc_buf();
        al.set_kernel(kern);
        al.set_traceback(tb);
        al.set_problem(a, b, p);
        al.compute_viterbi();
    };
    std::uniform_int_distribution<int> aa(0, 19);
    std::string A, B;
    for (int i = 0; i < 800; ++i) A += kAA[aa(rng)];
    for (int i = 0; i < 1200; ++i) B += kAA[aa(rng)];
    for (int kern : grid_backends()) {
        Aligner<GapModel::Affine, AlignMode::Glocal, AlignBand::Full, float> gf, rf;
        Aligner<GapModel::Affine, AlignMode::Glocal, AlignBand::Full, double> gd, rd;
        run(gf, TracebackMode::Default, kern, A, B); run(rf, TracebackMode::Pointers, kern, A, B);
        run(gd, TracebackMode::Default, kern, A, B); run(rd, TracebackMode::Pointers, kern, A, B);
        REQUIRE(gf.ran_grid() == (kern != kBackendScalar));
        REQUIRE(gd.ran_grid() == (kern != kBackendScalar));
        REQUIRE(same_bits(gd.score(), rd.score()));
        REQUIRE(gd.alignment() == rd.alignment());
        if (kern != kBackendScalar) {   // the scalar fallback is pmax at float32
            REQUIRE(same_bits(gf.score(), rf.score()));
            REQUIRE(gf.alignment() == rf.alignment());
        }
        // Local within the budget: the one-pass Grid (every gap parameter >= 0), Pointers' result;
        // float32 on AVX-512 only from 2048 rows (kLocalGridMinRows512), so 800 rows: Pointers
        Aligner<GapModel::Affine, AlignMode::Local, AlignBand::Full, float> lf, lfp;
        run(lf, TracebackMode::Default, kern, A, B); run(lfp, TracebackMode::Pointers, kern, A, B);
        REQUIRE(lf.ran_grid() == (kern != kBackendScalar && !is_avx512(kern)));
        REQUIRE(same_bits(lf.score(), lfp.score()));
        REQUIRE(lf.alignment() == lfp.alignment());
        Aligner<GapModel::Affine, AlignMode::Local, AlignBand::Full, double> ld;
        run(ld, TracebackMode::Default, kern, A, B);
        REQUIRE(ld.ran_grid() == (kern != kBackendScalar));
    }
    // above 2^25 cells, on the best level only (a 5900 x 5900 Pointers reference)
    const auto levels = detect_available_levels();
    if (levels.empty()) return;
    const int kern = static_cast<int>(levels.back());
    std::string LA, LB;
    for (int i = 0; i < 5900; ++i) LA += kAA[aa(rng)];
    for (char c : LA) LB += aa(rng) < 6 ? kAA[aa(rng)] : c;
    Aligner<GapModel::Affine, AlignMode::Local, AlignBand::Full, float> lg, lp;
    run(lg, TracebackMode::Default, kern, LA, LB);
    run(lp, TracebackMode::Pointers, kern, LA, LB);
    REQUIRE(lg.ran_grid());
    REQUIRE(same_bits(lg.score(), lp.score()));
    REQUIRE(lg.alignment() == lp.alignment());
}

TEST_CASE("auto takes Grid for integer T where it splits", "[grid][traceback]") {
    // run_tb_ (2026-11-05): integer `auto` is hirschberg_pmax; a pair that splits (past
    // hb_cutoff = 512 rows; Global int32 and Glocal have no pointer budget, Global int64 has
    // 2^25) runs Grid on a simd level, Pointers within the budget; int16 Global / Glocal keep
    // pmax; Local (not int16) takes Grid past hb_cutoff at every size.
    std::mt19937 rng(20261110);
    const AlignParams p = random_int_params(rng, 0);
    auto run = [&](auto& al, TracebackMode tb, int kern, const std::string& a, const std::string& b) {
        al.alloc_buf();
        al.set_kernel(kern);
        al.set_traceback(tb);
        al.set_problem(a, b, p);
        al.compute_viterbi();
    };
    std::uniform_int_distribution<int> aa(0, 19);
    std::string A, B;
    for (int i = 0; i < 800; ++i) A += kAA[aa(rng)];
    for (char c : A) B += aa(rng) < 5 ? kAA[aa(rng)] : c;
    for (int kern : grid_backends()) {
        const bool simd = kern != kBackendScalar;
        Aligner<GapModel::Affine, AlignMode::Global, AlignBand::Full, int32_t> g, gp;
        run(g, TracebackMode::Default, kern, A, B); run(gp, TracebackMode::Pointers, kern, A, B);
        REQUIRE(g.ran_grid() == simd);
        REQUIRE(g.score() == gp.score());
        if (simd) REQUIRE(g.alignment() == gp.alignment());
        Aligner<GapModel::Affine, AlignMode::Glocal, AlignBand::Full, int32_t> c, cp;
        run(c, TracebackMode::Default, kern, A, B); run(cp, TracebackMode::Pointers, kern, A, B);
        REQUIRE(c.ran_grid() == simd);
        if (simd) REQUIRE(c.alignment() == cp.alignment());
        // int16 Global / Glocal keep hirschberg_pmax (it beat Grid at 32 lanes, AVX-512)
        Aligner<GapModel::Affine, AlignMode::Global, AlignBand::Full, int16_t> g16, g16p;
        run(g16, TracebackMode::Default, kern, A, B); run(g16p, TracebackMode::Pointers, kern, A, B);
        REQUIRE_FALSE(g16.ran_grid());
        REQUIRE(g16.score() == g16p.score());
        Aligner<GapModel::Affine, AlignMode::Glocal, AlignBand::Full, int16_t> c16;
        run(c16, TracebackMode::Default, kern, A, B);
        REQUIRE_FALSE(c16.ran_grid());
        Aligner<GapModel::Affine, AlignMode::Local, AlignBand::Full, int32_t> l, lp;
        run(l, TracebackMode::Default, kern, A, B); run(lp, TracebackMode::Pointers, kern, A, B);
        REQUIRE(l.ran_grid() == simd);   // Local: the one-pass Grid inside the budget too
        if (simd) REQUIRE(l.alignment() == lp.alignment());
        Aligner<GapModel::Affine, AlignMode::Local, AlignBand::Full, int16_t> l16;
        run(l16, TracebackMode::Default, kern, A, B);
        REQUIRE_FALSE(l16.ran_grid());   // int16 Local inside the budget: Pointers (a tie)
        Aligner<GapModel::Affine, AlignMode::Global, AlignBand::Full, int64_t> g64;
        run(g64, TracebackMode::Default, kern, A, B);
        REQUIRE_FALSE(g64.ran_grid());
    }
}

// ── Linear gaps (2026-11-07): Grid over H (viterbi_linear_grid), bit-identical to the linear
// Pointers fill (striped or scalar) -- score bits, path, ties, guide, aligned strings,
// gradient -- in every mode, every scalar type, every backend (the scalar one runs as
// Pointers).  Floating: ties (small-integer matrices), 0.1 / 0.3 extends, a zero extend,
// negative extends (Global / Glocal; Local with one is not gridded, above); integer: int16
// in both regimes, int32, int64 (the fused prefix-max forward from 4 lanes, lazy-F below).
// Block edge cases: cutoff 1, one-column and narrow B, short A, the whole problem one base
// block (Local's end from the base block's scan, Glocal's from its last row).
namespace {

AlignParams random_linear_params(std::mt19937& rng, int scheme, bool allow_negative) {
    AlignParams p = random_params(rng, scheme % 3);
    if (scheme == 3) { p.gap_extend_a = 0.0; p.gap_extend_b = 0.7; }
    if (scheme == 4 && allow_negative) { p.gap_extend_a = -0.25; p.gap_extend_b = 0.5; }
    if (scheme == 5 && allow_negative) { p.gap_extend_a = 1.5; p.gap_extend_b = -0.1; }
    return p;
}

std::pair<std::string, std::string> linear_edge_pair(std::mt19937& rng, long t, int maxlen) {
    auto [A, B] = random_pair(rng, t % 10 == 0 ? 3 * maxlen : maxlen);
    std::uniform_int_distribution<int> aa(0, 19), L(0, 150);
    switch (t % 9) {
        case 1: B = B.substr(0, std::min<std::size_t>(B.size(), 1)); break;   // one column
        case 2: B = B.substr(0, B.size() / 9); break;                          // narrow B
        case 3: A = A.substr(0, A.size() / 7); break;                          // short A
        case 4: {                                                               // homology inside flanks
            std::string f1, f2, f3, f4;
            for (int k = L(rng); k > 0; --k) f1 += kAA[aa(rng)];
            for (int k = L(rng); k > 0; --k) f2 += kAA[aa(rng)];
            for (int k = L(rng); k > 0; --k) f3 += kAA[aa(rng)];
            for (int k = L(rng); k > 0; --k) f4 += kAA[aa(rng)];
            A = f1 + A + f2; B = f3 + B + f4;
            break;
        }
        default: break;
    }
    return {A, B};
}

template <AlignMode AM>
void grid_linear_float_round(std::uint32_t seed, long iters) {
    std::mt19937 rng(seed);
    long ng = 0;
    std::uniform_int_distribution<int> cut(1, 60);
    for (long t = 0; t < iters; ++t) {
        const AlignParams p = random_linear_params(rng, static_cast<int>(t % 6), AM != AlignMode::Local);
        auto [A, B] = linear_edge_pair(rng, t, 400);
        const int cutoff = t % 4 == 0 ? 1 : cut(rng);
        for (int kern : grid_backends()) {
            check_grid<GapModel::Linear, AM, double>(A, B, p, kern, cutoff, ng);
            check_grid<GapModel::Linear, AM, float>(A, B, p, kern, cutoff, ng);
        }
    }
    REQUIRE(ng > 0);
}

template <AlignMode AM>
void grid_linear_int_round(std::uint32_t seed, long iters) {
    std::mt19937 rng(seed);
    long ng[3] = {0, 0, 0};   // int32, int64, int16
    std::uniform_int_distribution<int> cut(1, 60);
    for (long t = 0; t < iters; ++t) {
        auto [A, B] = linear_edge_pair(rng, t, 350);
        const int cutoff = t % 4 == 0 ? 1 : cut(rng);
        const int scheme = static_cast<int>(t % 5);
        AlignParams p = random_int_params(rng, scheme == 3 ? 3 : (scheme == 4 ? 0 : scheme));
        if (scheme == 4) { p.gap_extend_a = 0; p.gap_extend_b = 3; }
        if (scheme == 2 && AM != AlignMode::Local) { p.gap_extend_a = -1; p.gap_extend_b = 2; }
        for (int kern : grid_backends()) {
            check_grid<GapModel::Linear, AM, int32_t>(A, B, p, kern, cutoff, ng[0]);
            check_grid<GapModel::Linear, AM, int64_t>(A, B, p, kern, cutoff, ng[1]);
            if (A.size() + B.size() < 1400 || scheme == 0)
                check_grid<GapModel::Linear, AM, int16_t>(A, B, p, kern, cutoff, ng[2]);
        }
    }
    for (long c : ng) REQUIRE(c > 0);
}

}  // namespace

TEST_CASE("Grid traceback is bit-identical to Pointers (linear, floating T)", "[grid][traceback][linear]") {
    grid_linear_float_round<AlignMode::Global>(20261111, grid_iters(40));
    grid_linear_float_round<AlignMode::Local>(20261112, grid_iters(40));
    grid_linear_float_round<AlignMode::Glocal>(20261113, grid_iters(40));
}

TEST_CASE("Grid traceback is bit-identical to Pointers (linear, integer T)", "[grid][traceback][linear]") {
    grid_linear_int_round<AlignMode::Global>(20261114, grid_iters(30));
    grid_linear_int_round<AlignMode::Local>(20261115, grid_iters(30));
    grid_linear_int_round<AlignMode::Glocal>(20261116, grid_iters(30));
}

TEST_CASE("Grid traceback, linear: int16 in the saturating regime", "[grid][traceback][linear]") {
    // BLOSUM-like range with large values (scheme 3) at L400-900: past the quarter rule, so
    // the saturating kernels run (the fill and the grid forward alike).
    std::mt19937 rng(20261117);
    long ng = 0;
    for (long t = 0; t < grid_iters(12); ++t) {
        const AlignParams p = random_int_params(rng, 3);
        std::uniform_int_distribution<int> aa(0, 19), L(400, 900);
        std::string A, B;
        for (int k = L(rng); k > 0; --k) A += kAA[aa(rng)];
        for (char c : A) B += aa(rng) < 4 ? kAA[aa(rng)] : c;
        for (int kern : grid_backends()) {
            check_grid<GapModel::Linear, AlignMode::Global, int16_t>(A, B, p, kern, 16, ng);
            check_grid<GapModel::Linear, AlignMode::Local, int16_t>(A, B, p, kern, 16, ng);
            check_grid<GapModel::Linear, AlignMode::Glocal, int16_t>(A, B, p, kern, 16, ng);
        }
    }
    REQUIRE(ng > 0);
}

TEST_CASE("auto takes Grid for linear gaps past hb_cutoff", "[grid][traceback][linear]") {
    // run_tb_ (kLinearGridAuto): linear `auto` (Pointers) runs Grid on a simd level past
    // hb_cutoff (512 rows); the H / Pointers fill below it and on the scalar backend.
    std::mt19937 rng(20261118);
    const AlignParams p = random_int_params(rng, 0);
    auto run = [&](auto& al, TracebackMode tb, int kern, const std::string& a, const std::string& b) {
        al.alloc_buf();
        al.set_kernel(kern);
        al.set_traceback(tb);
        al.set_problem(a, b, p);
        al.compute_viterbi();
    };
    std::uniform_int_distribution<int> aa(0, 19);
    std::string A, B;
    for (int i = 0; i < 800; ++i) A += kAA[aa(rng)];
    for (char c : A) B += aa(rng) < 5 ? kAA[aa(rng)] : c;
    const std::string S = A.substr(0, 300);
    for (int kern : grid_backends()) {
        const bool simd = kern != kBackendScalar;
        Aligner<GapModel::Linear, AlignMode::Global, AlignBand::Full, float> g, gp, sh;
        run(g, TracebackMode::Default, kern, A, B); run(gp, TracebackMode::Pointers, kern, A, B);
        REQUIRE(g.ran_grid() == simd);
        REQUIRE(same_bits(g.score(), gp.score()));
        REQUIRE(g.alignment() == gp.alignment());
        run(sh, TracebackMode::Default, kern, S, B);
        REQUIRE_FALSE(sh.ran_grid());
        Aligner<GapModel::Linear, AlignMode::Local, AlignBand::Full, int32_t> l, lp;
        run(l, TracebackMode::Default, kern, A, B); run(lp, TracebackMode::Pointers, kern, A, B);
        REQUIRE(l.ran_grid() == simd);
        REQUIRE(l.score() == lp.score());
        REQUIRE(l.alignment() == lp.alignment());
        // int16: per-level floors (linear_grid_min_rows): x86 below AVX-512 Global from 800 rows,
        // Local from 1500; A has 800 rows
        Aligner<GapModel::Linear, AlignMode::Global, AlignBand::Full, int16_t> g16, g16s;
        Aligner<GapModel::Linear, AlignMode::Local, AlignBand::Full, int16_t> l16;
        run(g16, TracebackMode::Default, kern, A, B);
        run(g16s, TracebackMode::Default, kern, A.substr(0, 700), B);
        run(l16, TracebackMode::Default, kern, A, B);
#if defined(__aarch64__) || defined(_M_ARM64)
        REQUIRE_FALSE(g16.ran_grid());
#else
        REQUIRE(g16.ran_grid() == (simd && !is_avx512(kern)));
#endif
        REQUIRE_FALSE(g16s.ran_grid());
        REQUIRE_FALSE(l16.ran_grid());
        Aligner<GapModel::Linear, AlignMode::Glocal, AlignBand::Full, double> c, cp;
        run(c, TracebackMode::Default, kern, A, B); run(cp, TracebackMode::Pointers, kern, A, B);
        REQUIRE(c.ran_grid() == simd);
        REQUIRE(c.alignment() == cp.alignment());
    }
}
