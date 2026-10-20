// Tracks (aligner.hpp, "Tracks"): K parameter sets, each DP row scored by its set.
//
// Two contracts, asserted on bit patterns:
//   * every leveled kernel is BIT-EXACT with the scalar tracked fill — tables, score,
//     gradient, path — at both precisions, Full and banded, Scores and Pointers, the
//     row-wise fill, score-only, and with the shared profile or the per-row one;
//   * a track whose sets all hold the same values IS the untracked problem: the same
//     tables bit for bit (the arrays, the column-0 border's run rule and the stacked block
//     are built for it), whatever the track and backend.
// The scalar tracked fill itself is checked against an exhaustive oracle from Python
// (tests/Python/test_tracks.py).  Run once per ISA level by ctest (NWGRAD_ISA).

#include "catch.hpp"
#include "align_params.hpp"
#include "aligner.hpp"
#include "batch_engine.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <random>
#include <span>
#include <tuple>
#include <string>
#include <type_traits>
#include <vector>

namespace {

template <class T> using Bits = std::conditional_t<sizeof(T) == 8, uint64_t, uint32_t>;

std::string random_seq(std::mt19937_64& rng, const Alphabet& alpha, int len) {
    std::uniform_int_distribution<int> d(0, alpha.size() - 1);
    std::string s(static_cast<size_t>(len), 'A');
    for (char& c : s) c = alpha.symbol_at(d(rng));
    return s;
}

// K sets over `alpha`: integral (ties everywhere) or real-valued; all equal to set 0 when
// `equal`.  Gaps positive and non-representable unless integral.
AlignParams random_sets(std::mt19937_64& rng, const Alphabet& alpha, int K, bool integral, bool equal) {
    const int N = alpha.size();
    std::vector<double> mats(static_cast<size_t>(K * N * N)), gaps(static_cast<size_t>(4 * K));
    std::uniform_int_distribution<int> di(-3, 4);
    std::uniform_int_distribution<int> dg(0, 3);
    std::normal_distribution<double> dn(0.0, 1.5);
    std::uniform_real_distribution<double> du(0.1, 2.0);
    for (int k = 0; k < K; ++k) {
        for (int c = 0; c < N * N; ++c)
            mats[static_cast<size_t>(k * N * N + c)] =
                (equal && k) ? mats[static_cast<size_t>(c)] : (integral ? di(rng) : dn(rng));
        for (int f = 0; f < 4; ++f)
            gaps[static_cast<size_t>(4 * k + f)] =
                (equal && k) ? gaps[static_cast<size_t>(f)] : (integral ? dg(rng) : du(rng));
    }
    return AlignParams(SubstMatrix(mats.data(), alpha, K), gaps.data());
}

std::vector<uint16_t> random_track(std::mt19937_64& rng, size_t len, int K) {
    std::uniform_int_distribution<int> d(0, K - 1);
    std::vector<uint16_t> t(len);
    for (auto& x : t) x = static_cast<uint16_t>(d(rng));
    return t;
}

template <class T>
struct Run {
    double score = 0, score_only = 0;
    std::vector<std::vector<T>> tables;   // row-major (Scores only)
    std::vector<double> grad;             // the K-set gradient, flattened
    std::string aa, bb;
};

std::vector<double> flatten(const AlignParams& g) {
    const size_t nn = static_cast<size_t>(g.matrix.size()) * g.matrix.size() * g.num_sets();
    std::vector<double> out(g.matrix.data(), g.matrix.data() + nn);
    const auto gp = g.all_gaps();
    out.insert(out.end(), gp.begin(), gp.end());
    return out;
}

template <GapModel GM, AlignMode AM, AlignBand AB, class T>
Run<T> run(const AlignParams& p, const std::string& a, const std::string& b, int band,
           std::span<const uint16_t> track, std::span<const uint16_t> gtrack, int backend,
           TracebackMode tb, bool rowwise = false) {
    Aligner<GM, AM, AB, T> al;
    al.set_kernel(backend);
    al.set_traceback(tb);
    al.set_rowwise_full(rowwise);
    al.set_problem(a, b, p, band, {}, track, gtrack);
    DpBufferT<T> buf;
    al.compute_viterbi(buf);
    Run<T> r;
    r.score = al.score();
    AlignParams g = AlignParams::zeros_like(p);
    al.hard_grad(buf, g);
    r.grad = flatten(g);
    std::tie(r.aa, r.bb) = al.aligned(buf);
    if (tb == TracebackMode::Scores) {
        if constexpr (GM == GapModel::Linear) r.tables.push_back(al.to_row_major(buf.H));
        else {
            r.tables.push_back(al.to_row_major(buf.VM));
            r.tables.push_back(al.to_row_major(buf.VX));
            r.tables.push_back(al.to_row_major(buf.VY));
        }
    }
    DpBufferT<T> buf2;
    r.score_only = al.compute_score(buf2);
    return r;
}

template <class T>
bool same_bits(const std::vector<T>& x, const std::vector<T>& y) {
    if (x.size() != y.size()) return false;
    for (size_t k = 0; k < x.size(); ++k)
        if (std::bit_cast<Bits<T>>(x[k]) != std::bit_cast<Bits<T>>(y[k])) return false;
    return true;
}
bool same_bits_d(double x, double y) { return std::bit_cast<uint64_t>(x) == std::bit_cast<uint64_t>(y); }

// Score-only may differ from the fill only in the sign of a zero score.
bool score_only_ok(double fill, double only) { return fill == only; }

template <class T>
void require_same(const Run<T>& x, const Run<T>& y, bool tables) {
    if (tables) {
        REQUIRE(x.tables.size() == y.tables.size());
        for (size_t t = 0; t < x.tables.size(); ++t) REQUIRE(same_bits(x.tables[t], y.tables[t]));
    }
    REQUIRE(same_bits_d(x.score, y.score));
    REQUIRE(same_bits(x.grad, y.grad));
    REQUIRE(x.aa == y.aa);
    REQUIRE(x.bb == y.bb);
    REQUIRE(score_only_ok(x.score, x.score_only));
    REQUIRE(score_only_ok(y.score, y.score_only));
}

// One problem: every backend and traceback against the scalar Scores fill.
template <GapModel GM, AlignMode AM, AlignBand AB, class T>
void check_problem(const AlignParams& p, const std::string& a, const std::string& b, int band,
                   std::span<const uint16_t> track, std::span<const uint16_t> gtrack) {
    const auto ref = run<GM, AM, AB, T>(p, a, b, band, track, gtrack, kBackendScalar, TracebackMode::Scores);
    const auto simd = run<GM, AM, AB, T>(p, a, b, band, track, gtrack, kBackendAuto, TracebackMode::Scores);
    require_same(ref, simd, true);
    if constexpr (AB == AlignBand::Full) {
        for (int be : {kBackendScalar, kBackendAuto}) {
            const auto ptr = run<GM, AM, AB, T>(p, a, b, band, track, gtrack, be, TracebackMode::Pointers);
            require_same(ref, ptr, false);
        }
        if constexpr (GM == GapModel::Affine && std::is_same_v<T, double>) {
            const auto rw = run<GM, AM, AB, T>(p, a, b, band, track, gtrack, kBackendAuto,
                                               TracebackMode::Scores, /*rowwise=*/true);
            require_same(ref, rw, true);
        }
    }
}

template <GapModel GM, AlignMode AM, class T>
void check_modes(const AlignParams& p, const std::string& a, const std::string& b,
                 std::span<const uint16_t> track, std::span<const uint16_t> gtrack) {
    check_problem<GM, AM, AlignBand::Full, T>(p, a, b, 0, track, gtrack);
    for (int band : {1, 5})
        check_problem<GM, AM, AlignBand::GuideBanded, T>(p, a, b, band, track, gtrack);
}

template <class T>
void check_all(const AlignParams& p, const std::string& a, const std::string& b,
               std::span<const uint16_t> track, std::span<const uint16_t> gtrack) {
    check_modes<GapModel::Linear, AlignMode::Global, T>(p, a, b, track, gtrack);
    check_modes<GapModel::Linear, AlignMode::Local, T>(p, a, b, track, gtrack);
    check_modes<GapModel::Affine, AlignMode::Global, T>(p, a, b, track, gtrack);
    check_modes<GapModel::Affine, AlignMode::Local, T>(p, a, b, track, gtrack);
}

// RAII: the shared-profile cap, restored on exit.
struct ProfileCap {
    size_t old;
    explicit ProfileCap(size_t v) : old(track_profile_bytes().load()) { track_profile_bytes().store(v); }
    ~ProfileCap() { track_profile_bytes().store(old); }
};

template <class T>
void sweep(uint64_t seed, const Alphabet& alpha, int count, int maxlen) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> dl(0, maxlen), dk(1, 6), dc(0, 1);
    for (int it = 0; it < count; ++it) {
        const int m = dl(rng), n = dl(rng), K = dk(rng);
        const std::string a = random_seq(rng, alpha, m), b = random_seq(rng, alpha, n);
        const AlignParams p = random_sets(rng, alpha, K, dc(rng), false);
        const auto track = random_track(rng, static_cast<size_t>(m), K);
        const auto gtrack = random_track(rng, static_cast<size_t>(m) + 1, K);
        INFO("seed " << seed << " it " << it << " m " << m << " n " << n << " K " << K);
        check_all<T>(p, a, b, track, dc(rng) ? std::span<const uint16_t>(gtrack) : std::span<const uint16_t>());
    }
}

}  // namespace

TEST_CASE("tracked kernels are bit-exact with the scalar tracked fill", "[tracks]") {
    for (const size_t cap : {size_t(1) << 20, size_t(0)}) {   // shared profile, then per-row
        ProfileCap pc(cap);
        INFO("profile cap " << cap);
        sweep<double>(1, Alphabet::get("ACGT"), 60, 40);
        sweep<float>(2, Alphabet::get("ACGT"), 60, 40);
        sweep<double>(3, Alphabet::get(AA_ORDER), 25, 70);
        sweep<float>(4, Alphabet::get(AA_ORDER), 25, 70);
    }
}

TEST_CASE("tracked kernels: degenerate shapes", "[tracks]") {
    std::mt19937_64 rng(5);
    const Alphabet& alpha = Alphabet::get("ACGT");
    for (int m : {0, 1, 2, 7})
        for (int n : {0, 1, 2, 9}) {
            const std::string a = random_seq(rng, alpha, m), b = random_seq(rng, alpha, n);
            const AlignParams p = random_sets(rng, alpha, 3, false, false);
            const auto track = random_track(rng, static_cast<size_t>(m), 3);
            const auto gtrack = random_track(rng, static_cast<size_t>(m) + 1, 3);
            check_all<double>(p, a, b, track, gtrack);
            check_all<float>(p, a, b, track, gtrack);
        }
}

// Equal sets: the tracked problem is the untracked one, table for table.
template <GapModel GM, AlignMode AM, AlignBand AB, class T>
void check_equal_sets(const AlignParams& p1, const AlignParams& pk, const std::string& a,
                      const std::string& b, int band, std::span<const uint16_t> track,
                      std::span<const uint16_t> gtrack) {
    for (int be : {kBackendScalar, kBackendAuto}) {
        const auto u = run<GM, AM, AB, T>(p1, a, b, band, {}, {}, be, TracebackMode::Scores);
        const auto t = run<GM, AM, AB, T>(pk, a, b, band, track, gtrack, be, TracebackMode::Scores);
        REQUIRE(u.tables.size() == t.tables.size());
        for (size_t k = 0; k < u.tables.size(); ++k) REQUIRE(same_bits(u.tables[k], t.tables[k]));
        REQUIRE(same_bits_d(u.score, t.score));
        REQUIRE(u.score_only == t.score_only);
        REQUIRE(u.aa == t.aa);
        REQUIRE(u.bb == t.bb);
    }
}

template <class T>
void equal_sets_sweep(uint64_t seed, const Alphabet& alpha, int count, int maxlen) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> dl(0, maxlen), dk(1, 5), dc(0, 1);
    for (int it = 0; it < count; ++it) {
        const int m = dl(rng), n = dl(rng), K = dk(rng);
        const std::string a = random_seq(rng, alpha, m), b = random_seq(rng, alpha, n);
        const AlignParams pk = random_sets(rng, alpha, K, dc(rng), /*equal=*/true);
        const AlignParams p1 = pk.set(0);
        const auto track = random_track(rng, static_cast<size_t>(m), K);
        const auto gtrack = random_track(rng, static_cast<size_t>(m) + 1, K);
        INFO("seed " << seed << " it " << it);
        check_equal_sets<GapModel::Linear, AlignMode::Global, AlignBand::Full, T>(p1, pk, a, b, 0, track, gtrack);
        check_equal_sets<GapModel::Linear, AlignMode::Local, AlignBand::Full, T>(p1, pk, a, b, 0, track, gtrack);
        check_equal_sets<GapModel::Affine, AlignMode::Global, AlignBand::Full, T>(p1, pk, a, b, 0, track, gtrack);
        check_equal_sets<GapModel::Affine, AlignMode::Local, AlignBand::Full, T>(p1, pk, a, b, 0, track, gtrack);
        check_equal_sets<GapModel::Linear, AlignMode::Global, AlignBand::GuideBanded, T>(p1, pk, a, b, 3, track, gtrack);
        check_equal_sets<GapModel::Affine, AlignMode::Global, AlignBand::GuideBanded, T>(p1, pk, a, b, 3, track, gtrack);
        check_equal_sets<GapModel::Affine, AlignMode::Local, AlignBand::GuideBanded, T>(p1, pk, a, b, 3, track, gtrack);
    }
}

TEST_CASE("equal sets reproduce the untracked tables bit for bit", "[tracks]") {
    equal_sets_sweep<double>(11, Alphabet::get("ACGT"), 50, 40);
    equal_sets_sweep<float>(12, Alphabet::get("ACGT"), 50, 40);
    equal_sets_sweep<double>(13, Alphabet::get(AA_ORDER), 20, 60);
    equal_sets_sweep<float>(14, Alphabet::get(AA_ORDER), 20, 60);
}

TEST_CASE("track validation", "[tracks]") {
    const Alphabet& alpha = Alphabet::get("ACGT");
    std::mt19937_64 rng(9);
    const AlignParams p = random_sets(rng, alpha, 2, false, false);
    Aligner<GapModel::Affine, AlignMode::Global> al;
    const std::vector<uint16_t> t1{0}, t3{0, 1, 2}, g2{0, 0};
    REQUIRE_THROWS_AS(al.set_problem("AC", "AC", p, 0, {}, t1, {}), std::invalid_argument);
    REQUIRE_THROWS_AS(al.set_problem("AC", "AC", p, 0, {}, {}, g2), std::invalid_argument);
    const std::vector<uint16_t> t2bad{0, 2};
    REQUIRE_THROWS_AS(al.set_problem("AC", "AC", p, 0, {}, t2bad, {}), std::invalid_argument);
    REQUIRE_THROWS_AS(al.set_problem("ACG", "AC", p, 0, {}, t3, {}), std::invalid_argument);
    AlignParams g1(alpha);   // a one-set gradient for a two-set problem
    const std::vector<uint16_t> t2{0, 1};
    al.set_problem("AC", "AC", p, 0, {}, t2, {});
    al.alloc_buf();
    al.compute_viterbi();
    REQUIRE_THROWS_AS(al.hard_grad(g1), std::invalid_argument);
}

// ── Hirschberg with tracks ─────────────────────────────────────────────────────
//
// The reverse sweeps run on rewritten rows (Aligner::hb_reverse_rows_), the join refunds
// the open of the first row below the cut.  Checked with a small hb_cutoff so the
// recursion really splits: the path's score is the optimum (Pointers'), up to the
// floating-point slack Hirschberg has always had; the gradient dotted with the params is
// that score; scalar and simd are bit-identical; equal sets are the untracked run.
namespace {

template <GapModel GM, AlignMode AM, class T>
Run<T> run_hb(const AlignParams& p, const std::string& a, const std::string& b,
              std::span<const uint16_t> track, std::span<const uint16_t> gtrack, int backend,
              TracebackMode tb, int cutoff) {
    Aligner<GM, AM, AlignBand::Full, T> al;
    al.set_kernel(backend);
    al.set_traceback(tb);
    al.set_hb_cutoff(cutoff);
    al.set_problem(a, b, p, 0, {}, track, gtrack);
    DpBufferT<T> buf;
    al.compute_viterbi(buf);
    Run<T> r;
    r.score = al.score();
    AlignParams g = AlignParams::zeros_like(p);
    al.hard_grad(buf, g);
    r.grad = flatten(g);
    std::tie(r.aa, r.bb) = al.aligned(buf);
    return r;
}

double dot(const std::vector<double>& g, const AlignParams& p) {
    const auto f = flatten(p);
    double s = 0;
    for (size_t k = 0; k < g.size(); ++k) s += g[k] * f[k];
    return s;
}

template <GapModel GM, AlignMode AM, class T>
void check_hb(const AlignParams& p, const std::string& a, const std::string& b,
              std::span<const uint16_t> track, std::span<const uint16_t> gtrack) {
    const double tol = std::is_same_v<T, double> ? 1e-9 : 2e-3;
    const auto ptr = run_hb<GM, AM, T>(p, a, b, track, gtrack, kBackendScalar, TracebackMode::Pointers, 1);
    for (TracebackMode tb : {TracebackMode::Hirschberg, TracebackMode::HirschbergPmax}) {
        for (int cutoff : {1, 3}) {
            const auto sc = run_hb<GM, AM, T>(p, a, b, track, gtrack, kBackendScalar, tb, cutoff);
            const auto sv = run_hb<GM, AM, T>(p, a, b, track, gtrack, kBackendAuto, tb, cutoff);
            INFO("tb " << int(tb) << " cutoff " << cutoff << " a " << a << " b " << b);
            REQUIRE(same_bits_d(sc.score, sv.score));
            REQUIRE(same_bits(sc.grad, sv.grad));
            REQUIRE(sc.aa == sv.aa);
            REQUIRE(sc.bb == sv.bb);
            // optimal: never above the optimum, the exact sweep within rounding of it
            REQUIRE(sc.score <= ptr.score + tol);
            if (tb == TracebackMode::Hirschberg || GM == GapModel::Linear)
                REQUIRE(sc.score >= ptr.score - tol);
            else
                REQUIRE(sc.score >= ptr.score - 100 * tol);
            REQUIRE(std::abs(dot(sc.grad, p) - sc.score) <= tol * (1 + std::abs(sc.score)));
        }
    }
}

template <class T>
void hb_sweep_tracks(uint64_t seed, const Alphabet& alpha, int count, int maxlen) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> dl(0, maxlen), dk(1, 5), dc(0, 1);
    for (int it = 0; it < count; ++it) {
        const int m = dl(rng), n = dl(rng), K = dk(rng);
        const std::string a = random_seq(rng, alpha, m), b = random_seq(rng, alpha, n);
        const AlignParams p = random_sets(rng, alpha, K, dc(rng), false);   // gaps >= 0
        const auto track = random_track(rng, static_cast<size_t>(m), K);
        const auto gtrack = random_track(rng, static_cast<size_t>(m) + 1, K);
        INFO("seed " << seed << " it " << it << " K " << K);
        check_hb<GapModel::Affine, AlignMode::Global, T>(p, a, b, track, gtrack);
        check_hb<GapModel::Affine, AlignMode::Local, T>(p, a, b, track, gtrack);
        check_hb<GapModel::Linear, AlignMode::Global, T>(p, a, b, track, gtrack);
        check_hb<GapModel::Linear, AlignMode::Local, T>(p, a, b, track, gtrack);
    }
}

template <class T>
void hb_equal_sets(uint64_t seed, const Alphabet& alpha, int count, int maxlen) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> dl(0, maxlen), dk(1, 5), dc(0, 1);
    for (int it = 0; it < count; ++it) {
        const int m = dl(rng), n = dl(rng), K = dk(rng);
        const std::string a = random_seq(rng, alpha, m), b = random_seq(rng, alpha, n);
        const AlignParams pk = random_sets(rng, alpha, K, dc(rng), true);
        const AlignParams p1 = pk.set(0);
        const auto track = random_track(rng, static_cast<size_t>(m), K);
        const auto gtrack = random_track(rng, static_cast<size_t>(m) + 1, K);
        for (int be : {kBackendScalar, kBackendAuto})
            for (TracebackMode tb : {TracebackMode::Hirschberg, TracebackMode::HirschbergPmax}) {
                const auto u = run_hb<GapModel::Affine, AlignMode::Global, T>(p1, a, b, {}, {}, be, tb, 2);
                const auto t = run_hb<GapModel::Affine, AlignMode::Global, T>(pk, a, b, track, gtrack, be, tb, 2);
                REQUIRE(same_bits_d(u.score, t.score));
                REQUIRE(u.aa == t.aa);
                const auto ul = run_hb<GapModel::Affine, AlignMode::Local, T>(p1, a, b, {}, {}, be, tb, 2);
                const auto tl = run_hb<GapModel::Affine, AlignMode::Local, T>(pk, a, b, track, gtrack, be, tb, 2);
                REQUIRE(same_bits_d(ul.score, tl.score));
                REQUIRE(ul.aa == tl.aa);
                const auto un = run_hb<GapModel::Linear, AlignMode::Global, T>(p1, a, b, {}, {}, be, tb, 2);
                const auto tn = run_hb<GapModel::Linear, AlignMode::Global, T>(pk, a, b, track, gtrack, be, tb, 2);
                REQUIRE(same_bits_d(un.score, tn.score));
                REQUIRE(un.aa == tn.aa);
            }
    }
}

}  // namespace

TEST_CASE("tracked Hirschberg: optimal, scalar == simd, gradient is the path's", "[tracks]") {
    hb_sweep_tracks<double>(21, Alphabet::get("ACGT"), 40, 30);
    hb_sweep_tracks<float>(22, Alphabet::get("ACGT"), 40, 30);
    hb_sweep_tracks<double>(23, Alphabet::get(AA_ORDER), 15, 40);
    hb_sweep_tracks<float>(24, Alphabet::get(AA_ORDER), 15, 40);
}

TEST_CASE("tracked Hirschberg: equal sets are the untracked recursion", "[tracks]") {
    hb_equal_sets<double>(31, Alphabet::get("ACGT"), 40, 30);
    hb_equal_sets<float>(32, Alphabet::get("ACGT"), 40, 30);
}

namespace {
// A tracked batch on 8 threads (the inter-pair fill), against each pair's own Aligner:
// whole-batch fills, every pair recomputed alone over and over (its stored gradient
// rewritten each time), wide bands, and the weighted sum after each — for the
// sanitizers as much as for the values.  In either store of the hard gradients.
template <GapModel GM, AlignMode AM, class T>
void check_batch(uint64_t seed, bool counts) {
    std::mt19937_64 rng(seed);
    const Alphabet& alpha = Alphabet::get("ACGT");
    const int K = 4;
    const AlignParams p = random_sets(rng, alpha, K, false, false);
    const size_t N = 400;
    std::uniform_int_distribution<int> dl(0, 60);
    std::vector<std::string> A, B;
    std::vector<std::vector<uint16_t>> tk, gk;
    for (size_t i = 0; i < N; ++i) {
        A.push_back(random_seq(rng, alpha, dl(rng)));
        B.push_back(random_seq(rng, alpha, dl(rng)));
        tk.push_back(random_track(rng, A.back().size(), K));
        gk.push_back(random_track(rng, A.back().size() + 1, K));
    }
    std::vector<std::string_view> Av(A.begin(), A.end()), Bv(B.begin(), B.end());
    BatchEngine<T, GM, AM> eng(BatchSettings{8, TracebackMode::Pointers}, GradMode::Hard, Av, Bv, p,
                               kBackendAuto, &tk, &gk);
    eng.set_fill(false, true);
    eng.set_track_counts(counts);
    std::vector<Run<T>> ref;
    for (size_t i = 0; i < N; ++i)
        ref.push_back(run<GM, AM, AlignBand::Full, T>(p, A[i], B[i], 0, tk[i], gk[i], kBackendAuto,
                                                      TracebackMode::Pointers));
    std::vector<double> w(N);
    for (auto& x : w) x = std::uniform_real_distribution<double>(-2, 2)(rng);
    auto check = [&]() {
        std::vector<double> acc;
        for (size_t i = 0; i < N; ++i) {
            INFO("pair " << i);
            REQUIRE(same_bits_d(eng.score(i), ref[i].score));
            AlignParams g(alpha, K);
            eng.grad(i, g);
            const std::vector<double> f = flatten(g);
            REQUIRE(same_bits(f, ref[i].grad));
            if (acc.empty()) acc.assign(f.size(), 0.0);
            for (size_t k = 0; k < f.size(); ++k) acc[k] += w[i] * f[k];   // one block: pair order
        }
        REQUIRE(same_bits(flatten(eng.weighted_grad(w.data(), N)), acc));
    };
    for (int round = 0; round < 3; ++round) {
        eng.score_and_grad();
        check();
        for (int rep = 0; rep < 5; ++rep)
            for (size_t i = 0; i < N; ++i) eng.score_and_grad_one(i, false);
        check();
        eng.banded_grad(128);
        check();
    }
}
}  // namespace

TEST_CASE("tracked batch: threaded fills, single-pair recomputes, weighted sums", "[tracks]") {
    for (bool counts : {false, true}) {   // the two stores of the hard gradients
        INFO("counts " << counts);
        check_batch<GapModel::Linear, AlignMode::Global, double>(41, counts);
        check_batch<GapModel::Linear, AlignMode::Local, double>(42, counts);
        check_batch<GapModel::Affine, AlignMode::Global, double>(43, counts);
        check_batch<GapModel::Affine, AlignMode::Local, double>(44, counts);
        check_batch<GapModel::Affine, AlignMode::Global, float>(45, counts);
        check_batch<GapModel::Linear, AlignMode::Local, float>(46, counts);
    }
}
