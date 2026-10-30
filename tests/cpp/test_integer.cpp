// ── Integer scalar types: every hard-path result equals the double oracle ────────
//
// Aligner<..., int16_t / int32_t / int64_t> runs the Viterbi DP in integer arithmetic.
// The contract is stronger than any floating one in this suite: on integral parameters
// double arithmetic is EXACT (every value below 2^53), so the double aligner is not an
// approximation to compare against within a tolerance — it is the ground truth the
// integer aligner must reproduce to the bit: the score, the alignment, the guide, the
// hard gradient, ties included.  That holds for every gap model, mode, band, traceback
// mode (the prefix-max carry is exact in integers, so it must agree with the serial
// sweep AND with Pointers at ties only where Hirschberg's own split allows — hence
// Hirschberg is compared on score and path validity, the others on everything) and
// every backend.
//
// Then the guards: a non-integral parameter throws, a problem whose a-priori bound
// (m+n+2) * step exceeds the type's cap throws (int16 refuses it where int32 takes it),
// and the soft path throws on an integer type.

#include "catch.hpp"

#include <cmath>
#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

#include "align_params.hpp"
#include "aligner.hpp"
#include "simd_levels.hpp"

namespace {

std::vector<int> backends_under_test() {
    std::vector<int> v{kBackendScalar};
    for (SimdLevel l : detect_available_levels()) v.push_back(static_cast<int>(l));
    return v;
}

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed ? seed : 1) {}
    uint32_t next() {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        return static_cast<uint32_t>(s >> 32);
    }
    int in(int lo, int hi) { return lo + static_cast<int>(next() % static_cast<uint32_t>(hi - lo + 1)); }
};

// Small integral parameters (so int16's bound holds on every fixture), tie-heavy, with
// K sets when `sets` > 1 (a tracked problem).
AlignParams int_params(Rng& rng, const Alphabet& alpha, bool affine, int sets,
                       bool allow_negative_gaps) {
    const int n = alpha.size();
    std::vector<double> stack(static_cast<size_t>(n) * n * sets);
    for (int k = 0; k < sets; ++k)
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                stack[(static_cast<size_t>(k) * n + i) * n + j] =
                    (i == j) ? rng.in(1, 5) : rng.in(-4, 2);
    AlignParams p(SubstMatrix(stack.data(), alpha, sets));
    for (int k = 0; k < sets; ++k) {
        const int lo = allow_negative_gaps ? -2 : 0;
        p.gap_ref(k, AlignParams::kOpenA)   = affine ? rng.in(lo, 6) : 0;
        p.gap_ref(k, AlignParams::kExtendA) = rng.in(lo, 3);
        p.gap_ref(k, AlignParams::kOpenB)   = affine ? rng.in(lo, 6) : 0;
        p.gap_ref(k, AlignParams::kExtendB) = rng.in(lo, 3);
    }
    return p;
}

std::string random_seq(Rng& rng, const Alphabet& alpha, int len) {
    std::string s(static_cast<size_t>(len), alpha.symbol_at(0));
    for (char& c : s) c = alpha.symbol_at(rng.in(0, alpha.size() - 1));
    return s;
}

std::vector<uint16_t> random_track(Rng& rng, int len, int sets) {
    std::vector<uint16_t> t(static_cast<size_t>(len));
    for (auto& v : t) v = static_cast<uint16_t>(rng.in(0, sets - 1));
    return t;
}

// Replay a hard gradient's path cost: <grad, params> must equal the score.  Hirschberg
// paths may differ from the oracle's at ties, so this is their check.
double dot(const AlignParams& g, const AlignParams& p) {
    double s = 0;
    const int N = p.matrix.size();
    for (int k = 0; k < p.num_sets(); ++k) {
        const double* gb = g.matrix.data() + static_cast<size_t>(k) * N * N;
        const double* pb = p.matrix.data() + static_cast<size_t>(k) * N * N;
        for (int q = 0; q < N * N; ++q) s += gb[q] * pb[q];
        for (int f = 0; f < 4; ++f) s += g.gap(k, f) * p.gap(k, f);
    }
    return s;
}

struct Result {
    double score = 0;
    std::pair<std::string, std::string> aligned;
    std::vector<int> guide;
    AlignParams grad;
    bool threw = false;
    std::string what;
};

template <GapModel GM, AlignMode AM, AlignBand AB, class T>
Result run_one(const std::vector<uint8_t>& ea, const std::vector<uint8_t>& eb, const AlignParams& p,
               int band, const std::vector<uint16_t>& track, int backend, TracebackMode tb, int cutoff) {
    Result r{0, {}, {}, AlignParams::zeros_like(p)};
    Aligner<GM, AM, AB, T> al;
    al.set_kernel(backend);
    al.set_traceback(tb);
    al.set_hb_cutoff(cutoff);
    DpBufferT<T> buf;
    try {
        if (track.empty()) al.set_problem(ea, eb, p, band);
        else               al.set_problem(ea, eb, p, band, {}, track, {});
        al.compute_viterbi(buf);
    } catch (const std::exception& e) {
        r.threw = true; r.what = e.what();
        return r;
    }
    r.score = al.score();
    r.aligned = al.aligned(buf);
    r.guide = al.guide_j_from_viterbi(buf);
    al.hard_grad(buf, r.grad);
    return r;
}

bool same_grad(const AlignParams& a, const AlignParams& b) {
    const int N = a.matrix.size();
    for (int k = 0; k < a.num_sets(); ++k) {
        for (int q = 0; q < N * N; ++q)
            if (a.matrix.data()[static_cast<size_t>(k) * N * N + q] != b.matrix.data()[static_cast<size_t>(k) * N * N + q]) return false;
        for (int f = 0; f < 4; ++f) if (a.gap(k, f) != b.gap(k, f)) return false;
    }
    return true;
}

template <GapModel GM, AlignMode AM, AlignBand AB, class T>
void compare_case(const std::string& A, const std::string& B, const AlignParams& p, int band,
                  const std::vector<uint16_t>& track, int backend, TracebackMode tb, int cutoff,
                  int& mismatches, int& cases, std::string& first) {
    const auto ea = p.matrix.alphabet().encode(A), eb = p.matrix.alphabet().encode(B);
    const Result o = run_one<GM, AM, AB, double>(ea, eb, p, band, track, backend, tb, cutoff);
    const Result x = run_one<GM, AM, AB, T>(ea, eb, p, band, track, backend, tb, cutoff);
    ++cases;
    bool ok = !o.threw && !x.threw;
    if (ok) {
        ok = o.score == x.score;
        if (ok) {
            // <grad, params> == score, in both (the oracle's own sanity, and the integer's)
            ok = dot(x.grad, p) == x.score && dot(o.grad, p) == o.score;
        }
        if (ok && !is_hirschberg(tb)) {
            // exact modes: the same path, guide and gradient, ties included
            ok = o.aligned == x.aligned && o.guide == x.guide && same_grad(o.grad, x.grad);
        }
        if (ok && is_hirschberg(tb) && static_cast<int>(A.size()) <= cutoff) {
            // below the cutoff Hirschberg runs AS Pointers in both: exact
            ok = o.aligned == x.aligned && o.guide == x.guide && same_grad(o.grad, x.grad);
        }
    }
    if (!ok) {
        ++mismatches;
        if (first.empty()) {
            first = std::string(nw_type_name<T>()) + " backend " + std::to_string(backend) +
                    " tb " + std::to_string(static_cast<int>(tb)) + " |A|=" + std::to_string(A.size()) +
                    " |B|=" + std::to_string(B.size()) + " oracle " + std::to_string(o.score) +
                    (o.threw ? " (threw: " + o.what + ")" : "") + " int " + std::to_string(x.score) +
                    (x.threw ? " (threw: " + x.what + ")" : "");
        }
    }
}

template <class T>
void sweep(uint64_t seed, int rounds, int& mismatches, int& cases, std::string& first, int maxlen = 40) {
    Rng rng(seed);
    const Alphabet* alphas[] = {&Alphabet::get("ACGT"), &Alphabet::get("ACDEFGHIKLMNPQRSTVWY"), &Alphabet::get("AB")};
    const auto backends = backends_under_test();
    for (int r = 0; r < rounds; ++r) {
        const Alphabet& alpha = *alphas[rng.in(0, 2)];
        const bool affine = rng.in(0, 1) == 1;
        const int sets = rng.in(0, 3) == 0 ? rng.in(2, 4) : 1;
        // Hirschberg needs non-negative penalties; negative ones only on the exact modes
        const int which = rng.in(0, 4);   // 0 scores, 1 pointers, 2 hirschberg, 3 pmax, 4 banded
        const bool neg = which <= 1 && rng.in(0, 3) == 0;
        const AlignParams p = int_params(rng, alpha, affine, sets, neg);
        const int la = rng.in(0, 3) == 0 ? rng.in(0, 3) : rng.in(1, maxlen);
        const int lb = rng.in(0, 3) == 0 ? rng.in(0, 3) : rng.in(1, maxlen);
        const std::string A = random_seq(rng, alpha, la), B = random_seq(rng, alpha, lb);
        const std::vector<uint16_t> track = sets > 1 ? random_track(rng, la, sets) : std::vector<uint16_t>{};
        const int backend = backends[static_cast<size_t>(rng.in(0, static_cast<int>(backends.size()) - 1))];
        const int cutoff = rng.in(0, 1) ? 1 : rng.in(2, 8);
        const bool global = rng.in(0, 1) == 1;
        if (which == 4) {
            const int band = rng.in(1, 12);
            if (affine) {
                if (global) compare_case<GapModel::Affine, AlignMode::Global, AlignBand::GuideBanded, T>(A, B, p, band, track, backend, TracebackMode::Scores, cutoff, mismatches, cases, first);
                else        compare_case<GapModel::Affine, AlignMode::Local,  AlignBand::GuideBanded, T>(A, B, p, band, track, backend, TracebackMode::Scores, cutoff, mismatches, cases, first);
            } else {
                if (global) compare_case<GapModel::Linear, AlignMode::Global, AlignBand::GuideBanded, T>(A, B, p, band, track, backend, TracebackMode::Scores, cutoff, mismatches, cases, first);
                else        compare_case<GapModel::Linear, AlignMode::Local,  AlignBand::GuideBanded, T>(A, B, p, band, track, backend, TracebackMode::Scores, cutoff, mismatches, cases, first);
            }
            continue;
        }
        const TracebackMode tb = which == 0 ? TracebackMode::Scores : which == 1 ? TracebackMode::Pointers
                               : which == 2 ? TracebackMode::Hirschberg : TracebackMode::HirschbergPmax;
        if (affine) {
            if (global) compare_case<GapModel::Affine, AlignMode::Global, AlignBand::Full, T>(A, B, p, 0, track, backend, tb, cutoff, mismatches, cases, first);
            else        compare_case<GapModel::Affine, AlignMode::Local,  AlignBand::Full, T>(A, B, p, 0, track, backend, tb, cutoff, mismatches, cases, first);
        } else {
            if (global) compare_case<GapModel::Linear, AlignMode::Global, AlignBand::Full, T>(A, B, p, 0, track, backend, tb, cutoff, mismatches, cases, first);
            else        compare_case<GapModel::Linear, AlignMode::Local,  AlignBand::Full, T>(A, B, p, 0, track, backend, tb, cutoff, mismatches, cases, first);
        }
    }
}

}  // namespace

TEMPLATE_TEST_CASE("integer aligners reproduce the double oracle bit for bit", "[integer][simd]",
                   int16_t, int32_t, int64_t) {
    int mismatches = 0, cases = 0;
    std::string first;
    sweep<TestType>(0x1234abcd, 1500, mismatches, cases, first);
    INFO("first mismatch: " << first);
    CHECK(mismatches == 0);
    CHECK(cases == 1500);
}

// int16 past the old quarter-range bound (saturating arithmetic, the value-range bound):
// the same sweep at lengths up to 500, where (m + n + 66) * step once refused most cases
// (random |s| <= 5, gaps <= 6: step up to 12, so ~L300).  A refusal counts as a mismatch
// (double never throws), so the fixture must fit every form of the bound, the guide-
// banded Global one included (its floor is the move-count bound, (m + n + 66) * step).
TEST_CASE("int16 reproduces the double oracle on long pairs", "[integer][simd][long]") {
    int mismatches = 0, cases = 0;
    std::string first;
    sweep<int16_t>(0x5eedf00d, 60, mismatches, cases, first, 500);
    INFO("first mismatch: " << first);
    CHECK(mismatches == 0);
    CHECK(cases == 60);
}

TEST_CASE("integer aligners refuse non-integral parameters", "[integer]") {
    const Alphabet& alpha = Alphabet::get("ACGT");
    std::vector<double> m(16, -1.0);
    for (int i = 0; i < 4; ++i) m[static_cast<size_t>(i) * 5] = 2.0;
    AlignParams p(SubstMatrix(m.data(), alpha));
    p.gap_open_a = 2; p.gap_extend_a = 1; p.gap_open_b = 2; p.gap_extend_b = 1;
    const auto ea = alpha.encode("ACGTAC"), eb = alpha.encode("ACTTAC");

    Aligner<GapModel::Affine, AlignMode::Global, AlignBand::Full, int32_t> al;
    DpBufferT<int32_t> buf;
    al.set_problem(ea, eb, p); al.compute_viterbi(buf);
    Aligner<GapModel::Affine, AlignMode::Global, AlignBand::Full, double> ad;
    DpBufferT<double> bd;
    ad.set_problem(ea, eb, p); ad.compute_viterbi(bd);
    CHECK(al.score() == ad.score());

    AlignParams q = p;
    q.gap_extend_a = 0.5;
    CHECK_THROWS_AS(al.set_problem(ea, eb, q), std::invalid_argument);
    AlignParams q2 = p;
    q2.matrix.at(1, 2) = -1.25;
    CHECK_THROWS_AS(al.set_problem(ea, eb, q2), std::invalid_argument);
    AlignParams q3 = p;
    q3.gap_open_b = std::nan("");
    CHECK_THROWS_AS(al.set_problem(ea, eb, q3), std::invalid_argument);
    // a rejected problem leaves the aligner without a problem
    CHECK_THROWS_AS(al.compute_viterbi(buf), std::logic_error);
}

TEST_CASE("integer aligners refuse problems that could overflow; a wider type takes them", "[integer]") {
    const Alphabet& alpha = Alphabet::get("ACGT");
    std::vector<double> m(16, -100.0);
    for (int i = 0; i < 4; ++i) m[static_cast<size_t>(i) * 5] = 100.0;
    AlignParams p(SubstMatrix(m.data(), alpha));
    p.gap_open_a = 50; p.gap_extend_a = 10; p.gap_open_b = 50; p.gap_extend_b = 10;
    // int16 saturates and is bounded by the VALUES a path can collect: 2 * 200 * 100 + 50
    // (a Hirschberg join of two halves) exceeds 32767, so 200 x 200 is refused; 60 x 60
    // (once refused by the quarter rule, (60 + 60 + 66) * 100 > 8191) now fits.
    const auto ea = alpha.encode(std::string(200, 'A')), eb = alpha.encode(std::string(200, 'C'));
    const auto sa = alpha.encode(std::string(60, 'A')), sb = alpha.encode(std::string(60, 'C'));

    Aligner<GapModel::Affine, AlignMode::Global, AlignBand::Full, int16_t> a16;
    DpBufferT<int16_t> b16;
    CHECK_THROWS_AS(a16.set_problem(ea, eb, p), std::overflow_error);
    a16.set_problem(sa, sb, p); a16.compute_viterbi(b16);
    {
        Aligner<GapModel::Affine, AlignMode::Global, AlignBand::Full, double> ad60;
        DpBufferT<double> bd60;
        ad60.set_problem(sa, sb, p); ad60.compute_viterbi(bd60);
        CHECK(a16.score() == ad60.score());
        CHECK(a16.aligned(b16) == ad60.aligned(bd60));
    }

    Aligner<GapModel::Affine, AlignMode::Global, AlignBand::Full, int32_t> a32;
    DpBufferT<int32_t> b32;
    a32.set_problem(ea, eb, p); a32.compute_viterbi(b32);
    Aligner<GapModel::Affine, AlignMode::Global, AlignBand::Full, double> ad;
    DpBufferT<double> bd;
    ad.set_problem(ea, eb, p); ad.compute_viterbi(bd);
    CHECK(a32.score() == ad.score());
    CHECK(a32.aligned(b32) == ad.aligned(bd));

    // a parameter beyond the cap is refused as a parameter, not as a bound
    AlignParams big = p;
    big.matrix.at(0, 0) = 9000.0;
    CHECK_THROWS_AS(a16.set_problem(sa, sb, big), std::invalid_argument);
}

TEST_CASE("the soft path throws on an integer scalar type", "[integer]") {
    const Alphabet& alpha = Alphabet::get("ACGT");
    std::vector<double> m(16, -1.0);
    for (int i = 0; i < 4; ++i) m[static_cast<size_t>(i) * 5] = 2.0;
    AlignParams p(SubstMatrix(m.data(), alpha));
    p.gap_open_a = 2; p.gap_extend_a = 1; p.gap_open_b = 2; p.gap_extend_b = 1;
    const auto ea = alpha.encode("ACGTAC"), eb = alpha.encode("ACTTAC");
    Aligner<GapModel::Affine, AlignMode::Global, AlignBand::Full, int64_t> al;
    DpBufferT<int64_t> buf;
    al.set_problem(ea, eb, p);
    CHECK_THROWS_AS(al.compute_forward_back(buf), std::invalid_argument);
    al.alloc_buf();
    CHECK_THROWS_AS(al.compute_forward_back(), std::invalid_argument);
}

TEST_CASE("integer score-only equals the integer fill", "[integer][simd]") {
    Rng rng(99);
    const Alphabet& alpha = Alphabet::get("ACDEFGHIKLMNPQRSTVWY");
    int bad = 0;
    for (int r = 0; r < 200; ++r) {
        const bool affine = rng.in(0, 1) == 1;
        const AlignParams p = int_params(rng, alpha, affine, 1, rng.in(0, 3) == 0);
        const std::string A = random_seq(rng, alpha, rng.in(0, 60)), B = random_seq(rng, alpha, rng.in(0, 60));
        const auto ea = alpha.encode(A), eb = alpha.encode(B);
        for (int backend : backends_under_test()) {
            auto one = [&](auto tag_gm, auto tag_am) {
                constexpr GapModel GM = decltype(tag_gm)::value;
                constexpr AlignMode AM = decltype(tag_am)::value;
                Aligner<GM, AM, AlignBand::Full, int32_t> al;
                al.set_kernel(backend);
                al.set_traceback(TracebackMode::Pointers);
                DpBufferT<int32_t> buf;
                al.set_problem(ea, eb, p);
                const double s = al.compute_score(buf);
                al.compute_viterbi(buf);
                if (s != al.score()) ++bad;
                Aligner<GM, AM, AlignBand::Full, double> ad;
                ad.set_kernel(backend);
                DpBufferT<double> bd;
                ad.set_problem(ea, eb, p);
                if (ad.compute_score(bd) != s) ++bad;
            };
            if (affine) {
                one(std::integral_constant<GapModel, GapModel::Affine>{}, std::integral_constant<AlignMode, AlignMode::Global>{});
                one(std::integral_constant<GapModel, GapModel::Affine>{}, std::integral_constant<AlignMode, AlignMode::Local>{});
            } else {
                one(std::integral_constant<GapModel, GapModel::Linear>{}, std::integral_constant<AlignMode, AlignMode::Global>{});
                one(std::integral_constant<GapModel, GapModel::Linear>{}, std::integral_constant<AlignMode, AlignMode::Local>{});
            }
        }
    }
    CHECK(bad == 0);
}
