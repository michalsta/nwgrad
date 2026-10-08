// ── Score only (Aligner::compute_score): bit-identical to the exact fill ─────────
//
// compute_score() keeps rolling rows instead of tables and returns the optimal score.
// Its contract is bit-identity with score() after compute_viterbi() under an exact
// traceback mode (Pointers and Scores both checked) — at every backend the CPU offers,
// both precisions, Global and Local, affine and linear (linear runs through the affine
// kernel with zero opens).  Random problems mix ties (small-integer matrices and gaps),
// non-representable extends (0.1), empty sequences and planted homology (long gap runs
// for lazy-F to carry across lanes).  NWGRAD_SCORE_FUZZ=<n> scales the iteration count
// (the fleet runs use it); NWGRAD_SCORE_VARIANT=twopass checks the unfused kernel.

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

std::vector<int> score_backends() {
    std::vector<int> v{kBackendScalar};
    for (SimdLevel l : detect_available_levels()) v.push_back(static_cast<int>(l));
    return v;
}

long fuzz_iters(long dflt) {
    const char* e = std::getenv("NWGRAD_SCORE_FUZZ");
    return e ? std::atol(e) : dflt;
}

bool same_bits(double x, double y) { return std::memcmp(&x, &y, sizeof x) == 0; }

template <GapModel GM, AlignMode AM, class T>
void check_one(const std::string& a, const std::string& b, const AlignParams& p, int kern,
               long& n_checked) {
    for (TracebackMode tb : {TracebackMode::Pointers, TracebackMode::Scores}) {
        Aligner<GM, AM, AlignBand::Full, T> ref;
        ref.set_kernel(kern);
        ref.set_traceback(tb);
        DpBufferT<T> rbuf;
        ref.set_problem(a, b, p);
        ref.compute_viterbi(rbuf);
        const double want = ref.score();

        Aligner<GM, AM, AlignBand::Full, T> al;
        al.set_kernel(kern);
        DpBufferT<T> buf;
        al.set_problem(a, b, p);
        const double got = al.compute_score(buf);
        INFO("a=" << a << " b=" << b << " kern=" << kern << " tb=" << static_cast<int>(tb)
             << " T=" << sizeof(T) << " GM=" << static_cast<int>(GM)
             << " AM=" << static_cast<int>(AM) << " want=" << want << " got=" << got);
        REQUIRE(same_bits(got, want));
        ++n_checked;
    }
}

template <class T>
void check_all(const std::string& a, const std::string& b, const AlignParams& p, int kern,
               long& n) {
    check_one<GapModel::Affine, AlignMode::Global, T>(a, b, p, kern, n);
    check_one<GapModel::Affine, AlignMode::Local,  T>(a, b, p, kern, n);
    check_one<GapModel::Linear, AlignMode::Global, T>(a, b, p, kern, n);
    check_one<GapModel::Linear, AlignMode::Local,  T>(a, b, p, kern, n);
}

}  // namespace

TEST_CASE("compute_score is bit-identical to the exact fill's score", "[score]") {
    std::mt19937_64 rng(20261008);
    auto ri = [&](int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi)(rng); };
    const Alphabet* alphas[2] = {&Alphabet::dna(), &Alphabet::protein()};
    const std::vector<int> kerns = score_backends();
    const long iters = fuzz_iters(300);
    long n_checked = 0;
    for (long it = 0; it < iters; ++it) {
        const Alphabet& A = *alphas[ri(0, 2) == 0];
        const int na = A.size();
        AlignParams p(A);
        const int style = ri(0, 2);   // 0 integer (ties), 1 real, 2 integer + 0.1 extends
        for (int x = 0; x < na; ++x)
            for (int y = 0; y < na; ++y)
                p.matrix.at(x, y) = style == 1
                    ? std::uniform_real_distribution<double>(-2, 3)(rng)
                    : static_cast<double>(x == y ? ri(1, 3) : ri(-2, 1));
        auto gap = [&](bool ext) {
            if (style == 1) return std::uniform_real_distribution<double>(0, ext ? 1.5 : 4)(rng);
            if (style == 2 && ext) return 0.1 * ri(1, 9);
            return static_cast<double>(ri(0, ext ? 2 : 4));
        };
        p.gap_open_a = gap(false); p.gap_extend_a = gap(true);
        p.gap_open_b = gap(false); p.gap_extend_b = gap(true);
        auto seq = [&](int len) {
            std::string s;
            for (int k = 0; k < len; ++k) s.push_back(A.symbols()[ri(0, na - 1)]);
            return s;
        };
        // lengths: some empty / tiny, most up to ~90 (several segments at every width),
        // a few long enough for many lazy-F rounds
        auto len = [&] {
            const int r = ri(0, 19);
            if (r == 0) return ri(0, 2);
            if (r == 1) return ri(150, 400);
            return ri(1, 90);
        };
        std::string a = seq(len()), b = seq(len());
        if (ri(0, 2) == 0 && !a.empty()) {   // planted homology: b = a with indels
            b = a;
            for (int k = ri(1, 6); k > 0 && !b.empty(); --k) {
                if (ri(0, 1)) b.erase(ri(0, (int)b.size() - 1), ri(1, 8));
                else          b.insert(ri(0, (int)b.size()), seq(ri(1, 8)));
            }
        }
        const int kern = kerns[ri(0, (int)kerns.size() - 1)];
        check_all<double>(a, b, p, kern, n_checked);
        check_all<float>(a, b, p, kern, n_checked);
    }
    INFO(n_checked);
    REQUIRE(n_checked == iters * 2 * 4 * 2);
}

TEST_CASE("compute_score: edge shapes at every backend", "[score]") {
    AlignParams p(Alphabet::dna());
    for (int x = 0; x < 4; ++x)
        for (int y = 0; y < 4; ++y) p.matrix.at(x, y) = x == y ? 2.0 : -1.0;
    p.gap_open_a = 3.0; p.gap_extend_a = 0.1; p.gap_open_b = 2.5; p.gap_extend_b = 0.7;
    long n = 0;
    const std::vector<std::pair<std::string, std::string>> cases = {
        {"", ""}, {"A", ""}, {"", "A"}, {"ACGT", ""}, {"", "ACGTACGT"}, {"A", "A"},
        {"A", "C"}, {"ACGTACGTACGTACGTACGT", "A"}, {"A", "ACGTACGTACGTACGTACGTACGTACGTACGT"},
        // one residue past each multiple of the common lane counts, both sides
        {std::string(17, 'A'), std::string(33, 'A')}, {std::string(65, 'C'), std::string(9, 'C')},
        {"ACGTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTACGT", "ACGTACGT"},
    };
    for (int kern : score_backends())
        for (const auto& [a, b] : cases) {
            check_all<double>(a, b, p, kern, n);
            check_all<float>(a, b, p, kern, n);
        }
    REQUIRE(n == static_cast<long>(score_backends().size() * cases.size() * 16));
}

TEST_CASE("compute_score leaves the Viterbi state alone", "[score]") {
    AlignParams p(Alphabet::dna());
    for (int x = 0; x < 4; ++x) p.matrix.at(x, x) = 1.0;
    p.gap_open_a = p.gap_open_b = 1.0; p.gap_extend_a = p.gap_extend_b = 0.5;
    Aligner<GapModel::Affine, AlignMode::Global> al;
    al.alloc_buf();
    al.set_problem("ACGT", "ACGT", p);
    REQUIRE_THROWS(al.score());                 // nothing computed yet ...
    const double s = al.compute_score();
    REQUIRE(s == 4.0);
    REQUIRE_THROWS(al.score());                 // ... and score-only leaves no Viterbi state
    REQUIRE_THROWS(al.aligned());
    al.compute_viterbi();
    REQUIRE(al.score() == s);
}
