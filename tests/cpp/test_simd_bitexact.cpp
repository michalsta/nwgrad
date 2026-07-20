// The Simd Viterbi kernel must be BIT-EXACT with the scalar one — not close, not
// within a tolerance, but identical down to the last bit of every cell.
//
// This is not a nicety.  The tracebacks in aligner.hpp do not store direction
// pointers; they re-derive the path by exact floating-point equality against the
// stored table (`rat(H,i,j) == rat(H,i-1,j-1) + sub(i,j)`).  A kernel that shifted a
// single ULP would match no traceback branch, fall through to the final else, and
// emit a gap where a match belongs — leaving the score right and the alignment and
// the hard gradient quietly wrong.  Bit-exactness is what buys the tracebacks the
// right to keep working untouched, so it is asserted directly rather than assumed.
//
// Comparison is on the bit patterns, not with ==, so that -inf and any stray signed
// zero are caught rather than silently compared equal.

#include "catch.hpp"
#include "align_params.hpp"
#include "aligner.hpp"

#include <array>
#include <bit>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace {

// An integer-valued matrix, deliberately.  Ties are what make tie-breaking
// observable, and with an integer matrix exact ties are everywhere — which is
// exactly the condition under which a careless argmax would diverge.
AlignParams int_params(double open_a, double ext_a, double open_b, double ext_b) {
    std::array<double, 400> src{};
    for (int i = 0; i < 20; ++i)
        for (int j = 0; j < 20; ++j)
            src[static_cast<size_t>(i * 20 + j)] = (i == j) ? 4.0 : ((i + j) % 3 == 0 ? -1.0 : -2.0);

    AlignParams p(SubstMatrix(src.data()));
    p.gap_open_a   = open_a;
    p.gap_extend_a = ext_a;
    p.gap_open_b   = open_b;
    p.gap_extend_b = ext_b;
    return p;
}

std::string random_seq(std::mt19937_64& rng, int len) {
    static constexpr std::string_view AA = "ACDEFGHIKLMNPQRSTVWY";
    std::uniform_int_distribution<int> d(0, 19);
    std::string s(static_cast<size_t>(len), 'A');
    for (char& c : s) c = AA[static_cast<size_t>(d(rng))];
    return s;
}

// Bit-identity over the first `n` doubles of two tables.  Returns the index of the
// first difference, or -1.
long first_bit_diff(const std::vector<double>& a, const std::vector<double>& b, size_t n) {
    if (a.size() < n || b.size() < n) return 0;
    for (size_t k = 0; k < n; ++k)
        if (std::bit_cast<uint64_t>(a[k]) != std::bit_cast<uint64_t>(b[k]))
            return static_cast<long>(k);
    return -1;
}

// Run one problem under both kernels and demand the tables, the score and the hard
// gradient all agree exactly.
template<GapModel GM, AlignMode AM, AlignBand AB>
void check_bit_exact(const AlignParams& p, const std::string& a, const std::string& b,
                     int band = 0) {
    Aligner<GM, AM, AB> al;
    const size_t sz = (a.size() + 1) * (b.size() + 1);

    DpBuffer buf_scalar, buf_simd;

    al.set_problem(a, b, p, band);
    al.set_kernel(kBackendScalar);
    al.compute_viterbi(buf_scalar);
    const double score_scalar = al.score();
    AlignParams grad_scalar = AlignParams::zeros_like(p);
    al.hard_grad(buf_scalar, grad_scalar);
    // Snapshot the scalar tables in canonical row-major NOW, while the aligner is in
    // row-major mode — the simd run below flips it to striped for the affine Full path.
    std::vector<double> rmH, rmM, rmX, rmY;
    if constexpr (GM == GapModel::Linear) rmH = al.to_row_major(buf_scalar.H);
    else {
        rmM = al.to_row_major(buf_scalar.VM);
        rmX = al.to_row_major(buf_scalar.VX);
        rmY = al.to_row_major(buf_scalar.VY);
    }

    al.set_problem(a, b, p, band);
    al.set_kernel(kBackendAuto);
    al.compute_viterbi(buf_simd);
    const double score_simd = al.score();
    AlignParams grad_simd = AlignParams::zeros_like(p);
    al.hard_grad(buf_simd, grad_simd);

    // Compare tables in canonical row-major: the simd Full kernel writes them striped,
    // so to_row_major de-stripes buf_simd back to (m+1)×(n+1) order for the comparison.
    if constexpr (GM == GapModel::Linear) {
        REQUIRE(first_bit_diff(rmH, al.to_row_major(buf_simd.H), sz) == -1);
    } else {
        REQUIRE(first_bit_diff(rmM, al.to_row_major(buf_simd.VM), sz) == -1);
        REQUIRE(first_bit_diff(rmX, al.to_row_major(buf_simd.VX), sz) == -1);
        REQUIRE(first_bit_diff(rmY, al.to_row_major(buf_simd.VY), sz) == -1);
    }

    REQUIRE(std::bit_cast<uint64_t>(score_scalar) == std::bit_cast<uint64_t>(score_simd));

    // The gradient depends on *which* optimal path the traceback picked, so an equal
    // gradient is the real proof that tie-breaking survived vectorization.
    const int N = p.matrix.size();
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            REQUIRE(std::bit_cast<uint64_t>(grad_scalar.matrix.at(i, j)) ==
                    std::bit_cast<uint64_t>(grad_simd.matrix.at(i, j)));
    REQUIRE(grad_scalar.gap_open_a   == grad_simd.gap_open_a);
    REQUIRE(grad_scalar.gap_extend_a == grad_simd.gap_extend_a);
    REQUIRE(grad_scalar.gap_open_b   == grad_simd.gap_open_b);
    REQUIRE(grad_scalar.gap_extend_b == grad_simd.gap_extend_b);
}

// Every combination of gap model, align mode and band, for one params/sequence pair.
void check_all_modes(const AlignParams& p, const std::string& a, const std::string& b) {
    check_bit_exact<GapModel::Linear, AlignMode::Global, AlignBand::Full>(p, a, b);
    check_bit_exact<GapModel::Linear, AlignMode::Local,  AlignBand::Full>(p, a, b);
    check_bit_exact<GapModel::Affine, AlignMode::Global, AlignBand::Full>(p, a, b);
    check_bit_exact<GapModel::Affine, AlignMode::Local,  AlignBand::Full>(p, a, b);

    for (int band : {4, 16}) {
        check_bit_exact<GapModel::Linear, AlignMode::Global, AlignBand::GuideBanded>(p, a, b, band);
        check_bit_exact<GapModel::Linear, AlignMode::Local,  AlignBand::GuideBanded>(p, a, b, band);
        check_bit_exact<GapModel::Affine, AlignMode::Global, AlignBand::GuideBanded>(p, a, b, band);
        check_bit_exact<GapModel::Affine, AlignMode::Local,  AlignBand::GuideBanded>(p, a, b, band);
    }
}

}  // namespace

TEST_CASE("simd viterbi is bit-exact with scalar — random sequences", "[simd]") {
    std::mt19937_64 rng(20260714);
    AlignParams p = int_params(11.0, 1.0, 11.0, 1.0);

    for (int trial = 0; trial < 12; ++trial) {
        std::uniform_int_distribution<int> len(1, 90);
        check_all_modes(p, random_seq(rng, len(rng)), random_seq(rng, len(rng)));
    }
}

TEST_CASE("simd viterbi is bit-exact with scalar — asymmetric gaps", "[simd]") {
    std::mt19937_64 rng(7);
    // Asymmetric in every field, so a kernel that confused the _a and _b penalties
    // (or the X and Y states) cannot hide behind symmetry.
    AlignParams p = int_params(7.0, 3.0, 13.0, 0.5);

    for (int trial = 0; trial < 8; ++trial)
        check_all_modes(p, random_seq(rng, 40), random_seq(rng, 55));
}

TEST_CASE("simd viterbi is bit-exact with scalar — zero gap-extend", "[simd]") {
    // The degenerate case, and the one most likely to expose the lazy-F reasoning:
    // with ge_a == 0 a gap costs nothing to extend, so the carry propagates the whole
    // width of the row instead of dying out after a few columns.  If the fixpoint were
    // reached by a different arithmetic route than the scalar chain, it would show here.
    std::mt19937_64 rng(99);
    AlignParams p = int_params(5.0, 0.0, 5.0, 0.0);

    for (int trial = 0; trial < 8; ++trial)
        check_all_modes(p, random_seq(rng, 50), random_seq(rng, 50));
}

TEST_CASE("simd viterbi is bit-exact with scalar — degenerate shapes", "[simd]") {
    AlignParams p = int_params(11.0, 1.0, 11.0, 1.0);

    // Empty and single-character sequences: the row loop runs zero or one time and the
    // band clamps hard.  These are where an off-by-one in lo/hi lives.
    check_all_modes(p, "", "");
    check_all_modes(p, "A", "");
    check_all_modes(p, "", "A");
    check_all_modes(p, "A", "A");
    check_all_modes(p, "A", "WYWYWY");
    check_all_modes(p, "WYWYWY", "A");
    check_all_modes(p, "AAAAAAAAAA", "AAAAAAAAAA");  // maximal ties
}
