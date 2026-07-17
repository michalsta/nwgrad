// Float32 Viterbi: the simd striped kernel must be BIT-EXACT with the scalar fill at
// the SAME precision.  This is the float analogue of test_simd_bitexact.cpp, and the
// contract is identical in spirit but redefined for T=float: the double reference no
// longer applies (float rounds differently and ties differently), so the reference is
// the scalar *float* fill, and both are compared on 32-bit bit patterns.
//
// Why it must hold: the striped kernel reads the very block subT() reads (blkT_, the
// substitution matrix converted to float once per problem) and the gap penalties
// converted to float the same way; every recurrence op is (v - go) - ge left-associated
// in float, and max(a,b)-c == max(a-c,b-c) is bitwise-true in IEEE float as in double.
// So the argmax tracebacks — which re-derive the path from the stored float cells — see
// exactly the tables the fill wrote, and the hard gradient is identical.
//
// It also confirms the float striped kernel is genuinely exercised: nwgrad_tests links
// the level TUs, so active_kernels().viterbi_f is non-null and the Full path dispatches
// to it rather than falling back to the scalar fill.

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

// Bit-identity over the first `n` floats of two tables (32-bit patterns, so -inf and
// signed zero are caught rather than compared equal).  Returns first differing index or -1.
long first_bit_diff_f(const std::vector<float>& a, const std::vector<float>& b, size_t n) {
    if (a.size() < n || b.size() < n) return 0;
    for (size_t k = 0; k < n; ++k)
        if (std::bit_cast<uint32_t>(a[k]) != std::bit_cast<uint32_t>(b[k]))
            return static_cast<long>(k);
    return -1;
}

// Affine Full only: that is the striped-kernel path.  (GuideBanded float falls back to
// the scalar fill, so scalar==simd there is trivially true and tests nothing new.)
template<AlignMode AM>
void check_bit_exact_f32(const AlignParams& p, const std::string& a, const std::string& b) {
    Aligner<GapModel::Affine, AM, AlignBand::Full, float> al;
    const size_t sz = (a.size() + 1) * (b.size() + 1);

    DpBufferT<float> buf_scalar, buf_simd;

    al.set_problem(a, b, p);
    al.set_kernel(DpKernel::Scalar);
    al.compute_viterbi(buf_scalar);
    const double score_scalar = al.score();
    AlignParams grad_scalar = AlignParams::zeros_like(p);
    al.hard_grad(buf_scalar, grad_scalar);
    // Snapshot scalar tables row-major while still row-major (simd flips to striped).
    std::vector<float> rmM = al.to_row_major(buf_scalar.VM);
    std::vector<float> rmX = al.to_row_major(buf_scalar.VX);
    std::vector<float> rmY = al.to_row_major(buf_scalar.VY);

    al.set_problem(a, b, p);
    al.set_kernel(DpKernel::Simd);
    al.compute_viterbi(buf_simd);
    const double score_simd = al.score();
    AlignParams grad_simd = AlignParams::zeros_like(p);
    al.hard_grad(buf_simd, grad_simd);

    // Tables (de-striped back to row-major for the simd run).
    REQUIRE(first_bit_diff_f(rmM, al.to_row_major(buf_simd.VM), sz) == -1);
    REQUIRE(first_bit_diff_f(rmX, al.to_row_major(buf_simd.VX), sz) == -1);
    REQUIRE(first_bit_diff_f(rmY, al.to_row_major(buf_simd.VY), sz) == -1);

    // Score (reported as double; equal float value promotes to the same double).
    REQUIRE(std::bit_cast<uint64_t>(score_scalar) == std::bit_cast<uint64_t>(score_simd));

    // Gradient: proof the tie-break survived vectorization at float precision.
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

void check_both_modes_f32(const AlignParams& p, const std::string& a, const std::string& b) {
    check_bit_exact_f32<AlignMode::Global>(p, a, b);
    check_bit_exact_f32<AlignMode::Local >(p, a, b);
}

}  // namespace

TEST_CASE("f32 striped kernel is actually dispatched (not a scalar fallback)", "[simd]") {
    // Bit-exactness would hold trivially if the float Full path silently fell back to the
    // scalar fill.  Prove it does not: the level's float striped entry is registered, and
    // a Simd run over a length that is not a multiple of the vector width leaves the table
    // in the padded striped layout (buffer larger than the row-major (m+1)(n+1)).
    REQUIRE(active_kernels().viterbi_f != nullptr);

    AlignParams p = int_params(11.0, 1.0, 11.0, 1.0);
    std::mt19937_64 rng(1234);
    const std::string a = random_seq(rng, 37), b = random_seq(rng, 41);  // n=41, not a mult of W

    Aligner<GapModel::Affine, AlignMode::Global, AlignBand::Full, float> al;
    DpBufferT<float> buf;
    al.set_problem(a, b, p);
    al.set_kernel(DpKernel::Simd);
    al.compute_viterbi(buf);
    const size_t row_major = (a.size() + 1) * (b.size() + 1);
    REQUIRE(buf.VM.size() > row_major);   // striped rows are padded to (seg+1)*W > (n+1)
}

TEST_CASE("f32 gives the correct score vs double — integer matrix", "[simd]") {
    // Self-consistency (scalar==simd) is not correctness.  For an integer-valued matrix
    // and integer gap penalties the optimal Global score is an integer far below 2^24, so
    // float represents it exactly — float and double must agree on the score to the bit.
    // (Ties may still route the two precisions down different equal-score paths, so the
    // gradient is NOT required equal here; the score is.)
    std::mt19937_64 rng(555);
    AlignParams p = int_params(11.0, 1.0, 11.0, 1.0);
    for (int trial = 0; trial < 16; ++trial) {
        std::uniform_int_distribution<int> len(1, 120);
        const std::string a = random_seq(rng, len(rng)), b = random_seq(rng, len(rng));

        Aligner<GapModel::Affine, AlignMode::Global, AlignBand::Full, double> ad;
        DpBufferT<double> bd;
        ad.set_problem(a, b, p); ad.set_kernel(DpKernel::Simd); ad.compute_viterbi(bd);

        Aligner<GapModel::Affine, AlignMode::Global, AlignBand::Full, float> af;
        DpBufferT<float> bf;
        af.set_problem(a, b, p); af.set_kernel(DpKernel::Simd); af.compute_viterbi(bf);

        REQUIRE(af.score() == ad.score());   // integer optimum, exact in both
    }
}

TEST_CASE("f32 simd viterbi is bit-exact with f32 scalar — random sequences", "[simd]") {
    std::mt19937_64 rng(20260717);
    AlignParams p = int_params(11.0, 1.0, 11.0, 1.0);
    for (int trial = 0; trial < 12; ++trial) {
        std::uniform_int_distribution<int> len(1, 90);
        check_both_modes_f32(p, random_seq(rng, len(rng)), random_seq(rng, len(rng)));
    }
}

TEST_CASE("f32 simd viterbi is bit-exact with f32 scalar — asymmetric gaps", "[simd]") {
    std::mt19937_64 rng(7);
    AlignParams p = int_params(7.0, 3.0, 13.0, 0.5);
    for (int trial = 0; trial < 8; ++trial)
        check_both_modes_f32(p, random_seq(rng, 40), random_seq(rng, 55));
}

TEST_CASE("f32 simd viterbi is bit-exact with f32 scalar — zero gap-extend", "[simd]") {
    // ge_a == 0: the lazy-F carry propagates the whole row width — the case most likely
    // to expose a divergent fixpoint between the scalar chain and the vectorized one.
    std::mt19937_64 rng(99);
    AlignParams p = int_params(5.0, 0.0, 5.0, 0.0);
    for (int trial = 0; trial < 8; ++trial)
        check_both_modes_f32(p, random_seq(rng, 50), random_seq(rng, 50));
}

TEST_CASE("f32 simd viterbi is bit-exact with f32 scalar — degenerate shapes", "[simd]") {
    AlignParams p = int_params(11.0, 1.0, 11.0, 1.0);
    check_both_modes_f32(p, "", "");
    check_both_modes_f32(p, "A", "");
    check_both_modes_f32(p, "", "A");
    check_both_modes_f32(p, "A", "A");
    check_both_modes_f32(p, "A", "WYWYWY");
    check_both_modes_f32(p, "WYWYWY", "A");
    check_both_modes_f32(p, "AAAAAAAAAA", "AAAAAAAAAA");  // maximal ties
}
