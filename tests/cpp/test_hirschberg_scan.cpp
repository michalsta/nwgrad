// ── Local Hirschberg: the endpoint scan must be bit-identical scalar vs every level ──
//
// Local (Smith-Waterman) linear-space alignment adds one new vectorized kernel: the
// endpoint SCAN (hb_scan_impl) that finds the optimal cell's end (clamped forward pass)
// and start (unclamped global-suffix reverse pass) before the divide-and-conquer aligns
// the box between them.  Like every other simd kernel here it is held to bit-identity
// with its scalar reference — score, alignment, guide_j and gradient — because forcing
// NWGRAD_ISA (here: set_kernel per aligner) must never move the endpoint it picks.  A
// tie-broken argmax that drifted between the scalar and vector reductions would show up
// as a different-but-valid alignment, and only a tie-heavy integral matrix makes it
// visible, so that is what this uses.
//
// This loops every backend the CPU offers via set_kernel, so a CI runner with AVX2
// exercises the W=4 argmax reduction that a W=2 machine cannot — the reason it is worth
// having in C++ and not only in the Python suite.

#include "catch.hpp"

#include <string>
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

// Local Hirschberg at `backend` must equal local Hirschberg at the scalar fallback, to
// the bit.  cutoff = 8 forces the recursion to split, so the scan+join are genuinely
// exercised and not just the base-case Pointers fill.
template <class T>
void scalar_vs_level(const std::string& A, const std::string& B, const AlignParams& p,
                     int backend, int& mismatches) {
    Aligner<GapModel::Affine, AlignMode::Local, AlignBand::Full, T> ref, dut;
    ref.set_kernel(kBackendScalar); dut.set_kernel(backend);
    ref.set_traceback(TracebackMode::Hirschberg); dut.set_traceback(TracebackMode::Hirschberg);
    ref.set_hb_cutoff(8); dut.set_hb_cutoff(8);

    const auto ea = p.matrix.alphabet().encode(A);
    const auto eb = p.matrix.alphabet().encode(B);
    DpBufferT<T> br, bd;
    ref.set_problem(ea, eb, p); ref.compute_viterbi(br);
    dut.set_problem(ea, eb, p); dut.compute_viterbi(bd);

    bool ok = (ref.score() == dut.score());
    if (ref.aligned(br) != dut.aligned(bd)) ok = false;
    if (ref.guide_j_from_viterbi(br) != dut.guide_j_from_viterbi(bd)) ok = false;

    AlignParams gr(p.matrix.alphabet()), gd(p.matrix.alphabet());
    ref.hard_grad(br, gr);
    dut.hard_grad(bd, gd);
    const int N = p.matrix.alphabet().size();
    for (int x = 0; x < N && ok; ++x)
        for (int y = 0; y < N; ++y)
            if (gr.matrix.at(x, y) != gd.matrix.at(x, y)) { ok = false; break; }
    if (gr.gap_open_a != gd.gap_open_a || gr.gap_extend_a != gd.gap_extend_a ||
        gr.gap_open_b != gd.gap_open_b || gr.gap_extend_b != gd.gap_extend_b) ok = false;

    if (!ok) ++mismatches;
}

// Optimality: local Hirschberg's score must equal local Pointers' (Smith-Waterman) score
// — a valid but possibly different path, never a worse one.  At double the two agree to
// floating associativity; the gap here is a whole gap-open, so a loose tolerance is safe.
template <class T>
void hb_score_matches_sw(const std::string& A, const std::string& B, const AlignParams& p,
                         int& mismatches) {
    Aligner<GapModel::Affine, AlignMode::Local, AlignBand::Full, T> ptr, hb;
    ptr.set_kernel(kBackendScalar); hb.set_kernel(kBackendScalar);
    ptr.set_traceback(TracebackMode::Pointers); hb.set_traceback(TracebackMode::Hirschberg);
    hb.set_hb_cutoff(8);
    const auto ea = p.matrix.alphabet().encode(A);
    const auto eb = p.matrix.alphabet().encode(B);
    DpBufferT<T> bp, bh;
    ptr.set_problem(ea, eb, p); ptr.compute_viterbi(bp);
    hb.set_problem(ea, eb, p);  hb.compute_viterbi(bh);
    if (std::abs(ptr.score() - hb.score()) > 1e-6) ++mismatches;
}

}  // namespace

TEST_CASE("local Hirschberg endpoint scan is bit-identical scalar vs every level",
          "[hirschberg][simd]") {
    const Alphabet& al = Alphabet::get("ACDEFGHIKLMNPQRSTVWY");
    SubstMatrix M(al);
    for (int x = 0; x < al.size(); ++x)
        for (int y = 0; y < al.size(); ++y)
            M.at(x, y) = (x == y) ? 4.0 : -1.0;      // integral => ties are common
    const AlignParams p(M, 11.0, 1.0, 11.0, 1.0);

    for (int backend : backends_under_test()) {
        INFO("backend = " << backend_name(backend));
        Rng rng(0xC0FFEEu);
        int mismatches = 0, subopt = 0;
        for (int t = 0; t < 300; ++t) {
            std::string A, B;
            for (int k = rng.in(0, 90); k > 0; --k) A += al.symbol_at(rng.in(0, al.size() - 1));
            for (int k = rng.in(0, 90); k > 0; --k) B += al.symbol_at(rng.in(0, al.size() - 1));
            scalar_vs_level<double>(A, B, p, backend, mismatches);
            scalar_vs_level<float >(A, B, p, backend, mismatches);
            hb_score_matches_sw<double>(A, B, p, subopt);
            // self-pair: unique optimum, so bit-exact with SW pointers on both axes
            scalar_vs_level<double>(A, A, p, backend, mismatches);
        }
        REQUIRE(mismatches == 0);
        REQUIRE(subopt == 0);
    }
}

TEST_CASE("local Hirschberg handles degenerate and empty shapes", "[hirschberg]") {
    const Alphabet& al = Alphabet::get("ACDEFGHIKLMNPQRSTVWY");
    SubstMatrix M(al);
    for (int x = 0; x < al.size(); ++x)
        for (int y = 0; y < al.size(); ++y) M.at(x, y) = (x == y) ? 4.0 : -1.0;
    const AlignParams p(M, 11.0, 1.0, 11.0, 1.0);

    const std::vector<std::string> odd{"", "A", "AC", "ACDEFGHIK"};
    for (int backend : backends_under_test()) {
        INFO("backend = " << backend_name(backend));
        int mismatches = 0, subopt = 0;
        for (const auto& A : odd)
            for (const auto& B : odd) {
                scalar_vs_level<double>(A, B, p, backend, mismatches);
                scalar_vs_level<float >(A, B, p, backend, mismatches);
                hb_score_matches_sw<double>(A, B, p, subopt);
            }
        REQUIRE(mismatches == 0);
        REQUIRE(subopt == 0);
    }
}
