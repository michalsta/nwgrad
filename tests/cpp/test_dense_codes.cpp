// ── Dense direction codes: the same codes, fewer bytes ───────────────────────
//
// A leveled Pointers fill can write its direction codes dense (Aligner::set_dense_codes,
// dense_codes_row_bytes in simd_levels.hpp): 2-bit fields, four segments a byte — the
// linear fill one plane (0.25 B/cell), the affine fill three, M / X / Y (0.75 B/cell).
// It relocates bits and recomputes nothing, so everything downstream must be bit-identical
// to the one-byte format: score, alignment, guide, hard gradient — for every gap model,
// mode, scalar type, level, tracked or not, and at every segment count mod 4 (the last
// group of a row may hold one to four segments).  And the dense format must actually be
// written (a liveness check), or this would compare the one-byte fill with itself.

#include "catch.hpp"

#include <cstdint>
#include <cstdlib>
#include <string>
#include <type_traits>
#include <vector>

#include "align_params.hpp"
#include "aligner.hpp"
#include "simd_levels.hpp"

namespace {

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed ? seed : 1) {}
    uint32_t next() {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        return static_cast<uint32_t>(s >> 32);
    }
    int in(int lo, int hi) { return lo + static_cast<int>(next() % static_cast<uint32_t>(hi - lo + 1)); }
};

// Tie-heavy integral parameters (valid for every scalar type), K sets for a tracked problem.
AlignParams params(Rng& rng, const Alphabet& alpha, bool affine, int sets) {
    const int n = alpha.size();
    std::vector<double> stack(static_cast<size_t>(n) * n * sets);
    for (int k = 0; k < sets; ++k)
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                stack[(static_cast<size_t>(k) * n + i) * n + j] = (i == j) ? rng.in(1, 5) : rng.in(-4, 2);
    AlignParams p(SubstMatrix(stack.data(), alpha, sets));
    for (int k = 0; k < sets; ++k) {
        p.gap_ref(k, AlignParams::kOpenA)   = affine ? rng.in(0, 6) : 0;
        p.gap_ref(k, AlignParams::kExtendA) = rng.in(0, 3);
        p.gap_ref(k, AlignParams::kOpenB)   = affine ? rng.in(0, 6) : 0;
        p.gap_ref(k, AlignParams::kExtendB) = rng.in(0, 3);
    }
    return p;
}

std::string seq(Rng& rng, const Alphabet& alpha, int len) {
    std::string s(static_cast<size_t>(len), alpha.symbol_at(0));
    for (char& c : s) c = alpha.symbol_at(rng.in(0, alpha.size() - 1));
    return s;
}

struct Out {
    double score = 0;
    std::pair<std::string, std::string> aligned;
    std::vector<int> guide;
    AlignParams grad;
    bool dense = false;
};

template <GapModel GM, AlignMode AM, class T>
Out run(const std::vector<uint8_t>& ea, const std::vector<uint8_t>& eb, const AlignParams& p,
        const std::vector<uint16_t>& track, int backend, int dense) {
    Out o{0, {}, {}, AlignParams::zeros_like(p), false};
    Aligner<GM, AM, AlignBand::Full, T> al;
    al.set_kernel(backend);
    al.set_traceback(TracebackMode::Pointers);
    al.set_dense_codes(dense);
    DpBufferT<T> buf;
    if (track.empty()) al.set_problem(ea, eb, p, 0);
    else               al.set_problem(ea, eb, p, 0, {}, track, {});
    al.compute_viterbi(buf);
    o.score = al.score();
    o.aligned = al.aligned(buf);
    o.guide = al.guide_j_from_viterbi(buf);
    al.hard_grad(buf, o.grad);
    o.dense = al.dense_written();
    return o;
}

bool same(const AlignParams& a, const AlignParams& b) {
    const int N = a.matrix.size();
    for (int k = 0; k < a.num_sets(); ++k) {
        for (int q = 0; q < N * N; ++q)
            if (a.matrix.data()[static_cast<size_t>(k) * N * N + q] != b.matrix.data()[static_cast<size_t>(k) * N * N + q]) return false;
        for (int f = 0; f < 4; ++f) if (a.gap(k, f) != b.gap(k, f)) return false;
    }
    return true;
}

template <GapModel GM, AlignMode AM, class T>
void sweep(uint64_t seed) {
    const Alphabet& alpha = Alphabet::protein();
    Rng rng(seed);
    int cases = 0, mismatches = 0, dense_written = 0;
    for (SimdLevel lvl : detect_available_levels()) {
        const int backend = static_cast<int>(lvl);
        for (int it = 0; it < 60; ++it) {
            const int sets = (it % 3 == 0) ? 3 : 1;
            const AlignParams p = params(rng, alpha, GM == GapModel::Affine, sets);
            // lengths over a range that puts seg % 4 at every value for every width
            const int la = rng.in(0, 3) == 0 ? rng.in(0, 3) : rng.in(1, 260);
            const int lb = rng.in(0, 4) == 0 ? rng.in(0, 9) : rng.in(1, 260);
            const std::string A = seq(rng, alpha, la);
            std::string B = seq(rng, alpha, lb);
            if (rng.in(0, 2) == 0 && la > 4) { B = A; B[static_cast<size_t>(la / 2)] = alpha.symbol_at(0); }   // related
            const auto ea = alpha.encode(A), eb = alpha.encode(B);
            std::vector<uint16_t> track;
            if (sets > 1) {
                track.resize(ea.size());
                for (auto& t : track) t = static_cast<uint16_t>(rng.in(0, sets - 1));
            }
            const Out one = run<GM, AM, T>(ea, eb, p, track, backend, 0);
            const Out dns = run<GM, AM, T>(ea, eb, p, track, backend, 1);
            ++cases;
            CHECK_FALSE(one.dense);
            if (dns.dense) ++dense_written;
            if (one.score != dns.score || one.aligned != dns.aligned || one.guide != dns.guide ||
                !same(one.grad, dns.grad)) {
                ++mismatches;
                UNSCOPED_INFO("level " << backend << " A " << A << " B " << B);
            }
        }
    }
    CHECK(mismatches == 0);
    // liveness: the leveled kernels write the dense format (int16 never; nor int64 linear,
    // which runs the scalar fill below eight lanes -- linear_kernel_pays -- and so on every
    // level this suite's hosts have but AVX-512)
    const bool scalar_linear = GM == GapModel::Linear && std::is_same_v<T, int64_t>;
    if (sizeof(T) >= 4 && !scalar_linear && !detect_available_levels().empty())
        CHECK(dense_written > cases / 4);
    INFO(cases << " cases, " << dense_written << " written dense");
}

template <GapModel GM>
bool gate_writes(Rng& rng, int len, int backend) {
    const Alphabet& alpha = Alphabet::protein();
    Aligner<GM, AlignMode::Global, AlignBand::Full, int32_t> al;
    al.set_kernel(backend);
    al.set_traceback(TracebackMode::Pointers);   // auto dense (dense_codes() == -1)
    const AlignParams p = params(rng, alpha, GM == GapModel::Affine, 1);
    const auto e = alpha.encode(seq(rng, alpha, len));
    DpBufferT<int32_t> buf;
    al.set_problem(e, e, p, 0);
    al.compute_viterbi(buf);
    return al.dense_written();
}

}  // namespace

TEMPLATE_TEST_CASE("dense direction codes reproduce the one-byte codes bit for bit",
                   "[dense][pointers]", float, double, int32_t, int64_t, int16_t) {
    sweep<GapModel::Affine, AlignMode::Global, TestType>(11);
    sweep<GapModel::Affine, AlignMode::Local, TestType>(12);
    sweep<GapModel::Linear, AlignMode::Global, TestType>(13);
    sweep<GapModel::Linear, AlignMode::Local, TestType>(14);
}

// The auto gate (Aligner::dense_gate_): dense from kDenseFromCells cells a pair, never for
// int16.  int32 affine Global has the smallest threshold (2^20 cells), so it is the one to
// exercise at a test-friendly size.  NWGRAD_DENSE_CODES, if set in the environment, would
// override auto: skip then.
TEST_CASE("dense codes: the auto gate engages by pair size", "[dense][pointers]") {
    if (std::getenv("NWGRAD_DENSE_CODES")) { WARN("NWGRAD_DENSE_CODES overrides the gate: skipped"); return; }
    Rng rng(99);
    // the unmeasured levels and the scalar fill never write them
    STATIC_REQUIRE(Aligner<GapModel::Affine, AlignMode::Global, AlignBand::Full, int16_t>::kDenseFromCellsAvx2 == 0);
    STATIC_REQUIRE(Aligner<GapModel::Affine, AlignMode::Global, AlignBand::GuideBanded, double>::kDenseFromCellsAvx2 == 0);
    STATIC_REQUIRE(Aligner<GapModel::Affine, AlignMode::Global, AlignBand::Full, float>::kDenseFromCellsNeon == 0);
    STATIC_REQUIRE(Aligner<GapModel::Linear, AlignMode::Global, AlignBand::Full, double>::kDenseFromCellsNeon == 0);
    int level = -1;
    for (SimdLevel l : detect_available_levels()) {
#if defined(__aarch64__) || defined(_M_ARM64)
        level = static_cast<int>(l);
#else
        if (l == SimdLevel::Avx2 || l == SimdLevel::Avx512) level = static_cast<int>(l);
#endif
    }
#if !(defined(__aarch64__) || defined(_M_ARM64))
    // sse2 / sse4: off at any size (the gated levels are the last one found above)
    for (SimdLevel l : detect_available_levels())
        if (l != SimdLevel::Avx2 && l != SimdLevel::Avx512 && static_cast<int>(l) != level)
            CHECK_FALSE(gate_writes<GapModel::Affine>(rng, 1100, static_cast<int>(l)));
#endif
    if (level < 0) { WARN("no gated level here: the positive half skipped"); return; }
#if defined(__aarch64__) || defined(_M_ARM64)
    CHECK_FALSE(gate_writes<GapModel::Linear>(rng, 500, level));    // 251 k cells
    CHECK(gate_writes<GapModel::Linear>(rng, 1100, level));         // 1.21 M cells
    CHECK_FALSE(gate_writes<GapModel::Affine>(rng, 1100, level));   // affine: never on NEON
#else
    CHECK_FALSE(gate_writes<GapModel::Affine>(rng, 500, level));    // 251 k cells: in cache
    CHECK(gate_writes<GapModel::Affine>(rng, 2100, level));         // 4.4 M: past avx2 2^20, avx512 2^22
#endif
}
