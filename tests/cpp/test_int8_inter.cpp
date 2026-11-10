// ── The inter-pair int8 score tier (inter8_kernel_impl.inl, BatchEngine::score_only_) ──
//
// Score-only integer batches with the tier forced on (inter_score_int8 = 2) against the
// tier off, at every level the CPU runs (NEON under qemu in the aarch64 cross run):
// every pair's score bit for bit, short DNA-like pairs (miRNA x site shapes, ragged),
// parameters up to the int8 limits, and problems on both sides of each overflow test
// (Local 127, Global / Glocal G - 128), so the re-runs and the predictor both take part.

#include "catch.hpp"

#include <random>
#include <string>
#include <string_view>
#include <vector>

#include "batch_engine.hpp"

namespace {

std::string rs(std::mt19937& rng, const std::string& sym, int n) {
    std::string s(static_cast<size_t>(n), 'A');
    for (auto& c : s) c = sym[rng() % sym.size()];
    return s;
}

template <class T, GapModel GM, AlignMode AM>
long i8_case(std::mt19937& rng, int kern, long& n_big) {
    const std::string sym = (rng() % 4 == 0) ? "ACGTN" : "ACGT";
    const Alphabet& alpha = Alphabet::get(sym);
    AlignParams p(alpha);
    const int style = static_cast<int>(rng() % 3);
    const int n = alpha.size();
    for (int x = 0; x < n; ++x)
        for (int y = 0; y < n; ++y) {
            int v;
            if (style == 0) v = x == y ? 2 : -4;
            else if (style == 1) v = x == y ? 1 + static_cast<int>(rng() % 8) : -static_cast<int>(rng() % 6);
            else v = x == y ? 127 : (rng() % 2 ? -127 : 0);
            p.matrix.at(x, y) = v;
        }
    auto g = [&](int hi) { return static_cast<double>(rng() % static_cast<unsigned>(hi + 1)); };
    if (style == 2) { p.gap_open_a = g(63); p.gap_extend_a = g(64); p.gap_open_b = g(63); p.gap_extend_b = g(64); }
    else            { p.gap_open_a = g(10); p.gap_extend_a = g(3);  p.gap_open_b = g(10); p.gap_extend_b = g(3); }
    if (GM == GapModel::Linear) { p.gap_open_a = p.gap_open_b = 0; }
    const int N = 20 + static_cast<int>(rng() % 60);
    const int la0 = 10 + static_cast<int>(rng() % 110), lb0 = la0 + static_cast<int>(rng() % 40);
    std::vector<std::string> A, B;
    for (int k = 0; k < N; ++k) {
        const int la = std::max(1, la0 + static_cast<int>(rng() % 5) - 2);
        const int lb = std::max(1, std::min(130, lb0 + static_cast<int>(rng() % 5) - 2));
        std::string a = rs(rng, sym, la), b = rs(rng, sym, lb);
        if (rng() % 2) b.replace(0, std::min(a.size(), b.size()), a.substr(0, std::min(a.size(), b.size())));
        A.push_back(a); B.push_back(b);
    }
    std::vector<std::string_view> va(A.begin(), A.end()), vb(B.begin(), B.end());
    std::vector<double> got[2];
    for (int arm = 0; arm < 2; ++arm) {
        set_gate_override(Gate::InterScoreInt8, arm == 0 ? 2 : 0);
        BatchEngine<T, GM, AM> eng(BatchSettings{1 + static_cast<int>(rng() % 3), TracebackMode::Default},
                                   GradMode::None, va, vb, p, kern);
        eng.set_fill(false, true);
        try {
            eng.score_and_grad();
        } catch (const std::overflow_error&) {
            set_gate_override(Gate::InterScoreInt8, -1);
            return 0;   // an int16 batch past its bound: refused alike in both arms
        }
        got[arm] = eng.scores();
    }
    set_gate_override(Gate::InterScoreInt8, -1);
    for (size_t i = 0; i < got[0].size(); ++i) {
        INFO("pair " << i << " a=" << A[i] << " b=" << B[i] << " kern " << kern << " GM " << static_cast<int>(GM)
             << " AM " << static_cast<int>(AM) << " T " << sizeof(T) << " int8 " << got[0][i] << " off " << got[1][i]);
        REQUIRE(got[0][i] == got[1][i]);
        n_big += AM == AlignMode::Local ? got[1][i] >= 127 : got[1][i] <= -84;
    }
    return static_cast<long>(got[0].size());
}

}  // namespace

TEST_CASE("inter-pair int8 score tier: every score equals the tier off", "[score][interpair][int8]") {
    std::mt19937 rng(8);
    set_gate_override(Gate::InterScoreSmallCells, kGateOff);   // groups up to L127 on the shared kernel
    long n = 0, big = 0, levels = 0;
    for (SimdLevel lv : detect_available_levels()) {
        const int kern = static_cast<int>(lv);
        if (!level_kernels(kern).inter_score_i8) continue;
        ++levels;
        for (int it = 0; it < 12; ++it) {
            n += i8_case<int16_t, GapModel::Affine, AlignMode::Global>(rng, kern, big);
            n += i8_case<int16_t, GapModel::Affine, AlignMode::Local >(rng, kern, big);
            n += i8_case<int16_t, GapModel::Affine, AlignMode::Glocal>(rng, kern, big);
            n += i8_case<int16_t, GapModel::Linear, AlignMode::Global>(rng, kern, big);
            n += i8_case<int16_t, GapModel::Linear, AlignMode::Local >(rng, kern, big);
            n += i8_case<int16_t, GapModel::Linear, AlignMode::Glocal>(rng, kern, big);
            n += i8_case<int32_t, GapModel::Affine, AlignMode::Global>(rng, kern, big);
            n += i8_case<int32_t, GapModel::Affine, AlignMode::Local >(rng, kern, big);
        }
    }
    set_gate_override(Gate::InterScoreSmallCells, -1);
    INFO(levels << " levels, " << n << " pairs, " << big << " past an int8 test");
    REQUIRE(levels > 0);
    REQUIRE(big > 0);
}
