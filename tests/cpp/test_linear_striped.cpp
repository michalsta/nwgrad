// The leveled striped LINEAR kernels (kernels_linear_impl.inl) against the scalar fills
// they replace (viterbi_linear / viterbi_linear_ptr, aligner.hpp), at every ISA level the
// running CPU offers, every scalar type, both modes, both tracebacks, tracked or not:
// score, aligned strings, guide and hard gradient equal bit for bit, ties included — the
// contract every leveled kernel in this tree carries, since the walkers re-derive the path
// by exact equality on the table.  Fixtures include every striping edge: n from 0 to past
// several segments at every lane count, m = 0, non-representable extends (0.1) and zero
// and negative ones on the floating types, tie-heavy integral parameters on the integer
// ones.  The Pointers form must also equal the H form at the same level (one walk).
//
// run with NWGRAD_ISA=<level> (ctest does, per level) or let it loop every level.
#include "catch.hpp"

#include <cstdint>
#include <string>
#include <vector>

#include "align_params.hpp"
#include "aligner.hpp"
#include "simd_levels.hpp"

namespace {

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed ? seed : 1) {}
    uint32_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return static_cast<uint32_t>(s >> 32); }
    int in(int lo, int hi) { return lo + static_cast<int>(next() % static_cast<uint32_t>(hi - lo + 1)); }
};

// Integer-valued for an integer T (small, tie-heavy); real-valued for a floating T, with
// extends drawn from {0.1 k}, zero and negative ones included (the exact modes take them).
template <class T>
AlignParams make_params(Rng& rng, const Alphabet& alpha, int sets, bool nonneg = false) {
    const int n = alpha.size();
    std::vector<double> stack(static_cast<size_t>(n) * n * sets);
    for (int k = 0; k < sets; ++k)
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                double v = (i == j) ? rng.in(1, 5) : rng.in(-4, 2);
                if constexpr (!nw_is_int<T>) if (rng.in(0, 1)) v += 0.1 * rng.in(-3, 3);
                stack[(static_cast<size_t>(k) * n + i) * n + j] = v;
            }
    AlignParams p(SubstMatrix(stack.data(), alpha, sets));
    for (int k = 0; k < sets; ++k) {
        double ea, eb;
        const int lo = nonneg ? 0 : -1;
        if constexpr (nw_is_int<T>) { ea = rng.in(lo, 3); eb = rng.in(lo, 3); }
        else { ea = 0.1 * rng.in(5 * lo, 30); eb = 0.1 * rng.in(5 * lo, 30); if (rng.in(0, 4) == 0) ea = 0; }
        p.gap_ref(k, AlignParams::kOpenA) = 0; p.gap_ref(k, AlignParams::kExtendA) = ea;
        p.gap_ref(k, AlignParams::kOpenB) = 0; p.gap_ref(k, AlignParams::kExtendB) = eb;
    }
    return p;
}

std::string random_seq(Rng& rng, const Alphabet& alpha, int len) {
    std::string s(static_cast<size_t>(len), alpha.symbol_at(0));
    for (char& c : s) c = alpha.symbol_at(rng.in(0, alpha.size() - 1));
    return s;
}

struct Result {
    double score = 0;
    std::pair<std::string, std::string> aligned;
    std::vector<int> guide;
    AlignParams grad;
    int layout = 0;   // 1 if the aligner reports a striped table (the kernel ran)
};

template <AlignMode AM, class T>
Result run_one(const std::vector<uint8_t>& ea, const std::vector<uint8_t>& eb, const AlignParams& p,
               const std::vector<uint16_t>& track, int backend, TracebackMode tb, int cutoff = 0,
               int* lhb_runs = nullptr) {
    Result r{0, {}, {}, AlignParams::zeros_like(p), 0};
    Aligner<GapModel::Linear, AM, AlignBand::Full, T> al;
    al.set_kernel(backend);
    al.set_traceback(tb);
    if (cutoff > 0) al.set_hb_cutoff(cutoff);
    DpBufferT<T> buf;
    if (track.empty()) al.set_problem(ea, eb, p, 0);
    else               al.set_problem(ea, eb, p, 0, {}, track, {});
    al.compute_viterbi(buf);
    r.score = al.score();
    r.aligned = al.aligned(buf);
    r.guide = al.guide_j_from_viterbi(buf);
    al.hard_grad(buf, r.grad);
    r.layout = al.tables_striped() ? 1 : 0;
    if (lhb_runs) *lhb_runs += al.lhb_kernel_runs();
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

bool same(const Result& a, const Result& b) {
    return a.score == b.score && a.aligned == b.aligned && a.guide == b.guide && same_grad(a.grad, b.grad);
}

template <class T>
void sweep(uint64_t seed, int rounds, int& mismatches, int& cases, int& kernel_runs, int& expected_runs,
           std::string& first) {
    Rng rng(seed);
    const Alphabet* alphas[] = {&Alphabet::get("ACGT"), &Alphabet::get("ACDEFGHIKLMNPQRSTVWY"), &Alphabet::get("AB")};
    std::vector<int> levels;
    for (SimdLevel l : detect_available_levels()) levels.push_back(static_cast<int>(l));
    if (levels.empty()) return;
    for (int r = 0; r < rounds; ++r) {
        const Alphabet& alpha = *alphas[rng.in(0, 2)];
        const int sets = rng.in(0, 3) == 0 ? rng.in(2, 4) : 1;
        const AlignParams p = make_params<T>(rng, alpha, sets);
        const int pick = rng.in(0, 3);
        // B spans the gate (kLinearStripedMinN, 24-96) so both sides run; A short or long
        const int la = pick == 0 ? rng.in(0, 3) : pick == 1 ? rng.in(1, 20) : rng.in(1, 90);
        const int lb = pick == 0 ? rng.in(0, 3) : pick == 1 ? rng.in(1, 40) : rng.in(90, 200);
        const std::string A = random_seq(rng, alpha, la), B = random_seq(rng, alpha, lb);
        std::vector<uint16_t> track;
        if (sets > 1) { track.resize(static_cast<size_t>(la)); for (auto& t : track) t = static_cast<uint16_t>(rng.in(0, sets - 1)); }
        const int level = levels[static_cast<size_t>(rng.in(0, static_cast<int>(levels.size()) - 1))];
        const bool global = rng.in(0, 1) == 1;
        const auto ea = alpha.encode(A), eb = alpha.encode(B);
        auto one = [&](auto am_tag) {
            constexpr AlignMode AM = decltype(am_tag)::value;
            const Result sH = run_one<AM, T>(ea, eb, p, track, kBackendScalar, TracebackMode::Scores);
            const Result sP = run_one<AM, T>(ea, eb, p, track, kBackendScalar, TracebackMode::Pointers);
            const Result kH = run_one<AM, T>(ea, eb, p, track, level, TracebackMode::Scores);
            const Result kP = run_one<AM, T>(ea, eb, p, track, level, TracebackMode::Pointers);
            ++cases;
            kernel_runs += kH.layout + kP.layout;
            const bool has = level_kernels(level).template of<T>().linear != nullptr;
            const int min_n = Aligner<GapModel::Linear, AM, AlignBand::Full, T>::linear_striped_min_n(level);
            expected_runs += (has && lb >= min_n) ? 2 : 0;
            const bool ok = same(sH, kH) && same(sP, kP) && same(kH, kP);
            if (!ok) {
                ++mismatches;
                if (first.empty())
                    first = std::string(nw_type_name<T>()) + (AM == AlignMode::Global ? " global" : " local") +
                            " level " + std::to_string(level) + " |A|=" + std::to_string(la) + " |B|=" + std::to_string(lb) +
                            " sets " + std::to_string(sets) + " scalar H " + std::to_string(sH.score) +
                            " kernel H " + std::to_string(kH.score) + " kernel P " + std::to_string(kP.score) +
                            " layout " + std::to_string(kH.layout) + "/" + std::to_string(kP.layout);
            }
        };
        if (global) one(std::integral_constant<AlignMode, AlignMode::Global>{});
        else        one(std::integral_constant<AlignMode, AlignMode::Local>{});
    }
}

}  // namespace

TEMPLATE_TEST_CASE("striped linear kernels equal the scalar linear fills bit for bit", "[linear][simd]",
                   double, float, int16_t, int32_t, int64_t) {
    int mismatches = 0, cases = 0, kernel_runs = 0, expected_runs = 0;
    std::string first;
    sweep<TestType>(0x5eedf00d, 800, mismatches, cases, kernel_runs, expected_runs, first);
    INFO("first mismatch: " << first);
    CHECK(mismatches == 0);
    CHECK(cases == 800);
    // liveness: the leveled kernel must have run exactly where the level has it and the
    // length gate admits the pair (striped layout reported) — a silent fallback to the
    // scalar fill would pass every comparison above vacuously, and a gate that stopped
    // routing would too
    CHECK(kernel_runs == expected_runs);
    if (!std::is_same_v<TestType, int64_t>) CHECK(expected_runs > cases / 2);
}

// The linear Hirschberg kernel (lhb_striped: both sweeps, the base case, the Local
// forward and reverse endpoint scans) against the scalar loops it replaces, at every
// level: Hirschberg may pick another optimal path than Pointers at ties, but the SAME
// algorithm on two backends must not — score, path, guide and gradient bit for bit.
// Small cutoffs keep the recursion splitting; B spans the length gate.
template <class T>
void sweep_hb(uint64_t seed, int rounds, int& mismatches, int& cases, int& kernel_calls, std::string& first) {
    Rng rng(seed);
    const Alphabet* alphas[] = {&Alphabet::get("ACGT"), &Alphabet::get("ACDEFGHIKLMNPQRSTVWY"), &Alphabet::get("AB")};
    std::vector<int> levels;
    for (SimdLevel l : detect_available_levels()) levels.push_back(static_cast<int>(l));
    if (levels.empty()) return;
    for (int r = 0; r < rounds; ++r) {
        const Alphabet& alpha = *alphas[rng.in(0, 2)];
        const int sets = rng.in(0, 3) == 0 ? rng.in(2, 4) : 1;
        const AlignParams p = make_params<T>(rng, alpha, sets, true);
        const int la = rng.in(0, 4) == 0 ? rng.in(0, 6) : rng.in(10, 70);
        const int lb = rng.in(0, 4) == 0 ? rng.in(0, 30) : rng.in(90, 220);
        const std::string A = random_seq(rng, alpha, la), B = random_seq(rng, alpha, lb);
        std::vector<uint16_t> track;
        if (sets > 1) { track.resize(static_cast<size_t>(la)); for (auto& t : track) t = static_cast<uint16_t>(rng.in(0, sets - 1)); }
        const int level = levels[static_cast<size_t>(rng.in(0, static_cast<int>(levels.size()) - 1))];
        const int cutoff = rng.in(1, 8);
        const TracebackMode tb = rng.in(0, 1) ? TracebackMode::Hirschberg : TracebackMode::HirschbergPmax;
        const auto ea = alpha.encode(A), eb = alpha.encode(B);
        auto one = [&](auto am_tag) {
            constexpr AlignMode AM = decltype(am_tag)::value;
            int runs = 0;
            const Result sc = run_one<AM, T>(ea, eb, p, track, kBackendScalar, tb, cutoff);
            const Result kv = run_one<AM, T>(ea, eb, p, track, level, tb, cutoff, &runs);
            ++cases;
            kernel_calls += runs;
            if (!same(sc, kv)) {
                ++mismatches;
                if (first.empty())
                    first = std::string(nw_type_name<T>()) + (AM == AlignMode::Global ? " global" : " local") +
                            " level " + std::to_string(level) + " cutoff " + std::to_string(cutoff) +
                            " |A|=" + std::to_string(la) + " |B|=" + std::to_string(lb) + " sets " + std::to_string(sets) +
                            " scalar " + std::to_string(sc.score) + " kernel " + std::to_string(kv.score) +
                            " (" + sc.aligned.first + " / " + kv.aligned.first + ")";
            }
        };
        if (rng.in(0, 1)) one(std::integral_constant<AlignMode, AlignMode::Global>{});
        else              one(std::integral_constant<AlignMode, AlignMode::Local>{});
    }
}

TEMPLATE_TEST_CASE("linear Hirschberg kernel equals the scalar linear Hirschberg bit for bit", "[linear][simd]",
                   double, float, int16_t, int32_t, int64_t) {
    int mismatches = 0, cases = 0, kernel_calls = 0;
    std::string first;
    sweep_hb<TestType>(0x0b5e55ed, 500, mismatches, cases, kernel_calls, first);
    INFO("first mismatch: " << first);
    CHECK(mismatches == 0);
    CHECK(cases == 500);
    // liveness: unless this type has no kernel on any level here (int64 below 8 lanes),
    // the kernel must have run — a silent fallback would pass the comparison vacuously
    bool any = false;
    for (SimdLevel l : detect_available_levels())
        any |= level_kernels(static_cast<int>(l)).template of<TestType>().lhb != nullptr;
    if (any) CHECK(kernel_calls > cases);
    else     CHECK(kernel_calls == 0);
}
