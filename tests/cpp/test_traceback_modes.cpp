// ── TracebackMode::Pointers must be bit-identical to TracebackMode::Scores ───
//
// Pointers replaces the three retained score tables with one byte per cell per state
// and two rolling rows: 3 B/cell against 12.  The claim it has to earn is not "close"
// but IDENTICAL — same score, same alignment, same guide_j, same gradient, ties
// included.
// That is the property Hirschberg could not offer (its midpoint split picks a different
// valid subgradient at ties), and it is the whole reason Pointers was adopted and
// Hirschberg was not, so it is worth testing directly rather than by proxy.
//
// The substitution matrix here is deliberately integral (+4 / -1) with integral gap
// costs, so ties are COMMON rather than incidental.  A tie-break that drifted by one
// `>=` would show up here and nowhere else — under a random real-valued matrix exact
// ties essentially never occur, and the test would pass while the property was broken.

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

// A deterministic tie-heavy generator: xorshift so the sequence is identical on every
// platform (a std::mt19937 distribution is not portable across libstdc++/libc++).
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed ? seed : 1) {}
    uint32_t next() {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        return static_cast<uint32_t>(s >> 32);
    }
    int in(int lo, int hi) { return lo + static_cast<int>(next() % static_cast<uint32_t>(hi - lo + 1)); }
};

template <AlignMode AM, class T>
void compare_one(const std::string& A, const std::string& B, const AlignParams& p,
                 int backend, int& mismatches) {
    Aligner<GapModel::Affine, AM, AlignBand::Full, T> a, b;
    a.set_kernel(backend);
    b.set_kernel(backend);
    a.set_traceback(TracebackMode::Scores);
    b.set_traceback(TracebackMode::Pointers);

    const auto ea = p.matrix.alphabet().encode(A);
    const auto eb = p.matrix.alphabet().encode(B);
    DpBufferT<T> ba, bb;
    a.set_problem(ea, eb, p); a.compute_viterbi(ba);
    b.set_problem(ea, eb, p); b.compute_viterbi(bb);

    bool ok = (a.score() == b.score());
    if (a.guide_j_from_viterbi(ba) != b.guide_j_from_viterbi(bb)) ok = false;
    if (a.aligned(ba) != b.aligned(bb)) ok = false;

    AlignParams ga(p.matrix.alphabet()), gb(p.matrix.alphabet());
    a.hard_grad(ba, ga);
    b.hard_grad(bb, gb);
    const int N = p.matrix.alphabet().size();
    for (int x = 0; x < N && ok; ++x)
        for (int y = 0; y < N; ++y)
            if (ga.matrix.at(x, y) != gb.matrix.at(x, y)) { ok = false; break; }
    if (ga.gap_open_a   != gb.gap_open_a   || ga.gap_extend_a != gb.gap_extend_a ||
        ga.gap_open_b   != gb.gap_open_b   || ga.gap_extend_b != gb.gap_extend_b) ok = false;

    if (!ok) ++mismatches;
}

}  // namespace

TEST_CASE("traceback=pointers is bit-identical to traceback=scores", "[simd][traceback]") {
    const Alphabet& al = Alphabet::get("ACDEFGHIKLMNPQRSTVWY");
    SubstMatrix M(al);
    for (int x = 0; x < al.size(); ++x)
        for (int y = 0; y < al.size(); ++y)
            M.at(x, y) = (x == y) ? 4.0 : -1.0;      // integral => ties are common
    const AlignParams p(M, 11.0, 1.0, 11.0, 1.0);

    for (int backend : backends_under_test()) {
        INFO("backend = " << backend_name(backend));
        Rng rng(0x9E3779B9u);
        int mismatches = 0;
        for (int t = 0; t < 300; ++t) {
            std::string A, B;
            for (int k = rng.in(0, 70); k > 0; --k) A += al.symbol_at(rng.in(0, al.size() - 1));
            for (int k = rng.in(0, 70); k > 0; --k) B += al.symbol_at(rng.in(0, al.size() - 1));
            compare_one<AlignMode::Global, double>(A, B, p, backend, mismatches);
            compare_one<AlignMode::Local,  double>(A, B, p, backend, mismatches);
            compare_one<AlignMode::Global, float >(A, B, p, backend, mismatches);
            compare_one<AlignMode::Local,  float >(A, B, p, backend, mismatches);
        }
        REQUIRE(mismatches == 0);
    }
}

TEST_CASE("traceback=pointers handles the degenerate shapes", "[traceback]") {
    const Alphabet& al = Alphabet::get("ACDEFGHIKLMNPQRSTVWY");
    SubstMatrix M(al);
    for (int x = 0; x < al.size(); ++x)
        for (int y = 0; y < al.size(); ++y) M.at(x, y) = (x == y) ? 4.0 : -1.0;
    const AlignParams p(M, 11.0, 1.0, 11.0, 1.0);

    // Empty sequences, single residues, and one-sided emptiness all walk the BORDER
    // cells of the direction tables — the ones the recurrence never visits and which
    // therefore have to be written explicitly by the fill.
    const std::vector<std::string> odd{"", "A", "AC", "ACDEFGHIK"};
    for (int backend : backends_under_test()) {
        INFO("backend = " << backend_name(backend));
        int mismatches = 0;
        for (const auto& A : odd)
            for (const auto& B : odd) {
                compare_one<AlignMode::Global, double>(A, B, p, backend, mismatches);
                compare_one<AlignMode::Local,  double>(A, B, p, backend, mismatches);
                compare_one<AlignMode::Global, float >(A, B, p, backend, mismatches);
                compare_one<AlignMode::Local,  float >(A, B, p, backend, mismatches);
            }
        REQUIRE(mismatches == 0);
    }
}
