// ── One DpBuffer, many problems: reuse must be invisible ────────────────────
//
// A batch worker owns one DpBuffer and runs every problem it is handed through it, with
// whatever Aligner instantiation that problem needs.  Each fill grows the vectors IT uses;
// a fill that tests one vector's size and then resizes a whole group trusts every other
// fill to keep that group in step.  The linear pointer fill (1 B/cell direction bytes)
// grew DM, rM and qM alone, and the affine pointer kernels, gating DX/DY and rX..qY on DM
// and rM, then wrote through empty or short vectors: a segfault, or heap corruption, on a
// float32 batch holding a linear pair followed by a narrower affine one.
//
// Here one buffer runs a sequence of problems across all four (gap model, mode) pairs,
// full and banded, every traceback mode, sizes rising and falling, and each result must
// equal the same problem's on a fresh buffer — bit for bit.  Under ASan an overrun is a
// hard failure even where the numbers would happen to survive it.

#include "catch.hpp"

#include <random>
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

struct Result {
    double score;
    std::pair<std::string, std::string> aln;
    std::vector<double> grad;
    bool operator==(const Result&) const = default;
};

std::vector<double> flat(const AlignParams& g) {
    const int N = g.matrix.alphabet().size();
    std::vector<double> v;
    for (int a = 0; a < N; ++a)
        for (int b = 0; b < N; ++b) v.push_back(g.matrix.at(a, b));
    v.insert(v.end(), {g.gap_open_a, g.gap_extend_a, g.gap_open_b, g.gap_extend_b});
    return v;
}

struct Problem {
    int gm, am, banded;       // GapModel::Linear=0/Affine=1, Global=0/Local=1
    TracebackMode tb;
    std::string a, b;
};

template <GapModel GM, AlignMode AM, AlignBand AB, class T>
Result run_one(const Problem& pr, const AlignParams& p, int backend, DpBufferT<T>& buf) {
    Aligner<GM, AM, AB, T> al;
    al.set_kernel(backend);
    al.set_traceback(pr.tb);
    al.set_hb_cutoff(8);     // keep the Hirschberg recursion (and its base case) in play
    const auto ea = p.matrix.alphabet().encode(pr.a);
    const auto eb = p.matrix.alphabet().encode(pr.b);
    al.set_problem(ea, eb, p, AB == AlignBand::GuideBanded ? 3 : 0);
    al.compute_viterbi(buf);
    AlignParams g(p.matrix.alphabet());
    al.hard_grad(buf, g);
    return {al.score(), al.aligned(buf), flat(g)};
}

template <class T>
Result run(const Problem& pr, const AlignParams& p, int backend, DpBufferT<T>& buf) {
#define NWG_ARM(G, M)                                                                    \
    if (pr.banded)                                                                       \
        return run_one<GapModel::G, AlignMode::M, AlignBand::GuideBanded, T>(pr, p, backend, buf); \
    return run_one<GapModel::G, AlignMode::M, AlignBand::Full, T>(pr, p, backend, buf);
    if (pr.gm == 0 && pr.am == 0) { NWG_ARM(Linear, Global) }
    if (pr.gm == 0 && pr.am == 1) { NWG_ARM(Linear, Local) }
    if (pr.gm == 1 && pr.am == 0) { NWG_ARM(Affine, Global) }
    NWG_ARM(Affine, Local)
#undef NWG_ARM
}

std::string rseq(std::mt19937& rng, int len) {
    static const char* al = "ACGT";
    std::string s(static_cast<size_t>(len), 'A');
    for (auto& c : s) c = al[rng() % 4];
    return s;
}

template <class T>
void check_sequence(unsigned seed) {
    std::mt19937 rng(seed);
    std::vector<double> m(16);
    for (auto& x : m) x = static_cast<double>(static_cast<int>(rng() % 9) - 3);
    const AlignParams lin(SubstMatrix(m.data(), "ACGT"), 0.0, 1.0, 0.0, 0.8);
    const AlignParams aff(SubstMatrix(m.data(), "ACGT"), 2.5, 0.5, 3.0, 0.5);
    const TracebackMode tbs[] = {TracebackMode::Pointers, TracebackMode::Scores,
                                 TracebackMode::Hirschberg, TracebackMode::Default};
    std::vector<Problem> seq;
    for (int k = 0; k < 40; ++k) {
        Problem pr;
        pr.gm = static_cast<int>(rng() % 2);
        pr.am = static_cast<int>(rng() % 2);
        pr.banded = (rng() % 4 == 0);
        pr.tb = tbs[rng() % 4];
        if (pr.banded && pr.tb == TracebackMode::Hirschberg) pr.tb = TracebackMode::Scores;
        // Sizes that rise and fall, so a later, smaller problem meets vectors an earlier
        // one grew — the case the per-group guards got wrong.
        const int big = (k % 3 == 0) ? 150 : 40;
        pr.a = rseq(rng, 1 + static_cast<int>(rng() % static_cast<unsigned>(big)));
        pr.b = rseq(rng, 1 + static_cast<int>(rng() % static_cast<unsigned>(big)));
        seq.push_back(pr);
    }
    // The bug's own shape, first: a wide linear pointer fill, then a narrower affine one.
    seq.insert(seq.begin(), {Problem{1, 0, 0, TracebackMode::Pointers, rseq(rng, 42), rseq(rng, 58)}});
    seq.insert(seq.begin(), {Problem{0, 0, 0, TracebackMode::Pointers, rseq(rng, 154), rseq(rng, 51)}});

    for (int backend : backends_under_test()) {
        DpBufferT<T> shared;
        for (size_t k = 0; k < seq.size(); ++k) {
            const Problem& pr = seq[k];
            const AlignParams& p = pr.gm ? aff : lin;
            INFO("seed=" << seed << " backend=" << backend_name(backend) << " step=" << k
                 << " gm=" << pr.gm << " am=" << pr.am << " banded=" << pr.banded
                 << " tb=" << static_cast<int>(pr.tb) << " m=" << pr.a.size()
                 << " n=" << pr.b.size());
            DpBufferT<T> fresh;
            const Result want = run<T>(pr, p, backend, fresh);
            const Result got  = run<T>(pr, p, backend, shared);
            CHECK(got == want);
        }
    }
}

} // namespace

TEST_CASE("one DpBuffer reused across gap models, modes, bands and sizes", "[buffer_reuse]") {
    for (unsigned seed : {1u, 2u, 3u}) {
        check_sequence<double>(seed);
        check_sequence<float>(seed);
    }
}
