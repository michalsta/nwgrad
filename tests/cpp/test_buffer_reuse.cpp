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

#include <cstdint>
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

// ── Row placement (dp_buffer.hpp): the 4 KB-aliasing fix must actually be in force ──
//
// Every result above is bit-identical whether or not the rows are spread, so nothing there
// would notice the placement silently not happening.  These pin the mechanics: PlacedVec
// lands where asked and keeps its contents, spread_stride / table_lines obey their rules,
// and a striped Scores fill leaves VM at line 0 and VM / VX / VY >= 4 lines apart.
TEST_CASE("rolling rows and striped tables are placed apart", "[buffer_reuse][layout]") {
    auto line_of = [](const void* p) {
        return static_cast<int>(reinterpret_cast<std::uintptr_t>(p) % 4096 / 64);
    };
    auto dist = [](int x) { x = ((x % 64) + 64) % 64; return x < 64 - x ? x : 64 - x; };

    PlacedVec<double> v;
    v.place(5);
    v.resize(100);
    CHECK(line_of(v.data()) == 5);
    for (size_t k = 0; k < 100; ++k) v[k] = static_cast<double>(k);
    v.place(17);
    v.resize(300000);   // past 2 MiB: the allocator's huge-page alignment, then the offset
    CHECK(line_of(v.data()) == 17);
    for (size_t k = 0; k < 100; ++k) CHECK(v[k] == static_cast<double>(k));

    for (std::size_t len : {1u, 16u, 1004u, 1016u, 2040u, 4100u})
        for (int nrows = 2; nrows <= 12; ++nrows) {
            const std::size_t st = spread_stride<int>(len, nrows);
            const int r = static_cast<int>(st / 16 % 64);
            CHECK(st >= len);
            CHECK(r % 4 != 0);
            for (int k = 1; k < nrows; ++k) CHECK(dist(k * r) >= 4);
        }
    for (int r = 0; r < 64; ++r) {
        if (r % 4 == 0 || dist(r) < 4) continue;   // never chosen by spread_stride(len, 2)
        const TableLines tl = table_lines(static_cast<std::size_t>(r));
        const int s[6] = {0, tl.a, tl.b, -r, tl.a - r, tl.b - r};
        for (int x = 0; x < 6; ++x)
            for (int y = x + 1; y < 6; ++y) CHECK(dist(s[x] - s[y]) >= 4);
    }

    std::mt19937 rng(7);
    std::vector<double> mat(16);
    for (auto& x : mat) x = static_cast<double>(static_cast<int>(rng() % 9) - 3);
    const AlignParams p(SubstMatrix(mat.data(), "ACGT"), 2.5, 0.5, 3.0, 0.5);
    const auto ea = p.matrix.alphabet().encode(rseq(rng, 300));   // the aligner views them
    const auto eb = p.matrix.alphabet().encode(rseq(rng, 300));
    struct Spread {   // table_spread() forced on for the placement checks, auto after
        explicit Spread(int m) { table_spread_mode().store(m); }
        ~Spread() { table_spread_mode().store(-1); }
    };
    // Gated on a narrow L1D (dp_buffer.hpp), so a host's own setting runs ONE of the two
    // table layouts: both run here, on the same problems, and must agree bit for bit.
    {
        std::vector<Problem> probs;
        for (int k = 0; k < 24; ++k) {
            const int len = (k % 4 == 0) ? 260 : 1 + static_cast<int>(rng() % 120);
            probs.push_back({k % 2, (k / 2) % 2, 0, (k % 3 == 0) ? TracebackMode::Pointers : TracebackMode::Scores,
                             rseq(rng, len), rseq(rng, 1 + static_cast<int>(rng() % 300))});
        }
        const AlignParams lin(SubstMatrix(mat.data(), "ACGT"), 0.0, 1.0, 0.0, 0.8);
        for (int backend : backends_under_test())
            for (const Problem& pr : probs) {
                INFO("backend=" << backend_name(backend) << " gm=" << pr.gm << " am=" << pr.am
                     << " m=" << pr.a.size() << " n=" << pr.b.size());
                DpBufferT<double> b0, b1;
                Result r0, r1;
                { Spread off(0); r0 = run<double>(pr, pr.gm ? p : lin, backend, b0); }
                { Spread on(1);  r1 = run<double>(pr, pr.gm ? p : lin, backend, b1); }
                CHECK(r0 == r1);
            }
    }
    Spread on(1);
    int striped = 0;
    for (int backend : backends_under_test()) {
        INFO("backend=" << backend_name(backend));
        Aligner<GapModel::Affine, AlignMode::Global, AlignBand::Full, double> al;
        al.set_kernel(backend);
        al.set_traceback(TracebackMode::Scores);
        al.set_problem(ea, eb, p, 0);
        DpBufferT<double> buf;
        al.compute_viterbi(buf);
        if (!al.tables_striped()) continue;   // scalar, or a build without leveled kernels
        ++striped;
        const int lm = line_of(buf.VM.data()), lx = line_of(buf.VX.data()), ly = line_of(buf.VY.data());
        CHECK(lm == 0);
        CHECK(dist(lx - lm) >= 4);
        CHECK(dist(ly - lm) >= 4);
        CHECK(dist(ly - lx) >= 4);
    }
    if (!striped) WARN("no backend ran the striped Scores kernel: placement unchecked");
}
