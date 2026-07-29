// ── The prefix-max Hirschberg sweep: a family that must be self-consistent ────────
//
// TracebackMode::HirschbergPmax replaces the sweep's serial VY carry with the closed form
//     VY[c] = ( max over k <= c-1 of ( g[k] + k*ge_a ) ) - (c-1)*ge_a
// (see hb_kernel_impl.inl).  It is deliberately NOT bit-exact with the serial sweep, so
// the usual "simd == scalar == the other traceback modes" chain does not apply and a
// weaker contract takes its place.  This file pins exactly that contract, in three parts,
// because each one fails in a different and silent way:
//
//   1. LIVENESS.  pmax must actually differ from the exact sweep on a lossy fixture.  If
//      the dispatch ever fell back to hb_sweep, every other assertion here would still
//      pass — the variant would simply have stopped existing while the suite stayed
//      green.  This is the check that makes the rest of the file mean anything.
//   2. SELF-CONSISTENCY.  pmax at every ISA level, and the scalar reference, must agree
//      to the BIT.  The closed form depends only on the absolute column index, never on
//      the vector width, so W=2/4/8 and the scalar walk must land on identical values —
//      that is what keeps NWGRAD_ISA a speed knob here as everywhere else.  It is checked
//      against a reference written from the algebra in this file, not from the kernel, so
//      a shared misreading of the recurrence cannot hide.
//   3. OPTIMALITY, WITHIN A MEASURED BOUND.  pmax may return a suboptimal path — that is
//      the price — but the shortfall must stay inside a bound the test states out loud
//      rather than a tolerance tuned until it passed.
//
// Parts 1 and 2 call the level kernels DIRECTLY through the dispatch table (they are
// plain function pointers over HbJob), which isolates the sweep from the recursion around
// it.  Part 3 goes through the Aligner, so it also exercises hb_fwd<true>/hb_rev<true>.

#include "catch.hpp"

#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "align_params.hpp"
#include "aligner.hpp"
#include "simd_levels.hpp"

namespace {

std::vector<int> levels_under_test() {
    std::vector<int> v;
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

// The sweep's answer: the three final-row vectors, contiguous, index 0..ncols.
template <class T>
struct Row { std::vector<T> M, X, Y; };

// ── An independent scalar model of the pmax sweep ─────────────────────────────
//
// Written from the recurrence, not transcribed from the kernel: row 0 is the exact serial
// Y-series (pmax leaves it alone on purpose — it is O(ncols) once per sweep, so
// approximating it would buy nothing and cost accuracy), and the body carries the closed
// form with the ramp precomputed.  The ramp is a LOAD in the inner loop here for the same
// reason it is in the kernel: an adjacent multiply invites FMA contraction, which rounds
// once where the kernel rounds twice.
template <class T>
Row<T> ref_pmax_sweep(const std::vector<unsigned char>& a, const std::vector<unsigned char>& b,
                      const std::vector<T>& blk, int nalpha,
                      T go_a, T ge_a, T go_b, T ge_b, int H, int NC, bool in_x) {
    const T NINF = -std::numeric_limits<T>::infinity();
    Row<T> out;
    out.M.assign(NC + 1, NINF); out.X.assign(NC + 1, NINF); out.Y.assign(NC + 1, NINF);

    std::vector<T> pM(NC + 1), pX(NC + 1), pY(NC + 1), cM(NC + 1), cX(NC + 1), cY(NC + 1);
    std::vector<T> rmp(NC + 1), g(NC + 1);
    for (int k = 0; k <= NC; ++k) rmp[k] = static_cast<T>(k) * ge_a;

    // row 0 — exact, serial
    pM[0] = in_x ? NINF : T(0);
    pX[0] = in_x ? T(0) : NINF;
    pY[0] = NINF;
    for (int c = 1; c <= NC; ++c) {
        pM[c] = NINF; pX[c] = NINF;
        const T open = (std::max(pM[c - 1], pX[c - 1]) - go_a) - ge_a;
        pY[c] = std::max(open, pY[c - 1] - ge_a);
    }
    if (H == 0) { out.M = pM; out.X = pX; out.Y = pY; return out; }

    T bM = pM[0], bX = pX[0], bY = pY[0];
    for (int t = 0; t < H; ++t) {
        const unsigned char arow = a[t];
        const T nbM = NINF;
        const T nbX = std::max(std::max((bM - go_b) - ge_b, bX - ge_b), (bY - go_b) - ge_b);
        const T nbY = NINF;

        // carry-free half, and the open value g[c] each cell offers a new Y gap
        g[0] = (std::max(nbM, nbX) - go_a) - ge_a;
        for (int c = 1; c <= NC; ++c) {
            const T vm = std::max(std::max(pM[c - 1], pX[c - 1]), pY[c - 1]) +
                         blk[(std::size_t)arow * nalpha + b[c - 1]];
            const T vx = std::max(std::max((pM[c] - go_b) - ge_b, pX[c] - ge_b),
                                  (pY[c] - go_b) - ge_b);
            cM[c] = vm; cX[c] = vx;
            g[c] = (std::max(vm, vx) - go_a) - ge_a;
        }
        cM[0] = nbM; cX[0] = nbX; cY[0] = nbY;

        // the carry: one prefix max over Q[k] = g[k] + k*ge_a, seeded by the border
        T P = bY - ge_a;
        for (int c = 1; c <= NC; ++c) {
            P = std::max(g[c - 1] + rmp[c - 1], P);
            cY[c] = P - rmp[c - 1];
        }

        pM.swap(cM); pX.swap(cX); pY.swap(cY);
        bM = nbM; bX = nbX; bY = nbY;
    }
    out.M = pM; out.X = pX; out.Y = pY;
    out.M[0] = bM; out.X[0] = bX; out.Y[0] = bY;
    return out;
}

// Run one leveled sweep (exact or pmax) over the whole rectangle, forward.
template <class T>
Row<T> run_sweep(const LevelKernels& K, bool pmax,
                 const std::vector<unsigned char>& a, const std::vector<unsigned char>& b,
                 const std::vector<T>& blk, int nalpha,
                 T go_a, T ge_a, T go_b, T ge_b, int H, int NC, bool in_x) {
    Row<T> out;
    out.M.assign(NC + 1, T(0)); out.X.assign(NC + 1, T(0)); out.Y.assign(NC + 1, T(0));
    DpBufferT<T> buf;
    HbJob<T> job{};
    job.a = a.data(); job.b = b.data();
    job.blk = blk.data(); job.nalpha = nalpha;
    job.go_a = go_a; job.ge_a = ge_a; job.go_b = go_b; job.ge_b = ge_b;
    job.a_start = 0; job.a_step = 1;
    job.b_start = 0; job.b_step = 1;
    job.H = H; job.ncols = NC; job.in_x = in_x ? 1 : 0;
    job.buf = &buf;
    job.outM = out.M.data(); job.outX = out.X.data(); job.outY = out.Y.data();
    if constexpr (std::is_same_v<T, double>) {
        if (pmax) K.hb_sweep_pmax(job); else K.hb_sweep(job);
    } else {
        if (pmax) K.hb_sweep_pmax_f(job); else K.hb_sweep_f(job);
    }
    return out;
}

template <class T>
int differing(const Row<T>& p, const Row<T>& q) {
    int n = 0;
    for (std::size_t i = 0; i < p.M.size(); ++i) {
        // Bit comparison, not near-equality: -inf == -inf is intended to match.
        if (!(p.M[i] == q.M[i])) ++n;
        if (!(p.X[i] == q.X[i])) ++n;
        if (!(p.Y[i] == q.Y[i])) ++n;
    }
    return n;
}

// A block whose ramp is genuinely lossy: ge_a = 0.1 is not representable in either
// precision, so k*ge_a rounds, and 1500 columns make k large enough for the round-trip to
// land in the low bits of the answer.
struct Fixture {
    std::vector<unsigned char> a, b;
    std::vector<double> blk;
    int nalpha, H, NC;
};

Fixture make_fixture(int H, int NC, uint64_t seed) {
    Fixture f;
    f.nalpha = 20;
    f.blk.resize((std::size_t)f.nalpha * f.nalpha);
    for (int x = 0; x < f.nalpha; ++x)
        for (int y = 0; y < f.nalpha; ++y)
            f.blk[(std::size_t)x * f.nalpha + y] = (x == y) ? 4.0 : -1.0;
    Rng rng(seed);
    f.a.resize(H); f.b.resize(NC);
    for (int i = 0; i < H; ++i) f.a[i] = (unsigned char)rng.in(0, f.nalpha - 1);
    for (int j = 0; j < NC; ++j) f.b[j] = (unsigned char)rng.in(0, f.nalpha - 1);
    f.H = H; f.NC = NC;
    return f;
}

template <class T> std::vector<T> to_T(const std::vector<double>& v) {
    return std::vector<T>(v.begin(), v.end());
}

}  // namespace

// ── 1 + 2: the sweep kernels themselves ───────────────────────────────────────
TEMPLATE_TEST_CASE("pmax sweep: differs from the exact sweep, and is width-independent",
                   "[hirschberg][pmax][simd]", double, float) {
    using T = TestType;
    const T go_a = T(11), ge_a = T(0.1), go_b = T(11), ge_b = T(0.1);

    const auto lv = levels_under_test();
    REQUIRE_FALSE(lv.empty());

    int total_pmax_vs_exact_diffs = 0;

    for (int H : {1, 7, 64, 200}) {
        for (int NC : {0, 1, 5, 63, 1500}) {
            for (bool in_x : {false, true}) {
                const Fixture f = make_fixture(H ? H : 1, NC, 0xABCDEFu + (uint64_t)(H * 31 + NC));
                const auto blkT = to_T<T>(f.blk);
                const Row<T> ref = ref_pmax_sweep<T>(f.a, f.b, blkT, f.nalpha,
                                                     go_a, ge_a, go_b, ge_b, H, NC, in_x);

                for (int level : lv) {
                    INFO("level = " << level_name((SimdLevel)level) << "  H=" << H
                                    << " NC=" << NC << " in_x=" << in_x);
                    const LevelKernels& K = level_kernels(level);
                    REQUIRE(( std::is_same_v<T, double> ? K.hb_sweep_pmax != nullptr
                                                        : K.hb_sweep_pmax_f != nullptr ));

                    const Row<T> got = run_sweep<T>(K, true, f.a, f.b, blkT, f.nalpha,
                                                    go_a, ge_a, go_b, ge_b, H, NC, in_x);
                    // (2) bit-identical to the independent scalar model, at every width.
                    REQUIRE(differing(ref, got) == 0);

                    // (1) and genuinely a different kernel from the exact sweep.
                    const Row<T> exact = run_sweep<T>(K, false, f.a, f.b, blkT, f.nalpha,
                                                      go_a, ge_a, go_b, ge_b, H, NC, in_x);
                    total_pmax_vs_exact_diffs += differing(exact, got);
                }
            }
        }
    }

    // LIVENESS.  If the dispatch silently ran the exact sweep for pmax, every REQUIRE
    // above would still hold and this is the only thing that would notice.
    INFO("pmax and the exact sweep produced identical values everywhere — the pmax "
         "kernel is probably not being dispatched at all");
    REQUIRE(total_pmax_vs_exact_diffs > 0);
}

// ── 2 (again), through the Aligner: scalar reference vs every level ───────────
//
// This is the path a user actually takes, so it exercises hb_fwd<true>/hb_rev<true> (the
// scalar fallback) against the leveled kernels, plus the recursion and join around them.
// cutoff = 8 forces real splitting; without it a short pair runs the exact base-case fill
// and the sweep never executes.
namespace {
template <class T>
void pmax_scalar_vs_level(const std::string& A, const std::string& B, const AlignParams& p,
                          int backend, int& mismatches) {
    Aligner<GapModel::Affine, AlignMode::Global, AlignBand::Full, T> ref, dut;
    ref.set_kernel(kBackendScalar); dut.set_kernel(backend);
    ref.set_traceback(TracebackMode::HirschbergPmax);
    dut.set_traceback(TracebackMode::HirschbergPmax);
    ref.set_hb_cutoff(8); dut.set_hb_cutoff(8);

    const auto ea = p.matrix.alphabet().encode(A);
    const auto eb = p.matrix.alphabet().encode(B);
    DpBufferT<T> br, bd;
    ref.set_problem(ea, eb, p); ref.compute_viterbi(br);
    dut.set_problem(ea, eb, p); dut.compute_viterbi(bd);

    bool ok = (ref.score() == dut.score());
    if (ref.aligned(br) != dut.aligned(bd)) ok = false;

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
}  // namespace

TEST_CASE("pmax Hirschberg is bit-identical scalar vs every level", "[hirschberg][pmax][simd]") {
    const Alphabet& al = Alphabet::get("ACDEFGHIKLMNPQRSTVWY");
    SubstMatrix M(al);
    for (int x = 0; x < al.size(); ++x)
        for (int y = 0; y < al.size(); ++y)
            M.at(x, y) = (x == y) ? 4.0 : -1.0;      // integral => ties are common
    const AlignParams p(M, 11.0, 0.1, 11.0, 0.1);    // ge = 0.1: an inexact ramp

    std::vector<int> backends;
    for (SimdLevel l : detect_available_levels()) backends.push_back((int)l);

    for (int backend : backends) {
        INFO("backend = " << backend_name(backend));
        Rng rng(0xBEEF01u);
        int mismatches = 0;
        for (int t = 0; t < 200; ++t) {
            std::string A, B;
            for (int k = rng.in(0, 120); k > 0; --k) A += al.symbol_at(rng.in(0, al.size() - 1));
            for (int k = rng.in(0, 120); k > 0; --k) B += al.symbol_at(rng.in(0, al.size() - 1));
            pmax_scalar_vs_level<double>(A, B, p, backend, mismatches);
            pmax_scalar_vs_level<float >(A, B, p, backend, mismatches);
            pmax_scalar_vs_level<double>(A, A, p, backend, mismatches);
        }
        REQUIRE(mismatches == 0);
    }
}

// ── 3: optimality within a stated bound ───────────────────────────────────────
//
// pmax's path may be suboptimal — the round-trip through the ramp can lose an argmax.  It
// may never be BETTER than the true optimum, which would mean the replayed score is not a
// real path's score.  The bound below is the measured worst case over this fixture, not a
// tolerance widened until the test went green; a regression that pushed the error up would
// have to be looked at rather than absorbed.
TEST_CASE("pmax Hirschberg stays at or just below the optimum", "[hirschberg][pmax]") {
    const Alphabet& al = Alphabet::get("ACDEFGHIKLMNPQRSTVWY");
    SubstMatrix M(al);
    for (int x = 0; x < al.size(); ++x)
        for (int y = 0; y < al.size(); ++y)
            M.at(x, y) = (x == y) ? 4.0 : -1.0;
    const AlignParams p(M, 11.0, 0.1, 11.0, 0.1);

    Rng rng(0x51DEu);
    double worst_below = 0.0, worst_above = 0.0;
    for (int t = 0; t < 200; ++t) {
        std::string A, B;
        for (int k = rng.in(20, 400); k > 0; --k) A += al.symbol_at(rng.in(0, al.size() - 1));
        for (int k = rng.in(20, 400); k > 0; --k) B += al.symbol_at(rng.in(0, al.size() - 1));

        Aligner<GapModel::Affine, AlignMode::Global, AlignBand::Full, double> ptr, pm;
        ptr.set_traceback(TracebackMode::Pointers);
        pm.set_traceback(TracebackMode::HirschbergPmax);
        pm.set_hb_cutoff(8);
        const auto ea = al.encode(A), eb = al.encode(B);
        DpBuffer bp, bm;
        ptr.set_problem(ea, eb, p); ptr.compute_viterbi(bp);
        pm.set_problem(ea, eb, p);  pm.compute_viterbi(bm);

        const double d = ptr.score() - pm.score();     // >= 0 when pmax is merely suboptimal
        if (d > worst_below) worst_below = d;
        if (-d > worst_above) worst_above = -d;
    }
    INFO("worst shortfall below the optimum = " << worst_below
         << ", worst excess above it = " << worst_above);
    REQUIRE(worst_above <= 1e-9);      // never better than optimal: that would be a bug
    REQUIRE(worst_below <= 1e-6);      // measured 0 on this fixture; a real regression trips this
}
