// ── BatchEngine against the classes it replaces ─────────────────────────────
//
// BatchEngine<T, GM, AM> keeps every pair's state in flat arrays and runs each pair
// through a per-worker Aligner; SeqPairBatchT<T> held one SeqPair object (two Aligners)
// per pair.  The engine's per-pair operations are ports of SeqPair's, so every result
// must be BIT-identical: scores, gradients, guides and validity flags after
// score_and_grad(), after set_params() + banded_grad(), the stored paths against
// align_full() + aligned() / coordinates(), and weighted_grad().  Hard compute_grad()
// is integer counts, so exact too; the soft one is compared with a tolerance, because
// the old one merged per-thread sums in completion order (not reproducible), while the
// engine sums in fixed blocks.
//
// Random configurations cover both precisions, the four problem types, the three grad
// modes, every fill and scheduler, every traceback, soft impl / temperature / guide,
// small Hirschberg cutoffs (so short pairs split), two parameter segments, and empty
// and one-residue sequences.

#include "catch.hpp"

#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "batch_engine.hpp"
#include "batch.hpp"
#include "seq_pair_batch.hpp"

namespace {

struct Cfg {
    GradMode gd;
    TracebackMode tb;
    bool inter, rowwise, sorted;
    int soft_guide;        // 0 eager, 1 lazy, 2 posterior
    SoftImpl soft_impl;
    double temp;
    int hb_cutoff;
    int threads;
    bool protein;
    bool two_segments, scalar_second;
    int bw;
};

std::string rseq(std::mt19937& rng, const std::string& al, int len) {
    std::string s(static_cast<size_t>(len), al[0]);
    for (auto& c : s) c = al[rng() % al.size()];
    return s;
}
int rlen(std::mt19937& rng) {
    const unsigned r = rng() % 100;
    if (r < 4) return 0;
    if (r < 8) return 1;
    if (r < 90) return 2 + static_cast<int>(rng() % 50);
    return 60 + static_cast<int>(rng() % 120);
}

AlignParams rparams(std::mt19937& rng, const Alphabet& alpha, GapModel gm, int kind) {
    const int n = alpha.size();
    std::vector<double> m(static_cast<size_t>(n) * n);
    std::normal_distribution<double> nd(0.0, 1.0);
    for (int a = 0; a < n; ++a)
        for (int b = 0; b < n; ++b)
            m[static_cast<size_t>(a) * n + b] =
                kind == 0 ? (a == b ? 2.0 : -1.0)                       // ties
              : kind == 1 ? static_cast<double>(static_cast<int>(rng() % 9) - 3)
                          : nd(rng);
    const double go = gm == GapModel::Linear ? 0.0 : (kind == 0 ? 3.0 : 2.5);
    const double ge = kind == 2 ? 0.1 : 0.7;
    return AlignParams(SubstMatrix(m.data(), alpha), go, ge, go * 1.3, ge * 0.8);
}

bool same(double x, double y) { return x == y || (std::isnan(x) && std::isnan(y)); }

template <class T, GapModel GM, AlignMode AM>
void run_case(unsigned seed, const Cfg& c) {
    std::mt19937 rng(seed);
    const Alphabet& alpha = Alphabet::get(c.protein ? "ACDEFGHIKLMNPQRSTVWY" : "ACGT");
    const int kind = static_cast<int>(rng() % 3);
    const AlignParams p1 = rparams(rng, alpha, GM, kind);
    const AlignParams p1b = rparams(rng, alpha, GM, (kind + 1) % 3);
    const AlignParams p2 = rparams(rng, alpha, GM, (kind + 2) % 3);

    const int N1 = 1 + static_cast<int>(rng() % 30), N2 = c.two_segments ? 1 + static_cast<int>(rng() % 20) : 0;
    std::vector<std::string> A, B;
    const int lb = rlen(rng);
    for (int k = 0; k < N1 + N2; ++k) {
        A.push_back(rseq(rng, alpha.symbols(), rlen(rng)));
        B.push_back(rseq(rng, alpha.symbols(), rng() % 4 ? lb : rlen(rng)));
    }
    std::vector<std::string_view> A1(A.begin(), A.begin() + N1), B1(B.begin(), B.begin() + N1);
    std::vector<std::string_view> A2(A.begin() + N1, A.end()), B2(B.begin() + N1, B.end());
    const int k2 = c.scalar_second ? kBackendScalar : kBackendAuto;

    SeqPairBatchT<T> old(c.threads, c.tb);
    old.inter_fill = c.inter; old.rowwise_full = c.rowwise; old.sorted_schedule = c.sorted;
    old.soft_impl = c.soft_impl; old.soft_temperature = c.temp;
    old.soft_guide_lazy = c.soft_guide == 1; old.soft_guide_posterior = c.soft_guide == 2;
    old.hb_cutoff = c.hb_cutoff;
    old.add_many(A1, B1, p1, GM, AM, c.gd);
    if (N2) old.add_many(A2, B2, p1b, GM, AM, c.gd, k2);

    BatchEngine<T, GM, AM> eng(c.threads, c.gd, c.tb);
    eng.inter_fill = c.inter; eng.rowwise_full = c.rowwise; eng.sorted_schedule = c.sorted;
    eng.set_soft_impl(c.soft_impl); eng.set_soft_temperature(c.temp);
    eng.set_soft_guide(c.soft_guide == 1, c.soft_guide == 2);
    eng.set_hb_cutoff(c.hb_cutoff);
    eng.add_many(A1, B1, p1);
    if (N2) eng.add_many(A2, B2, p1b, k2);

    const size_t N = old.size();
    REQUIRE(eng.size() == N);
    AlignParams go(alpha), ge(alpha);

    auto compare = [&](const char* stage) {
        INFO("stage " << stage);
        for (size_t i = 0; i < N; ++i) {
            INFO("pair " << i << " m=" << A[i].size() << " n=" << B[i].size());
            const auto& op = old[i];
            REQUIRE(op.path_valid()  == eng.path_valid(i));
            REQUIRE(op.score_valid() == eng.score_valid(i));
            REQUIRE(op.grad_valid()  == eng.grad_valid(i));
            REQUIRE(op.guide_pending() == eng.guide_pending(i));
            if (op.score_valid()) CHECK(same(op.score(), eng.score(i)));
            if (op.grad_valid()) {
                go = op.grad(); eng.grad(i, ge);
                bool eq = same(go.gap_open_a, ge.gap_open_a) && same(go.gap_extend_a, ge.gap_extend_a) &&
                          same(go.gap_open_b, ge.gap_open_b) && same(go.gap_extend_b, ge.gap_extend_b);
                for (int k = 0; k < alpha.size() * alpha.size(); ++k)
                    eq = eq && same(go.matrix.data()[k], ge.matrix.data()[k]);
                CHECK(eq);
            }
            if (op.path_valid() && !op.guide_pending())
                CHECK(op.guide_j_raw() == eng.guide_raw(i));
        }
    };

    // 1. Full DP.
    const double so = old.score_and_grad(), se = eng.score_and_grad();
    CHECK(same(so, se));
    compare("score_and_grad");

    // 2. weighted_grad with arbitrary weights: same blocked arithmetic.
    if (c.gd != GradMode::None && N) {
        std::vector<double> w(N);
        for (auto& x : w) x = std::uniform_real_distribution<double>(-2, 2)(rng);
        const AlignParams wo = old.weighted_grad(w.data(), N), we = eng.weighted_grad(w.data(), N);
        bool eq = same(wo.gap_open_a, we.gap_open_a) && same(wo.gap_extend_b, we.gap_extend_b);
        for (int k = 0; k < alpha.size() * alpha.size(); ++k)
            eq = eq && same(wo.matrix.data()[k], we.matrix.data()[k]);
        CHECK(eq);
        const AlignParams co = old.compute_grad(), ce = eng.compute_grad();
        for (int k = 0; k < alpha.size() * alpha.size(); ++k) {
            if (c.gd == GradMode::Hard) CHECK(same(co.matrix.data()[k], ce.matrix.data()[k]));
            else CHECK(std::abs(co.matrix.data()[k] - ce.matrix.data()[k]) <=
                       1e-9 * (1.0 + std::abs(co.matrix.data()[k])));
        }
    }

    // 3. New params, banded step around the cached guides.
    old.set_params(p2); eng.set_params(p2);
    compare("set_params");
    const double bo = old.banded_grad(c.bw), be = eng.banded_grad(c.bw);
    CHECK(same(bo, be));
    compare("banded_grad");

    // 4. Stored paths against the old pair-owned tables.  align_full() ran every pair's
    // own fill; score_and_grad(keep_paths) honours fill="interpair", whose shared soft
    // pass agrees with the per-pair forward-backward within tolerance only (the soft
    // path's contract) — so soft scores here are compared with it.  Paths are Viterbi:
    // exact.
    auto close = [&](double x, double y) {
        if (c.gd != GradMode::Soft) return same(x, y);
        return std::abs(x - y) <= 1e-11 * (1.0 + std::abs(x));
    };
    old.alloc_dp();
    const double ao = old.align_full(), ae = eng.score_and_grad(/*keep_paths=*/true);
    CHECK(close(ao, ae));
    for (size_t i = 0; i < N; ++i) {
        INFO("aligned, pair " << i);
        CHECK(close(old[i].score(), eng.score(i)));
        CHECK(old[i].aligned() == eng.aligned(i));
        CHECK(old[i].coordinates() == eng.coordinates(i));
    }
    // ... and the banded realignment's.
    const double ro = old.realign_banded(c.bw), re = eng.banded_grad(c.bw, /*keep_paths=*/true);
    CHECK(same(ro, re));
    for (size_t i = 0; i < N; ++i) {
        INFO("realigned, pair " << i);
        CHECK(same(old[i].score(), eng.score(i)));
        CHECK(old[i].aligned() == eng.aligned(i));
    }
}

template <class T, GapModel GM, AlignMode AM>
void sweep(int n) {
    std::mt19937 rng(1000u + static_cast<unsigned>(GM) * 10u + static_cast<unsigned>(AM) +
                     (std::is_same_v<T, float> ? 100u : 0u));
    const GradMode gds[] = {GradMode::Hard, GradMode::Soft, GradMode::None};
    const TracebackMode tbs[] = {TracebackMode::Default, TracebackMode::Pointers,
                                 TracebackMode::Scores, TracebackMode::Hirschberg};
    const SoftImpl sis[] = {SoftImpl::Scaled, SoftImpl::ScaledOrLog, SoftImpl::Log};
    for (int k = 0; k < n; ++k) {
        Cfg c;
        c.gd = gds[k % 3];
        c.tb = tbs[rng() % 4];
        c.inter = rng() % 3 != 0;
        c.rowwise = rng() % 3 == 0;
        c.sorted = rng() % 3 == 0;
        c.soft_guide = static_cast<int>(rng() % 3);
        c.soft_impl = sis[rng() % 3];
        c.temp = rng() % 3 == 0 ? 0.5 : 1.0;
        c.hb_cutoff = rng() % 2 ? 512 : 1 + static_cast<int>(rng() % 24);
        c.threads = 1 + static_cast<int>(rng() % 4);
        c.protein = rng() % 4 == 0;
        c.two_segments = rng() % 2;
        c.scalar_second = rng() % 2;
        c.bw = 1 + static_cast<int>(rng() % 8);
        const unsigned seed = static_cast<unsigned>(rng());
        INFO("case " << k << " seed " << seed << " gd " << static_cast<int>(c.gd) << " tb "
             << static_cast<int>(c.tb) << " inter " << c.inter << " rowwise " << c.rowwise
             << " sorted " << c.sorted << " guide " << c.soft_guide << " impl "
             << static_cast<int>(c.soft_impl) << " T " << c.temp << " cutoff " << c.hb_cutoff
             << " threads " << c.threads << " protein " << c.protein << " segs "
             << c.two_segments << " bw " << c.bw);
        run_case<T, GM, AM>(seed, c);
    }
}

} // namespace

TEST_CASE("BatchEngine == SeqPairBatch, double", "[batch_engine]") {
    sweep<double, GapModel::Affine, AlignMode::Global>(24);
    sweep<double, GapModel::Affine, AlignMode::Local>(24);
    sweep<double, GapModel::Linear, AlignMode::Global>(24);
    sweep<double, GapModel::Linear, AlignMode::Local>(24);
}

TEST_CASE("BatchEngine == SeqPairBatch, float32", "[batch_engine]") {
    sweep<float, GapModel::Affine, AlignMode::Global>(24);
    sweep<float, GapModel::Affine, AlignMode::Local>(24);
    sweep<float, GapModel::Linear, AlignMode::Global>(24);
    sweep<float, GapModel::Linear, AlignMode::Local>(24);
}

// ── align_stream against BatchAligner ───────────────────────────────────────

namespace {

template <class T, GapModel GM, AlignMode AM>
void stream_sweep(int n) {
    std::mt19937 rng(77u + static_cast<unsigned>(GM) * 10u + static_cast<unsigned>(AM) +
                     (std::is_same_v<T, float> ? 100u : 0u));
    using BA = BatchAlignerT<T>;
    const GradMode gds[] = {GradMode::Hard, GradMode::Soft, GradMode::None};
    const typename BA::GradMode bgds[] = {BA::GradMode::Hard, BA::GradMode::Soft, BA::GradMode::None};
    for (int k = 0; k < n; ++k) {
        const int gi = k % 3;
        const bool protein = rng() % 4 == 0, inter = rng() % 3 != 0;
        const int band = rng() % 3 == 0 ? 1 + static_cast<int>(rng() % 6) : 0;
        const bool guides = rng() % 3 == 0;
        const int threads = 1 + static_cast<int>(rng() % 4);
        const double temp = rng() % 3 == 0 ? 0.5 : 1.0;
        const Alphabet& alpha = Alphabet::get(protein ? "ACDEFGHIKLMNPQRSTVWY" : "ACGT");
        const AlignParams p = rparams(rng, alpha, GM, static_cast<int>(rng() % 3));
        const int N = 1 + static_cast<int>(rng() % 40), lb = rlen(rng);
        std::vector<std::string> A, B;
        for (int i = 0; i < N; ++i) {
            A.push_back(rseq(rng, alpha.symbols(), rlen(rng)));
            B.push_back(rseq(rng, alpha.symbols(), rng() % 4 ? lb : rlen(rng)));
        }
        std::vector<ProblemInstance> probs;
        for (int i = 0; i < N; ++i) {
            ProblemInstance pi{A[i], B[i], {}};
            if (guides) {   // a diagonal guide, as a caller's aligned strings would give
                const int m = static_cast<int>(A[i].size()), nb = static_cast<int>(B[i].size());
                for (int r = 0; r <= m; ++r) pi.guide_j.push_back(m ? r * nb / m : 0);
                if (!pi.guide_j.empty()) pi.guide_j.back() = nb;
            }
            probs.push_back(pi);
        }
        INFO("stream case " << k << " gd " << gi << " band " << band << " guides " << guides
             << " inter " << inter << " threads " << threads << " protein " << protein);
        BA ba(p, band, GM, AM, bgds[gi], threads);
        ba.inter_fill = inter; ba.soft_temperature = temp;
        BatchEngine<T, GM, AM> eng(threads, gds[gi]);
        eng.inter_fill = inter; eng.set_soft_temperature(temp);
        const BatchResult ro = ba.align(probs);
        const BatchResult re = eng.align_stream(probs, p, band);
        REQUIRE(ro.scores.size() == re.scores.size());
        for (int i = 0; i < N; ++i) {
            INFO("problem " << i);
            if (gds[gi] == GradMode::Soft)
                CHECK(std::abs(ro.scores[i] - re.scores[i]) <= 1e-11 * (1 + std::abs(ro.scores[i])));
            else
                CHECK(same(ro.scores[i], re.scores[i]));
        }
        for (int q = 0; q < alpha.size() * alpha.size(); ++q) {
            const double x = ro.grad.matrix.data()[q], y = re.grad.matrix.data()[q];
            if (gds[gi] == GradMode::Soft) CHECK(std::abs(x - y) <= 1e-9 * (1 + std::abs(x)));
            else                           CHECK(same(x, y));
        }
    }
}

} // namespace

TEST_CASE("BatchEngine::align_stream == BatchAligner::align", "[batch_engine]") {
    stream_sweep<double, GapModel::Affine, AlignMode::Global>(30);
    stream_sweep<double, GapModel::Affine, AlignMode::Local>(30);
    stream_sweep<double, GapModel::Linear, AlignMode::Global>(30);
    stream_sweep<double, GapModel::Linear, AlignMode::Local>(30);
    stream_sweep<float, GapModel::Affine, AlignMode::Global>(30);
    stream_sweep<float, GapModel::Affine, AlignMode::Local>(30);
    stream_sweep<float, GapModel::Linear, AlignMode::Global>(30);
    stream_sweep<float, GapModel::Linear, AlignMode::Local>(30);
}
