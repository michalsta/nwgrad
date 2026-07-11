// SeqPair and SeqPairBatch.
//
// These two headers were never instantiated by the C++ test binary — they were
// only ever reached through the Python bindings.  Anyone using nwgrad as a
// header-only library hits this code directly, so it needs coverage here:
// the state machine (alloc_dp / align_full / drop_dp and the *_valid flags),
// all four GapModel x AlignMode variants of the std::variant type erasure,
// every GradMode, banded re-alignment, and the threaded batch.
//
// Gap penalties are deliberately asymmetric throughout (gap_a != gap_b) so that
// a crossed or dropped parameter set shows up.

#include "catch.hpp"
#include "align_params.hpp"
#include "aligner.hpp"
#include "seq_pair.hpp"
#include "seq_pair_batch.hpp"

#include <array>
#include <string>
#include <vector>

// Match = +2, mismatch = -1, over the canonical AA alphabet.
static AlignParams asym_params(double open_a, double ext_a,
                               double open_b, double ext_b) {
    std::array<double, 400> src{};
    for (int i = 0; i < 20; ++i)
        for (int j = 0; j < 20; ++j)
            src[i * 20 + j] = (i == j) ? 2.0 : -1.0;

    AlignParams p;
    p.matrix       = SubstMatrix(src.data());
    p.gap_open_a   = open_a;
    p.gap_extend_a = ext_a;
    p.gap_open_b   = open_b;
    p.gap_extend_b = ext_b;
    return p;
}

// Reference score straight from the Aligner, bypassing SeqPair entirely.
template<GapModel GM, AlignMode AM>
static double aligner_score(const AlignParams& p, const std::string& a,
                            const std::string& b) {
    Aligner<GM, AM> al;
    al.alloc_buf();
    al.set_problem(a.c_str(), b.c_str(), p);
    al.compute_viterbi();
    return al.score();
}

static const std::string A = "WWKKLLMMFFAAGG";
static const std::string B = "WWKLLMMFFAACGG";

// ── SeqPair: agreement with the raw Aligner across all four variants ─────────

TEST_CASE("SeqPair: linear/global score matches Aligner", "[seq_pair]") {
    auto p = asym_params(0.0, 1.5, 0.0, 0.5);   // asymmetric extend
    SeqPair sp(A, B, p, GapModel::Linear, AlignMode::Global, GradMode::None);
    sp.alloc_dp();
    sp.align_full();
    REQUIRE(sp.score() ==
            Approx(aligner_score<GapModel::Linear, AlignMode::Global>(p, A, B)));
}

TEST_CASE("SeqPair: linear/local score matches Aligner", "[seq_pair]") {
    auto p = asym_params(0.0, 1.5, 0.0, 0.5);
    SeqPair sp(A, B, p, GapModel::Linear, AlignMode::Local, GradMode::None);
    sp.alloc_dp();
    sp.align_full();
    REQUIRE(sp.score() ==
            Approx(aligner_score<GapModel::Linear, AlignMode::Local>(p, A, B)));
}

TEST_CASE("SeqPair: affine/global score matches Aligner", "[seq_pair]") {
    auto p = asym_params(4.0, 0.5, 1.5, 2.0);   // asymmetric open and extend
    SeqPair sp(A, B, p, GapModel::Affine, AlignMode::Global, GradMode::None);
    sp.alloc_dp();
    sp.align_full();
    REQUIRE(sp.score() ==
            Approx(aligner_score<GapModel::Affine, AlignMode::Global>(p, A, B)));
}

TEST_CASE("SeqPair: affine/local score matches Aligner", "[seq_pair]") {
    auto p = asym_params(4.0, 0.5, 1.5, 2.0);
    SeqPair sp(A, B, p, GapModel::Affine, AlignMode::Local, GradMode::None);
    sp.alloc_dp();
    sp.align_full();
    REQUIRE(sp.score() ==
            Approx(aligner_score<GapModel::Affine, AlignMode::Local>(p, A, B)));
}

// ── SeqPair: state machine ───────────────────────────────────────────────────

TEST_CASE("SeqPair: accessors throw before anything is computed", "[seq_pair]") {
    auto p = asym_params(4.0, 0.5, 1.5, 2.0);
    SeqPair sp(A, B, p, GapModel::Affine, AlignMode::Global, GradMode::Hard);

    REQUIRE_FALSE(sp.score_valid());
    REQUIRE_FALSE(sp.grad_valid());
    REQUIRE_FALSE(sp.path_valid());
    REQUIRE_FALSE(sp.dp_valid());

    REQUIRE_THROWS_AS(sp.score(), std::logic_error);
    REQUIRE_THROWS_AS(sp.grad(),  std::logic_error);
}

TEST_CASE("SeqPair: align_full without alloc_dp throws", "[seq_pair]") {
    auto p = asym_params(4.0, 0.5, 1.5, 2.0);
    SeqPair sp(A, B, p, GapModel::Affine, AlignMode::Global, GradMode::Hard);
    REQUIRE_THROWS(sp.align_full());
}

TEST_CASE("SeqPair: validity flags follow the lifecycle", "[seq_pair]") {
    auto p = asym_params(4.0, 0.5, 1.5, 2.0);
    SeqPair sp(A, B, p, GapModel::Affine, AlignMode::Global, GradMode::Hard);

    sp.alloc_dp();
    sp.align_full();
    REQUIRE(sp.score_valid());
    REQUIRE(sp.path_valid());
    REQUIRE(sp.dp_valid());
    REQUIRE_FALSE(sp.grad_valid());     // gradient is lazy

    sp.compute_grad();
    REQUIRE(sp.grad_valid());

    sp.drop_dp();
    REQUIRE_FALSE(sp.dp_valid());
    REQUIRE(sp.score_valid());          // the score survives dropping the tables
    REQUIRE(sp.grad_valid());
}

TEST_CASE("SeqPair: set_params invalidates the cached results", "[seq_pair]") {
    auto p = asym_params(4.0, 0.5, 1.5, 2.0);
    SeqPair sp(A, B, p, GapModel::Affine, AlignMode::Global, GradMode::Hard);
    sp.alloc_dp();
    sp.align_full();
    sp.compute_grad();
    const double first = sp.score();

    auto p2 = asym_params(1.0, 0.25, 1.0, 0.25);   // cheaper gaps
    sp.set_params(p2);
    REQUIRE_FALSE(sp.score_valid());
    REQUIRE_FALSE(sp.grad_valid());

    sp.align_full();
    REQUIRE(sp.score() ==
            Approx(aligner_score<GapModel::Affine, AlignMode::Global>(p2, A, B)));
    REQUIRE(sp.score() != Approx(first));           // the new params actually bit
}

TEST_CASE("SeqPair: score_and_grad matches align_full + compute_grad", "[seq_pair]") {
    auto p = asym_params(4.0, 0.5, 1.5, 2.0);

    SeqPair a(A, B, p, GapModel::Affine, AlignMode::Global, GradMode::Hard);
    a.alloc_dp();
    a.align_full();
    a.compute_grad();

    SeqPair b(A, B, p, GapModel::Affine, AlignMode::Global, GradMode::Hard);
    auto [score, grad] = b.score_and_grad();

    REQUIRE(score == Approx(a.score()));
    REQUIRE(grad.gap_open_a   == Approx(a.grad().gap_open_a));
    REQUIRE(grad.gap_extend_a == Approx(a.grad().gap_extend_a));
    REQUIRE(grad.gap_open_b   == Approx(a.grad().gap_open_b));
    REQUIRE(grad.gap_extend_b == Approx(a.grad().gap_extend_b));
    for (int i = 0; i < 256; ++i)
        for (int j = 0; j < 256; ++j)
            REQUIRE(grad.matrix.mat[i][j] == Approx(a.grad().matrix.mat[i][j]));
}

TEST_CASE("SeqPair: aligned() strings reconstruct the sequences", "[seq_pair]") {
    auto p = asym_params(4.0, 0.5, 1.5, 2.0);
    SeqPair sp(A, B, p, GapModel::Affine, AlignMode::Global, GradMode::None);
    sp.alloc_dp();
    sp.align_full();

    auto [ga, gb] = sp.aligned();
    REQUIRE(ga.size() == gb.size());

    std::string ungapped_a, ungapped_b;
    for (char c : ga) if (c != '-') ungapped_a.push_back(c);
    for (char c : gb) if (c != '-') ungapped_b.push_back(c);
    REQUIRE(ungapped_a == A);   // global alignment spans both sequences
    REQUIRE(ungapped_b == B);
}

TEST_CASE("SeqPair: GradMode::Soft scores the log-partition function", "[seq_pair]") {
    auto p = asym_params(4.0, 0.5, 1.5, 2.0);

    SeqPair soft(A, B, p, GapModel::Affine, AlignMode::Global, GradMode::Soft);
    soft.alloc_dp();
    soft.align_full();

    // log Z sums over every alignment, so it is >= the single best score.
    const double viterbi =
        aligner_score<GapModel::Affine, AlignMode::Global>(p, A, B);
    REQUIRE(soft.score() >= Approx(viterbi));

    soft.compute_grad();
    REQUIRE(soft.grad_valid());
    // Gap derivatives are negated expected counts: raising a penalty lowers the
    // score, so d(log Z)/d(gap penalty) can never be positive.
    REQUIRE(soft.grad().gap_extend_a <= 0.0);
    REQUIRE(soft.grad().gap_extend_b <= 0.0);
}

TEST_CASE("SeqPair: GradMode::None still scores but refuses a gradient", "[seq_pair]") {
    auto p = asym_params(4.0, 0.5, 1.5, 2.0);
    SeqPair sp(A, B, p, GapModel::Affine, AlignMode::Global, GradMode::None);
    sp.alloc_dp();
    sp.align_full();

    REQUIRE(sp.score() ==
            Approx(aligner_score<GapModel::Affine, AlignMode::Global>(p, A, B)));
    REQUIRE_THROWS_AS(sp.grad(), std::logic_error);
}

TEST_CASE("SeqPair: a wide band reproduces the full-DP score", "[seq_pair][banded]") {
    auto p = asym_params(4.0, 0.5, 1.5, 2.0);
    SeqPair sp(A, B, p, GapModel::Affine, AlignMode::Global, GradMode::Hard);
    sp.alloc_dp();
    sp.align_full();
    const double full = sp.score();

    // A band wider than both sequences cannot exclude any cell, so the banded
    // re-alignment must recover exactly the full-DP optimum.
    sp.realign_banded(static_cast<int>(A.size() + B.size()));
    REQUIRE(sp.score() == Approx(full));
}

TEST_CASE("SeqPair: realign_banded picks up new params", "[seq_pair][banded]") {
    auto p = asym_params(4.0, 0.5, 1.5, 2.0);
    SeqPair sp(A, B, p, GapModel::Affine, AlignMode::Global, GradMode::Hard);
    sp.alloc_dp();
    sp.align_full();

    auto p2 = asym_params(0.5, 0.25, 0.5, 0.25);
    sp.set_params(p2);
    sp.realign_banded(static_cast<int>(A.size() + B.size()));

    REQUIRE(sp.score() ==
            Approx(aligner_score<GapModel::Affine, AlignMode::Global>(p2, A, B)));
}

// ── SeqPairBatch ─────────────────────────────────────────────────────────────

namespace {

struct Corpus {
    std::vector<std::string> as{"WWKKLLMMFF", "AAGGCC", "MMFFAAGGCCWW", "KK"};
    std::vector<std::string> bs{"WWKLLMMFFA", "AGGC",   "MFFAAGGCWW",   "KKLL"};
};

}  // namespace

TEST_CASE("SeqPairBatch: align_full matches per-pair alignment", "[seq_pair_batch]") {
    auto p = asym_params(4.0, 0.5, 1.5, 2.0);
    Corpus c;

    std::vector<SeqPair> pairs;
    pairs.reserve(c.as.size());
    for (size_t i = 0; i < c.as.size(); ++i)
        pairs.emplace_back(c.as[i], c.bs[i], p,
                           GapModel::Affine, AlignMode::Global, GradMode::Hard);

    SeqPairBatch batch(2);
    for (auto& sp : pairs) batch.add(&sp);
    REQUIRE(batch.size() == c.as.size());

    batch.alloc_dp();
    const double total = batch.align_full();

    double expected = 0.0;
    for (size_t i = 0; i < c.as.size(); ++i)
        expected += aligner_score<GapModel::Affine, AlignMode::Global>(
            p, c.as[i], c.bs[i]);

    REQUIRE(total == Approx(expected));
    for (auto& sp : pairs) REQUIRE(sp.score_valid());
}

TEST_CASE("SeqPairBatch: compute_grad sums the per-pair gradients",
          "[seq_pair_batch]") {
    auto p = asym_params(4.0, 0.5, 1.5, 2.0);
    Corpus c;

    std::vector<SeqPair> pairs;
    pairs.reserve(c.as.size());
    for (size_t i = 0; i < c.as.size(); ++i)
        pairs.emplace_back(c.as[i], c.bs[i], p,
                           GapModel::Affine, AlignMode::Global, GradMode::Hard);

    SeqPairBatch batch(3);
    for (auto& sp : pairs) batch.add(&sp);
    batch.alloc_dp();
    batch.align_full();
    const AlignParams total = batch.compute_grad();

    AlignParams expected;
    for (size_t i = 0; i < c.as.size(); ++i) {
        SeqPair one(c.as[i], c.bs[i], p,
                    GapModel::Affine, AlignMode::Global, GradMode::Hard);
        expected += one.score_and_grad().second;
    }

    REQUIRE(total.gap_open_a   == Approx(expected.gap_open_a));
    REQUIRE(total.gap_extend_a == Approx(expected.gap_extend_a));
    REQUIRE(total.gap_open_b   == Approx(expected.gap_open_b));
    REQUIRE(total.gap_extend_b == Approx(expected.gap_extend_b));
    for (int i = 0; i < 256; ++i)
        for (int j = 0; j < 256; ++j)
            REQUIRE(total.matrix.mat[i][j] == Approx(expected.matrix.mat[i][j]));
}

TEST_CASE("SeqPairBatch: score_and_grad matches align_full + compute_grad",
          "[seq_pair_batch]") {
    auto p = asym_params(4.0, 0.5, 1.5, 2.0);
    Corpus c;

    auto build = [&](std::vector<SeqPair>& out) {
        out.reserve(c.as.size());
        for (size_t i = 0; i < c.as.size(); ++i)
            out.emplace_back(c.as[i], c.bs[i], p,
                             GapModel::Affine, AlignMode::Global, GradMode::Hard);
    };

    std::vector<SeqPair> stepwise, fused;
    build(stepwise);
    build(fused);

    SeqPairBatch b1(2);
    for (auto& sp : stepwise) b1.add(&sp);
    b1.alloc_dp();
    const double total_stepwise = b1.align_full();
    const AlignParams grad_stepwise = b1.compute_grad();

    SeqPairBatch b2(2);
    for (auto& sp : fused) b2.add(&sp);
    const double total_fused = b2.score_and_grad();
    const AlignParams grad_fused = b2.compute_grad();

    REQUIRE(total_fused == Approx(total_stepwise));
    REQUIRE(grad_fused.gap_open_b   == Approx(grad_stepwise.gap_open_b));
    REQUIRE(grad_fused.gap_extend_b == Approx(grad_stepwise.gap_extend_b));
    for (int i = 0; i < 256; ++i)
        for (int j = 0; j < 256; ++j)
            REQUIRE(grad_fused.matrix.mat[i][j] ==
                    Approx(grad_stepwise.matrix.mat[i][j]));
}

TEST_CASE("SeqPairBatch: set_params then realign_banded", "[seq_pair_batch][banded]") {
    auto p = asym_params(4.0, 0.5, 1.5, 2.0);
    Corpus c;

    std::vector<SeqPair> pairs;
    pairs.reserve(c.as.size());
    for (size_t i = 0; i < c.as.size(); ++i)
        pairs.emplace_back(c.as[i], c.bs[i], p,
                           GapModel::Affine, AlignMode::Global, GradMode::Hard);

    SeqPairBatch batch(2);
    for (auto& sp : pairs) batch.add(&sp);
    batch.alloc_dp();
    batch.align_full();

    auto p2 = asym_params(0.5, 0.25, 0.5, 0.25);
    batch.set_params(p2);
    const double total = batch.realign_banded(64);   // band wider than any pair

    double expected = 0.0;
    for (size_t i = 0; i < c.as.size(); ++i)
        expected += aligner_score<GapModel::Affine, AlignMode::Global>(
            p2, c.as[i], c.bs[i]);
    REQUIRE(total == Approx(expected));

    batch.drop_dp();
    for (auto& sp : pairs) REQUIRE_FALSE(sp.dp_valid());
}

TEST_CASE("SeqPairBatch: an empty batch is harmless", "[seq_pair_batch]") {
    SeqPairBatch batch(2);
    REQUIRE(batch.size() == 0);
    batch.alloc_dp();
    REQUIRE(batch.align_full() == Approx(0.0));

    const AlignParams grad = batch.compute_grad();
    REQUIRE(grad.gap_open_a == Approx(0.0));
    REQUIRE(grad.gap_extend_b == Approx(0.0));
}

TEST_CASE("SeqPairBatch: single-threaded and multi-threaded agree",
          "[seq_pair_batch][threads]") {
    auto p = asym_params(4.0, 0.5, 1.5, 2.0);
    Corpus c;

    auto run = [&](int threads) {
        std::vector<SeqPair> pairs;
        pairs.reserve(c.as.size());
        for (size_t i = 0; i < c.as.size(); ++i)
            pairs.emplace_back(c.as[i], c.bs[i], p,
                               GapModel::Affine, AlignMode::Global, GradMode::Hard);

        SeqPairBatch batch(threads);
        for (auto& sp : pairs) batch.add(&sp);
        const double total = batch.score_and_grad();
        return std::make_pair(total, batch.compute_grad());
    };

    auto [total1, grad1] = run(1);
    auto [total8, grad8] = run(8);

    REQUIRE(total8 == Approx(total1));
    REQUIRE(grad8.gap_open_a   == Approx(grad1.gap_open_a));
    REQUIRE(grad8.gap_extend_b == Approx(grad1.gap_extend_b));
    for (int i = 0; i < 256; ++i)
        for (int j = 0; j < 256; ++j)
            REQUIRE(grad8.matrix.mat[i][j] == Approx(grad1.matrix.mat[i][j]));
}
