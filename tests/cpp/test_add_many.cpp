// Tests for SeqPairBatch::add_many() — bulk, parallel, C++-owned construction.
//
// The Python tests cover the API contract.  These exist for what only the
// sanitizer build can see: add_many() constructs on N threads, takes ownership,
// and can throw out of a worker mid-construction.  Under ASan+UBSan a leaked
// staged pair, a double free, or a use-after-free on a partially built batch is
// a hard failure here rather than a mystery in someone's training run.

#include "catch.hpp"
#include "align_params.hpp"
#include "aligner.hpp"
#include "seq_pair.hpp"
#include "seq_pair_batch.hpp"

#include <array>
#include <string>
#include <string_view>
#include <vector>

// Match = +2, mismatch = -1, over the canonical AA alphabet.
static AlignParams aa_params() {
    std::array<double, 400> src{};
    for (int i = 0; i < 20; ++i)
        for (int j = 0; j < 20; ++j)
            src[i * 20 + j] = (i == j) ? 2.0 : -1.0;
    AlignParams p(SubstMatrix(src.data()));
    p.gap_open_a = 2.0; p.gap_extend_a = 1.0;
    p.gap_open_b = 2.0; p.gap_extend_b = 1.0;
    return p;
}

static AlignParams dna_params() {
    std::array<double, 16> src{};
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            src[i * 4 + j] = (i == j) ? 2.0 : -1.0;
    AlignParams p(SubstMatrix(src.data(), Alphabet::get("ACGT")));
    p.gap_open_a = 2.0; p.gap_extend_a = 1.0;
    p.gap_open_b = 2.0; p.gap_extend_b = 1.0;
    return p;
}

static const std::vector<std::string> AA_A = {
    "WWKKLLMMFFAAGG", "MKLPQR", "ACDEFG", "K", "HHHHPPPP", "YYWWAA"};
static const std::vector<std::string> AA_B = {
    "WWKLLMMFFAACGG", "MKLPQ",  "ACDEF",  "KKK", "HHPPP",  "YWWAAC"};

static std::vector<std::string_view> views(const std::vector<std::string>& v) {
    return std::vector<std::string_view>(v.begin(), v.end());
}

// Sum of the per-pair matrix gradients, as a flat vector — a cheap fingerprint
// that two batches computed the same thing.
static std::vector<double> grad_fingerprint(SeqPairBatch& b) {
    AlignParams g = b.compute_grad();
    std::vector<double> out(400);
    g.matrix.to_array(out.data());
    out.push_back(g.gap_open_a);
    out.push_back(g.gap_extend_a);
    out.push_back(g.gap_open_b);
    out.push_back(g.gap_extend_b);
    return out;
}

// ── add_many() is indistinguishable from adding pairs by hand ────────────────

TEST_CASE("add_many: matches a hand-built batch", "[add_many]") {
    auto p = aa_params();

    // Hand-built: the pairs are borrowed, so they must outlive the batch.
    std::vector<SeqPair> hand;
    hand.reserve(AA_A.size());
    SeqPairBatch ref(1);
    for (size_t i = 0; i < AA_A.size(); ++i)
        hand.emplace_back(AA_A[i], AA_B[i], p, GapModel::Affine,
                          AlignMode::Local, GradMode::Hard);
    for (auto& sp : hand) ref.add(&sp);

    SeqPairBatch bulk(4);
    bulk.add_many(views(AA_A), views(AA_B), p, GapModel::Affine,
                  AlignMode::Local, GradMode::Hard);

    REQUIRE(bulk.size() == ref.size());
    double ref_sum  = ref.score_and_grad();
    double bulk_sum = bulk.score_and_grad();
    REQUIRE(bulk_sum == Approx(ref_sum));

    for (size_t i = 0; i < AA_A.size(); ++i) {
        REQUIRE(bulk[i].score() == Approx(ref[i].score()));
        REQUIRE(bulk[i].seq_a() == AA_A[i]);
        REQUIRE(bulk[i].seq_b() == AA_B[i]);
    }
    REQUIRE(grad_fingerprint(bulk) == grad_fingerprint(ref));
}

TEST_CASE("add_many: parallel construction preserves order and content", "[add_many]") {
    auto p = aa_params();
    std::vector<std::string> a, b;
    for (int rep = 0; rep < 200; ++rep)
        for (size_t i = 0; i < AA_A.size(); ++i) {
            a.push_back(AA_A[i]);
            b.push_back(AA_B[i]);
        }

    SeqPairBatch one(1), many(8);
    one.add_many(views(a), views(b), p, GapModel::Affine, AlignMode::Local, GradMode::Hard);
    many.add_many(views(a), views(b), p, GapModel::Affine, AlignMode::Local, GradMode::Hard);

    REQUIRE(one.score_and_grad() == Approx(many.score_and_grad()));
    for (size_t i = 0; i < a.size(); ++i) {
        REQUIRE(many[i].seq_a() == a[i]);          // slot i really is pair i
        REQUIRE(many[i].score() == Approx(one[i].score()));
    }
    REQUIRE(grad_fingerprint(one) == grad_fingerprint(many));
}

TEST_CASE("add_many: owned pairs support the full SeqPair lifecycle", "[add_many]") {
    auto p = aa_params();
    SeqPairBatch batch(4);
    batch.add_many(views(AA_A), views(AA_B), p, GapModel::Affine,
                   AlignMode::Local, GradMode::Hard);

    batch.alloc_dp();
    batch.align_full();
    for (size_t i = 0; i < batch.size(); ++i) {
        REQUIRE(batch[i].path_valid());
        REQUIRE(batch[i].dp_valid());
        auto [aa, bb] = batch[i].aligned();
        REQUIRE(aa.size() == bb.size());
    }
    batch.compute_grad();
    batch.realign_banded(8);
    batch.drop_dp();
    for (size_t i = 0; i < batch.size(); ++i)
        REQUIRE_FALSE(batch[i].dp_valid());
}

TEST_CASE("add_many: mixes with add()", "[add_many]") {
    auto p = aa_params();
    SeqPair hand(AA_A[0], AA_B[0], p, GapModel::Affine, AlignMode::Local, GradMode::Hard);

    SeqPairBatch batch(2);
    batch.add(&hand);
    batch.add_many(views({AA_A.begin() + 1, AA_A.end()}),
                   views({AA_B.begin() + 1, AA_B.end()}),
                   p, GapModel::Affine, AlignMode::Local, GradMode::Hard);
    REQUIRE(batch.size() == AA_A.size());
    batch.score_and_grad();
    REQUIRE(batch[0].seq_a() == AA_A[0]);
    REQUIRE(batch[1].seq_a() == AA_A[1]);
}

// ── failures leave the batch exactly as it was, and leak nothing ─────────────

TEST_CASE("add_many: length mismatch throws, batch unchanged", "[add_many]") {
    auto p = aa_params();
    SeqPairBatch batch(2);
    std::vector<std::string> short_b(AA_B.begin(), AA_B.end() - 1);
    REQUIRE_THROWS_AS(
        batch.add_many(views(AA_A), views(short_b), p, GapModel::Affine,
                       AlignMode::Local, GradMode::Hard),
        std::invalid_argument);
    REQUIRE(batch.size() == 0);
}

TEST_CASE("add_many: bad character throws from a worker, batch unchanged", "[add_many]") {
    auto p = dna_params();
    SeqPairBatch batch(4);

    std::vector<std::string> good_a = {"ACGT", "ACGTACGT", "GGGG"};
    std::vector<std::string> good_b = {"ACGT", "ACGTTCGT", "GGAG"};
    batch.add_many(views(good_a), views(good_b), p, GapModel::Affine,
                   AlignMode::Local, GradMode::Hard);
    REQUIRE(batch.size() == 3);

    // Enough pairs that several threads are mid-construction when one throws:
    // the staged pairs built by the others must all be destroyed, not leaked.
    std::vector<std::string> bad_a, bad_b;
    for (int i = 0; i < 500; ++i) { bad_a.push_back("ACGTACGT"); bad_b.push_back("ACGTACGT"); }
    bad_a[250] = "ACGZ";                      // 'Z' is not in ACGT
    REQUIRE_THROWS_AS(
        batch.add_many(views(bad_a), views(bad_b), p, GapModel::Affine,
                       AlignMode::Local, GradMode::Hard),
        std::invalid_argument);

    REQUIRE(batch.size() == 3);               // the failed call added nothing
    batch.score_and_grad();                   // and the survivors still align
    REQUIRE(batch[0].score_valid());
}

TEST_CASE("add_many: alphabet mismatch throws, batch unchanged", "[add_many]") {
    auto aa = aa_params();
    auto dna = dna_params();
    SeqPairBatch batch(2);
    batch.add_many(views(AA_A), views(AA_B), aa, GapModel::Affine,
                   AlignMode::Local, GradMode::Hard);

    std::vector<std::string> da = {"ACGT"}, db = {"ACGT"};
    REQUIRE_THROWS_AS(
        batch.add_many(views(da), views(db), dna, GapModel::Affine,
                       AlignMode::Local, GradMode::Hard),
        std::invalid_argument);
    REQUIRE(batch.size() == AA_A.size());
    batch.score_and_grad();
}

TEST_CASE("add_many: empty is a no-op", "[add_many]") {
    auto p = aa_params();
    SeqPairBatch batch(2);
    batch.add_many({}, {}, p, GapModel::Affine, AlignMode::Local, GradMode::Hard);
    REQUIRE(batch.size() == 0);
    REQUIRE_THROWS(batch.compute_grad());     // empty batch has no alphabet
}
