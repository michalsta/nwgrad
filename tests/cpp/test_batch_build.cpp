// Tests for building a SeqPairBatch from its pairs — bulk, parallel, C++-owned
// construction, SeqPairBatch(seqs_a, seqs_b, params, ...).
//
// The Python tests cover the API contract.  These exist for what only the
// sanitizer build can see: construction runs on N threads, takes ownership,
// and can throw out of a worker mid-construction.  The deprecated add_many()
// (appending; removed in 0.8.0) is tested at the end.  Under ASan+UBSan a leaked
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

// ── a batch is indistinguishable from aligning the pairs one by one ──────────

static_assert(!std::is_constructible_v<SeqPairBatch, int>);
static_assert(!std::is_default_constructible_v<SeqPairBatch>);
static_assert(std::is_constructible_v<SeqPairBatch, GapModel, AlignMode, GradMode>);

TEST_CASE("batch build: matches standalone pairs", "[batch_build]") {
    auto p = aa_params();

    SeqPairBatch bulk(views(AA_A), views(AA_B), p,
                      GapModel::Affine, AlignMode::Local, GradMode::Hard, 4);
    REQUIRE(bulk.size() == AA_A.size());
    const double bulk_sum = bulk.score_and_grad();

    double ref_sum = 0.0;
    AlignParams ref_grad(p.matrix.alphabet());
    for (size_t i = 0; i < AA_A.size(); ++i) {
        SeqPair one(AA_A[i], AA_B[i], p, GapModel::Affine, AlignMode::Local, GradMode::Hard);
        const auto [s, g] = one.score_and_grad();
        ref_sum += s;
        ref_grad += g;
        REQUIRE(bulk[i].score() == s);
        REQUIRE(bulk[i].seq_a() == AA_A[i]);
        REQUIRE(bulk[i].seq_b() == AA_B[i]);
    }
    REQUIRE(bulk_sum == ref_sum);
    std::vector<double> ref(400);
    ref_grad.matrix.to_array(ref.data());
    ref.insert(ref.end(), {ref_grad.gap_open_a, ref_grad.gap_extend_a,
                           ref_grad.gap_open_b, ref_grad.gap_extend_b});
    REQUIRE(grad_fingerprint(bulk) == ref);   // integer path counts: exact in any order
}

TEST_CASE("batch build: parallel construction preserves order and content", "[batch_build]") {
    auto p = aa_params();
    std::vector<std::string> a, b;
    for (int rep = 0; rep < 200; ++rep)
        for (size_t i = 0; i < AA_A.size(); ++i) {
            a.push_back(AA_A[i]);
            b.push_back(AA_B[i]);
        }

    SeqPairBatch one(views(a), views(b), p, GapModel::Affine, AlignMode::Local, GradMode::Hard, 1);
    SeqPairBatch many(views(a), views(b), p, GapModel::Affine, AlignMode::Local, GradMode::Hard, 8);

    REQUIRE(one.score_and_grad() == Approx(many.score_and_grad()));
    for (size_t i = 0; i < a.size(); ++i) {
        REQUIRE(many[i].seq_a() == a[i]);          // slot i really is pair i
        REQUIRE(many[i].score() == Approx(one[i].score()));
    }
    REQUIRE(grad_fingerprint(one) == grad_fingerprint(many));
}

TEST_CASE("batch build: owned pairs support the full SeqPair lifecycle", "[batch_build]") {
    auto p = aa_params();
    SeqPairBatch batch(views(AA_A), views(AA_B), p,
                       GapModel::Affine, AlignMode::Local, GradMode::Hard, 4);

    batch.score_and_grad(/*keep_paths=*/true);
    for (size_t i = 0; i < batch.size(); ++i) {
        REQUIRE(batch[i].path_valid());
        REQUIRE(batch[i].stored_path_valid());
        auto [aa, bb] = batch[i].aligned();
        REQUIRE(aa.size() == bb.size());
    }
    batch.compute_grad();
    batch.banded_grad(8, /*keep_paths=*/true);
    batch.drop_paths();
    for (size_t i = 0; i < batch.size(); ++i)
        REQUIRE_FALSE(batch[i].stored_path_valid());
}

TEST_CASE("batch build: a batch holds one problem type", "[batch_build]") {
    auto p = aa_params();
    SeqPairBatch batch(views(AA_A), views(AA_B), p,
                       GapModel::Affine, AlignMode::Local, GradMode::Hard, 2);
    REQUIRE(batch.size() == AA_A.size());
    REQUIRE(batch.gap_model() == GapModel::Affine);
    REQUIRE(batch.align_mode() == AlignMode::Local);
    REQUIRE(batch.grad_mode() == GradMode::Hard);
}

// ── failures throw, and leak nothing ──────────────────────────────────────────

TEST_CASE("batch build: length mismatch throws", "[batch_build]") {
    auto p = aa_params();
    std::vector<std::string> short_b(AA_B.begin(), AA_B.end() - 1);
    REQUIRE_THROWS_AS(SeqPairBatch(views(AA_A), views(short_b), p,
                                   GapModel::Affine, AlignMode::Local, GradMode::Hard, 2),
                      std::invalid_argument);
}

TEST_CASE("batch build: bad character throws from a worker", "[batch_build]") {
    // Enough pairs that several threads are mid-construction when one throws:
    // what the others encoded must all be released, not leaked.
    auto p = dna_params();
    std::vector<std::string> bad_a, bad_b;
    for (int i = 0; i < 500; ++i) { bad_a.push_back("ACGTACGT"); bad_b.push_back("ACGTACGT"); }
    bad_a[250] = "ACGZ";                      // 'Z' is not in ACGT
    REQUIRE_THROWS_AS(SeqPairBatch(views(bad_a), views(bad_b), p,
                                   GapModel::Affine, AlignMode::Local, GradMode::Hard, 4),
                      std::invalid_argument);
}

TEST_CASE("batch build: empty is a no-op", "[batch_build]") {
    auto p = aa_params();
    SeqPairBatch batch({}, {}, p,
                       GapModel::Affine, AlignMode::Local, GradMode::Hard, 2);
    REQUIRE(batch.size() == 0);
    REQUIRE_THROWS(batch.compute_grad());     // empty batch has no alphabet
}

// ── the deprecated add_many(): appending (removed in 0.8.0) ──────────────────

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

TEST_CASE("add_many: appending keeps the problem type", "[add_many]") {
    auto p = aa_params();
    SeqPairBatch batch(views(AA_A), views(AA_B), p,
                       GapModel::Affine, AlignMode::Local, GradMode::Hard, 2);
    batch.add_many(views(AA_A), views(AA_B), p);
    REQUIRE(batch.size() == 2 * AA_A.size());
    REQUIRE(batch.gap_model() == GapModel::Affine);
    batch.score_and_grad();
    REQUIRE(batch[AA_A.size()].score() == batch[0].score());
}

TEST_CASE("add_many: length mismatch throws, batch unchanged", "[add_many]") {
    auto p = aa_params();
    SeqPairBatch batch(GapModel::Affine, AlignMode::Local, GradMode::Hard, 2);
    std::vector<std::string> short_b(AA_B.begin(), AA_B.end() - 1);
    REQUIRE_THROWS_AS(
        batch.add_many(views(AA_A), views(short_b), p),
        std::invalid_argument);
    REQUIRE(batch.size() == 0);
}

TEST_CASE("add_many: bad character throws from a worker, batch unchanged", "[add_many]") {
    auto p = dna_params();
    std::vector<std::string> good_a = {"ACGT", "ACGTACGT", "GGGG"};
    std::vector<std::string> good_b = {"ACGT", "ACGTTCGT", "GGAG"};
    SeqPairBatch batch(views(good_a), views(good_b), p, GapModel::Affine, AlignMode::Local, GradMode::Hard, 4);
    REQUIRE(batch.size() == 3);

    // Enough pairs that several threads are mid-construction when one throws:
    // the staged pairs built by the others must all be destroyed, not leaked.
    std::vector<std::string> bad_a, bad_b;
    for (int i = 0; i < 500; ++i) { bad_a.push_back("ACGTACGT"); bad_b.push_back("ACGTACGT"); }
    bad_a[250] = "ACGZ";                      // 'Z' is not in ACGT
    REQUIRE_THROWS_AS(
        batch.add_many(views(bad_a), views(bad_b), p),
        std::invalid_argument);

    REQUIRE(batch.size() == 3);               // the failed call added nothing
    batch.score_and_grad();                   // and the survivors still align
    REQUIRE(batch[0].score_valid());
}

TEST_CASE("add_many: alphabet mismatch throws, batch unchanged", "[add_many]") {
    auto aa = aa_params();
    auto dna = dna_params();
    SeqPairBatch batch(views(AA_A), views(AA_B), aa,
                       GapModel::Affine, AlignMode::Local, GradMode::Hard, 2);

    std::vector<std::string> da = {"ACGT"}, db = {"ACGT"};
    REQUIRE_THROWS_AS(
        batch.add_many(views(da), views(db), dna),
        std::invalid_argument);
    REQUIRE(batch.size() == AA_A.size());
    batch.score_and_grad();
}

#pragma GCC diagnostic pop
