// Tests for AlignParams::add_scaled(), SeqPairBatch::scores() and
// SeqPairBatch::weighted_grad().

#include "catch.hpp"
#include "align_params.hpp"
#include "seq_pair.hpp"
#include "seq_pair_batch.hpp"

#include <array>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

static AlignParams dna_params() {
    std::array<double, 16> src{};
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            src[i * 4 + j] = (i == j) ? 2.0 : -1.0;
    src[0 * 4 + 2] = 0.5;   // asymmetric
    AlignParams p(SubstMatrix(src.data(), Alphabet::get("ACGT")));
    p.gap_open_a = 2.0; p.gap_extend_a = 1.0;
    p.gap_open_b = 1.5; p.gap_extend_b = 0.5;
    return p;
}

static const std::vector<std::string> SEQS_A = {"ACGTACGT", "AAT", "GATTACA", "CCGGA", "T"};
static const std::vector<std::string> SEQS_B = {"AGTACT", "ACT", "GATCA", "CGGTA", "TT"};
static const std::vector<double> WEIGHTS = {0.5, -1.25, 2.0, 0.0, -0.75};

static std::vector<std::string_view> views(const std::vector<std::string>& v) {
    return std::vector<std::string_view>(v.begin(), v.end());
}

static std::vector<double> flat(const AlignParams& g) {
    std::vector<double> out(16);
    g.matrix.to_array(out.data());
    out.push_back(g.gap_open_a);
    out.push_back(g.gap_extend_a);
    out.push_back(g.gap_open_b);
    out.push_back(g.gap_extend_b);
    return out;
}

TEST_CASE("add_scaled: equals += s * o over every field", "[weighted_grad]") {
    AlignParams a = dna_params(), b = dna_params();
    b.gap_open_a = 7.0;
    AlignParams expected = a + (-0.5) * b;
    a.add_scaled(b, -0.5);
    REQUIRE(flat(a) == flat(expected));
}

TEST_CASE("add_scaled: different alphabets throw", "[weighted_grad]") {
    AlignParams a = dna_params();
    AlignParams rna(Alphabet::get("ACGU"));
    REQUIRE_THROWS_AS(a.add_scaled(rna, 1.0), std::invalid_argument);
}

TEST_CASE("scores / weighted_grad: match the per-pair values", "[weighted_grad]") {
    auto p = dna_params();
    SeqPairBatchT<double> batch(3);
    batch.add_many(views(SEQS_A), views(SEQS_B), p, GapModel::Affine,
                   AlignMode::Local, GradMode::Hard);
    batch.score_and_grad();

    std::vector<double> s = batch.scores();
    REQUIRE(s.size() == SEQS_A.size());
    for (size_t i = 0; i < s.size(); ++i) REQUIRE(s[i] == batch[i].score());

    AlignParams expected(p.alphabet());
    for (size_t i = 0; i < batch.size(); ++i) expected += WEIGHTS[i] * batch[i].grad();
    AlignParams got = batch.weighted_grad(WEIGHTS.data(), WEIGHTS.size());
    std::vector<double> fg = flat(got), fe = flat(expected);
    for (size_t k = 0; k < fg.size(); ++k) REQUIRE(fg[k] == Approx(fe[k]).margin(1e-12));
}

TEST_CASE("weighted_grad: bit-identical across thread counts", "[weighted_grad]") {
    auto p = dna_params();
    std::vector<std::vector<double>> results;
    for (int threads : {1, 2, 7}) {
        SeqPairBatchT<double> batch(threads);
        batch.add_many(views(SEQS_A), views(SEQS_B), p, GapModel::Linear,
                       AlignMode::Global, GradMode::Hard);
        batch.score_and_grad();
        results.push_back(flat(batch.weighted_grad(WEIGHTS.data(), WEIGHTS.size())));
    }
    for (const auto& r : results)
        REQUIRE(std::memcmp(r.data(), results[0].data(), r.size() * sizeof(double)) == 0);
}

TEST_CASE("weighted_grad: preconditions throw", "[weighted_grad]") {
    auto p = dna_params();
    SeqPairBatchT<double> empty(1);
    REQUIRE_THROWS_AS(empty.weighted_grad(nullptr, 0), std::logic_error);
    REQUIRE(empty.scores().empty());

    SeqPairBatchT<double> batch(2);
    batch.add_many(views(SEQS_A), views(SEQS_B), p, GapModel::Affine,
                   AlignMode::Global, GradMode::Hard);
    REQUIRE_THROWS_AS(batch.weighted_grad(WEIGHTS.data(), WEIGHTS.size()), std::logic_error);
    REQUIRE_THROWS_AS(batch.scores(), std::logic_error);

    batch.score_and_grad();
    REQUIRE_THROWS_AS(batch.weighted_grad(WEIGHTS.data(), WEIGHTS.size() - 1),
                      std::invalid_argument);
}
