// Tests for SeqPairBatch::scores(), SeqPairBatch::weighted_grad() and
// SeqPairBatch::grads_into().

#include "catch.hpp"
#include "align_params.hpp"
#include "seq_pair.hpp"
#include "seq_pair_batch.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
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

// Every product is rounded on its own before it is added: no FMA.  The reference
// forces that with a volatile store.  On an FMA target (the ARM64 CI job) a
// contracted sum in weighted_grad() would miss this bit-exact equality.
TEST_CASE("weighted_grad: products are rounded before they are summed", "[weighted_grad]") {
    auto p = dna_params();
    SeqPairBatchT<double> batch(2);
    batch.add_many(views(SEQS_A), views(SEQS_B), p, GapModel::Affine,
                   AlignMode::Local, GradMode::Hard);
    batch.score_and_grad();
    // Weights with long mantissas, so a fused multiply-add would round differently.
    const std::vector<double> w = {0.1, -1.0 / 3.0, 2.0 / 7.0, 1e-3 / 9.0, -0.7};

    std::vector<double> expected(20, 0.0);
    for (size_t i = 0; i < batch.size(); ++i) {
        std::vector<double> g = flat(batch[i].grad());
        for (size_t k = 0; k < g.size(); ++k) {
            volatile double product = w[i] * g[k];
            expected[k] += product;
        }
    }
    REQUIRE(flat(batch.weighted_grad(w.data(), w.size())) == expected);
}

TEST_CASE("grads_into: copies every pair's cached gradient", "[weighted_grad]") {
    auto p = dna_params();
    SeqPairBatchT<double> batch(3);
    batch.add_many(views(SEQS_A), views(SEQS_B), p, GapModel::Affine,
                   AlignMode::Local, GradMode::Hard);
    batch.score_and_grad();
    const size_t N = batch.size();
    std::vector<double> mats(N * 16, -1.0), gaps(N * 4, -1.0);
    batch.grads_into(mats.data(), gaps.data());
    for (size_t i = 0; i < N; ++i) {
        std::vector<double> e = flat(batch[i].grad());
        for (size_t k = 0; k < 16; ++k) REQUIRE(mats[i * 16 + k] == e[k]);
        for (size_t k = 0; k < 4; ++k) REQUIRE(gaps[i * 4 + k] == e[16 + k]);
    }
}

TEST_CASE("grads_into: preconditions throw", "[weighted_grad]") {
    auto p = dna_params();
    SeqPairBatchT<double> empty(1);
    REQUIRE_THROWS_AS(empty.grads_into(nullptr, nullptr), std::logic_error);

    SeqPairBatchT<double> batch(2);
    batch.add_many(views(SEQS_A), views(SEQS_B), p, GapModel::Affine,
                   AlignMode::Global, GradMode::Hard);
    std::vector<double> mats(batch.size() * 16), gaps(batch.size() * 4);
    REQUIRE_THROWS_AS(batch.grads_into(mats.data(), gaps.data()), std::logic_error);
}

// More pairs than one block: the sum is per block in pair order, then over the
// blocks in order, whatever the thread count.  The reference rounds every
// product on its own (volatile), as in the test above, and adds in that order.
TEST_CASE("weighted_grad: block order, bit-identical across thread counts", "[weighted_grad]") {
    const size_t B = BatchEngine<double, GapModel::Linear, AlignMode::Global>::WEIGHTED_GRAD_BLOCK;
    const size_t N = 3 * B + 17;
    static const char ACGT[] = "ACGT";
    std::vector<std::string> a(N), b(N);
    uint64_t state = 12345;
    auto next = [&]() { state = state * 6364136223846793005ULL + 1442695040888963407ULL; return state >> 33; };
    for (size_t i = 0; i < N; ++i) {
        for (size_t k = 0, len = 4 + next() % 9; k < len; ++k) a[i] += ACGT[next() % 4];
        for (size_t k = 0, len = 4 + next() % 13; k < len; ++k) b[i] += ACGT[next() % 4];
    }
    std::vector<double> w(N);
    for (size_t i = 0; i < N; ++i) w[i] = (static_cast<double>(next() % 2000001) - 1e6) / 7e5;

    auto p = dna_params();
    std::vector<std::vector<double>> results;
    std::vector<double> expected(20, 0.0);
    for (int threads : {1, 2, 7}) {
        SeqPairBatchT<double> batch(threads);
        batch.add_many(views(a), views(b), p, GapModel::Affine, AlignMode::Local, GradMode::Hard);
        batch.score_and_grad();
        results.push_back(flat(batch.weighted_grad(w.data(), N)));
        if (threads == 1) {
            for (size_t lo = 0; lo < N; lo += B) {
                std::vector<double> block(20, 0.0);
                for (size_t i = lo; i < std::min(N, lo + B); ++i) {
                    std::vector<double> g = flat(batch[i].grad());
                    for (size_t k = 0; k < g.size(); ++k) {
                        volatile double product = w[i] * g[k];
                        block[k] += product;
                    }
                }
                for (size_t k = 0; k < 20; ++k) expected[k] += block[k];
            }
        }
    }
    REQUIRE(results[0] == expected);
    for (const auto& r : results)
        REQUIRE(std::memcmp(r.data(), results[0].data(), r.size() * sizeof(double)) == 0);
}
