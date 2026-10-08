// Tests for nwgrad::logistic, and for the parallel SeqPairBatch::set_params()
// and scores() it relies on.

#include "catch.hpp"
#include "align_params.hpp"
#include "logistic/logistic.hpp"
#include "seq_pair.hpp"
#include "seq_pair_batch.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

AlignParams params_over(const char* symbols, double match) {
    std::array<double, 16> src{};
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) src[i * 4 + j] = (i == j) ? match : -1.0;
    AlignParams p(SubstMatrix(src.data(), Alphabet::get(symbols)));
    p.gap_open_a = p.gap_open_b = 2.0;
    p.gap_extend_a = p.gap_extend_b = 0.5;
    return p;
}

struct Fixture {
    std::vector<std::string> a, b;
    std::vector<double> y;
    explicit Fixture(size_t n) : a(n), b(n), y(n) {
        uint64_t state = 99;
        auto next = [&]() { state = state * 6364136223846793005ULL + 1442695040888963407ULL; return state >> 33; };
        for (size_t i = 0; i < n; ++i) {
            for (size_t k = 0, len = 4 + next() % 9; k < len; ++k) a[i] += "ACGT"[next() % 4];
            for (size_t k = 0, len = 4 + next() % 13; k < len; ++k) b[i] += "ACGT"[next() % 4];
            y[i] = static_cast<double>(next() % 2);
        }
        y[0] = 0.0; y[1] = 1.0;
    }
    std::vector<std::string_view> va() const { return {a.begin(), a.end()}; }
    std::vector<std::string_view> vb() const { return {b.begin(), b.end()}; }
};

std::vector<double> flat(const AlignParams& g) {
    std::vector<double> out(16);
    g.matrix.to_array(out.data());
    for (double v : {g.gap_open_a, g.gap_extend_a, g.gap_open_b, g.gap_extend_b}) out.push_back(v);
    return out;
}

}  // namespace

TEST_CASE("logistic: sums and the fitted alpha do not depend on the thread count", "[logistic]") {
    const size_t n = 3 * nwgrad::logistic::BLOCK + 11;
    std::vector<double> s(n), y(n);
    for (size_t i = 0; i < n; ++i) {
        s[i] = std::sin(0.37 * static_cast<double>(i)) * 4.0 - 2.0;
        y[i] = (std::cos(0.11 * static_cast<double>(i)) > 0.2) ? 1.0 : 0.0;
    }
    const auto ref = nwgrad::logistic::evaluate(s.data(), y.data(), n, -0.3, 1);
    const double ref_alpha = nwgrad::logistic::fit_alpha(s.data(), y.data(), n, 7.0, 1);
    for (int t : {2, 3, 8}) {
        const auto e = nwgrad::logistic::evaluate(s.data(), y.data(), n, -0.3, t);
        REQUIRE(std::memcmp(&e, &ref, sizeof e) == 0);
        const double a = nwgrad::logistic::fit_alpha(s.data(), y.data(), n, 7.0, t);
        REQUIRE(std::memcmp(&a, &ref_alpha, sizeof a) == 0);
    }
    // alpha is the root of dL/dalpha.
    const auto at = nwgrad::logistic::evaluate(s.data(), y.data(), n, ref_alpha, 1);
    REQUIRE(std::abs(at.g) < 1e-8 * static_cast<double>(n));
}

TEST_CASE("logistic: invalid labels throw", "[logistic]") {
    std::vector<double> s = {0.0, 1.0, 2.0};
    for (std::vector<double> y : {std::vector<double>{0, 0, 0}, std::vector<double>{1, 1, 1},
                                  std::vector<double>{0, 1, 2}}) {
        REQUIRE_THROWS_AS(nwgrad::logistic::fit_alpha(s.data(), y.data(), 3, 0.0, 1),
                          std::invalid_argument);
    }
}

TEST_CASE("logistic: step is its parts composed", "[logistic]") {
    Fixture f(2 * nwgrad::logistic::BLOCK + 5);
    auto p = params_over("ACGT", 2.0);
    SeqPairBatchT<double> batch(4);
    batch.add_many(f.va(), f.vb(), p, GapModel::Affine, AlignMode::Local, GradMode::Hard);
    batch.score_and_grad();
    const auto st = nwgrad::logistic::step(batch, f.y.data(), f.y.size(), -0.4);
    const auto s = batch.scores();
    REQUIRE(st.alpha == nwgrad::logistic::fit_alpha(s.data(), f.y.data(), s.size(), -0.4, 4));
    std::vector<double> w(s.size());
    for (size_t i = 0; i < s.size(); ++i) w[i] = f.y[i] - nwgrad::logistic::expit(st.alpha + s[i]);
    REQUIRE(flat(st.grad) == flat(batch.weighted_grad(w.data(), w.size())));
    REQUIRE_THROWS_AS(nwgrad::logistic::step(batch, f.y.data(), f.y.size() - 1, 0.0), std::invalid_argument);
}

TEST_CASE("SeqPairBatch: parallel set_params and scores match the pairs", "[logistic]") {
    Fixture f(2 * 4096 + 9);   // more than one of the old 4096-pair scores() blocks
    auto p1 = params_over("ACGT", 2.0);
    auto p2 = params_over("ACGT", 3.0);
    SeqPairBatchT<double> batch(5);
    batch.add_many(f.va(), f.vb(), p1, GapModel::Affine, AlignMode::Global, GradMode::Hard);
    batch.score_and_grad();
    batch.set_params(p2);
    for (size_t i = 0; i < batch.size(); ++i) REQUIRE_FALSE(batch[i].score_valid());
    batch.score_and_grad();
    const auto s = batch.scores();
    for (size_t i = 0; i < batch.size(); ++i) REQUIRE(s[i] == batch[i].score());

    // A different alphabet is rejected before any pair changes.
    auto other = params_over("TGCA", 2.0);
    REQUIRE_THROWS_AS(batch.set_params(other), std::invalid_argument);
    for (size_t i = 0; i < batch.size(); ++i) REQUIRE(batch[i].score_valid());
}
