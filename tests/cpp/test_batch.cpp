#include "catch.hpp"
#include "batch.hpp"

static BlosumMatrix unit_matrix() {
    std::array<double, 400> src{};
    for (int i = 0; i < 20; ++i)
        src[i * 20 + i] = 1.0;
    return BlosumMatrix(src.data());
}

static BatchAligner make_aligner(int n_threads, bool grad = true,
                                  GapModel gm = GapModel::Linear,
                                  AlignMode am = AlignMode::Global) {
    auto gd = grad ? BatchAligner::GradMode::Hard : BatchAligner::GradMode::None;
    return BatchAligner(unit_matrix(), /*gap_open=*/0.0, /*gap_extend=*/1.0, /*band=*/0,
                        gm, am, gd, n_threads);
}

TEST_CASE("BatchAligner: empty problem list", "[batch]") {
    auto ba = make_aligner(2);
    auto result = ba.align({});
    REQUIRE(result.scores.empty());
}

TEST_CASE("BatchAligner: single pair, single thread", "[batch]") {
    auto ba = make_aligner(1);
    std::string a = "ACDE", b = "ACDE";
    std::vector<ProblemInstance> problems{{a, b, {}}};
    auto result = ba.align(problems);

    REQUIRE(result.scores.size() == 1);
    REQUIRE(result.scores[0] == Approx(4.0));
}

TEST_CASE("BatchAligner: gradient matches GradAligner single-threaded", "[batch]") {
    auto ba = make_aligner(1, /*grad=*/true);
    std::string a = "ACDE", b = "ACDE";
    std::vector<ProblemInstance> problems{{a, b, {}}};
    auto result = ba.align(problems);

    // 4 identical AAs → diagonal entries each 1.0
    REQUIRE(result.grad[(unsigned char)'A'][(unsigned char)'A'] == Approx(1.0));
    REQUIRE(result.grad[(unsigned char)'C'][(unsigned char)'C'] == Approx(1.0));
    REQUIRE(result.grad[(unsigned char)'D'][(unsigned char)'D'] == Approx(1.0));
    REQUIRE(result.grad[(unsigned char)'E'][(unsigned char)'E'] == Approx(1.0));
}

TEST_CASE("BatchAligner: multi-thread scores match single-thread", "[batch]") {
    // Build a batch of varied pairs
    std::vector<std::string> seqs_a = {"ACDE", "ADE", "MMMADE", "A", "ACDEFGHIK"};
    std::vector<std::string> seqs_b = {"ACDE", "ACDE", "ADE",   "",  "ACDEFGHIK"};

    std::vector<ProblemInstance> problems;
    for (size_t i = 0; i < seqs_a.size(); ++i)
        problems.push_back({seqs_a[i], seqs_b[i], {}});

    auto single = make_aligner(1);
    auto multi  = make_aligner(4);

    auto r1 = single.align(problems);
    auto r2 = multi.align(problems);

    REQUIRE(r1.scores.size() == r2.scores.size());
    for (size_t i = 0; i < r1.scores.size(); ++i)
        REQUIRE(r1.scores[i] == Approx(r2.scores[i]));
}

TEST_CASE("BatchAligner: multi-thread gradient matches single-thread", "[batch]") {
    std::vector<std::string> seqs_a = {"ACDE", "ADE", "A"};
    std::vector<std::string> seqs_b = {"ACDE", "ADE", "A"};
    std::vector<ProblemInstance> problems;
    for (size_t i = 0; i < seqs_a.size(); ++i)
        problems.push_back({seqs_a[i], seqs_b[i], {}});

    auto single = make_aligner(1);
    auto multi  = make_aligner(4);

    auto r1 = single.align(problems);
    auto r2 = multi.align(problems);

    for (int i = 0; i < 256; ++i)
        for (int j = 0; j < 256; ++j)
            REQUIRE(r1.grad[i][j] == Approx(r2.grad[i][j]));
}

TEST_CASE("BatchAligner: no gradient mode skips accumulation", "[batch]") {
    auto ba = make_aligner(2, /*grad=*/false);
    std::string a = "ACDE", b = "ACDE";
    std::vector<ProblemInstance> problems{{a, b, {}}};
    auto result = ba.align(problems);

    REQUIRE(result.scores[0] == Approx(4.0));
    // grad should remain all zeros
    double total = 0.0;
    for (int i = 0; i < 256; ++i)
        for (int j = 0; j < 256; ++j)
            total += result.grad[i][j];
    REQUIRE(total == Approx(0.0));
}

TEST_CASE("BatchAligner: scores are indexed correctly (not scrambled by threading)", "[batch]") {
    // Pairs with known distinct scores: pair i scores i+1 matches
    std::vector<std::string> seqs_a = {"A", "AC", "ACD", "ACDE"};
    std::vector<std::string> seqs_b = {"A", "AC", "ACD", "ACDE"};
    std::vector<ProblemInstance> problems;
    for (size_t i = 0; i < seqs_a.size(); ++i)
        problems.push_back({seqs_a[i], seqs_b[i], {}});

    auto ba = make_aligner(4);
    auto result = ba.align(problems);

    for (size_t i = 0; i < problems.size(); ++i)
        REQUIRE(result.scores[i] == Approx(static_cast<double>(i + 1)));
}
