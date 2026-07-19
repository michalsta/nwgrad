#include "catch.hpp"
#include "align_params.hpp"
#include "batch.hpp"
#include <array>

static AlignParams unit_params(double gap_extend = 1.0, double gap_open = 0.0) {
    std::array<double, 400> src{};
    for (int i = 0; i < 20; ++i)
        src[i * 20 + i] = 1.0;
    AlignParams p(SubstMatrix(src.data()));
    p.gap_extend_a = p.gap_extend_b = gap_extend;
    p.gap_open_a   = p.gap_open_b   = gap_open;
    return p;
}

static BatchAligner make_aligner(int n_threads, bool grad = true,
                                  GapModel gm = GapModel::Linear,
                                  AlignMode am = AlignMode::Global) {
    auto gd = grad ? BatchAligner::GradMode::Hard : BatchAligner::GradMode::None;
    return BatchAligner(unit_params(), /*band=*/0, gm, am, gd, n_threads);
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

    REQUIRE(result.grad.matrix.at(Alphabet::protein().index_of('A'), Alphabet::protein().index_of('A')) == Approx(1.0));
    REQUIRE(result.grad.matrix.at(Alphabet::protein().index_of('C'), Alphabet::protein().index_of('C')) == Approx(1.0));
    REQUIRE(result.grad.matrix.at(Alphabet::protein().index_of('D'), Alphabet::protein().index_of('D')) == Approx(1.0));
    REQUIRE(result.grad.matrix.at(Alphabet::protein().index_of('E'), Alphabet::protein().index_of('E')) == Approx(1.0));
}

TEST_CASE("BatchAligner: multi-thread scores match single-thread", "[batch]") {
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

    for (int i = 0; i < 20; ++i)
        for (int j = 0; j < 20; ++j)
            REQUIRE(r1.grad.matrix.at(i, j) == Approx(r2.grad.matrix.at(i, j)));
}

TEST_CASE("BatchAligner: no gradient mode skips accumulation", "[batch]") {
    auto ba = make_aligner(2, /*grad=*/false);
    std::string a = "ACDE", b = "ACDE";
    std::vector<ProblemInstance> problems{{a, b, {}}};
    auto result = ba.align(problems);

    REQUIRE(result.scores[0] == Approx(4.0));
    double total = 0.0;
    for (int i = 0; i < 20; ++i)
        for (int j = 0; j < 20; ++j)
            total += result.grad.matrix.at(i, j);
    REQUIRE(total == Approx(0.0));
}

TEST_CASE("BatchAligner: mixed guided/unguided problems throw when band == 0", "[batch]") {
    auto ba = make_aligner(2);
    std::vector<ProblemInstance> problems{
        {"ACDE", "ACDE", {}},                                    // unguided
        {"ACDE", "ACDE", guide_j_from_aligned("ACDE", "ACDE")},  // guided
    };
    REQUIRE_THROWS_AS(ba.align(problems), std::invalid_argument);
}

TEST_CASE("BatchAligner: all-unguided batch with band == 0 does not throw", "[batch]") {
    auto ba = make_aligner(2);
    std::vector<ProblemInstance> problems{
        {"ACDE", "ACDE", {}},
        {"ADE",  "ACDE", {}},
    };
    REQUIRE_NOTHROW(ba.align(problems));
}

TEST_CASE("BatchAligner: all-guided batch with band == 0 does not throw", "[batch]") {
    auto ba = make_aligner(2);
    std::vector<ProblemInstance> problems{
        {"ACDE", "ACDE", guide_j_from_aligned("ACDE", "ACDE")},
        {"ADE",  "ADE",  guide_j_from_aligned("ADE",  "ADE")},
    };
    REQUIRE_NOTHROW(ba.align(problems));
}

TEST_CASE("BatchAligner: mixed guided/unguided problems do not throw when band > 0", "[batch]") {
    auto ba = make_aligner(2);
    ba.band = 2;
    std::vector<ProblemInstance> problems{
        {"ACDE", "ACDE", {}},
        {"ACDE", "ACDE", guide_j_from_aligned("ACDE", "ACDE")},
    };
    REQUIRE_NOTHROW(ba.align(problems));
}

TEST_CASE("BatchAligner: scores are indexed correctly (not scrambled by threading)", "[batch]") {
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
