#include "catch.hpp"
#include "aligner.hpp"

// Unit substitution matrix: match=1, mismatch=0
static BlosumMatrix unit_matrix() {
    std::array<double, 400> src{};
    for (int i = 0; i < 20; ++i)
        src[i * 20 + i] = 1.0;
    return BlosumMatrix(src.data());
}

// Convenience: set problem, run viterbi, return score.
template<GapModel GM, AlignMode AM>
static double score(Aligner<GM,AM>& al, const BlosumMatrix& mat,
                    const char* a, const char* b,
                    double gap_extend, double gap_open = 0.0) {
    al.set_problem(a, b, mat, gap_extend, gap_open);
    al.compute_viterbi();
    return al.score();
}

// ── Global / Linear ──────────────────────────────────────────────────────────

TEST_CASE("NW linear: identical sequences", "[aligner][global][linear]") {
    auto mat = unit_matrix();
    Aligner<GapModel::Linear, AlignMode::Global> al;
    REQUIRE(score(al, mat, "ACDE", "ACDE", 1.0) == Approx(4.0));
}

TEST_CASE("NW linear: empty sequences", "[aligner][global][linear]") {
    auto mat = unit_matrix();
    Aligner<GapModel::Linear, AlignMode::Global> al;
    REQUIRE(score(al, mat, "", "", 1.0)     == Approx(0.0));
    // One empty: n gaps, each costing 1.0
    REQUIRE(score(al, mat, "ACDE", "", 1.0) == Approx(-4.0));
    REQUIRE(score(al, mat, "", "ACDE", 1.0) == Approx(-4.0));
}

TEST_CASE("NW linear: single gap in middle", "[aligner][global][linear]") {
    auto mat = unit_matrix();
    Aligner<GapModel::Linear, AlignMode::Global> al;
    // "ACDE" vs "ADE": best is A-A, C→gap(-1), D-D, E-E = 3 - 1 = 2
    REQUIRE(score(al, mat, "ACDE", "ADE", 1.0) == Approx(2.0));
}

TEST_CASE("NW linear: all mismatches score lower than gaps", "[aligner][global][linear]") {
    auto mat = unit_matrix();  // mismatch = 0, gap_extend = 1
    Aligner<GapModel::Linear, AlignMode::Global> al;
    // "AA" vs "CC": optimal is align A/C, A/C → 0 + 0 = 0
    REQUIRE(score(al, mat, "AA", "CC", 1.0) == Approx(0.0));
}

TEST_CASE("NW linear: gap penalty scales with length", "[aligner][global][linear]") {
    auto mat = unit_matrix();
    Aligner<GapModel::Linear, AlignMode::Global> al;
    REQUIRE(score(al, mat, "A",    "", 2.0) == Approx(-2.0));
    REQUIRE(score(al, mat, "AAAA", "", 2.0) == Approx(-8.0));
}

// ── Local / Linear ───────────────────────────────────────────────────────────

TEST_CASE("SW linear: score >= 0", "[aligner][local][linear]") {
    auto mat = unit_matrix();
    Aligner<GapModel::Linear, AlignMode::Local> al;
    REQUIRE(score(al, mat, "AAAA", "CCCC", 1.0) >= 0.0);
}

TEST_CASE("SW linear: finds local match inside longer sequence", "[aligner][local][linear]") {
    auto mat = unit_matrix();
    Aligner<GapModel::Linear, AlignMode::Local> al;
    // "ADE" embedded in "MMMADEMM"
    REQUIRE(score(al, mat, "ADE", "MMMADEMM", 1.0) == Approx(3.0));
}

TEST_CASE("SW linear: identical sequences equal NW score", "[aligner][local][linear]") {
    auto mat = unit_matrix();
    Aligner<GapModel::Linear, AlignMode::Global> nw;
    Aligner<GapModel::Linear, AlignMode::Local>  sw;
    // For identical sequences with positive scores, SW == NW
    REQUIRE(score(sw, mat, "ACDE", "ACDE", 1.0) ==
            Approx(score(nw, mat, "ACDE", "ACDE", 1.0)));
}

// ── Global / Affine ──────────────────────────────────────────────────────────

TEST_CASE("NW affine: identical sequences", "[aligner][global][affine]") {
    auto mat = unit_matrix();
    Aligner<GapModel::Affine, AlignMode::Global> al;
    REQUIRE(score(al, mat, "ACDE", "ACDE", /*gap_ext=*/1.0, /*gap_open=*/10.0) == Approx(4.0));
}

TEST_CASE("NW affine: single gap costs open+extend", "[aligner][global][affine]") {
    auto mat = unit_matrix();
    Aligner<GapModel::Affine, AlignMode::Global> al;
    // "A" vs "": 1 gap → -(10+1) = -11
    REQUIRE(score(al, mat, "A", "", 1.0, 10.0) == Approx(-11.0));
}

TEST_CASE("NW affine: extending gap cheaper than opening new one", "[aligner][global][affine]") {
    auto mat = unit_matrix();
    Aligner<GapModel::Affine, AlignMode::Global> al;
    // gap_open=10, gap_extend=1; "AAA" vs "A": best is 1 match + gap of length 2
    // → 1 - (10+2) = -11
    REQUIRE(score(al, mat, "AAA", "A", 1.0, 10.0) == Approx(-11.0));
}

TEST_CASE("NW affine: gap_open=0 matches linear model", "[aligner][global][affine]") {
    auto mat = unit_matrix();
    Aligner<GapModel::Linear, AlignMode::Global> linear;
    Aligner<GapModel::Affine, AlignMode::Global> affine;
    REQUIRE(score(affine, mat, "ACDE", "ADE",  1.0, 0.0) == Approx(score(linear, mat, "ACDE", "ADE",  1.0)));
    REQUIRE(score(affine, mat, "ACDE", "ACDE", 1.0, 0.0) == Approx(score(linear, mat, "ACDE", "ACDE", 1.0)));
    REQUIRE(score(affine, mat, "AAAA", "",     1.0, 0.0) == Approx(score(linear, mat, "AAAA", "",     1.0)));
}

// ── Local / Affine ───────────────────────────────────────────────────────────

TEST_CASE("SW affine: score >= 0", "[aligner][local][affine]") {
    auto mat = unit_matrix();
    Aligner<GapModel::Affine, AlignMode::Local> al;
    REQUIRE(score(al, mat, "AAAA", "CCCC", 1.0, 10.0) >= 0.0);
}

TEST_CASE("SW affine: finds local match inside longer sequence", "[aligner][local][affine]") {
    auto mat = unit_matrix();
    Aligner<GapModel::Affine, AlignMode::Local> al;
    REQUIRE(score(al, mat, "ADE", "MMMADEMM", 1.0, 10.0) == Approx(3.0));
}
