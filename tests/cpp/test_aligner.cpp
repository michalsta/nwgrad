#include "catch.hpp"
#include "align_params.hpp"
#include "aligner.hpp"
#include <cstring>

static AlignParams unit_params(double gap_extend, double gap_open = 0.0) {
    std::array<double, 400> src{};
    for (int i = 0; i < 20; ++i)
        src[i * 20 + i] = 1.0;
    AlignParams p;
    p.matrix       = SubstMatrix(src.data());
    p.gap_extend_a = p.gap_extend_b = gap_extend;
    p.gap_open_a   = p.gap_open_b   = gap_open;
    return p;
}

template<GapModel GM, AlignMode AM>
static double score(Aligner<GM,AM>& al, const AlignParams& params,
                    const char* a, const char* b) {
    al.alloc_buf();
    al.set_problem(a, b, params);
    al.compute_viterbi();
    return al.score();
}

// ── Global / Linear ──────────────────────────────────────────────────────────

TEST_CASE("NW linear: identical sequences", "[aligner][global][linear]") {
    auto p = unit_params(1.0);
    Aligner<GapModel::Linear, AlignMode::Global> al;
    REQUIRE(score(al, p, "ACDE", "ACDE") == Approx(4.0));
}

TEST_CASE("NW linear: empty sequences", "[aligner][global][linear]") {
    auto p = unit_params(1.0);
    Aligner<GapModel::Linear, AlignMode::Global> al;
    REQUIRE(score(al, p, "", "")     == Approx(0.0));
    REQUIRE(score(al, p, "ACDE", "") == Approx(-4.0));
    REQUIRE(score(al, p, "", "ACDE") == Approx(-4.0));
}

TEST_CASE("NW linear: single gap in middle", "[aligner][global][linear]") {
    auto p = unit_params(1.0);
    Aligner<GapModel::Linear, AlignMode::Global> al;
    REQUIRE(score(al, p, "ACDE", "ADE") == Approx(2.0));
}

TEST_CASE("NW linear: all mismatches score lower than gaps", "[aligner][global][linear]") {
    auto p = unit_params(1.0);
    Aligner<GapModel::Linear, AlignMode::Global> al;
    REQUIRE(score(al, p, "AA", "CC") == Approx(0.0));
}

TEST_CASE("NW linear: gap penalty scales with length", "[aligner][global][linear]") {
    Aligner<GapModel::Linear, AlignMode::Global> al;
    REQUIRE(score(al, unit_params(2.0), "A",    "") == Approx(-2.0));
    REQUIRE(score(al, unit_params(2.0), "AAAA", "") == Approx(-8.0));
}

// ── Local / Linear ───────────────────────────────────────────────────────────

TEST_CASE("SW linear: score >= 0", "[aligner][local][linear]") {
    auto p = unit_params(1.0);
    Aligner<GapModel::Linear, AlignMode::Local> al;
    REQUIRE(score(al, p, "AAAA", "CCCC") >= 0.0);
}

TEST_CASE("SW linear: finds local match inside longer sequence", "[aligner][local][linear]") {
    auto p = unit_params(1.0);
    Aligner<GapModel::Linear, AlignMode::Local> al;
    REQUIRE(score(al, p, "ADE", "MMMADEMM") == Approx(3.0));
}

TEST_CASE("SW linear: identical sequences equal NW score", "[aligner][local][linear]") {
    auto p = unit_params(1.0);
    Aligner<GapModel::Linear, AlignMode::Global> nw;
    Aligner<GapModel::Linear, AlignMode::Local>  sw;
    REQUIRE(score(sw, p, "ACDE", "ACDE") == Approx(score(nw, p, "ACDE", "ACDE")));
}

// ── Global / Affine ──────────────────────────────────────────────────────────

TEST_CASE("NW affine: identical sequences", "[aligner][global][affine]") {
    auto p = unit_params(/*gap_ext=*/1.0, /*gap_open=*/10.0);
    Aligner<GapModel::Affine, AlignMode::Global> al;
    REQUIRE(score(al, p, "ACDE", "ACDE") == Approx(4.0));
}

TEST_CASE("NW affine: single gap costs open+extend", "[aligner][global][affine]") {
    auto p = unit_params(1.0, 10.0);
    Aligner<GapModel::Affine, AlignMode::Global> al;
    REQUIRE(score(al, p, "A", "") == Approx(-11.0));
}

TEST_CASE("NW affine: extending gap cheaper than opening new one", "[aligner][global][affine]") {
    auto p = unit_params(1.0, 10.0);
    Aligner<GapModel::Affine, AlignMode::Global> al;
    REQUIRE(score(al, p, "AAA", "A") == Approx(-11.0));
}

TEST_CASE("NW affine: gap_open=0 matches linear model", "[aligner][global][affine]") {
    Aligner<GapModel::Linear, AlignMode::Global> linear;
    Aligner<GapModel::Affine, AlignMode::Global> affine;
    auto pl = unit_params(1.0);
    auto pa = unit_params(1.0, 0.0);
    REQUIRE(score(affine, pa, "ACDE", "ADE")  == Approx(score(linear, pl, "ACDE", "ADE")));
    REQUIRE(score(affine, pa, "ACDE", "ACDE") == Approx(score(linear, pl, "ACDE", "ACDE")));
    REQUIRE(score(affine, pa, "AAAA", "")     == Approx(score(linear, pl, "AAAA", "")));
}

// ── Local / Affine ───────────────────────────────────────────────────────────

TEST_CASE("SW affine: score >= 0", "[aligner][local][affine]") {
    auto p = unit_params(1.0, 10.0);
    Aligner<GapModel::Affine, AlignMode::Local> al;
    REQUIRE(score(al, p, "AAAA", "CCCC") >= 0.0);
}

TEST_CASE("SW affine: finds local match inside longer sequence", "[aligner][local][affine]") {
    auto p = unit_params(1.0, 10.0);
    Aligner<GapModel::Affine, AlignMode::Local> al;
    REQUIRE(score(al, p, "ADE", "MMMADEMM") == Approx(3.0));
}
