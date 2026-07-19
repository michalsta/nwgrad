#include "catch.hpp"
#include "align_params.hpp"
#include "aligner.hpp"
#include <array>
#include <cstring>

static AlignParams unit_params(double gap_extend, double gap_open = 0.0) {
    std::array<double, 400> src{};
    for (int i = 0; i < 20; ++i)
        src[i * 20 + i] = 1.0;
    AlignParams p(SubstMatrix(src.data()));
    p.gap_extend_a = p.gap_extend_b = gap_extend;
    p.gap_open_a   = p.gap_open_b   = gap_open;
    return p;
}

static double sum_grad(const AlignParams& g) {
    double s = 0.0;
    for (int i = 0; i < 20; ++i)
        for (int j = 0; j < 20; ++j)
            s += g.matrix.at(i, j);
    return s;
}

template<GapModel GM, AlignMode AM>
static void setup_aligner(Aligner<GM,AM>& al, const char* a, const char* b, const AlignParams& p) {
    al.alloc_buf();
    al.set_problem(a, b, p);
}

// ── Linear / Global ──────────────────────────────────────────────────────────

TEST_CASE("Gradient linear global: identical sequences — all matches", "[gradient][linear][global]") {
    auto p = unit_params(1.0);
    Aligner<GapModel::Linear, AlignMode::Global> al;
    al.alloc_buf();
    al.set_problem("ACDE", "ACDE", p);
    al.compute_viterbi();
    AlignParams grad(Alphabet::protein());
    al.hard_grad(grad);

    REQUIRE(al.score() == Approx(4.0));
    REQUIRE(grad.matrix.at(Alphabet::protein().index_of('A'), Alphabet::protein().index_of('A')) == Approx(1.0));
    REQUIRE(grad.matrix.at(Alphabet::protein().index_of('C'), Alphabet::protein().index_of('C')) == Approx(1.0));
    REQUIRE(grad.matrix.at(Alphabet::protein().index_of('D'), Alphabet::protein().index_of('D')) == Approx(1.0));
    REQUIRE(grad.matrix.at(Alphabet::protein().index_of('E'), Alphabet::protein().index_of('E')) == Approx(1.0));
    REQUIRE(sum_grad(grad) == Approx(4.0));
}

TEST_CASE("Gradient linear global: gaps produce zero gradient contribution", "[gradient][linear][global]") {
    auto p = unit_params(1.0);
    Aligner<GapModel::Linear, AlignMode::Global> al;
    al.alloc_buf();
    al.set_problem("A", "", p);
    al.compute_viterbi();
    AlignParams grad(Alphabet::protein());
    al.hard_grad(grad);
    REQUIRE(sum_grad(grad) == Approx(0.0));
}

TEST_CASE("Gradient linear global: repeated calls accumulate", "[gradient][linear][global]") {
    auto p = unit_params(1.0);
    Aligner<GapModel::Linear, AlignMode::Global> al;
    al.alloc_buf();
    AlignParams grad(Alphabet::protein());

    al.set_problem("AA", "AA", p);
    al.compute_viterbi();
    al.hard_grad(grad);

    al.set_problem("AA", "AA", p);
    al.compute_viterbi();
    al.hard_grad(grad);

    REQUIRE(grad.matrix.at(Alphabet::protein().index_of('A'), Alphabet::protein().index_of('A')) == Approx(4.0));
}

TEST_CASE("Gradient linear global: one-gap alignment has correct count", "[gradient][linear][global]") {
    auto p = unit_params(0.5);
    Aligner<GapModel::Linear, AlignMode::Global> al;
    al.alloc_buf();
    al.set_problem("ADE", "ACDE", p);
    al.compute_viterbi();
    AlignParams grad(Alphabet::protein());
    al.hard_grad(grad);
    REQUIRE(sum_grad(grad) == Approx(3.0));
}

// ── Linear / Local ───────────────────────────────────────────────────────────

TEST_CASE("Gradient linear local: only matching region contributes", "[gradient][linear][local]") {
    auto p = unit_params(1.0);
    Aligner<GapModel::Linear, AlignMode::Local> al;
    al.alloc_buf();
    al.set_problem("ADE", "MMMADEM", p);
    al.compute_viterbi();
    AlignParams grad(Alphabet::protein());
    al.hard_grad(grad);

    REQUIRE(sum_grad(grad) == Approx(3.0));
    REQUIRE(grad.matrix.at(Alphabet::protein().index_of('A'), Alphabet::protein().index_of('A')) == Approx(1.0));
    REQUIRE(grad.matrix.at(Alphabet::protein().index_of('D'), Alphabet::protein().index_of('D')) == Approx(1.0));
    REQUIRE(grad.matrix.at(Alphabet::protein().index_of('E'), Alphabet::protein().index_of('E')) == Approx(1.0));
    REQUIRE(grad.matrix.at(Alphabet::protein().index_of('M'), Alphabet::protein().index_of('M')) == Approx(0.0));
}

TEST_CASE("Gradient linear local: no match gives zero gradient", "[gradient][linear][local]") {
    auto p = unit_params(1.0);
    Aligner<GapModel::Linear, AlignMode::Local> al;
    al.alloc_buf();
    al.set_problem("AAAA", "CCCC", p);
    al.compute_viterbi();
    AlignParams grad(Alphabet::protein());
    al.hard_grad(grad);
    REQUIRE(sum_grad(grad) == Approx(0.0));
}

// ── Affine / Global ──────────────────────────────────────────────────────────

TEST_CASE("Gradient affine global: identical sequences", "[gradient][affine][global]") {
    auto p = unit_params(1.0, 10.0);
    Aligner<GapModel::Affine, AlignMode::Global> al;
    al.alloc_buf();
    al.set_problem("ACDE", "ACDE", p);
    al.compute_viterbi();
    AlignParams grad(Alphabet::protein());
    al.hard_grad(grad);

    REQUIRE(al.score() == Approx(4.0));
    REQUIRE(sum_grad(grad) == Approx(4.0));
}

TEST_CASE("Gradient affine global: gap produces no gradient", "[gradient][affine][global]") {
    auto p = unit_params(1.0, 10.0);
    Aligner<GapModel::Affine, AlignMode::Global> al;
    al.alloc_buf();
    al.set_problem("A", "", p);
    al.compute_viterbi();
    AlignParams grad(Alphabet::protein());
    al.hard_grad(grad);
    REQUIRE(sum_grad(grad) == Approx(0.0));
}

// ── Affine / Local ───────────────────────────────────────────────────────────

TEST_CASE("Gradient affine local: finds and counts local match", "[gradient][affine][local]") {
    auto p = unit_params(1.0, 10.0);
    Aligner<GapModel::Affine, AlignMode::Local> al;
    al.alloc_buf();
    al.set_problem("ADE", "MMMADEM", p);
    al.compute_viterbi();
    AlignParams grad(Alphabet::protein());
    al.hard_grad(grad);
    REQUIRE(sum_grad(grad) == Approx(3.0));
}
