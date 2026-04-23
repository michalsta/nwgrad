#include "catch.hpp"
#include "align_params.hpp"
#include "aligner.hpp"

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

static double sum_grad(const AlignParams& g) {
    double s = 0.0;
    for (int i = 0; i < 256; ++i)
        for (int j = 0; j < 256; ++j)
            s += g.matrix.mat[i][j];
    return s;
}

// ── Linear / Global ──────────────────────────────────────────────────────────

TEST_CASE("Gradient linear global: identical sequences — all matches", "[gradient][linear][global]") {
    auto p = unit_params(1.0);
    Aligner<GapModel::Linear, AlignMode::Global> al;
    al.set_problem("ACDE", "ACDE", p);
    al.compute_viterbi();
    AlignParams grad{};
    al.hard_grad(grad);

    REQUIRE(al.score() == Approx(4.0));
    REQUIRE(grad.matrix.mat[(unsigned char)'A'][(unsigned char)'A'] == Approx(1.0));
    REQUIRE(grad.matrix.mat[(unsigned char)'C'][(unsigned char)'C'] == Approx(1.0));
    REQUIRE(grad.matrix.mat[(unsigned char)'D'][(unsigned char)'D'] == Approx(1.0));
    REQUIRE(grad.matrix.mat[(unsigned char)'E'][(unsigned char)'E'] == Approx(1.0));
    REQUIRE(sum_grad(grad) == Approx(4.0));
}

TEST_CASE("Gradient linear global: gaps produce zero gradient contribution", "[gradient][linear][global]") {
    auto p = unit_params(1.0);
    Aligner<GapModel::Linear, AlignMode::Global> al;
    al.set_problem("A", "", p);
    al.compute_viterbi();
    AlignParams grad{};
    al.hard_grad(grad);
    REQUIRE(sum_grad(grad) == Approx(0.0));
}

TEST_CASE("Gradient linear global: repeated calls accumulate", "[gradient][linear][global]") {
    auto p = unit_params(1.0);
    Aligner<GapModel::Linear, AlignMode::Global> al;
    AlignParams grad{};

    al.set_problem("AA", "AA", p);
    al.compute_viterbi();
    al.hard_grad(grad);

    al.set_problem("AA", "AA", p);
    al.compute_viterbi();
    al.hard_grad(grad);

    REQUIRE(grad.matrix.mat[(unsigned char)'A'][(unsigned char)'A'] == Approx(4.0));
}

TEST_CASE("Gradient linear global: one-gap alignment has correct count", "[gradient][linear][global]") {
    auto p = unit_params(0.5);
    Aligner<GapModel::Linear, AlignMode::Global> al;
    al.set_problem("ADE", "ACDE", p);
    al.compute_viterbi();
    AlignParams grad{};
    al.hard_grad(grad);
    REQUIRE(sum_grad(grad) == Approx(3.0));
}

// ── Linear / Local ───────────────────────────────────────────────────────────

TEST_CASE("Gradient linear local: only matching region contributes", "[gradient][linear][local]") {
    auto p = unit_params(1.0);
    Aligner<GapModel::Linear, AlignMode::Local> al;
    al.set_problem("ADE", "MMMADEM", p);
    al.compute_viterbi();
    AlignParams grad{};
    al.hard_grad(grad);

    REQUIRE(sum_grad(grad) == Approx(3.0));
    REQUIRE(grad.matrix.mat[(unsigned char)'A'][(unsigned char)'A'] == Approx(1.0));
    REQUIRE(grad.matrix.mat[(unsigned char)'D'][(unsigned char)'D'] == Approx(1.0));
    REQUIRE(grad.matrix.mat[(unsigned char)'E'][(unsigned char)'E'] == Approx(1.0));
    REQUIRE(grad.matrix.mat[(unsigned char)'M'][(unsigned char)'M'] == Approx(0.0));
}

TEST_CASE("Gradient linear local: no match gives zero gradient", "[gradient][linear][local]") {
    auto p = unit_params(1.0);
    Aligner<GapModel::Linear, AlignMode::Local> al;
    al.set_problem("AAAA", "CCCC", p);
    al.compute_viterbi();
    AlignParams grad{};
    al.hard_grad(grad);
    REQUIRE(sum_grad(grad) == Approx(0.0));
}

// ── Affine / Global ──────────────────────────────────────────────────────────

TEST_CASE("Gradient affine global: identical sequences", "[gradient][affine][global]") {
    auto p = unit_params(1.0, 10.0);
    Aligner<GapModel::Affine, AlignMode::Global> al;
    al.set_problem("ACDE", "ACDE", p);
    al.compute_viterbi();
    AlignParams grad{};
    al.hard_grad(grad);

    REQUIRE(al.score() == Approx(4.0));
    REQUIRE(sum_grad(grad) == Approx(4.0));
}

TEST_CASE("Gradient affine global: gap produces no gradient", "[gradient][affine][global]") {
    auto p = unit_params(1.0, 10.0);
    Aligner<GapModel::Affine, AlignMode::Global> al;
    al.set_problem("A", "", p);
    al.compute_viterbi();
    AlignParams grad{};
    al.hard_grad(grad);
    REQUIRE(sum_grad(grad) == Approx(0.0));
}

// ── Affine / Local ───────────────────────────────────────────────────────────

TEST_CASE("Gradient affine local: finds and counts local match", "[gradient][affine][local]") {
    auto p = unit_params(1.0, 10.0);
    Aligner<GapModel::Affine, AlignMode::Local> al;
    al.set_problem("ADE", "MMMADEM", p);
    al.compute_viterbi();
    AlignParams grad{};
    al.hard_grad(grad);
    REQUIRE(sum_grad(grad) == Approx(3.0));
}
