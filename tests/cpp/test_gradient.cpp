#include "catch.hpp"
#include "aligner.hpp"

static BlosumMatrix unit_matrix() {
    std::array<double, 400> src{};
    for (int i = 0; i < 20; ++i)
        src[i * 20 + i] = 1.0;
    return BlosumMatrix(src.data());
}

static void zero_grad(double g[256][256]) {
    for (int i = 0; i < 256; ++i)
        for (int j = 0; j < 256; ++j)
            g[i][j] = 0.0;
}

static double sum_grad(const double g[256][256]) {
    double s = 0.0;
    for (int i = 0; i < 256; ++i)
        for (int j = 0; j < 256; ++j)
            s += g[i][j];
    return s;
}

// ── Linear / Global ──────────────────────────────────────────────────────────

TEST_CASE("Gradient linear global: identical sequences — all matches", "[gradient][linear][global]") {
    auto mat = unit_matrix();
    Aligner<GapModel::Linear, AlignMode::Global> al;
    al.set_problem("ACDE", "ACDE", mat, 1.0);
    al.compute_viterbi();
    double grad[256][256];
    zero_grad(grad);
    al.hard_grad(grad);

    REQUIRE(al.score() == Approx(4.0));
    REQUIRE(grad[(unsigned char)'A'][(unsigned char)'A'] == Approx(1.0));
    REQUIRE(grad[(unsigned char)'C'][(unsigned char)'C'] == Approx(1.0));
    REQUIRE(grad[(unsigned char)'D'][(unsigned char)'D'] == Approx(1.0));
    REQUIRE(grad[(unsigned char)'E'][(unsigned char)'E'] == Approx(1.0));
    REQUIRE(sum_grad(grad) == Approx(4.0));
}

TEST_CASE("Gradient linear global: gaps produce zero gradient contribution", "[gradient][linear][global]") {
    auto mat = unit_matrix();
    Aligner<GapModel::Linear, AlignMode::Global> al;
    al.set_problem("A", "", mat, 1.0);
    al.compute_viterbi();
    double grad[256][256];
    zero_grad(grad);
    al.hard_grad(grad);
    REQUIRE(sum_grad(grad) == Approx(0.0));
}

TEST_CASE("Gradient linear global: repeated calls accumulate", "[gradient][linear][global]") {
    auto mat = unit_matrix();
    Aligner<GapModel::Linear, AlignMode::Global> al;
    double grad[256][256];
    zero_grad(grad);

    al.set_problem("AA", "AA", mat, 1.0);
    al.compute_viterbi();
    al.hard_grad(grad);

    al.set_problem("AA", "AA", mat, 1.0);
    al.compute_viterbi();
    al.hard_grad(grad);  // second call adds on top

    REQUIRE(grad[(unsigned char)'A'][(unsigned char)'A'] == Approx(4.0));
}

TEST_CASE("Gradient linear global: one-gap alignment has correct count", "[gradient][linear][global]") {
    auto mat = unit_matrix();
    Aligner<GapModel::Linear, AlignMode::Global> al;
    // gap=0.5 so matching is always preferred; single inserted residue forces a gap
    al.set_problem("ADE", "ACDE", mat, 0.5);
    al.compute_viterbi();
    double grad[256][256];
    zero_grad(grad);
    al.hard_grad(grad);
    // "ADE" vs "ACDE": optimal is A-A, gap-C, D-D, E-E → 3 matched pairs
    REQUIRE(sum_grad(grad) == Approx(3.0));
}

// ── Linear / Local ───────────────────────────────────────────────────────────

TEST_CASE("Gradient linear local: only matching region contributes", "[gradient][linear][local]") {
    auto mat = unit_matrix();
    Aligner<GapModel::Linear, AlignMode::Local> al;
    al.set_problem("ADE", "MMMADEM", mat, 1.0);
    al.compute_viterbi();
    double grad[256][256];
    zero_grad(grad);
    al.hard_grad(grad);

    REQUIRE(sum_grad(grad) == Approx(3.0));
    REQUIRE(grad[(unsigned char)'A'][(unsigned char)'A'] == Approx(1.0));
    REQUIRE(grad[(unsigned char)'D'][(unsigned char)'D'] == Approx(1.0));
    REQUIRE(grad[(unsigned char)'E'][(unsigned char)'E'] == Approx(1.0));
    REQUIRE(grad[(unsigned char)'M'][(unsigned char)'M'] == Approx(0.0));
}

TEST_CASE("Gradient linear local: no match gives zero gradient", "[gradient][linear][local]") {
    auto mat = unit_matrix();
    Aligner<GapModel::Linear, AlignMode::Local> al;
    al.set_problem("AAAA", "CCCC", mat, 1.0);
    al.compute_viterbi();
    double grad[256][256];
    zero_grad(grad);
    al.hard_grad(grad);
    REQUIRE(sum_grad(grad) == Approx(0.0));
}

// ── Affine / Global ──────────────────────────────────────────────────────────

TEST_CASE("Gradient affine global: identical sequences", "[gradient][affine][global]") {
    auto mat = unit_matrix();
    Aligner<GapModel::Affine, AlignMode::Global> al;
    al.set_problem("ACDE", "ACDE", mat, 1.0, 10.0);
    al.compute_viterbi();
    double grad[256][256];
    zero_grad(grad);
    al.hard_grad(grad);

    REQUIRE(al.score() == Approx(4.0));
    REQUIRE(sum_grad(grad) == Approx(4.0));
}

TEST_CASE("Gradient affine global: gap produces no gradient", "[gradient][affine][global]") {
    auto mat = unit_matrix();
    Aligner<GapModel::Affine, AlignMode::Global> al;
    al.set_problem("A", "", mat, 1.0, 10.0);
    al.compute_viterbi();
    double grad[256][256];
    zero_grad(grad);
    al.hard_grad(grad);
    REQUIRE(sum_grad(grad) == Approx(0.0));
}

// ── Affine / Local ───────────────────────────────────────────────────────────

TEST_CASE("Gradient affine local: finds and counts local match", "[gradient][affine][local]") {
    auto mat = unit_matrix();
    Aligner<GapModel::Affine, AlignMode::Local> al;
    al.set_problem("ADE", "MMMADEM", mat, 1.0, 10.0);
    al.compute_viterbi();
    double grad[256][256];
    zero_grad(grad);
    al.hard_grad(grad);
    REQUIRE(sum_grad(grad) == Approx(3.0));
}
