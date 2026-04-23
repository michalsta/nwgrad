#include "catch.hpp"
#include "aligner.hpp"

static SubstMatrix unit_matrix() {
    std::array<double, 400> src{};
    for (int i = 0; i < 20; ++i)
        src[i * 20 + i] = 1.0;
    return SubstMatrix(src.data());
}

// Score helpers
template<GapModel GM, AlignMode AM>
static double score_full(const SubstMatrix& mat, const char* a, const char* b,
                          double gap_ext, double gap_open = 0.0) {
    Aligner<GM, AM, AlignBand::Full> al;
    al.set_problem(a, b, mat, gap_ext, gap_open);
    al.compute_viterbi();
    return al.score();
}

template<GapModel GM, AlignMode AM>
static double score_banded(const SubstMatrix& mat, const char* a, const char* b,
                             double gap_ext, double gap_open, int band) {
    Aligner<GM, AM, AlignBand::GuideBanded> al;
    al.set_problem(a, b, mat, gap_ext, gap_open, band);
    al.compute_viterbi();
    return al.score();
}

// ── Wide band: banded == full ─────────────────────────────────────────────────

TEST_CASE("Banded linear global: wide band matches full DP", "[banded][linear][global]") {
    auto mat = unit_matrix();
    REQUIRE(score_banded<GapModel::Linear, AlignMode::Global>(mat, "ACDE", "ACDE", 1.0, 0.0, 100)
         == Approx(score_full <GapModel::Linear, AlignMode::Global>(mat, "ACDE", "ACDE", 1.0)));
    REQUIRE(score_banded<GapModel::Linear, AlignMode::Global>(mat, "ACDE", "ADE",  1.0, 0.0, 100)
         == Approx(score_full <GapModel::Linear, AlignMode::Global>(mat, "ACDE", "ADE",  1.0)));
    REQUIRE(score_banded<GapModel::Linear, AlignMode::Global>(mat, "AAAA", "", 2.0, 0.0, 100)
         == Approx(score_full <GapModel::Linear, AlignMode::Global>(mat, "AAAA", "", 2.0)));
}

TEST_CASE("Banded affine global: wide band matches full DP", "[banded][affine][global]") {
    auto mat = unit_matrix();
    REQUIRE(score_banded<GapModel::Affine, AlignMode::Global>(mat, "ACDE", "ACDE", 1.0, 10.0, 100)
         == Approx(score_full <GapModel::Affine, AlignMode::Global>(mat, "ACDE", "ACDE", 1.0, 10.0)));
    REQUIRE(score_banded<GapModel::Affine, AlignMode::Global>(mat, "AAA", "A", 1.0, 10.0, 100)
         == Approx(score_full <GapModel::Affine, AlignMode::Global>(mat, "AAA", "A",    1.0, 10.0)));
}

TEST_CASE("Banded linear local: wide band matches full DP", "[banded][linear][local]") {
    auto mat = unit_matrix();
    REQUIRE(score_banded<GapModel::Linear, AlignMode::Local>(mat, "ADE", "MMMADEM", 1.0, 0.0, 100)
         == Approx(score_full <GapModel::Linear, AlignMode::Local>(mat, "ADE", "MMMADEM", 1.0)));
    REQUIRE(score_banded<GapModel::Linear, AlignMode::Local>(mat, "AAAA", "CCCC", 1.0, 0.0, 100)
         == Approx(score_full <GapModel::Linear, AlignMode::Local>(mat, "AAAA", "CCCC", 1.0)));
}

TEST_CASE("Banded affine local: wide band matches full DP", "[banded][affine][local]") {
    auto mat = unit_matrix();
    REQUIRE(score_banded<GapModel::Affine, AlignMode::Local>(mat, "ADE", "MMMADEM", 1.0, 10.0, 100)
         == Approx(score_full <GapModel::Affine, AlignMode::Local>(mat, "ADE", "MMMADEM", 1.0, 10.0)));
}

// ── Tight band: correct when alignment stays in band ─────────────────────────

TEST_CASE("Banded linear global: identical sequences, band=0", "[banded][linear][global]") {
    auto mat = unit_matrix();
    // No gaps needed → band=0 (main diagonal only) is sufficient.
    REQUIRE(score_banded<GapModel::Linear, AlignMode::Global>(mat, "ACDE", "ACDE", 1.0, 0.0, 0)
         == Approx(4.0));
}

TEST_CASE("Banded linear global: one gap, band=1", "[banded][linear][global]") {
    auto mat = unit_matrix();
    // "ACDE" vs "ADE": one insertion → needs band >= 1
    REQUIRE(score_banded<GapModel::Linear, AlignMode::Global>(mat, "ACDE", "ADE", 1.0, 0.0, 1)
         == Approx(2.0));
}

TEST_CASE("Banded affine global: identical sequences, band=0", "[banded][affine][global]") {
    auto mat = unit_matrix();
    REQUIRE(score_banded<GapModel::Affine, AlignMode::Global>(mat, "ACDE", "ACDE", 1.0, 10.0, 0)
         == Approx(4.0));
}

// ── Hard gradient through banded DP ──────────────────────────────────────────

static double sum_grad(const double g[256][256]) {
    double s = 0.0;
    for (int i = 0; i < 256; ++i)
        for (int j = 0; j < 256; ++j)
            s += g[i][j];
    return s;
}

TEST_CASE("Banded hard grad: identical sequences", "[banded][gradient]") {
    auto mat = unit_matrix();
    Aligner<GapModel::Linear, AlignMode::Global, AlignBand::GuideBanded> al;
    al.set_problem("ACDE", "ACDE", mat, 1.0, 0.0, 0);
    al.compute_viterbi();
    double grad[256][256]{};
    al.hard_grad(grad);
    REQUIRE(sum_grad(grad) == Approx(4.0));
    REQUIRE(grad[(unsigned char)'A'][(unsigned char)'A'] == Approx(1.0));
}

TEST_CASE("Banded hard grad: matches full DP grad", "[banded][gradient]") {
    auto mat = unit_matrix();

    Aligner<GapModel::Linear, AlignMode::Global, AlignBand::Full>   full;
    Aligner<GapModel::Linear, AlignMode::Global, AlignBand::GuideBanded> banded;

    full.set_problem("ADE", "ACDE", mat, 0.5);
    full.compute_viterbi();
    double gfull[256][256]{};
    full.hard_grad(gfull);

    banded.set_problem("ADE", "ACDE", mat, 0.5, 0.0, 2);
    banded.compute_viterbi();
    double gbanded[256][256]{};
    banded.hard_grad(gbanded);

    REQUIRE(sum_grad(gfull) == Approx(sum_grad(gbanded)));
}

// ── Soft gradient through banded DP ──────────────────────────────────────────

TEST_CASE("Banded soft grad: wide band matches full for global linear", "[banded][soft]") {
    auto mat = unit_matrix();

    Aligner<GapModel::Linear, AlignMode::Global, AlignBand::Full>   full;
    Aligner<GapModel::Linear, AlignMode::Global, AlignBand::GuideBanded> banded;

    full.set_problem("ACDE", "ACDE", mat, 1.0);
    full.compute_forward_back();
    double gfull[256][256]{};
    full.soft_grad(gfull);

    banded.set_problem("ACDE", "ACDE", mat, 1.0, 0.0, 100);
    banded.compute_forward_back();
    double gbanded[256][256]{};
    banded.soft_grad(gbanded);

    REQUIRE(full.log_z() == Approx(banded.log_z()).epsilon(1e-9));
    REQUIRE(sum_grad(gfull) == Approx(sum_grad(gbanded)).epsilon(1e-9));
}

// ── guide_j_from_aligned utility ─────────────────────────────────────────────

TEST_CASE("guide_j_from_aligned: all matches", "[banded][guide]") {
    // "ACDE" vs "ACDE" — 4 match columns, guide_j = [0,1,2,3,4]
    auto gj = guide_j_from_aligned("ACDE", "ACDE");
    REQUIRE(gj.size() == 5);
    for (int i = 0; i <= 4; ++i) REQUIRE(gj[i] == i);
}

TEST_CASE("guide_j_from_aligned: gap in b (deletion in b)", "[banded][guide]") {
    // "ACDE" aligned to "A-DE": a gap in b at position 1 → i advances, j stays
    // columns: A↔A, C↔-, D↔D, E↔E
    // guide_j: [0, 1, 1, 2, 3]
    auto gj = guide_j_from_aligned("ACDE", "A-DE");
    REQUIRE(gj.size() == 5);
    REQUIRE(gj[0] == 0);
    REQUIRE(gj[1] == 1);
    REQUIRE(gj[2] == 1);  // after consuming 'C' of a, j is still 1 (gap in b)
    REQUIRE(gj[3] == 2);
    REQUIRE(gj[4] == 3);
}

TEST_CASE("guide_j_from_aligned: gap in a (insertion in b)", "[banded][guide]") {
    // "A-DE" aligned to "ACDE": gap in a at position 1 → j advances, i stays
    // guide_j for a = [0, 1, 3, 4]  (a has 3 non-gap chars)
    auto gj = guide_j_from_aligned("A-DE", "ACDE");
    REQUIRE(gj.size() == 4);
    REQUIRE(gj[0] == 0);
    REQUIRE(gj[1] == 1);
    REQUIRE(gj[2] == 3);  // after A (j=1), gap in a advances j to 2, then D→j=3
    REQUIRE(gj[3] == 4);
}

// ── Guide-banded: score matches full DP when guide is correct ─────────────────

TEST_CASE("Guide-banded: correct guide, band=0, unequal sequences", "[banded][guide]") {
    auto mat = unit_matrix();
    // "ACDE" vs "ADE": optimal global is A-ADE, score = 3 - 1*gap = 2
    // Guide: align "ACDE" to "A-DE" → guide_j = [0,1,1,2,3]
    auto gj = guide_j_from_aligned("ACDE", "A-DE");

    Aligner<GapModel::Linear, AlignMode::Global, AlignBand::Full>       full;
    Aligner<GapModel::Linear, AlignMode::Global, AlignBand::GuideBanded> guided;

    full.set_problem("ACDE", "ADE", mat, 1.0);
    full.compute_viterbi();

    guided.set_problem("ACDE", "ADE", mat, 1.0, 0.0, 0, gj);
    guided.compute_viterbi();

    REQUIRE(guided.score() == Approx(full.score()));
}

TEST_CASE("Guide-banded: wide band around trivial guide matches full DP", "[banded][guide]") {
    auto mat = unit_matrix();
    // Trivial guide (default empty → diagonal) with large band = full DP equivalent
    Aligner<GapModel::Linear, AlignMode::Global, AlignBand::Full>       full;
    Aligner<GapModel::Linear, AlignMode::Global, AlignBand::GuideBanded> guided;

    full.set_problem("MMMADE", "ADE", mat, 1.0);
    full.compute_viterbi();

    guided.set_problem("MMMADE", "ADE", mat, 1.0, 0.0, 100);
    guided.compute_viterbi();

    REQUIRE(guided.score() == Approx(full.score()));
}

TEST_CASE("Guide-banded: endpoint always reachable for complete guide", "[banded][guide]") {
    auto mat = unit_matrix();
    // Very different lengths — diagonal band would need band >= 4, but guide-band with
    // the correct guide works with band=0.
    // "ACDEFG" vs "DE": guide "AC-DE-FG" / "--DE--" (just use the right alignment)
    // guide_j_from_aligned of the actual optimal alignment
    auto gj = guide_j_from_aligned("ACDEFG", "--DE--");
    // a has 6 non-gap chars; b has 2 non-gap chars
    // columns: A↔-, C↔-, D↔D, E↔E, F↔-, G↔-
    // guide_j: after A: j=0, after C: j=0, after D: j=1, after E: j=2, after F: j=2, after G: j=2
    REQUIRE(gj.size() == 7);
    REQUIRE(gj[6] == 2);  // endpoint: m=6, guide_j[6]=2=n, always in-band

    Aligner<GapModel::Linear, AlignMode::Global, AlignBand::Full>       full;
    Aligner<GapModel::Linear, AlignMode::Global, AlignBand::GuideBanded> guided;

    full.set_problem("ACDEFG", "DE", mat, 1.0);
    full.compute_viterbi();

    guided.set_problem("ACDEFG", "DE", mat, 1.0, 0.0, 0, gj);
    guided.compute_viterbi();

    REQUIRE(guided.score() == Approx(full.score()));
}
