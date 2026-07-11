#include "catch.hpp"
#include "align_params.hpp"
#include "aligner.hpp"
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

template<GapModel GM, AlignMode AM>
static double score_full(const AlignParams& p, const char* a, const char* b) {
    Aligner<GM, AM, AlignBand::Full> al;
    al.alloc_buf();
    al.set_problem(a, b, p);
    al.compute_viterbi();
    return al.score();
}

template<GapModel GM, AlignMode AM>
static double score_banded(const AlignParams& p, const char* a, const char* b, int band) {
    Aligner<GM, AM, AlignBand::GuideBanded> al;
    al.alloc_buf();
    al.set_problem(a, b, p, band);
    al.compute_viterbi();
    return al.score();
}

// ── Wide band: banded == full ─────────────────────────────────────────────────

TEST_CASE("Banded linear global: wide band matches full DP", "[banded][linear][global]") {
    auto p = unit_params(1.0);
    REQUIRE(score_banded<GapModel::Linear, AlignMode::Global>(p, "ACDE", "ACDE", 100)
         == Approx(score_full<GapModel::Linear, AlignMode::Global>(p, "ACDE", "ACDE")));
    REQUIRE(score_banded<GapModel::Linear, AlignMode::Global>(p, "ACDE", "ADE",  100)
         == Approx(score_full<GapModel::Linear, AlignMode::Global>(p, "ACDE", "ADE")));
    REQUIRE(score_banded<GapModel::Linear, AlignMode::Global>(p, "AAAA", "", 100)
         == Approx(score_full<GapModel::Linear, AlignMode::Global>(p, "AAAA", "")));
}

TEST_CASE("Banded affine global: wide band matches full DP", "[banded][affine][global]") {
    auto p = unit_params(1.0, 10.0);
    REQUIRE(score_banded<GapModel::Affine, AlignMode::Global>(p, "ACDE", "ACDE", 100)
         == Approx(score_full<GapModel::Affine, AlignMode::Global>(p, "ACDE", "ACDE")));
    REQUIRE(score_banded<GapModel::Affine, AlignMode::Global>(p, "AAA", "A", 100)
         == Approx(score_full<GapModel::Affine, AlignMode::Global>(p, "AAA", "A")));
}

TEST_CASE("Banded linear local: wide band matches full DP", "[banded][linear][local]") {
    auto p = unit_params(1.0);
    REQUIRE(score_banded<GapModel::Linear, AlignMode::Local>(p, "ADE", "MMMADEM", 100)
         == Approx(score_full<GapModel::Linear, AlignMode::Local>(p, "ADE", "MMMADEM")));
    REQUIRE(score_banded<GapModel::Linear, AlignMode::Local>(p, "AAAA", "CCCC", 100)
         == Approx(score_full<GapModel::Linear, AlignMode::Local>(p, "AAAA", "CCCC")));
}

TEST_CASE("Banded affine local: wide band matches full DP", "[banded][affine][local]") {
    auto p = unit_params(1.0, 10.0);
    REQUIRE(score_banded<GapModel::Affine, AlignMode::Local>(p, "ADE", "MMMADEM", 100)
         == Approx(score_full<GapModel::Affine, AlignMode::Local>(p, "ADE", "MMMADEM")));
}

// ── Tight band: correct when alignment stays in band ─────────────────────────

TEST_CASE("Banded linear global: identical sequences, band=0", "[banded][linear][global]") {
    auto p = unit_params(1.0);
    REQUIRE(score_banded<GapModel::Linear, AlignMode::Global>(p, "ACDE", "ACDE", 0) == Approx(4.0));
}

TEST_CASE("Banded linear global: one gap, band=1", "[banded][linear][global]") {
    auto p = unit_params(1.0);
    REQUIRE(score_banded<GapModel::Linear, AlignMode::Global>(p, "ACDE", "ADE", 1) == Approx(2.0));
}

TEST_CASE("Banded affine global: identical sequences, band=0", "[banded][affine][global]") {
    auto p = unit_params(1.0, 10.0);
    REQUIRE(score_banded<GapModel::Affine, AlignMode::Global>(p, "ACDE", "ACDE", 0) == Approx(4.0));
}

// ── Hard gradient through banded DP ──────────────────────────────────────────

static double sum_grad(const AlignParams& g) {
    double s = 0.0;
    for (int i = 0; i < 20; ++i)
        for (int j = 0; j < 20; ++j)
            s += g.matrix.at(i, j);
    return s;
}

TEST_CASE("Banded hard grad: identical sequences", "[banded][gradient]") {
    auto p = unit_params(1.0);
    Aligner<GapModel::Linear, AlignMode::Global, AlignBand::GuideBanded> al;
    al.alloc_buf();
    al.set_problem("ACDE", "ACDE", p, 0);
    al.compute_viterbi();
    AlignParams grad(Alphabet::protein());
    al.hard_grad(grad);
    REQUIRE(sum_grad(grad) == Approx(4.0));
    REQUIRE(grad.matrix.at(Alphabet::protein().index_of('A'), Alphabet::protein().index_of('A')) == Approx(1.0));
}

TEST_CASE("Banded hard grad: matches full DP grad", "[banded][gradient]") {
    auto pfull   = unit_params(0.5);
    auto pbanded = unit_params(0.5);

    Aligner<GapModel::Linear, AlignMode::Global, AlignBand::Full>        full;
    Aligner<GapModel::Linear, AlignMode::Global, AlignBand::GuideBanded> banded;

    full.alloc_buf();
    full.set_problem("ADE", "ACDE", pfull);
    full.compute_viterbi();
    AlignParams gfull(Alphabet::protein());
    full.hard_grad(gfull);

    banded.alloc_buf();
    banded.set_problem("ADE", "ACDE", pbanded, 2);
    banded.compute_viterbi();
    AlignParams gbanded(Alphabet::protein());
    banded.hard_grad(gbanded);

    REQUIRE(sum_grad(gfull) == Approx(sum_grad(gbanded)));
}

// ── Soft gradient through banded DP ──────────────────────────────────────────

TEST_CASE("Banded soft grad: wide band matches full for global linear", "[banded][soft]") {
    auto p = unit_params(1.0);

    Aligner<GapModel::Linear, AlignMode::Global, AlignBand::Full>        full;
    Aligner<GapModel::Linear, AlignMode::Global, AlignBand::GuideBanded> banded;

    full.alloc_buf();
    full.set_problem("ACDE", "ACDE", p);
    full.compute_forward_back();
    AlignParams gfull(Alphabet::protein());
    full.soft_grad(gfull);

    banded.alloc_buf();
    banded.set_problem("ACDE", "ACDE", p, 100);
    banded.compute_forward_back();
    AlignParams gbanded(Alphabet::protein());
    banded.soft_grad(gbanded);

    REQUIRE(full.log_z()    == Approx(banded.log_z()).epsilon(1e-9));
    REQUIRE(sum_grad(gfull) == Approx(sum_grad(gbanded)).epsilon(1e-9));
}

// ── guide_j_from_aligned utility ─────────────────────────────────────────────

TEST_CASE("guide_j_from_aligned: all matches", "[banded][guide]") {
    auto gj = guide_j_from_aligned("ACDE", "ACDE");
    REQUIRE(gj.size() == 5);
    for (int i = 0; i <= 4; ++i) REQUIRE(gj[i] == i);
}

TEST_CASE("guide_j_from_aligned: gap in b (deletion in b)", "[banded][guide]") {
    auto gj = guide_j_from_aligned("ACDE", "A-DE");
    REQUIRE(gj.size() == 5);
    REQUIRE(gj[0] == 0);
    REQUIRE(gj[1] == 1);
    REQUIRE(gj[2] == 1);
    REQUIRE(gj[3] == 2);
    REQUIRE(gj[4] == 3);
}

TEST_CASE("guide_j_from_aligned: gap in a (insertion in b)", "[banded][guide]") {
    auto gj = guide_j_from_aligned("A-DE", "ACDE");
    REQUIRE(gj.size() == 4);
    REQUIRE(gj[0] == 0);
    REQUIRE(gj[1] == 1);
    REQUIRE(gj[2] == 3);
    REQUIRE(gj[3] == 4);
}

// ── Guide-banded: score matches full DP when guide is correct ─────────────────

TEST_CASE("Guide-banded: correct guide, band=0, unequal sequences", "[banded][guide]") {
    auto p  = unit_params(1.0);
    auto gj = guide_j_from_aligned("ACDE", "A-DE");

    Aligner<GapModel::Linear, AlignMode::Global, AlignBand::Full>        full;
    Aligner<GapModel::Linear, AlignMode::Global, AlignBand::GuideBanded> guided;

    full.alloc_buf();
    full.set_problem("ACDE", "ADE", p);
    full.compute_viterbi();

    guided.alloc_buf();
    guided.set_problem("ACDE", "ADE", p, 0, gj);
    guided.compute_viterbi();

    REQUIRE(guided.score() == Approx(full.score()));
}

TEST_CASE("Guide-banded: wide band around trivial guide matches full DP", "[banded][guide]") {
    auto p = unit_params(1.0);

    Aligner<GapModel::Linear, AlignMode::Global, AlignBand::Full>        full;
    Aligner<GapModel::Linear, AlignMode::Global, AlignBand::GuideBanded> guided;

    full.alloc_buf();
    full.set_problem("MMMADE", "ADE", p);
    full.compute_viterbi();

    guided.alloc_buf();
    guided.set_problem("MMMADE", "ADE", p, 100);
    guided.compute_viterbi();

    REQUIRE(guided.score() == Approx(full.score()));
}

TEST_CASE("Guide-banded: endpoint always reachable for complete guide", "[banded][guide]") {
    auto p  = unit_params(1.0);
    auto gj = guide_j_from_aligned("ACDEFG", "--DE--");
    REQUIRE(gj.size() == 7);
    REQUIRE(gj[6] == 2);

    Aligner<GapModel::Linear, AlignMode::Global, AlignBand::Full>        full;
    Aligner<GapModel::Linear, AlignMode::Global, AlignBand::GuideBanded> guided;

    full.alloc_buf();
    full.set_problem("ACDEFG", "DE", p);
    full.compute_viterbi();

    guided.alloc_buf();
    guided.set_problem("ACDEFG", "DE", p, 0, gj);
    guided.compute_viterbi();

    REQUIRE(guided.score() == Approx(full.score()));
}
