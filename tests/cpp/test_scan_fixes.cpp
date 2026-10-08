// ── Regressions from the 2026-10-07 package scan (scan-raport.md) ───────────────────
//
// B1: a failed string set_problem() freed A's codes under the old problem's spans.
// B6: a negative band slipped through every entry point that picks Full for band <= 0.
// B7: guide_j_[i] + band_ overflowed int for a huge band.
// B2: Aligner copies kept spans into the source's vectors (now: not copyable).
// B3: a batch borrowed its params; it now owns a copy, one per distinct value.
// B5: fit_alpha "converged" on NaN scores.

#include "catch.hpp"

#include <climits>
#include <cmath>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include "align_params.hpp"
#include "aligner.hpp"
#include "batch_engine.hpp"
#include "logistic/logistic.hpp"

TEST_CASE("B1: a rejected string problem leaves the previous one intact", "[scan]") {
    AlignParams p(Alphabet::dna());
    Aligner<GapModel::Linear, AlignMode::Global> al;
    al.alloc_buf();
    al.set_problem("AA", "AA", p);
    al.compute_viterbi();
    const auto before = al.aligned();
    // Encoding fails before anything is replaced, so the previous problem is intact
    // (under ASan this was a heap-use-after-free: A's codes were already freed).
    REQUIRE_THROWS_AS(al.set_problem(std::string(1000, 'A'), "!", p), std::invalid_argument);
    REQUIRE(al.aligned() == before);
    REQUIRE_THROWS_AS(al.set_problem("!", "AA", p), std::invalid_argument);
    REQUIRE(al.aligned() == before);
}

TEST_CASE("B6: a negative band is rejected on the Full path too", "[scan]") {
    AlignParams p(Alphabet::dna());
    Aligner<GapModel::Affine, AlignMode::Global> full;
    full.alloc_buf();
    REQUIRE_THROWS_AS(full.set_problem("AA", "AA", p, -1), std::invalid_argument);
}

TEST_CASE("B7: a huge band is full coverage, not signed overflow", "[scan]") {
    AlignParams p(Alphabet::dna());
    for (int i = 0; i < 4; ++i) p.matrix.at(i, i) = 2.0;
    p.gap_open_a = p.gap_open_b = 1.0;
    p.gap_extend_a = p.gap_extend_b = 1.0;
    Aligner<GapModel::Affine, AlignMode::Global, AlignBand::GuideBanded> big;
    Aligner<GapModel::Affine, AlignMode::Global> full;
    big.alloc_buf();
    full.alloc_buf();
    big.set_problem("ACGTAC", "AGTTACG", p, INT_MAX);   // UBSan: used to overflow in jhi
    full.set_problem("ACGTAC", "AGTTACG", p);
    big.compute_viterbi();
    full.compute_viterbi();
    REQUIRE(big.score() == full.score());
}

TEST_CASE("B2: Aligner is movable but not copyable", "[scan]") {
    using A = Aligner<GapModel::Affine, AlignMode::Global, AlignBand::Full, float>;
    STATIC_REQUIRE(!std::is_copy_constructible_v<A>);
    STATIC_REQUIRE(!std::is_copy_assignable_v<A>);
    STATIC_REQUIRE(std::is_move_constructible_v<A>);
    AlignParams p(Alphabet::dna());
    A a;
    a.alloc_buf();
    a.set_problem("ACGT", "ACGT", p);
    a.compute_viterbi();
    const auto before = a.aligned();
    A b(std::move(a));   // owned codes and the float block keep their buffers
    REQUIRE(b.aligned() == before);
}

TEST_CASE("B3: the batch owns its params, shared per distinct value", "[scan]") {
    BatchEngine<double, GapModel::Affine, AlignMode::Local> e(1, GradMode::Hard);
    AlignParams p1(Alphabet::dna()), p2(Alphabet::dna()), p3(Alphabet::dna());
    p1.gap_open_a = p2.gap_open_a = 2.0;
    p3.gap_open_a = 3.0;
    e.add_many({"ACGT"}, {"ACGT"}, p1);
    e.add_many({"ACGT"}, {"ACGT"}, p2);   // equal value: the same copy (the plan groups by address)
    e.add_many({"ACGT"}, {"ACGT"}, p3);
    REQUIRE(&e.params(0) != &p1);
    REQUIRE(&e.params(0) == &e.params(1));
    REQUIRE(&e.params(0) != &e.params(2));
    REQUIRE(e.params(2).gap_open_a == 3.0);
    p1.gap_open_a = 99.0;                  // the caller's object is not the batch's
    REQUIRE(e.params(0).gap_open_a == 2.0);
    e.set_params(p3);
    REQUIRE(&e.params(0) == &e.params(2));
    REQUIRE(e.params(1).gap_open_a == 3.0);
}

TEST_CASE("B5: fit_alpha rejects non-finite scores", "[scan]") {
    const double s[2] = {std::numeric_limits<double>::quiet_NaN(), 0.0};
    const double y[2] = {0.0, 1.0};
    REQUIRE_THROWS_AS(nwgrad::logistic::fit_alpha(s, y, 2, 0.0, 1), std::invalid_argument);
    const double ok[2] = {-1.0, 1.0};
    REQUIRE_THROWS_AS(nwgrad::logistic::fit_alpha(ok, y, 2, 0.0, 1, std::nan("")),
                      std::invalid_argument);
    REQUIRE(std::isfinite(nwgrad::logistic::fit_alpha(ok, y, 2, 0.0, 1)));
}
