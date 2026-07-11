// Aligner precondition checks.
//
// The Aligner is a pipeline: set_problem() -> compute_viterbi() /
// compute_forward_back() -> score() / hard_grad() / soft_grad().  Calling a
// stage out of order throws std::logic_error with a message telling the caller
// what to run first.  Those guards were entirely uncovered — the Python bindings
// happen to call the stages in order, so nothing ever tripped them, leaving the
// misuse contract that header-only users rely on unverified.

#include "catch.hpp"
#include "align_params.hpp"
#include "aligner.hpp"

#include <array>

static AlignParams unit_params() {
    std::array<double, 400> src{};
    for (int i = 0; i < 20; ++i)
        src[i * 20 + i] = 1.0;

    AlignParams p(SubstMatrix(src.data()));
    p.gap_open_a   = 2.0;
    p.gap_extend_a = 1.0;
    p.gap_open_b   = 2.0;
    p.gap_extend_b = 1.0;
    return p;
}

using GlobalAffine = Aligner<GapModel::Affine, AlignMode::Global>;

TEST_CASE("Aligner: score() before any DP throws", "[aligner][preconditions]") {
    GlobalAffine al;
    REQUIRE_THROWS_AS(al.score(), std::logic_error);
}

TEST_CASE("Aligner: log_z() before forward-backward throws",
          "[aligner][preconditions]") {
    auto p = unit_params();

    GlobalAffine al;
    al.alloc_buf();
    al.set_problem("WKLM", "WKLM", p);

    REQUIRE_THROWS_AS(al.log_z(), std::logic_error);

    // Viterbi alone is still not enough: log Z needs the forward pass.
    al.compute_viterbi();
    REQUIRE_THROWS_AS(al.log_z(), std::logic_error);

    al.compute_forward_back();
    REQUIRE(al.log_z() >= al.score() - 1e-9);   // now it is available
}

TEST_CASE("Aligner: compute_viterbi() without set_problem() throws",
          "[aligner][preconditions]") {
    GlobalAffine al;
    al.alloc_buf();
    REQUIRE_THROWS_AS(al.compute_viterbi(), std::logic_error);
}

TEST_CASE("Aligner: compute_viterbi() without alloc_buf() throws",
          "[aligner][preconditions]") {
    auto p = unit_params();
    GlobalAffine al;
    al.set_problem("WKLM", "WKLM", p);
    REQUIRE_THROWS_AS(al.compute_viterbi(), std::logic_error);
}

TEST_CASE("Aligner: traceback before compute_viterbi() throws",
          "[aligner][preconditions]") {
    auto p = unit_params();
    GlobalAffine al;
    al.alloc_buf();
    al.set_problem("WKLM", "WKLM", p);

    REQUIRE_THROWS_AS(al.alignment(), std::logic_error);
    REQUIRE_THROWS_AS(al.aligned(),   std::logic_error);

    AlignParams grad(Alphabet::protein());
    REQUIRE_THROWS_AS(al.hard_grad(grad), std::logic_error);
}

TEST_CASE("Aligner: soft_grad() before forward-backward throws",
          "[aligner][preconditions]") {
    auto p = unit_params();
    GlobalAffine al;
    al.alloc_buf();
    al.set_problem("WKLM", "WKLM", p);
    al.compute_viterbi();   // the *other* DP does not satisfy soft_grad

    AlignParams grad(Alphabet::protein());
    REQUIRE_THROWS_AS(al.soft_grad(grad), std::logic_error);

    al.compute_forward_back();
    REQUIRE_NOTHROW(al.soft_grad(grad));
}

TEST_CASE("Aligner: score() reports whichever DP ran most recently",
          "[aligner][preconditions]") {
    // score() is documented to return the Viterbi score or log Z depending on
    // which pass ran last, so the order of the two calls has to matter.
    auto p = unit_params();

    GlobalAffine al;
    al.alloc_buf();
    al.set_problem("WKLMWK", "WKLM", p);

    al.compute_viterbi();
    const double viterbi = al.score();

    al.compute_forward_back();
    const double log_z = al.score();     // now the newest result is log Z
    REQUIRE(log_z == Approx(al.log_z()));

    // log Z sums over every alignment, so it strictly dominates the best one.
    REQUIRE(log_z >= Approx(viterbi));

    al.compute_viterbi();                // re-running Viterbi makes it newest again
    REQUIRE(al.score() == Approx(viterbi));
}
