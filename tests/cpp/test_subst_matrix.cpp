#include "catch.hpp"
#include "subst_matrix.hpp"

#include <stdexcept>

// Build a flat 20×20 identity-like matrix for predictable round-trip tests.
static std::array<double, 400> make_flat(double diag, double off) {
    std::array<double, 400> m{};
    for (int i = 0; i < 20; ++i)
        for (int j = 0; j < 20; ++j)
            m[i * 20 + j] = (i == j) ? diag : off;
    return m;
}

TEST_CASE("SubstMatrix round-trip: identity-like matrix", "[subst_matrix]") {
    auto src = make_flat(4.0, -1.0);
    SubstMatrix sm(src.data());

    std::array<double, 400> dst{};
    sm.to_array(dst.data());

    for (int i = 0; i < 400; ++i)
        REQUIRE(dst[i] == Approx(src[i]));
}

TEST_CASE("SubstMatrix score() uses ASCII index directly", "[subst_matrix]") {
    auto src = make_flat(5.0, -2.0);
    SubstMatrix sm(src.data());

    // Diagonal entries (same AA) should give 5.0
    for (char aa : AA_ORDER)
        REQUIRE(sm.score(aa, aa) == Approx(5.0));

    // Off-diagonal
    REQUIRE(sm.score('A', 'C') == Approx(-2.0));
    REQUIRE(sm.score('C', 'A') == Approx(-2.0));
}

TEST_CASE("SubstMatrix preserves asymmetric values", "[subst_matrix]") {
    std::array<double, 400> src{};
    src[0 * 20 + 1] = 3.0;  // A→C
    src[1 * 20 + 0] = 7.0;  // C→A (different)
    SubstMatrix sm(src.data());

    REQUIRE(sm.score('A', 'C') == Approx(3.0));
    REQUIRE(sm.score('C', 'A') == Approx(7.0));
}

TEST_CASE("SubstMatrix rejects characters outside the alphabet", "[subst_matrix]") {
    auto src = make_flat(1.0, 1.0);
    SubstMatrix sm(src.data());

    // '*' is not in AA_ORDER.  The 256x256 ASCII table used to have a cell for
    // it, so it scored a silent 0.0 -- an out-of-alphabet residue would quietly
    // align at no cost instead of being reported.  There is no such cell now,
    // and no such silence.
    REQUIRE_THROWS_AS(sm.score('*', 'A'), std::invalid_argument);
    REQUIRE_THROWS_AS(sm.score('A', '*'), std::invalid_argument);
    REQUIRE_THROWS_AS(sm.score('*', '*'), std::invalid_argument);

    // Case is significant: lowercase is out of alphabet like any other symbol.
    REQUIRE_THROWS_AS(sm.score('a', 'A'), std::invalid_argument);
}
