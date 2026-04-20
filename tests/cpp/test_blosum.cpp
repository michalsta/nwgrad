#include "catch.hpp"
#include "blosum.hpp"

// Build a flat 20×20 identity-like matrix for predictable round-trip tests.
static std::array<double, 400> make_flat(double diag, double off) {
    std::array<double, 400> m{};
    for (int i = 0; i < 20; ++i)
        for (int j = 0; j < 20; ++j)
            m[i * 20 + j] = (i == j) ? diag : off;
    return m;
}

TEST_CASE("BlosumMatrix round-trip: identity-like matrix", "[blosum]") {
    auto src = make_flat(4.0, -1.0);
    BlosumMatrix bm(src.data());

    std::array<double, 400> dst{};
    bm.to_array(dst.data());

    for (int i = 0; i < 400; ++i)
        REQUIRE(dst[i] == Approx(src[i]));
}

TEST_CASE("BlosumMatrix score() uses ASCII index directly", "[blosum]") {
    auto src = make_flat(5.0, -2.0);
    BlosumMatrix bm(src.data());

    // Diagonal entries (same AA) should give 5.0
    for (char aa : AA_ORDER)
        REQUIRE(bm.score(aa, aa) == Approx(5.0));

    // Off-diagonal
    REQUIRE(bm.score('A', 'C') == Approx(-2.0));
    REQUIRE(bm.score('C', 'A') == Approx(-2.0));  // symmetry
}

TEST_CASE("BlosumMatrix enforces symmetry on construction", "[blosum]") {
    // Build an asymmetric flat array; constructor should symmetrise.
    std::array<double, 400> src{};
    src[0 * 20 + 1] = 3.0;  // A→C
    src[1 * 20 + 0] = 7.0;  // C→A  (different)
    BlosumMatrix bm(src.data());

    // Both directions should be the same value (constructor uses v for both)
    // The constructor writes mat[a][b] = v and mat[b][a] = v for each (i,j),
    // so the last write for pair (0,1) and (1,0) wins; here i=0,j=1 writes 3.0
    // and then i=1,j=0 overwrites with 7.0.  Both should equal the final write.
    REQUIRE(bm.score('A', 'C') == bm.score('C', 'A'));
}

TEST_CASE("BlosumMatrix non-AA entries are zero", "[blosum]") {
    auto src = make_flat(1.0, 1.0);
    BlosumMatrix bm(src.data());
    // '*' is not in AA_ORDER
    REQUIRE(bm.score('*', 'A') == Approx(0.0));
    REQUIRE(bm.score('A', '*') == Approx(0.0));
    REQUIRE(bm.score('*', '*') == Approx(0.0));
}
