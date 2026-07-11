// Arithmetic on AlignParams: +, +=, -, -=, unary -, *, *=, and scalar * params.
//
// align_params.hpp advertises these as the gradient-descent update API
// ("params = params - learning_rate * grad") and defaults every field to zero so
// an AlignParams can act as a gradient accumulator.  They were entirely
// unexercised by the C++ tests, which matters for header-only consumers that
// never go through the Python bindings.
//
// Each operator must act element-wise across the whole 256x256 substitution
// table and all four gap fields, and must carry the alphabet (order_) through.

#include "catch.hpp"
#include "align_params.hpp"

#include <array>

static AlignParams sample(double base) {
    std::array<double, 400> src{};
    for (int i = 0; i < 400; ++i)
        src[i] = base + i * 0.25;  // no two entries equal

    AlignParams p;
    p.matrix       = SubstMatrix(src.data());
    p.gap_open_a   = base + 1.0;
    p.gap_extend_a = base + 2.0;
    p.gap_open_b   = base + 3.0;
    p.gap_extend_b = base + 4.0;
    return p;
}

// Compare over the full 256x256 table, not just the 20x20 alphabet block, so a
// loop bound that misses the tail of the table is caught.
static void require_matrix_equals(const SubstMatrix& got, const SubstMatrix& want) {
    for (int i = 0; i < 256; ++i)
        for (int j = 0; j < 256; ++j)
            REQUIRE(got.mat[i][j] == Approx(want.mat[i][j]));
}

static void require_gaps(const AlignParams& p, double oa, double ea,
                         double ob, double eb) {
    REQUIRE(p.gap_open_a   == Approx(oa));
    REQUIRE(p.gap_extend_a == Approx(ea));
    REQUIRE(p.gap_open_b   == Approx(ob));
    REQUIRE(p.gap_extend_b == Approx(eb));
}

TEST_CASE("AlignParams: default construction is a zero accumulator", "[params]") {
    AlignParams z;
    require_gaps(z, 0.0, 0.0, 0.0, 0.0);
    for (int i = 0; i < 256; ++i)
        for (int j = 0; j < 256; ++j)
            REQUIRE(z.matrix.mat[i][j] == Approx(0.0));
}

TEST_CASE("AlignParams: operator+", "[params]") {
    auto x = sample(1.0), y = sample(10.0);
    auto r = x + y;

    for (int i = 0; i < 256; ++i)
        for (int j = 0; j < 256; ++j)
            REQUIRE(r.matrix.mat[i][j] == Approx(x.matrix.mat[i][j] + y.matrix.mat[i][j]));
    require_gaps(r, x.gap_open_a + y.gap_open_a, x.gap_extend_a + y.gap_extend_a,
                    x.gap_open_b + y.gap_open_b, x.gap_extend_b + y.gap_extend_b);
    REQUIRE(r.matrix.order_ == x.matrix.order_);   // alphabet must survive
}

TEST_CASE("AlignParams: operator+ leaves its operands untouched", "[params]") {
    auto x = sample(1.0), y = sample(10.0);
    const auto x0 = x, y0 = y;

    (void)(x + y);

    require_matrix_equals(x.matrix, x0.matrix);
    require_matrix_equals(y.matrix, y0.matrix);
    require_gaps(x, x0.gap_open_a, x0.gap_extend_a, x0.gap_open_b, x0.gap_extend_b);
    require_gaps(y, y0.gap_open_a, y0.gap_extend_a, y0.gap_open_b, y0.gap_extend_b);
}

TEST_CASE("AlignParams: operator+=", "[params]") {
    auto x = sample(1.0);
    const auto x0 = x;
    const auto y = sample(10.0);

    AlignParams& ref = (x += y);
    REQUIRE(&ref == &x);  // returns *this, not a copy

    for (int i = 0; i < 256; ++i)
        for (int j = 0; j < 256; ++j)
            REQUIRE(x.matrix.mat[i][j] == Approx(x0.matrix.mat[i][j] + y.matrix.mat[i][j]));
    require_gaps(x, x0.gap_open_a + y.gap_open_a, x0.gap_extend_a + y.gap_extend_a,
                    x0.gap_open_b + y.gap_open_b, x0.gap_extend_b + y.gap_extend_b);
}

TEST_CASE("AlignParams: operator* by a scalar", "[params]") {
    auto x = sample(1.0);
    auto r = x * 2.5;

    for (int i = 0; i < 256; ++i)
        for (int j = 0; j < 256; ++j)
            REQUIRE(r.matrix.mat[i][j] == Approx(x.matrix.mat[i][j] * 2.5));
    require_gaps(r, x.gap_open_a * 2.5, x.gap_extend_a * 2.5,
                    x.gap_open_b * 2.5, x.gap_extend_b * 2.5);
    REQUIRE(r.matrix.order_ == x.matrix.order_);
}

TEST_CASE("AlignParams: scalar on the left (free operator*)", "[params]") {
    auto x = sample(1.0);
    auto lhs = 2.5 * x;   // the form "learning_rate * grad" needs
    auto rhs = x * 2.5;

    require_matrix_equals(lhs.matrix, rhs.matrix);
    require_gaps(lhs, rhs.gap_open_a, rhs.gap_extend_a,
                      rhs.gap_open_b, rhs.gap_extend_b);
}

TEST_CASE("AlignParams: operator*=", "[params]") {
    auto x = sample(1.0);
    const auto x0 = x;

    AlignParams& ref = (x *= 3.0);
    REQUIRE(&ref == &x);

    for (int i = 0; i < 256; ++i)
        for (int j = 0; j < 256; ++j)
            REQUIRE(x.matrix.mat[i][j] == Approx(x0.matrix.mat[i][j] * 3.0));
    require_gaps(x, x0.gap_open_a * 3.0, x0.gap_extend_a * 3.0,
                    x0.gap_open_b * 3.0, x0.gap_extend_b * 3.0);
}

TEST_CASE("AlignParams: unary minus equals multiplication by -1", "[params]") {
    auto x = sample(1.0);
    auto neg = -x;
    auto mul = x * -1.0;

    require_matrix_equals(neg.matrix, mul.matrix);
    require_gaps(neg, -x.gap_open_a, -x.gap_extend_a, -x.gap_open_b, -x.gap_extend_b);
    REQUIRE(neg.matrix.order_ == x.matrix.order_);
}

TEST_CASE("AlignParams: operator-", "[params]") {
    auto x = sample(1.0), y = sample(10.0);
    auto r = x - y;

    for (int i = 0; i < 256; ++i)
        for (int j = 0; j < 256; ++j)
            REQUIRE(r.matrix.mat[i][j] == Approx(x.matrix.mat[i][j] - y.matrix.mat[i][j]));
    require_gaps(r, x.gap_open_a - y.gap_open_a, x.gap_extend_a - y.gap_extend_a,
                    x.gap_open_b - y.gap_open_b, x.gap_extend_b - y.gap_extend_b);
}

TEST_CASE("AlignParams: operator-=", "[params]") {
    auto x = sample(1.0);
    const auto x0 = x;
    const auto y = sample(10.0);

    AlignParams& ref = (x -= y);
    REQUIRE(&ref == &x);

    for (int i = 0; i < 256; ++i)
        for (int j = 0; j < 256; ++j)
            REQUIRE(x.matrix.mat[i][j] == Approx(x0.matrix.mat[i][j] - y.matrix.mat[i][j]));
    require_gaps(x, x0.gap_open_a - y.gap_open_a, x0.gap_extend_a - y.gap_extend_a,
                    x0.gap_open_b - y.gap_open_b, x0.gap_extend_b - y.gap_extend_b);
}

TEST_CASE("AlignParams: subtracting self yields zero", "[params]") {
    auto x = sample(1.0);
    auto z = x - x;

    for (int i = 0; i < 256; ++i)
        for (int j = 0; j < 256; ++j)
            REQUIRE(z.matrix.mat[i][j] == Approx(0.0));
    require_gaps(z, 0.0, 0.0, 0.0, 0.0);
}

TEST_CASE("AlignParams: the advertised gradient-descent step", "[params]") {
    // params = params - learning_rate * grad, verbatim from the header comment.
    auto params = sample(1.0);
    const auto before = params;
    const auto grad = sample(10.0);
    const double learning_rate = 0.1;

    params = params - learning_rate * grad;

    for (int i = 0; i < 256; ++i)
        for (int j = 0; j < 256; ++j)
            REQUIRE(params.matrix.mat[i][j] ==
                    Approx(before.matrix.mat[i][j] - learning_rate * grad.matrix.mat[i][j]));
    require_gaps(params,
                 before.gap_open_a   - learning_rate * grad.gap_open_a,
                 before.gap_extend_a - learning_rate * grad.gap_extend_a,
                 before.gap_open_b   - learning_rate * grad.gap_open_b,
                 before.gap_extend_b - learning_rate * grad.gap_extend_b);
}

TEST_CASE("AlignParams: accumulating into a default-constructed instance", "[params]") {
    // The documented accumulator pattern: start from zero, += each contribution.
    AlignParams total;
    const auto g1 = sample(1.0), g2 = sample(10.0), g3 = sample(100.0);

    total += g1;
    total += g2;
    total += g3;

    for (int i = 0; i < 256; ++i)
        for (int j = 0; j < 256; ++j)
            REQUIRE(total.matrix.mat[i][j] ==
                    Approx(g1.matrix.mat[i][j] + g2.matrix.mat[i][j] + g3.matrix.mat[i][j]));
    require_gaps(total,
                 g1.gap_open_a   + g2.gap_open_a   + g3.gap_open_a,
                 g1.gap_extend_a + g2.gap_extend_a + g3.gap_extend_a,
                 g1.gap_open_b   + g2.gap_open_b   + g3.gap_open_b,
                 g1.gap_extend_b + g2.gap_extend_b + g3.gap_extend_b);
}
