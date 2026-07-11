#pragma once

#include <stdexcept>

#include "alphabet.hpp"
#include "subst_matrix.hpp"

// AlignParams bundles the substitution matrix with (potentially asymmetric)
// gap penalties.  "a" parameters apply to gaps in sequence A (A gets a '-',
// consuming B — the Y/VY state).  "b" parameters apply to gaps in sequence B
// (B gets a '-', consuming A — the X/VX state).
//
// One type serves two roles: a point in parameter space, and a gradient — a
// direction in that space.  Addition and scalar multiplication operate
// element-wise over all fields, so the update rule reads
//
//   params = params - learning_rate * grad;
//
// When an AlignParams holds a gradient (as returned by hard_grad / soft_grad),
// every field — the matrix entries and all four gap fields alike — is a
// derivative of the score with respect to that parameter.  Because the score
// *subtracts* the gap penalties, the gap fields of a gradient come out negative
// where the matrix fields come out positive; that asymmetry is the point, and it
// is what makes a single update rule correct for the whole struct.
//
// An AlignParams always has an alphabet: there is no default constructor and no
// alphabet-less state.  A gradient accumulator is built with zeros_like(params),
// which fixes its alphabet up front.  Two AlignParams can be combined only if
// they share an alphabet, and since alphabets are interned that check is a
// pointer comparison.

struct AlignParams {
    SubstMatrix matrix;
    double gap_open_a   = 0.0;
    double gap_extend_a = 0.0;
    double gap_open_b   = 0.0;
    double gap_extend_b = 0.0;

    explicit AlignParams(const SubstMatrix& m,
                         double go_a = 0.0, double ge_a = 0.0,
                         double go_b = 0.0, double ge_b = 0.0)
        : matrix(m),
          gap_open_a(go_a), gap_extend_a(ge_a),
          gap_open_b(go_b), gap_extend_b(ge_b) {}

    // All-zero params over `alpha` — the identity for gradient accumulation.
    explicit AlignParams(const Alphabet& alpha) : matrix(alpha) {}

    // All-zero params over the same alphabet as `p`.  The way to build an
    // accumulator: it can be summed with any gradient derived from `p`.
    static AlignParams zeros_like(const AlignParams& p) {
        return AlignParams(p.matrix.alphabet());
    }

    const Alphabet& alphabet() const noexcept { return matrix.alphabet(); }

    // Reset every field to zero, keeping the alphabet and the allocation.
    // Cheaper than reconstructing, and the alphabet cannot drift.
    void zero() noexcept {
        matrix.zero();
        gap_open_a = gap_extend_a = gap_open_b = gap_extend_b = 0.0;
    }

    // ── Arithmetic ───────────────────────────────────────────────────────────
    // Combining params over different alphabets throws; see SubstMatrix.

    AlignParams& operator+=(const AlignParams& o) {
        matrix       += o.matrix;
        gap_open_a   += o.gap_open_a;
        gap_extend_a += o.gap_extend_a;
        gap_open_b   += o.gap_open_b;
        gap_extend_b += o.gap_extend_b;
        return *this;
    }

    AlignParams& operator*=(double s) noexcept {
        matrix       *= s;
        gap_open_a   *= s;
        gap_extend_a *= s;
        gap_open_b   *= s;
        gap_extend_b *= s;
        return *this;
    }

    AlignParams& operator-=(const AlignParams& o) { return *this += (-1.0 * o); }

    AlignParams operator+(const AlignParams& o) const {
        AlignParams r(*this);
        r += o;
        return r;
    }

    AlignParams operator*(double s) const {
        AlignParams r(*this);
        r *= s;
        return r;
    }

    AlignParams operator-() const { return *this * -1.0; }

    AlignParams operator-(const AlignParams& o) const { return *this + (-1.0 * o); }

    friend AlignParams operator*(double s, const AlignParams& p) { return p * s; }
};
