#pragma once

#include "subst_matrix.hpp"

// AlignParams bundles the substitution matrix with (potentially asymmetric)
// gap penalties.  "a" parameters apply to gaps in sequence A (A gets a '-',
// consuming B — the Y/VY state).  "b" parameters apply to gaps in sequence B
// (B gets a '-', consuming A — the X/VX state).
//
// All fields default to 0.0, which is the correct zero-initialiser for use
// as a gradient accumulator.  For alignment, supply non-zero gap costs.
//
// When an AlignParams holds a gradient (as returned by hard_grad / soft_grad),
// every field — the matrix entries and all four gap fields alike — is a
// derivative of the score with respect to that parameter.  Because the score
// *subtracts* the gap penalties, the gap fields of a gradient come out negative
// where the matrix fields come out positive; that asymmetry is the point, and it
// is what makes a single update rule correct for the whole struct.
//
// Addition and scalar multiplication operate element-wise over all fields,
// enabling gradient-descent update loops:
//   params = params - learning_rate * grad;

struct AlignParams {
    SubstMatrix matrix;
    double gap_open_a   = 0.0;
    double gap_extend_a = 0.0;
    double gap_open_b   = 0.0;
    double gap_extend_b = 0.0;

    AlignParams() = default;

    AlignParams operator+(const AlignParams& o) const {
        AlignParams r;
        r.matrix.order_ = matrix.order_;
        for (int i = 0; i < 256; ++i)
            for (int j = 0; j < 256; ++j)
                r.matrix.mat[i][j] = matrix.mat[i][j] + o.matrix.mat[i][j];
        r.gap_open_a   = gap_open_a   + o.gap_open_a;
        r.gap_extend_a = gap_extend_a + o.gap_extend_a;
        r.gap_open_b   = gap_open_b   + o.gap_open_b;
        r.gap_extend_b = gap_extend_b + o.gap_extend_b;
        return r;
    }

    AlignParams& operator+=(const AlignParams& o) {
        for (int i = 0; i < 256; ++i)
            for (int j = 0; j < 256; ++j)
                matrix.mat[i][j] += o.matrix.mat[i][j];
        gap_open_a   += o.gap_open_a;
        gap_extend_a += o.gap_extend_a;
        gap_open_b   += o.gap_open_b;
        gap_extend_b += o.gap_extend_b;
        return *this;
    }

    AlignParams operator*(double s) const {
        AlignParams r;
        r.matrix.order_ = matrix.order_;
        for (int i = 0; i < 256; ++i)
            for (int j = 0; j < 256; ++j)
                r.matrix.mat[i][j] = matrix.mat[i][j] * s;
        r.gap_open_a   = gap_open_a   * s;
        r.gap_extend_a = gap_extend_a * s;
        r.gap_open_b   = gap_open_b   * s;
        r.gap_extend_b = gap_extend_b * s;
        return r;
    }

    AlignParams& operator*=(double s) {
        for (int i = 0; i < 256; ++i)
            for (int j = 0; j < 256; ++j)
                matrix.mat[i][j] *= s;
        gap_open_a   *= s;
        gap_extend_a *= s;
        gap_open_b   *= s;
        gap_extend_b *= s;
        return *this;
    }

    AlignParams operator-() const { return *this * -1.0; }

    AlignParams operator-(const AlignParams& o) const { return *this + (-1.0 * o); }

    AlignParams& operator-=(const AlignParams& o) { return *this += (-1.0 * o); }

    friend AlignParams operator*(double s, const AlignParams& p) { return p * s; }
};
