#pragma once

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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
//
// ── The matrix track: position-dependent substitution ────────────────────────
//
// An AlignParams holds K >= 1 substitution matrices, called SLOTS, all over one
// alphabet.  Slot 0 is `matrix` and is the only one that exists unless slots are
// added, so a K == 1 AlignParams is bit-for-bit what it always was: the vector
// below stays empty and allocates nothing.
//
// A problem may then carry a TRACK — one slot index per position of sequence A —
// and position i of A is scored by slot track[i].  Position 5 of A can be scored
// by a transmembrane matrix and position 600 by a loop matrix, in one alignment.
// Sequence B has no track; the column axis is always the plain alphabet.
//
// WHY SLOTS AND A TRACK, rather than one matrix literally per position: the
// gradient has to come back as a POINT IN THIS SAME PARAMETER SPACE, because that
// is the whole purpose of the library.  With slots it does — the gradient is K
// matrices, and every position sharing a slot ties into the same K*N*N learnable
// parameters.  The literal per-position reading is not lost, it is the special
// case K == len(A) with track = [0, 1, 2, ...]; it is supported and it is simply
// the one where nothing is tied.  See aligner.hpp for what that costs.
//
// Arithmetic requires the same K on both sides as well as the same alphabet: a
// gradient over 3 slots is not a direction in a 1-slot space, and silently adding
// its slot 0 would be a wrong answer rather than an error.

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

    // Multi-slot construction: `ms` must be non-empty and all over one alphabet.
    explicit AlignParams(const std::vector<SubstMatrix>& ms,
                         double go_a = 0.0, double ge_a = 0.0,
                         double go_b = 0.0, double ge_b = 0.0)
        : matrix(ms.empty()
                     ? throw std::invalid_argument(
                           "nwgrad: AlignParams needs at least one substitution matrix")
                     : ms.front()),
          gap_open_a(go_a), gap_extend_a(ge_a),
          gap_open_b(go_b), gap_extend_b(ge_b)
    {
        for (size_t k = 1; k < ms.size(); ++k) add_matrix(ms[k]);
    }

    // All-zero params over `alpha` — the identity for gradient accumulation.
    explicit AlignParams(const Alphabet& alpha) : matrix(alpha) {}

    // All-zero params over the same alphabet as `p`, WITH THE SAME NUMBER OF
    // SLOTS.  The way to build an accumulator: it can be summed with any gradient
    // derived from `p`, which the slot count is now part of.
    static AlignParams zeros_like(const AlignParams& p) {
        AlignParams r(p.matrix.alphabet());
        r.extra_.assign(p.extra_.size(), SubstMatrix(p.matrix.alphabet()));
        return r;
    }

    const Alphabet& alphabet() const noexcept { return matrix.alphabet(); }

    // ── Slots ────────────────────────────────────────────────────────────────

    // K: always >= 1.  Slot 0 is `matrix`; slots 1..K-1 live in extra_.
    int matrix_count() const noexcept { return 1 + static_cast<int>(extra_.size()); }

    const SubstMatrix& matrix_at(int k) const {
        check_slot(k);
        return (k == 0) ? matrix : extra_[static_cast<size_t>(k) - 1];
    }
    SubstMatrix& matrix_at(int k) {
        check_slot(k);
        return (k == 0) ? matrix : extra_[static_cast<size_t>(k) - 1];
    }

    // Append a slot; returns its index.  Must be over the same alphabet — a track
    // indexes both by position, so a slot in a different order would silently
    // score different residues for the positions that select it.
    int add_matrix(const SubstMatrix& m) {
        if (&m.alphabet() != &matrix.alphabet())
            throw std::invalid_argument(
                "nwgrad: cannot add a matrix over alphabet \"" +
                m.alphabet().symbols() + "\" to params over \"" +
                matrix.alphabet().symbols() + "\"");
        extra_.push_back(m);
        return static_cast<int>(extra_.size());
    }

    // Reset every field to zero, keeping the alphabet, the slots and the
    // allocations.  Cheaper than reconstructing, and neither the alphabet nor the
    // slot count can drift.
    void zero() noexcept {
        matrix.zero();
        for (auto& e : extra_) e.zero();
        gap_open_a = gap_extend_a = gap_open_b = gap_extend_b = 0.0;
    }

    // ── Arithmetic ───────────────────────────────────────────────────────────
    // Combining params over different alphabets throws; see SubstMatrix.  So does
    // combining different slot counts — see the header note.

    AlignParams& operator+=(const AlignParams& o) {
        check_same_slots(o);
        matrix       += o.matrix;
        for (size_t k = 0; k < extra_.size(); ++k) extra_[k] += o.extra_[k];
        gap_open_a   += o.gap_open_a;
        gap_extend_a += o.gap_extend_a;
        gap_open_b   += o.gap_open_b;
        gap_extend_b += o.gap_extend_b;
        return *this;
    }

    AlignParams& operator*=(double s) noexcept {
        matrix       *= s;
        for (auto& e : extra_) e *= s;
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

private:
    void check_slot(int k) const {
        if (k < 0 || k >= matrix_count())
            throw std::out_of_range(
                "nwgrad: matrix slot " + std::to_string(k) + " out of range; params "
                "hold " + std::to_string(matrix_count()) + " matrix slot(s)");
    }

    void check_same_slots(const AlignParams& o) const {
        if (extra_.size() != o.extra_.size())
            throw std::invalid_argument(
                "nwgrad: cannot combine params with " + std::to_string(matrix_count()) +
                " matrix slot(s) and params with " + std::to_string(o.matrix_count()) +
                "; build the accumulator with zeros_like(), which copies the slot count");
    }

    // Slots 1..K-1.  Empty for the ordinary single-matrix case, so nothing about
    // a K == 1 AlignParams costs anything it did not cost before.
    std::vector<SubstMatrix> extra_;
};
