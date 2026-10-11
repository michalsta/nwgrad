#pragma once

#include <stdexcept>
#include <string>
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
// ── Parameter sets ───────────────────────────────────────────────────────────
// An AlignParams holds K >= 1 SETS, each a full substitution matrix plus the four
// gap fields.  A problem may carry a TRACK (one set index per residue of sequence A)
// and a GAP TRACK (one per DP row 0..m of A, for gaps in A); every alignment column
// is then scored by the set its row selects (see aligner.hpp, "Tracks").  Set 0 is
// what the plain fields hold — `matrix` block 0 and gap_open_a .. gap_extend_b — so
// every K = 1 use is unchanged; sets 1..K-1 live in the rest of the matrix stack
// (matrix.set_data(k)) and in more_gaps_.  Arithmetic needs equal K as well as equal
// alphabets, and zeros_like keeps K, so a gradient of a K-set problem is again K sets:
// the update rule above still reads params + lr * grad.

struct AlignParams {
    SubstMatrix matrix;
    double gap_open_a   = 0.0;
    double gap_extend_a = 0.0;
    double gap_open_b   = 0.0;
    double gap_extend_b = 0.0;

    // Gap fields of one set, in the order of the four members above.
    enum GapField : int { kOpenA = 0, kExtendA = 1, kOpenB = 2, kExtendB = 3 };

    explicit AlignParams(const SubstMatrix& m,
                         double go_a = 0.0, double ge_a = 0.0,
                         double go_b = 0.0, double ge_b = 0.0)
        : matrix(m),
          gap_open_a(go_a), gap_extend_a(ge_a),
          gap_open_b(go_b), gap_extend_b(ge_b),
          more_gaps_(4 * static_cast<size_t>(m.sets() - 1), 0.0) {}

    // A K-set params from a stack of K matrices and 4*K gap values, set by set
    // (go_a, ge_a, go_b, ge_b for set 0, then set 1, ...).
    AlignParams(const SubstMatrix& stack, const double* gaps)
        : matrix(stack),
          gap_open_a(gaps[0]), gap_extend_a(gaps[1]),
          gap_open_b(gaps[2]), gap_extend_b(gaps[3]),
          more_gaps_(gaps + 4, gaps + 4 * static_cast<size_t>(stack.sets())) {}

    // All-zero params over `alpha` (K sets) — the identity for gradient accumulation.
    explicit AlignParams(const Alphabet& alpha, int nsets = 1)
        : matrix(alpha, nsets), more_gaps_(4 * static_cast<size_t>(nsets - 1), 0.0) {}

    // All-zero params over the same alphabet, with the same number of sets, as `p`.
    // The way to build an accumulator: it can be summed with any gradient derived from `p`.
    static AlignParams zeros_like(const AlignParams& p) {
        return AlignParams(p.matrix.alphabet(), p.num_sets());
    }

    // K one-set params (same alphabet) stacked into one K-set params, in order.  Each
    // operand may itself hold several sets; they are concatenated.
    static AlignParams stack(const std::vector<AlignParams>& parts) {
        if (parts.empty())
            throw std::invalid_argument("nwgrad: AlignParams.stack() needs at least one set");
        AlignParams out = parts[0];
        for (size_t k = 1; k < parts.size(); ++k) {
            const AlignParams& p = parts[k];
            out.matrix.append_sets(p.matrix);   // checks the alphabet and the size limit
            for (int s = 0; s < p.num_sets(); ++s)
                for (int f = 0; f < 4; ++f) out.more_gaps_.push_back(p.gap(s, f));
        }
        return out;
    }

    const Alphabet& alphabet() const noexcept { return matrix.alphabet(); }
    int num_sets() const noexcept { return matrix.sets(); }

    // Gap field f (GapField) of set k.
    double gap(int k, int f) const noexcept {
        if (k == 0) {
            switch (f) {
                case kOpenA:   return gap_open_a;
                case kExtendA: return gap_extend_a;
                case kOpenB:   return gap_open_b;
                default:       return gap_extend_b;
            }
        }
        return more_gaps_[4 * static_cast<size_t>(k - 1) + static_cast<size_t>(f)];
    }
    double& gap_ref(int k, int f) noexcept {
        if (k == 0) {
            switch (f) {
                case kOpenA:   return gap_open_a;
                case kExtendA: return gap_extend_a;
                case kOpenB:   return gap_open_b;
                default:       return gap_extend_b;
            }
        }
        return more_gaps_[4 * static_cast<size_t>(k - 1) + static_cast<size_t>(f)];
    }

    // Set k as a one-set params.
    AlignParams set(int k) const {
        AlignParams out(matrix.slice(k));   // checks k
        out.gap_open_a = gap(k, kOpenA); out.gap_extend_a = gap(k, kExtendA);
        out.gap_open_b = gap(k, kOpenB); out.gap_extend_b = gap(k, kExtendB);
        return out;
    }
    // Replace set k with the one-set params `p` (same alphabet).
    void assign_set(int k, const AlignParams& p) {
        if (p.num_sets() != 1)
            throw std::invalid_argument("nwgrad: assign_set() takes a one-set AlignParams");
        matrix.assign_set(k, p.matrix);     // checks the alphabet and k
        gap_ref(k, kOpenA) = p.gap_open_a; gap_ref(k, kExtendA) = p.gap_extend_a;
        gap_ref(k, kOpenB) = p.gap_open_b; gap_ref(k, kExtendB) = p.gap_extend_b;
    }

    // Every gap value, set by set (4*K), as the K-set constructor takes them.
    std::vector<double> all_gaps() const {
        std::vector<double> g(4 * static_cast<size_t>(num_sets()));
        for (int k = 0; k < num_sets(); ++k)
            for (int f = 0; f < 4; ++f) g[4 * static_cast<size_t>(k) + static_cast<size_t>(f)] = gap(k, f);
        return g;
    }
    // The gap values of sets 1..K-1 (4 each), contiguous: for hashing and comparing.
    const std::vector<double>& more_gaps() const noexcept { return more_gaps_; }

    // Reset every field to zero, keeping the alphabet, the set count and the allocation.
    // Cheaper than reconstructing, and the alphabet cannot drift.
    void zero() noexcept {
        matrix.zero();
        gap_open_a = gap_extend_a = gap_open_b = gap_extend_b = 0.0;
        for (auto& v : more_gaps_) v = 0.0;
    }

    // ── Arithmetic ───────────────────────────────────────────────────────────
    // Combining params over different alphabets, or with different set counts, throws;
    // see SubstMatrix.

    AlignParams& operator+=(const AlignParams& o) {
        matrix       += o.matrix;   // checks alphabet and set count first
        gap_open_a   += o.gap_open_a;
        gap_extend_a += o.gap_extend_a;
        gap_open_b   += o.gap_open_b;
        gap_extend_b += o.gap_extend_b;
        for (size_t k = 0; k < more_gaps_.size(); ++k) more_gaps_[k] += o.more_gaps_[k];
        return *this;
    }

    AlignParams& operator*=(double s) noexcept {
        matrix       *= s;
        gap_open_a   *= s;
        gap_extend_a *= s;
        gap_open_b   *= s;
        gap_extend_b *= s;
        for (auto& v : more_gaps_) v *= s;
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
    std::vector<double> more_gaps_;   // sets 1..K-1: go_a, ge_a, go_b, ge_b each
};
