#pragma once

#include <algorithm>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "alphabet.hpp"

// Canonical amino-acid ordering, the default alphabet.
static constexpr std::string_view AA_ORDER = "ACDEFGHIKLMNPQRSTVWY";

// SubstMatrix stores an N x N block of substitution scores, indexed by alphabet
// position rather than by ASCII value.  The alphabet is an interned Alphabet;
// the matrix holds a non-owning pointer to it, which is safe because interned
// alphabets are immortal.
//
// The block is dense over the alphabet and nothing else exists: for DNA it is
// 16 doubles.  (The previous 256x256 ASCII-indexed table was 512 KiB whether or
// not you used four symbols, which made every gradient 512 KiB too.)
//
// Asymmetric matrices are fully supported: the row alphabet and the column
// alphabet are the same, but M[a][b] need not equal M[b][a].
//
// The DP never calls score(char, char).  Sequences are encoded to indices once,
// at the boundary, and the inner loop reads data()[ia * n + ib].
//
// A SubstMatrix may hold a STACK of K >= 1 matrices over one alphabet (sets()), K
// blocks of N x N back to back: block k at data() + k*N*N.  This is how AlignParams
// keeps K parameter sets (see align_params.hpp): the stacked block is exactly what a
// tracked DP indexes, row (k*N + a), with no copy.  Everything that predates the stack
// reads block 0 (at(), score(), to_array()), so a K = 1 matrix behaves as it always did.

struct SubstMatrix {
    // Zero matrix (or a stack of `nsets` zero matrices) over `alpha`.  This is the
    // correct starting point for a gradient accumulator.
    explicit SubstMatrix(const Alphabet& alpha, int nsets = 1)
        : alpha_(&alpha), nsets_(check_sets(nsets)),
          blk_(block_size(alpha) * static_cast<size_t>(nsets_), 0.0) {}

    // Construct from caller-supplied row-major data over `alpha`: one N x N block, or
    // `nsets` of them back to back.
    SubstMatrix(const double* src, const Alphabet& alpha, int nsets = 1)
        : alpha_(&alpha), nsets_(check_sets(nsets)),
          blk_(block_size(alpha) * static_cast<size_t>(nsets_))
    {
        for (size_t k = 0; k < blk_.size(); ++k) blk_[k] = src[k];
    }

    // Construct from an N x N array plus a symbol string; interns the alphabet.
    explicit SubstMatrix(const double* src, std::string_view order = AA_ORDER)
        : SubstMatrix(src, Alphabet::get(order)) {}

    const Alphabet&    alphabet() const noexcept { return *alpha_; }
    int                size()     const noexcept { return alpha_->size(); }
    const std::string& order()    const noexcept { return alpha_->symbols(); }
    // Number of stacked matrices, K >= 1.
    int                sets()     const noexcept { return nsets_; }

    // ── Index-based access: the hot path ─────────────────────────────────────

    // Raw block, row-major, size() x size() — the whole stack when sets() > 1, block k
    // at data() + k*size()*size().  Cached by the Aligner at set_problem() so the DP
    // inner loop dereferences nothing else.
    const double* data() const noexcept { return blk_.data(); }
    double*       data()       noexcept { return blk_.data(); }
    const double* set_data(int k) const noexcept { return blk_.data() + static_cast<size_t>(k) * block_size(*alpha_); }
    double*       set_data(int k)       noexcept { return blk_.data() + static_cast<size_t>(k) * block_size(*alpha_); }

    // Matrix k of the stack as a one-matrix SubstMatrix.
    SubstMatrix slice(int k) const {
        if (k < 0 || k >= nsets_)
            throw std::out_of_range("nwgrad: matrix set " + std::to_string(k) + " out of range (K = " +
                                    std::to_string(nsets_) + ")");
        return SubstMatrix(set_data(k), *alpha_);
    }
    // Replace matrix k of the stack with `m` (one matrix, same alphabet).
    void assign_set(int k, const SubstMatrix& m) {
        check_same_alphabet(m);
        if (m.nsets_ != 1)
            throw std::invalid_argument("nwgrad: assign_set() takes a single matrix");
        if (k < 0 || k >= nsets_)
            throw std::out_of_range("nwgrad: matrix set " + std::to_string(k) + " out of range (K = " +
                                    std::to_string(nsets_) + ")");
        std::copy(m.blk_.begin(), m.blk_.end(), set_data(k));
    }
    // The stack of this matrix followed by the stack of `o` (same alphabet).
    void append_sets(const SubstMatrix& o) {
        check_same_alphabet(o);
        check_sets(nsets_ + o.nsets_);
        blk_.insert(blk_.end(), o.blk_.begin(), o.blk_.end());
        nsets_ += o.nsets_;
    }

    double& at(int i, int j) noexcept {
        return blk_[static_cast<size_t>(i) * static_cast<size_t>(alpha_->size()) +
                    static_cast<size_t>(j)];
    }
    double at(int i, int j) const noexcept {
        return blk_[static_cast<size_t>(i) * static_cast<size_t>(alpha_->size()) +
                    static_cast<size_t>(j)];
    }

    // ── Character-based access: convenience, not for the DP ───────────────────

    // Throws if either character is outside the alphabet.
    double score(char a, char b) const {
        int ia = alpha_->index_of(a);
        int ib = alpha_->index_of(b);
        if (ia < 0) throw_not_in_alphabet(a);
        if (ib < 0) throw_not_in_alphabet(b);
        return at(ia, ib);
    }

    // Export matrix 0 as an N x N row-major array in alphabet order.
    void to_array(double* dst) const {
        const size_t nn = block_size(*alpha_);
        for (size_t k = 0; k < nn; ++k) dst[k] = blk_[k];
    }
    // Export the whole stack: sets() blocks of N x N.
    void to_array_all(double* dst) const {
        for (size_t k = 0; k < blk_.size(); ++k) dst[k] = blk_[k];
    }

    // Set every entry to zero, keeping the alphabet and the allocation.
    void zero() noexcept {
        for (auto& v : blk_) v = 0.0;
    }

    // ── Element-wise arithmetic ──────────────────────────────────────────────
    // Both operands must share an alphabet and a stack height.  Since alphabets are
    // interned, the first is a pointer comparison.

    SubstMatrix& operator+=(const SubstMatrix& o) {
        check_same_alphabet(o);
        if (nsets_ != o.nsets_)
            throw std::invalid_argument(
                "nwgrad: cannot combine parameters with different numbers of sets (" +
                std::to_string(nsets_) + " vs " + std::to_string(o.nsets_) + ")");
        for (size_t k = 0; k < blk_.size(); ++k) blk_[k] += o.blk_[k];
        return *this;
    }

    SubstMatrix& operator*=(double s) noexcept {
        for (auto& v : blk_) v *= s;
        return *this;
    }

private:
    void check_same_alphabet(const SubstMatrix& o) const {
        if (alpha_ != o.alpha_)
            throw std::invalid_argument(
                "nwgrad: cannot combine matrices over different alphabets (\"" +
                alpha_->symbols() + "\" vs \"" + o.alpha_->symbols() + "\")");
    }

    [[noreturn]] void throw_not_in_alphabet(char c) const {
        std::string msg = "nwgrad: character '";
        msg += c;
        msg += "' is not in alphabet \"" + alpha_->symbols() + "\"";
        throw std::invalid_argument(msg);
    }

    static size_t block_size(const Alphabet& a) noexcept {
        return static_cast<size_t>(a.size()) * static_cast<size_t>(a.size());
    }
    // A set index is stored as uint16_t in tracks, so a stack holds at most 65536.
    static int check_sets(int k) {
        if (k < 1 || k > kMaxSets)
            throw std::invalid_argument("nwgrad: the number of parameter sets must be in [1, " +
                                        std::to_string(kMaxSets) + "], got " + std::to_string(k));
        return k;
    }

public:
    static constexpr int kMaxSets = 65536;

private:
    const Alphabet*     alpha_;   // interned, immortal, never null
    int                 nsets_ = 1;
    std::vector<double> blk_;     // nsets_ blocks of n x n, row-major
};
