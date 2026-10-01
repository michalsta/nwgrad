#pragma once

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

struct SubstMatrix {
    // Zero matrix over `alpha`.  This is the correct starting point for a
    // gradient accumulator.
    explicit SubstMatrix(const Alphabet& alpha)
        : alpha_(&alpha),
          blk_(static_cast<size_t>(alpha.size()) * static_cast<size_t>(alpha.size()), 0.0) {}

    // Construct from a caller-supplied N x N row-major array over `alpha`.
    SubstMatrix(const double* src, const Alphabet& alpha)
        : alpha_(&alpha),
          blk_(static_cast<size_t>(alpha.size()) * static_cast<size_t>(alpha.size()))
    {
        const size_t n = static_cast<size_t>(alpha.size());
        for (size_t k = 0; k < n * n; ++k) blk_[k] = src[k];
    }

    // Construct from an N x N array plus a symbol string; interns the alphabet.
    explicit SubstMatrix(const double* src, std::string_view order = AA_ORDER)
        : SubstMatrix(src, Alphabet::get(order)) {}

    const Alphabet&    alphabet() const noexcept { return *alpha_; }
    int                size()     const noexcept { return alpha_->size(); }
    const std::string& order()    const noexcept { return alpha_->symbols(); }

    // ── Index-based access: the hot path ─────────────────────────────────────

    // Raw block, row-major, size() x size().  Cached by the Aligner at
    // set_problem() so the DP inner loop dereferences nothing else.
    const double* data() const noexcept { return blk_.data(); }
    double*       data()       noexcept { return blk_.data(); }

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

    // Export as an N x N row-major array in alphabet order.
    void to_array(double* dst) const {
        for (size_t k = 0; k < blk_.size(); ++k) dst[k] = blk_[k];
    }

    // Set every entry to zero, keeping the alphabet and the allocation.
    void zero() noexcept {
        for (auto& v : blk_) v = 0.0;
    }

    // ── Element-wise arithmetic ──────────────────────────────────────────────
    // Both operands must share an alphabet.  Since alphabets are interned, that
    // is a pointer comparison.

    SubstMatrix& operator+=(const SubstMatrix& o) {
        check_same_alphabet(o);
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

    const Alphabet*     alpha_;   // interned, immortal, never null
    std::vector<double> blk_;     // n x n, row-major
};
