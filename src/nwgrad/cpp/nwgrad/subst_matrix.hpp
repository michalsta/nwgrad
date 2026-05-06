#pragma once

#include <stdexcept>
#include <string>
#include <string_view>

// Canonical amino-acid ordering used as the default alphabet.
static constexpr std::string_view AA_ORDER = "ACDEFGHIKLMNPQRSTVWY";

// SubstMatrix stores substitution scores in a 256×256 table indexed directly
// by ASCII character values, eliminating any char→index mapping in hot paths.
// The active N×N sub-block is determined by the stored alphabet; all other
// entries are 0.  Asymmetric matrices are fully supported.
struct SubstMatrix {
    double mat[256][256]{};                  // zero-initialised
    std::string order_ = std::string(AA_ORDER);  // alphabet, length = N

    SubstMatrix() = default;

    // Construct from a caller-supplied N×N row-major array with given alphabet.
    // Default alphabet is the 20-AA canonical order (backward-compatible).
    explicit SubstMatrix(const double* src, std::string_view order = AA_ORDER)
        : order_(order) {
        int n = static_cast<int>(order.size());
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                unsigned char a = static_cast<unsigned char>(order[i]);
                unsigned char b = static_cast<unsigned char>(order[j]);
                mat[a][b] = src[i * n + j];
            }
    }

    int         size()  const noexcept { return static_cast<int>(order_.size()); }
    const std::string& order() const noexcept { return order_; }

    // Direct score lookup — O(1), no branching.
    double score(char a, char b) const {
        return mat[static_cast<unsigned char>(a)][static_cast<unsigned char>(b)];
    }

    // Export back as an N×N row-major array in alphabet order.
    void to_array(double* dst) const {
        int n = size();
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                dst[i * n + j] =
                    mat[static_cast<unsigned char>(order_[i])]
                       [static_cast<unsigned char>(order_[j])];
    }
};
