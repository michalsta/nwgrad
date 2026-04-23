#pragma once

#include <array>
#include <cstring>
#include <stdexcept>
#include <string_view>

// Canonical amino-acid ordering used by the Python API (20×20 matrix).
static constexpr std::string_view AA_ORDER = "ACDEFGHIKLMNPQRSTVWY";

// SubstMatrix stores substitution scores in a 256×256 table indexed directly
// by ASCII character values, eliminating any char→index mapping in hot paths.
// Only the 20×20 amino-acid sub-block is populated; all other entries are 0.
// Asymmetric matrices are fully supported: score(a, b) and score(b, a) may differ.
struct SubstMatrix {
    double mat[256][256]{};  // zero-initialised

    SubstMatrix() noexcept = default;

    // Construct from a caller-supplied 20×20 row-major array (canonical AA order).
    explicit SubstMatrix(const double* src20x20) {
        for (int i = 0; i < 20; ++i) {
            for (int j = 0; j < 20; ++j) {
                unsigned char a = static_cast<unsigned char>(AA_ORDER[i]);
                unsigned char b = static_cast<unsigned char>(AA_ORDER[j]);
                mat[a][b] = src20x20[i * 20 + j];
            }
        }
    }

    // Direct score lookup — O(1), no branching.
    double score(char a, char b) const {
        return mat[static_cast<unsigned char>(a)][static_cast<unsigned char>(b)];
    }

    // Export back as a 20×20 row-major array (canonical AA order).
    void to_array(double* dst20x20) const {
        for (int i = 0; i < 20; ++i)
            for (int j = 0; j < 20; ++j)
                dst20x20[i * 20 + j] =
                    mat[static_cast<unsigned char>(AA_ORDER[i])]
                       [static_cast<unsigned char>(AA_ORDER[j])];
    }
};
