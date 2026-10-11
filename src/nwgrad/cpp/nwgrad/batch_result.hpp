#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

#include "align_params.hpp"

// One problem of a streaming align() call: the two sequences and, optionally, a guide
// (an empty guide_j means the automatic diagonal one when banded).
struct ProblemInstance {
    std::string_view seq_a;
    std::string_view seq_b;
    std::vector<int> guide_j;  // empty → trivial diagonal guide (band around main diagonal)
    // Tracks (aligner.hpp, "Tracks"); empty → the defaults (all set 0 / derived).
    std::vector<uint16_t> track;
    std::vector<uint16_t> gap_track;
};

// What a streaming align() returns: one score per problem, and the gradient summed over
// all of them (nothing per pair is kept).
struct BatchResult {
    std::vector<double> scores;
    AlignParams grad;

    explicit BatchResult(const Alphabet& alpha) : grad(alpha) {}
    // A zero gradient shaped like `params` (its alphabet and its number of sets).
    explicit BatchResult(const AlignParams& params) : grad(AlignParams::zeros_like(params)) {}
};
