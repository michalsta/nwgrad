#pragma once

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include "align_params.hpp"

// One problem of a streaming align() call: the two sequences and, optionally, a guide
// (an empty guide_j means the automatic diagonal one when banded).
struct ProblemInstance {
    std::string_view seq_a;
    std::string_view seq_b;
    std::vector<int> guide_j;  // empty → trivial diagonal guide (band around main diagonal)
};

// The tracks of a streaming align() call (aligner.hpp, "Tracks"), kept beside the
// problems rather than in them so an untracked call builds what it always built: null,
// or one entry per problem (an empty entry: that problem's default).
struct StreamTracks {
    const std::vector<std::vector<uint16_t>>* tracks = nullptr;
    const std::vector<std::vector<uint16_t>>* gap_tracks = nullptr;
    std::span<const uint16_t> track(size_t i) const noexcept {
        return tracks ? std::span<const uint16_t>((*tracks)[i]) : std::span<const uint16_t>();
    }
    std::span<const uint16_t> gap_track(size_t i) const noexcept {
        return gap_tracks ? std::span<const uint16_t>((*gap_tracks)[i]) : std::span<const uint16_t>();
    }
    bool any() const noexcept { return tracks || gap_tracks; }
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
