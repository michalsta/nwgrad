#pragma once

#include <cstddef>
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

// Per-pair tracks (aligner.hpp, "Tracks"): one run of set indices per pair, either
// nested vectors or one flat buffer cut by offsets (pair i: flat[off[i] .. off[i+1]),
// off has n + 1 entries).  An empty run is that pair's default; a null PairTracks
// means every pair's.  A view: the storage must outlive it.
struct PairTracks {
    const std::vector<std::vector<uint16_t>>* nested = nullptr;
    const uint16_t* flat = nullptr;
    const size_t* off = nullptr;
    size_t n = 0;

    PairTracks() = default;
    PairTracks(std::nullptr_t) {}
    PairTracks(const std::vector<std::vector<uint16_t>>* v) : nested(v), n(v ? v->size() : 0) {}
    PairTracks(const uint16_t* f, const size_t* o, size_t count) : flat(f), off(o), n(count) {}

    explicit operator bool() const noexcept { return nested != nullptr || flat != nullptr; }
    size_t size() const noexcept { return n; }
    std::span<const uint16_t> operator[](size_t i) const noexcept {
        if (nested) return std::span<const uint16_t>((*nested)[i]);
        return std::span<const uint16_t>(flat + off[i], off[i + 1] - off[i]);
    }
    // The largest entry of any pair (0 when there are none): one pass, vectorizable.
    uint16_t max_entry() const noexcept {
        uint16_t mx = 0;
        if (flat) {
            const size_t total = off[n] - off[0];
            const uint16_t* p = flat + off[0];
            for (size_t q = 0; q < total; ++q) mx = p[q] > mx ? p[q] : mx;
        } else if (nested) {
            for (const auto& t : *nested)
                for (const uint16_t v : t) mx = v > mx ? v : mx;
        }
        return mx;
    }
};

// The tracks of a streaming align() call, kept beside the problems rather than in them
// so an untracked call builds what it always built.
struct StreamTracks {
    PairTracks tracks, gap_tracks;
    std::span<const uint16_t> track(size_t i) const noexcept {
        return tracks ? tracks[i] : std::span<const uint16_t>();
    }
    std::span<const uint16_t> gap_track(size_t i) const noexcept {
        return gap_tracks ? gap_tracks[i] : std::span<const uint16_t>();
    }
    bool any() const noexcept { return static_cast<bool>(tracks) || static_cast<bool>(gap_tracks); }
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
