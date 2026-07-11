// Aligner::alignment() — the traceback that returns the aligned (i,j) index pairs.
//
// The header documents alignment() as part of the public pipeline, but nothing
// called it: the Python bindings expose the gapped *strings* (aligned()), and no
// test touched the index-pair form.  Its whole affine traceback
// (traceback_affine -> traceback_affine_impl) was therefore dead to the suite.
//
// The two tracebacks are separate implementations of the same walk, so the
// strongest check is to hold them against each other: the index pairs from
// alignment() must be exactly the non-gap columns of aligned().

#include "catch.hpp"
#include "align_params.hpp"
#include "aligner.hpp"

#include <array>
#include <string>
#include <utility>
#include <vector>

static AlignParams path_params(double open_a, double ext_a,
                               double open_b, double ext_b) {
    std::array<double, 400> src{};
    for (int i = 0; i < 20; ++i)
        for (int j = 0; j < 20; ++j)
            src[i * 20 + j] = (i == j) ? 2.0 : -1.0;

    AlignParams p;
    p.matrix       = SubstMatrix(src.data());
    p.gap_open_a   = open_a;
    p.gap_extend_a = ext_a;
    p.gap_open_b   = open_b;
    p.gap_extend_b = ext_b;
    return p;
}

// The (i,j) pairs implied by a gapped alignment: every column where neither
// side is a '-', reported as indices into the original ungapped sequences.
static std::vector<std::pair<int,int>> pairs_from_gapped(const std::string& ga,
                                                         const std::string& gb) {
    std::vector<std::pair<int,int>> out;
    int i = 0, j = 0;
    for (size_t k = 0; k < ga.size(); ++k) {
        const bool gap_a = (ga[k] == '-');
        const bool gap_b = (gb[k] == '-');
        if (!gap_a && !gap_b) out.emplace_back(i, j);
        if (!gap_a) ++i;
        if (!gap_b) ++j;
    }
    return out;
}

template<GapModel GM, AlignMode AM>
static void check_path_matches_gapped_strings(const AlignParams& p,
                                              const std::string& a,
                                              const std::string& b) {
    Aligner<GM, AM> al;
    al.alloc_buf();
    al.set_problem(a.c_str(), b.c_str(), p);
    al.compute_viterbi();

    const auto path = al.alignment();
    const auto [ga, gb] = al.aligned();

    // alignment() reports absolute indices into a and b, while a *local*
    // aligned() only spans the local window — so its columns are offset by
    // wherever that window starts.  Anchor the two on the first matched pair.
    const auto windowed = pairs_from_gapped(ga, gb);
    REQUIRE(path.size() == windowed.size());

    if (!path.empty()) {
        const int off_i = path[0].first  - windowed[0].first;
        const int off_j = path[0].second - windowed[0].second;
        if constexpr (AM == AlignMode::Global)
            REQUIRE((off_i == 0 && off_j == 0));   // a global window starts at (0,0)
        REQUIRE(off_i >= 0);
        REQUIRE(off_j >= 0);

        for (size_t k = 0; k < path.size(); ++k)
            REQUIRE(path[k] == std::make_pair(windowed[k].first  + off_i,
                                              windowed[k].second + off_j));
    }

    // Tie the indices to the sequences themselves: each reported pair must name
    // the characters that the gapped strings actually aligned.
    for (size_t k = 0; k < path.size(); ++k) {
        REQUIRE(path[k].first  >= 0);
        REQUIRE(path[k].second >= 0);
        REQUIRE(path[k].first  < static_cast<int>(a.size()));
        REQUIRE(path[k].second < static_cast<int>(b.size()));
        if (k > 0) {
            REQUIRE(path[k].first  > path[k-1].first);   // strictly increasing
            REQUIRE(path[k].second > path[k-1].second);
        }
    }

    size_t k = 0;
    for (size_t col = 0; col < ga.size(); ++col) {
        if (ga[col] == '-' || gb[col] == '-') continue;
        REQUIRE(a[static_cast<size_t>(path[k].first)]  == ga[col]);
        REQUIRE(b[static_cast<size_t>(path[k].second)] == gb[col]);
        ++k;
    }
    REQUIRE(k == path.size());
}

static const std::string PA = "WWKKLLMMFFAAGG";
static const std::string PB = "WWKLLMMFFAACGG";

TEST_CASE("alignment(): linear/global path matches aligned()", "[aligner][path]") {
    check_path_matches_gapped_strings<GapModel::Linear, AlignMode::Global>(
        path_params(0.0, 1.5, 0.0, 0.5), PA, PB);
}

TEST_CASE("alignment(): linear/local path matches aligned()", "[aligner][path]") {
    check_path_matches_gapped_strings<GapModel::Linear, AlignMode::Local>(
        path_params(0.0, 1.5, 0.0, 0.5), PA, PB);
}

TEST_CASE("alignment(): affine/global path matches aligned()", "[aligner][path]") {
    check_path_matches_gapped_strings<GapModel::Affine, AlignMode::Global>(
        path_params(4.0, 0.5, 1.5, 2.0), PA, PB);
}

TEST_CASE("alignment(): affine/local path matches aligned()", "[aligner][path]") {
    check_path_matches_gapped_strings<GapModel::Affine, AlignMode::Local>(
        path_params(4.0, 0.5, 1.5, 2.0), PA, PB);
}

TEST_CASE("alignment(): identical sequences align position-for-position",
          "[aligner][path]") {
    auto p = path_params(4.0, 0.5, 1.5, 2.0);
    Aligner<GapModel::Affine, AlignMode::Global> al;
    al.alloc_buf();
    al.set_problem("WKLMFA", "WKLMFA", p);
    al.compute_viterbi();

    const auto path = al.alignment();
    REQUIRE(path.size() == 6);
    for (int k = 0; k < 6; ++k)
        REQUIRE(path[static_cast<size_t>(k)] == std::make_pair(k, k));
}

TEST_CASE("alignment(): a gap in A shows up as a skipped B index",
          "[aligner][path]") {
    // Cheap gaps in A, punitive in B: B is one longer, so the extra B character
    // must be absorbed by a gap in A, leaving a jump in the B index.
    auto p = path_params(0.5, 0.5, 20.0, 20.0);
    Aligner<GapModel::Affine, AlignMode::Global> al;
    al.alloc_buf();
    al.set_problem("WKLM", "WKXLM", p);
    al.compute_viterbi();

    const auto path = al.alignment();
    const auto [ga, gb] = al.aligned();

    REQUIRE(ga.find('-') != std::string::npos);   // the gap landed in A
    REQUIRE(gb.find('-') == std::string::npos);
    REQUIRE(path == pairs_from_gapped(ga, gb));

    // every A index is matched; exactly one B index (the inserted X) is skipped
    REQUIRE(path.size() == 4);
}

TEST_CASE("alignment(): affine traceback crosses a multi-character gap",
          "[aligner][path]") {
    // A long insertion in B exercises the X-state run inside traceback_affine.
    auto p = path_params(2.0, 0.5, 2.0, 0.5);
    Aligner<GapModel::Affine, AlignMode::Global> al;
    al.alloc_buf();
    al.set_problem("WKLMFFFFAA", "WKLMAA", p);
    al.compute_viterbi();

    const auto path = al.alignment();
    const auto [ga, gb] = al.aligned();
    REQUIRE(path == pairs_from_gapped(ga, gb));
    REQUIRE(gb.find("----") != std::string::npos);  // the FFFF run is gapped out
}

TEST_CASE("alignment(): empty on a local alignment with no positive match",
          "[aligner][path]") {
    auto p = path_params(10.0, 10.0, 10.0, 10.0);
    Aligner<GapModel::Affine, AlignMode::Local> al;
    al.alloc_buf();
    al.set_problem("WWWW", "KKKK", p);   // all mismatches, every extension negative
    al.compute_viterbi();

    REQUIRE(al.score() == Approx(0.0));
    REQUIRE(al.alignment().empty());
}
