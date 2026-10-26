#pragma once

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <cstdlib>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "align_params.hpp"

enum class GapModel  { Linear, Affine };
enum class AlignMode { Global, Local  };
enum class AlignBand { Full,   GuideBanded };

// How the traceback recovers a cell's predecessor.
//
//   Scores    retain VM/VX/VY for the whole DP and RE-DERIVE the argmax by repeating
//             the forward pass's comparison chain.  12 B/cell (float32).
//   Pointers  record the predecessors in one byte per cell (the three states' codes
//             packed) during the fill and simply FOLLOW them; scores then need only two
//             rolling rows.  1 B/cell (3 until 2026-10-25) — and measured 1.4-2.2x faster, because the footprint is what
//             drives the page-fault cost that dominates the memory-bound regime.
//
// The two are bit-identical — same score, alignment, guide_j and gradient, ties
// included — because Pointers records exactly the argmax Scores would re-derive.
// Affine Full only; the linear and banded paths always keep their score tables.
//
//   Hirschberg  divide and conquer in O(n) memory: sweep to the middle row from both
//               ends keeping only rolling rows, join them to find where the optimal
//               path crosses, recurse on the two halves.  Nothing above the base case
//               is ever materialized, so the footprint is a few rows rather than a
//               table — ~0.9 MB for a 36,000 aa self-pair against 3.9 GB for Pointers.
//               It pays ~2x the cell work for that (each level re-sweeps its half).
//
// *** Hirschberg is NOT bit-exact with the other two, by construction. ***
//
// The join picks a midpoint argmax, and that choice cannot reproduce the backward-greedy
// M>X>Y tie-break the other two share, because the tie-break depends on the rows below
// the split — which Hirschberg has already discarded.  Where the optimum is unique the
// answers agree exactly; where alignments tie, it returns a DIFFERENT optimal path, hence
// a valid but different subgradient (path score == Viterbi score, always).  BUT: a pair
// no longer than hb_cutoff never splits, and is run AS Pointers (see run_viterbi), so it
// IS bit-exact — divergence is confined to pairs longer than hb_cutoff.  Affine Full Global
// only; explicitly asking for it on Local / linear / banded THROWS rather than silently
// running a different algorithm than requested.
//
//   Default    a per-problem SENTINEL, never stored on an Aligner: it resolves at compile
//              time to a Hirschberg mode for affine+global+full (the case Hirschberg
//              implements) and to Pointers for everything else.  This is what "traceback
//              defaults to Hirschberg" means without breaking Local/linear/banded, which
//              have no Hirschberg variant.  It resolves further on the SCALAR TYPE —
//              HirschbergPmax at T=float, Hirschberg at T=double — so the two precisions
//              do not merely differ in width, they run different carries; see kDefaultTb
//              for why.  SeqPair/SeqPairBatch leave the Aligner's own default in place
//              when handed this.
//
//   HirschbergPmax
//              Hirschberg with the VY carry computed by the CLOSED-FORM PREFIX MAX
//              (VY[c] = prefixmax_k(open[k] + k*ge_a) - (c-1)*ge_a) instead of the serial
//              `- ge_a` chain + lazy-F.  It is the one mode in this library that can
//              return a SUBOPTIMAL path, because the ramp k*ge_a is added and then
//              subtracted again and the round-trip costs a rounding proportional to the
//              column index.  Read that sentence before selecting it: no other mode here
//              trades optimality, and this one buys speed with it.  It is nonetheless the
//              DEFAULT AT T=float (see kDefaultTb) — at that precision the ramp is
//              two orders of magnitude below float32's own rounding, so the trade is
//              already paid for; at T=double it is strictly opt-in.  Everything else
//              about it is Hirschberg — same recursion, same
//              exact base case below hb_cutoff, same exact row 0 — so pairs that never
//              split are unaffected.  It remains bit-identical scalar-vs-simd across
//              every ISA level (the closed form depends only on the absolute column
//              index, not on the vector width).  See hb_kernel_impl.inl.
//
//   Grid       Pointers' result — score, path, guide, gradient, ties included — from one
//              score-only forward pass that saves a grid of rows and columns (every
//              state), then a walk back from (m, n) that re-fills, with direction codes,
//              only the blocks the path crosses, recursively down to small blocks
//              (viterbi_affine_grid).  O(k (m + n)) memory and ~1x the cell work
//              (the re-filled blocks are ~2/k of the cells per level).  Bit-identical to
//              Pointers because every re-filled cell is computed from the full fill's
//              exact values by the same expressions.  Implemented for affine Global Full,
//              untracked, floating T, opens >= 0, on a simd level with the kernel, and
//              pairs longer than hb_cutoff; everything else runs AS Pointers (the same
//              result by definition, so no fallback can change an answer).
enum class TracebackMode { Scores, Pointers, Hirschberg, HirschbergPmax, Grid, Default };

// How compute_forward_back() evaluates log Z and the expected counts.
//   Scaled (default): probability space with exact power-of-two row rescaling and the
//     gradient fused into the backward pass — no transcendental call per cell.  Throws
//     std::domain_error when a pair's dynamic range does not fit a double (see
//     run_fwdbwd and soft_kernel_impl.inl); it never returns a silently degraded result.
//   ScaledOrLog: Scaled, falling back per problem to Log where Scaled would throw.
//   Log: the original log-space recurrences (lse per cell).  Unlimited range, slow.
// The soft path is NOT bit-exact across modes, ISAs or compilers; it is tested with
// tolerances.  The Viterbi/hard path is unaffected by any of this.
enum class SoftImpl { Scaled, ScaledOrLog, Log };

// Reassociating reductions for the scaled soft path only (needs -fopenmp-simd, which
// the build sets; without it the pragma is ignored and the loop stays scalar).
#define NWGRAD_PRAGMA_(x) _Pragma(#x)
#define NWGRAD_SOFT_SIMD_SUM(...) NWGRAD_PRAGMA_(omp simd reduction(+:__VA_ARGS__))

// Both Hirschberg modes share the whole divide-and-conquer driver — the recursion, the
// join, the base case, the memory story — and differ only in which carry the SWEEP uses.
// Everything structural therefore asks this rather than naming one of them.
inline constexpr bool is_hirschberg(TracebackMode t) noexcept {
    return t == TracebackMode::Hirschberg || t == TracebackMode::HirschbergPmax;
}

// Where `auto` resolves to a Hirschberg mode with a pointer budget (Aligner::
// kAutoPointerBudget), a pair of up to this many cells runs as Pointers -- the same fill,
// table and traceback as traceback="pointers", so bit-exact with it -- and splits only
// above it, where the 1 B/cell code table grows past what a thread should hold.
inline constexpr std::size_t kAutoHbPointerCells = std::size_t(1) << 25;

// Whether a pair splits -- runs the Hirschberg recursion -- rather than the Pointers fill:
// every Hirschberg mode splits past hb_cutoff rows, except, under `auto` for a type with a
// pointer budget, a pair within that budget (kAutoHbPointerCells; Global double 2x that).  The aligner's dispatch
// and the batch engine's planning (which pairs may share a fill) ask the same question.
// auto_cells: the budget that applies (Aligner::kAutoPointerCells under `auto`), 0 = none.
inline constexpr bool hb_splits(TracebackMode tb, std::size_t auto_cells, int cutoff,
                                std::size_t la, std::size_t lb) noexcept {
    if (!is_hirschberg(tb) || la <= static_cast<std::size_t>(cutoff)) return false;
    return !(auto_cells != 0 && (la + 1) * (lb + 1) <= auto_cells);
}

// Which Viterbi backend fills the DP tables is a runtime field (backend_, below), one of
// the unified vocabulary in simd_levels.hpp: "scalar_fallback" (the original plain
// kernel), "auto" (defer to the global default = best simd), or a named simd level
// ("sse2"/"sse4"/"avx2"/"avx512"/"neon").  Every simd level is *bit-exact* with the scalar one —
// identical tables to the last bit — which is what lets the exact-float-equality
// tracebacks keep working unchanged; so the backend is a speed knob, never a correctness
// one.  It selects the Viterbi (hence hard_grad) path only; forward-backward and soft_grad
// are the same shared code either way.  Runtime, not a template parameter: the branch is
// taken once per compute_viterbi() and amortized over m*n cells.  Backends are encoded as
// ints (kBackendAuto / kBackendScalar / a SimdLevel index) so an aligner can store one.

// ── Utility: convert a pair of aligned strings to a guide_j vector ────────────
//
// a_aligned, b_aligned: equal-length strings with '-' for gaps.
// Returns a vector of length (m+1) where m = non-gap chars in a_aligned.
// guide_j[i] = the column j in the DP table after consuming i characters of a.
// Pass the result to Aligner::set_problem when using AlignBand::GuideBanded.
//
// When no explicit guide is given to set_problem, a proportional guide
// guide_j[i] = round(i * n / m) is used, tracking the main diagonal of the
// rectangular DP matrix.

inline std::vector<int> guide_j_from_aligned(std::string_view a_aligned,
                                              std::string_view b_aligned) {
    if (a_aligned.size() != b_aligned.size())
        throw std::invalid_argument("aligned strings must have equal length");
    std::vector<int> gj;
    int j = 0;
    gj.push_back(j);
    for (size_t c = 0; c < a_aligned.size(); ++c) {
        bool a_gap = (a_aligned[c] == '-');
        bool b_gap = (b_aligned[c] == '-');
        if (a_gap && b_gap)
            throw std::invalid_argument("aligned strings have a column with two gaps");
        if (!a_gap && !b_gap) { ++j; gj.push_back(j); }  // match/mismatch
        else if (a_gap)         ++j;                       // gap in a: j advances
        else                    gj.push_back(j);           // gap in b: i advances
    }
    return gj;
}

// ── log-sum-exp helpers ───────────────────────────────────────────────────────

static inline double lse2(double a, double b) noexcept {
    if (a == -std::numeric_limits<double>::infinity()) return b;
    if (b == -std::numeric_limits<double>::infinity()) return a;
    if (a > b) return a + std::log1p(std::exp(b - a));
    return b + std::log1p(std::exp(a - b));
}

static inline double lse3(double a, double b, double c) noexcept {
    return lse2(lse2(a, b), c);
}

// DpBuffer (the DP table storage) now lives in its own header so the leveled kernel
// TUs can include it without the whole Aligner.
#include "dp_buffer.hpp"

// Runtime per-ISA-level dispatch table for the leveled affine kernels.
#include "simd_levels.hpp"

// ── Integer scalar types: checked parameter casts and the largest step ──────────
//
// An integer T takes integral parameters only (the user quantizes; a silent truncation
// of 0.5 to 0 would be the alternative), each within +-nw_int_cap, and every problem's
// a-priori bound (m + n + 2 + 64) * step_max must fit the cap (scalar_traits.hpp);
// step_max is the largest magnitude one move can add — max |s| over the matrix stack,
// |go| + |ge| per side over the sets.  nw_int_prepare casts K sets of `p` (the whole
// stack, K*4 gaps) with those checks and reports step_max; a batch runs it once per
// params and hands the results to every pair's set_problem(), a standalone aligner
// runs it per problem.
template <class T>
T nw_int_checked(double x, const char* what) {
    static_assert(nw_is_int<T>);
    if (!(x == std::nearbyint(x)) || !(std::fabs(x) <= static_cast<double>(nw_int_cap<T>)))
        throw std::invalid_argument(
            "nwgrad: " + std::string(what) + " " + std::to_string(x) + " is not an integer within +-" +
            std::to_string(static_cast<long long>(nw_int_cap<T>)) + " (the " + nw_type_name<T>() +
            " scalar type takes integral parameters only; quantize them first)");
    return static_cast<T>(x);
}
template <class T>
void nw_int_prepare(const AlignParams& p, int K, std::vector<T>& stack, std::vector<T>& gaps,
                    T& step_max) {
    static_assert(nw_is_int<T>);
    auto abs_t = [](T v) noexcept { return v < T(0) ? static_cast<T>(-v) : v; };
    const std::size_t nn = static_cast<std::size_t>(p.matrix.size()) * p.matrix.size() *
                           static_cast<std::size_t>(K);
    stack.resize(nn);
    T smax = T(0);
    const double* src = p.matrix.data();
    for (std::size_t k = 0; k < nn; ++k) {
        const T v = nw_int_checked<T>(src[k], "substitution score");
        stack[k] = v;
        smax = std::max(smax, abs_t(v));
    }
    gaps.resize(4 * static_cast<std::size_t>(K));
    const long long cap = static_cast<long long>(nw_int_cap<T>);
    for (int k = 0; k < K; ++k) {
        T* g = gaps.data() + 4 * static_cast<std::size_t>(k);
        g[AlignParams::kOpenA]   = nw_int_checked<T>(p.gap(k, AlignParams::kOpenA), "gap_open_a");
        g[AlignParams::kExtendA] = nw_int_checked<T>(p.gap(k, AlignParams::kExtendA), "gap_extend_a");
        g[AlignParams::kOpenB]   = nw_int_checked<T>(p.gap(k, AlignParams::kOpenB), "gap_open_b");
        g[AlignParams::kExtendB] = nw_int_checked<T>(p.gap(k, AlignParams::kExtendB), "gap_extend_b");
        // |go| + |ge| <= 2 * cap fits a long long; clamped to the cap (the bound check
        // then refuses all but the smallest problems).
        const long long sa = static_cast<long long>(abs_t(g[0])) + abs_t(g[1]);
        const long long sb = static_cast<long long>(abs_t(g[2])) + abs_t(g[3]);
        smax = std::max(smax, static_cast<T>(std::min(cap, std::max(sa, sb))));
    }
    step_max = smax;
}

// The per-pair scaled soft kernels at the baseline ISA (scalar_fallback, header-only
// builds, levels without their own copy).  The level TUs compile the same file again
// with their -march; see soft_kernel_impl.inl.
namespace nwgrad_soft_base {
#include "soft_kernel_impl.inl"
}

// ── Aligner ───────────────────────────────────────────────────────────────────
//
// Pipeline API (call in order):
//   1. set_problem(seq_a, seq_b, params[, band[, guide_j]])
//   2a. compute_viterbi()       — max-based DP; enables score(), alignment(), hard_grad()
//   2b. compute_forward_back()  — lse forward+backward; enables log_z(), soft_grad()
//   3. score() / log_z() / alignment() / hard_grad() / soft_grad()
//
// Gap model and alignment mode are fixed at compile time via template parameters.
// AlignParams carries asymmetric gap penalties: gap_{open,extend}_{a,b} where
// suffix _a applies to gaps in sequence A (Y/VY state, consuming B) and
// suffix _b applies to gaps in sequence B (X/VX state, consuming A).
//
// Each of 2a/2b/3 has an overload that accepts a DpBuffer& to use instead of
// the internal own_buf_.  These overloads do NOT touch viterbi_done_/fwdbwd_done_
// and are intended for callers that manage their own buffer lifetime (e.g. a
// per-thread DpBuffer in a batch worker).  The scalar results (score, log_z,
// best_i/j/tbl) are always written to the Aligner regardless of which buffer is used.
//
// AlignBand::Full (default): full (m+1)×(n+1) DP, no restriction.
//
// AlignBand::GuideBanded: band of half-width `band` centered on a reference alignment.
//   guide_j[i] = the column the guide visits after consuming i chars of a.
//   If guide_j is omitted (or empty), a proportional diagonal guide
//   guide_j[i] = round(i * n / m) is used.  Unlike a literal guide_j[i] = i, this
//   tracks the main diagonal of the rectangular DP, so it stays centred for
//   unequal-length sequences without requiring band >= |m-n|.
//   For a custom guide, compute it with guide_j_from_aligned(a_aligned, b_aligned);
//   the endpoint (m, n) is always in-band as long as the guide is a complete alignment.
//
// In GuideBanded mode the full (m+1)×(n+1) table is still allocated (peak memory is
// unchanged), but the per-call init fill and the DP/backward loops are all restricted
// to the band, so the work per re-alignment is O(m·band) rather than O(m·n).  Only the
// band region (plus a one-column margin and adjacent-row overlap, see band_row_span)
// is initialised to NEG_INF; band-edge reads therefore always land on an initialised
// cell and cannot corrupt max/lse operations with stale data from a previous problem.

// A tracked problem's shared query profile (one row per distinct substitution row it
// uses) is kept only while it fits in this many bytes; past it the kernels build each DP
// row's profile row as they reach it — O(n) memory, one gather pass per row, the same
// cell work as the profile when no two rows share a row.  1 MiB (an L2) by default;
// NWGRAD_TRACK_PROFILE_BYTES overrides it at load, and the tests set it to 0 to drive
// the per-row path.  Results are identical either way: it is a speed / memory knob.
inline std::atomic<size_t>& track_profile_bytes() {
    static std::atomic<size_t> v{[] {
        if (const char* e = std::getenv("NWGRAD_TRACK_PROFILE_BYTES")) {
            char* end = nullptr;
            const unsigned long long x = std::strtoull(e, &end, 10);
            if (end != e) return static_cast<size_t>(x);
        }
        return size_t(1) << 20;
    }()};
    return v;
}

template<GapModel GM, AlignMode AM, AlignBand AB = AlignBand::Full, class T = double>
struct Aligner {

    // Viterbi-precision buffer type.  T is the scalar precision of the Viterbi /
    // hard-gradient DP: double by default, or float32 for ~2x SIMD lanes and half
    // the table footprint.  This member alias deliberately *shadows* the global
    // ::DpBuffer (= DpBufferT<double>) throughout the class body, so every DpBuffer&
    // signature below already means "the buffer at this Aligner's precision" with no
    // per-site edit.  For T=double it resolves to exactly the old type, which is what
    // keeps the double build byte-for-byte unchanged.  The forward-backward / soft
    // tables inside DpBufferT<T> are double regardless of T.
    using DpBuffer = DpBufferT<T>;

    // Not copyable: a_idx_/b_idx_ and blkT_ may point into this aligner's own vectors
    // (a_own_, b_own_, blkT_storage_), which a copy duplicates without re-aiming them.
    // Moving keeps each vector's heap buffer, so the pointers stay valid.
    Aligner() = default;
    Aligner(const Aligner&) = delete;
    Aligner& operator=(const Aligner&) = delete;
    Aligner(Aligner&&) = default;
    Aligner& operator=(Aligner&&) = default;

    // ── Public pipeline API — own internal buffer ─────────────────────────────

    // Convenience: take characters, validate and encode them against the params'
    // alphabet into aligner-owned buffers.  An out-of-alphabet character throws.
    //
    // Callers that align the same sequences repeatedly should encode once and
    // use the span overload below — SeqPair does exactly that, so a banded
    // re-alignment under a new matrix costs no re-encoding.
    void set_problem(std::string_view a, std::string_view b,
                     const AlignParams& params,
                     int band = 0,
                     std::vector<int> guide_j = {},
                     std::span<const uint16_t> track = {},
                     std::span<const uint16_t> gap_track = {}) {
        // Encode both before touching a_own_/b_own_: a bad character in B must not
        // free A's codes under the previous problem's spans.
        const Alphabet& alpha = params.matrix.alphabet();
        std::vector<uint8_t> ea = alpha.encode(a), eb = alpha.encode(b);
        a_own_ = std::move(ea);
        b_own_ = std::move(eb);
        set_problem(std::span<const uint8_t>(a_own_),
                    std::span<const uint8_t>(b_own_),
                    params, band, std::move(guide_j), track, gap_track);
    }

    // Sequences arrive already encoded to alphabet indices — the DP never sees a
    // char.  Encoding (and validation) happens once, at the boundary, in SeqPair
    // / BatchAligner.  The spans must outlive the DP calls that follow.
    //
    // Tracks (see "Tracks" below): `track` gives the parameter set of each residue of A
    // (m entries, each < params.num_sets()), `gap_track` the set of each DP row 0..m for
    // gaps in A (m+1 entries).  Either may be empty: the track then defaults to all
    // zeros, the gap track to row 0 -> track[0], row i -> track[i-1].  A problem is
    // TRACKED when params has more than one set or either track is given; an untracked
    // problem runs exactly the code it always ran.  The spans must outlive the DP calls,
    // like the sequences'.  `blkT`, when given, is params' whole matrix stack already in
    // precision T (the batch prepares it once per params); otherwise a float32 tracked
    // problem converts the stack here.
    // The untracked entry (no tracks): what it always was, except that params with more
    // than one set make the problem tracked with the default tracks.
    void set_problem(std::span<const uint8_t> a, std::span<const uint8_t> b,
                     const AlignParams& params,
                     int band = 0,
                     std::vector<int> guide_j = {}) {
        problem_common_(a, b, params, band, std::move(guide_j), params.num_sets() > 1, nullptr, T(-1));
        if (tr_) setup_tracks_({}, {}, nullptr);
        problem_set_      = true;
    }
    // `step_max` (integer T, with blkT and gapsT): the params' largest step, prepared by the
    // batch with the casts; T(-1) = compute it here (floating T: ignored).
    void set_problem(std::span<const uint8_t> a, std::span<const uint8_t> b,
                     const AlignParams& params, int band, std::vector<int> guide_j,
                     std::span<const uint16_t> track, std::span<const uint16_t> gap_track,
                     const T* blkT = nullptr, const T* gapsT = nullptr, T step_max = T(-1)) {
        problem_common_(a, b, params, band, std::move(guide_j),
                        params.num_sets() > 1 || !track.empty() || !gap_track.empty(), blkT, step_max);
        if (tr_) setup_tracks_(track, gap_track, gapsT);
        problem_set_      = true;
    }

    // Whether the current problem is tracked (see set_problem).
    bool tracked() const noexcept { return tr_; }
    // The current tracked problem's per-row parameters for a kernel without a query
    // profile (the inter-pair fills).  Valid until the next set_problem().
    void track_job(TrackJob<T>& tj) { fill_track_job_(tj, false, 0, GM == GapModel::Linear); }

    // A tracked soft result (after compute_forward_back()): the expected counts PER DP
    // ROW — cnt[(i-1)*n + c], row i's match posteriors against letter c (its set and
    // residue are the row's), and gaps[4*i + f], row i's four gap fields (negative: the
    // gradient's sign) — from which soft_grad() scatters into the K sets.  The batch
    // stores this record per pair (it does not grow with K).
    const std::vector<double>& soft_record_counts() const noexcept { return srec_cnt_; }
    const std::vector<double>& soft_record_gaps() const noexcept { return srec_gap_; }

    // The optimal path as forward moves (0 = diagonal, 1 = gap in B, 2 = gap in A) from
    // its start cell (i0, j0), and (with gj) the guide, after compute_viterbi() — for
    // storing a path instead of its gradient (a tracked batch's hard gradients: see
    // BatchEngine).  The same walk hard_grad() takes, so the same moves.
    void hard_path_and_guide(const DpBuffer& buf, std::vector<uint8_t>& moves, int& i0, int& j0,
                             std::vector<int>* gj) const {
        moves.clear();
        if (hirschberg_) {
            moves.assign(hops_.begin(), hops_.end());
            i0 = hb_start_i_; j0 = hb_start_j_;
            if (gj) *gj = guide_j_affine_hb();
            return;
        }
        path_walk_(buf, [&](uint8_t k) { moves.push_back(k); }, i0, j0, gj);
        std::reverse(moves.begin(), moves.end());
    }
    // The same path packed 2 bits a move straight into `dst` (whose (cap+3)/4 bytes the
    // caller zeroed; cap >= the path's length, len_a + len_b serves): written from the
    // back, so it ends at move cap-1; returns the first move's index, cap - steps.
    int hard_path_packed(const DpBuffer& buf, uint8_t* dst, int cap, int& i0, int& j0,
                         std::vector<int>* gj) const {
        auto put = [dst](int pos, uint8_t k) {
            dst[pos >> 2] = static_cast<uint8_t>(dst[pos >> 2] | (k << ((pos & 3) * 2)));
        };
        if (hirschberg_) {
            const int off = cap - static_cast<int>(hops_.size());
            for (size_t q = 0; q < hops_.size(); ++q) put(off + static_cast<int>(q), hops_[q]);
            i0 = hb_start_i_; j0 = hb_start_j_;
            if (gj) *gj = guide_j_affine_hb();
            return off;
        }
        int pos = cap;
        path_walk_(buf, [&](uint8_t k) { put(--pos, k); }, i0, j0, gj);
        return pos;
    }
    // The walk of both: emit(k) for each move, LAST move first; the start cell into
    // (i0, j0) and the guide into gj.
    template <class Emit>
    void path_walk_(const DpBuffer& buf, Emit&& emit, int& i0, int& j0, std::vector<int>* gj) const {
        int ci = 0, cj = 0;   // the cell the last emitted move left from
        if constexpr (GM == GapModel::Linear) {
            ci = (AM == AlignMode::Global) ? m_ : best_i_;
            cj = (AM == AlignMode::Global) ? n_ : best_j_;
            if (gj) {
                gj->assign(static_cast<size_t>(m_ + 1), -1);
                (*gj)[0] = 0;
                (*gj)[static_cast<size_t>(m_)] = n_;
                const int i_end = (AM == AlignMode::Global) ? m_ : best_i_;
                (*gj)[static_cast<size_t>(i_end)] = (AM == AlignMode::Global) ? n_ : best_j_;
            }
            walk_linear(buf, [&](int k, int i, int j) {
                emit(static_cast<uint8_t>(k));
                if (gj) {
                    if (k == 0)      (*gj)[static_cast<size_t>(i - 1)] = j - 1;
                    else if (k == 1) (*gj)[static_cast<size_t>(i - 1)] = j;
                }
                ci = (k == 2) ? i : i - 1;
                cj = (k == 1) ? j : j - 1;
            });
        } else {
            ci = best_i_; cj = best_j_;
            if (gj) guide_affine_init(*gj);
            walk_affine(buf, [&](TBTable st, int i, int j, bool) {
                const uint8_t k = st == TBTable::M ? 0 : st == TBTable::X ? 1 : 2;
                emit(k);
                if (gj) {
                    if (k == 0)      (*gj)[static_cast<size_t>(i - 1)] = j - 1;
                    else if (k == 1) (*gj)[static_cast<size_t>(i - 1)] = j;
                }
                ci = (k == 2) ? i : i - 1;
                cj = (k == 1) ? j : j - 1;
            });
        }
        if (gj) fill_guide_gaps(*gj);
        i0 = ci; j0 = cj;
    }

    // The hard gradient as unit counts on the flat K-set layout (BatchEngine's: matrix
    // stack, then 4 gap fields per set): add(key) once per count — counts on gap fields
    // stand for -1 each — and (with gj) the guide, from one walk: the same walk, and so
    // the same path, as hard_grad() (a tracked batch's hard gradients stored as counts:
    // see BatchEngine).
    template <class Add>
    void hard_fields(const DpBuffer& buf, Add&& add, std::vector<int>* gj) const {
        const uint32_t gbase = static_cast<uint32_t>(params_->num_sets()) * static_cast<uint32_t>(nalpha_ * nalpha_);
        auto gk = [&](int i, int f) {
            const int set = (f == AlignParams::kOpenA || f == AlignParams::kExtendA) ? gset(i) : tset(i);
            return gbase + 4 * static_cast<uint32_t>(set) + static_cast<uint32_t>(f);
        };
        auto sk = [&](int i, int j) { return static_cast<uint32_t>(sub_off(i, j)); };
        if (hirschberg_) {
            int i = hb_start_i_, j = hb_start_j_;
            unsigned char prev = 3;
            for (unsigned char op : hops_) {
                if (op == 0) { ++i; ++j; add(sk(i, j)); }
                else if (op == 1) {
                    ++i;
                    if (GM == GapModel::Affine && prev != 1) add(gk(i, AlignParams::kOpenB));
                    add(gk(i, AlignParams::kExtendB));
                } else {
                    if (GM == GapModel::Affine && prev != 2) add(gk(i, AlignParams::kOpenA));
                    add(gk(i, AlignParams::kExtendA)); ++j;
                }
                prev = op;
            }
            if (gj) *gj = guide_j_affine_hb();
            return;
        }
        if constexpr (GM == GapModel::Linear) {
            if (gj) {
                gj->assign(static_cast<size_t>(m_ + 1), -1);
                (*gj)[0] = 0;
                (*gj)[static_cast<size_t>(m_)] = n_;
                const int i_end = (AM == AlignMode::Global) ? m_ : best_i_;
                (*gj)[static_cast<size_t>(i_end)] = (AM == AlignMode::Global) ? n_ : best_j_;
            }
            walk_linear(buf, [&](int k, int i, int j) {
                if (k == 0) {
                    add(sk(i, j));
                    if (gj) (*gj)[static_cast<size_t>(i - 1)] = j - 1;
                } else if (k == 1) {
                    add(gk(i, AlignParams::kExtendB));
                    if (gj) (*gj)[static_cast<size_t>(i - 1)] = j;
                } else {
                    add(gk(i, AlignParams::kExtendA));
                }
            });
        } else {
            if (gj) guide_affine_init(*gj);
            walk_affine(buf, [&](TBTable st, int i, int j, bool open) {
                if (st == TBTable::M) {
                    add(sk(i, j));
                    if (gj) (*gj)[static_cast<size_t>(i - 1)] = j - 1;
                } else if (st == TBTable::X) {
                    add(gk(i, AlignParams::kExtendB));
                    if (open) add(gk(i, AlignParams::kOpenB));
                    if (gj) (*gj)[static_cast<size_t>(i - 1)] = j;
                } else {
                    add(gk(i, AlignParams::kExtendA));
                    if (open) add(gk(i, AlignParams::kOpenA));
                }
            });
        }
        if (gj) fill_guide_gaps(*gj);
    }

    // Allocate an empty buffer shell. Must be called once before compute_viterbi()
    // / compute_forward_back() with own buffer. Buffers grow implicitly as needed.
    void alloc_buf() {
        own_buf_allocated_ = true;
    }

    // Select the DP kernel.  Scalar (the default) and Simd write bit-identical
    // tables; Simd is the vectorized one.  Affects compute_viterbi() only —
    // compute_forward_back() is shared and ignores this.
    // Set the Viterbi backend: kBackendAuto (defer to the global default), kBackendScalar
    // (scalar fallback), or a SimdLevel index.  simd_levels.hpp::parse_backend() turns the
    // string vocabulary ("scalar_fallback"/"auto"/"sse2"/"avx2"/...) into this int.
    void set_kernel(int backend) noexcept { backend_ = backend; }
    // How the traceback recovers predecessors — see TracebackMode.  Defaults to
    // Pointers (smaller and faster); set Scores when you need VM/VX/VY to survive
    // the fill, e.g. to compare tables across kernels.
    // Default resolves to this Aligner's compile-time kDefaultTb (Hirschberg for the
    // affine+global+full case, Pointers elsewhere) — so a caller can pass the sentinel
    // through without knowing the problem type.
    // Under `auto`, these types run a pair of up to kAutoHbPointerCells cells as Pointers
    // and split only above it (hb_splits): affine Global Full at T=double (exact
    // Hirschberg above the budget) and int64 (hirschberg_pmax above it), and affine Local
    // Full at T=float and the integer types (hirschberg_pmax above it).  Measured
    // 2026-10-25 with one-byte codes (AGENTS.md, "`auto` with a pointer budget"): Global
    // double, Pointers beat the exact recursion at every length and thread count within
    // the budget (nighthaven, wloczykij 64 cores); Global int64, Pointers beat pmax on both
    // fixtures; Local, Pointers beat pmax 1.5-3x on homologous pairs and lose 0-14 %
    // (float32) / 6-37 % (integers) on unrelated ones.  Local double keeps plain Pointers
    // (kDefaultTb); Global float32 / int16 / int32 keep hirschberg_pmax past hb_cutoff,
    // which won there (homologous float32, both int32 fixtures).
    static constexpr bool kAutoPointerBudget =
        GM == GapModel::Affine && AB == AlignBand::Full &&
        ((AM == AlignMode::Global && (std::is_same_v<T, double> || std::is_same_v<T, int64_t>)) ||
         (AM == AlignMode::Local && (std::is_same_v<T, float> || nw_is_int<T>)));
    // ... and up to how many cells (0 = no budget).  Global double 2^26: with the dense codes
    // (dense_gate_) Pointers kept beating the exact recursion to L8000 at 1 and 6 threads
    // (nighthaven 2026-10-26: Hirschberg 1.07-1.32x unrelated, 1.38-1.81x homologous); only
    // 12 threads on 6 cores (SMT) lost there (0.81 / 0.96).  Local float32 / integers keep
    // 2^25: above it pmax won on unrelated pairs by 2-5x.
    static constexpr std::size_t kAutoPointerCells =
        !kAutoPointerBudget ? 0
        : (AM == AlignMode::Global && std::is_same_v<T, double>) ? 2 * kAutoHbPointerCells
                                                                  : kAutoHbPointerCells;
    void set_traceback(TracebackMode t) noexcept {
        tb_auto_ = (t == TracebackMode::Default);
        tb_ = tb_auto_ ? kDefaultTb : t;
    }
    TracebackMode traceback() const noexcept { return tb_; }
    // Rows per block at which the Hirschberg recursion stops splitting and solves
    // outright — see hb_cutoff_.  Unlike set_traceback this is safe to change between
    // runs: it alters how the DP divides, not what it retains.  It is a THREE-way knob
    // (base-case memory, per-block overhead, tie-break fidelity), not just a speed one:
    // at cutoff >= m no split happens at all and the result is the Pointers fill, hence
    // bit-exact — the memory win is what pays for the divergence.
    // Which direction-code format a leveled Pointers fill writes: -1 auto (the gate,
    // dense_gate_), 0 one byte a cell, 1 the dense codes (0.75 B/cell affine, 0.25 B/cell
    // linear) wherever the kernel can (not int16).  Bit-identical results either way.
    // NWGRAD_DENSE_CODES=0/1 overrides auto.
    void set_dense_codes(int mode) noexcept { dense_mode_ = mode < 0 ? -1 : (mode ? 1 : 0); }
    int dense_codes() const noexcept { return dense_mode_; }
    bool dense_written() const noexcept { return dense_; }
    // The auto gate's thresholds (dense_gate_): cells a pair from which the dense codes are
    // written, per level, gap model, mode and type; 0 = never.  Only where measured faster
    // (2026-10-26; AGENTS.md "Dense direction codes"); unmeasured levels stay off.
    // AVX2 (nighthaven, 1-12 threads): from the size where dense won at every thread count.
    static constexpr std::size_t kDenseFromCellsAvx2 = [] {
        constexpr std::size_t M1 = std::size_t(1) << 20;
        if constexpr (sizeof(T) < 4 || AB != AlignBand::Full) return std::size_t(0);
        else if constexpr (GM == GapModel::Affine && AM == AlignMode::Global)
            return nw_is_int<T> && sizeof(T) == 4 ? M1 : 16 * M1;         // int32 L1000; else L4000
        else if constexpr (GM == GapModel::Affine)                         // Local: its row scan
            return nw_is_int<T> && sizeof(T) == 4 ? 16 * M1 : 32 * M1;    // dilutes the code stream
        else if constexpr (AM == AlignMode::Global)                        // linear
            return std::is_same_v<T, double> ? 16 * M1 : 4 * M1;
        else                                                               // linear Local
            return std::is_same_v<T, float> ? 4 * M1 : 16 * M1;
    }();
    // AVX-512: every crossover comes earlier than AVX2's.  Re-measured 2026-10-27 on PR #8's head (solace WSL, exact L x L, L700-5800, 1 / 8 / 16
    // threads): each threshold sits between the last size that did not win and the first from
    // which dense won (or tied within its control) at every thread count and size above.
    // (m+1)(n+1) at L700 / 1000 / 1450 / 2000 / 2900 / 4000: 0.49 / 1.00 / 2.11 / 4.00 / 8.41 / 16.0 M.
    static constexpr std::size_t kDenseFromCellsAvx512 = [] {
        constexpr std::size_t M1 = std::size_t(1) << 20;
        constexpr bool D = std::is_same_v<T, double>, F = std::is_same_v<T, float>;
        if constexpr (sizeof(T) < 4 || AB != AlignBand::Full) return std::size_t(0);
        else if constexpr (std::is_same_v<T, int64_t> && !(GM == GapModel::Affine && AM == AlignMode::Global))
            return kDenseFromCellsAvx2;                                    // unmeasured: AVX2's
        else if constexpr (GM == GapModel::Affine && AM == AlignMode::Global)
            return D ? M1 / 2 : F ? 3 * M1 : nw_is_int<T> && sizeof(T) == 4 ? 2 * M1 : M1 / 2;
        else if constexpr (GM == GapModel::Affine)                         // Local
            return D ? M1 / 2 : 6 * M1;
        else if constexpr (AM == AlignMode::Global)                        // linear
            return D ? 6 * M1 : F ? 2 * M1 : M1 / 2;
        else                                                               // linear Local
            return D ? 12 * M1 : 3 * M1;
    }();
    // sse2 / sse4 (wloczykij, Piledriver, 1 / 16 / 64 threads): dense lost 6-24 % almost
    // everywhere (gains of 2-4 % only at 64 threads, int32 affine and linear double) -- off.
    // NEON (M1, 1-8 threads): affine and linear double lost 4-18 % at every size (no memory
    // wall there); linear float32 / int32 won 7-10 % at every size measured (L1000-5800).
    // NEON (M1): linear float32 / int32 from 2^20; affine int32 and Local double from 2^18
    // (0.92-0.98 from L500), float32 from 2^22 (parity at L500, 0.95-0.98 above); affine
    // Global double off (1.00-1.05 even with the weighted pass 1); linear double off.
    static constexpr std::size_t kDenseFromCellsNeon =
        AB != AlignBand::Full ? 0
        : GM == GapModel::Linear
            ? ((std::is_same_v<T, float> || std::is_same_v<T, int32_t>) ? std::size_t(1) << 20 : 0)
        : std::is_same_v<T, int32_t> ? std::size_t(1) << 18
        : std::is_same_v<T, float>   ? std::size_t(1) << 22
        : (std::is_same_v<T, double> && AM == AlignMode::Local) ? std::size_t(1) << 18 : 0;
    void set_hb_cutoff(int rows) {
        if (rows < 1)
            throw std::invalid_argument("nwgrad: hb_cutoff must be >= 1");
        hb_cutoff_ = rows;
    }
    int hb_cutoff() const noexcept { return hb_cutoff_; }
    // Whether the retained Viterbi tables (or direction bytes) are in the striped layout
    // of a leveled kernel — i.e. one ran, not the scalar fill.  For tests' liveness checks.
    bool tables_striped() const noexcept { return tables_striped_; }
    // The B length from which the linear Full fill takes the leveled striped kernel.
    // ... on `backend` (a level id): AVX-512's lanes move some crossovers later.
    static int linear_striped_min_n(int backend) noexcept {
        if (const int64_t o = gate_override(Gate::LinearStripedMinN); o >= 0)   // nwgrad._tuning
            return o > INT32_MAX ? INT32_MAX : static_cast<int>(o);
#if !(defined(__aarch64__) || defined(_M_ARM64))
        if (backend == static_cast<int>(SimdLevel::Avx512)) return kLinearStripedMinN512;
#endif
        (void)backend;
        return kLinearStripedMinN;
    }
    int  kernel() const noexcept { return backend_; }
    // Which simd fill the Full band uses at double precision: the striped kernel (false,
    // the default) or the row-wise one the guide-banded path uses (true).  Both write
    // tables bit-identical to the scalar fill, so this is a speed knob only.  Row-wise
    // pays no lazy-F fixpoint, which wins on short pairs (miRNA x site, ~22 x 50:
    // 1.5-1.8x, more with cheap gaps); striped wins on long ones.  Row-wise keeps
    // VM/VX/VY (24 B/cell) even under traceback Pointers, which it cannot record.
    void set_rowwise_full(bool on) noexcept { rowwise_full_ = on; }
    bool rowwise_full() const noexcept { return rowwise_full_; }

    void compute_viterbi() {
        check_problem();
        check_own_buf_allocated();
        ensure_viterbi_ptruf(own_buf_);  // Grow if needed
        run_viterbi(own_buf_);
        viterbi_done_ = any_viterbi_done_ = true;
        fwdbwd_is_newest_ = false;
    }

    void compute_forward_back() {
        check_problem();
        check_own_buf_allocated();
        if constexpr (nw_is_int<T>) throw_soft_int_();
        else {
            run_fwdbwd(own_buf_);
            fwdbwd_done_ = any_fwdbwd_done_ = true;
            fwdbwd_is_newest_ = true;
        }
    }

    // Returns the score from whichever DP ran most recently (Viterbi score or log Z),
    // regardless of which buffer was used.
    double score() const {
        if (fwdbwd_is_newest_) return log_z_;
        if (any_viterbi_done_) return static_cast<double>(viterbi_score_);
        throw std::logic_error(
            "nwgrad: call compute_viterbi() or compute_forward_back() before score()");
    }

    double log_z() const {
        if (!any_fwdbwd_done_)
            throw std::logic_error("nwgrad: call compute_forward_back() before log_z()");
        return log_z_;
    }

    std::vector<std::pair<int,int>> alignment() const {
        check_viterbi();
        if (hirschberg_) return alignment_hb();
        if constexpr (GM == GapModel::Linear) return traceback_linear(own_buf_);
        else                                   return traceback_affine(own_buf_);
    }

    std::vector<int> guide_j_from_viterbi() const {
        check_viterbi();
        return guide_j_from_viterbi(own_buf_);
    }

    // Reconstruct the gapped aligned sequences (a, b) from the Viterbi traceback.
    // Each string uses '-' for gaps and the two are of equal length.  Works for
    // both global (full-length) and local (matched sub-region only) alignment.
    std::pair<std::string, std::string> aligned() const {
        check_viterbi();
        return aligned(own_buf_);
    }

    void hard_grad(AlignParams& grad) const {
        check_viterbi();
        hard_grad(own_buf_, grad);
    }

    void soft_grad(AlignParams& grad) const {
        check_fwdbwd();
        soft_grad(own_buf_, grad);
    }

    // Release own DP table memory and reset computed-state flags.
    // The buffer shell remains allocated; set_problem() must be called for next DP run.
    void free_dp() noexcept {
        own_buf_.clear();
        problem_set_     = false;
        viterbi_done_    = false;
        fwdbwd_done_     = false;
        any_viterbi_done_ = false;
        any_fwdbwd_done_ = false;
        fwdbwd_is_newest_ = false;
    }

    // ── Extended API — caller-supplied DpBuffer ───────────────────────────────
    //
    // set_problem() must have been called first.
    // viterbi_done_ / fwdbwd_done_ are NOT touched; callers manage their own state.
    // Scalar results (viterbi_score_, log_z_, best_i_, best_j_, best_tbl_) are
    // always written to the Aligner so score() / log_z() work after these calls.

    void compute_viterbi(DpBuffer& buf) {
        check_problem();
        ensure_viterbi_ptruf(buf);  // Grow if needed (allows implicit growth from size 0)
        run_viterbi(buf);
        any_viterbi_done_ = true;
        fwdbwd_is_newest_ = false;
    }

    // ── Score only ────────────────────────────────────────────────────────────
    //
    // The optimal alignment score and nothing else: no table, no traceback, O(n) memory.
    // Bit-identical to score() after compute_viterbi() under an EXACT traceback mode
    // (Pointers / Scores — every cell is computed by the same expressions as the fill),
    // at every backend; it does not touch the Viterbi state, so path, guide and gradient
    // accessors are unaffected (and still need compute_viterbi()).  Note the float32
    // `auto` default for affine Full is hirschberg_pmax, whose path-replayed score may sit
    // a measured hair below this exact optimum (see AGENTS.md).
    //
    // Full: the leveled striped kernel (score_kernel_impl.inl) on a simd backend, a scalar
    // rolling-row copy of the fill otherwise.  Linear gaps run through the affine kernel
    // with zero opens, which reduces the recurrence to the linear one exactly (v - 0 is
    // exact, and rounding is monotone, so max(a-g, b-g) == max(a,b) - g bit for bit;
    // the borders -(0 + i*ge) are -i*ge).  GuideBanded runs the banded fill: its tables
    // are band-sized already, and it is not what score-only callers use.
    double compute_score(DpBuffer& buf) {
        check_problem();
        if constexpr (AB == AlignBand::GuideBanded) {
            compute_viterbi(buf);
            return viterbi_score_;
        } else {
            const int backend = (backend_ == kBackendAuto) ? global_default_backend() : backend_;
            if (backend != kBackendScalar) {
                const auto fn = level_kernels(backend).template of<T>().score;
                if (fn) {
                    ScoreJob<T> job{};
                    job.a = a_idx_.data(); job.m = m_;
                    job.b = b_idx_.data(); job.n = n_;
                    job.blk = blkT_; job.nalpha = nalpha_;
                    const bool lin = GM == GapModel::Linear;
                    job.go_a = lin ? T(0) : static_cast<T>(params_->gap_open_a);
                    job.ge_a = static_cast<T>(params_->gap_extend_a);
                    job.go_b = lin ? T(0) : static_cast<T>(params_->gap_open_b);
                    job.ge_b = static_cast<T>(params_->gap_extend_b);
                    job.local = AM == AlignMode::Local;
                    job.variant = score_variant_env();
                    job.buf = &buf;
                    if (tr_) fill_track_job_(job.tr, true, static_cast<size_t>(n_) + 16, lin);
                    fn(job);
                    return static_cast<double>(job.score);
                }
            }
            if constexpr (GM == GapModel::Linear) return score_linear_scalar(buf);
            else                                   return score_affine_scalar(buf);
        }
    }
    double compute_score() {
        check_problem();
        check_own_buf_allocated();
        return compute_score(own_buf_);
    }

    // soft_guide="posterior": the next forward-backward also records, per row of A, the
    // column of greatest posterior mass; posterior_guide() turns those rows into a guide
    // (non-decreasing, in [0, n]) — a band centre without running a Viterbi.
    void set_posterior_guide(bool on) noexcept { want_post_ = on; }
    static std::vector<int> monotone_guide(const int* rows, int m, int n) {
        std::vector<int> gj(static_cast<size_t>(m) + 1);
        int run = 0;
        for (int i = 0; i <= m; ++i) {
            run = std::max(run, std::min(std::max(rows[i], 0), n));
            gj[static_cast<size_t>(i)] = run;
        }
        return gj;
    }
    std::vector<int> posterior_guide() const { return monotone_guide(post_rows_.data(), m_, n_); }

    // For a GuideBanded inter-pair fill (InterJobT::blo): this problem's rows as lane
    // `lane` of W — per row i (0..M) the computed columns [jlo, jhi] (empty past m) into
    // blo/bhi[i*W + lane], the initialised span (band_row_span, clipped to columns >= 1)
    // into slo/shi[i] — min/max-merged with the other lanes' — and the Global border
    // extents.  set_problem() first.  Full: nothing to do.
    void banded_lane_rows(int W, int lane, int M, int* blo, int* bhi, int* slo, int* shi,
                          int& bri, int& brj) const {
        if constexpr (AB == AlignBand::GuideBanded) {
            for (int i = 1; i <= M; ++i) {
                const size_t k = static_cast<size_t>(i) * W + lane;
                if (i > m_) { blo[k] = 1; bhi[k] = 0; continue; }
                blo[k] = jlo(i); bhi[k] = jhi(i);
                int lo, hi; band_row_span(i, lo, hi);
                slo[i] = std::min(slo[i], std::max(1, lo));
                shi[i] = std::max(shi[i], hi);
            }
            bri = border_rows(); brj = border_cols();
        }
    }

    // In place of compute_viterbi(buf): adopt lane `lane` of an inter-pair fill
    // (InterJob, inter_kernel_impl.inl) that left W pairs' tables interleaved in
    // buf.VM/VX/VY, this aligner's pair being that lane.  The tables are bit-identical
    // to this pair's own affine Full fill, so score(), the traceback and hard_grad()
    // behave exactly as after compute_viterbi(buf).  Local passes the lane's best cell
    // as the kernel found it; Global reads the score at (m, n).  set_problem() first.
    // `stride`: the group's row length (n+1 of its longest B) when B lengths are ragged
    // (InterJobT::nb); 0 = this problem's own n+1.
    void adopt_interleaved(const DpBuffer& buf, int W, int lane,
                           T local_best, int best_i, int best_j, size_t stride = 0) {
        check_problem();
        inter_stride_ = stride ? stride : stride_;
        tables_striped_ = false;
        pointers_       = false;
        hirschberg_     = false;
        grid_           = false;
        inter_w_ = W; inter_lane_ = lane;
        if constexpr (GM == GapModel::Linear) {
            // Linear: H in buf.H's slot of the interleaved fill (InterJob.linear).
            if constexpr (AM == AlignMode::Local) {
                viterbi_score_ = local_best;
                best_i_ = best_i; best_j_ = best_j;
            } else {
                viterbi_score_ = rat(buf.H, m_, n_);
            }
        } else if constexpr (AM == AlignMode::Local) {
            viterbi_score_ = local_best;
            best_i_ = best_i; best_j_ = best_j; best_tbl_ = TBTable::M;
            if (best_i > 0) {
                const T mv = rat(buf.VM, best_i, best_j), xv = rat(buf.VX, best_i, best_j),
                        yv = rat(buf.VY, best_i, best_j);
                if      (mv >= xv && mv >= yv) best_tbl_ = TBTable::M;
                else if (xv >= yv)             best_tbl_ = TBTable::X;
                else                            best_tbl_ = TBTable::Y;
            }
        } else {
            const T vm = rat(buf.VM, m_, n_), vx = rat(buf.VX, m_, n_), vy = rat(buf.VY, m_, n_);
            viterbi_score_ = std::max({vm, vx, vy});
            best_i_ = m_; best_j_ = n_;
            if      (vm >= vx && vm >= vy) best_tbl_ = TBTable::M;
            else if (vx >= vy)             best_tbl_ = TBTable::X;
            else                            best_tbl_ = TBTable::Y;
        }
        any_viterbi_done_ = true;
        fwdbwd_is_newest_ = false;
    }

    void compute_forward_back(DpBuffer& buf) {
        check_problem();
        if constexpr (nw_is_int<T>) throw_soft_int_();
        else {
            run_fwdbwd(buf);
            any_fwdbwd_done_ = true;
            fwdbwd_is_newest_ = true;
        }
    }

    std::vector<int> guide_j_from_viterbi(const DpBuffer& buf) const {
        // Hirschberg holds the path itself, so there is nothing to walk back — replay it.
        if (hirschberg_) return guide_j_affine_hb();   // the move list is gap-model free
        if constexpr (GM == GapModel::Linear) return guide_j_linear(buf);
        else                                   return guide_j_affine(buf);
    }

    // The cell the alignment ends at, (i, j) in the DP: (m, n) for Global, the best cell
    // for Local.  Every traceback walks back from it, so with the aligned strings it
    // fixes where the alignment starts too (SeqPair::coordinates).
    std::pair<int, int> alignment_end() const {
        if constexpr (AM == AlignMode::Global) return {m_, n_};   // not every fill records it
        else                                   return {best_i_, best_j_};
    }

    std::pair<std::string, std::string> aligned(const DpBuffer& buf) const {
        std::string a, b;
        if (hirschberg_) { aligned_hb(a, b); return {std::move(a), std::move(b)}; }
        if constexpr (GM == GapModel::Linear) aligned_linear(buf, a, b);
        else                                   aligned_affine(buf, a, b);
        return {std::move(a), std::move(b)};
    }

    // guide_j_from_viterbi(buf) and hard_grad(buf, grad) in one traceback walk where
    // the path is walked back over the DP's own record (scores or Pointers, either gap
    // model); two walks for Hirschberg.  Same guide, same gradient either way.
    void hard_grad_and_guide(const DpBuffer& buf, AlignParams& grad, std::vector<int>& gj) const {
        if (!hirschberg_) {
            if constexpr (GM == GapModel::Affine) hard_grad_affine(buf, grad, &gj);
            else                                   hard_grad_linear(buf, grad, &gj);
            return;
        }
        gj = guide_j_from_viterbi(buf);
        hard_grad(buf, grad);
    }

    void hard_grad(const DpBuffer& buf, AlignParams& grad) const {
        if constexpr (GM == GapModel::Linear)
            if (hirschberg_) { hard_grad_linear_hb(grad); return; }
        if constexpr (GM == GapModel::Affine) {
            if (hirschberg_) { hard_grad_affine_hb(grad); return; }
        }
        if constexpr (GM == GapModel::Linear) hard_grad_linear(buf, grad);
        else                                   hard_grad_affine(buf, grad);
    }

    // Introspection / testing: copy a DP table into canonical (m+1)×(n+1) row-major
    // order, reading through the current layout (de-stripes when the simd Full kernel
    // left it striped).  Used by the bit-exactness test to compare across layouts.
    template <class V>
    std::vector<typename V::value_type> to_row_major(const V& t) const {
        using E = typename V::value_type;
        const size_t rm_stride = static_cast<size_t>(n_) + 1;
        // Pointers leaves the score tables unallocated; say so rather than read
        // past the end of an empty vector.
        if (t.size() < cell_index(m_, n_) + 1)
            throw std::logic_error(
                "nwgrad: DP table is empty — TracebackMode::Pointers retains predecessor "
                "codes, not score tables; use set_traceback(TracebackMode::Scores) to "
                "inspect VM/VX/VY");
        std::vector<E> out(static_cast<size_t>(m_ + 1) * rm_stride);
        for (int i = 0; i <= m_; ++i)
            for (int j = 0; j <= n_; ++j)
                out[static_cast<size_t>(i) * rm_stride + j] = rat(t, i, j);
        return out;
    }

    void soft_grad(const DpBuffer& buf, AlignParams& grad) const {
        if (tr_) { add_soft_record_(grad); return; }   // the per-row record (run_fwdbwd)
        if (soft_counts_ready_) { add_scaled_counts(grad); return; }
        if constexpr (GM == GapModel::Linear) soft_grad_linear(buf, grad);
        else                                   soft_grad_affine(buf, grad);
    }

    // See SoftImpl.  Takes effect at the next compute_forward_back().
    void set_soft_impl(SoftImpl s) noexcept { soft_impl_ = s; }
    SoftImpl soft_impl() const noexcept { return soft_impl_; }
    // Whether the most recent compute_forward_back() ran the scaled path (false: log).
    bool soft_scaled_used() const noexcept { return soft_scaled_used_; }

    // Soft temperature T > 0 (default 1).  The soft path then evaluates the smoothed
    // score T·log Z(θ/T) — log_z() returns that — and its gradient with respect to θ,
    // which is the expected counts under θ/T.  T → 0 recovers the Viterbi score and
    // (on tie-free inputs) the hard counts; T = 1 is the plain log Z.
    void set_soft_temperature(double t) {
        if (!(t > 0.0 && t <= std::numeric_limits<double>::max()))
            throw std::invalid_argument("nwgrad: soft temperature must be finite and > 0, got " +
                                        std::to_string(t));
        soft_temp_ = t;
    }
    double soft_temperature() const noexcept { return soft_temp_; }

private:
    // The soft path's sentinel (always double) and the hard path's (in T; see
    // scalar_traits.hpp: -inf for floating T, lowest()/2 for an integer T).
    static constexpr double NEG_INF = -std::numeric_limits<double>::infinity();
    static constexpr T      NINF    = nw_ninf<T>;

    // ── Problem state ─────────────────────────────────────────────────────────
    std::span<const uint8_t> a_idx_, b_idx_;   // alphabet indices, not characters
    // Backing store for the string_view overload of set_problem(); empty when
    // the caller supplied encoded spans directly.
    std::vector<uint8_t> a_own_, b_own_;
    const AlignParams*  params_     = nullptr;
    const double*       blk_        = nullptr; // params_->matrix.data(), cached (double master)
    // The substitution block in the Viterbi precision T.  For T=double it aliases blk_
    // (zero copy); for float32 it points into blkT_storage_.  subT() reads it; sub()
    // keeps reading the double master, so the soft path is untouched by T.
    const T*            blkT_       = nullptr;
    std::vector<T>      blkT_storage_;
    int                 nalpha_     = 0;       // params_->matrix.size(), cached
    int                 band_       = 0;
    int                 m_ = 0, n_ = 0;
    size_t              stride_ = 0, sz_ = 0;
    std::vector<int>    guide_j_;  // used only when AB == GuideBanded

    // The body both set_problem overloads share (inlined into each: the untracked one
    // must cost what it always cost).  Leaves problem_set_ to the caller.
    [[gnu::always_inline]] inline void problem_common_(std::span<const uint8_t> a, std::span<const uint8_t> b,
                                                       const AlignParams& params, int band,
                                                       std::vector<int>&& guide_j, bool tracked,
                                                       const T* blkT, T step_max) {
        // Invalidate first: a guide rejected below must not leave the previous
        // problem's results readable against this problem's sequences.
        problem_set_      = false;
        viterbi_done_     = false;
        fwdbwd_done_      = false;
        any_viterbi_done_ = false;
        any_fwdbwd_done_  = false;
        fwdbwd_is_newest_ = false;
        a_idx_  = a;
        b_idx_  = b;
        params_ = &params;
        blk_    = params.matrix.data();
        nalpha_ = params.matrix.size();
        tr_     = tracked;
        // The Viterbi-precision copy of the substitution block (the whole stack when
        // tracked).  T=double aliases the master (no copy); float32 converts once per
        // problem into blkT_storage_, unless the caller prepared it.
        if constexpr (std::is_same_v<T, double>) {
            (void)blkT; (void)step_max;
            blkT_ = blk_;
        } else if constexpr (nw_is_int<T>) {
            // The batch prepared the stack, the gaps and step_max once per params
            // (nw_int_prepare) and hands them to every pair, tracked or not; a standalone
            // problem prepares its own (the sets it reads: K when tracked, else set 0).
            if (blkT && step_max >= T(0)) {
                blkT_ = blkT;
                step_max_ = step_max;
            } else {
                nw_int_prepare<T>(params, tr_ ? params.num_sets() : 1, blkT_storage_,
                                  int_gaps_scratch_, step_max_);
                blkT_ = blkT_storage_.data();
            }
        } else {
            (void)step_max;
            if (tr_ && blkT) {
                blkT_ = blkT;
            } else {
                const std::size_t nn = static_cast<std::size_t>(nalpha_) * nalpha_ *
                                       static_cast<std::size_t>(tr_ ? params.num_sets() : 1);
                blkT_storage_.resize(nn);
                for (std::size_t k = 0; k < nn; ++k)
                    blkT_storage_[k] = static_cast<T>(blk_[k]);
                blkT_ = blkT_storage_.data();
            }
        }
        band_   = band;
        m_      = static_cast<int>(a.size());
        n_      = static_cast<int>(b.size());
        stride_ = static_cast<size_t>(n_ + 1);
        sz_     = static_cast<size_t>(m_ + 1) * stride_;
        if constexpr (nw_is_int<T>) int_bound_check_();
        // Checked for Full too: the entry points pick GuideBanded only for band > 0, so
        // a negative band would otherwise slip through as a full DP.
        if (band_ < 0)
            throw std::invalid_argument(
                "nwgrad: band must be >= 0, got " + std::to_string(band_));
        // Every guide entry is in [0, n], so a band >= n already covers whole rows;
        // clamping keeps guide_j_[i] + band_ (jhi/jhi0) from overflowing int.
        if (band_ > n_) band_ = n_;
        if constexpr (AB == AlignBand::GuideBanded) {
            if (guide_j.empty()) {
                guide_j_.resize(static_cast<size_t>(m_ + 1));
                if (m_ > 0)
                    for (int i = 0; i <= m_; ++i)
                        guide_j_[i] = static_cast<int>(
                            std::llround(static_cast<double>(i) * n_ / m_));
                else
                    guide_j_[0] = 0;
            } else {
                // The band helpers index guide_j_[0..m] unchecked, so a supplied guide
                // must describe THIS problem: one entry per row, each a column of B,
                // never moving left.  guide_j_from_aligned() checks only the aligned
                // strings' own syntax, not that they spell this a and b.
                validate_guide(guide_j);
                guide_j_ = std::move(guide_j);
            }
        }
    }


    // ── Tracks ────────────────────────────────────────────────────────────────
    //
    // A tracked problem scores each DP row with the parameter set its track selects
    // (params_ holds K sets: matrix block k, gap fields gap(k, .)).  Row i (1..m)
    // consumes a[i-1]; with s = track[i-1]:
    //   * a match (i-1, j-1) -> (i, j) scores matrix s at (a[i-1], b[j-1]);
    //   * a gap in B INTO row i (vertical, the X state, consuming a[i-1]) costs set s's
    //     gap_extend_b, plus its gap_open_b when the move opens the run — so a vertical
    //     run is charged the open of the row it STARTS in and each row's own extend;
    //   * a gap in A WITHIN row i (horizontal, the Y state, consuming b) costs set
    //     gap_track[i]'s gap_open_a / gap_extend_a; row 0 (before a[0]) and row m (after
    //     a[m-1]) included.  A horizontal run lies in one row, so it has one set.
    // Nothing else changes: the recurrences are the untracked ones with each penalty
    // read from its row.  Every DP below reads its row's penalties from the arrays here
    // (in precision T, cast exactly as the untracked fill casts params_'s fields), and
    // the substitution row from the STACKED block: row rs_[i-1] = s*nalpha + a[i-1] of
    // blkT_, the K matrices back to back (params_->matrix is that stack).  The gradient
    // is K sets too, each count landing in the set its row selects.
    //
    // The Global column-0 border is a vertical run from the origin: row i holds
    // -(go_b(1) + C_i), C_i the sum of ge_b over rows 1..i.  The untracked fill writes
    // the closed form -(go_b + i*ge_b), not a running sum, so C_i is accumulated by RUNS
    // of bitwise-equal extends: C_i = i*ge when rows 1..i share one value, and
    // C_{s-1} + (i-s+1)*ge from a run's start s on.  A track whose sets all hold the same
    // values is therefore bit-identical to the untracked problem — every penalty, every
    // border, every substitution score is the same number — which the tests assert.
    bool                       tr_ = false;
    // (The tracked state proper lives at the end of the class, away from the hot members.)

    // ── Hirschberg's reverse sweeps with tracks ──────────────────────────────
    // A reverse sweep runs the recurrence MIRRORED (hb_rev, hb_kernel_impl.inl's a_step
    // < 0): it meets a vertical run at the run's LAST row first, and charges the open
    // there.  With one open for every row that was the same number; with a track it is
    // not — the run must pay the open of its FIRST row.  So the mirror is given rewritten
    // rows: open'(r) = open(r+1) and extend'(r) = extend(r) + (open(r) - open(r+1)), with
    // open(m+1) := open(m).  A run over rows s..e then pays open(e+1) + sum extend'(r)
    // = open(s) + sum extend(r) — its true cost — by telescoping, whatever it straddles
    // inside the sweep.  A run the sweep enters already open (out_x, continuing past the
    // block's corner) pays sum extend' over its rows, its true cost minus open(i1+1) for
    // the block's last row i1: one constant for every path of that sweep, which the join's
    // argmax cannot see.  The join itself refunds open(p+1), the open of the first row
    // below the cut, where the plain sum charges a straddling run twice (hb_solve).
    // Equal rows give open' = open and extend' = extend + 0 = extend exactly, so equal sets
    // stay bit-identical to the untracked sweep.  (Exact in real arithmetic; in floating
    // point the mirror's sums round differently from the forward fill's, as the untracked
    // reverse sweep's always have — Hirschberg's score is replayed from its path.)
    void hb_reverse_rows_() const {
        if (rev_ready_) return;
        const size_t m = static_cast<size_t>(m_);
        trgob_.assign(m + 1, static_cast<T>(0));
        trgeb_.assign(m + 1, static_cast<T>(0));
        for (size_t r = 1; r <= m; ++r) {
            const T go_next = (r < m) ? tgob_[r + 1] : tgob_[r];
            trgob_[r] = go_next;
            trgeb_[r] = tgeb_[r] + (tgob_[r] - go_next);
        }
        rev_ready_ = true;
    }
    // The penalties a scalar Hirschberg sweep row reads: gaps in A of DP row `hr`, the
    // vertical move into DP row `vr` (reverse: the rewritten rows).
    T hb_go_a(int hr) const noexcept { return tr_ ? tgoa_[static_cast<size_t>(hr)] : static_cast<T>(params_->gap_open_a); }
    T hb_ge_a(int hr) const noexcept { return tr_ ? tgea_[static_cast<size_t>(hr)] : static_cast<T>(params_->gap_extend_a); }
    T hb_go_b(int vr, bool rev) const noexcept {
        if (!tr_) return static_cast<T>(params_->gap_open_b);
        return (rev ? trgob_ : tgob_)[static_cast<size_t>(vr)];
    }
    T hb_ge_b(int vr, bool rev) const noexcept {
        if (!tr_) return static_cast<T>(params_->gap_extend_b);
        return (rev ? trgeb_ : tgeb_)[static_cast<size_t>(vr)];
    }

    // The per-row parameters a leveled kernel reads (TrackJob), `profile`: whether it
    // builds a query profile (rows `row_len` elements long), so whether to offer one.
    // `linear_opens_zero`: a linear problem run through an affine kernel (score only).
    void fill_track_job_(TrackJob<T>& tj, bool profile, size_t row_len, bool linear_opens_zero = false) const {
        tj.rs = rs_.data();
        tj.go_a = tgoa_.data(); tj.ge_a = tgea_.data();
        tj.go_b = tgob_.data(); tj.ge_b = tgeb_.data();
        tj.bx0 = tbx0_.data();
        if (rev_ready_) { tj.rgo_b = trgob_.data(); tj.rge_b = trgeb_.data(); }
        if (linear_opens_zero) {
            tzero_.assign(static_cast<size_t>(m_) + 1, static_cast<T>(0));
            tj.go_a = tj.go_b = tzero_.data();
        }
        tj.ra = nullptr; tj.crow = nullptr; tj.nrow = 0;
        if (!profile) return;
        if (!compact_ready_) {
            crow_.assign(rs_.begin(), rs_.end());
            std::sort(crow_.begin(), crow_.end());
            crow_.erase(std::unique(crow_.begin(), crow_.end()), crow_.end());
            ra_.resize(rs_.size());
            for (size_t i = 0; i < rs_.size(); ++i)
                ra_[i] = static_cast<uint32_t>(std::lower_bound(crow_.begin(), crow_.end(), rs_[i]) - crow_.begin());
            compact_ready_ = true;
        }
        if (crow_.size() * row_len * sizeof(T) <= track_profile_bytes().load(std::memory_order_relaxed)) {
            tj.ra = ra_.data(); tj.crow = crow_.data(); tj.nrow = static_cast<int>(crow_.size());
        }
    }

    // The set of DP row i (1..m) for matches and gaps in B, and of row i (0..m) for gaps in A.
    int tset(int i) const noexcept { return trk_[static_cast<size_t>(i) - 1]; }
    int gset(int i) const noexcept { return gtrk_[static_cast<size_t>(i)]; }
    // Row i's penalties in precision T, tracked or not (for the walkers and the
    // non-hot paths; the fills read the arrays directly).
    T row_go_a(int i) const noexcept { return tr_ ? tgoa_[static_cast<size_t>(i)] : static_cast<T>(params_->gap_open_a); }
    T row_ge_a(int i) const noexcept { return tr_ ? tgea_[static_cast<size_t>(i)] : static_cast<T>(params_->gap_extend_a); }
    T row_go_b(int i) const noexcept { return tr_ ? tgob_[static_cast<size_t>(i)] : static_cast<T>(params_->gap_open_b); }
    T row_ge_b(int i) const noexcept { return tr_ ? tgeb_[static_cast<size_t>(i)] : static_cast<T>(params_->gap_extend_b); }
    // The same in double, from params_ (the soft path's precision).
    double drow_go_a(int i) const noexcept { return tr_ ? params_->gap(gset(i), AlignParams::kOpenA)   : params_->gap_open_a; }
    double drow_ge_a(int i) const noexcept { return tr_ ? params_->gap(gset(i), AlignParams::kExtendA) : params_->gap_extend_a; }
    double drow_go_b(int i) const noexcept { return tr_ ? params_->gap(tset(i), AlignParams::kOpenB)   : params_->gap_open_b; }
    double drow_ge_b(int i) const noexcept { return tr_ ? params_->gap(tset(i), AlignParams::kExtendB) : params_->gap_extend_b; }

    // `gapsT`: params' gap fields already cast to T, 4 per set (the batch prepares them
    // once per params); null: read and cast here.
    void setup_tracks_(std::span<const uint16_t> track, std::span<const uint16_t> gap_track,
                       const T* gapsT = nullptr) {
        const int K = params_->num_sets();
        const size_t m = static_cast<size_t>(m_);
        if (!track.empty() && track.size() != m)
            throw std::invalid_argument(
                "nwgrad: track has " + std::to_string(track.size()) + " entries, but sequence A has " +
                std::to_string(m) + " residues (one set per residue)");
        if (!gap_track.empty() && gap_track.size() != m + 1)
            throw std::invalid_argument(
                "nwgrad: gap_track has " + std::to_string(gap_track.size()) + " entries, but sequence A of " +
                "length " + std::to_string(m) + " needs " + std::to_string(m + 1) + " (one per DP row 0..m)");
        // (the max first: a batch re-checks every pair on every fill, so keep it a
        // vectorizable pass; the loops below only find the index to report)
        uint16_t mx = 0;
        for (const uint16_t v : track) mx = std::max(mx, v);
        for (const uint16_t v : gap_track) mx = std::max(mx, v);
        if (mx >= K) {
            for (size_t k = 0; k < track.size(); ++k)
                if (track[k] >= K)
                    throw std::invalid_argument(
                        "nwgrad: track[" + std::to_string(k) + "] = " + std::to_string(track[k]) +
                        " is not a parameter set (params has " + std::to_string(K) + ")");
            for (size_t k = 0; k < gap_track.size(); ++k)
                if (gap_track[k] >= K)
                    throw std::invalid_argument(
                        "nwgrad: gap_track[" + std::to_string(k) + "] = " + std::to_string(gap_track[k]) +
                        " is not a parameter set (params has " + std::to_string(K) + ")");
        }
        if (track.empty()) { trk_own_.assign(m, 0); trk_ = trk_own_; }
        else                 trk_ = track;
        if (gap_track.empty()) {
            gtrk_own_.resize(m + 1);
            gtrk_own_[0] = m ? trk_[0] : 0;
            for (size_t i = 1; i <= m; ++i) gtrk_own_[i] = trk_[i - 1];
            gtrk_ = gtrk_own_;
        } else {
            gtrk_ = gap_track;
        }
        rs_.resize(m);
        for (size_t i = 0; i < m; ++i)
            rs_[i] = static_cast<uint32_t>(trk_[i]) * static_cast<uint32_t>(nalpha_) + a_idx_[i];
        compact_ready_ = false;
        rev_ready_ = false;
        tgoa_.resize(m + 1); tgea_.resize(m + 1); tgob_.resize(m + 1); tgeb_.resize(m + 1);
        tbx0_.resize(m + 1);
        if (gapsT) {
            for (size_t i = 0; i <= m; ++i) {
                const T* g = gapsT + 4 * static_cast<size_t>(gtrk_[i]);
                tgoa_[i] = g[AlignParams::kOpenA]; tgea_[i] = g[AlignParams::kExtendA];
            }
            for (size_t i = 1; i <= m; ++i) {
                const T* g = gapsT + 4 * static_cast<size_t>(trk_[i - 1]);
                tgob_[i] = g[AlignParams::kOpenB]; tgeb_[i] = g[AlignParams::kExtendB];
            }
        } else {
            for (size_t i = 0; i <= m; ++i) {
                const int g = gtrk_[i];
                tgoa_[i] = static_cast<T>(params_->gap(g, AlignParams::kOpenA));
                tgea_[i] = static_cast<T>(params_->gap(g, AlignParams::kExtendA));
            }
            for (size_t i = 1; i <= m; ++i) {
                const int t = trk_[i - 1];
                tgob_[i] = static_cast<T>(params_->gap(t, AlignParams::kOpenB));
                tgeb_[i] = static_cast<T>(params_->gap(t, AlignParams::kExtendB));
            }
        }
        tgob_[0] = tgeb_[0] = tbx0_[0] = static_cast<T>(0);
        // The column-0 border, by runs of bitwise-equal extends (see above).
        T csum = static_cast<T>(0);     // C_{s-1}: the sum over rows before the current run
        size_t s = 1;                   // the current run's first row
        for (size_t i = 1; i <= m; ++i) {
            if (i > 1 && std::bit_cast<nw_bits_t<T>>(tgeb_[i]) != std::bit_cast<nw_bits_t<T>>(tgeb_[i - 1])) {
                // close the run [s, i-1]
                csum = (s == 1) ? static_cast<T>(i - 1) * tgeb_[i - 1]
                                : csum + static_cast<T>(i - s) * tgeb_[i - 1];
                s = i;
            }
            const T ge = tgeb_[i];
            if constexpr (GM == GapModel::Linear) {
                tbx0_[i] = (s == 1) ? -static_cast<T>(i) * ge
                                    : -(csum + static_cast<T>(i - s + 1) * ge);
            } else {
                const T go = tgob_[1];
                tbx0_[i] = (s == 1) ? -(go + static_cast<T>(i) * ge)
                                    : -(go + (csum + static_cast<T>(i - s + 1) * ge));
            }
        }
    }

    // ── Striped table layout ──────────────────────────────────────────────────
    // The leveled simd Viterbi (affine Full) writes VM/VX/VY in Farrar striped layout
    // to skip a per-row de-stripe copy.  When tables_striped_ is set, rat/at index those
    // tables striped (cell_index); every row-major fill clears it.  striped_seg_ and
    // striped_w_ come straight from the kernel (ViterbiJob.seg / .width); a striped row
    // is striped_seg_*striped_w_ + 1 doubles, slot 0 being column 0, at a stride of
    // striped_rs_ (ViterbiJob.rstride: the Scores and linear H kernels pad it, dp_buffer.hpp).
    bool   tables_striped_ = false;
    bool   rowwise_full_   = false;   // see set_rowwise_full()
    // Set by adopt_interleaved(): VM/VX/VY hold inter_w_ pairs' tables interleaved per
    // cell and this aligner's pair is lane inter_lane_.  0 = not interleaved; every fill
    // of this aligner's own clears it (run_viterbi).
    bool             want_post_ = false;   // soft_guide="posterior": fill post_rows_
    std::vector<int> post_rows_;
    int    inter_w_ = 0, inter_lane_ = 0;
    size_t inter_stride_ = 0;   // the interleaved tables' row length (>= stride_ if ragged)
    // Pointers mode: DM holds predecessor codes, VM/VX/VY are NOT retained.
    bool        pointers_ = false;
    // The last Pointers fill wrote the DENSE codes (dense_codes_row_bytes, simd_levels.hpp:
    // 2-bit fields, four segments a byte -- linear one plane, affine three).  dense_mode_ is
    // the request: -1 auto (dense_gate_), 0 never, 1 always (where the kernel can).
    bool        dense_ = false;
    int         dense_mode_ = -1;
    // Hirschberg mode: no tables at all survive the fill.  The recursion recovers the
    // path directly, so hops_ IS the result — every consumer (gradient, alignment,
    // guide_j) replays it instead of walking a table.  One byte per step, so a path
    // is O(m+n) where a table would be O(m·n).
    bool        hirschberg_ = false;
    std::vector<unsigned char> hops_;   // 0=M diagonal, 1=X gap-in-b, 2=Y gap-in-a
    // hops_ came from the Grid traceback (hirschberg_ is set too: every consumer replays
    // hops_), whose guide follows Pointers' convention (guide_j_affine_hb).
    bool grid_ = false;
    // Local Hirschberg only: the 0-based cell where the recovered local path STARTS.
    // hops_ is a path within the sub-rectangle A[hb_start_i_..ie) x B[hb_start_j_..je),
    // so every consumer replays it from this origin rather than (0, 0).  For Global (and
    // any non-local mode) both are 0, so the consumers are byte-for-byte unchanged there.
    int         hb_start_i_ = 0, hb_start_j_ = 0;
    // Rows per block at which the recursion stops splitting and runs the Pointers fill
    // instead.  1 would be pure Hirschberg (minimum memory, maximum re-sweeping); a
    // block of `cutoff` rows costs cutoff*n direction bytes and saves a level of
    // recursion.  512 from a fleet sweep AFTER hb_base was vectorized (sse2/avx2/avx512/
    // neon × len 500..8000): with a vectorized base case the optimum jumped from 32 to
    // ~512, because a bigger base case is now cheap and it means fewer 2x-work splits.
    // 512 sits in the sweet spot on every host — pairs <= 512 don't split, so they run
    // the exact Pointers fill at full speed (0.99-1.04x pointers, bit-exact); longer
    // pairs split and win 1.4-2.7x at high thread counts by staying out of the memory
    // wall.  A much larger cutoff (2048) re-enters that wall on long pairs.
    int         hb_cutoff_ = 512;
    // The Aligner's OWN default, resolved at compile time from the template case: the
    // affine+global+full specialization is the one Hirschberg implements, so it defaults
    // there; every other case keeps Pointers (which the linear/banded paths ignore
    // anyway).  A default-constructed Aligner therefore never lands on a mode that would
    // throw.  The TracebackMode::Default sentinel maps to exactly this.
    //
    // The Hirschberg case defaults FURTHER, on the scalar type: at T=float the carry is
    // the closed-form prefix max, at T=double the exact serial chain.  That split is a
    // deliberate judgment about what each precision is FOR.  float32 is the throughput
    // mode — it has already accepted ~8.4e-3 of its own rounding against a double oracle,
    // and the ramp's worst measured contribution on top of that is 4.9e-4, two orders of
    // magnitude below the error the caller has already agreed to pay, in exchange for
    // 1.5-3.9x on homologous data.  T=double is chosen BY people who want exactness, so
    // it keeps the carry that has it: there the ramp would be the largest error term in
    // the computation (~9.1e-13 against the exact mode's ~1.6e-12) rather than a rounding
    // lost in the noise.  Either default is overridable per problem.
    //
    // LOCAL affine Full at T=float defaults to HirschbergPmax too (2026-10-05), since its
    // endpoint scans took the prefix-max carry: measured (nighthaven, AVX2, float32,
    // 1000-3000 aa, 1 / 12 threads) 1.6x / 1.5-2.3x Pointers on 30%-mutated homologues
    // and 2.8-3.6x / 3.7-5x on unrelated pairs, identical totals; pairs <= hb_cutoff run
    // AS Pointers.  Local double stays Pointers (exact Local Hirschberg loses there; pmax
    // at double is opt-in by the rule above).  Linear and banded resolve to Pointers.
    //
    // An INTEGER T resolves affine Full (both modes) to HirschbergPmax outright: integer
    // arithmetic is exact and associative, so the prefix-max carry is bit-identical to the
    // serial one there (scalar_traits.hpp) and the only difference left is its speed.
    // Affine Global Full at floating T: Grid (2026-10-26) — Pointers' result at ~1-1.5x a
    // score-only pass, against 2-3.5x for the Hirschberg modes it replaced (AGENTS.md, "Grid
    // traceback").  Where Grid does not apply, `auto` runs kGridFallbackTb instead (see
    // run_viterbi), so a tracked or scalar-backend pair keeps Hirschberg's O(m+n) memory.
    static constexpr TracebackMode kGridFallbackTb =
        std::is_same_v<T, float> ? TracebackMode::HirschbergPmax : TracebackMode::Hirschberg;
    static constexpr TracebackMode kDefaultTb =
        (GM == GapModel::Affine && AB == AlignBand::Full && AM == AlignMode::Global)
            ? (nw_is_int<T> ? TracebackMode::HirschbergPmax
                            : TracebackMode::Grid)
        : (GM == GapModel::Affine && AB == AlignBand::Full && (std::is_same_v<T, float> || nw_is_int<T>))
            ? TracebackMode::HirschbergPmax
            : TracebackMode::Pointers;
    TracebackMode tb_ = kDefaultTb;
    bool          tb_auto_ = true;   // tb_ came from "auto" (see linear_ptr_fill)
    int    striped_seg_    = 0;
    int    striped_w_      = 1;
    size_t striped_rs_     = 0;   // a striped row's stride (>= (seg+1)*W: a kernel may pad it)

    int backend_ = kBackendAuto;  // which Viterbi backend fills the tables (default: auto)

    bool problem_set_       = false;
    bool own_buf_allocated_ = false; // own buffer shell has been allocated
    bool viterbi_done_      = false; // own_buf_ has valid viterbi data
    bool fwdbwd_done_       = false; // own_buf_ has valid fwdbwd data
    bool any_viterbi_done_  = false; // viterbi scalar results are valid (either buffer)
    bool any_fwdbwd_done_   = false; // fwdbwd scalar results are valid (either buffer)
    bool fwdbwd_is_newest_  = false; // true if forward-back ran more recently than viterbi

    // ── Scalar results (written by every compute_viterbi / compute_forward_back)
    T      viterbi_score_ = T(0);   // in T; score() promotes (exact below 2^53)
    double log_z_         = 0.0;
    int    best_i_ = 0, best_j_ = 0;

    // ── Scaled soft path state ────────────────────────────────────────────────
    SoftImpl soft_impl_ = SoftImpl::Scaled;
    bool     soft_scaled_used_ = false;
    bool     soft_counts_ready_ = false;   // scnt_/sg_* hold this run's gradient
    double   soft_temp_ = 1.0;
    // exp(substitution block), and the expected counts the scaled backward pass
    // accumulated (matrix: +count; gaps: -count, i.e. already the gradient's sign).
    std::vector<double> es_, scnt_, srow_, srs_;
    std::vector<int> sband_;   // per-row band ranges handed to the soft kernel
    double sg_go_a_ = 0, sg_ge_a_ = 0, sg_go_b_ = 0, sg_ge_b_ = 0;
    enum class TBTable { M, X, Y };
    mutable TBTable best_tbl_ = TBTable::M;

    // ── Own DP buffer (used by the no-arg public API) ─────────────────────────
    DpBuffer own_buf_;

    // ── Tracked state (see "Tracks"): declared last, so the untracked hot members keep
    // their places and cache lines.
    std::span<const uint16_t>  trk_, gtrk_;         // m: set per residue of A; m+1: per row
    std::vector<uint16_t>      trk_own_, gtrk_own_;  // defaults, when a span was empty
    std::vector<uint32_t>      rs_;                  // m: stacked substitution row of row i at [i-1]
    // Per DP row 0..m, precision T: gaps in A within row i (open, extend); gaps in B into
    // row i (open, extend; [0] unused); the Global column-0 border of row i ([0] unused).
    std::vector<T>             tgoa_, tgea_, tgob_, tgeb_, tbx0_;
    mutable std::vector<T>     tzero_;                  // m+1 zeros: linear's opens, for the affine kernels
    mutable std::vector<T>     trgob_, trgeb_;          // Hirschberg's reverse gaps in B (hb_reverse_rows_)
    // Tracked soft: the per-row record (soft_record_counts / _gaps), and the scaled
    // kernel's per-row weights (4 per row) and exp'd profile rows.
    std::vector<double>        srec_cnt_, srec_gap_, srw_;
    std::vector<double>        dbx0_;                   // the log path's column-0 border, double
    mutable bool               rev_ready_ = false;
    // The shared query profile's rows (TrackJob::ra/crow): each distinct stacked row once.
    mutable std::vector<uint32_t> ra_, crow_;
    mutable bool               compact_ready_ = false;  // ra_/crow_ describe this problem

    // ── Precondition checks ───────────────────────────────────────────────────
    void check_problem() const {
        if (!problem_set_)
            throw std::logic_error("nwgrad: call set_problem() first");
    }
    void check_viterbi() const {
        check_problem();
        if (!viterbi_done_)
            throw std::logic_error("nwgrad: call compute_viterbi() first");
    }
    void check_fwdbwd() const {
        check_problem();
        if (!fwdbwd_done_)
            throw std::logic_error("nwgrad: call compute_forward_back() first");
    }

    // The soft path (forward-backward, log Z, expected counts) is a computation in
    // exp/log space: it has no integer form.  An integer T throws here, and the
    // `if constexpr` at the two call sites keeps the double-space body from being
    // instantiated against integer tables at all.
    [[noreturn]] static void throw_soft_int_() {
        throw std::invalid_argument(
            std::string("nwgrad: the soft path (forward-backward / soft gradient) needs a "
                        "floating-point scalar type; this aligner's is ") + nw_type_name<T>());
    }

    // ── Integer T: parameter checks and the overflow bound (see scalar_traits.hpp) ──
    //
    // Every parameter must be an integer within ±nw_int_cap (the user quantizes; a
    // silent truncation of 0.5 to 0 would be the alternative), and the problem's a-priori
    // bound (m+n+2) * step_max_ must not exceed the cap, step_max_ being the largest
    // magnitude any one move can add: max |s| over the block, |go| + |ge| per side.
    T step_max_ = T(0);
    mutable std::vector<T> int_gaps_scratch_;   // nw_int_prepare's gaps, when run here
    void int_bound_check_() const {
        if (step_max_ > T(0) &&
            static_cast<long long>(m_) + n_ + 2 + nw_int_pad_cols > static_cast<long long>(nw_int_cap<T> / step_max_))
            throw std::overflow_error(
                "nwgrad: a " + std::to_string(m_) + " x " + std::to_string(n_) + " problem with a largest "
                "step of " + std::to_string(static_cast<long long>(step_max_)) + " could overflow the " +
                nw_type_name<T>() + " scalar type ((m + n + 2 + 64 padding columns) * step exceeds " +
                std::to_string(static_cast<long long>(nw_int_cap<T>)) + "); use a wider type");
    }

    // Throw if own buffer shell has never been allocated.
    void check_own_buf_allocated() const {
        if (!own_buf_allocated_)
            throw std::logic_error(
                "nwgrad: own DP buffer not allocated; call alloc_buf() before compute_viterbi() / compute_forward_back()");
    }

    // Throw if external viterbi buffer has never been allocated (size is 0).
    // After initial allocation, buffers may grow implicitly as needed.
    void check_external_viterbi_ptruf(const DpBuffer& buf) const {
        bool allocated;
        if constexpr (GM == GapModel::Linear)
            allocated = (buf.H.size() > 0);
        else
            allocated = (buf.VM.size() > 0);
        if (!allocated) {
            std::string gap_model_str = (GM == GapModel::Linear) ? "linear" : "affine";
            throw std::logic_error(
                "nwgrad: external viterbi DP buffer not allocated; provide a pre-allocated buffer or allocate with buf.H.resize(sz) before compute_viterbi(buf)");
        }
    }

    // Throw if external forward-backward buffer has never been allocated (size is 0).
    // After initial allocation, buffers may grow implicitly as needed.
    void check_external_fwdbwd_buf(const DpBuffer& buf) const {
        bool allocated;
        if constexpr (GM == GapModel::Linear)
            allocated = (buf.F.size() > 0);
        else
            allocated = (buf.FM.size() > 0);
        if (!allocated) {
            std::string gap_model_str = (GM == GapModel::Linear) ? "linear" : "affine";
            throw std::logic_error(
                "nwgrad: external forward-backward DP buffer not allocated; provide a pre-allocated buffer or allocate with buf.F.resize(sz) before compute_forward_back(buf)");
        }
    }

    // Whether this run actually selects rowwise score tables. The preference
    // alone is insufficient: splitting Hirschberg and scalar kernels ignore it. Share the predicate with dispatch so those paths keep their
    // smaller memory bounds even when rowwise was requested.
    bool uses_rowwise_full() const {
        if constexpr (GM == GapModel::Affine && AB == AlignBand::Full) {
            if (!rowwise_full_ || hb_splits(tb_, tb_auto_ ? kAutoPointerCells : 0, hb_cutoff_, m_, n_) ||
                (tb_ == TracebackMode::Grid && m_ > hb_cutoff_)) return false;
            const int backend = (backend_ == kBackendAuto) ? global_default_backend() : backend_;
            return backend >= 0 && level_kernels(backend).template of<T>().banded_row_local;
        }
        return false;
    }

    // Grow external buffer to fit current problem (thread-owned path).
    // Linear Full: whether the fill keeps direction bytes (viterbi_linear_ptr) rather than
    // H.  Explicit "pointers" (and Hirschberg's short pairs) always do.  Under "auto" —
    // which resolves to pointers for linear — only a pair whose H table exceeds 2 MiB
    // does: the pointer fill is ~1.35x slower per thread, but past L2 the H table's
    // traffic dominates.  Measured (nighthaven, AVX2, double, Global, 12 threads): H
    // wins 1.6x up to L=350 (1 MB), ties at L=500 (2 MB), loses 2.4x at 700 and 3.5x
    // at 1000.  Same path either way, so this is a speed choice only.
    static constexpr size_t kLinearHTableBytes = size_t(2) << 20;
    bool linear_ptr_fill() const noexcept {
        if constexpr (GM != GapModel::Linear || AB != AlignBand::Full) return false;
        else {
            if (is_hirschberg(tb_) || tb_ == TracebackMode::Grid) return true;
            if (tb_ != TracebackMode::Pointers) return false;
            return !tb_auto_ || sz_ * sizeof(T) > kLinearHTableBytes;
        }
    }

    void ensure_viterbi_ptruf(DpBuffer& buf) const {
        if constexpr (GM == GapModel::Linear) {
            // The pointer / Hirschberg fills keep no H (they allocate their own).
            if (linear_ptr_fill()) return;
            if (buf.H.size() < sz_) buf.H.resize(sz_);
        } else {
            // Pointers mode never touches VM/VX/VY — it keeps two rolling rows and byte
            // codes — so allocating them here would hand back the entire footprint
            // saving before the kernel ever runs.  Measured: skipping this is the
            // difference between B costing MORE memory than A and costing ~4x less.
            // Hirschberg allocates even less: its scratch is O(n) rows sized on demand
            // inside the recursion, and the whole point is that no O(m*n) table exists.
            if constexpr (AB == AlignBand::Full)
                if ((tb_ == TracebackMode::Pointers || is_hirschberg(tb_) || tb_ == TracebackMode::Grid) &&
                    !uses_rowwise_full()) return;
            for (auto* v : {&buf.VM, &buf.VX, &buf.VY}) if (v->size() < sz_) v->resize(sz_);
        }
    }

    void ensure_fwdbwd_buf(DpBuffer& buf) const {
        if constexpr (GM == GapModel::Linear) {
            // F and B checked separately: the scaled path grows F alone.
            if (buf.F.size() < sz_) buf.F.resize(sz_);
            if (buf.B.size() < sz_) buf.B.resize(sz_);
        } else {
            if (buf.FM.size() < sz_ || buf.BM.size() < sz_) {
                buf.FM.resize(sz_); buf.FX.resize(sz_); buf.FY.resize(sz_);
                buf.BM.resize(sz_); buf.BX.resize(sz_); buf.BY.resize(sz_);
            }
        }
    }

    // ── Cell accessors ────────────────────────────────────────────────────────
    // Every table access in the DP, the tracebacks and the gradient goes through these,
    // so switching VM/VX/VY between row-major and striped is a change to cell_index alone.
    // Striped layout MUST match the kernel in kernels_impl.inl: row size striped_seg_*
    // striped_w_ + 1, slot 0 = column 0, column j (1..n) at 1 + ((j-1)%seg)*W + (j-1)/seg.
    __attribute__((always_inline)) size_t cell_index(int i, int j) const noexcept {
        if (inter_w_) {
            // Inter-pair fill (adopt_interleaved): W pairs' tables interleaved per cell.
            return (static_cast<size_t>(i) * inter_stride_ + static_cast<size_t>(j)) *
                   static_cast<size_t>(inter_w_) + static_cast<size_t>(inter_lane_);
        }
        if (tables_striped_) {
            // rowsz = (seg+1)*W: slot 0 is column 0, slots [W, W+seg*W) are the striped
            // columns 1..n (started at W so every W-wide access lands on an aligned
            // address, given the 64-byte-aligned allocator), slots 1..W-1 pad.  Column j
            // is at W + ((j-1)%seg)*W + (j-1)/seg.  Must match kernels_impl.inl exactly.
            const size_t W = static_cast<size_t>(striped_w_);
            const size_t rowsz = striped_rs_;   // (seg+1)*W, or a kernel's padded stride
            if (j == 0) return static_cast<size_t>(i) * rowsz;
            const int jj = j - 1;
            return static_cast<size_t>(i) * rowsz + W +
                   static_cast<size_t>(jj % striped_seg_) * W +
                   static_cast<size_t>(jj / striped_seg_);
        }
        return static_cast<size_t>(i) * stride_ + static_cast<size_t>(j);
    }
    // Element-generic so they serve both the T-typed Viterbi tables (VM/VX/VY/H) and the
    // always-double soft tables (F/B/FM..BY) from one definition; the layout switch in
    // cell_index() is identical for either element type.
    template <class V> typename V::value_type& at(V& t, int i, int j) const {
        return t[cell_index(i, j)];
    }
    template <class V> typename V::value_type rat(const V& t, int i, int j) const {
        return t[cell_index(i, j)];
    }
    // The forward-backward tables (F/B, FM..BY) are ALWAYS row-major, whatever layout
    // the Viterbi fill left VM/VX/VY or DM/DX/DY in.  They get their own accessors so
    // the soft path never has to touch tables_striped_ / pointers_: those describe the
    // retained Viterbi state, which a traceback after compute_forward_back() still
    // reads (SeqPair's soft mode runs Viterbi, then forward-backward, then aligned()).
    // Clearing them here once made Pointers walk unallocated score tables and a
    // striped Scores walk read its tables row-major.
    double& sat(DVec& t, int i, int j) const noexcept {
        return t[static_cast<size_t>(i) * stride_ + static_cast<size_t>(j)];
    }
    double srat(const DVec& t, int i, int j) const noexcept {
        return t[static_cast<size_t>(i) * stride_ + static_cast<size_t>(j)];
    }

    // ── Substitution lookup (the DP hot path) ─────────────────────────────────
    // Offset of the (a[i-1], b[j-1]) cell in an n_alpha × n_alpha block.  DP
    // coordinates are 1-based, so i-1 / j-1 index the sequences.
    // The untracked offset, for the untracked fills' hot loops (no branch on tr_ there).
    size_t sub_off_u(int i, int j) const noexcept {
        return static_cast<size_t>(a_idx_[static_cast<size_t>(i) - 1]) *
                   static_cast<size_t>(nalpha_) +
               static_cast<size_t>(b_idx_[static_cast<size_t>(j) - 1]);
    }
    T subTu(int i, int j) const noexcept { return blkT_[sub_off_u(i, j)]; }
    // Tracked: the row is row i's stacked row (its set's block, then a[i-1]), which is
    // also the gradient's index, since a K-set gradient's matrix is the same stack.
    size_t sub_off(int i, int j) const noexcept {
        const size_t r = tr_ ? static_cast<size_t>(rs_[static_cast<size_t>(i) - 1])
                             : static_cast<size_t>(a_idx_[static_cast<size_t>(i) - 1]);
        return r * static_cast<size_t>(nalpha_) +
               static_cast<size_t>(b_idx_[static_cast<size_t>(j) - 1]);
    }

    // Substitution score for a[i-1] against b[j-1] (double master; used by the soft path).
    double sub(int i, int j) const noexcept { return blk_[sub_off(i, j)]; }

    // Substitution score in the Viterbi precision T (reads the T-converted block).  The
    // Viterbi fill and its tracebacks use this so the traceback float-equality
    // re-derivation happens in the SAME precision the forward pass rounded in — which is
    // load-bearing for bit-exactness when T=float.  For T=double it equals sub().
    T subT(int i, int j) const noexcept { return blkT_[sub_off(i, j)]; }
    // DP row i's substitution row (subT(i, j) == sub_row_(i)[b[j-1]]), for hoisting the
    // row lookup out of a column loop.
    const T* sub_row_(int i) const noexcept {
        const size_t r = tr_ ? static_cast<size_t>(rs_[static_cast<size_t>(i) - 1])
                             : static_cast<size_t>(a_idx_[static_cast<size_t>(i) - 1]);
        return blkT_ + r * static_cast<size_t>(nalpha_);
    }

    // The characters behind those indices — used only to render alignments.
    char sym_a(int i) const noexcept {
        return params_->matrix.alphabet().symbol_at(
            static_cast<int>(a_idx_[static_cast<size_t>(i) - 1]));
    }
    char sym_b(int j) const noexcept {
        return params_->matrix.alphabet().symbol_at(
            static_cast<int>(b_idx_[static_cast<size_t>(j) - 1]));
    }

    // The gradient's matrix block, checked to be over the same alphabet as the
    // params.  Checked once per grad call, not once per cell.
    double* grad_block(AlignParams& grad) const {
        if (&grad.matrix.alphabet() != &params_->matrix.alphabet())
            throw std::invalid_argument(
                "nwgrad: gradient alphabet \"" + grad.matrix.alphabet().symbols() +
                "\" does not match params alphabet \"" +
                params_->matrix.alphabet().symbols() + "\"");
        if (grad.num_sets() != params_->num_sets())
            throw std::invalid_argument(
                "nwgrad: gradient has " + std::to_string(grad.num_sets()) +
                " parameter sets, params have " + std::to_string(params_->num_sets()));
        return grad.matrix.data();
    }
    // The gradient's gap field f of the set row i charges: a gap in A (f = kOpenA /
    // kExtendA) its gap-track set, a gap in B (kOpenB / kExtendB) its track set.
    double& grad_gap(AlignParams& grad, int i, int f) const noexcept {
        if (!tr_) return grad.gap_ref(0, f);
        return grad.gap_ref((f == AlignParams::kOpenA || f == AlignParams::kExtendA) ? gset(i) : tset(i), f);
    }

    // ── Band helpers ──────────────────────────────────────────────────────────
    void validate_guide(const std::vector<int>& gj) const {
        if (gj.size() != static_cast<size_t>(m_) + 1)
            throw std::invalid_argument(
                "nwgrad: guide_j has " + std::to_string(gj.size()) +
                " entries, but sequence A of length " + std::to_string(m_) +
                " needs " + std::to_string(m_ + 1) +
                " (do the aligned strings spell the input sequences?)");
        for (size_t i = 0; i < gj.size(); ++i) {
            if (gj[i] < 0 || gj[i] > n_)
                throw std::invalid_argument(
                    "nwgrad: guide_j[" + std::to_string(i) + "] = " +
                    std::to_string(gj[i]) + " is outside [0, " +
                    std::to_string(n_) + "] for sequence B of length " +
                    std::to_string(n_));
            if (i > 0 && gj[i] < gj[i - 1])
                throw std::invalid_argument(
                    "nwgrad: guide_j decreases at row " + std::to_string(i) +
                    " (" + std::to_string(gj[i - 1]) + " -> " +
                    std::to_string(gj[i]) + ")");
        }
    }

    int jlo(int i) const noexcept {
        if constexpr (AB == AlignBand::Full) return 1;
        else                                  return std::max(1, guide_j_[i] - band_);
    }
    // Global: the last row's band always reaches n.  Every global path ends at (m, n),
    // but a guide may stop short of it — trailing gaps in A consume B residues without
    // appending an entry (guide_j_from_aligned) — and a band that misses (m, n) left the
    // score and the traceback's start reading a cell no fill had written: 0.0 instead of
    // 10.0 for the optimal path itself (band 0, "ACGTACGTAC-------" vs a 17-mer).
    int jhi(int i) const noexcept {
        if constexpr (AB == AlignBand::Full) return n_;
        else {
            if constexpr (AM == AlignMode::Global) if (i == m_) return n_;
            int hi = guide_j_[i] + band_;
            if (i < m_) hi = std::max(hi, guide_j_[i + 1] - 1 + band_);
            return std::min(n_, hi);
        }
    }
    int jlo0(int i) const noexcept {
        if constexpr (AB == AlignBand::Full) return 0;
        else                                  return std::max(0, guide_j_[i] - band_);
    }
    int jhi0(int i) const noexcept {
        if constexpr (AB == AlignBand::Full) return n_;
        else {
            if constexpr (AM == AlignMode::Global) if (i == m_) return n_;   // see jhi
            int hi = guide_j_[i] + band_;
            if (i < m_) hi = std::max(hi, guide_j_[i + 1] - 1 + band_);
            return std::min(n_, hi);
        }
    }

    int border_rows() const noexcept {
        if constexpr (AB == AlignBand::Full) return m_;
        else {
            int bi = 0;
            while (bi < m_ && guide_j_[bi + 1] <= band_) ++bi;
            return bi;
        }
    }
    int border_cols() const noexcept {
        if constexpr (AB == AlignBand::Full) return n_;
        else                                  return jhi(0);
    }

    // Inclusive column window [lo, hi] that must be initialised for row i in
    // GuideBanded mode.  It is the union of row i's band with its neighbours'
    // bands, widened by one column on each side.  The neighbour union covers the
    // cells row i±1 read from / write into row i (the recurrence reads row i-1,
    // the backward pass writes into row i-1); the one-column margin covers the
    // NEG_INF guard cell just left of each band edge, so every band-edge read
    // lands on an initialised cell rather than stale data from a previous problem.
    void band_row_span(int i, int& lo, int& hi) const noexcept {
        lo = jlo0(i); hi = jhi0(i);
        if (i > 0)  { lo = std::min(lo, jlo0(i - 1)); hi = std::max(hi, jhi0(i - 1)); }
        if (i < m_) { lo = std::min(lo, jlo0(i + 1)); hi = std::max(hi, jhi0(i + 1)); }
        lo = std::max(0,  lo - 1);
        hi = std::min(n_, hi + 1);
    }

    // GuideBanded: NEG_INF-fill only the band region (O(m·band)).  Full: no-op —
    // the linear forward tables need no pre-fill (every in-band cell is written).
    template <class V> void banded_fill(V& vec) const {
        if constexpr (AB == AlignBand::GuideBanded) {
            for (int i = 0; i <= m_; ++i) {
                int lo, hi; band_row_span(i, lo, hi);
                auto* row = vec.data() + static_cast<size_t>(i) * stride_;
                std::fill(row + lo, row + hi + 1, nw_ninf<typename V::value_type>);
            }
        }
    }

    // Fill `vec` with `value`.  Full: the whole (m+1)×(n+1) table.  GuideBanded:
    // only the band region (O(m·band)).  Element-generic: `value` is the table's own
    // element type (NINF for the T tables, NEG_INF for the double soft tables).
    template <class V> void band_fill(V& vec, typename V::value_type value) const {
        if constexpr (AB == AlignBand::Full) {
            std::fill(vec.begin(), vec.begin() + static_cast<ptrdiff_t>(sz_), value);
        } else {
            for (int i = 0; i <= m_; ++i) {
                int lo, hi; band_row_span(i, lo, hi);
                auto* row = vec.data() + static_cast<size_t>(i) * stride_;
                std::fill(row + lo, row + hi + 1, value);
            }
        }
    }

    // ── guide_j extraction helpers ────────────────────────────────────────────

    static void fill_guide_gaps(std::vector<int>& gj) {
        int n = static_cast<int>(gj.size());
        int lo = 0;
        while (lo < n) {
            if (gj[lo] < 0) { ++lo; continue; }
            int hi = lo + 1;
            while (hi < n && gj[hi] < 0) ++hi;
            if (hi == n) {
                for (int k = lo + 1; k < n; ++k) gj[k] = gj[lo];
                break;
            }
            for (int k = lo + 1; k < hi; ++k)
                gj[k] = gj[lo] + (gj[hi] - gj[lo]) * (k - lo) / (hi - lo);
            lo = hi;
        }
    }

    // The four linear walkers (guide_j_linear, traceback_linear, aligned_linear,
    // hard_grad_linear) re-derive each step by exact float equality, like the affine
    // ones — but unlike them they had no border guard.  Column 0 is seeded closed-form,
    // H[i][0] = -i*ge_b, which is NOT bit-equal to the stepwise H[i-1][0] - ge_b when
    // ge_b is not representable (0.1: some i differ by an ULP).  The gap-in-B test then
    // failed on the border, the walk fell through to --j, and read H[i][-1] — off the
    // row: out of bounds in a release build, an assertion under _GLIBCXX_ASSERTIONS.
    // On column 0 only an upward move remains, so `j == 0` now takes it outright, as the
    // affine walkers' "col 0: only upward moves remain" does.  (Row 0 was already safe:
    // both earlier branches need i > 0, so the else takes --j.)
    std::vector<int> guide_j_linear(const DpBuffer& buf) const {
        std::vector<int> gj(static_cast<size_t>(m_ + 1), -1);
        gj[0] = 0;
        gj[static_cast<size_t>(m_)] = n_;
        const int i0 = (AM == AlignMode::Global) ? m_ : best_i_;
        gj[static_cast<size_t>(i0)] = (AM == AlignMode::Global) ? n_ : best_j_;
        walk_linear(buf, [&](int k, int i, int j) {
            if (k == 0)      gj[static_cast<size_t>(i - 1)] = j - 1;
            else if (k == 1) gj[static_cast<size_t>(i - 1)] = j;   // gap in a: no update
        });
        fill_guide_gaps(gj);
        return gj;
    }

    // The one walk behind the four linear walkers: emit(k, i, j) for each move out of
    // (i, j) — k = 0 diagonal, 1 up (gap in B), 2 left (gap in A) — from the optimum
    // back to the start.  The move is read from the recorded code (pointers_, see
    // viterbi_linear_ptr) or re-derived by exact equality on H, in the same order the
    // fill's max chose it, so both give the same walk, ties included.
    template <class Emit>
    void walk_linear(const DpBuffer& buf, Emit&& emit) const {
        if (tr_) walk_linear_impl<true>(buf, emit);
        else     walk_linear_impl<false>(buf, emit);
    }
    // Tr: read each row's extend (row_ge_b) rather than one hoisted value — a template
    // parameter, so the untracked walk is the code it always was.
    template <bool Tr, class Emit>
    void walk_linear_impl(const DpBuffer& buf, Emit& emit) const {
        int i = (AM == AlignMode::Global) ? m_ : best_i_;
        int j = (AM == AlignMode::Global) ? n_ : best_j_;
        while (true) {
            if (i == 0 && j == 0) break;
            int k;
            if (pointers_) {
                k = dense_ ? dense_code(buf.DM, i, j, 1, 0) : dcode(buf.DM, i, j);
                if (k == 3) break;
            } else {
                if constexpr (AM == AlignMode::Local)
                    if (rat(buf.H, i, j) <= 0.0) break;
                if (i > 0 && j > 0 &&
                    rat(buf.H, i, j) == rat(buf.H, i-1, j-1) + (Tr ? subT(i, j) : subTu(i, j)))
                    k = 0;
                else if (i > 0 && (j == 0 || rat(buf.H, i, j) == rat(buf.H, i-1, j) -
                                                 (Tr ? tgeb_[static_cast<size_t>(i)]
                                                     : static_cast<T>(params_->gap_extend_b))))
                    k = 1;
                else
                    k = 2;
            }
            emit(k, i, j);
            if (k == 0)      { --i; --j; }
            else if (k == 1) --i;
            else             --j;
        }
    }

    // ── The affine traceback: one walk, every consumer ───────────────────────
    //
    // Directions (Pointers, variant B) live in whatever layout the fill used -- row-major
    // from the scalar B fill, striped from the leveled B kernel -- so index them through
    // the same cell_index() the score tables use.  One copy of the striping arithmetic.
    // Affine: one byte per cell, M | X << 2 | Y << 4; linear: the move, 0..3.
    unsigned char dcode(const BVec& d, int i, int j) const noexcept {
        return d[cell_index(i, j)];
    }
    // The dense codes (dense_): plane `plane` of `planes` at (i, j); column 0 keeps the
    // one-byte format whole (linear: the move; affine: the packed M | X << 2 | Y << 4).
    unsigned char dense_code(const BVec& d, int i, int j, int planes, int plane) const noexcept {
        const unsigned char* row = d.data() +
            static_cast<size_t>(i) * dense_codes_row_bytes(striped_seg_, striped_w_, planes);
        if (j == 0) return row[0];
        const int s = (j - 1) % striped_seg_, l = (j - 1) / striped_seg_;
        return static_cast<unsigned char>(dense_code_at(row, striped_seg_, striped_w_, planes, plane, s, l));
    }
    // Whether this fill asks its Pointers kernel for the dense codes (dense_mode_, then
    // NWGRAD_DENSE_CODES, then the gate).
    bool want_dense_() const noexcept {
        if (dense_mode_ >= 0) return dense_mode_ == 1;
        static const int env = [] {
            const char* e = std::getenv("NWGRAD_DENSE_CODES");
            return e ? (std::atoi(e) != 0 ? 1 : 0) : -1;
        }();
        if (env >= 0) return env == 1;
        return dense_gate_();
    }
    // The auto choice: the dense codes from kDenseFromCells{Avx2,Neon} cells a pair, by the
    // level the fill runs at (0 = never).  On AVX2 they cost 2-13 % in cache (more shifting
    // a segment) and win once a thread's code table presses on L2 / L3 -- from a size that
    // hardly moves with the thread count, so the gate is on cells alone; on NEON the
    // answer does not depend on size (see the constants).
    bool dense_gate_() const noexcept {
        const int backend = (backend_ == kBackendAuto) ? global_default_backend() : backend_;
#if defined(__aarch64__) || defined(_M_ARM64)
        const std::size_t from = backend >= 0 ? kDenseFromCellsNeon : 0;
#else
        const std::size_t from = backend == static_cast<int>(SimdLevel::Avx2)   ? kDenseFromCellsAvx2
                               : backend == static_cast<int>(SimdLevel::Avx512) ? kDenseFromCellsAvx512 : 0;
#endif
        return from != 0 &&
               (static_cast<std::size_t>(m_) + 1) * (static_cast<std::size_t>(n_) + 1) >= from;
    }
    // The affine cell's three codes in the one-byte packing, from whichever format holds them.
    unsigned char affine_codes(const BVec& d, int i, int j) const noexcept {
        if (!dense_) return dcode(d, i, j);
        if (j == 0) return dense_code(d, i, 0, 3, 0);
        return static_cast<unsigned char>(dense_code(d, i, j, 3, 0) | dense_code(d, i, j, 3, 1) << 2 |
                                          dense_code(d, i, j, 3, 2) << 4);
    }

    // emit(s, i, j, open) for each move out of (i, j), from the optimum back to the
    // start: s = M diagonal to (i-1, j-1), X up to (i-1, j) (gap in B), Y left to
    // (i, j-1) (gap in A); `open` is true when that gap move opens its run (the state
    // it came from is not the same gap), always false for M.  The predecessor state is
    // read from the code the fill recorded (pointers_) or re-derived from VM/VX/VY by
    // the same `>=` M>X>Y chain the fill ran, in the same T-precision penalties, so the
    // two walks take the same steps, ties included -- the property Hirschberg cannot
    // offer.  VM/VX/VY share one layout, so a move indexes its predecessor cell once
    // (cell_index costs a division in the striped layout) and reads all three there.  (This replaced six hand-written copies of the walk -- guide, gradient,
    // aligned strings and path, each over codes or scores -- whose border fixes had to
    // be made in every copy, and once were not: a missing pointers branch segfaulted.)
    //
    // Global borders: on row 0 or column 0 one move remains all the way to the origin,
    // so the rest of the path is one leading gap run, opened at its last move, out of
    // M(0,0).  It is emitted without reading the tables, because a guide band need not
    // cover the border: guide_j[0] is where the old path LEFT row 0, so after a long
    // leading gap the band misses the origin, and stepping read cells never initialised
    // for this pair (stale data from the thread's previous one: banded_grad gap-open
    // counts that changed with the thread count).  It is the walk a Full DP's border
    // values give (VY(0,k) extends from VY(0,k-1) back to (0,1), which opens), and
    // exactly the codes viterbi_affine_ptr records there.
    //
    // Local borders are free starts: the M stop ends the walk there.  The forced moves
    // below remain for a GuideBanded band narrower than the path needs: the band can
    // fail to admit any route back to the origin, leaving every predecessor at -inf.
    // The M>=X>=Y tie-break then picks M, and M steps DIAGONALLY -- so i or j goes
    // negative, cell_index() turns that into a huge size_t, and the walk reads off the
    // end of the table.  That was a segfault, not a suboptimal score (realign_banded:
    // "A"*22 vs "C"*87 at band<=32).  At a border only one move is legal; forcing it is
    // what a well-formed DP would have chosen anyway.
    template <class Emit>
    void walk_affine(const DpBuffer& buf, Emit&& emit) const {
        if (tr_) {
            if (pointers_) walk_affine_impl<true, true>(buf, emit);
            else           walk_affine_impl<false, true>(buf, emit);
        } else {
            if (pointers_) walk_affine_impl<true, false>(buf, emit);
            else           walk_affine_impl<false, false>(buf, emit);
        }
    }

    // Tr: each move reads the penalties of its own row (a gap in B the row it enters, a
    // gap in A the row it lies in); otherwise the four are hoisted, as they always were.
    template <bool Ptr, bool Tr, class Emit>
    void walk_affine_impl(const DpBuffer& buf, Emit& emit) const {
        auto from_code = [](unsigned char c) {
            return c == 1 ? TBTable::X : (c == 2 ? TBTable::Y : TBTable::M);
        };
        auto argmax = [](T vm, T vx, T vy) {
            if (vm >= vx && vm >= vy) return TBTable::M;
            return vx >= vy ? TBTable::X : TBTable::Y;
        };
        T go_a = static_cast<T>(params_->gap_open_a);
        T ge_a = static_cast<T>(params_->gap_extend_a);
        T go_b = static_cast<T>(params_->gap_open_b);
        T ge_b = static_cast<T>(params_->gap_extend_b);

        int i = best_i_, j = best_j_;
        TBTable tbl = best_tbl_;
        while (true) {
            if (i == 0 && j == 0) break;
            // Ptr: the cell's packed byte, read once for whichever state the walk is in
            // (Global: past the border, which is emitted without reading the codes)
            unsigned char c = 0;
            if constexpr (AM == AlignMode::Local) {
                if constexpr (Ptr) c = affine_codes(buf.DM, i, j);
                if (tbl == TBTable::M) {
                    if constexpr (Ptr) { if ((c & 3) == 3) break; }
                    else               { if (rat(buf.VM, i, j) <= 0.0) break; }
                }
            } else {
                if (i == 0) { for (; j > 0; --j) emit(TBTable::Y, 0, j, j == 1); break; }
                if (j == 0) { for (; i > 0; --i) emit(TBTable::X, i, 0, i == 1); break; }
                if constexpr (Ptr) c = affine_codes(buf.DM, i, j);
            }
            if      (i == 0) tbl = TBTable::Y;   // row 0: only leftward moves remain
            else if (j == 0) tbl = TBTable::X;   // col 0: only upward moves remain

            TBTable prev;
            if (tbl == TBTable::M) {
                if constexpr (Ptr) prev = from_code(c & 3);
                else { const size_t k = cell_index(i-1, j-1);
                       prev = argmax(buf.VM[k], buf.VX[k], buf.VY[k]); }
                emit(TBTable::M, i, j, false);
                --i; --j;
            } else if (tbl == TBTable::X) {   // gap in B: advance i, gap_b penalties
                if constexpr (Tr) { go_b = tgob_[static_cast<size_t>(i)]; ge_b = tgeb_[static_cast<size_t>(i)]; }
                if constexpr (Ptr) prev = from_code((c >> 2) & 3);
                else { const size_t k = cell_index(i-1, j);
                       prev = argmax(buf.VM[k] - go_b - ge_b,
                                     buf.VX[k]        - ge_b,
                                     buf.VY[k] - go_b - ge_b); }
                emit(TBTable::X, i, j, prev != TBTable::X);
                --i;
            } else {                          // gap in A: advance j, gap_a penalties
                if constexpr (Tr) { go_a = tgoa_[static_cast<size_t>(i)]; ge_a = tgea_[static_cast<size_t>(i)]; }
                if constexpr (Ptr) prev = from_code(c >> 4);
                else { const size_t k = cell_index(i, j-1);
                       prev = argmax(buf.VM[k] - go_a - ge_a,
                                     buf.VX[k] - go_a - ge_a,
                                     buf.VY[k]        - ge_a); }
                emit(TBTable::Y, i, j, prev != TBTable::Y);
                --j;
            }
            tbl = prev;
        }
    }

    // gj[i] = the column after consuming i residues of A: set by M and X moves (Y moves
    // stay in the row), the end cell, row 0 and row m; fill_guide_gaps interpolates the
    // rows a Local path does not cover.
    void guide_affine_init(std::vector<int>& gj) const {
        gj.assign(static_cast<size_t>(m_ + 1), -1);
        gj[0] = 0;
        gj[static_cast<size_t>(m_)] = n_;
        gj[static_cast<size_t>(best_i_)] = best_j_;
    }

    std::vector<int> guide_j_affine(const DpBuffer& buf) const {
        std::vector<int> gj;
        guide_affine_init(gj);
        walk_affine(buf, [&](TBTable s, int i, int j, bool) {
            if      (s == TBTable::M) gj[static_cast<size_t>(i - 1)] = j - 1;
            else if (s == TBTable::X) gj[static_cast<size_t>(i - 1)] = j;
        });
        fill_guide_gaps(gj);
        return gj;
    }

    // gj != nullptr: also record the guide path in the same walk.  Each gap move charges
    // one extend, plus one open if it opens its run; the score subtracts penalties, so
    // the gap fields count down (see the sign convention).
    void hard_grad_affine(const DpBuffer& buf, AlignParams& grad,
                          std::vector<int>* gj = nullptr) const {
        if (tr_) { hard_grad_affine_impl<true>(buf, grad, gj); return; }
        double* gblk = grad_block(grad);
        if (gj) guide_affine_init(*gj);
        walk_affine(buf, [&](TBTable s, int i, int j, bool open) {
            if (s == TBTable::M) {
                gblk[sub_off_u(i, j)] += 1.0;
                if (gj) (*gj)[static_cast<size_t>(i - 1)] = j - 1;
            } else if (s == TBTable::X) {
                grad.gap_extend_b -= 1.0;
                if (open) grad.gap_open_b -= 1.0;
                if (gj) (*gj)[static_cast<size_t>(i - 1)] = j;
            } else {
                grad.gap_extend_a -= 1.0;
                if (open) grad.gap_open_a -= 1.0;
            }
        });
        if (gj) fill_guide_gaps(*gj);
    }
    // Tr: each count lands in its row's set (sub_off / grad_gap); otherwise the walk
    // the untracked gradient always took, with no per-step branch on tr_.
    template <bool Tr>
    void hard_grad_affine_impl(const DpBuffer& buf, AlignParams& grad, std::vector<int>* gj) const {
        double* gblk = grad_block(grad);
        if (gj) guide_affine_init(*gj);
        walk_affine(buf, [&](TBTable s, int i, int j, bool open) {
            if (s == TBTable::M) {
                gblk[Tr ? sub_off(i, j) : sub_off_u(i, j)] += 1.0;
                if (gj) (*gj)[static_cast<size_t>(i - 1)] = j - 1;
            } else if (s == TBTable::X) {
                if constexpr (Tr) {
                    grad_gap(grad, i, AlignParams::kExtendB) -= 1.0;
                    if (open) grad_gap(grad, i, AlignParams::kOpenB) -= 1.0;
                } else {
                    grad.gap_extend_b -= 1.0;
                    if (open) grad.gap_open_b -= 1.0;
                }
                if (gj) (*gj)[static_cast<size_t>(i - 1)] = j;
            } else {
                if constexpr (Tr) {
                    grad_gap(grad, i, AlignParams::kExtendA) -= 1.0;
                    if (open) grad_gap(grad, i, AlignParams::kOpenA) -= 1.0;
                } else {
                    grad.gap_extend_a -= 1.0;
                    if (open) grad.gap_open_a -= 1.0;
                }
            }
        });
        if (gj) fill_guide_gaps(*gj);
    }

    void aligned_affine(const DpBuffer& buf, std::string& a, std::string& b) const {
        walk_affine(buf, [&](TBTable s, int i, int j, bool) {
            a.push_back(s == TBTable::Y ? '-' : sym_a(i));
            b.push_back(s == TBTable::X ? '-' : sym_b(j));
        });
        std::reverse(a.begin(), a.end());
        std::reverse(b.begin(), b.end());
    }

    // The aligned residue pairs (0-based), start to end.
    std::vector<std::pair<int,int>> traceback_affine(const DpBuffer& buf) const {
        std::vector<std::pair<int,int>> path;
        walk_affine(buf, [&](TBTable s, int i, int j, bool) {
            if (s == TBTable::M) path.emplace_back(i - 1, j - 1);
        });
        std::reverse(path.begin(), path.end());
        return path;
    }

    // ── Backend dispatch ──────────────────────────────────────────────────────
    //
    // The one place in the library where the scalar/simd choice exists.  The branch is
    // taken once per DP, not once per cell.  Every simd level writes tables bit-identical
    // to the scalar one, so nothing downstream can tell them apart.  The linear model has
    // no PER-PAIR simd kernel — deliberately.  Its recurrence collapses to a single carry
    // that is a pure latency chain along the row (~6 cycles/cell), and the scalar loop
    // already runs at ~9 cycles/cell against that floor; a row-wise vectorized linear
    // kernel was written, measured at 0.90x, and deleted.  So any simd backend is a legal
    // request for a linear aligner and runs the scalar fill here.  (Batches of short pairs
    // vectorize linear ACROSS pairs instead — SeqPairBatch fill="interpair", one pair per
    // lane, bit-identical — which sidesteps the chain.)
    void run_viterbi(DpBuffer& buf) {
        // `auto` resolved to Grid where Grid does not apply: this run takes the previous
        // default (kGridFallbackTb) — its memory bound — and tb_ is restored after it.
        struct TbRestore {
            TracebackMode& t; TracebackMode saved;
            ~TbRestore() { t = saved; }
        } tb_restore{tb_, tb_};
        if constexpr (GM == GapModel::Affine && AM == AlignMode::Global && AB == AlignBand::Full &&
                      std::is_floating_point_v<T>)
            if (tb_ == TracebackMode::Grid && tb_auto_) {
                const int be = (backend_ == kBackendAuto) ? global_default_backend() : backend_;
                if (be == kBackendScalar || !grid_ok_(level_kernels(be))) tb_ = kGridFallbackTb;
            }
        // Default to row-major; only the striped Full kernel (run_dispatched_affine) flips
        // this back on.  Every other fill here writes VM/VX/VY row-major.
        tables_striped_ = false;
        pointers_     = false;
        dense_        = false;
        hirschberg_   = false;
        grid_         = false;
        inter_w_      = 0;
        // Hirschberg: divide and conquer, no tables above the base case.  It is a
        // DIFFERENT algorithm with a different tie-break, not a faster spelling of the
        // same one, so an unsupported combination throws instead of falling back — a
        // silent substitution would make a benchmark of it meaningless and a gradient
        // from it unattributable.
        if (is_hirschberg(tb_)) {
            if constexpr (AB != AlignBand::Full)
                throw std::logic_error(
                    "nwgrad: traceback=\"hirschberg\" is implemented for full DP only; a "
                    "guide band already bounds memory, which is the only thing Hirschberg buys");
            else if (hb_splits(tb_, tb_auto_ ? kAutoPointerCells : 0, hb_cutoff_, m_, n_) && !hb_penalties_ok_()) {
                // A NEGATIVE gap penalty breaks the recursion (characterized 2026-10-11,
                // after the boundary-state fix):
                //  * a negative OPEN (either sequence): the join's plain form adds a prefix
                //    ending in a gap to a suffix starting in the same gap — one run, two
                //    opens — which with go < 0 OVERvalues it, and nothing stops the halves
                //    from realizing it (36 % of random Global pairs below the optimum);
                //  * Local with a negative EXTEND: a gap run that gains score makes Local
                //    degenerate, and the fill (no gap on row 0 / column 0) and the box's
                //    global solve (may open one at its corner) resolve it differently —
                //    Hirschberg scores up to ~0.1-3 ABOVE Pointers — and the simd endpoint
                //    scan lets a gap run growing through the padding lanes win, so the box
                //    overruns B (heap-buffer-overflow in hb_scan_impl).
                // Global with negative extends and opens >= 0 is correct (0 / 12000).
                // `auto` runs such a pair as Pointers (exact; only the memory differs); an
                // explicit request raises rather than return a wrong answer.
                if (!tb_auto_)
                    throw std::invalid_argument(
                        "nwgrad: traceback=\"hirschberg\" / \"hirschberg_pmax\" needs non-negative "
                        "gap penalties (got a negative one); use traceback=\"pointers\" or \"auto\"");
            }
            else if (hb_splits(tb_, tb_auto_ ? kAutoPointerCells : 0, hb_cutoff_, m_, n_)) {
                // Linear: one exact sweep for both "hirschberg" and "hirschberg_pmax" —
                // the prefix-max carry exists to remove affine's lazy-F fixpoint, and the
                // scalar linear sweep has none.
                if constexpr (GM == GapModel::Linear) { viterbi_linear_hirschberg(buf); return; }
                else if constexpr (AM == AlignMode::Local) { viterbi_affine_hirschberg_local(buf); return; }
                else { viterbi_affine_hirschberg(buf); return; }
            }
            // A pair no longer than hb_cutoff never splits, so it is run AS Pointers —
            // the same fill, table and traceback, hence bit-exact with traceback=
            // "pointers" by construction.  It used to run the Hirschberg base case
            // instead, which is the Pointers fill in spirit but not in arithmetic: its
            // borders are seeded and carried differently, so with a non-representable
            // gap_extend (0.1) it could settle a float tie on a path one ULP worse, and
            // it replayed the score from the path rather than reading the table — 27/41
            // short pairs differed from Pointers.  Same memory (the base case kept
            // a Pointers-sized table too) and the same speed (measured 0.99-1.04x).
        }
        // Variant B: direction pointers + rolling rows, 1 B/cell instead of 12/24.
        // Full affine only; the banded and linear paths keep their score tables.
        // (the pointer kernels are chosen after the backend is resolved, below: every
        // simd level has its own, and only the scalar path falls back here.)
        if constexpr (GM == GapModel::Linear) {
            // The pointer fill keeps direction bytes, not H, where linear_ptr_fill() says
            // so.  Full: the leveled striped linear kernel (kernels_linear_impl.inl) when
            // the backend has one — bit-exact with the scalar fills below, which remain
            // the scalar backend's, the banded path's and the header-only build's.
            const bool lin_ptr = linear_ptr_fill();
            if constexpr (AB == AlignBand::Full) {
                const int backend = (backend_ == kBackendAuto) ? global_default_backend() : backend_;
                if (backend != kBackendScalar && n_ >= linear_striped_min_n(backend)) {
                    const LevelKernels& K = level_kernels(backend);
                    const TypedKernels<T>& KT = K.template of<T>();
                    if (lin_ptr ? KT.linear_ptr != nullptr : KT.linear != nullptr) {
                        run_dispatched_linear(buf, K, lin_ptr);
                        return;
                    }
                }
            }
            if (lin_ptr) { viterbi_linear_ptr(buf); return; }
            viterbi_linear(buf);
            return;
        }
        // Resolve this aligner's backend: auto -> the global default, then scalar or a level.
        const int backend = (backend_ == kBackendAuto) ? global_default_backend() : backend_;
        // Pointers is Full-affine only; banded/linear always keep score tables.
        // (a Hirschberg mode reaches here only for a short pair — see above)
        const bool use_ptr = (tb_ == TracebackMode::Pointers || is_hirschberg(tb_) ||
                              tb_ == TracebackMode::Grid) && (AB == AlignBand::Full);
        if (backend == kBackendScalar) {
            if (use_ptr) viterbi_affine_ptr(buf); else viterbi_affine(buf);
            return;
        }
        // Grid: where it is implemented (grid_ok_), else Pointers below — the same result.
        if constexpr (GM == GapModel::Affine && AM == AlignMode::Global && AB == AlignBand::Full &&
                      std::is_floating_point_v<T>)
            if (tb_ == TracebackMode::Grid && grid_ok_(level_kernels(backend))) {
                viterbi_affine_grid(buf, level_kernels(backend));
                return;
            }

        // A simd level: its kernels.  If that level's TU was not linked (header-only
        // single-level build), every pointer is null and we fall through to scalar.
        const LevelKernels& K = level_kernels(backend);
        const TypedKernels<T>& KT = K.template of<T>();
        if constexpr (AB == AlignBand::Full) {
            // Opt-in row-wise fill (set_rowwise_full): score tables, row-major, read back
            // by the Scores traceback whatever tb_ says — the paths are identical.
            if (uses_rowwise_full()) { viterbi_affine_simd(buf, K); return; }
            // The striped Full kernel, this level's instantiation for T.
            if (use_ptr && KT.viterbi_ptr) { run_dispatched_affine(buf, K, true);  return; }
            if (KT.viterbi)                { run_dispatched_affine(buf, K, false); return; }
            if (use_ptr) { viterbi_affine_ptr(buf); return; }   // level lacks the ptr kernel
        }
        // The row-wise banded kernel (every T); a level without it falls back to the
        // scalar fill.
        if (KT.banded_row_local) { viterbi_affine_simd(buf, K); return; }
        viterbi_affine(buf);
    }

    // Build a plain ViterbiJob from Aligner state, run the dispatched (leveled) kernel,
    // and copy its scalar results back.  The kernel fills buf.VM/VX/VY in STRIPED layout
    // (table_layout == 1) and reports seg/width; rat/at then read them striped.
    void run_dispatched_affine(DpBuffer& buf, const LevelKernels& K, bool use_ptr = false) {
        ViterbiJob<T> job{};
        job.a = a_idx_.data(); job.m = m_;
        job.b = b_idx_.data(); job.n = n_;
        // blkT_ is the substitution block already in the Viterbi precision T (= blk_ when
        // T is double); the penalties convert to T on assignment.
        job.blk = blkT_;       job.nalpha = nalpha_;
        job.go_a = static_cast<T>(params_->gap_open_a);   job.ge_a = static_cast<T>(params_->gap_extend_a);
        job.go_b = static_cast<T>(params_->gap_open_b);   job.ge_b = static_cast<T>(params_->gap_extend_b);
        job.align_mode = (AM == AlignMode::Local) ? 1 : 0;
        job.align_band = 0; job.band = 0;
        job.guide_j = nullptr; job.guide_len = 0;
        job.buf = &buf;
        if (tr_) fill_track_job_(job.tr, true, static_cast<size_t>(n_) + 16);
        const TypedKernels<T>& KT = K.template of<T>();
        job.dense = (use_ptr && want_dense_()) ? 1 : 0;
        if (use_ptr) KT.viterbi_ptr(job); else KT.viterbi(job);
        pointers_ = use_ptr;   // Pointers leaves codes, not score tables
        dense_ = use_ptr && job.dense;
        // Adopt the layout the kernel produced (striped for the Full striped kernel).
        if (job.table_layout == 1) {
            tables_striped_ = true;
            striped_seg_ = job.seg; striped_w_ = job.width;
            striped_rs_ = job.rstride ? job.rstride
                                      : (static_cast<size_t>(job.seg) + 1) * static_cast<size_t>(job.width);
        } else {
            tables_striped_ = false;
        }
        viterbi_score_ = job.score;
        best_i_ = job.best_i; best_j_ = job.best_j;
        best_tbl_ = (job.best_tbl == 0) ? TBTable::M
                  : (job.best_tbl == 1) ? TBTable::X : TBTable::Y;
    }

    // Below these B lengths the scalar linear fill beats the striped kernel: each row pays
    // a fixed cost (two passes, the carry's cross-lane step, the profile row) that few
    // columns cannot amortize.  Measured, striped / scalar time reaching 1.0, own fill,
    // hard, 1 thread, protein and DNA, nighthaven (AVX2) and spot (NEON) agreeing within
    // one fixture step (2026-10-23): Global n ~48 int16, ~64 int32, ~80 float32, ~96
    // double; Local ~32, ~40, ~24, ~24.  A column count, nearly independent of the ISA:
    // the 22 x 50 DNA cells fall on the same lines.  (int64 has no striped linear kernel
    // below AVX-512 at all: kernels_linear_impl.inl, linear_kernel_pays.)
    static constexpr int kLinearStripedMinN =
        (AM == AlignMode::Global)
            ? (std::is_same_v<T, int16_t> ? 48 : std::is_same_v<T, int32_t> ? 64
               : std::is_same_v<T, float> ? 80 : 96)
            : (std::is_same_v<T, int16_t> ? 32 : std::is_same_v<T, int32_t> ? 40 : 24);
    // AVX-512 (solace, Tiger Lake, gcc 15.2, 2026-10-23): the lanes double AVX2's and some
    // crossovers move a step later — striped / scalar at the shared thresholds: int16 Global
    // n 48 1.13, Local 32 1.18; float32 Local 24 1.36; int64 Local 24 1.50, 32 1.20 — each
    // winning a step later (int16 Global 64 0.79, Local 48 0.84; float32 Local 32 0.89;
    // int64 Local 48 0.88).  The rest keep the shared values (crossing at or before them).
    static constexpr int kLinearStripedMinN512 =
        (AM == AlignMode::Global)
            ? (std::is_same_v<T, int16_t> ? 56 : kLinearStripedMinN)
            : (std::is_same_v<T, int16_t> ? 40 : std::is_same_v<T, float> ? 30
               : std::is_same_v<T, int64_t> ? 44 : kLinearStripedMinN);

    // The leveled linear kernel (Full): H in STRIPED layout (or direction bytes in it),
    // read back through cell_index like the affine tables.  No gap opens in the job.
    void run_dispatched_linear(DpBuffer& buf, const LevelKernels& K, bool use_ptr) {
        ViterbiJob<T> job{};
        job.a = a_idx_.data(); job.m = m_;
        job.b = b_idx_.data(); job.n = n_;
        job.blk = blkT_;       job.nalpha = nalpha_;
        job.go_a = static_cast<T>(0); job.ge_a = static_cast<T>(params_->gap_extend_a);
        job.go_b = static_cast<T>(0); job.ge_b = static_cast<T>(params_->gap_extend_b);
        job.align_mode = (AM == AlignMode::Local) ? 1 : 0;
        job.align_band = 0; job.band = 0;
        job.guide_j = nullptr; job.guide_len = 0;
        job.buf = &buf;
        if (tr_) fill_track_job_(job.tr, true, static_cast<size_t>(n_) + 16, true);
        const TypedKernels<T>& KT = K.template of<T>();
        job.dense = (use_ptr && want_dense_()) ? 1 : 0;
        if (use_ptr) KT.linear_ptr(job); else KT.linear(job);
        pointers_ = use_ptr;
        dense_ = use_ptr && job.dense;
        if (job.table_layout == 1) {
            tables_striped_ = true;
            striped_seg_ = job.seg; striped_w_ = job.width;
            striped_rs_ = job.rstride ? job.rstride
                                      : (static_cast<size_t>(job.seg) + 1) * static_cast<size_t>(job.width);
        } else {
            tables_striped_ = false;
        }
        viterbi_score_ = job.score;
        best_i_ = job.best_i; best_j_ = job.best_j;
        best_tbl_ = TBTable::M;
    }

    // ── Simd kernel — defined out-of-line in aligner_simd.hpp ─────────────────
    //
    // Declared here, defined there, and that header is included at the bottom of
    // this one so both are always visible at the point of instantiation.  The
    // vectorization machinery (query profile, lazy-F) stays out of this file
    // entirely; K supplies the row-wise leaf kernels for the active ISA level.
    void viterbi_affine_simd(DpBuffer& buf, const LevelKernels& K);

    // Substitution scores for row i, contiguous in j over [lo, hi] — a vector
    // load, not a gather.  Full mode serves a slice of the prebuilt query
    // profile; banded mode gathers the row's short span into buf.subbuf, since a
    // full-width profile would outcost the banded DP.  Defined in aligner_simd.hpp.
    void build_profile(DpBuffer& buf);
    bool rw_prof_ = false;   // tracked row-wise Full: build_profile kept a compact profile
    const T* subrow(DpBuffer& buf, int i, int lo, int hi) const;

    // ═════════════════════════════════════════════════════════════════════════
    // Viterbi — Linear gap model
    // ═════════════════════════════════════════════════════════════════════════

    // In precision T throughout, like the affine fill: the walkers below re-derive
    // each step by exact equality against this table, so they must round where it
    // rounded.  It used to add in double (sub(), the raw double gap costs) and store
    // into a float table at T=float; the walkers' double recomputation then rarely
    // matched the stored float for a non-representable cost, every test fell through
    // to "gap in A", and the returned path was not the one scored (-18.7 vs -7.9).
    // At T=double every cast here is the identity: results are unchanged bit for bit.
    void viterbi_linear(DpBuffer& buf) {
        if (tr_) viterbi_linear_impl<true>(buf);
        else     viterbi_linear_impl<false>(buf);
    }
    // Tr: each row's penalties and substitution row from the track arrays (the same
    // expressions with the row's values, so equal sets reproduce the untracked fill).
    template <bool Tr>
    void viterbi_linear_impl(DpBuffer& buf) {
        banded_fill(buf.H);

        if constexpr (AM == AlignMode::Global) {
            at(buf.H, 0, 0) = 0.0;
            const int bi = border_rows(), bj = border_cols();
            if constexpr (Tr) {
                for (int i = 1; i <= bi; ++i) at(buf.H, i, 0) = tbx0_[static_cast<size_t>(i)];
                for (int j = 1; j <= bj; ++j) at(buf.H, 0, j) = -static_cast<T>(j) * tgea_[0];
            } else {
                for (int i = 1; i <= bi; ++i) at(buf.H, i, 0) = -static_cast<T>(i) * static_cast<T>(params_->gap_extend_b);
                for (int j = 1; j <= bj; ++j) at(buf.H, 0, j) = -static_cast<T>(j) * static_cast<T>(params_->gap_extend_a);
            }
        } else {
            for (int i = 0; i <= m_; ++i) at(buf.H, i, 0) = 0.0;
            for (int j = 0; j <= n_; ++j) at(buf.H, 0, j) = 0.0;
        }

        best_i_ = 0; best_j_ = 0;
        T best_local = T(0);
        // Row pointers, not rat()/at(): run_viterbi cleared the layout flags, so H is
        // row-major here, and cell_index()'s layout switch inside the loop is dead weight
        // that gcc unswitches and clang does not (clang/gcc 1.77x on this fill).  The
        // gap costs and the substitution row are hoisted per row for the same reason;
        // same operands, same order, so the table is unchanged bit for bit.
        const uint8_t* bcode = b_idx_.data();
        for (int i = 1; i <= m_; ++i) {
            T ge_b_r, ge_a_r;
            const T* srow;
            if constexpr (Tr) {
                ge_b_r = tgeb_[static_cast<size_t>(i)]; ge_a_r = tgea_[static_cast<size_t>(i)];
                srow = blkT_ + static_cast<size_t>(rs_[static_cast<size_t>(i) - 1]) * static_cast<size_t>(nalpha_);
            } else {
                ge_b_r = static_cast<T>(params_->gap_extend_b);   // gap in B (advance i)
                ge_a_r = static_cast<T>(params_->gap_extend_a);   // gap in A (advance j)
                srow = blkT_ + static_cast<size_t>(a_idx_[static_cast<size_t>(i) - 1]) * static_cast<size_t>(nalpha_);
            }
            const T* hp = buf.H.data() + static_cast<size_t>(i - 1) * stride_;
            T*       hc = buf.H.data() + static_cast<size_t>(i) * stride_;
            const int j0 = jlo(i), j1 = jhi(i);
            T rowmax = static_cast<T>(0);   // Local: see local_row_best_
            for (int j = j0; j <= j1; ++j) {
                T v = std::max({
                    hp[j-1] + srow[bcode[static_cast<size_t>(j) - 1]],
                    hp[j]   - ge_b_r,
                    hc[j-1] - ge_a_r,
                });
                if constexpr (AM == AlignMode::Local) {
                    v = std::max(v, static_cast<T>(0));
                    rowmax = std::max(rowmax, v);
                }
                hc[j] = v;
            }
            if constexpr (AM == AlignMode::Local)
                if (rowmax > best_local) {
                    int j = j0;
                    while (!(hc[j] == rowmax)) ++j;
                    best_local = rowmax; best_i_ = i; best_j_ = j;
                }
        }

        viterbi_score_ = (AM == AlignMode::Global) ? rat(buf.H, m_, n_) : best_local;
    }

    // TracebackMode::Pointers for linear gaps (Full band): viterbi_linear's fill, in the
    // same precision and the same order, retaining one direction byte per cell (buf.DM,
    // row-major) and two rolling rows instead of H — 1 B/cell against 4 (float32) / 8
    // (double), and no O(m*n) score table at all.  Codes: 0 diagonal, 1 up (gap in B),
    // 2 left (gap in A), 3 stop.  Bit-exact with the H walk, ties included: the max below
    // keeps the FIRST largest of (diag+s, up-ge_b, left-ge_a), exactly as std::max over
    // the initializer list does, and the H walk tests the same three in the same order by
    // equality — so both pick the first largest.  Local: stop where the clamped value is
    // <= 0 (the H walk's stop test); borders stop (Local), go up / left (Global).
    void viterbi_linear_ptr(DpBuffer& buf) {
        if (tr_) viterbi_linear_ptr_impl<true>(buf);
        else     viterbi_linear_ptr_impl<false>(buf);
    }
    template <bool Tr>
    void viterbi_linear_ptr_impl(DpBuffer& buf) {
        const size_t st = stride_;
        const size_t dsz = static_cast<size_t>(m_ + 1) * st;
        if (buf.DM.size() < dsz) buf.DM.resize(dsz);
        if (buf.rM.size() < st) { buf.rM.resize(st); buf.qM.resize(st); }
        T* prev = buf.qM.data(); T* cur = buf.rM.data();
        unsigned char* D = buf.DM.data();
        T ge_a = static_cast<T>(params_->gap_extend_a);
        T ge_b = static_cast<T>(params_->gap_extend_b);
        if constexpr (Tr) ge_a = tgea_[0];
        constexpr bool local = AM == AlignMode::Local;
        prev[0] = static_cast<T>(0); D[0] = 3;
        for (int j = 1; j <= n_; ++j) {
            prev[j] = local ? static_cast<T>(0) : -static_cast<T>(j) * ge_a;
            D[j] = local ? 3 : 2;
        }
        int bi = 0, bj = 0;   // locals: best_i_/best_j_ are members (see below)
        const int n = n_;     // ditto: a member bound is reloaded after every byte store
        T best_local = T(0);
        const unsigned char* __restrict bcode = b_idx_.data();
        for (int i = 1; i <= m_; ++i) {
            unsigned char* __restrict d = D + static_cast<size_t>(i) * st;
            // Locals, not members: the byte stores below may alias anything, so a member
            // read in the loop (subT's blkT_/a_idx_/b_idx_) would be reloaded every cell.
            const T* __restrict prow;
            if constexpr (Tr) {
                prow = blkT_ + static_cast<size_t>(rs_[static_cast<size_t>(i) - 1]) * nalpha_;
                ge_a = tgea_[static_cast<size_t>(i)]; ge_b = tgeb_[static_cast<size_t>(i)];
            } else {
                prow = blkT_ + static_cast<size_t>(a_idx_[static_cast<size_t>(i) - 1]) * nalpha_;
            }
            const T* __restrict pv = prev;
            T* __restrict cv = cur;
            if constexpr (Tr) cv[0] = local ? static_cast<T>(0) : tbx0_[static_cast<size_t>(i)];
            else              cv[0] = local ? static_cast<T>(0) : -static_cast<T>(i) * ge_b;
            d[0] = local ? 3 : 1;
            // Two passes, as gcc -O3 splits viterbi_linear's own loop: the carry-free
            // max(diag+s, up-ge_b) and its code vectorize across the row; only
            // max(t, left-ge_a) stays on the serial chain.  One fused loop (the code on
            // the carry) ran 2.6x slower than the H fill; this, ~1.6x (a third pass
            // reading the codes off the values was slower still: 2x).  The values are the
            // same std::max calls in the same order, so bit-exact.
            for (int j = 1; j <= n; ++j) {
                const T d0 = pv[j - 1] + prow[bcode[j - 1]];
                const T u = pv[j] - ge_b;
                cv[j] = std::max(d0, u);
                d[j] = static_cast<unsigned char>(d0 < u);
            }
            T left = cv[0];
            int hit = 0;
            for (int j = 1; j <= n; ++j) {
                const T t = cv[j], l = left - ge_a;
                T v = std::max(t, l);
                const int tl = t < l;
                unsigned char c = static_cast<unsigned char>(d[j] + tl * (2 - d[j]));
                if constexpr (local) {
                    v = std::max(v, static_cast<T>(0));
#if defined(__GNUC__) && !defined(__clang__) && defined(__aarch64__)
                  if constexpr (std::is_floating_point_v<T>) {
                    // gcc on AArch64 turned every form of the float stop compare into a
                    // branch around the code (out of line, mispredicted where many cells
                    // clamp to 0) and branched on the best cell too: 1.8x clang.  So: stop
                    // where v <= 0, which after the clamp (v >= 0, -0 or NaN) is "the bits
                    // without the sign are 0", an integer test; the best cell as selects,
                    // its row taken once after the row (0.53x).  x86 keeps the plain form:
                    // this one was 5-7 % slower there under both compilers.
                    using U = nw_bits_t<T>;
                    const unsigned z = static_cast<unsigned>((std::bit_cast<U>(v) << 1) == 0);
                    c = static_cast<unsigned char>(c | (3u * z));
                    const bool up = v > best_local;
                    best_local = up ? v : best_local;
                    bj = up ? j : bj;
                    hit |= static_cast<int>(up);
                  } else {   // integer T: the plain form (a `v <= 0` on an int is one compare)
                    c = static_cast<unsigned char>(c | (3u * static_cast<unsigned>(v <= static_cast<T>(0))));
                    if (v > best_local) { best_local = v; bi = i; bj = j; }
                  }
#else
                    // c is 0..2: OR-ing 3 sets the stop code without the branch gcc made
                    c = static_cast<unsigned char>(c | (3u * static_cast<unsigned>(v <= static_cast<T>(0))));
                    if (v > best_local) { best_local = v; bi = i; bj = j; }
#endif
                }
                cv[j] = v; d[j] = c;
                left = v;
            }
            if constexpr (local) bi = hit ? i : bi;   // hit stays 0 on the plain form
            std::swap(prev, cur);
        }
        best_i_ = bi; best_j_ = bj;
        viterbi_score_ = local ? best_local : prev[n_];
        pointers_ = true;
    }

    std::vector<std::pair<int,int>> traceback_linear(const DpBuffer& buf) const {
        std::vector<std::pair<int,int>> path;
        walk_linear(buf, [&](int k, int i, int j) { if (k == 0) path.emplace_back(i - 1, j - 1); });
        std::reverse(path.begin(), path.end());
        return path;
    }

    void aligned_linear(const DpBuffer& buf, std::string& a, std::string& b) const {
        walk_linear(buf, [&](int k, int i, int j) {
            if (k == 0)      { a.push_back(sym_a(i)); b.push_back(sym_b(j)); }
            else if (k == 1) { a.push_back(sym_a(i)); b.push_back('-'); }   // gap in B
            else             { a.push_back('-'); b.push_back(sym_b(j)); }   // gap in A
        });
        std::reverse(a.begin(), a.end());
        std::reverse(b.begin(), b.end());
    }

    // With gj: the guide too, from the same walk (guide_j_linear's, move for move).
    void hard_grad_linear(const DpBuffer& buf, AlignParams& grad, std::vector<int>* gj = nullptr) const {
        double* gblk = grad_block(grad);
        if (gj) {
            gj->assign(static_cast<size_t>(m_ + 1), -1);
            (*gj)[0] = 0;
            (*gj)[static_cast<size_t>(m_)] = n_;
            const int i0 = (AM == AlignMode::Global) ? m_ : best_i_;
            (*gj)[static_cast<size_t>(i0)] = (AM == AlignMode::Global) ? n_ : best_j_;
        }
        if (tr_) {
            walk_linear(buf, [&](int k, int i, int j) {
                if (k == 0) {
                    gblk[sub_off(i, j)] += 1.0;
                    if (gj) (*gj)[static_cast<size_t>(i - 1)] = j - 1;
                } else if (k == 1) {
                    grad_gap(grad, i, AlignParams::kExtendB) -= 1.0;
                    if (gj) (*gj)[static_cast<size_t>(i - 1)] = j;
                } else {
                    grad_gap(grad, i, AlignParams::kExtendA) -= 1.0;
                }
            });
            if (gj) fill_guide_gaps(*gj);
            return;
        }
        walk_linear(buf, [&](int k, int i, int j) {
            if (k == 0) {
                gblk[sub_off_u(i, j)] += 1.0;
                if (gj) (*gj)[static_cast<size_t>(i - 1)] = j - 1;
            } else if (k == 1) {
                grad.gap_extend_b -= 1.0;   // the score subtracts this penalty
                if (gj) (*gj)[static_cast<size_t>(i - 1)] = j;
            } else {
                grad.gap_extend_a -= 1.0;
            }
        });
        if (gj) fill_guide_gaps(*gj);
    }

    // ═════════════════════════════════════════════════════════════════════════
    // Viterbi — Affine gap model
    // ═════════════════════════════════════════════════════════════════════════

    // ── TracebackMode::Pointers: affine Viterbi, scalar reference ─────────────
    //
    // Computes the same DP as viterbi_affine() but retains, instead of three score
    // tables, one byte per cell naming the predecessor each state's forward pass chose,
    // packed `M | X << 2 | Y << 4` (buf.DM).  Scores live in two rolling rows, so the
    // retained footprint is 1 B/cell against 12 (float32) / 24 (double).
    //
    // BIT-EXACTNESS is the whole contract, and it holds for a specific reason: the
    // codes below are produced by the *identical* comparison chain the traceback
    // would otherwise re-derive from the tables — `>=` in M, X, Y order.  Change
    // one `>=` to `>` here and B silently returns a different (still valid)
    // subgradient at ties, which is exactly the defect that disqualified Hirschberg.
    //
    // DM additionally carries code 3 = "VM <= 0 at this cell", which is how Local's
    // `rat(VM,i,j) <= 0` stop survives having no VM table to consult.
    //
    // Borders are recorded too, not just the recurrence interior: the traceback
    // walks column 0 and row 0 (X at (1,0) resolves to M, at (i>1,0) to X), and
    // those cells are border-initialised, never visited by the main loop.
    void viterbi_affine_ptr(DpBuffer& buf) {
        if (tr_) viterbi_affine_ptr_impl<true>(buf);
        else     viterbi_affine_ptr_impl<false>(buf);
    }
    template <bool Tr>
    void viterbi_affine_ptr_impl(DpBuffer& buf) {
        T go_a = static_cast<T>(params_->gap_open_a);
        T ge_a = static_cast<T>(params_->gap_extend_a);
        T go_b = static_cast<T>(params_->gap_open_b);
        T ge_b = static_cast<T>(params_->gap_extend_b);
        if constexpr (Tr) { go_a = tgoa_[0]; ge_a = tgea_[0]; }

        const std::size_t stride = stride_;   // row-major, matches cell_index()
        const std::size_t dsz    = static_cast<std::size_t>(m_ + 1) * stride;
        if (buf.DM.size() < dsz) buf.DM.resize(dsz);
        // Per vector, not keyed on rM: viterbi_linear_ptr grows only rM and qM.
        for (auto* r : {&buf.rM, &buf.rX, &buf.rY, &buf.qM, &buf.qX, &buf.qY})
            if (r->size() < stride) r->resize(stride);
        T* pM = buf.qM.data(); T* pX = buf.qX.data(); T* pY = buf.qY.data();  // row i-1
        T* cM = buf.rM.data(); T* cX = buf.rX.data(); T* cY = buf.rY.data();  // row i
        unsigned char* D = buf.DM.data();   // M | X << 2 | Y << 4 per cell

        // Same three-way argmax the traceback runs, in one place so the two cannot drift.
        // Selects between constants on one compare each (the values are finite or -inf,
        // never NaN, so vm >= max(vx, vy) is vm >= vx && vm >= vy): gcc compiled the
        // && / ?: form to branches, mispredicted on real alignments — its scalar pointer
        // fill ran 2-3x slower than clang's selects.
        auto pick = [](T vm, T vx, T vy) -> unsigned char {
            const unsigned char kxy = vx >= vy ? 1 : 2;
            return vm >= std::max(vx, vy) ? static_cast<unsigned char>(0) : kxy;
        };

        // ── row 0 ────────────────────────────────────────────────────────────
        for (std::size_t j = 0; j <= static_cast<std::size_t>(n_); ++j) {
            if constexpr (AM == AlignMode::Local) { pM[j] = 0; pX[j] = NINF; pY[j] = NINF; }
            else {
                pM[j] = (j == 0) ? static_cast<T>(0) : NINF;
                pX[j] = NINF;
                pY[j] = (j == 0) ? NINF : -(go_a + static_cast<T>(j) * ge_a);
            }
            // row 0 is a start (M 3, X 0); Y along it extends leftward: (0,1) opens from
            // M(0,0), the rest extend Y.
            D[j] = (j <= 1) ? 3 : 3 | 2 << 4;
        }

        best_i_ = 0; best_j_ = 0; best_tbl_ = TBTable::M;
        T best_local = static_cast<T>(0);

        for (int i = 1; i <= m_; ++i) {
            const std::size_t rb = static_cast<std::size_t>(i) * stride;
            const T* srow = nullptr;
            if constexpr (Tr) {
                go_a = tgoa_[static_cast<size_t>(i)]; ge_a = tgea_[static_cast<size_t>(i)];
                go_b = tgob_[static_cast<size_t>(i)]; ge_b = tgeb_[static_cast<size_t>(i)];
                srow = blkT_ + static_cast<size_t>(rs_[static_cast<size_t>(i) - 1]) * static_cast<size_t>(nalpha_);
            }
            // column 0 of this row
            if constexpr (AM == AlignMode::Local) { cM[0] = 0; cX[0] = NINF; }
            else if constexpr (Tr) { cM[0] = NINF; cX[0] = tbx0_[static_cast<size_t>(i)]; }
            else { cM[0] = NINF; cX[0] = -(go_b + static_cast<T>(i) * ge_b); }
            cY[0] = NINF;
            D[rb] = (i <= 1) ? 3 : 3 | 1 << 2;   // M 3, Y 0; X: (1,0) opens from M(0,0), deeper rows extend
            T rowmax = static_cast<T>(0);   // Local: see local_row_best_

            for (int j = 1; j <= n_; ++j) {
                const std::size_t k = static_cast<std::size_t>(j);
                const T dgM = pM[k-1], dgX = pX[k-1], dgY = pY[k-1];
                T m_val;
                if constexpr (Tr) m_val = std::max({dgM, dgX, dgY}) + srow[b_idx_[k - 1]];
                else              m_val = std::max({dgM, dgX, dgY}) + subTu(i, j);
                unsigned char km = pick(dgM, dgX, dgY);

                const T ux = pM[k] - go_b - ge_b, vx = pX[k] - ge_b, wx = pY[k] - go_b - ge_b;
                const T x_val = std::max({ux, vx, wx});
                const unsigned char kx = pick(ux, vx, wx);

                const T uy = cM[k-1] - go_a - ge_a, vy = cX[k-1] - go_a - ge_a, wy = cY[k-1] - ge_a;
                const T y_val = std::max({uy, vy, wy});
                const unsigned char ky = pick(uy, vy, wy);

                if constexpr (AM == AlignMode::Local) {
                    m_val = std::max(m_val, static_cast<T>(0));
                    rowmax = std::max(rowmax, std::max({m_val, x_val, y_val}));
                    // traceback stops here; km is 0..2, so OR-ing 3 sets 3 (no branch: gcc
                    // branched on the sign, a coin toss on unrelated sequences)
                    km = static_cast<unsigned char>(km | (3u * static_cast<unsigned>(m_val <= static_cast<T>(0))));
                }

                cM[k] = m_val; cX[k] = x_val; cY[k] = y_val;
                D[rb + k] = static_cast<unsigned char>(km | kx << 2 | ky << 4);
            }
            if constexpr (AM == AlignMode::Local)
                if (rowmax > best_local) local_row_best_(cM, cX, cY, 1, rowmax, i, best_local);
            std::swap(pM, cM); std::swap(pX, cX); std::swap(pY, cY);
        }

        if constexpr (AM == AlignMode::Global) {
            const std::size_t k = static_cast<std::size_t>(n_);
            const T vm = pM[k], vx = pX[k], vy = pY[k];       // pM is row m_ after the swap
            viterbi_score_ = std::max({vm, vx, vy});
            best_i_ = m_; best_j_ = n_;
            if      (vm >= vx && vm >= vy) best_tbl_ = TBTable::M;
            else if (vx >= vy)             best_tbl_ = TBTable::X;
            else                            best_tbl_ = TBTable::Y;
        } else {
            viterbi_score_ = best_local;
        }
        pointers_ = true;
    }

    // ── Hirschberg / Myers-Miller: linear space, no retained tables ───────────
    //
    // Everything below works on a SUB-RECTANGLE of the DP: rows a[i0..i1), columns
    // b[j0..j1), in the sequences' own 0-based coordinates.  Two boundary flags carry
    // the affine state across a cut, and they are the whole reason this is Myers-Miller
    // rather than plain Hirschberg:
    //
    //   in_x   the path is ALREADY inside a gap-in-b run when it enters this block, so
    //          the block's first X move extends (-ge_b) instead of opening (-go_b-ge_b).
    //   out_x  the path is still inside such a run when it leaves.
    //
    // Only X can straddle a row cut.  A Y move stays within its row, and the cut is
    // between rows, so a Y run is wholly on one side of it — which is why there is no
    // in_y/out_y.  Getting this wrong does not crash; it silently charges one extra
    // gap open per crossing, so the score comes out slightly low.  The score assertion
    // in the tests is what actually pins it.
    //
    // A block is entered in one of two states (in_x), so the DP's origin is seeded at
    // whichever of M/X is legal: M[0][0]=0 for a fresh arrival, X[0][0]=0 to continue a
    // run.  That single seed is the entire mechanism.

    // Grow a scratch row vector to hold `n` elements.
    static void hb_fit(typename DpBufferT<T>::TVec& v, std::size_t n) {
        if (v.size() < n) v.resize(n);
    }

    // The leveled striped sweep for this aligner's backend, or null to go scalar.
    // Resolved per sweep rather than cached: it is one predicated load against an
    // O(H*ncols) sweep, and caching it would have to be invalidated on set_kernel().
    // True when this aligner's sweep uses the closed-form prefix-max carry rather than
    // the exact serial one.  Only the SWEEP differs: the recursion, the join, the base
    // case and row 0 are shared, which is why this is a predicate and not a second driver.
    bool hb_pmax() const noexcept { return tb_ == TracebackMode::HirschbergPmax; }

    const LevelKernels* hb_level() const noexcept {
        const int backend = (backend_ == kBackendAuto) ? global_default_backend() : backend_;
        if (backend == kBackendScalar) return nullptr;
        const LevelKernels& K = level_kernels(backend);
        const TypedKernels<T>& KT = K.template of<T>();
        const bool have = hb_pmax() ? KT.hb_sweep_pmax != nullptr : KT.hb_sweep != nullptr;
        return have ? &K : nullptr;
    }

    // Run one half-sweep through the leveled kernel.  Forward and reverse differ only in
    // the walk direction, so they are the same call with the steps negated.
    void hb_run_level(const LevelKernels& K, DpBuffer& buf, int a_start, int a_step,
                      int b_start, int b_step, int H, int ncols, bool in_x,
                      T* oM, T* oX, T* oY) const {
        HbJob<T> job{};
        job.a = a_idx_.data(); job.b = b_idx_.data();
        job.blk = blkT_; job.nalpha = nalpha_;
        job.go_a = static_cast<T>(params_->gap_open_a);
        job.ge_a = static_cast<T>(params_->gap_extend_a);
        job.go_b = static_cast<T>(params_->gap_open_b);
        job.ge_b = static_cast<T>(params_->gap_extend_b);
        job.a_start = a_start; job.a_step = a_step;
        job.b_start = b_start; job.b_step = b_step;
        job.H = H; job.ncols = ncols; job.in_x = in_x ? 1 : 0;
        job.buf = &buf;
        job.outM = oM; job.outX = oX; job.outY = oY;
        if (tr_) { hb_reverse_rows_(); fill_track_job_(job.tr, false, 0); }
        const TypedKernels<T>& KT = K.template of<T>();
        if (hb_pmax()) KT.hb_sweep_pmax(job); else KT.hb_sweep(job);
    }

    // Forward sweep of rows (i0, i1], keeping two rolling rows.  On return the three
    // out-pointers hold row i1 of the block, indexed by column offset c = j - j0.
    // Costs (i1-i0)*(j1-j0) cells and O(j1-j0) memory: nothing is retained.
    //
    // Pmax = true swaps the VY carry for the closed-form prefix max — see the long note
    // in hb_kernel_impl.inl.  `rmp` then holds the gap ramp k*ge_a for k = 0..(j1-j0),
    // precomputed by the caller.  It is a LOAD and not a multiply on purpose: an
    // adjacent multiply would let the compiler contract `g + k*ge_a` into an FMA, which
    // rounds once instead of twice and would make this reference disagree with the simd
    // kernel on any -mfma level.  Row 0 stays exact in both variants.
    // Tracked: row r reads its own penalties (hb_go_a etc.) and substitution row, and the
    // pmax ramp is that row's (`rmp` then a scratch of W+1 the sweep refills per row).
    template <bool Pmax = false>
    void hb_fwd(int i0, int i1, int j0, int j1, bool in_x,
                T* pM, T* pX, T* pY, T* cM, T* cX, T* cY,
                T*& oM, T*& oX, T*& oY, const T* rmp = nullptr) const {
        T go_a = hb_go_a(i0), ge_a = hb_ge_a(i0);
        T go_b = static_cast<T>(params_->gap_open_b);
        T ge_b = static_cast<T>(params_->gap_extend_b);
        const int W = j1 - j0;

        // Row i0: the block's origin, then Y moves rightward along it.
        pM[0] = in_x ? NINF : static_cast<T>(0);
        // An inherited X run (in_x) may only CONTINUE: the origin feeds nothing but the
        // first row's column-0 X (below) — see hb_solve's note on boundary states.
        pX[0] = NINF;
        pY[0] = NINF;
        for (int c = 1; c <= W; ++c) {
            pM[c] = NINF;
            pX[c] = NINF;
            pY[c] = std::max({pM[c-1] - go_a - ge_a, pX[c-1] - go_a - ge_a, pY[c-1] - ge_a});
        }

        T* rr = const_cast<T*>(rmp);
        for (int r = i0 + 1; r <= i1; ++r) {
            if (tr_) {
                go_a = hb_go_a(r); ge_a = hb_ge_a(r); go_b = hb_go_b(r, false); ge_b = hb_ge_b(r, false);
                if constexpr (Pmax) for (int k = 0; k <= W; ++k) rr[k] = static_cast<T>(k) * ge_a;
            }
            cM[0] = NINF;
            cX[0] = std::max({pM[0] - go_b - ge_b, pX[0] - ge_b, pY[0] - go_b - ge_b});
            if (in_x && r == i0 + 1) cX[0] = static_cast<T>(0) - ge_b;   // the run continues
            cY[0] = NINF;
            // The running prefix max over Q[k] = g[k] + k*ge_a, seeded with the column-0
            // border folded in as k = -1.  Unused (and cold) when Pmax is false.
            T P = cY[0] - ge_a;
            for (int c = 1; c <= W; ++c) {
                // subT is 1-based over the full sequences; row r consumes a[r-1].
                cM[c] = std::max({pM[c-1], pX[c-1], pY[c-1]}) + (tr_ ? subT(r, j0 + c) : subTu(r, j0 + c));
                cX[c] = std::max({pM[c] - go_b - ge_b, pX[c] - ge_b, pY[c] - go_b - ge_b});
                if constexpr (Pmax) {
                    // g[k] for k = c-1.  Max-then-subtract mirrors the kernel's `ov`
                    // exactly (subtracting a common value is order-preserving, so this is
                    // bit-identical to the three-way form the exact branch uses).
                    const T g = (std::max(cM[c-1], cX[c-1]) - go_a) - ge_a;
                    P = std::max<T>(g + rmp[c-1], P);
                    cY[c] = P - rmp[c-1];
                } else {
                    cY[c] = std::max({cM[c-1] - go_a - ge_a, cX[c-1] - go_a - ge_a, cY[c-1] - ge_a});
                }
            }
            std::swap(pM, cM); std::swap(pX, cX); std::swap(pY, cY);
        }
        oM = pM; oX = pX; oY = pY;      // after the last swap, p holds row i1
    }

    // Reverse sweep of rows [i0, i1), the mirror image of hb_fwd: it aligns the SUFFIX
    // a[r..i1) with b[.. j1), so on return oS[c] is the best score of aligning
    // a[i0..i1) with b[(j0+c)..j1) given the block's FIRST forward move has type S.
    // out_x seeds the far end exactly as in_x seeds the near one.
    // Tracked: the mirror's rows (see hb_reverse_rows_) — reversed row r (consuming a[r])
    // holds DP row r's gaps in A and DP row r+1's rewritten vertical move.
    template <bool Pmax = false>
    void hb_rev(int i0, int i1, int j0, int j1, bool out_x,
                T* pM, T* pX, T* pY, T* cM, T* cX, T* cY,
                T*& oM, T*& oX, T*& oY, const T* rmp = nullptr) const {
        T go_a = hb_go_a(i1), ge_a = hb_ge_a(i1);
        T go_b = static_cast<T>(params_->gap_open_b);
        T ge_b = static_cast<T>(params_->gap_extend_b);
        const int W = j1 - j0;

        // Reversed column offset d = j1 - j, so d grows leftward from the block's end.
        pM[0] = out_x ? NINF : static_cast<T>(0);
        // A run continuing past the corner (out_x) must ENTER it vertically: the corner
        // feeds nothing but the first reversed row's column-0 X (below).
        pX[0] = NINF;
        pY[0] = NINF;
        for (int d = 1; d <= W; ++d) {
            pM[d] = NINF;
            pX[d] = NINF;
            pY[d] = std::max({pM[d-1] - go_a - ge_a, pX[d-1] - go_a - ge_a, pY[d-1] - ge_a});
        }

        T* rr = const_cast<T*>(rmp);
        for (int r = i1 - 1; r >= i0; --r) {
            if (tr_) {
                go_a = hb_go_a(r); ge_a = hb_ge_a(r); go_b = hb_go_b(r + 1, true); ge_b = hb_ge_b(r + 1, true);
                if constexpr (Pmax) for (int k = 0; k <= W; ++k) rr[k] = static_cast<T>(k) * ge_a;
            }
            cM[0] = NINF;
            cX[0] = std::max({pM[0] - go_b - ge_b, pX[0] - ge_b, pY[0] - go_b - ge_b});
            if (out_x && r == i1 - 1) cX[0] = static_cast<T>(0) - ge_b;   // enters the corner vertically
            cY[0] = NINF;
            T P = cY[0] - ge_a;
            for (int d = 1; d <= W; ++d) {
                // Mirrored diagonal: this step pairs a[r] with b[j1-d], i.e. subT(r+1, j1-d+1).
                cM[d] = std::max({pM[d-1], pX[d-1], pY[d-1]}) + (tr_ ? subT(r + 1, j1 - d + 1) : subTu(r + 1, j1 - d + 1));
                cX[d] = std::max({pM[d] - go_b - ge_b, pX[d] - ge_b, pY[d] - go_b - ge_b});
                if constexpr (Pmax) {
                    const T g = (std::max(cM[d-1], cX[d-1]) - go_a) - ge_a;
                    P = std::max<T>(g + rmp[d-1], P);
                    cY[d] = P - rmp[d-1];
                } else {
                    cY[d] = std::max({cM[d-1] - go_a - ge_a, cX[d-1] - go_a - ge_a, cY[d-1] - ge_a});
                }
            }
            std::swap(pM, cM); std::swap(pX, cX); std::swap(pY, cY);
        }
        oM = pM; oX = pX; oY = pY;
    }

    // Base case: a block short enough to solve outright.  This is the Pointers fill,
    // restricted to the sub-rectangle and seeded by in_x/out_x, recording one direction
    // byte per cell per state and walking them back.  Appends the block's moves to hops_
    // in forward order.  Memory is 3*(rows+1)*(cols+1) bytes, bounded by hb_cutoff_.
    // The leveled base-case kernel for this backend, or null to go scalar.  Registered
    // together with hb_sweep, but checked separately so a partially-populated level table
    // cannot dispatch one without the other.
    const LevelKernels* hb_base_level() const noexcept {
        const int backend = (backend_ == kBackendAuto) ? global_default_backend() : backend_;
        if (backend == kBackendScalar) return nullptr;
        const LevelKernels& K = level_kernels(backend);
        return K.template of<T>().hb_base ? &K : nullptr;
    }

    // Vectorized base case: the striped kernel fills buf.hbD (packed codes, striped)
    // and returns seg/width + the final-cell scores; the walk-back below reads them
    // striped.  Bit-identical to the scalar fill, so which one runs is a speed choice.
    void hb_base_striped_run(const LevelKernels& K, DpBuffer& buf,
                             int i0, int i1, int j0, int j1, bool in_x, bool out_x) {
        const int H = i1 - i0, NC = j1 - j0;
        HbBaseJob<T> job{};
        job.a = a_idx_.data(); job.b = b_idx_.data();
        job.blk = blkT_; job.nalpha = nalpha_;
        job.go_a = static_cast<T>(params_->gap_open_a);
        job.ge_a = static_cast<T>(params_->gap_extend_a);
        job.go_b = static_cast<T>(params_->gap_open_b);
        job.ge_b = static_cast<T>(params_->gap_extend_b);
        job.a_start = i0; job.a_step = 1;      // row r=1..H consumes a[i0 + (r-1)]
        job.b_start = j0; job.b_step = 1;      // col c=1..NC pairs   b[j0 + (c-1)]
        job.H = H; job.ncols = NC; job.in_x = in_x ? 1 : 0; job.buf = &buf;
        if (tr_) fill_track_job_(job.tr, false, 0);
        K.template of<T>().hb_base(job);

        // Striped walk-back over buf.hbD.  Column c lives at slot (c==0 ? 0 : W+scol(c)).
        const int seg = job.seg, Wl = job.width;
        const std::size_t rowsz = (std::size_t)(seg + 1) * Wl;
        const std::size_t off   = (std::size_t)Wl;
        const unsigned char* D = buf.hbD.data();   // M | X << 2 | Y << 4
        auto idx = [&](int r, int c) -> std::size_t {
            return (std::size_t)r * rowsz + (c == 0 ? 0 : off + (std::size_t)((c - 1) % seg) * Wl + (c - 1) / seg);
        };
        int r = H, c = NC;
        TBTable tbl;
        if (out_x) tbl = TBTable::X;
        else tbl = (job.fM >= job.fX && job.fM >= job.fY) ? TBTable::M
                 : ((job.fX >= job.fY) ? TBTable::X : TBTable::Y);
        const std::size_t first = hops_.size();
        while (r > 0 || c > 0) {
            const unsigned char p = D[idx(r, c)];
            unsigned char code;
            if (tbl == TBTable::M)      { code = p & 3;        hops_.push_back(0); --r; --c; }
            else if (tbl == TBTable::X) { code = (p >> 2) & 3; hops_.push_back(1); --r; }
            else                        { code = p >> 4;       hops_.push_back(2); --c; }
            tbl = (code == 1) ? TBTable::X : (code == 2 ? TBTable::Y : TBTable::M);
        }
        std::reverse(hops_.begin() + static_cast<std::ptrdiff_t>(first), hops_.end());
    }

    void hb_base(DpBuffer& buf, int i0, int i1, int j0, int j1,
                 bool in_x, bool out_x) {
        T go_a = hb_go_a(i0), ge_a = hb_ge_a(i0);
        T go_b = static_cast<T>(params_->gap_open_b);
        T ge_b = static_cast<T>(params_->gap_extend_b);
        const int H = i1 - i0, W = j1 - j0;
        if (H == 0 && W == 0) return;
        if (const LevelKernels* K = hb_base_level()) {
            hb_base_striped_run(*K, buf, i0, i1, j0, j1, in_x, out_x);
            return;
        }

        const std::size_t stride = static_cast<std::size_t>(W) + 1;
        hb_fit_b(buf.hbD, static_cast<std::size_t>(H + 1) * stride);
        unsigned char* D = buf.hbD.data();   // M | X << 2 | Y << 4, as the striped kernel's
        auto pack = [](unsigned km, unsigned kx, unsigned ky) {
            return static_cast<unsigned char>(km | kx << 2 | ky << 4);
        };

        // Score rows: two rolling is not enough here because Y reads the current row,
        // but M/X read only the row above — the same shape as the Pointers fill.
        hb_fit(buf.hfa, stride); hb_fit(buf.hfb, stride); hb_fit(buf.hfc, stride);
        hb_fit(buf.hfd, stride); hb_fit(buf.hfe, stride); hb_fit(buf.hff, stride);
        T* pM = buf.hfa.data(); T* pX = buf.hfb.data(); T* pY = buf.hfc.data();
        T* cM = buf.hfd.data(); T* cX = buf.hfe.data(); T* cY = buf.hff.data();

        auto pick = [](T vm, T vx, T vy) -> unsigned char {
            if (vm >= vx && vm >= vy) return 0;
            return (vx >= vy) ? 1 : 2;
        };

        pM[0] = in_x ? NINF : static_cast<T>(0);
        pX[0] = NINF;   // in_x: only the first row's column-0 X continues it
        pY[0] = NINF;
        D[0] = pack(3, 3, 3);
        for (int c = 1; c <= W; ++c) {
            pM[c] = NINF;
            pX[c] = NINF;
            const T uy = pM[c-1] - go_a - ge_a, vy = pX[c-1] - go_a - ge_a, wy = pY[c-1] - ge_a;
            pY[c] = std::max({uy, vy, wy});
            D[c] = pack(3, 3, pick(uy, vy, wy));
        }

        for (int r = 1; r <= H; ++r) {
            const std::size_t rb = static_cast<std::size_t>(r) * stride;
            if (tr_) {
                go_a = hb_go_a(i0 + r); ge_a = hb_ge_a(i0 + r);
                go_b = hb_go_b(i0 + r, false); ge_b = hb_ge_b(i0 + r, false);
            }
            const T ux0 = pM[0] - go_b - ge_b, vx0 = pX[0] - ge_b, wx0 = pY[0] - go_b - ge_b;
            cM[0] = NINF;
            cX[0] = std::max({ux0, vx0, wx0});
            cY[0] = NINF;
            D[rb] = pack(3, pick(ux0, vx0, wx0), 3);
            if (in_x && r == 1) { cX[0] = static_cast<T>(0) - ge_b; D[rb] = pack(3, 1, 3); }   // continues the run
            for (int c = 1; c <= W; ++c) {
                const T dgM = pM[c-1], dgX = pX[c-1], dgY = pY[c-1];
                cM[c] = std::max({dgM, dgX, dgY}) + (tr_ ? subT(i0 + r, j0 + c) : subTu(i0 + r, j0 + c));
                const T ux = pM[c] - go_b - ge_b, vx = pX[c] - ge_b, wx = pY[c] - go_b - ge_b;
                cX[c] = std::max({ux, vx, wx});
                const T uy = cM[c-1] - go_a - ge_a, vy = cX[c-1] - go_a - ge_a, wy = cY[c-1] - ge_a;
                cY[c] = std::max({uy, vy, wy});
                D[rb + c] = pack(pick(dgM, dgX, dgY), pick(ux, vx, wx), pick(uy, vy, wy));
            }
            std::swap(pM, cM); std::swap(pX, cX); std::swap(pY, cY);
        }

        // Where does the block end?  out_x forces X; otherwise take the best of the three.
        int r = H, c = W;
        TBTable tbl;
        if (out_x) tbl = TBTable::X;
        else {
            const T vm = pM[W], vx = pX[W], vy = pY[W];
            tbl = (vm >= vx && vm >= vy) ? TBTable::M : ((vx >= vy) ? TBTable::X : TBTable::Y);
        }

        // Walk back, emitting moves; reverse at the end since we produce them backwards.
        const std::size_t first = hops_.size();
        while (r > 0 || c > 0) {
            const unsigned char p = D[static_cast<std::size_t>(r) * stride + static_cast<std::size_t>(c)];
            unsigned char code;
            if (tbl == TBTable::M)      { code = p & 3;        hops_.push_back(0); --r; --c; }
            else if (tbl == TBTable::X) { code = (p >> 2) & 3; hops_.push_back(1); --r; }
            else                        { code = p >> 4;       hops_.push_back(2); --c; }
            tbl = (code == 1) ? TBTable::X : (code == 2 ? TBTable::Y : TBTable::M);
        }
        std::reverse(hops_.begin() + static_cast<std::ptrdiff_t>(first), hops_.end());
    }

    static void hb_fit_b(BVec& v, std::size_t n) { if (v.size() < n) v.resize(n); }

    // One half-sweep, simd if this backend has a kernel and scalar otherwise, writing its
    // final row contiguously into (oM,oX,oY).  The two are bit-identical by construction
    // — same operations, same order, same left-association — so which one runs is a
    // speed decision and never a correctness one, exactly as with the Viterbi backends.
    void hb_half(DpBuffer& buf, bool reverse, int i0, int i1, int j0, int j1,
                 bool flag, T* oM, T* oX, T* oY) {
        const int H = i1 - i0, NC = j1 - j0;
        if (const LevelKernels* K = hb_level()) {
            if (reverse) hb_run_level(*K, buf, i1 - 1, -1, j1 - 1, -1, H, NC, flag, oM, oX, oY);
            else         hb_run_level(*K, buf, i0,     +1, j0,     +1, H, NC, flag, oM, oX, oY);
            return;
        }
        T *sM, *sX, *sY;
        if (tr_) hb_reverse_rows_();
        if (hb_pmax()) {
            // The ramp, contiguous: rmp[k] = k*ge_a.  Same scalar multiply of the same
            // exactly-representable integer the striped kernel uses for the same absolute
            // column k, so the two agree to the bit — and built here, outside the row
            // loop, so the carry loop below contains no multiply to contract into an FMA.
            // (Tracked: hb_fwd / hb_rev rebuild it per row, before the row's loop.)
            const T ge_a = static_cast<T>(params_->gap_extend_a);
            hb_fit(buf.hramp, static_cast<std::size_t>(NC) + 1);
            T* rmp = buf.hramp.data();
            for (int k = 0; k <= NC; ++k) rmp[k] = static_cast<T>(k) * ge_a;
            if (reverse)
                hb_rev<true>(i0, i1, j0, j1, flag,
                             buf.hfa.data(), buf.hfb.data(), buf.hfc.data(),
                             buf.hfd.data(), buf.hfe.data(), buf.hff.data(), sM, sX, sY, rmp);
            else
                hb_fwd<true>(i0, i1, j0, j1, flag,
                             buf.hfa.data(), buf.hfb.data(), buf.hfc.data(),
                             buf.hfd.data(), buf.hfe.data(), buf.hff.data(), sM, sX, sY, rmp);
        } else if (reverse)
            hb_rev(i0, i1, j0, j1, flag,
                   buf.hfa.data(), buf.hfb.data(), buf.hfc.data(),
                   buf.hfd.data(), buf.hfe.data(), buf.hff.data(), sM, sX, sY);
        else
            hb_fwd(i0, i1, j0, j1, flag,
                   buf.hfa.data(), buf.hfb.data(), buf.hfc.data(),
                   buf.hfd.data(), buf.hfe.data(), buf.hff.data(), sM, sX, sY);
        std::copy(sM, sM + NC + 1, oM);
        std::copy(sX, sX + NC + 1, oX);
        std::copy(sY, sY + NC + 1, oY);
    }

    // The recursion.  Splits the block's rows in half, sweeps to the split row from both
    // ends, and joins the two halves to find the column the optimal path crosses at.
    void hb_solve(DpBuffer& buf, int i0, int i1, int j0, int j1, bool in_x, bool out_x) {
        const int H = i1 - i0, W = j1 - j0;
        if (H <= hb_cutoff_) { hb_base(buf, i0, i1, j0, j1, in_x, out_x); return; }

        const int p = i0 + H / 2;
        const std::size_t stride = static_cast<std::size_t>(W) + 1;
        hb_fit(buf.hfa, stride); hb_fit(buf.hfb, stride); hb_fit(buf.hfc, stride);
        hb_fit(buf.hfd, stride); hb_fit(buf.hfe, stride); hb_fit(buf.hff, stride);
        hb_fit(buf.hra, stride); hb_fit(buf.hrb, stride); hb_fit(buf.hrc, stride);
        hb_fit(buf.hrd, stride); hb_fit(buf.hre, stride); hb_fit(buf.hrf, stride);
        hb_fit(buf.hsM, stride); hb_fit(buf.hsX, stride); hb_fit(buf.hsY, stride);

        // Both halves land in their own contiguous output rows — hs* for the forward,
        // hr{a,b,c} for the reverse — so the two never contend for scratch and the
        // simd and scalar paths deliver their results in the same place.
        hb_half(buf, false, i0, p, j0, j1, in_x,
                buf.hsM.data(), buf.hsX.data(), buf.hsY.data());
        const T* FM = buf.hsM.data(); const T* FX = buf.hsX.data(); const T* FY = buf.hsY.data();

        hb_half(buf, true, p, i1, j0, j1, out_x,
                buf.hra.data(), buf.hrb.data(), buf.hrc.data());
        const T* rM = buf.hra.data(); const T* rX = buf.hrb.data(); const T* rY = buf.hrc.data();

        // Join.  For each crossing column c the prefix ends in some state and the suffix
        // begins in some state; every combination is a legal path, so the plain sum over
        // the best of each is admissible.  The one case that is NOT a plain sum is an X
        // run straddling the cut: both halves then charge the open, so one -go_b is
        // refunded.  That refund is strictly positive, so taking the max over both forms
        // picks it automatically wherever it applies.
        //
        // Boundary states are STRICT, and must be for the join's values to be achievable:
        // in_x means the block's path LEAVES its origin by a vertical move continuing the
        // inherited run, out_x that it ENTERS its far corner by one.  The sweeps used to seed
        // the X state at the origin / corner instead, which also let the inherited state
        // feed a diagonal or a horizontal move there — so a half promised more than its
        // solver (whose base case ends strictly in X) could deliver, and the shortfall
        // compounded up the recursion: up to 4.8 below the optimum at float32, 3.5 at
        // double, on random pairs at small hb_cutoff (fixed 2026-10-09).  Now the origin /
        // corner is unreachable and the first swept row's column 0 is seeded 0 - ge_b.
        // Tracked: the open refunded is that of the first row below the cut, DP row p+1
        // (see hb_reverse_rows_).
        const T go_b = tr_ ? tgob_[static_cast<size_t>(p) + 1] : static_cast<T>(params_->gap_open_b);
        T best = NINF;
        int cstar = 0;
        bool span = false;
        for (int c = 0; c <= W; ++c) {
            const int d = W - c;                       // reverse sweep is indexed leftward
            const T f = std::max({FM[c], FX[c], FY[c]});
            const T g = std::max({rM[d], rX[d], rY[d]});
            const T plain = f + g;
            if (plain > best) { best = plain; cstar = c; span = false; }
            if (FX[c] > NINF && rX[d] > NINF) {
                const T joined = FX[c] + rX[d] + go_b;
                if (joined > best) { best = joined; cstar = c; span = true; }
            }
        }

        const int jstar = j0 + cstar;
        hb_solve(buf, i0, p, j0, jstar, in_x, span);
        hb_solve(buf, p, i1, jstar, j1, span, out_x);
    }

    // Entry point: recover the whole path, then read the score back off it.  The score is
    // recomputed from the path rather than taken from the join, which is a genuine check
    // and not bookkeeping — if the affine boundary handling were wrong the two would
    // disagree, and the tests compare this score against the other tracebacks'.
    void viterbi_affine_hirschberg(DpBuffer& buf) {
        hops_.clear();
        hops_.reserve(static_cast<std::size_t>(m_ + n_));
        hb_solve(buf, 0, m_, 0, n_, false, false);

        // Replay for the score.  Also fixes best_i_/best_j_/best_tbl_, which the shared
        // accessors still expect even though no table backs them here.
        //
        // Accumulate in T, not double.  The DP's own score is the path's prefix sum with
        // a rounding at every step; replaying in double instead rounds nowhere and lands
        // 1e-3 away under T=float — small, but enough to make "does Hirschberg agree with
        // pointers" untestable by exact comparison, which is the one question worth asking
        // of it.  Same precision, same order, same operations: identical paths now give
        // identical scores to the bit, so any difference that remains is a real one.
        T s = static_cast<T>(0);
        int i = 0, j = 0;
        unsigned char prev = 3;                     // 3 = no previous move
        for (unsigned char op : hops_) {
            if (op == 0) { ++i; ++j; s += subT(i, j); }
            else if (op == 1) {
                ++i;   // a gap in B is charged by the row it enters
                if (prev != 1) s -= row_go_b(i);
                s -= row_ge_b(i);
            } else {
                if (prev != 2) s -= row_go_a(i);
                s -= row_ge_a(i); ++j;
            }
            prev = op;
        }
        viterbi_score_ = s;
        best_i_ = m_; best_j_ = n_;
        best_tbl_ = hops_.empty() ? TBTable::M
                  : (hops_.back() == 1 ? TBTable::X
                     : (hops_.back() == 2 ? TBTable::Y : TBTable::M));
        hirschberg_ = true;
    }

    // ── Local (Smith-Waterman) linear space: endpoint scans + a global box ─────
    //
    // The leveled endpoint-scan kernel for this backend, or null to go scalar — the
    // twin of hb_level()/hb_base_level().
    const LevelKernels* hb_scan_level() const noexcept {
        const int backend = (backend_ == kBackendAuto) ? global_default_backend() : backend_;
        if (backend == kBackendScalar) return nullptr;
        const LevelKernels& K = level_kernels(backend);
        return K.template of<T>().hb_scan ? &K : nullptr;
    }

    // Run one endpoint scan over the sub-rectangle rows [i0, i1), cols [j0, j1): simd if
    // this backend has the kernel, scalar otherwise.  `local` picks clamped-SW (forward
    // end cell) vs unclamped global-suffix (reverse start cell); `reverse` negates the
    // walk.  Reports the best cell in BLOCK-1-based coords (row 1..H, col 1..NC) and its
    // score.  The two paths are bit-identical by construction — same recurrence, same
    // topmost/leftmost tie-break — so which runs is a speed choice, exactly as elsewhere.
    void hb_scan(DpBuffer& buf, bool local, bool reverse, int i0, int i1, int j0, int j1,
                 T& best, int& bi, int& bj) {
        const int H = i1 - i0, NC = j1 - j0;
        if (const LevelKernels* K = hb_scan_level()) {
            HbScanJob<T> job{};
            job.a = a_idx_.data(); job.b = b_idx_.data();
            job.blk = blkT_; job.nalpha = nalpha_;
            job.go_a = static_cast<T>(params_->gap_open_a);
            job.ge_a = static_cast<T>(params_->gap_extend_a);
            job.go_b = static_cast<T>(params_->gap_open_b);
            job.ge_b = static_cast<T>(params_->gap_extend_b);
            if (reverse) { job.a_start = i1 - 1; job.a_step = -1; job.b_start = j1 - 1; job.b_step = -1; }
            else         { job.a_start = i0;     job.a_step = +1; job.b_start = j0;     job.b_step = +1; }
            job.H = H; job.ncols = NC; job.local = local ? 1 : 0;
            job.pmax = hb_pmax() ? 1 : 0;
            job.buf = &buf;
            if (tr_) { hb_reverse_rows_(); fill_track_job_(job.tr, false, 0); }
            K->template of<T>().hb_scan(job);
            best = job.best; bi = job.best_i; bj = job.best_j;
            return;
        }
        hb_scan_scalar(buf, local, reverse, i0, i1, j0, j1, best, bi, bj);
    }

    // Scalar reference endpoint scan.  Rolling rows only; the borders and the M-clamp
    // switch on `local`, the walk direction on `reverse`.  The argmax is topmost-row /
    // leftmost-column, strict > — the SAME tie-break the simd kernel uses, so the two
    // agree to the bit (the property NWGRAD_ISA relies on).
    void hb_scan_scalar(DpBuffer& buf, bool local, bool reverse, int i0, int i1, int j0, int j1,
                        T& best, int& bi, int& bj) {
        const int H = i1 - i0, NC = j1 - j0;
        best = static_cast<T>(0); bi = 0; bj = 0;
        if (H == 0 || NC == 0) return;
        if (tr_) hb_reverse_rows_();
        // Tracked: the rows as the leveled scan reads them (HbRows): forward, row t's
        // cells are DP row a[t]+1; reverse (the mirror), DP row a[t], with the rewritten
        // vertical moves; the block's row 0 is DP row i0 forward, i1 reverse.
        T go_a = hb_go_a(reverse ? i1 : i0), ge_a = hb_ge_a(reverse ? i1 : i0);
        T go_b = static_cast<T>(params_->gap_open_b);
        T ge_b = static_cast<T>(params_->gap_extend_b);
        const int a_start = reverse ? i1 - 1 : i0, a_step = reverse ? -1 : +1;
        const int b_start = reverse ? j1 - 1 : j0, b_step = reverse ? -1 : +1;

        const std::size_t stride = static_cast<std::size_t>(NC) + 1;
        hb_fit(buf.hfa, stride); hb_fit(buf.hfb, stride); hb_fit(buf.hfc, stride);
        hb_fit(buf.hfd, stride); hb_fit(buf.hfe, stride); hb_fit(buf.hff, stride);
        T* pM = buf.hfa.data(); T* pX = buf.hfb.data(); T* pY = buf.hfc.data();
        T* cM = buf.hfd.data(); T* cX = buf.hfe.data(); T* cY = buf.hff.data();
        // Pmax: the closed-form carry, as hb_scan_impl<..., true> (the ramp loaded, k*ge_a).
        const bool pmax = hb_pmax();
        if (pmax) hb_fit(buf.hramp, stride);
        T* rmp = buf.hramp.data();
        if (pmax) for (int k = 0; k < NC; ++k) rmp[k] = static_cast<T>(k) * ge_a;

        // Row 0.  Local: M = 0 for every column (fresh start anywhere).  Global: the
        // Y-gap-open series, so the reverse pass computes the suffixes' global alignment.
        pM[0] = static_cast<T>(0); pX[0] = NINF; pY[0] = NINF;
        for (int c = 1; c <= NC; ++c) {
            if (local) { pM[c] = static_cast<T>(0); pX[c] = NINF; pY[c] = NINF; }
            else {
                pM[c] = NINF; pX[c] = NINF;
                pY[c] = std::max((std::max(pM[c-1], pX[c-1]) - go_a) - ge_a, pY[c-1] - ge_a);
            }
        }

        T gbest = static_cast<T>(0); int gi = 0, gj = 0;
        for (int t = 0; t < H; ++t) {
            const int arow = a_start + t * a_step;                 // 0-based sequence row
            if (tr_) {
                const int hr = reverse ? arow : arow + 1;
                go_a = hb_go_a(hr); ge_a = hb_ge_a(hr);
                go_b = hb_go_b(arow + 1, reverse); ge_b = hb_ge_b(arow + 1, reverse);
                if (pmax) for (int k = 0; k < NC; ++k) rmp[k] = static_cast<T>(k) * ge_a;
            }
            cM[0] = local ? static_cast<T>(0) : NINF;
            cX[0] = std::max(std::max((pM[0] - go_b) - ge_b, pX[0] - ge_b), (pY[0] - go_b) - ge_b);
            cY[0] = NINF;
            T P = cY[0] - ge_a;   // Pmax: the running prefix max, the border as its seed
            for (int c = 1; c <= NC; ++c) {
                const int bcol = b_start + (c - 1) * b_step;       // 0-based sequence col
                T m = std::max({pM[c-1], pX[c-1], pY[c-1]}) + (tr_ ? subT(arow + 1, bcol + 1) : subTu(arow + 1, bcol + 1));
                if (local) m = std::max(m, static_cast<T>(0));
                const T x = std::max(std::max((pM[c] - go_b) - ge_b, pX[c] - ge_b), (pY[c] - go_b) - ge_b);
                T y;
                if (pmax) {
                    const T g = (std::max(cM[c-1], cX[c-1]) - go_a) - ge_a;
                    P = std::max<T>(g + rmp[c-1], P);
                    y = P - rmp[c-1];
                } else {
                    y = std::max(std::max((cM[c-1] - go_a) - ge_a, (cX[c-1] - go_a) - ge_a), cY[c-1] - ge_a);
                }
                cM[c] = m; cX[c] = x; cY[c] = y;
                const T here = std::max({m, x, y});
                if (here > gbest) { gbest = here; gi = t + 1; gj = c; }
            }
            std::swap(pM, cM); std::swap(pX, cX); std::swap(pY, cY);
        }
        best = gbest; bi = gi; bj = gj;
    }

    // Entry: linear-space Smith-Waterman.  Forward clamped scan -> end cell (ie, je) and
    // score S; reverse global-suffix scan over the prefix rectangle -> start (is, js);
    // then hb_solve GLOBALLY aligns the box A[is..ie) x B[js..je), which scores exactly S.
    // Everything is O(n) memory.  Below hb_cutoff the box never splits and the fill is the
    // exact Pointers one, so short local pairs are bit-exact; longer ones return a valid
    // (equal-score) but possibly different optimal path at ties, like Global Hirschberg.
    void viterbi_affine_hirschberg_local(DpBuffer& buf) {
        hops_.clear();
        hb_start_i_ = 0; hb_start_j_ = 0;

        T s_fwd; int ie, je;
        hb_scan(buf, /*local=*/true, /*reverse=*/false, 0, m_, 0, n_, s_fwd, ie, je);

        // No positive-scoring local alignment: the empty alignment (score 0) is optimal.
        if (s_fwd <= static_cast<T>(0) || ie <= 0 || je <= 0) {
            viterbi_score_ = T(0);
            best_i_ = 0; best_j_ = 0; best_tbl_ = TBTable::M;
            hirschberg_ = true;
            return;
        }

        T s_rev; int p, q;
        hb_scan(buf, /*local=*/false, /*reverse=*/true, 0, ie, 0, je, s_rev, p, q);
        const int is = ie - p, js = je - q;

        hb_solve(buf, is, ie, js, je, false, false);
        hb_start_i_ = is; hb_start_j_ = js;

        // Replay for the score, seeded at the local start (subT is 1-based).  Same order
        // and precision as the DP, so the path's score is exact — the value the tests pin
        // against SW Pointers.
        T s = static_cast<T>(0);
        int i = is, j = js;
        unsigned char prev = 3;
        for (unsigned char op : hops_) {
            if (op == 0) { ++i; ++j; s += subT(i, j); }
            else if (op == 1) { ++i; if (prev != 1) s -= row_go_b(i); s -= row_ge_b(i); }
            else              { if (prev != 2) s -= row_go_a(i); s -= row_ge_a(i); ++j; }
            prev = op;
        }
        viterbi_score_ = s;
        best_i_ = ie; best_j_ = je;
        best_tbl_ = hops_.empty() ? TBTable::M
                  : (hops_.back() == 1 ? TBTable::X
                     : (hops_.back() == 2 ? TBTable::Y : TBTable::M));
        hirschberg_ = true;
    }

    // ── Linear gaps in linear space ─────────────────────────────────────────────
    //
    // Hirschberg for linear gaps: no gap state crosses a row cut (a gap move costs ge
    // whatever precedes it), so the join is a plain argmax of forward + reverse scores —
    // none of the affine path's boundary flags or open refunds.  Scalar sweeps (linear
    // has no simd per-pair kernel), O(n) rows; the base case is the Pointers fill on the
    // sub-rectangle, walked back into hops_.  Local: the affine path's endpoint reduction
    // — a clamped forward scan for the end (ie, je) and its score, an unclamped global-
    // suffix scan for the start, then the global solve of the box.  Like affine
    // Hirschberg it may pick a different optimal path at ties above hb_cutoff, never a
    // worse one (the score is replayed from the path); a pair <= hb_cutoff runs AS
    // Pointers (run_viterbi), bit-exact.

    // Forward sweep from (i0, j0), Global borders, to row i1: out[c] = best score of
    // A[i0..i1) x B[j0..j0+c).  Reverse (rev): from (i1, j1) back to row i0, out[d] =
    // best score of A[i0..i1) x B[j1-d..j1).
    // The leveled linear Hirschberg kernel (kernels_linear_impl.inl, lhb_striped) for a
    // loop of `ncols` columns, or null: the scalar backend, a level without it (int64
    // below 8 lanes), a header-only build, or a row too short to pay (the linear fill's
    // measured gate, kLinearStripedMinN).  Every caller keeps its scalar loop as the
    // fallback, and the two are bit-identical, so the choice moves no path.
    using lhb_fn_t = void (*)(LhbJob<T>&);
    lhb_fn_t lhb_kernel_(int ncols) const {
        const int backend = (backend_ == kBackendAuto) ? global_default_backend() : backend_;
        if (backend == kBackendScalar) return nullptr;
        if (ncols < linear_striped_min_n(backend)) return nullptr;
        return level_kernels(backend).template of<T>().lhb;
    }
    uint32_t lhb_srow_(int i) const noexcept {   // DP row i's stacked substitution row
        return static_cast<uint32_t>((sub_row_(i) - blkT_) / nalpha_);
    }
    std::vector<unsigned char> lhb_b_;
    std::vector<uint32_t> lhb_rows_;
    std::vector<T> lhb_ga_, lhb_gb_, lhb_r0_, lhb_c0_;
    int lhb_runs_ = 0;   // kernel calls, for the tests' liveness check
    LhbJob<T> lhb_job_(DpBuffer& buf, int ncols, int nrows) {
        LhbJob<T> job{};
        job.b = lhb_b_.data(); job.n = ncols;
        job.blk = blkT_; job.nalpha = nalpha_;
        job.srow = lhb_rows_.data(); job.h = nrows;
        job.ga = lhb_ga_.data(); job.gb = lhb_gb_.data();
        job.row0 = lhb_r0_.data(); job.col0 = lhb_c0_.data();
        job.by_symbol = (!tr_ && nrows >= nalpha_) ? 1 : 0;
        job.buf = &buf;
        return job;
    }
public:
    int lhb_kernel_runs() const noexcept { return lhb_runs_; }
private:

    void lhb_sweep(DpBuffer& buf, bool rev, int i0, int i1, int j0, int j1, T* out) {
        const int W = j1 - j0;
        if (const lhb_fn_t fn = lhb_kernel_(W); fn && i1 > i0 && W > 0) {
            // the scalar sweep's arrays, in its order and with its expressions
            const int Hh = i1 - i0;
            lhb_b_.resize(static_cast<size_t>(W));
            for (int c = 1; c <= W; ++c)
                lhb_b_[static_cast<size_t>(c) - 1] = rev ? b_idx_[static_cast<size_t>(j1 - c)]
                                                         : b_idx_[static_cast<size_t>(j0 + c - 1)];
            lhb_rows_.resize(static_cast<size_t>(Hh));
            lhb_ga_.resize(static_cast<size_t>(Hh)); lhb_gb_.resize(static_cast<size_t>(Hh));
            T ga = hb_ge_a(rev ? i1 : i0), gb = static_cast<T>(params_->gap_extend_b);
            lhb_r0_.resize(static_cast<size_t>(W) + 1); lhb_c0_.resize(static_cast<size_t>(Hh) + 1);
            lhb_r0_[0] = static_cast<T>(0);
            for (int c = 1; c <= W; ++c) lhb_r0_[static_cast<size_t>(c)] = lhb_r0_[static_cast<size_t>(c) - 1] - ga;
            lhb_c0_[0] = lhb_r0_[0];
            for (int r = 1; r <= Hh; ++r) {
                const int i = rev ? i1 - r + 1 : i0 + r;
                if (tr_) { gb = tgeb_[static_cast<size_t>(i)]; ga = hb_ge_a(rev ? i - 1 : i); }
                lhb_rows_[static_cast<size_t>(r) - 1] = lhb_srow_(i);
                lhb_ga_[static_cast<size_t>(r) - 1] = ga; lhb_gb_[static_cast<size_t>(r) - 1] = gb;
                lhb_c0_[static_cast<size_t>(r)] = lhb_c0_[static_cast<size_t>(r) - 1] - gb;
            }
            LhbJob<T> job = lhb_job_(buf, W, Hh);
            job.out = out;
            fn(job);
            ++lhb_runs_;
            return;
        }
        hb_fit(buf.hfa, static_cast<std::size_t>(W) + 1);
        T* row = buf.hfa.data();
        // Tracked: no opens in linear, so the mirror needs no rewritten rows — only the
        // row of each gap in A: forward DP row i, reverse (the suffix from row i-1) row i-1.
        T ge_a = hb_ge_a(rev ? i1 : i0);
        T ge_b = static_cast<T>(params_->gap_extend_b);
        row[0] = static_cast<T>(0);
        for (int c = 1; c <= W; ++c) row[c] = row[c - 1] - ge_a;
        // The direction and the substitution row are resolved outside the column loop
        // (gcc unswitched them, clang did not: clang/gcc 1.65x); the same arithmetic.
        auto sweep = [&](auto step) {
            constexpr int S = decltype(step)::value;
            for (int r = 1; r <= i1 - i0; ++r) {
                const int i = S < 0 ? i1 - r + 1 : i0 + r;   // 1-based DP row of A[i-1]
                if (tr_) { ge_b = tgeb_[static_cast<size_t>(i)]; ge_a = hb_ge_a(S < 0 ? i - 1 : i); }
                const T* srow = sub_row_(i);
                // B code of column c: forward b[j0 + c - 1], reverse b[j1 - c].
                const uint8_t* bc = b_idx_.data();
                const int jb = S < 0 ? j1 : j0 - 1;
                const T gb = ge_b, ga = ge_a;
                T diag = row[0];
                row[0] = row[0] - gb;
                // The left neighbour rides in a register: clang re-loaded row[c-1] just
                // stored, a store-forward on the serial chain.
                T left = row[0];
                for (int c = 1; c <= W; ++c) {
                    T v = diag + srow[bc[jb + S * c]];
                    const T u = row[c] - gb, l = left - ga;
                    if (v < u) v = u;
                    if (v < l) v = l;
                    diag = row[c];
                    row[c] = v;
                    left = v;
                }
            }
        };
        if (rev) sweep(std::integral_constant<int, -1>{});
        else     sweep(std::integral_constant<int, 1>{});
        std::copy(row, row + W + 1, out);
    }

    // Base case: Pointers on the sub-rectangle, Global borders, walked back from
    // (i1, j1) to (i0, j0); the moves appended to hops_ in forward order.
    void lhb_base(DpBuffer& buf, int i0, int i1, int j0, int j1) {
        const int H = i1 - i0, W = j1 - j0;
        if (const lhb_fn_t fn = lhb_kernel_(W); fn && H > 0 && W > 0) {
            lhb_b_.resize(static_cast<size_t>(W));
            for (int c = 1; c <= W; ++c) lhb_b_[static_cast<size_t>(c) - 1] = b_idx_[static_cast<size_t>(j0 + c - 1)];
            lhb_rows_.resize(static_cast<size_t>(H));
            lhb_ga_.resize(static_cast<size_t>(H)); lhb_gb_.resize(static_cast<size_t>(H));
            T ga = hb_ge_a(i0), gb = static_cast<T>(params_->gap_extend_b);
            lhb_r0_.resize(static_cast<size_t>(W) + 1); lhb_c0_.resize(static_cast<size_t>(H) + 1);
            lhb_r0_[0] = static_cast<T>(0);
            for (int c = 1; c <= W; ++c) lhb_r0_[static_cast<size_t>(c)] = lhb_r0_[static_cast<size_t>(c) - 1] - ga;
            lhb_c0_[0] = lhb_r0_[0];
            for (int r = 1; r <= H; ++r) {
                if (tr_) { ga = tgea_[static_cast<size_t>(i0 + r)]; gb = tgeb_[static_cast<size_t>(i0 + r)]; }
                lhb_rows_[static_cast<size_t>(r) - 1] = lhb_srow_(i0 + r);
                lhb_ga_[static_cast<size_t>(r) - 1] = ga; lhb_gb_[static_cast<size_t>(r) - 1] = gb;
                lhb_c0_[static_cast<size_t>(r)] = lhb_c0_[static_cast<size_t>(r) - 1] - gb;
            }
            // rowsz = (seg + 1) * lanes <= W + 2 * 64 for every level's lane count
            const std::size_t cap = (static_cast<std::size_t>(H) + 1) * (static_cast<std::size_t>(W) + 130);
            if (buf.hbD.size() < cap) buf.hbD.resize(cap);
            LhbJob<T> job = lhb_job_(buf, W, H);
            job.codes = buf.hbD.data();
            fn(job);
            ++lhb_runs_;
            const std::size_t lw = static_cast<std::size_t>(job.width), sg = static_cast<std::size_t>(job.seg);
            const std::size_t rs = (sg + 1) * lw;
            auto at = [&](int r, int c) -> std::size_t {
                if (c == 0) return static_cast<std::size_t>(r) * rs;
                const std::size_t cc = static_cast<std::size_t>(c) - 1;
                return static_cast<std::size_t>(r) * rs + lw + (cc % sg) * lw + cc / sg;
            };
            const unsigned char* D = buf.hbD.data();
            const std::size_t at0 = hops_.size();
            int r = H, c = W;
            while (r > 0 || c > 0) {
                const unsigned char k = D[at(r, c)];
                hops_.push_back(k);
                if (k == 0) { --r; --c; } else if (k == 1) --r; else --c;
            }
            std::reverse(hops_.begin() + static_cast<std::ptrdiff_t>(at0), hops_.end());
            return;
        }
        const std::size_t st = static_cast<std::size_t>(W) + 1;
        if (buf.DM.size() < (static_cast<std::size_t>(H) + 1) * st)
            buf.DM.resize((static_cast<std::size_t>(H) + 1) * st);
        hb_fit(buf.hfa, st); hb_fit(buf.hfb, st);
        T* prev = buf.hfa.data(); T* cur = buf.hfb.data();
        unsigned char* D = buf.DM.data();
        T ge_a = hb_ge_a(i0);
        T ge_b = static_cast<T>(params_->gap_extend_b);
        prev[0] = static_cast<T>(0);
        for (int c = 1; c <= W; ++c) { prev[c] = prev[c - 1] - ge_a; D[c] = 2; }
        for (int r = 1; r <= H; ++r) {
            unsigned char* d = D + static_cast<std::size_t>(r) * st;
            if (tr_) { ge_a = tgea_[static_cast<size_t>(i0 + r)]; ge_b = tgeb_[static_cast<size_t>(i0 + r)]; }
            cur[0] = prev[0] - ge_b; d[0] = 1;
            const T* srow = sub_row_(i0 + r);
            const uint8_t* bc = b_idx_.data();
            const T gb = ge_b, ga = ge_a;
            T left = cur[0];   // in a register, as in lhb_sweep
            for (int c = 1; c <= W; ++c) {
                T v = prev[c - 1] + srow[bc[j0 + c - 1]];
                unsigned char k = 0;
                const T u = prev[c] - gb, l = left - ga;
                if (v < u) { v = u; k = 1; }
                if (v < l) { v = l; k = 2; }
                cur[c] = v; d[c] = k; left = v;
            }
            std::swap(prev, cur);
        }
        const std::size_t at = hops_.size();
        int r = H, c = W;
        while (r > 0 || c > 0) {
            const unsigned char k = D[static_cast<std::size_t>(r) * st + c];
            hops_.push_back(k);
            if (k == 0) { --r; --c; } else if (k == 1) --r; else --c;
        }
        std::reverse(hops_.begin() + static_cast<std::ptrdiff_t>(at), hops_.end());
    }

    void lhb_solve(DpBuffer& buf, int i0, int i1, int j0, int j1) {
        const int H = i1 - i0, W = j1 - j0;
        if (H <= hb_cutoff_) { lhb_base(buf, i0, i1, j0, j1); return; }
        const int p = i0 + H / 2;
        hb_fit(buf.hsM, static_cast<std::size_t>(W) + 1);
        hb_fit(buf.hra, static_cast<std::size_t>(W) + 1);
        lhb_sweep(buf, false, i0, p, j0, j1, buf.hsM.data());
        lhb_sweep(buf, true, p, i1, j0, j1, buf.hra.data());
        const T* F = buf.hsM.data(); const T* R = buf.hra.data();
        T best = NINF;
        int cstar = 0;
        for (int c = 0; c <= W; ++c) {
            const T v = F[c] + R[W - c];
            if (v > best) { best = v; cstar = c; }
        }
        lhb_solve(buf, i0, p, j0, j0 + cstar);
        lhb_solve(buf, p, i1, j0 + cstar, j1);
    }

    void viterbi_linear_hirschberg(DpBuffer& buf) {
        hops_.clear();
        hb_start_i_ = 0; hb_start_j_ = 0;
        int i_end = m_, j_end = n_;
        // Tracked: row i's own extends (ga(i): gaps in A within DP row i; gb(i): into it).
        const T ge_a0 = static_cast<T>(params_->gap_extend_a);
        const T ge_b0 = static_cast<T>(params_->gap_extend_b);
        auto ga = [&](int i) { return tr_ ? tgea_[static_cast<size_t>(i)] : ge_a0; };
        auto gb = [&](int i) { return tr_ ? tgeb_[static_cast<size_t>(i)] : ge_b0; };
        if constexpr (AM == AlignMode::Local) {
            // Forward clamped scan: the best cell, first row then first column (strict >,
            // as viterbi_linear chooses it).
            hb_fit(buf.hfa, static_cast<std::size_t>(n_) + 1);
            T* row = buf.hfa.data();
            std::fill(row, row + n_ + 1, static_cast<T>(0));
            T best = static_cast<T>(0); int ie = 0, je = 0;
            const lhb_fn_t fscan = lhb_kernel_(n_);
            if (fscan && m_ > 0 && n_ > 0) {
                lhb_b_.assign(b_idx_.begin(), b_idx_.begin() + n_);
                lhb_rows_.resize(static_cast<size_t>(m_));
                lhb_ga_.resize(static_cast<size_t>(m_)); lhb_gb_.resize(static_cast<size_t>(m_));
                for (int i = 1; i <= m_; ++i) {
                    lhb_rows_[static_cast<size_t>(i) - 1] = lhb_srow_(i);
                    lhb_ga_[static_cast<size_t>(i) - 1] = ga(i); lhb_gb_[static_cast<size_t>(i) - 1] = gb(i);
                }
                lhb_r0_.assign(static_cast<size_t>(n_) + 1, static_cast<T>(0));
                lhb_c0_.assign(static_cast<size_t>(m_) + 1, static_cast<T>(0));
                LhbJob<T> job = lhb_job_(buf, n_, m_);
                job.clamp = 1; job.argmax = 1;
                job.best = static_cast<T>(0); job.bi = 0; job.bj = 0;
                fscan(job);
                ++lhb_runs_;
                best = job.best; ie = job.bi; je = job.bj;
            }
            for (int i = 1; i <= (fscan && m_ > 0 && n_ > 0 ? 0 : m_); ++i) {
                T diag = row[0];
                const T ge_a = ga(i), ge_b = gb(i);
                for (int j = 1; j <= n_; ++j) {
                    T v = diag + (tr_ ? subT(i, j) : subTu(i, j));
                    const T u = row[j] - ge_b, l = row[j - 1] - ge_a;
                    if (v < u) v = u;
                    if (v < l) v = l;
                    v = std::max(v, static_cast<T>(0));
                    diag = row[j];
                    row[j] = v;
                    if (v > best) { best = v; ie = i; je = j; }
                }
            }
            if (!(best > static_cast<T>(0))) {
                viterbi_score_ = T(0); best_i_ = 0; best_j_ = 0; hirschberg_ = true; return;
            }
            // Unclamped global-suffix scan over [0, ie) x [0, je): the start maximizing
            // the box's global score (any argmax scores the local optimum).
            hb_fit(buf.hfb, static_cast<std::size_t>(je) + 1);
            T* g = buf.hfb.data();                       // g[q] = score from (p, q) to (ie, je)
            g[je] = static_cast<T>(0);
            for (int q = je - 1; q >= 0; --q) g[q] = g[q + 1] - ga(ie);
            T sbest = g[0]; int is = ie, js = 0;
            for (int q = 0; q <= je; ++q) if (g[q] > sbest) { sbest = g[q]; js = q; }
            // Leveled: the suffix scan is a reverse sweep in mirrored coordinates (row r =
            // DP row pi = ie - r, column c = je - q), its row 0 the g just computed; the
            // scalar argmax order (rows pi descending; in each, q = je first, then q
            // descending; strict >) is the kernel's (rows ascending, column 0 first, then
            // columns ascending), seeded with row 0's best.
            const lhb_fn_t rscan = lhb_kernel_(je);
            const bool rk = rscan && ie > 0 && je > 0;
            if (rk) {
                lhb_b_.resize(static_cast<size_t>(je));
                for (int c = 1; c <= je; ++c) lhb_b_[static_cast<size_t>(c) - 1] = b_idx_[static_cast<size_t>(je - c)];
                lhb_rows_.resize(static_cast<size_t>(ie));
                lhb_ga_.resize(static_cast<size_t>(ie)); lhb_gb_.resize(static_cast<size_t>(ie));
                lhb_r0_.resize(static_cast<size_t>(je) + 1); lhb_c0_.resize(static_cast<size_t>(ie) + 1);
                for (int c = 0; c <= je; ++c) lhb_r0_[static_cast<size_t>(c)] = g[je - c];
                lhb_c0_[0] = g[je];
                for (int r = 1; r <= ie; ++r) {
                    const int pi = ie - r;
                    lhb_rows_[static_cast<size_t>(r) - 1] = lhb_srow_(pi + 1);
                    lhb_ga_[static_cast<size_t>(r) - 1] = ga(pi); lhb_gb_[static_cast<size_t>(r) - 1] = gb(pi + 1);
                    lhb_c0_[static_cast<size_t>(r)] = lhb_c0_[static_cast<size_t>(r) - 1] - gb(pi + 1);
                }
                LhbJob<T> job = lhb_job_(buf, je, ie);
                job.argmax = 2;
                job.best = sbest; job.bi = 0; job.bj = je - js;
                rscan(job);
                ++lhb_runs_;
                sbest = job.best; is = ie - job.bi; js = je - job.bj;
            }
            for (int pi = (rk ? -1 : ie - 1); pi >= 0; --pi) {
                // the suffix from DP row pi: its first move down is into row pi+1, its
                // gaps in A lie in row pi
                const T ge_a = ga(pi), ge_b = gb(pi + 1);
                T diag = g[je];
                g[je] = g[je] - ge_b;
                if (g[je] > sbest) { sbest = g[je]; is = pi; js = je; }
                for (int q = je - 1; q >= 0; --q) {
                    T v = diag + (tr_ ? subT(pi + 1, q + 1) : subTu(pi + 1, q + 1));
                    const T u = g[q] - ge_b, l = g[q + 1] - ge_a;
                    if (v < u) v = u;
                    if (v < l) v = l;
                    diag = g[q];
                    g[q] = v;
                    if (v > sbest) { sbest = v; is = pi; js = q; }
                }
            }
            hb_start_i_ = is; hb_start_j_ = js;
            i_end = ie; j_end = je;
            lhb_solve(buf, is, ie, js, je);
        } else {
            lhb_solve(buf, 0, m_, 0, n_);
        }
        // Replay for the score, in T and in path order (the tests compare it exactly).
        T sc = static_cast<T>(0);
        int i = hb_start_i_, j = hb_start_j_;
        for (unsigned char op : hops_) {
            if (op == 0)      { ++i; ++j; sc += subT(i, j); }
            else if (op == 1) { ++i; sc -= gb(i); }
            else              { sc -= ga(i); ++j; }
        }
        viterbi_score_ = sc;
        best_i_ = i_end; best_j_ = j_end;
        hirschberg_ = true;
    }

    void hard_grad_linear_hb(AlignParams& grad) const {
        double* gblk = grad_block(grad);
        int i = hb_start_i_, j = hb_start_j_;
        for (unsigned char op : hops_) {
            if (op == 0)      { ++i; ++j; gblk[sub_off(i, j)] += 1.0; }
            else if (op == 1) { ++i; grad_gap(grad, i, AlignParams::kExtendB) -= 1.0; }
            else              { ++j; grad_gap(grad, i, AlignParams::kExtendA) -= 1.0; }
        }
    }

    // ── Consumers of a Hirschberg path ────────────────────────────────────────
    // Each mirrors its table-walking twin, but replays hops_ forward instead of
    // walking predecessors backward.  Same emissions, same gradient convention.

    void hard_grad_affine_hb(AlignParams& grad) const {
        double* gblk = grad_block(grad);
        int i = hb_start_i_, j = hb_start_j_;   // local: the path starts at (is, js), not the origin
        unsigned char prev = 3;
        for (unsigned char op : hops_) {
            if (op == 0) { ++i; ++j; gblk[sub_off(i, j)] += 1.0; }
            else if (op == 1) {
                ++i;   // charged by the row it enters
                if (prev != 1) grad_gap(grad, i, AlignParams::kOpenB) -= 1.0;
                grad_gap(grad, i, AlignParams::kExtendB) -= 1.0;
            } else {
                if (prev != 2) grad_gap(grad, i, AlignParams::kOpenA) -= 1.0;
                grad_gap(grad, i, AlignParams::kExtendA) -= 1.0; ++j;
            }
            prev = op;
        }
    }

    std::vector<std::pair<int,int>> alignment_hb() const {
        // Emit the 0-based sequence position of each matched pair BEFORE advancing —
        // the same convention traceback_affine uses (it emits i-1,j-1 from 1-based DP
        // coords), so alignment() agrees across tracebacks and with pairs_from_gapped().
        std::vector<std::pair<int,int>> path;
        int i = hb_start_i_, j = hb_start_j_;
        for (unsigned char op : hops_) {
            if (op == 0)      { path.emplace_back(i, j); ++i; ++j; }
            else if (op == 1) { ++i; }
            else              { ++j; }
        }
        return path;
    }

    void aligned_hb(std::string& a, std::string& b) const {
        a.clear(); b.clear();
        int i = hb_start_i_, j = hb_start_j_;
        for (unsigned char op : hops_) {
            if (op == 0)      { ++i; ++j; a.push_back(sym_a(i)); b.push_back(sym_b(j)); }
            else if (op == 1) { ++i; a.push_back(sym_a(i)); b.push_back('-'); }
            else              { ++j; a.push_back('-');      b.push_back(sym_b(j)); }
        }
    }

    std::vector<int> guide_j_affine_hb() const {
        if (grid_) return guide_j_grid_();
        std::vector<int> gj(static_cast<std::size_t>(m_ + 1), -1);
        gj[0] = 0;
        // Local: the path covers only rows [is, ie); seed the start anchor and let the
        // rows outside it interpolate (fill_guide_gaps), exactly as guide_j_affine does
        // from its local traceback.  For Global (is=js=0) this is the original walk.
        int i = hb_start_i_, j = hb_start_j_;
        gj[static_cast<std::size_t>(i)] = j;
        for (unsigned char op : hops_) {
            if (op == 0)      { ++i; ++j; gj[static_cast<std::size_t>(i)] = j; }
            else if (op == 1) { ++i;      gj[static_cast<std::size_t>(i)] = j; }
            else              { ++j; }
        }
        gj[static_cast<std::size_t>(m_)] = n_;
        fill_guide_gaps(gj);
        return gj;
    }

    // The guide of a Grid path, in Pointers' convention (guide_j_affine): gj[i] is the
    // column where the path LEAVES row i (an M or X move out of it), not where it enters.
    std::vector<int> guide_j_grid_() const {
        std::vector<int> gj;
        guide_affine_init(gj);
        int i = 0, j = 0;
        for (unsigned char op : hops_) {
            if (op == 0)      { gj[static_cast<std::size_t>(i)] = j; ++i; ++j; }
            else if (op == 1) { gj[static_cast<std::size_t>(i)] = j; ++i; }
            else              { ++j; }
        }
        fill_guide_gaps(gj);
        return gj;
    }

    // ── TracebackMode::Grid: grid-checkpoint traceback (see the TracebackMode note) ──
    //
    // grid_solve(rect, entry cell) walks the optimal path back from the entry cell (i, j)
    // in state st until it reaches the rectangle's top row or left column.  Nothing below
    // or right of the entry cell can be on the path, so the rectangle is [r0, i] x [c0, j].
    // Above the base size it runs the leveled grid kernel (grid_kernel_impl.inl) over the
    // rectangle, saving every R-th row and segment-aligned columns, and hands each block
    // the path crosses to grid_solve one level down, with the saved lines as its borders;
    // a base rectangle is re-filled by the scalar Pointers fill (viterbi_affine_ptr's
    // expressions) and walked.  Each re-filled cell is the full fill's value, so each code
    // is the full fill's code and the walk takes Pointers' steps, ties included.
    // Tuning (measured on the prototype, tools/gridgrad): kGridK lines per dimension, base
    // blocks of kGridBase cells; memory ~ kGridK (m + n) * 3 values over all levels.
    static constexpr int  kGridK = 32;
    static constexpr long kGridBase = 64 * 64;
    static constexpr int  kGridMin = 8;      // rows / columns per block, at least
    static constexpr int  kGridDepth = 64;   // recursion levels (never reached: ~log_k(m))

    struct GridLine {                        // value(s, idx) = p[s*ss + idx*es]
        const T* p; std::size_t ss, es;
        T operator()(int s, int idx) const noexcept {
            return p[static_cast<std::size_t>(s) * ss + static_cast<std::size_t>(idx) * es];
        }
        GridLine shifted(int d) const noexcept {
            return {p + static_cast<std::ptrdiff_t>(d) * static_cast<std::ptrdiff_t>(es), ss, es};
        }
    };

    bool grid_ok_(const LevelKernels& K) const noexcept {
        if (!K.template of<T>().grid_fwd || tr_ || m_ <= hb_cutoff_ || n_ == 0) return false;
        return static_cast<T>(params_->gap_open_a) >= T(0) && static_cast<T>(params_->gap_open_b) >= T(0);
    }

    void viterbi_affine_grid(DpBuffer& buf, const LevelKernels& K) {
        const T go_a = static_cast<T>(params_->gap_open_a), ge_a = static_cast<T>(params_->gap_extend_a);
        const T go_b = static_cast<T>(params_->gap_open_b), ge_b = static_cast<T>(params_->gap_extend_b);
        // the borders, as viterbi_affine_ptr leaves row 0 and column 0 (M, X, Y)
        const std::size_t n1 = static_cast<std::size_t>(n_) + 1, m1 = static_cast<std::size_t>(m_) + 1;
        buf.gtop.resize(3 * n1); buf.gleft.resize(3 * m1);
        for (std::size_t j = 0; j < n1; ++j) {
            buf.gtop[j] = j == 0 ? T(0) : NEG_INF;
            buf.gtop[n1 + j] = NEG_INF;
            buf.gtop[2 * n1 + j] = j == 0 ? NEG_INF : -(go_a + static_cast<T>(j) * ge_a);
        }
        for (std::size_t i = 0; i < m1; ++i) {
            buf.gleft[i] = i == 0 ? T(0) : NEG_INF;
            buf.gleft[m1 + i] = i == 0 ? NEG_INF : -(go_b + static_cast<T>(i) * ge_b);
            buf.gleft[2 * m1 + i] = NEG_INF;
        }
        if (buf.glv.size() < static_cast<std::size_t>(kGridDepth)) buf.glv.resize(kGridDepth);   // never again: grid_solve holds references
        hops_.clear();
        hops_.reserve(static_cast<std::size_t>(m_ + n_));
        int i = m_, j = n_, st = -1;
        grid_solve(buf, K, 0, 0, GridLine{buf.gtop.data(), n1, 1}, GridLine{buf.gleft.data(), m1, 1},
                   0, i, j, st);
        // the borders: forced moves, as walk_affine's
        for (; i == 0 && j > 0; --j) hops_.push_back(2);
        for (; j == 0 && i > 0; --i) hops_.push_back(1);
        std::reverse(hops_.begin(), hops_.end());
        best_i_ = m_; best_j_ = n_;
        hb_start_i_ = 0; hb_start_j_ = 0;
        hirschberg_ = true;
        grid_ = true;
    }

    void grid_solve(DpBuffer& buf, const LevelKernels& K, int r0, int c0, const GridLine& top,
                    const GridLine& left, int depth, int& i, int& j, int& st) {
        const int r1 = i, c1 = j, h = r1 - r0, w = c1 - c0;
        if (static_cast<long>(h) * w <= kGridBase || h < 2 * kGridMin || w < 2 * kGridMin ||
            depth + 1 >= kGridDepth) {
            grid_base(buf, r0, r1, c0, c1, top, left, i, j, st);
            return;
        }
        auto& L = buf.glv[static_cast<std::size_t>(depth)];
        const int R = std::max(kGridMin, (h + kGridK - 1) / kGridK);
        GridFwdJob<T> J{};
        J.a = a_idx_.data(); J.b = b_idx_.data(); J.blk = blkT_; J.nalpha = nalpha_;
        J.go_a = static_cast<T>(params_->gap_open_a); J.ge_a = static_cast<T>(params_->gap_extend_a);
        J.go_b = static_cast<T>(params_->gap_open_b); J.ge_b = static_cast<T>(params_->gap_extend_b);
        J.r0 = r0; J.r1 = r1; J.c0 = c0; J.c1 = c1;
        J.top = top.p; J.top_ss = top.ss; J.top_es = top.es;
        J.left = left.p; J.left_ss = left.ss; J.left_es = left.es;
        J.R = R; J.spacing = std::max(kGridMin, (w + kGridK - 1) / kGridK);
        J.rows = &L.rows; J.cols = &L.cols; J.colid = &L.colid; J.buf = &buf;
        K.template of<T>().grid_fwd(J);
        const int W = J.W, q = J.q;
        L.gcols.clear();
        L.gcols.push_back({c0, -1});                     // the rectangle's left line
        for (int t = 0; t < q * W; ++t)
            if (L.colid[static_cast<std::size_t>(t)] >= 0) L.gcols.push_back({L.colid[static_cast<std::size_t>(t)], t});
        std::sort(L.gcols.begin(), L.gcols.end());
        while (i > r0 && j > c0) {
            const int br0 = r0 + ((i - r0 - 1) / R) * R;   // the band holding row i
            auto it = std::lower_bound(L.gcols.begin(), L.gcols.end(), std::make_pair(j, -2));
            --it;                                           // the last line left of column j
            const int bc0 = it->first, tl = it->second;
            const GridLine ctop = br0 == r0
                ? top.shifted(bc0 - c0)
                : GridLine{L.rows.data() + static_cast<std::size_t>((br0 - r0) / R - 1) * 3 * static_cast<std::size_t>(w + 1),
                           static_cast<std::size_t>(w + 1), 1}.shifted(bc0 - c0);
            const GridLine cleft = tl < 0
                ? left.shifted(br0 - r0)
                : GridLine{L.cols.data() + static_cast<std::size_t>(tl / W) * 3 * static_cast<std::size_t>(h + 1) * W + static_cast<std::size_t>(tl % W),
                           static_cast<std::size_t>(h + 1) * W, static_cast<std::size_t>(W)}.shifted(br0 - r0);
            grid_solve(buf, K, br0, bc0, ctop, cleft, depth + 1, i, j, st);
        }
    }

    // The base block: viterbi_affine_ptr's fill over [r0, r1] x [c0, c1] from its borders,
    // then the walk from (i, j) until the top row or the left column.
    void grid_base(DpBuffer& buf, int r0, int r1, int c0, int c1, const GridLine& top,
                   const GridLine& left, int& i, int& j, int& st) {
        const T go_a = static_cast<T>(params_->gap_open_a), ge_a = static_cast<T>(params_->gap_extend_a);
        const T go_b = static_cast<T>(params_->gap_open_b), ge_b = static_cast<T>(params_->gap_extend_b);
        const std::size_t bw = static_cast<std::size_t>(c1 - c0) + 1, sz = (static_cast<std::size_t>(r1 - r0) + 1) * bw;
        if (buf.gbM.size() < sz) {
            buf.gbM.resize(sz); buf.gbX.resize(sz); buf.gbY.resize(sz);
            buf.gdM.resize(sz); buf.gdX.resize(sz); buf.gdY.resize(sz);
        }
        T* bM = buf.gbM.data(); T* bX = buf.gbX.data(); T* bY = buf.gbY.data();
        unsigned char* dM = buf.gdM.data(); unsigned char* dX = buf.gdX.data(); unsigned char* dY = buf.gdY.data();
        auto at = [&](int ii, int jj) {
            return static_cast<std::size_t>(ii - r0) * bw + static_cast<std::size_t>(jj - c0);
        };
        auto pick = [](T vm, T vx, T vy) -> unsigned char {   // viterbi_affine_ptr's
            const unsigned char kxy = vx >= vy ? 1 : 2;
            return vm >= std::max(vx, vy) ? static_cast<unsigned char>(0) : kxy;
        };
        for (int jj = c0; jj <= c1; ++jj) {
            const std::size_t k = at(r0, jj);
            bM[k] = top(0, jj - c0); bX[k] = top(1, jj - c0); bY[k] = top(2, jj - c0);
        }
        for (int ii = r0 + 1; ii <= r1; ++ii) {
            const std::size_t k0 = at(ii, c0);
            bM[k0] = left(0, ii - r0); bX[k0] = left(1, ii - r0); bY[k0] = left(2, ii - r0);
            for (int jj = c0 + 1; jj <= c1; ++jj) {
                const std::size_t k = at(ii, jj), up = k - bw, dg = up - 1, lf = k - 1;
                const T dgM = bM[dg], dgX = bX[dg], dgY = bY[dg];
                bM[k] = std::max({dgM, dgX, dgY}) + subTu(ii, jj);
                dM[k] = pick(dgM, dgX, dgY);
                const T ux = bM[up] - go_b - ge_b, vx = bX[up] - ge_b, wx = bY[up] - go_b - ge_b;
                bX[k] = std::max({ux, vx, wx}); dX[k] = pick(ux, vx, wx);
                const T uy = bM[lf] - go_a - ge_a, vy = bX[lf] - go_a - ge_a, wy = bY[lf] - ge_a;
                bY[k] = std::max({uy, vy, wy}); dY[k] = pick(uy, vy, wy);
            }
        }
        if (st < 0) {   // the end cell (m, n): its argmax and the score, as viterbi_affine_ptr
            const std::size_t k = at(i, j);
            const T vm = bM[k], vx = bX[k], vy = bY[k];
            viterbi_score_ = std::max({vm, vx, vy});
            st = (vm >= vx && vm >= vy) ? 0 : (vx >= vy ? 1 : 2);
            best_tbl_ = st == 0 ? TBTable::M : (st == 1 ? TBTable::X : TBTable::Y);
        }
        while (i > r0 && j > c0) {
            const std::size_t k = at(i, j);
            hops_.push_back(static_cast<unsigned char>(st));
            if (st == 0)      { st = dM[k]; --i; --j; }
            else if (st == 1) { st = dX[k]; --i; }
            else              { st = dY[k]; --j; }
        }
    }

    // ── Score only, scalar: viterbi_affine / viterbi_linear on rolling rows ──
    // The same expressions in the same order (Full band), so the value at every cell —
    // and the score — is bit-identical to the fill's; only the tables are gone.
    // Hirschberg needs every gap penalty it uses >= 0 (NaN fails too); see run_viterbi.
    bool hb_penalties_ok_() const noexcept {
        if (tr_) {   // every row's (the rows this problem uses)
            bool ok = true;
            for (size_t r = 0; r <= static_cast<size_t>(m_); ++r) {
                ok &= tgea_[r] >= static_cast<T>(0);
                if constexpr (GM == GapModel::Affine) ok &= tgoa_[r] >= static_cast<T>(0);
                if (r > 0) {
                    ok &= tgeb_[r] >= static_cast<T>(0);
                    if constexpr (GM == GapModel::Affine) ok &= tgob_[r] >= static_cast<T>(0);
                }
            }
            return ok;
        }
        const AlignParams& p = *params_;
        const bool ext = p.gap_extend_a >= 0.0 && p.gap_extend_b >= 0.0;
        if constexpr (GM == GapModel::Linear) return ext;
        else return ext && p.gap_open_a >= 0.0 && p.gap_open_b >= 0.0;
    }
    // NWGRAD_SCORE_VARIANT: unset = auto (0), "twopass" (1), "fused3" (2) — for measuring.
    static int score_variant_env() {
        static const int v = [] {
            const char* e = std::getenv("NWGRAD_SCORE_VARIANT");
            if (!e) return 0;
            const std::string_view s(e);
            return s == "twopass" ? 1 : s == "fused3" ? 2 : 0;
        }();
        return v;
    }

    double score_affine_scalar(DpBuffer& buf) const {
        return tr_ ? score_affine_scalar_impl<true>(buf) : score_affine_scalar_impl<false>(buf);
    }
    template <bool Tr>
    double score_affine_scalar_impl(DpBuffer& buf) const {
        T go_a = static_cast<T>(params_->gap_open_a);
        T ge_a = static_cast<T>(params_->gap_extend_a);
        T go_b = static_cast<T>(params_->gap_open_b);
        T ge_b = static_cast<T>(params_->gap_extend_b);
        if constexpr (Tr) { go_a = tgoa_[0]; ge_a = tgea_[0]; }
        const std::size_t w = static_cast<std::size_t>(n_) + 1;
        for (auto* v : {&buf.rM, &buf.rX, &buf.rY, &buf.qM, &buf.qX, &buf.qY})
            if (v->size() < w) v->resize(w);
        T* pM = buf.qM.data(); T* pX = buf.qX.data(); T* pY = buf.qY.data();
        T* cM = buf.rM.data(); T* cX = buf.rX.data(); T* cY = buf.rY.data();
        constexpr bool local = AM == AlignMode::Local;
        const T ninf = NINF;
        pM[0] = static_cast<T>(0); pX[0] = ninf; pY[0] = ninf;
        for (int j = 1; j <= n_; ++j) {
            pM[j] = local ? static_cast<T>(0) : ninf;
            pX[j] = ninf;
            pY[j] = local ? ninf : -(go_a + static_cast<T>(j) * ge_a);
        }
        T best = static_cast<T>(0);
        for (int i = 1; i <= m_; ++i) {
            const T* sub;
            if constexpr (Tr) {
                go_a = tgoa_[static_cast<size_t>(i)]; ge_a = tgea_[static_cast<size_t>(i)];
                go_b = tgob_[static_cast<size_t>(i)]; ge_b = tgeb_[static_cast<size_t>(i)];
                cX[0] = local ? ninf : tbx0_[static_cast<size_t>(i)];
                sub = blkT_ + static_cast<std::size_t>(rs_[static_cast<size_t>(i) - 1]) * nalpha_;
            } else {
                cX[0] = local ? ninf : -(go_b + static_cast<T>(i) * ge_b);
                sub = blkT_ + static_cast<std::size_t>(a_idx_[i - 1]) * nalpha_;
            }
            cM[0] = local ? static_cast<T>(0) : ninf;
            cY[0] = ninf;
            // The left cell (j-1) rides in registers: read back from cM/cX/cY (which may
            // alias the p rows as far as the compiler knows), clang re-loaded the values
            // it had just stored, a store-forward on the Y chain.
            T lM = cM[0], lX = cX[0], lY = cY[0];
            const uint8_t* bcode = b_idx_.data();
            for (int j = 1; j <= n_; ++j) {
                T m_val = std::max({pM[j-1], pX[j-1], pY[j-1]}) + sub[bcode[j - 1]];
                // The open from max(M, Y) / max(M, X) first: (v - go) - ge is monotone in v,
                // so this is the three-way max's value exactly (the inter-pair kernels' form),
                // and it leaves clang's SLP vectorizer no isomorphic pairs to pack onto the
                // Y chain (it did, with shuffles: 2.3x gcc on scalar Global).
                const T x_val = std::max((std::max(pM[j], pY[j]) - go_b) - ge_b, pX[j] - ge_b);
                const T y_val = std::max((std::max(lM, lX) - go_a) - ge_a, lY - ge_a);
                if constexpr (local) {
                    m_val = std::max(m_val, static_cast<T>(0));
                    best = std::max(best, std::max({m_val, x_val, y_val}));
                }
                cM[j] = m_val; cX[j] = x_val; cY[j] = y_val;
                lM = m_val; lX = x_val; lY = y_val;
            }
            std::swap(pM, cM); std::swap(pX, cX); std::swap(pY, cY);
        }
        if constexpr (local) return static_cast<double>(best);
        else return static_cast<double>(std::max({pM[n_], pX[n_], pY[n_]}));
    }

    double score_linear_scalar(DpBuffer& buf) const {
        return tr_ ? score_linear_scalar_impl<true>(buf) : score_linear_scalar_impl<false>(buf);
    }
    template <bool Tr>
    double score_linear_scalar_impl(DpBuffer& buf) const {
        T ge_a = static_cast<T>(params_->gap_extend_a);
        T ge_b = static_cast<T>(params_->gap_extend_b);
        if constexpr (Tr) ge_a = tgea_[0];
        const std::size_t w = static_cast<std::size_t>(n_) + 1;
        if (buf.rM.size() < w) buf.rM.resize(w);
        if (buf.qM.size() < w) buf.qM.resize(w);
        T* prev = buf.qM.data(); T* cur = buf.rM.data();
        constexpr bool local = AM == AlignMode::Local;
        prev[0] = static_cast<T>(0);
        for (int j = 1; j <= n_; ++j) prev[j] = local ? static_cast<T>(0) : -static_cast<T>(j) * ge_a;
        double best = 0.0;
        for (int i = 1; i <= m_; ++i) {
            const T* sub;
            if constexpr (Tr) {
                ge_a = tgea_[static_cast<size_t>(i)]; ge_b = tgeb_[static_cast<size_t>(i)];
                cur[0] = local ? static_cast<T>(0) : tbx0_[static_cast<size_t>(i)];
                sub = blkT_ + static_cast<std::size_t>(rs_[static_cast<size_t>(i) - 1]) * nalpha_;
            } else {
                cur[0] = local ? static_cast<T>(0) : -static_cast<T>(i) * ge_b;
                sub = blkT_ + static_cast<std::size_t>(a_idx_[i - 1]) * nalpha_;
            }
            T left = cur[0];   // in a register, as in score_affine_scalar_impl
            const uint8_t* bcode = b_idx_.data();
            T rowmax = static_cast<T>(0);
            for (int j = 1; j <= n_; ++j) {
                T v = std::max({prev[j-1] + sub[bcode[j - 1]], prev[j] - ge_b, left - ge_a});
                if constexpr (local) {
                    v = std::max(v, static_cast<T>(0));
                    rowmax = std::max(rowmax, v);
                }
                cur[j] = v;
                left = v;
            }
            if constexpr (local) if (rowmax > best) best = rowmax;
            std::swap(prev, cur);
        }
        return local ? best : static_cast<double>(prev[n_]);
    }

    // Local fills: the best cell of row i, found AFTER the row.  The fill keeps only a
    // running rowmax = max over the row of max(M, X, Y) (one maxsd per cell, no branch);
    // when it beats best_local, the first column from j0 holding it is exactly the cell
    // the old in-loop strict-`>` scan stopped at (an equal later cell never replaced it),
    // and the table is picked by the same M>X>Y rule.  In the loop, gcc compiled that
    // scan to compare-and-branch per cell, mispredicted on unrelated sequences.
    template <class V>
    void local_row_best_(const V* cM, const V* cX, const V* cY, int j0, V rowmax, int i,
                         V& best_local) {
        int j = j0;
        while (!(std::max({cM[j], cX[j], cY[j]}) == rowmax)) ++j;
        const V m_val = cM[j], x_val = cX[j], y_val = cY[j];
        best_local = rowmax; best_i_ = i; best_j_ = j;
        if      (m_val >= x_val && m_val >= y_val) best_tbl_ = TBTable::M;
        else if (x_val >= y_val)                    best_tbl_ = TBTable::X;
        else                                         best_tbl_ = TBTable::Y;
    }

    void viterbi_affine(DpBuffer& buf) {
        if (tr_) viterbi_affine_impl<true>(buf);
        else     viterbi_affine_impl<false>(buf);
    }
    template <bool Tr>
    void viterbi_affine_impl(DpBuffer& buf) {
        band_fill(buf.VM, NINF);
        band_fill(buf.VX, NINF);
        band_fill(buf.VY, NINF);

        // Gap penalties in the Viterbi precision T.  Converting once, up front, is what
        // makes this scalar fill bit-exact with the T-typed simd kernel: both then do
        // (v - go) - ge in T with the same left association, so their VM/VX/VY tables
        // agree to the last bit (which the argmax tracebacks below depend on).
        // Tracked: the same, per row (the arrays hold the same casts).
        T go_a = static_cast<T>(params_->gap_open_a);
        T ge_a = static_cast<T>(params_->gap_extend_a);
        T go_b = static_cast<T>(params_->gap_open_b);
        T ge_b = static_cast<T>(params_->gap_extend_b);

        if constexpr (AM == AlignMode::Global) {
            at(buf.VM, 0, 0) = static_cast<T>(0);
            const int bi = border_rows(), bj = border_cols();
            if constexpr (Tr) {
                for (int i = 1; i <= bi; ++i) at(buf.VX, i, 0) = tbx0_[static_cast<size_t>(i)];
                for (int j = 1; j <= bj; ++j) at(buf.VY, 0, j) = -(tgoa_[0] + static_cast<T>(j) * tgea_[0]);
            } else {
            // VX along column 0: all gaps in B (consuming A), uses gap_b params
            for (int i = 1; i <= bi; ++i)
                at(buf.VX, i, 0) = -(go_b + static_cast<T>(i) * ge_b);
            // VY along row 0: all gaps in A (consuming B), uses gap_a params
            for (int j = 1; j <= bj; ++j)
                at(buf.VY, 0, j) = -(go_a + static_cast<T>(j) * ge_a);
            }
        } else {
            for (int i = 0; i <= m_; ++i) at(buf.VM, i, 0) = static_cast<T>(0);
            for (int j = 0; j <= n_; ++j) at(buf.VM, 0, j) = static_cast<T>(0);
        }

        best_i_ = 0; best_j_ = 0; best_tbl_ = TBTable::M;
        T best_local = static_cast<T>(0);

        // Row pointers, not rat()/at(): run_viterbi cleared the layout flags, so the
        // tables are row-major here, and cell_index()'s layout switch inside the loop is
        // dead weight that gcc unswitches and clang does not.  The left cell rides in
        // registers (clang re-loaded the cells it had just stored), the substitution row
        // is hoisted per row.  Same operands, same order: the tables are unchanged.
        const uint8_t* bcode = b_idx_.data();
        for (int i = 1; i <= m_; ++i) {
            const T* srow;
            if constexpr (Tr) {
                go_a = tgoa_[static_cast<size_t>(i)]; ge_a = tgea_[static_cast<size_t>(i)];
                go_b = tgob_[static_cast<size_t>(i)]; ge_b = tgeb_[static_cast<size_t>(i)];
                srow = blkT_ + static_cast<size_t>(rs_[static_cast<size_t>(i) - 1]) * static_cast<size_t>(nalpha_);
            } else {
                srow = blkT_ + static_cast<size_t>(a_idx_[static_cast<size_t>(i) - 1]) * static_cast<size_t>(nalpha_);
            }
            const size_t rp = static_cast<size_t>(i - 1) * stride_, rc = static_cast<size_t>(i) * stride_;
            const T* pM = buf.VM.data() + rp; const T* pX = buf.VX.data() + rp; const T* pY = buf.VY.data() + rp;
            T* cM = buf.VM.data() + rc; T* cX = buf.VX.data() + rc; T* cY = buf.VY.data() + rc;
            const int j0 = jlo(i), j1 = jhi(i);
            if (j0 > j1) continue;   // an empty band row reads nothing (as before)
            T lM = cM[j0 - 1], lX = cX[j0 - 1], lY = cY[j0 - 1];
            T rowmax = static_cast<T>(0);   // Local: see local_row_best_
            for (int j = j0; j <= j1; ++j) {
                T diag  = std::max({pM[j-1], pX[j-1], pY[j-1]});
                T m_val = diag + srow[bcode[static_cast<size_t>(j) - 1]];
                // X state: gap in B (advance i), uses gap_b params.  Y: gap in A, gap_a.
                // The open from the max of the two opening states first: (v - go) - ge is
                // monotone in v, so the value is the three-way max's exactly (the
                // inter-pair kernels' form), and clang's SLP vectorizer finds no isomorphic
                // subtraction pairs to pack onto the Y chain (it did, with shuffles).
                T x_val = std::max((std::max(pM[j], pY[j]) - go_b) - ge_b, pX[j] - ge_b);
                T y_val = std::max((std::max(lM, lX) - go_a) - ge_a, lY - ge_a);

                if constexpr (AM == AlignMode::Local) {
                    m_val = std::max(m_val, static_cast<T>(0));
                    rowmax = std::max(rowmax, std::max({m_val, x_val, y_val}));
                }

                cM[j] = m_val; cX[j] = x_val; cY[j] = y_val;
                lM = m_val; lX = x_val; lY = y_val;
            }
            if constexpr (AM == AlignMode::Local)
                if (rowmax > best_local) local_row_best_(cM, cX, cY, j0, rowmax, i, best_local);
        }

        if constexpr (AM == AlignMode::Global) {
            T vm = rat(buf.VM, m_, n_), vx = rat(buf.VX, m_, n_), vy = rat(buf.VY, m_, n_);
            viterbi_score_ = std::max({vm, vx, vy});
            best_i_ = m_; best_j_ = n_;
            if      (vm >= vx && vm >= vy) best_tbl_ = TBTable::M;
            else if (vx >= vy)             best_tbl_ = TBTable::X;
            else                            best_tbl_ = TBTable::Y;
        } else {
            viterbi_score_ = best_local;
        }
    }

    // ═════════════════════════════════════════════════════════════════════════
    // Forward-backward — Linear gap model
    // ═════════════════════════════════════════════════════════════════════════

    void fwdbwd_linear(DpBuffer& buf) {
        // ── Forward ──
        banded_fill(buf.F);

        if constexpr (AM == AlignMode::Global) {
            sat(buf.F, 0, 0) = 0.0;
            const int bi = border_rows(), bj = border_cols();
            if (tr_) build_dbx0_();
            for (int i = 1; i <= bi; ++i)
                sat(buf.F, i, 0) = tr_ ? dbx0_[static_cast<size_t>(i)] : -static_cast<double>(i) * params_->gap_extend_b;
            for (int j = 1; j <= bj; ++j)
                sat(buf.F, 0, j) = -static_cast<double>(j) * drow_ge_a(0);
        } else {
            for (int i = 0; i <= m_; ++i) sat(buf.F, i, 0) = 0.0;
            for (int j = 0; j <= n_; ++j) sat(buf.F, 0, j) = 0.0;
        }

        for (int i = 1; i <= m_; ++i) {
            for (int j = jlo(i); j <= jhi(i); ++j) {
                double v = lse3(
                    srat(buf.F, i-1, j-1) + sub(i, j),
                    srat(buf.F, i-1, j)   - drow_ge_b(i),
                    srat(buf.F, i,   j-1) - drow_ge_a(i)
                );
                if constexpr (AM == AlignMode::Local) v = lse2(v, 0.0);
                sat(buf.F, i, j) = v;
            }
        }

        if constexpr (AM == AlignMode::Global) {
            log_z_ = srat(buf.F, m_, n_);
        } else {
            log_z_ = NEG_INF;
            for (int i = 0; i <= m_; ++i)
                for (int j = jlo0(i); j <= jhi0(i); ++j)
                    log_z_ = lse2(log_z_, srat(buf.F, i, j));
        }

        // ── Backward ──
        if constexpr (AM == AlignMode::Global) {
            band_fill(buf.B, NEG_INF);
            sat(buf.B, m_, n_) = 0.0;
        } else {
            band_fill(buf.B, 0.0);
        }

        for (int i = m_; i >= 0; --i) {
            for (int j = jhi0(i); j >= jlo0(i); --j) {
                double bval = srat(buf.B, i, j);
                if (bval == NEG_INF) continue;
                if (i > 0 && j > 0)
                    sat(buf.B,i-1,j-1) = lse2(srat(buf.B,i-1,j-1),
                                              bval + sub(i, j));
                if (i > 0)
                    sat(buf.B,i-1,j)   = lse2(srat(buf.B,i-1,j),   bval - drow_ge_b(i));
                if (j > 0)
                    sat(buf.B,i,j-1)   = lse2(srat(buf.B,i,j-1),   bval - drow_ge_a(i));
            }
        }
    }

    // Local borders are constants (a free start), not gap moves: the forward pass sets
    // F(i,0) = F(0,j) = 0 rather than deriving them from a neighbour minus a gap cost,
    // so a gap transition INTO a border cell does not exist and must not be counted,
    // even though the backward pass leaves finite values there.  Global borders really
    // are charged gap runs, so there the border targets stay in.
    static constexpr int kGapTargetMin = (AM == AlignMode::Local) ? 1 : 0;

    void soft_grad_linear(const DpBuffer& buf, AlignParams& grad) const {
        double* gblk = grad_block(grad);
        // Matrix gradient: match steps (i-1,j-1) → (i,j)
        for (int i = 1; i <= m_; ++i) {
            for (int j = jlo(i); j <= jhi(i); ++j) {
                double bval = srat(buf.B, i, j);
                if (bval == NEG_INF) continue;
                double log_p = srat(buf.F, i-1, j-1)
                               + sub(i, j)
                               + bval - log_z_;
                gblk[sub_off(i, j)] += std::exp(log_p);
            }
        }

        // gap_extend_b: B-gap steps (i-1,j) → (i,j), cost = -gap_extend_b
        for (int i = 1; i <= m_; ++i) {
            for (int j = std::max(kGapTargetMin, jlo0(i)); j <= jhi0(i); ++j) {
                double bval = srat(buf.B, i, j);
                if (bval == NEG_INF) continue;
                double fval = srat(buf.F, i-1, j);
                if (fval == NEG_INF) continue;
                grad.gap_extend_b -= std::exp(fval - params_->gap_extend_b + bval - log_z_);
            }
        }

        // gap_extend_a: A-gap steps (i,j-1) → (i,j), cost = -gap_extend_a
        for (int i = kGapTargetMin; i <= m_; ++i) {
            for (int j = std::max(1, jlo0(i)); j <= jhi0(i); ++j) {
                double bval = srat(buf.B, i, j);
                if (bval == NEG_INF) continue;
                double fval = srat(buf.F, i, j-1);
                if (fval == NEG_INF) continue;
                grad.gap_extend_a -= std::exp(fval - params_->gap_extend_a + bval - log_z_);
            }
        }
    }

    // ═════════════════════════════════════════════════════════════════════════
    // Forward-backward — Affine gap model
    // ═════════════════════════════════════════════════════════════════════════

    void fwdbwd_affine(DpBuffer& buf) {
        // ── Forward ──
        band_fill(buf.FM, NEG_INF);
        band_fill(buf.FX, NEG_INF);
        band_fill(buf.FY, NEG_INF);

        if constexpr (AM == AlignMode::Global) {
            sat(buf.FM, 0, 0) = 0.0;
            const int bi = border_rows(), bj = border_cols();
            if (tr_) build_dbx0_();
            for (int i = 1; i <= bi; ++i)
                sat(buf.FX, i, 0) = tr_ ? dbx0_[static_cast<size_t>(i)]
                                        : -(params_->gap_open_b + static_cast<double>(i) * params_->gap_extend_b);
            for (int j = 1; j <= bj; ++j)
                sat(buf.FY, 0, j) = -(drow_go_a(0) + static_cast<double>(j) * drow_ge_a(0));
        } else {
            for (int i = 0; i <= m_; ++i) sat(buf.FM, i, 0) = 0.0;
            for (int j = 0; j <= n_; ++j) sat(buf.FM, 0, j) = 0.0;
        }

        for (int i = 1; i <= m_; ++i) {
            for (int j = jlo(i); j <= jhi(i); ++j) {
                double diag  = lse3(srat(buf.FM,i-1,j-1), srat(buf.FX,i-1,j-1), srat(buf.FY,i-1,j-1));
                double m_val = diag + sub(i, j);
                const double gob = drow_go_b(i), geb = drow_ge_b(i), goa = drow_go_a(i), gea = drow_ge_a(i);
                double x_val = lse3(
                    srat(buf.FM,i-1,j) - gob - geb,
                    srat(buf.FX,i-1,j)       - geb,
                    srat(buf.FY,i-1,j) - gob - geb);
                double y_val = lse3(
                    srat(buf.FM,i,j-1) - goa - gea,
                    srat(buf.FX,i,j-1) - goa - gea,
                    srat(buf.FY,i,j-1)       - gea);
                if constexpr (AM == AlignMode::Local) m_val = lse2(m_val, 0.0);
                sat(buf.FM, i, j) = m_val; sat(buf.FX, i, j) = x_val; sat(buf.FY, i, j) = y_val;
            }
        }

        if constexpr (AM == AlignMode::Global) {
            log_z_ = lse3(srat(buf.FM,m_,n_), srat(buf.FX,m_,n_), srat(buf.FY,m_,n_));
        } else {
            log_z_ = NEG_INF;
            for (int i = 0; i <= m_; ++i)
                for (int j = jlo0(i); j <= jhi0(i); ++j)
                    log_z_ = lse2(log_z_, lse3(srat(buf.FM,i,j), srat(buf.FX,i,j), srat(buf.FY,i,j)));
        }

        // ── Backward ──
        if constexpr (AM == AlignMode::Global) {
            band_fill(buf.BM, NEG_INF);
            band_fill(buf.BX, NEG_INF);
            band_fill(buf.BY, NEG_INF);
            sat(buf.BM,m_,n_) = 0.0; sat(buf.BX,m_,n_) = 0.0; sat(buf.BY,m_,n_) = 0.0;
        } else {
            band_fill(buf.BM, 0.0);
            band_fill(buf.BX, 0.0);
            band_fill(buf.BY, 0.0);
        }

        for (int i = m_; i >= 0; --i) {
            for (int j = jhi0(i); j >= jlo0(i); --j) {
                double bm = srat(buf.BM,i,j), bx = srat(buf.BX,i,j), by = srat(buf.BY,i,j);

                if (bm != NEG_INF && i > 0 && j > 0) {
                    double contrib = bm + sub(i, j);
                    sat(buf.BM,i-1,j-1) = lse2(srat(buf.BM,i-1,j-1), contrib);
                    sat(buf.BX,i-1,j-1) = lse2(srat(buf.BX,i-1,j-1), contrib);
                    sat(buf.BY,i-1,j-1) = lse2(srat(buf.BY,i-1,j-1), contrib);
                }
                // X state (gap in B) contribution to predecessors at (i-1,j)
                if (bx != NEG_INF && i > 0) {
                    const double gob = drow_go_b(i), geb = drow_ge_b(i);
                    sat(buf.BM,i-1,j) = lse2(srat(buf.BM,i-1,j), bx - gob - geb);
                    sat(buf.BX,i-1,j) = lse2(srat(buf.BX,i-1,j), bx - geb);
                    sat(buf.BY,i-1,j) = lse2(srat(buf.BY,i-1,j), bx - gob - geb);
                }
                // Y state (gap in A) contribution to predecessors at (i,j-1)
                if (by != NEG_INF && j > 0) {
                    const double goa = drow_go_a(i), gea = drow_ge_a(i);
                    sat(buf.BM,i,j-1) = lse2(srat(buf.BM,i,j-1), by - goa - gea);
                    sat(buf.BX,i,j-1) = lse2(srat(buf.BX,i,j-1), by - goa - gea);
                    sat(buf.BY,i,j-1) = lse2(srat(buf.BY,i,j-1), by - gea);
                }
            }
        }
    }

    void soft_grad_affine(const DpBuffer& buf, AlignParams& grad) const {
        double* gblk = grad_block(grad);
        // Matrix gradient: match steps
        for (int i = 1; i <= m_; ++i) {
            for (int j = jlo(i); j <= jhi(i); ++j) {
                double bm = srat(buf.BM, i, j);
                if (bm == NEG_INF) continue;
                double pred  = lse3(srat(buf.FM,i-1,j-1), srat(buf.FX,i-1,j-1), srat(buf.FY,i-1,j-1));
                double log_p = pred + sub(i, j) + bm - log_z_;
                gblk[sub_off(i, j)] += std::exp(log_p);
            }
        }

        // gap_extend_b: expected # of X-state steps (gap in B)
        for (int i = 1; i <= m_; ++i) {
            for (int j = jlo0(i); j <= jhi0(i); ++j) {
                double fx = srat(buf.FX, i, j), bx = srat(buf.BX, i, j);
                if (fx == NEG_INF || bx == NEG_INF) continue;
                grad.gap_extend_b -= std::exp(fx + bx - log_z_);
            }
        }

        // gap_extend_a: expected # of Y-state steps (gap in A)
        for (int i = 0; i <= m_; ++i) {
            for (int j = std::max(1, jlo0(i)); j <= jhi0(i); ++j) {
                double fy = srat(buf.FY, i, j), by = srat(buf.BY, i, j);
                if (fy == NEG_INF || by == NEG_INF) continue;
                grad.gap_extend_a -= std::exp(fy + by - log_z_);
            }
        }

        // gap_open_b: expected # of B-gap openings (M→X or Y→X transitions).  Local
        // border X cells are unreachable (FX = -inf there), so no opening lands on one.
        for (int i = 1; i <= m_; ++i) {
            for (int j = std::max(kGapTargetMin, jlo0(i)); j <= jhi0(i); ++j) {
                double bx = srat(buf.BX, i, j);
                if (bx == NEG_INF) continue;
                double fm = srat(buf.FM, i-1, j), fy_prev = srat(buf.FY, i-1, j);
                double log_open = lse2(fm, fy_prev)
                                  - params_->gap_open_b - params_->gap_extend_b
                                  + bx - log_z_;
                if (log_open > -700) grad.gap_open_b -= std::exp(log_open);
            }
        }

        // gap_open_a: expected # of A-gap openings (M→Y or X→Y transitions)
        for (int i = kGapTargetMin; i <= m_; ++i) {
            for (int j = std::max(1, jlo0(i)); j <= jhi0(i); ++j) {
                double by = srat(buf.BY, i, j);
                if (by == NEG_INF) continue;
                double fm = srat(buf.FM, i, j-1), fx_prev = srat(buf.FX, i, j-1);
                double log_open = lse2(fm, fx_prev)
                                  - params_->gap_open_a - params_->gap_extend_a
                                  + by - log_z_;
                if (log_open > -700) grad.gap_open_a -= std::exp(log_open);
            }
        }
    }

    // ═════════════════════════════════════════════════════════════════════════
    // Forward-backward — scaled probability space (SoftImpl::Scaled)
    // ═════════════════════════════════════════════════════════════════════════
    //
    // The same recurrences as fwdbwd_linear/fwdbwd_affine with exp() applied: every
    // log-space lse becomes a +, every +score a ×exp(score), so a cell costs only
    // multiply-adds.  exp() runs |Σ|² + 4 times per problem, log() once.
    //
    // Range.  Each forward row i is stored times 2^-S[i] (S[i] cumulative; the factor
    // is a power of two, so rescaling is exact; the row max lands in [1, 2)), each
    // backward row times 2^-T[i], and Z = 2^ze · zr.  Posteriors are then
    // F̂·B̂ · 2^(S+T-ze)/zr, an ldexp — no exp.
    //
    // All terms are non-negative, so normal values carry full relative precision, and
    // the only error is MASS LOST to the bottom of the double range: a term that went
    // subnormal or to 0 (a deep cell, a weight like exp(-100), a local free start
    // 2^-S).  Any lost term is below DBL_MIN in its row's scaled units (times 2^-k if
    // the rescale scaled the row UP), and mass lost at a cell moves any posterior —
    // and Z — by at most its own posterior, F_lost·B/Z = lost · B̂ · 2^(S+T-ze)/zr.
    // With B̂ (or F̂) at most 2 after rescaling, each row's possible damage is bounded
    // by quantities already in hand, so the run sums that bound and fails (returns
    // false) when it exceeds 2^-45 (kSpLossTol), or when a weight or a row overflows.  The
    // bound is rigorous, not a heuristic: a pair passes exactly when nothing that was
    // lost could matter at the tolerance.  What fails in practice: a row whose forward
    // and backward mass sit ~1000 binary orders apart (extreme score ranges, very low
    // temperature), or a step score beyond ~+709 (exp overflows).  SoftImpl decides
    // whether that throws or falls back to the log path.

    void run_fwdbwd(DpBuffer& buf) {
        soft_counts_ready_ = false;

        if (soft_impl_ != SoftImpl::Log) {
            bool ok;
            ok = fwdbwd_scaled(buf);
            if (ok) {
                soft_scaled_used_ = soft_counts_ready_ = true;
                log_z_ *= soft_temp_;
                return;
            }
            if (soft_impl_ == SoftImpl::Scaled)
                throw std::domain_error(
                    "nwgrad: soft path: this pair's dynamic range does not fit the scaled "
                    "probability-space forward-backward (lengths " + std::to_string(m_) +
                    " x " + std::to_string(n_) + "; mass lost to underflow could move the "
                    "result by more than 2^-45, or a step score exceeds ~709 so exp() "
                    "overflows — typically a very low temperature or extreme scores). "
                    "Use soft_impl=\"scaled_or_log\" (falls back per pair) or \"log\".");
        }
        soft_scaled_used_ = false;
        ensure_fwdbwd_buf(buf);
        if (soft_temp_ == 1.0) {
            if constexpr (GM == GapModel::Linear) fwdbwd_linear(buf);
            else                                   fwdbwd_affine(buf);
            if (want_post_) posterior_rows_log(buf);
            if (tr_) soft_record_log_(buf);   // tracked: the per-row record, now
            return;   // soft_grad() sweeps the tables lazily, as it always has
        }
        // T != 1: run the log path on θ/T, and sweep the gradient NOW, while params_ and
        // log_z_ still describe θ/T; the counts are kept like the scaled path's.
        const AlignParams* orig = params_;
        const double* orig_blk = blk_;
        const AlignParams tp = (*orig) * (1.0 / soft_temp_);
        params_ = &tp;
        blk_ = tp.matrix.data();
        try {
            if (tr_) {   // the per-row record, of theta/T
                if constexpr (GM == GapModel::Linear) fwdbwd_linear(buf); else fwdbwd_affine(buf);
                if (want_post_) posterior_rows_log(buf);
                soft_record_log_(buf);
                params_ = orig; blk_ = orig_blk;
                log_z_ *= soft_temp_;
                soft_counts_ready_ = true;
                return;
            }
            AlignParams g = AlignParams::zeros_like(tp);
            if constexpr (GM == GapModel::Linear) { fwdbwd_linear(buf); soft_grad_linear(buf, g); }
            else                                   { fwdbwd_affine(buf); soft_grad_affine(buf, g); }
            if (want_post_) posterior_rows_log(buf);
            const size_t nn = static_cast<size_t>(nalpha_) * nalpha_;
            scnt_.assign(g.matrix.data(), g.matrix.data() + nn);
            sg_go_a_ = g.gap_open_a; sg_ge_a_ = g.gap_extend_a;
            sg_go_b_ = g.gap_open_b; sg_ge_b_ = g.gap_extend_b;
        } catch (...) { params_ = orig; blk_ = orig_blk; throw; }
        params_ = orig;
        blk_ = orig_blk;
        log_z_ *= soft_temp_;
        soft_counts_ready_ = true;
    }

    void add_soft_record_(AlignParams& grad) const {
        double* gblk = grad_block(grad);
        const size_t na = static_cast<size_t>(nalpha_);
        for (int i = 1; i <= m_; ++i) {
            double* dst = gblk + static_cast<size_t>(rs_[static_cast<size_t>(i) - 1]) * na;
            const double* src = srec_cnt_.data() + static_cast<size_t>(i - 1) * na;
            for (size_t c = 0; c < na; ++c) dst[c] += src[c];
        }
        for (int i = 0; i <= m_; ++i) {
            const double* g = srec_gap_.data() + 4 * static_cast<size_t>(i);
            grad.gap_ref(gset(i), AlignParams::kOpenA) += g[0];
            grad.gap_ref(gset(i), AlignParams::kExtendA) += g[1];
            if (i > 0) {
                grad.gap_ref(tset(i), AlignParams::kOpenB) += g[2];
                grad.gap_ref(tset(i), AlignParams::kExtendB) += g[3];
            }
        }
    }
    // The tracked log path's record, from the forward/backward tables of params_ (theta,
    // or theta/T): soft_grad_linear / soft_grad_affine's sums, kept per row.
    void soft_record_log_(const DpBuffer& buf) {
        const size_t na = static_cast<size_t>(nalpha_);
        srec_cnt_.assign(static_cast<size_t>(m_) * na, 0.0);
        srec_gap_.assign(4 * (static_cast<size_t>(m_) + 1), 0.0);
        double* G = srec_gap_.data();
        if constexpr (GM == GapModel::Linear) {
            for (int i = 1; i <= m_; ++i)
                for (int j = jlo(i); j <= jhi(i); ++j) {
                    const double bval = srat(buf.B, i, j);
                    if (bval == NEG_INF) continue;
                    srec_cnt_[static_cast<size_t>(i - 1) * na + b_idx_[static_cast<size_t>(j) - 1]] +=
                        std::exp(srat(buf.F, i - 1, j - 1) + sub(i, j) + bval - log_z_);
                }
            for (int i = 1; i <= m_; ++i)
                for (int j = std::max(kGapTargetMin, jlo0(i)); j <= jhi0(i); ++j) {
                    const double bval = srat(buf.B, i, j), fval = srat(buf.F, i - 1, j);
                    if (bval == NEG_INF || fval == NEG_INF) continue;
                    G[4 * i + 3] -= std::exp(fval - drow_ge_b(i) + bval - log_z_);
                }
            for (int i = kGapTargetMin; i <= m_; ++i)
                for (int j = std::max(1, jlo0(i)); j <= jhi0(i); ++j) {
                    const double bval = srat(buf.B, i, j), fval = srat(buf.F, i, j - 1);
                    if (bval == NEG_INF || fval == NEG_INF) continue;
                    G[4 * i + 1] -= std::exp(fval - drow_ge_a(i) + bval - log_z_);
                }
        } else {
            for (int i = 1; i <= m_; ++i)
                for (int j = jlo(i); j <= jhi(i); ++j) {
                    const double bm = srat(buf.BM, i, j);
                    if (bm == NEG_INF) continue;
                    const double pred = lse3(srat(buf.FM, i-1, j-1), srat(buf.FX, i-1, j-1), srat(buf.FY, i-1, j-1));
                    srec_cnt_[static_cast<size_t>(i - 1) * na + b_idx_[static_cast<size_t>(j) - 1]] +=
                        std::exp(pred + sub(i, j) + bm - log_z_);
                }
            for (int i = 1; i <= m_; ++i)
                for (int j = jlo0(i); j <= jhi0(i); ++j) {
                    const double fx = srat(buf.FX, i, j), bx = srat(buf.BX, i, j);
                    if (fx == NEG_INF || bx == NEG_INF) continue;
                    G[4 * i + 3] -= std::exp(fx + bx - log_z_);
                }
            for (int i = 0; i <= m_; ++i)
                for (int j = std::max(1, jlo0(i)); j <= jhi0(i); ++j) {
                    const double fy = srat(buf.FY, i, j), by = srat(buf.BY, i, j);
                    if (fy == NEG_INF || by == NEG_INF) continue;
                    G[4 * i + 1] -= std::exp(fy + by - log_z_);
                }
            for (int i = 1; i <= m_; ++i)
                for (int j = std::max(kGapTargetMin, jlo0(i)); j <= jhi0(i); ++j) {
                    const double bx = srat(buf.BX, i, j);
                    if (bx == NEG_INF) continue;
                    const double lo = lse2(srat(buf.FM, i-1, j), srat(buf.FY, i-1, j))
                                      - drow_go_b(i) - drow_ge_b(i) + bx - log_z_;
                    if (lo > -700) G[4 * i + 2] -= std::exp(lo);
                }
            for (int i = kGapTargetMin; i <= m_; ++i)
                for (int j = std::max(1, jlo0(i)); j <= jhi0(i); ++j) {
                    const double by = srat(buf.BY, i, j);
                    if (by == NEG_INF) continue;
                    const double lo = lse2(srat(buf.FM, i, j-1), srat(buf.FX, i, j-1))
                                      - drow_go_a(i) - drow_ge_a(i) + by - log_z_;
                    if (lo > -700) G[4 * i] -= std::exp(lo);
                }
        }
    }
    // The log path's Global column-0 border (tracked), in double: -(go_b(1) + C_i), C_i
    // the sum of ge_b over rows 1..i (linear: -C_i), by the T-precision rule's runs.
    void build_dbx0_() {
        const size_t m = static_cast<size_t>(m_);
        dbx0_.assign(m + 1, 0.0);
        double csum = 0.0; size_t s0 = 1;
        for (size_t i = 1; i <= m; ++i) {
            const double ge = drow_ge_b(static_cast<int>(i));
            if (i > 1 && std::bit_cast<uint64_t>(ge) != std::bit_cast<uint64_t>(drow_ge_b(static_cast<int>(i) - 1))) {
                csum = (s0 == 1) ? static_cast<double>(i - 1) * drow_ge_b(static_cast<int>(i) - 1)
                                 : csum + static_cast<double>(i - s0) * drow_ge_b(static_cast<int>(i) - 1);
                s0 = i;
            }
            const double C = (s0 == 1) ? static_cast<double>(i) * ge : csum + static_cast<double>(i - s0 + 1) * ge;
            if constexpr (GM == GapModel::Linear) dbx0_[i] = -C;
            else                                   dbx0_[i] = -(drow_go_b(1) + C);
        }
    }

    void add_scaled_counts(AlignParams& grad) const {
        double* gblk = grad_block(grad);
        const size_t nn = static_cast<size_t>(nalpha_) * nalpha_;
        for (size_t k = 0; k < nn; ++k) gblk[k] += scnt_[k];
        grad.gap_open_a   += sg_go_a_;
        grad.gap_extend_a += sg_ge_a_;
        grad.gap_open_b   += sg_go_b_;
        grad.gap_extend_b += sg_ge_b_;
    }

    // Weights only need to be finite: a weight that underflows is lost mass, which
    // the per-row bound accounts for.
    static bool weight_bad(double w) noexcept {
        return !(w <= std::numeric_limits<double>::max());
    }
    // exp of the substitution block; false if any weight is out of range.
    bool prepare_scaled() {
        const size_t nn = static_cast<size_t>(nalpha_) * nalpha_;
        es_.resize(nn);
        scnt_.assign(nn, 0.0);
        srow_.resize(static_cast<size_t>(nalpha_));
        sg_go_a_ = sg_ge_a_ = sg_go_b_ = sg_ge_b_ = 0.0;
        bool bad = false;
        const double it = 1.0 / soft_temp_;
        for (size_t k = 0; k < nn; ++k) { es_[k] = std::exp(blk_[k] * it); bad |= weight_bad(es_[k]); }
        return !bad;
    }

    // Query profile: prof[a·w + j] = exp(score(a, b[j-1]) / T) for j = 1..n, 0 at j = 0
    // and j = n+1, so a row reads its substitution weights contiguously.
    const double* build_profile(DpBuffer& buf, size_t w) {
        const size_t na = static_cast<size_t>(nalpha_);
        buf.sqp.assign(na * w, 0.0);
        double* P = buf.sqp.data();
        for (size_t a = 0; a < na; ++a) {
            const double* er = es_.data() + a * na;
            double* pr = P + a * w;
            for (int j = 1; j <= n_; ++j) pr[j] = er[b_idx_[static_cast<size_t>(j) - 1]];
        }
        return P;
    }

    // soft_guide="posterior" on the log path: row i's column of greatest EXIT mass (see
    // SoftPairJob::gpost), from the full log tables (first on ties, over the band) — as
    // the scaled kernels.  Each row is taken relative to its largest posterior.
    void posterior_rows_log(const DpBuffer& buf) {
        post_rows_.assign(static_cast<size_t>(m_) + 1, 0);
        std::vector<double> lp, ly;
        for (int i = 0; i <= m_; ++i) {
            const double lea = -drow_ge_a(i);   // log ea (params_ is theta/T here)
            const int lo = jlo0(i), hi = jhi0(i);
            if (lo > hi) continue;
            lp.assign(static_cast<size_t>(hi - lo + 2), NEG_INF);
            ly.assign(static_cast<size_t>(hi - lo + 2), NEG_INF);
            double rmax = NEG_INF;
            for (int j = lo; j <= hi; ++j) {
                double v, y;
                if constexpr (GM == GapModel::Linear) {
                    v = srat(buf.F, i, j) + srat(buf.B, i, j);
                    y = (j < hi) ? srat(buf.F, i, j) + lea + srat(buf.B, i, j + 1) : NEG_INF;
                } else {
                    v = lse3(srat(buf.FM, i, j) + srat(buf.BM, i, j),
                             srat(buf.FX, i, j) + srat(buf.BX, i, j),
                             srat(buf.FY, i, j) + srat(buf.BY, i, j));
                    y = (j < hi) ? srat(buf.FY, i, j + 1) + srat(buf.BY, i, j + 1) : NEG_INF;
                }
                lp[static_cast<size_t>(j - lo)] = v; ly[static_cast<size_t>(j - lo)] = y;
                rmax = std::max(rmax, v);
            }
            double bv = -1.0; int bj = lo;
            for (int j = lo; j <= hi; ++j) {
                const double e = std::exp(lp[static_cast<size_t>(j - lo)] - rmax) -
                                 std::exp(ly[static_cast<size_t>(j - lo)] - rmax);
                if (e > bv) { bv = e; bj = j; }
            }
            post_rows_[static_cast<size_t>(i)] = bj;
        }
    }

    // Both gap models: build the SoftPairJob and run the per-pair kernel of this
    // aligner's backend (soft_kernel_impl.inl; the baseline copy when the backend is
    // scalar_fallback or its level has none).  False: out of range (see run_fwdbwd).
    bool fwdbwd_scaled(DpBuffer& buf) {
        constexpr bool lin = (GM == GapModel::Linear);
        const double it = 1.0 / soft_temp_;
        SoftPairJob J{};
        const size_t w = static_cast<size_t>(n_) + 2;
        if (tr_) {
            // Tracked: each row's weights, and a profile row per DP row (its stacked row,
            // exp'd: nalpha exps a row, not per cell); counts and gaps per row.
            const size_t na = static_cast<size_t>(nalpha_), m = static_cast<size_t>(m_);
            srw_.assign(4 * (m + 1), 0.0);
            bool bad = false;
            for (int i = 0; i <= m_; ++i) {
                double* r = srw_.data() + 4 * static_cast<size_t>(i);
                r[1] = std::exp(-drow_ge_a(i) * it);
                if (!lin) r[0] = std::exp(-(drow_go_a(i) + drow_ge_a(i)) * it);
                bad |= weight_bad(r[0]) || weight_bad(r[1]);
                if (i > 0) {
                    r[3] = std::exp(-drow_ge_b(i) * it);
                    if (!lin) r[2] = std::exp(-(drow_go_b(i) + drow_ge_b(i)) * it);
                    bad |= weight_bad(r[2]) || weight_bad(r[3]);
                }
            }
            buf.sqp.assign(m * w, 0.0);
            srow_.resize(na);
            for (size_t i = 0; i < m; ++i) {
                const double* br = blk_ + static_cast<size_t>(rs_[i]) * na;
                for (size_t c = 0; c < na; ++c) { srow_[c] = std::exp(br[c] * it); bad |= weight_bad(srow_[c]); }
                double* pr = buf.sqp.data() + i * w;
                for (int j = 1; j <= n_; ++j) pr[j] = srow_[b_idx_[static_cast<size_t>(j) - 1]];
            }
            if (bad) return false;
            scnt_.assign(m * na, 0.0);
            srec_gap_.assign(4 * (m + 1), 0.0);
            J.rw = srw_.data();
            J.grow = srec_gap_.data();
        } else {
            if (!prepare_scaled()) return false;
        }
        if (want_post_) { post_rows_.assign(static_cast<size_t>(m_) + 1, 0); J.gpost = post_rows_.data(); }
        if (!tr_) {
        J.ea = std::exp(-params_->gap_extend_a * it);
        J.eb = std::exp(-params_->gap_extend_b * it);
        if (!lin) {
            J.oa = std::exp(-(params_->gap_open_a + params_->gap_extend_a) * it);
            J.ob = std::exp(-(params_->gap_open_b + params_->gap_extend_b) * it);
            if (weight_bad(J.oa) || weight_bad(J.ob)) return false;
        }
        if (weight_bad(J.ea) || weight_bad(J.eb)) return false;
        }
        if constexpr (lin) { if (buf.F.size() < sz_) buf.F.resize(sz_); }
        else for (auto* v : {&buf.FM, &buf.FX, &buf.FY}) if (v->size() < sz_) v->resize(sz_);
        for (DVec* r : {&buf.sb0, &buf.sb1, &buf.sb2, &buf.sb3, &buf.sb4, &buf.sb5})
            if (r->size() < w) r->resize(w);
        const size_t mm = static_cast<size_t>(m_) + 1;
        sband_.resize(6 * mm);
        int* bl = sband_.data();
        for (int i = 0; i <= m_; ++i) {
            bl[i] = jlo(i); bl[mm + i] = jhi(i); bl[2 * mm + i] = jlo0(i); bl[3 * mm + i] = jhi0(i);
            int lo, hi;
            if constexpr (AB == AlignBand::Full) { lo = 0; hi = n_; } else band_row_span(i, lo, hi);
            bl[4 * mm + i] = lo; bl[5 * mm + i] = hi;
        }
        buf.sexp.resize(mm);
        srs_.resize(2 * mm);
        J.m = m_; J.n = n_; J.nalpha = nalpha_;
        J.a = a_idx_.data(); J.b = b_idx_.data();
        J.P = tr_ ? buf.sqp.data() : build_profile(buf, w);
        J.local = (AM == AlignMode::Local) ? 1 : 0;
        J.full = (AB == AlignBand::Full) ? 1 : 0;
        J.bi = border_rows(); J.bj = border_cols();
        J.jlo = bl; J.jhi = bl + mm; J.jlo0 = bl + 2 * mm; J.jhi0 = bl + 3 * mm;
        J.slo = bl + 4 * mm; J.shi = bl + 5 * mm;
        J.stride = stride_;
        if constexpr (lin) J.FM = buf.F.data();
        else { J.FM = buf.FM.data(); J.FX = buf.FX.data(); J.FY = buf.FY.data(); }
        J.r0 = buf.sb0.data(); J.r1 = buf.sb1.data(); J.r2 = buf.sb2.data();
        J.r3 = buf.sb3.data(); J.r4 = buf.sb4.data(); J.r5 = buf.sb5.data();
        J.S = buf.sexp.data(); J.rowsum = srs_.data(); J.fmax = srs_.data() + mm;
        J.scnt = scnt_.data(); J.srow = srow_.data();
        soft_pair_fn fn = lin ? &nwgrad_soft_base::soft_pair_linear : &nwgrad_soft_base::soft_pair_affine;
        const int be = (backend_ == kBackendAuto) ? global_default_backend() : backend_;
        if (be >= 0) {
            const LevelKernels& K = level_kernels(be);
            soft_pair_fn f = lin ? K.soft_pair_linear : K.soft_pair_affine;
            if (f) fn = f;
        }
        fn(J);
        if (!J.ok) return false;
        log_z_ = J.log_z;
        sg_go_a_ = J.g_oa; sg_ge_a_ = J.g_ea; sg_go_b_ = J.g_ob; sg_ge_b_ = J.g_eb;
        if (tr_) srec_cnt_.swap(scnt_);   // per row: the record (srec_gap_ filled by the kernel)
        return true;
    }
};

// The inter-pair soft pass's weights (InterSoftJob es/oa/ea/ob/eb, linear) for `P` at
// temperature T: exp(score / T), exp(-(go+ge)/T), exp(-ge/T).  False when one overflows
// — the pairs then run their own path (which raises or falls back per soft_impl).
// Shared by SeqPairBatch and BatchAligner so both batch the same weights.
inline bool inter_soft_weights(const AlignParams& P, double T, bool lin,
                               std::vector<double>& es, InterSoftJob& sj) {
    constexpr double big = std::numeric_limits<double>::max();
    const double it = 1.0 / T;
    // (The whole matrix stack: a tracked lane reads its stacked row; K = 1 otherwise.)
    const size_t nn = static_cast<size_t>(P.matrix.size()) * P.matrix.size() *
                      static_cast<size_t>(P.num_sets());
    es.resize(nn);
    bool fin = true;
    for (size_t k = 0; k < nn; ++k) { es[k] = std::exp(P.matrix.data()[k] * it); fin &= es[k] <= big; }
    sj.oa = std::exp(-(P.gap_open_a + P.gap_extend_a) * it);
    sj.ea = std::exp(-P.gap_extend_a * it);
    sj.ob = std::exp(-(P.gap_open_b + P.gap_extend_b) * it);
    sj.eb = std::exp(-P.gap_extend_b * it);
    sj.linear = lin ? 1 : 0;
    sj.es = es.data(); sj.nalpha = P.matrix.size();
    return fin && sj.ea <= big && sj.eb <= big && (lin || (sj.oa <= big && sj.ob <= big));
}

// The Simd kernel's out-of-line definitions.  Included last, once Aligner is a
// complete type, so every instantiation sees both kernels.
#define NWGRAD_ALIGNER_HPP_INCLUDED 1
#include "aligner_simd.hpp"
