#pragma once

// ── Runtime SIMD-level dispatch ───────────────────────────────────────────────
//
// The affine Viterbi kernels are compiled once *per instruction-set level*, each in
// its own translation unit with that level's real `-march` flag, so std::simd picks
// the right register width (the #pragma-GCC-target trick cannot widen std::simd — its
// ABI is fixed at instantiation by __AVX2__ etc., so a genuine compile flag is
// required).  Each level TU registers its kernels here; the library resolves the
// active level once at load from the CPU, and every dispatched call reads a cached
// function pointer.
//
// THE ODR RULE the level TUs must obey: no std::simd type may cross this boundary.
// Every registered function takes only plain data (see ViterbiJob) and uses std::simd
// on locals only.  Distinct native width per level (sse2/neon=2, avx2=4, avx512=8)
// keeps the std::simd instantiations from COMDAT-folding across TUs.
//
// This is header-only-friendly: a consumer who just includes the headers and compiles
// at their own -march gets a single level (no dispatch); the runtime table is a
// feature of the built extension, where CMake compiles the level TUs with flags.

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

template <class T> struct DpBufferT;   // defined in dp_buffer.hpp; referenced by pointer
using DpBuffer = DpBufferT<double>;    // ViterbiJob carries the double buffer for now

// ── Levels ────────────────────────────────────────────────────────────────────
// Ordered weakest → strongest within an architecture.  Plain AVX is deliberately
// absent: it is a measured regression on Bulldozer/Piledriver and buys nothing over
// SSE2 elsewhere, so AVX-only CPUs run the Sse4 level (or Baseline, if they lack SSE4.2).
// Sse4 is the same width as Baseline (W=2); it exists for the SSE4.1 blends/ptest the
// compiler then emits in place of SSE2's and/andn/or sequences.
enum class SimdLevel : int {
#if defined(__aarch64__) || defined(_M_ARM64)
    Neon = 0,          // mandatory baseline on AArch64; the only level (no dispatch)
    Baseline = 0,
#else
    Baseline = 0,      // SSE2 (x86-64 baseline)
    Sse4 = 1,          // x86-64-v2 (SSE4.2 + POPCNT), still W=2
    Avx2 = 2,
    Avx512 = 3,
#endif
};

inline const char* level_name(SimdLevel l) {
    switch (l) {
#if defined(__aarch64__) || defined(_M_ARM64)
        case SimdLevel::Neon:     return "neon";
#else
        case SimdLevel::Baseline: return "sse2";
        case SimdLevel::Sse4:     return "sse4";
        case SimdLevel::Avx2:     return "avx2";
        case SimdLevel::Avx512:   return "avx512";
#endif
    }
    return "sse2";
}

// ── What the CPU can actually run ─────────────────────────────────────────────
inline std::vector<SimdLevel> detect_available_levels() {
    std::vector<SimdLevel> out;
#if defined(__aarch64__) || defined(_M_ARM64)
    out.push_back(SimdLevel::Neon);
#elif defined(__GNUC__) || defined(__clang__)
    __builtin_cpu_init();
    out.push_back(SimdLevel::Baseline);
    // x86-64-v2's vector set: the level TU is built with -msse4.2 -mpopcnt.
    if (__builtin_cpu_supports("sse4.2") && __builtin_cpu_supports("popcnt"))
        out.push_back(SimdLevel::Sse4);
    if (__builtin_cpu_supports("avx2"))
        out.push_back(SimdLevel::Avx2);
    // W=8 needs the 512-bit lanes plus the VL/DQ/BW subsets the kernels use.
    if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512vl") &&
        __builtin_cpu_supports("avx512dq") && __builtin_cpu_supports("avx512bw"))
        out.push_back(SimdLevel::Avx512);
#else
    out.push_back(SimdLevel::Baseline);  // unknown compiler: baseline only
#endif
    return out;
}

// ── Backend vocabulary ────────────────────────────────────────────────────────
//
// One unified vocabulary selects the Viterbi backend, per aligner and as a global
// default: "scalar_fallback" (the plain scalar kernel), "auto" (the strongest simd
// level the CPU can run), or a named simd level — "sse2"/"sse4"/"avx2"/"avx512" on x86,
// "neon" on AArch64.  Backends are encoded as ints so an aligner can store one:
//   kBackendAuto (-2)   defer to the global default (a per-aligner value only)
//   kBackendScalar (-1) the scalar fallback
//   0..                 a SimdLevel index — a simd kernel at that ISA level
constexpr int kBackendAuto   = -2;
constexpr int kBackendScalar = -1;

// "scalar_fallback" | "auto" | a simd-level name the CPU can run → backend int.  Throws
// on an unknown name, or a simd level this CPU cannot run (we never select above the
// hardware — that would SIGILL).
inline int parse_backend(const std::string& name) {
    if (name == "scalar_fallback") return kBackendScalar;
    if (name == "auto")            return kBackendAuto;
    for (SimdLevel l : detect_available_levels())
        if (name == level_name(l)) return (int)l;
    // Every level name on every platform, so a real level this CPU lacks is told
    // apart from a name that is not a backend at all (e.g. the pre-0.3 "simd").
    if (name == "sse2" || name == "sse4" || name == "avx2" || name == "avx512" || name == "neon")
        throw std::invalid_argument(
            "nwgrad: backend \"" + name + "\" is not available on this CPU — expected "
            "\"scalar_fallback\", \"auto\", or a simd level this CPU runs (see available_isa_levels())");
    throw std::invalid_argument(
        "nwgrad: unknown backend \"" + name + "\" — expected \"auto\", \"scalar_fallback\", "
        "\"sse2\", \"sse4\", \"avx2\", \"avx512\" or \"neon\"");
}

inline std::string backend_name(int backend) {
    if (backend == kBackendScalar) return "scalar_fallback";
    if (backend == kBackendAuto)   return "auto";
    return level_name((SimdLevel)backend);
}

// The strongest simd level the CPU can run — what "auto" resolves to.
inline int best_simd_backend() { return (int)detect_available_levels().back(); }

// ── The dispatch table ────────────────────────────────────────────────────────
//
// One entry per level.  A level's `viterbi` is the whole affine forward pass for that
// level: it switches internally on mode/band (Global/Local × Full/GuideBanded) and
// runs striped (Global+Full) or row-wise (the rest).  Plain-typed in and out.

// ── Tracks: per-row parameters (see aligner.hpp, "Tracks") ───────────────────
//
// A tracked problem reads each DP row's penalties and substitution row from here; an
// untracked one leaves `go_a` null, and every kernel then runs its untracked code (a
// template instantiation of its own, so the untracked fill is the code it always was).
// Row i (1..m) consumes a[i-1].  Arrays are indexed by the ABSOLUTE DP row, so a
// sub-block sweep (Hirschberg) reads the rows it covers.
//   rs        m: the stacked substitution row of DP row i at [i-1] — set*nalpha + a[i-1],
//             a row of `blk`, which then holds the K matrices back to back
//   ra/crow   the query profile's rows: ra[i-1] is DP row i's profile row, crow[r] the
//             stacked row profile row r holds (nrow of them, each distinct).  ra null:
//             no shared profile — a kernel builds each DP row's profile row from rs as it
//             reaches it (the caller's choice when nrow profile rows would not fit)
//   go_a/ge_a m+1: gaps in A within row i (row 0 included)
//   go_b/ge_b m+1: gaps in B into row i ([0] unused)
//   bx0       m+1: the Global column-0 border of row i (affine VX, linear H; [0] unused)
//   rgo_b/rge_b m+1: the reverse sweeps' gaps in B (Hirschberg; see aligner.hpp)
template <class T>
struct TrackJob {
    const uint32_t* rs = nullptr;
    const uint32_t* ra = nullptr;
    const uint32_t* crow = nullptr;
    int nrow = 0;
    const T* go_a = nullptr; const T* ge_a = nullptr;
    const T* go_b = nullptr; const T* ge_b = nullptr;
    const T* bx0 = nullptr;
    const T* rgo_b = nullptr; const T* rge_b = nullptr;
    bool on() const noexcept { return go_a != nullptr; }
};

// Everything a forward pass needs, and everything it produces — no std::simd, no
// Aligner: the only currency that crosses a level boundary.  Templated on the Viterbi
// precision T (double or float32): the substitution block, gap penalties and DP buffer
// are all T, the reported score too.
template <class T>
struct ViterbiJob {
    bool sat = false;   // nw_sat T past the quarter-range bound: the saturating kernel variant
    // problem (sequences already encoded to alphabet indices)
    const unsigned char* a; int m;
    const unsigned char* b; int n;
    const T* blk; int nalpha;               // substitution block, row-major nalpha×nalpha
    T go_a, ge_a, go_b, ge_b;               // gap penalties (in the Viterbi precision)
    int   align_mode;                       // 0 = Global, 1 = Local
    int   align_band;                       // 0 = Full,   1 = GuideBanded
    int   band;                             // half-width (band>0)
    const int* guide_j; int guide_len;      // guide path for GuideBanded (may be null)

    // scratch/output tables live in the caller's DpBufferT<T>
    DpBufferT<T>* buf;

    // results
    T       score;                          // in T (the aligner promotes it)
    int     best_i, best_j, best_tbl;       // best_tbl: 0=M 1=X 2=Y
    int     table_layout;                   // 0 = row-major VM/VX/VY, 1 = striped
    int     seg, width;                     // striping geometry (set when layout==1)
    std::size_t rstride;                    // striped row stride in T; 0 = (seg+1)*width
    TrackJob<T> tr;                         // tracked rows (tr.on()), else untracked
    // Pointers fills only.  In: 1 asks for the DENSE direction codes (dense_codes_row_bytes,
    // below: 2-bit codes, four segments' worth per byte -- 0.25 B/cell linear, three planes
    // 0.75 B/cell affine).  Out: 1 if the kernel wrote them, 0 if it wrote one byte a cell
    // (it clears the request it cannot honour: int16 lanes).
    int dense = 0;
};

// The dense direction codes' geometry, shared by the kernels and the walks.  Per DP row:
// W bytes of slot area (byte 0 = column 0's code in the one-byte format), then one group
// per four segments: `planes` x W bytes (plane p, lane l at group_base + p*W + l), each byte
// holding that lane's 2-bit codes for the group's segments, the FIRST in the highest bits:
// segment r of a group at shift 2*(3-r), a short last group's missing fields 0 (Horner,
// acc = 4*acc + k, and a short group scaled up at its end; or the codes picked already
// weighted, as the affine pass 1 does).  Linear: one plane (the move); affine: three (M, X, Y predecessor).
inline std::size_t dense_codes_row_bytes(int seg, int width, int planes) noexcept {
    return static_cast<std::size_t>(width) *
           (1 + static_cast<std::size_t>(planes) * static_cast<std::size_t>((seg + 3) / 4));
}
// The 2-bit code of striped slot (s, l) of a dense row: s = (j-1) % seg, l = (j-1) / seg.
inline unsigned dense_code_at(const unsigned char* row, int seg, int width, int planes,
                              int plane, int s, int l) noexcept {
    (void)seg;
    const int q = s >> 2, r = s & 3;
    const unsigned char byte = row[static_cast<std::size_t>(width) *
                                   (1 + static_cast<std::size_t>(planes) * q + plane) + l];
    return (byte >> (2 * (3 - r))) & 3u;
}


// ── Score only: the optimal alignment score, no table, no traceback ───────────
//
// The affine Full recurrence of ViterbiJob, bit-identical in every cell it computes,
// but keeping only rolling rows (O(n) memory) and reporting just the score: VM/VX/VY at
// (m, n) for Global, the maximum over every cell for Local.  Linear gaps run through it
// with go_a = go_b = 0, which reduces the affine recurrence to the linear one exactly
// (v - 0 is exact and rounding is monotone, so max(a-g, b-g) == max(a,b) - g bit for
// bit) — see Aligner::compute_score.  Scratch (profile, rolling rows) lives in buf.
template <class T>
struct ScoreJob {
    bool sat = false;   // nw_sat T past the quarter-range bound: the saturating kernel variant
    const unsigned char* a; int m;
    const unsigned char* b; int n;
    const T* blk; int nalpha;
    T go_a, ge_a, go_b, ge_b;
    int local;                              // 0 = Global, 1 = Local
    int variant;                            // 0 auto, 1 two-pass, 2 fused three-state (measuring)
    DpBufferT<T>* buf;
    T score;                                // out
    TrackJob<T> tr;                         // tracked rows (tr.on()), else untracked
};

// ── Grid forward pass (TracebackMode::Grid) ──────────────────────────────────
//
// The score kernel's D/X recurrence over the rectangle rows r0..r1 x columns c0..c1
// (DP coordinates), from its top row and left column (all three states), saving every
// R-th row (dense, rows[(t-1)*3*(w+1) + s*(w+1) + (j-c0)] for row r0 + t*R < r1) and q
// sets of segment-aligned columns (cols[((t*3 + s)*(h+1) + (i-r0))*W + lane], the column
// of set t / lane l in colid[t*W + l], -1 past the rectangle).  The kernel picks q from
// `spacing` (columns between lines) and its own width, resizes the outputs, and reports
// W and q.  Every value is the full fill's, bit for bit (opens >= 0, floating T).
template <class T>
struct GridFwdJob {
    const unsigned char* a;                 // full encoded sequences
    const unsigned char* b;
    const T* blk; int nalpha;
    T go_a, ge_a, go_b, ge_b;
    int r0, r1, c0, c1;
    const T* top;  std::size_t top_ss, top_es;    // top(s, j - c0) = top[s*top_ss + (j-c0)*top_es]
    const T* left; std::size_t left_ss, left_es;  // left(s, i - r0), likewise
    int R, spacing;
    typename DpBufferT<T>::GVec* rows;      // out (resized by the kernel)
    typename DpBufferT<T>::GVec* cols;
    std::vector<int>* colid;
    DpBufferT<T>* buf;                      // scratch: grows, gprof, gbuilt
    int W, q;                               // out
};

// ── Hirschberg linear-space sweep ─────────────────────────────────────────────
//
// One half-sweep of a divide-and-conquer block: fill H rows keeping only rolling rows,
// and report the FINAL row.  No table is written — that is the whole point of the mode —
// so unlike ViterbiJob there is no layout to report back; the output row is handed back
// de-striped and contiguous, because the join that consumes it scans by column.
//
// Forward and reverse are the same sweep with different walks, so they are one entry
// rather than two: the caller supplies start/step for both sequences.  Reverse is
// a_step = b_step = -1 with the starts at the far end.  That keeps the striped carry
// logic in exactly one place.
template <class T>
struct HbJob {
    bool sat = false;   // nw_sat T past the quarter-range bound: the saturating kernel variant
    const unsigned char* a;                 // full encoded sequences; the block is
    const unsigned char* b;                 // addressed by start/step below
    const T* blk; int nalpha;
    T go_a, ge_a, go_b, ge_b;
    int a_start, a_step;                    // row t (0..H-1) consumes a[a_start + t*a_step]
    int b_start, b_step;                    // column c (1..ncols) pairs b[b_start + (c-1)*b_step]
    int H;                                  // rows to sweep
    int ncols;                              // columns in this block
    int in_x;                               // 1 = the path enters already inside a gap-in-b run
    DpBufferT<T>* buf;                      // scratch (striped rows, profile, open vector)
    T* outM; T* outX; T* outY;              // final row, CONTIGUOUS, index 0..ncols (0 = border)
    TrackJob<T> tr;                         // tracked rows (tr.on()), else untracked
};


// ── Hirschberg LOCAL endpoint scan: a striped affine sweep that reports its argmax ─
//
// Local (Smith-Waterman) linear-space alignment cannot begin the divide-and-conquer
// until it knows the optimal cell's END and START; the recursion then aligns the
// substring between them GLOBALLY.  This job is that scan: the same striped rolling-row
// recurrence as HbJob, but it RETAINS NOTHING — it tracks the single best cell over the
// whole (sub-)block and reports it.  Two modes, chosen by `local`:
//   local = 1  clamped local borders (M=0) and the M-clamp max(.,0) — the FORWARD pass
//              that finds the end cell (ie, je) and score S.
//   local = 0  hb_sweep's GLOBAL borders and no clamp — the REVERSE pass, walked
//              backward, that computes Rev[i][j] = best global alignment of the
//              suffixes and finds the start (its argmax box globally aligns to S, so
//              the tie-break can never cost score).
// Forward vs reverse is the usual negated start/step; the reported (best_i, best_j) are
// block-1-based, and the caller maps them to absolute end/start cells.
template <class T>
struct HbScanJob {
    bool sat = false;   // nw_sat T past the quarter-range bound: the saturating kernel variant
    const unsigned char* a;
    const unsigned char* b;
    const T* blk; int nalpha;
    T go_a, ge_a, go_b, ge_b;
    int a_start, a_step;                    // row t (0..H-1) consumes a[a_start + t*a_step]
    int b_start, b_step;                    // column c (1..ncols) pairs b[b_start + (c-1)*b_step]
    int H;                                  // rows to sweep
    int ncols;                              // columns in this block
    int local;                              // 1 = clamped local (forward end), 0 = global (reverse start)
    int pmax;                               // 1 = the prefix-max VY carry (HirschbergPmax)
    DpBufferT<T>* buf;                      // scratch (striped rows, profile, open vector)
    // outputs: the best cell and its block-1-based coordinates.  Default 0 at (0,0), so
    // an all-nonpositive local block reports the empty alignment.
    T best; int best_i; int best_j;
    TrackJob<T> tr;                         // tracked rows (tr.on()), else untracked
};


// ── Hirschberg base case: striped fill that RECORDS direction bytes ───────────
//
// The recursion bottoms out here.  Unlike HbJob (which keeps only rolling rows and
// returns one row), this keeps the whole block's direction tables — 3 bytes/cell in
// buf.hbD, striped — so the aligner can walk the path back.  It is the striped Pointers
// kernel restricted to a sub-rectangle with an affine boundary seed (in_x), which is
// why the base case is now vectorized instead of the scalar fill it began as.  The
// walk-back stays on the aligner side (O(H+ncols), negligible) and reads hbD striped.
template <class T>
struct HbBaseJob {
    bool sat = false;   // nw_sat T past the quarter-range bound: the saturating kernel variant
    const unsigned char* a;
    const unsigned char* b;
    const T* blk; int nalpha;
    T go_a, ge_a, go_b, ge_b;
    int a_start, a_step;
    int b_start, b_step;
    int H;                                   // rows (0..H), so H+1 rows of direction bytes
    int ncols;                               // columns (0..ncols)
    int in_x;                                // 1 = path enters already inside a gap-in-b run
    DpBufferT<T>* buf;                       // writes buf.hbD; scratch in hprof/hf*/hov/hs*
    // outputs: the final cell (H, ncols) scores, so the aligner picks the walk-back start
    T fM; T fX; T fY;
    int seg;                                 // striping segment count (walk-back needs it)
    int width;                               // vector lane count W (walk-back needs it)
    TrackJob<T> tr;                         // tracked rows (tr.on()), else untracked
};


// Inter-pair affine Full fill, double: W pairs at once, one per vector lane (W = the
// level's native double count, reported in LevelKernels::inter_w).  All W pairs share
// sequence B's length n; A's lengths may differ (a lane's rows past its own m are
// computed and ignored).  Writes VM/VX/VY INTERLEAVED — cell (i,j) of lane l at
// ((i*(n+1)+j)*W + l), rows 0..max m — with exactly the operations, in exactly the
// order, of the row-wise kernel (row_kernel_impl.inl), so each lane's table is bit-
// identical to that pair's own fill and the aligner's traceback/gradient read it
// unchanged through cell_index().  The substitution score is selected per lane by
// blending on the bits of the B residue: nalpha <= 8 only (the caller falls back
// otherwise).  Local: also returns each lane's best cell, chosen as the per-pair fill
// chooses it (first row, then first column, reaching the maximum; 0 if none > 0).
// Templated on the precision T (double or float32); at float32 W is twice the double
// lane count (LevelKernels::inter_w_f) and every value, border and comparison is in T,
// bit-identical to the scalar float fill.
template <class T>
struct InterJobT {
    const unsigned char* const* a;   // W pointers to A's codes
    const int* m;                    // W lengths of A
    const unsigned char* const* b;   // W pointers to B's codes (all of length n)
    int n;
    int M;                           // max m over the lanes (rows filled)
    const T* blk; int nalpha;        // substitution block, row-major nalpha x nalpha
    T go_a, ge_a, go_b, ge_b;
    int align_mode;                  // 0 = Global, 1 = Local
    int linear;                      // 1 = linear gaps: H in VM only (go_*, VX, VY unused)
    T* VM; T* VX; T* VY;             // (M+1)*(n+1)*W each
    T* best; int* best_i; int* best_j;   // W each, Local only
    // GuideBanded (nullptr = Full): each lane around its own guide, its tables bit-
    // identical to that pair's own banded fill (Aligner::banded_lane_rows fills these).
    // blo/bhi: (M+1)*W, row i lane l computes columns [blo, bhi] (empty past its m);
    // ulo/uhi: M+1, the columns >= 1 the row computes — the union of the lanes'
    // initialised spans; bri/brj: W, Global border_rows / border_cols.
    const int* blo = nullptr; const int* bhi = nullptr;
    const int* ulo = nullptr; const int* uhi = nullptr;
    const int* bri = nullptr; const int* brj = nullptr;
    // Ragged B (nullptr = every lane's B has length n): W lengths <= n.  A lane's columns
    // past its own length are computed from a padded residue and ignored — the DP flows
    // right and down only, so they cannot reach its cells — and excluded from its Local
    // best; its pair adopts the tables at the group's stride (n+1).
    const int* nb = nullptr;
    // Tracked (nullptr = untracked): W TrackJobs, one per lane, each lane's rows read from
    // its own (rs into `blk`, which is then the stacked block; the four gap arrays; bx0).
    // The go_*/ge_* scalars above are then unused.  ra/crow are not read: no profile here.
    const TrackJob<T>* tr = nullptr;
};
using InterJob = InterJobT<double>;

// Inter-pair SOFT pass (scaled forward-backward with the gradient fused), double: the
// same W-lane grouping as InterJob, Full band, affine or linear gaps, nalpha <= 8.  Per lane it is
// soft_pair_affine (soft_kernel_impl.inl) — tolerance-equal, not bit-equal (the soft path is not
// bit-exact).  The caller supplies exp'd weights (temperature already applied) and
// scratch; per lane it receives log Z (of params/T, NOT yet multiplied by T), the
// expected match counts (nalpha x nalpha) and the four gap gradient fields
// (go_a, ge_a, go_b, ge_b; already negative), and ok = 0 where the lane's lost-mass
// bound or overflow check failed (the caller then runs that pair on its own path).
// Body in inter_soft_impl.inl.
template <class F>
struct InterSoftJobT {
    const unsigned char* const* a; const int* m; const unsigned char* const* b;
    int n, M;
    const int* nb = nullptr;           // ragged B: W lengths <= n (see InterJobT::nb)
    const F* es; int nalpha;           // exp(score / T), row-major nalpha x nalpha
    F oa, ea, ob, eb;                  // exp(-(go+ge)/T), exp(-ge/T), per side
    int align_mode;                    // 0 = Global, 1 = Local
    int linear;                        // 1 = linear gaps (oa/ob unused, gap opens 0)
    F* scratch;                        // inter_soft_scratch(n, M, W) values of F, 64-byte aligned
    int* iscratch;                     // (M+1)*W ints
    double* logz; double* counts; double* gaps; int* ok;   // W, W*nalpha^2, W*4, W
    int* gpost = nullptr;              // soft_guide="posterior": (M+1)*W, as SoftPairJob
    // Tracked (rs != nullptr): per lane, rs[l][i-1] is DP row i's row of `es` (then the
    // exp'd matrix STACK) and rw[l][4i ..] row i's weights (oa, ea, ob, eb, as
    // SoftPairJob::rw).  Outputs per lane and row instead of per lane: counts at
    // (l*M + i-1)*nalpha (row i's match posteriors by B letter; W*M*nalpha) and gaps at
    // (l*(M+1) + i)*4 (W*(M+1)*4).
    const uint32_t* const* rs = nullptr;
    const F* const* rw = nullptr;
};
using InterSoftJob  = InterSoftJobT<double>;
using InterSoftJobF = InterSoftJobT<float>;
using inter_soft_fn   = void (*)(InterSoftJob&);
using inter_soft_fn_f = void (*)(InterSoftJobF&);

// Per-pair scaled forward-backward (Aligner's SoftImpl::Scaled), double: one pair, Full
// or GuideBanded, any alphabet.  Bodies in soft_kernel_impl.inl — compiled once per
// level (LevelKernels::soft_pair_*) and once at baseline inside aligner.hpp.  The Aligner
// fills it: sequences, exp'd profile P (nalpha x (n+2), temperature applied), gap
// weights, per-row band ranges (jlo/jhi: the computed cells j >= 1; jlo0/jhi0: the
// band; slo/shi: the initialized span), tables and scratch.  Outputs: log_z (of
// params/T), expected counts accumulated into scnt (zeroed by the caller), the gap
// gradient fields (negative), ok = 0 when out of range.
template <class F>
struct SoftPairJobT {
    int m, n, nalpha;
    const unsigned char* a; const unsigned char* b;
    const F* P;
    F oa, ea, ob, eb;               // linear: ea, eb only
    int local, full, bi, bj;        // bi/bj: Global border rows/cols
    const int *jlo, *jhi, *jlo0, *jhi0, *slo, *shi;   // m+1 each
    size_t stride;                  // n+1
    F *FM, *FX, *FY;                // forward tables (linear: FM only)
    F *r0, *r1, *r2, *r3, *r4, *r5; // n+2 each
    int* S; F* rowsum; F* fmax;     // m+1 each (fmax: forward row maxima)
    double* scnt; F* srow;          // nalpha^2 (double: the counts accumulate across rows), nalpha
    double log_z, g_oa, g_ea, g_ob, g_eb;
    int ok;
    // soft_guide="posterior" (nullptr = not wanted): m+1 entries, row i's column of
    // greatest EXIT mass (first on ties) — the posterior that the path leaves row i at
    // column j: the cell's posterior minus the mass moving on left into (i, j+1) (the Y
    // posterior there; linear: F(i,j)·ea·B(i,j+1)).  That is what a Viterbi guide
    // records (the last column of row i on the path), so along a gap run only its last
    // cell scores — a plain posterior argmax picks any cell of the run, on rounding.
    int* gpost = nullptr;
    // Tracked (rw != nullptr; see aligner.hpp, "Tracks"): row i's weights rw[4i ..] =
    // oa, ea (gaps in A within row i), ob, eb (gaps in B into row i; [0] unused); P has a
    // row per DP row (row i-1 for DP row i) and scnt likewise (m x nalpha: row i's match
    // posteriors by B letter); grow[4i ..] receives row i's gap gradient (g_oa, g_ea,
    // g_ob, g_eb, negative).  The scalar weights and g_* are then unused.
    const F* rw = nullptr;
    double* grow = nullptr;
};
using SoftPairJob  = SoftPairJobT<double>;
using SoftPairJobF = SoftPairJobT<float>;
using soft_pair_fn   = void (*)(SoftPairJob&);
using soft_pair_fn_f = void (*)(SoftPairJobF&);

// The soft (scaled forward-backward) path's floating type for a Viterbi type T: float32
// pairs run it in float32 (twice the lanes, half the tables), everything else in double.
template <class T>
using soft_float_t = std::conditional_t<std::is_same_v<T, float>, float, double>;
// Per soft type: the lazy-rescale window [1/win, win] a row max may roam before its
// row is rescaled (powers of two, exact), and the lost-mass tolerance (see
// Aligner::run_fwdbwd).  float32: exponents reach 2^127, so the window is 2^32 (a
// row grows by at most the largest weight per step, which itself is below 2^128);
// the tolerance 2^-20 sits above the bound any normal pair accumulates (~1e-34 with
// FLT_MIN terms) and below float32's own rounding, 2^-24 per operation.
template <class F> struct SoftRange;
template <> struct SoftRange<double> { static constexpr double win = 0x1p256;  static constexpr double tol = 0x1p-45; };
template <> struct SoftRange<float>  { static constexpr float  win = 0x1p32f;  static constexpr double tol = 0x1p-20; };

// ── Runtime overrides of the fill gates (nwgrad._tuning) ─────────────────────────
// The gates below (and Aligner::linear_striped_min_n) are thresholds measured on one
// host per ISA, so they are runtime-overridable for tuning sweeps: each reads
// gate_value(g, its compiled default).  Overrides are process-wide; setting one bumps
// gate_epoch(), which makes a batch's cached inter-pair plan regroup on its next call.
// kGateUnset: no override, the compiled default; kGateOff: the cap never rejects
// (LinearStripedMinN takes 0 for that instead: the striped kernel from any length).
// Only baseline TUs call these (the level TUs include this header but no gate code),
// so the one static table is never compiled at a level's -march.
enum class Gate : int {
    InterGroupBytes, InterSoftBytes, InterScoreGatherCells,
    InterLinearIntGatherCells, InterLinearF32GatherCells, InterLinearF64GatherCells,
    InterGlobalLinearIntCells, InterGlobalLinearF32Cells, LinearStripedMinN, Count_
};
inline constexpr const char* kGateNames[] = {
    "inter_group_bytes", "inter_soft_bytes", "inter_score_gather_cells",
    "inter_linear_int_gather_cells", "inter_linear_f32_gather_cells",
    "inter_linear_f64_gather_cells", "inter_global_linear_int_cells",
    "inter_global_linear_f32_cells", "linear_striped_min_n"};
static_assert(sizeof(kGateNames) / sizeof(kGateNames[0]) == static_cast<size_t>(Gate::Count_));
inline constexpr int64_t kGateUnset = -1;
inline constexpr int64_t kGateOff = INT64_MAX;
struct GateTable {
    std::atomic<int64_t> v[static_cast<size_t>(Gate::Count_)];
    std::atomic<unsigned> epoch{0};
    GateTable() { for (auto& x : v) x.store(kGateUnset, std::memory_order_relaxed); }
};
inline GateTable& gate_table() { static GateTable t; return t; }
inline int64_t gate_override(Gate g) {
    return gate_table().v[static_cast<size_t>(g)].load(std::memory_order_relaxed);
}
inline void set_gate_override(Gate g, int64_t v) {
    gate_table().v[static_cast<size_t>(g)].store(v < 0 ? kGateUnset : v, std::memory_order_relaxed);
    gate_table().epoch.fetch_add(1, std::memory_order_relaxed);
}
inline unsigned gate_epoch() { return gate_table().epoch.load(std::memory_order_relaxed); }
inline size_t gate_value(Gate g, size_t def) {
    const int64_t v = gate_override(g);
    return v < 0 ? def : static_cast<size_t>(v);
}
// Whether a pair of lengths (la, lb) may join an inter-pair group: the group's
// interleaved tables must stay in L2.  Measured (nighthaven, i5-12500, Golden Cove: 1.25
// MiB L2 per core; protein pairs, 12 threads): affine Viterbi wins up to 100-130
// residues (tables ~1.3 MB per group) and loses 2x at 130-160 (~1.6 MB); linear, one
// table, still wins at 160-200 and loses at 150-300 — the same byte crossover.  Past
// it W pairs' tables stream from L3/DRAM while the per-pair fills keep 3 B/cell
// (pointers) and win by up to 11x.  Viterbi: 3 (affine) / 1 (linear) T-tables of W
// lanes, capped at 1.25 MiB.  The soft pass crosses over lower: 30-60 residues (~0.3
// MB per group) win 0.51-0.70x at 12 threads, 60-120 (~1.3 MB) lose up to 1.95x — so
// its 5 (affine: 3 forward tables, the per-row weights, rows) / 2 (linear) double
// tables of Ws lanes are capped at 512 KiB.
inline constexpr size_t kInterGroupBytes = size_t(5) << 18;   // 1.25 MiB
inline constexpr size_t kInterSoftBytes  = size_t(1) << 19;   // 512 KiB
// Score only, alphabets over 8 (the gather): the inter-pair score kernel loads one scalar
// per lane-cell where the striped per-pair score kernel loads 1/W of a profile row per
// cell, so the shared score pays only for short pairs — measured on nighthaven (AVX2,
// BLOSUM62, 1 thread, inter / striped): L60 0.46-0.85, L100 0.80-1.14, L150 1.13-1.38,
// L200 1.25-1.57, for int16, int32, float32 and double alike (the crossover ~L100 is
// independent of the type and the lane count).  Hard fills are not gated by this: there
// the shared fill replaces the own LINEAR fill, which is slower at every length.
inline constexpr size_t kInterScoreGatherCells = size_t(8192);   // (M + 1) * (n + 1) of the group
// Hard fills, linear, integer T, alphabets over 8 (batch_engine.hpp, inter_eligible_): ~L120.
inline constexpr size_t kInterLinearIntGatherCells = size_t(15000);
// The same for FLOATING T on AArch64 (NEON, the only simd level there): the inter-pair
// gather loses to the leveled linear kernel from ~L120 (float32) / ~L90 (double) on the
// M1 (inter / own, Local protein, hard: float32 L100 0.87, L150 1.14, L200 1.18; double
// L60 0.95, L100 1.02, L150 1.11), while on AVX2 it still wins through L200 (0.63-0.84),
// so x86 keeps no cap.  0 = no cap.
#if defined(__aarch64__) || defined(_M_ARM64)
inline constexpr size_t kInterLinearF32GatherCells = size_t(15000);
inline constexpr size_t kInterLinearF64GatherCells = size_t(8000);
#else
inline constexpr size_t kInterLinearF32GatherCells = 0;
inline constexpr size_t kInterLinearF64GatherCells = 0;
#endif
// Global linear, alphabets over 8 (admitted to the shared fill 2026-10-23): it wins L20-80
// for every type, but at 12 threads (nighthaven AVX2, inter / own, hard) loses for the
// integers at L80-100 (1.04-1.20; 1 thread 0.82-1.01) and for float32 at L160-200
// (1.02-1.11; 1 thread 0.80-0.87) — the threaded case is a batch's default, so those
// bands are capped: ~L75 and ~L140.  double wins through L200 (0.77-0.93).
inline constexpr size_t kInterGlobalLinearIntCells = size_t(6000);
inline constexpr size_t kInterGlobalLinearF32Cells = size_t(20000);
// The gather cap for a linear hard pair of T over an alphabet over 8 letters, or
// SIZE_MAX (none): the smallest of the caps that apply (the mode's, the type's,
// AArch64's for floats).  A compiled default of 0 means no cap; an override is literal.
template <class T, bool Global>
inline size_t inter_linear_gather_cap() {
    size_t cap = SIZE_MAX;
    auto take = [&cap](Gate g, size_t def) {
        const size_t c = gate_override(g) < 0 ? (def > 0 ? def : SIZE_MAX) : gate_value(g, def);
        if (c < cap) cap = c;
    };
    if constexpr (std::is_integral_v<T>) take(Gate::InterLinearIntGatherCells, kInterLinearIntGatherCells);
    else if constexpr (std::is_same_v<T, float>) take(Gate::InterLinearF32GatherCells, kInterLinearF32GatherCells);
    else take(Gate::InterLinearF64GatherCells, kInterLinearF64GatherCells);
    if constexpr (Global) {
        if constexpr (std::is_integral_v<T>) take(Gate::InterGlobalLinearIntCells, kInterGlobalLinearIntCells);
        else if constexpr (std::is_same_v<T, float>) take(Gate::InterGlobalLinearF32Cells, kInterGlobalLinearF32Cells);
    }
    return cap;
}
inline bool inter_score_ok(size_t nalpha, int M, int n) {
    return nalpha <= 8 ||
           static_cast<size_t>(M + 1) * static_cast<size_t>(n + 1) <=
               gate_value(Gate::InterScoreGatherCells, kInterScoreGatherCells);
}
inline bool inter_pair_fits(size_t la, size_t lb, bool affine, bool soft, int W,
                            size_t tsize, int Ws) {
    const size_t cells = (la + 1) * (lb + 1);
    if (cells * static_cast<size_t>(W) * (affine ? 3 : 1) * tsize >
        gate_value(Gate::InterGroupBytes, kInterGroupBytes)) return false;
    return !soft || cells * static_cast<size_t>(Ws) * (affine ? 5 : 2) * sizeof(double) <=
                        gate_value(Gate::InterSoftBytes, kInterSoftBytes);
}

// Doubles of scratch the inter-pair soft pass needs for W lanes (the caller allocates it,
// 64-byte aligned): the forward tables, the per-row weights, 7 rolling rows, row sums and
// maxima, and 11 more rows for the letter masks (<= 8), the two ragged column masks and
// the gathered match row — kept here and NOT in thread_local vectors, which made every
// soft call abort on macOS (gcc's emulated TLS; see inter_soft_impl.inl).
inline size_t inter_soft_scratch(int n, int M, int W) {
    return (3 * static_cast<size_t>(M + 1) * (n + 1) + static_cast<size_t>(M + 1) * (n + 2) +
            18 * static_cast<size_t>(n + 2) + 2 * static_cast<size_t>(M + 1)) * W;
}

// Whole-row banded kernel for the GuideBanded path.  viterbi_affine_simd (in
// aligner_simd.hpp) owns the banded indexing and hands this one row's worth of
// contiguous slices; it runs the whole interleaved block loop (carry-free VM/VX, serial
// VY carry) plus the Local row max in a single call, compiled once per level with that
// level's real -march.  Plain-typed, so it obeys the ODR rule (no std::simd crosses the
// boundary).  Returns max3 for the Local entry; Global's return is ignored.  Bodies in
// row_kernel_impl.inl.
template <class T>
using banded_row_fn_t = T (*)(T* __restrict, T* __restrict, T* __restrict,
                              const T* __restrict, const T* __restrict,
                              const T* __restrict, const T* __restrict,
                              int, int, int, T, T, T, T, bool /* sat */);

// ── The kernels of one scalar type T (double, float, int16_t, int32_t, int64_t) ──
//
// Every hard-path kernel is instantiated per T by each level TU and registered in the
// level's TypedKernels<T> (LevelKernels::of<T>()).  A null entry means the level has no
// kernel of that kind for that T, and the aligner runs its scalar fill instead — which
// is what a header-only build (no level TU linked) sees for every entry.
// ── Linear Hirschberg, one striped kernel for its four O(m*n) loops ─────────────
// (kernels_linear_impl.inl, lhb_striped.)  The aligner hands over everything that
// differs between the forward / reverse sweeps, the base case and the two Local
// endpoint scans — the B codes and the rows in SWEEP order, each row's extends, and the
// row-0 and column-0 borders, computed in scalar exactly as the scalar loops compute
// them — so the kernel holds no border or direction logic of its own and its rows are
// bit for bit the scalar loops' rows.  Rows r = 1..h; columns c = 1..n.
template <class T>
struct LhbJob {
    bool sat = false;   // nw_sat T past the quarter-range bound: the saturating kernel variant
    const unsigned char* b; int n;          // B codes in sweep order (column c: b[c-1])
    const T* blk; int nalpha;               // substitution stack
    const uint32_t* srow; int h;            // row r: stacked substitution row srow[r-1]
    const T* ga; const T* gb;               // row r: extend of a gap in A / into the row (index r-1)
    const T* row0;                          // n+1 values: row 0
    const T* col0;                          // h+1 values: column 0 (col0[0] == row0[0])
    int by_symbol;                          // srow < nalpha for every row: one profile per symbol
    int clamp;                              // Local: max(., 0) after the diagonal/up maximum
    int argmax;                             // 0 none; 1 columns 1..n; 2 column 0 too (first)
    T best; int bi, bj;                     // in: the seed; out: the best cell (strict >,
                                            //   topmost row, then leftmost column)
    unsigned char* codes;                   // non-null: (h+1) x rowsz direction codes, striped
    T* out;                                 // non-null: row h, de-striped, n+1 values
    int seg, width;                         // out: striping geometry (for codes)
    DpBufferT<T>* buf;
};

template <class T>
struct TypedKernels {
    using viterbi_fn = void (*)(ViterbiJob<T>&);
    using hb_fn      = void (*)(HbJob<T>&);
    using hbbase_fn  = void (*)(HbBaseJob<T>&);
    using hbscan_fn  = void (*)(HbScanJob<T>&);
    using score_fn   = void (*)(ScoreJob<T>&);
    using inter_fn   = void (*)(InterJobT<T>&);
    viterbi_fn         viterbi = nullptr;            // striped affine Full (Global + Local)
    viterbi_fn         viterbi_ptr = nullptr;        // TracebackMode::Pointers
    viterbi_fn         linear = nullptr;             // striped LINEAR Full, H table (kernels_linear_impl.inl)
    viterbi_fn         linear_ptr = nullptr;         // ... recording direction bytes
    using lhb_fn = void (*)(LhbJob<T>&);
    lhb_fn             lhb = nullptr;                // linear Hirschberg sweeps / base / scans
    banded_row_fn_t<T> banded_row_global = nullptr;  // row-wise banded whole-row kernels
    banded_row_fn_t<T> banded_row_local = nullptr;
    hb_fn              hb_sweep = nullptr;           // Hirschberg linear-space sweep
    // The prefix-max sibling of the sweep (TracebackMode::HirschbergPmax): same job, same
    // output, a closed-form VY carry instead of the serial chain + lazy-F.  Bit-identical
    // to hb_sweep only for an integer T; its own slot, never overwriting the sweep's.
    hb_fn              hb_sweep_pmax = nullptr;
    hbbase_fn          hb_base = nullptr;            // Hirschberg base case (records directions)
    hbscan_fn          hb_scan = nullptr;            // Hirschberg local endpoint scan
    score_fn           score = nullptr;              // score only (rolling rows)
    inter_fn           inter_fill = nullptr;         // inter-pair Full fill, W pairs per call
    inter_fn           inter_score = nullptr;        // inter-pair score only (Full)
    int                inter_w = 0;                  // the inter-pair lane count (pairs per call)
    using grid_fn = void (*)(GridFwdJob<T>&);
    grid_fn            grid_fwd = nullptr;           // TracebackMode::Grid's forward pass (floating T)
};

struct LevelKernels {
    TypedKernels<double>  d;
    TypedKernels<float>   f;
    TypedKernels<int16_t> i16;
    TypedKernels<int32_t> i32;
    TypedKernels<int64_t> i64;
    int           row_block = 0;                // columns per interleaved block (per-µarch)
    inter_soft_fn inter_soft = nullptr;         // inter-pair soft pass, double (W = d.inter_w)
    soft_pair_fn  soft_pair_linear = nullptr;   // per-pair scaled forward-backward ↓
    soft_pair_fn  soft_pair_affine = nullptr;
    inter_soft_fn_f inter_soft_f = nullptr;       // the same in float32 (W = f.inter_w)
    soft_pair_fn_f  soft_pair_linear_f = nullptr;
    soft_pair_fn_f  soft_pair_affine_f = nullptr;
    // The soft kernels of soft type F.
    template <class F> auto inter_soft_of() const noexcept {
        if constexpr (std::is_same_v<F, float>) return inter_soft_f; else return inter_soft; }
    template <class F> auto soft_pair_of(bool linear) const noexcept {
        if constexpr (std::is_same_v<F, float>) return linear ? soft_pair_linear_f : soft_pair_affine_f;
        else                                    return linear ? soft_pair_linear : soft_pair_affine; }
    // The kernels of scalar type T.
    template <class T> const TypedKernels<T>& of() const noexcept {
        if constexpr (std::is_same_v<T, double>)        return d;
        else if constexpr (std::is_same_v<T, float>)    return f;
        else if constexpr (std::is_same_v<T, int16_t>)  return i16;
        else if constexpr (std::is_same_v<T, int32_t>)  return i32;
        else { static_assert(std::is_same_v<T, int64_t>, "unsupported scalar type"); return i64; }
    }
    template <class T> TypedKernels<T>& of() noexcept {
        return const_cast<TypedKernels<T>&>(static_cast<const LevelKernels*>(this)->of<T>());
    }
};

// One slot per possible level; index by (int)SimdLevel.  Populated by the level TUs'
// static registrars at load, before any dispatch.
inline LevelKernels* level_table() {
    static LevelKernels tbl[4];
    return tbl;
}

// Register a level's kernels.  Deliberately takes the pointers *individually*, not a
// TypedKernels / LevelKernels by value, and is defined out-of-line in simd_levels.cpp —
// which is compiled at the x86-64 baseline.  Both facts are load-bearing: the level TUs
// are compiled with -mavx512 etc. and their registrars run at static init on EVERY CPU,
// so any code they emit must be baseline-legal.  If they built a struct locally, GCC
// would zero-init it with an AVX512 GPR-broadcast (vpbroadcastd) and the program would
// SIGILL at load on a non-AVX512 box.  Passing bare pointers keeps the registrar to
// scalar `lea`s; the struct is assembled and stored here, in baseline code.  One typed
// call per scalar type (register_level_typed<T>, explicitly instantiated for the five)
// and one for the type-free entries (register_level_common).
// Out-of-line is not enough under LTO: the link-time inliner may put a baseline callee
// into an AVX-512 caller (a callee with fewer target features is inlinable), and gcc
// -flto did exactly that, SIGILLing at load (vinserti64x2) on an AVX2 CPU.  So the
// registrars are never inlined or IPA-specialized into their callers.
#if defined(__GNUC__) && !defined(__clang__)
#define NWGRAD_BASELINE_ONLY __attribute__((noipa))
#elif defined(__clang__)
#define NWGRAD_BASELINE_ONLY __attribute__((noinline))
#else
#define NWGRAD_BASELINE_ONLY
#endif
template <class T>
NWGRAD_BASELINE_ONLY
void register_level_typed(SimdLevel l,
                          typename TypedKernels<T>::viterbi_fn viterbi,
                          typename TypedKernels<T>::viterbi_fn viterbi_ptr,
                          typename TypedKernels<T>::viterbi_fn linear,
                          typename TypedKernels<T>::viterbi_fn linear_ptr,
                          void (*lhb)(LhbJob<T>&),
                          banded_row_fn_t<T> banded_row_global, banded_row_fn_t<T> banded_row_local,
                          typename TypedKernels<T>::hb_fn hb_sweep,
                          typename TypedKernels<T>::hb_fn hb_sweep_pmax,
                          typename TypedKernels<T>::hbbase_fn hb_base,
                          typename TypedKernels<T>::hbscan_fn hb_scan,
                          typename TypedKernels<T>::score_fn score,
                          typename TypedKernels<T>::inter_fn inter_fill,
                          typename TypedKernels<T>::inter_fn inter_score, int inter_w);

// TracebackMode::Grid's forward pass (null for a type without one: it runs as Pointers).
template <class T>
NWGRAD_BASELINE_ONLY
void register_level_grid(SimdLevel l, typename TypedKernels<T>::grid_fn grid_fwd);

NWGRAD_BASELINE_ONLY
void register_level_common(SimdLevel l, int row_block, inter_soft_fn inter_soft,
                           soft_pair_fn soft_pair_linear, soft_pair_fn soft_pair_affine,
                           inter_soft_fn_f inter_soft_f, soft_pair_fn_f soft_pair_linear_f,
                           soft_pair_fn_f soft_pair_affine_f);

// ── The global default backend ────────────────────────────────────────────────
//
// The backend used by any aligner whose backend is "auto".  Resolved lazily: NWGRAD_ISA
// if set and valid (it may name "scalar_fallback" / "auto" / a level), otherwise "auto"
// = the strongest simd level.  set_isa_level() changes it at runtime.  Changing it is
// NOT thread-safe — do it before dispatching work.
constexpr int kBackendUnset = -1000;

inline std::atomic<int>& default_backend_slot() {
    static std::atomic<int> slot{kBackendUnset};
    return slot;
}

inline int global_default_backend() {
    int v = default_backend_slot().load(std::memory_order_relaxed);
    if (v == kBackendUnset) {
        int b = best_simd_backend();   // "auto"
        if (const char* want = std::getenv("NWGRAD_ISA")) {
            try {
                int parsed = parse_backend(want);
                b = (parsed == kBackendAuto) ? best_simd_backend() : parsed;
            } catch (const std::exception&) {
                // Unknown/unavailable NWGRAD_ISA: keep auto (best simd), never throw at init.
            }
        }
        default_backend_slot().store(b, std::memory_order_relaxed);
        v = b;
    }
    return v;   // kBackendScalar or a level index, never Auto/Unset
}

// Set the global default backend.  Accepts "scalar_fallback" | "auto" | a level the CPU
// runs; throws on a level it cannot.  Forcing *down* (e.g. "sse2" on an AVX2 box) is how
// a level's bit-exactness is tested on capable hardware.
inline void set_isa_level(const std::string& name) {
    int b = parse_backend(name);
    if (b == kBackendAuto) b = best_simd_backend();
    default_backend_slot().store(b, std::memory_order_relaxed);
}

inline std::string get_isa_level() { return backend_name(global_default_backend()); }

// Every backend selectable on this CPU: the scalar fallback plus each simd level it can
// run (weakest first).  "auto" is always settable but omitted here — it is a meta-value
// that resolves to the strongest of these.
inline std::vector<std::string> available_isa_levels() {
    std::vector<std::string> out;
    out.push_back("scalar_fallback");
    for (SimdLevel l : detect_available_levels()) out.push_back(level_name(l));
    return out;
}

// The kernels for a specific simd level (a 0.. SimdLevel index).  The DP resolves its
// per-aligner backend to a level and calls this; if that level's TU was not linked (a
// header-only single-level build), the slot holds nullptrs and the DP falls back to the
// in-header scalar path.
inline const LevelKernels& level_kernels(int level) {
    return level_table()[level];
}
