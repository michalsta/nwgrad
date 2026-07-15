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

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

struct DpBuffer;  // defined in aligner.hpp; only referenced by pointer here

// ── Levels ────────────────────────────────────────────────────────────────────
// Ordered weakest → strongest within an architecture.  Plain AVX is deliberately
// absent: it is a measured regression on Bulldozer/Piledriver and buys nothing over
// SSE2 elsewhere, so AVX-only CPUs run the Baseline level.
enum class SimdLevel : int {
#if defined(__aarch64__) || defined(_M_ARM64)
    Neon = 0,          // mandatory baseline on AArch64; the only level (no dispatch)
    Baseline = 0,
#else
    Baseline = 0,      // SSE2 (x86-64 baseline)
    Avx2 = 1,
    Avx512 = 2,
#endif
};

inline const char* level_name(SimdLevel l) {
    switch (l) {
#if defined(__aarch64__) || defined(_M_ARM64)
        case SimdLevel::Neon:     return "neon";
#else
        case SimdLevel::Baseline: return "baseline";
        case SimdLevel::Avx2:     return "avx2";
        case SimdLevel::Avx512:   return "avx512";
#endif
    }
    return "baseline";
}

// ── What the CPU can actually run ─────────────────────────────────────────────
inline std::vector<SimdLevel> detect_available_levels() {
    std::vector<SimdLevel> out;
#if defined(__aarch64__) || defined(_M_ARM64)
    out.push_back(SimdLevel::Neon);
#elif defined(__GNUC__) || defined(__clang__)
    __builtin_cpu_init();
    out.push_back(SimdLevel::Baseline);
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

// The strongest available level, honouring a NWGRAD_ISA override at first use.
inline SimdLevel default_level() {
    const auto avail = detect_available_levels();
    SimdLevel best = avail.back();
    if (const char* want = std::getenv("NWGRAD_ISA")) {
        for (SimdLevel l : avail)
            if (std::strcmp(level_name(l), want) == 0) best = l;
        // an unsupported request silently keeps `best` — never select above the CPU.
    }
    return best;
}

// ── The dispatch table ────────────────────────────────────────────────────────
//
// One entry per level.  A level's `viterbi` is the whole affine forward pass for that
// level: it switches internally on mode/band (Global/Local × Full/GuideBanded) and
// runs striped (Global+Full) or row-wise (the rest).  Plain-typed in and out.

// Everything a forward pass needs, and everything it produces — no std::simd, no
// Aligner, no templates: the only currency that crosses a level boundary.
struct ViterbiJob {
    // problem (sequences already encoded to alphabet indices)
    const unsigned char* a; int m;
    const unsigned char* b; int n;
    const double* blk; int nalpha;          // substitution block, row-major nalpha×nalpha
    double go_a, ge_a, go_b, ge_b;          // gap penalties
    int   align_mode;                       // 0 = Global, 1 = Local
    int   align_band;                       // 0 = Full,   1 = GuideBanded
    int   band;                             // half-width (band>0)
    const int* guide_j; int guide_len;      // guide path for GuideBanded (may be null)

    // scratch/output tables live in the caller's DpBuffer
    DpBuffer* buf;

    // results
    double  score;
    int     best_i, best_j, best_tbl;       // best_tbl: 0=M 1=X 2=Y
    int     table_layout;                   // 0 = row-major VM/VX/VY, 1 = striped
    int     seg, width;                     // striping geometry (set when layout==1)
};

using viterbi_fn = void (*)(ViterbiJob&);

// Row-wise leaf kernels for the GuideBanded path.  viterbi_affine_simd (in
// aligner_simd.hpp) owns the banded indexing and hands each of these one row's
// worth of contiguous slices; they are the vectorized inner loops, compiled once
// per level with that level's real -march.  Plain-typed, so they obey the ODR rule
// (no std::simd crosses the boundary).  See aligner_simd.hpp for the bodies.
using row_mx_fn = void (*)(double* __restrict, double* __restrict,
                           const double* __restrict, const double* __restrict,
                           const double* __restrict, const double* __restrict,
                           int, int, double, double);
using row_y_fn  = void (*)(double* __restrict, const double* __restrict,
                           const double* __restrict, int, int, double, double);
using row_m3_fn = double (*)(const double* __restrict, const double* __restrict,
                             const double* __restrict, int, int);

struct LevelKernels {
    viterbi_fn viterbi = nullptr;        // striped affine Full (Global + Local)
    row_mx_fn  row_mx_global = nullptr;  // row-wise banded leaf kernels ↓
    row_mx_fn  row_mx_local  = nullptr;
    row_y_fn   row_y  = nullptr;
    row_m3_fn  row_m3 = nullptr;
    int        row_block = 0;            // columns per interleaved block (per-µarch)
};

// One slot per possible level; index by (int)SimdLevel.  Populated by the level TUs'
// static registrars at load, before any dispatch.
inline LevelKernels* level_table() {
    static LevelKernels tbl[4];
    return tbl;
}

// Register a level's kernels.  Deliberately takes the pointers *individually*, not a
// LevelKernels by value, and is defined out-of-line in simd_levels.cpp — which is
// compiled at the x86-64 baseline.  Both facts are load-bearing: the level TUs are
// compiled with -mavx512 etc. and their registrars run at static init on EVERY CPU,
// so any code they emit must be baseline-legal.  If they built a LevelKernels locally,
// GCC would zero-init its 56 bytes with an AVX512 GPR-broadcast (vpbroadcastd) and the
// program would SIGILL at load on a non-AVX512 box.  Passing bare pointers keeps the
// registrar to scalar `lea`s; the struct is assembled and stored here, in baseline code.
void register_level(SimdLevel l, viterbi_fn viterbi,
                    row_mx_fn row_mx_global, row_mx_fn row_mx_local,
                    row_y_fn row_y, row_m3_fn row_m3, int row_block);

// ── The active selection ──────────────────────────────────────────────────────
//
// Resolved once (lazily) to default_level(); overridable via set_isa_level for tests.
// Changing it is NOT thread-safe — do it before dispatching work.

inline std::atomic<int>& active_level_slot() {
    static std::atomic<int> slot{-1};
    return slot;
}

inline SimdLevel active_level() {
    int v = active_level_slot().load(std::memory_order_relaxed);
    if (v < 0) {
        v = (int)default_level();
        active_level_slot().store(v, std::memory_order_relaxed);
    }
    return (SimdLevel)v;
}

// Force a level.  Throws if the CPU cannot run it — we never select above hardware
// (that would SIGILL); forcing *down* (e.g. baseline on an AVX2 box) is how per-level
// bit-exactness is tested on capable hardware.
inline void set_isa_level(const std::string& name) {
    for (SimdLevel l : detect_available_levels())
        if (name == level_name(l)) {
            active_level_slot().store((int)l, std::memory_order_relaxed);
            return;
        }
    throw std::invalid_argument(
        "nwgrad: ISA level \"" + name + "\" is not available on this CPU");
}

inline std::string get_isa_level() { return level_name(active_level()); }

inline std::vector<std::string> available_isa_levels() {
    std::vector<std::string> out;
    for (SimdLevel l : detect_available_levels()) out.push_back(level_name(l));
    return out;
}

// The active level's kernels.  If the level TU for the active level was not linked
// (e.g. a header-only single-level build), its slot may hold nullptrs — callers fall
// back to the in-header scalar/templated path.
inline const LevelKernels& active_kernels() {
    return level_table()[(int)active_level()];
}
