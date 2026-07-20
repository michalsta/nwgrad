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

template <class T> struct DpBufferT;   // defined in dp_buffer.hpp; referenced by pointer
using DpBuffer = DpBufferT<double>;    // ViterbiJob carries the double buffer for now

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
        case SimdLevel::Baseline: return "sse2";
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
// level the CPU can run), or a named simd level — "sse2"/"avx2"/"avx512" on x86,
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
    throw std::invalid_argument(
        "nwgrad: backend \"" + name + "\" is not available on this CPU — expected "
        "\"scalar_fallback\", \"auto\", or a simd level this CPU runs (see available_isa_levels())");
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

// Everything a forward pass needs, and everything it produces — no std::simd, no
// Aligner: the only currency that crosses a level boundary.  Templated on the Viterbi
// precision T (double or float32): the substitution block, gap penalties and DP buffer
// are all T; the reported score stays double (a T score promotes to double exactly).
template <class T>
struct ViterbiJob {
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
    double  score;                          // reported as double regardless of T
    int     best_i, best_j, best_tbl;       // best_tbl: 0=M 1=X 2=Y
    int     table_layout;                   // 0 = row-major VM/VX/VY, 1 = striped
    int     seg, width;                     // striping geometry (set when layout==1)
};

// One striped-Full entry per precision.  Both are registered by each level TU.
using viterbi_fn   = void (*)(ViterbiJob<double>&);
using viterbi_fn_f = void (*)(ViterbiJob<float>&);

// Whole-row banded kernel for the GuideBanded path.  viterbi_affine_simd (in
// aligner_simd.hpp) owns the banded indexing and hands this one row's worth of
// contiguous slices; it runs the whole interleaved block loop (carry-free VM/VX, serial
// VY carry) plus the Local row max in a single call, compiled once per level with that
// level's real -march.  Plain-typed, so it obeys the ODR rule (no std::simd crosses the
// boundary).  Returns max3 for the Local entry; Global's return is ignored.  Bodies in
// row_kernel_impl.inl.
using banded_row_fn = double (*)(double* __restrict, double* __restrict, double* __restrict,
                                 const double* __restrict, const double* __restrict,
                                 const double* __restrict, const double* __restrict,
                                 int, int, int, double, double, double, double);

struct LevelKernels {
    viterbi_fn    viterbi = nullptr;            // striped affine Full, double (Global + Local)
    viterbi_fn_f  viterbi_f = nullptr;          // striped affine Full, float32
    banded_row_fn banded_row_global = nullptr;  // row-wise banded whole-row kernels ↓ (double)
    banded_row_fn banded_row_local  = nullptr;
    int           row_block = 0;                // columns per interleaved block (per-µarch)
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
void register_level(SimdLevel l, viterbi_fn viterbi, viterbi_fn_f viterbi_f,
                    banded_row_fn banded_row_global, banded_row_fn banded_row_local,
                    int row_block);

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
