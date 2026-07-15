// ── Shared body of a level translation unit ───────────────────────────────────
//
// Each level_<name>.cpp defines NWGRAD_LVL_NAME (the namespace) and NWGRAD_LVL_ENUM
// (the SimdLevel), then #include's this.  CMake compiles each with that level's real
// -march flag, so stdx::native_simd here has the level's native width — the whole
// reason for the per-TU scheme (a #pragma cannot widen std::simd; only a compile flag
// defining __AVX2__ etc. can).
//
// The registrar runs at load and only takes function addresses — it executes no
// wide-ISA instructions, so linking (say) the avx512 TU into a binary that also runs
// on a non-avx512 CPU is safe; the kernel bodies run only when the CPU selects the
// level (verified).

#include <experimental/simd>
namespace stdx = std::experimental;

#include <algorithm>
#include <array>
#include <limits>

#include "dp_buffer.hpp"
#include "simd_levels.hpp"

#define NWGRAD_LEVEL_NS NWGRAD_LVL_NAME
namespace NWGRAD_LVL_NAME {
#include "kernels_impl.inl"
}
#undef NWGRAD_LEVEL_NS

namespace {
struct NwgradLevelRegistrar {
    NwgradLevelRegistrar() {
        LevelKernels k;
        k.viterbi = &NWGRAD_LVL_NAME::striped_full_entry;   // affine Full: Global + Local
        register_level(NWGRAD_LVL_ENUM, k);
    }
};
const NwgradLevelRegistrar nwgrad_level_registrar_instance;
}
