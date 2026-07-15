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

// For the AVX-512 mask-op workaround in kernels_impl.inl (clang + libstdc++ std::simd
// bug).  Guarded identically to the workaround, and at global scope — intrinsics must
// not be pulled into a namespace.  See the long note at that site.
#if defined(__clang__) && defined(__AVX512F__) && !defined(NWGRAD_STD_SIMD_AVX512_MASK_OK)
#  include <immintrin.h>
#endif

#include "dp_buffer.hpp"
#include "simd_levels.hpp"

#define NWGRAD_LEVEL_NS NWGRAD_LVL_NAME
namespace NWGRAD_LVL_NAME {
#include "kernels_impl.inl"       // striped affine Full kernel (defines KW)
#include "row_kernel_impl.inl"    // row-wise banded leaf kernels (uses KW)
}
#undef NWGRAD_LEVEL_NS

namespace {
struct NwgradLevelRegistrar {
    NwgradLevelRegistrar() {
        // Pass bare function addresses (scalar `lea`s), never a locally-built
        // LevelKernels: this constructor runs at static init on every CPU, and in a
        // -mavx512 TU a local struct zero-init would emit an AVX512 broadcast that
        // SIGILLs on non-AVX512 hardware.  register_level assembles the struct in
        // baseline code (simd_levels.cpp).
        register_level(NWGRAD_LVL_ENUM,
                       &NWGRAD_LVL_NAME::striped_full_entry,  // affine Full: Global + Local
                       &NWGRAD_LVL_NAME::row_mx_global,       // affine GuideBanded: row-wise ↓
                       &NWGRAD_LVL_NAME::row_mx_local,
                       &NWGRAD_LVL_NAME::row_y,
                       &NWGRAD_LVL_NAME::row_m3,
                       NWGRAD_LVL_NAME::ROW_BLOCK);
    }
};
const NwgradLevelRegistrar nwgrad_level_registrar_instance;
}
