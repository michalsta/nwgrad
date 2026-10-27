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
#include <bit>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

// For the AVX-512 mask-op workaround in kernels_impl.inl (clang + libstdc++ std::simd
// bug).  Guarded identically to the workaround, and at global scope — intrinsics must
// not be pulled into a namespace.  See the long note at that site.
#if defined(__x86_64__) || defined(__i386__)
#  include <immintrin.h>   // lane_shift_in / store_codes intrinsic forms at every level
#endif
#if defined(__aarch64__)
#  include <arm_neon.h>    // carry_pin's AArch64 form (kernels_impl.inl)
#endif

#include "dp_buffer.hpp"
#include "simd_levels.hpp"

#define NWGRAD_LEVEL_NS NWGRAD_LVL_NAME
namespace NWGRAD_LVL_NAME {
#include "kernels_impl.inl"       // striped affine Full kernel (defines KW)
#include "kernels_pointers_impl.inl"  // same kernel, recording predecessor pointers
#include "kernels_linear_impl.inl"    // striped LINEAR Full: H table, or direction bytes
#include "hb_kernel_impl.inl"     // Hirschberg linear-space sweep (striped, no tables)
#include "row_kernel_impl.inl"    // row-wise banded leaf kernels (uses KW)
#include "inter_kernel_impl.inl"  // inter-pair affine Full fill (uses KW)
#include "inter_soft_impl.inl"    // inter-pair soft pass (uses the above's IW/ivd)
#include "soft_kernel_impl.inl"   // per-pair scaled forward-backward
#include "score_kernel_impl.inl"  // score only: rolling rows, no table
#include "grid_kernel_impl.inl"   // TracebackMode::Grid's forward pass (score recurrence + grid lines)
}
#undef NWGRAD_LEVEL_NS

namespace {
struct NwgradLevelRegistrar {
    // Pass bare function addresses (scalar `lea`s), never a locally-built struct: this
    // constructor runs at static init on every CPU, and in a -mavx512 TU a local struct
    // zero-init would emit an AVX512 broadcast that SIGILLs on non-AVX512 hardware.
    // register_level_* assemble the structs in baseline code (simd_levels.cpp).
    template <class T>
    static void typed() {
        register_level_typed<T>(NWGRAD_LVL_ENUM,
                                &NWGRAD_LVL_NAME::striped_full_entry<T>,      // affine Full: G+L
                                &NWGRAD_LVL_NAME::striped_full_ptr_entry<T>,  // TracebackMode::Pointers
                                NWGRAD_LVL_NAME::linear_kernel_pays<T>()       // linear Full: H table
                                    ? &NWGRAD_LVL_NAME::striped_linear_entry<T> : nullptr,
                                NWGRAD_LVL_NAME::linear_kernel_pays<T>()       // linear Full: direction bytes
                                    ? &NWGRAD_LVL_NAME::striped_linear_ptr_entry<T> : nullptr,
                                NWGRAD_LVL_NAME::linear_kernel_pays<T>()       // linear Hirschberg
                                    ? &NWGRAD_LVL_NAME::lhb_striped<T> : nullptr,
                                &NWGRAD_LVL_NAME::banded_row_global<T>,       // affine GuideBanded: whole-row
                                &NWGRAD_LVL_NAME::banded_row_local<T>,
                                &NWGRAD_LVL_NAME::hb_sweep_entry<T>,          // Hirschberg sweep
                                &NWGRAD_LVL_NAME::hb_sweep_pmax_entry<T>,     // prefix-max carry sweep
                                &NWGRAD_LVL_NAME::hb_base_entry<T>,           // Hirschberg base case
                                &NWGRAD_LVL_NAME::hb_scan_entry<T>,           // Hirschberg local endpoint scan
                                &NWGRAD_LVL_NAME::score_entry<T>,             // score only
                                &NWGRAD_LVL_NAME::inter_fill_entry<T>,        // inter-pair Full fill
                                &NWGRAD_LVL_NAME::inter_score_entry<T>,       // inter-pair score only
                                NWGRAD_LVL_NAME::IVT<T>::W);
        register_level_grid<T>(NWGRAD_LVL_ENUM, NWGRAD_LVL_NAME::grid_fwd_ptr<T>());
    }
    NwgradLevelRegistrar() {
        typed<double>();
        typed<float>();
        typed<int16_t>();
        typed<int32_t>();
        typed<int64_t>();
        register_level_common(NWGRAD_LVL_ENUM,
                              NWGRAD_LVL_NAME::ROW_BLOCK,
                              &NWGRAD_LVL_NAME::inter_soft_entry,     // inter-pair soft pass, double
                              &NWGRAD_LVL_NAME::soft_pair_linear,     // per-pair soft, linear
                              &NWGRAD_LVL_NAME::soft_pair_affine);    // per-pair soft, affine
    }
};
const NwgradLevelRegistrar nwgrad_level_registrar_instance;
}
