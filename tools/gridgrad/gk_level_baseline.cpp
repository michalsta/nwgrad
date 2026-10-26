// Grid kernel at the baseline level.
#include <experimental/simd>
namespace stdx = std::experimental;
#include <algorithm>
#include <limits>
#include <type_traits>
#include <vector>
#include <cstdio>
#if defined(__x86_64__)
#  include <immintrin.h>
#endif
#include "dp_buffer.hpp"
#include "gk.hpp"
namespace gk_baseline {
#include "gk_impl.inl"
}
__attribute__((noinline)) GridLevel gk_level_baseline() {
    return {"baseline", (int)stdx::native_simd<float>::size(), (int)stdx::native_simd<double>::size(),
            {&gk_baseline::grid_fwd_e<float, false, false>, &gk_baseline::grid_fwd_e<float, false, true>,
             &gk_baseline::grid_fwd_e<float, true, false>, &gk_baseline::grid_fwd_e<float, true, true>,
             &gk_baseline::grid_fwd_e<float, true, true, true>, &gk_baseline::grid_fwd_e<float, true, false, true>},
            {&gk_baseline::grid_fwd_e<double, false, false>, &gk_baseline::grid_fwd_e<double, false, true>,
             &gk_baseline::grid_fwd_e<double, true, false>, &gk_baseline::grid_fwd_e<double, true, true>,
             &gk_baseline::grid_fwd_e<double, true, true, true>, &gk_baseline::grid_fwd_e<double, true, false, true>}};
}
