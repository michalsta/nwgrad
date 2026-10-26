// Grid kernel at the avx512 level.
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
namespace gk_avx512 {
#include "gk_impl.inl"
}
__attribute__((noinline)) GridLevel gk_level_avx512() {
    return {"avx512", (int)stdx::native_simd<float>::size(), (int)stdx::native_simd<double>::size(),
            {&gk_avx512::grid_fwd_e<float, false, false>, &gk_avx512::grid_fwd_e<float, false, true>,
             &gk_avx512::grid_fwd_e<float, true, false>, &gk_avx512::grid_fwd_e<float, true, true>,
             &gk_avx512::grid_fwd_e<float, true, true, true>, &gk_avx512::grid_fwd_e<float, true, false, true>},
            {&gk_avx512::grid_fwd_e<double, false, false>, &gk_avx512::grid_fwd_e<double, false, true>,
             &gk_avx512::grid_fwd_e<double, true, false>, &gk_avx512::grid_fwd_e<double, true, true>,
             &gk_avx512::grid_fwd_e<double, true, true, true>, &gk_avx512::grid_fwd_e<double, true, false, true>}};
}
