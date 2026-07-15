// AVX-512 (F/DQ/VL/BW), native double width = 8.  CMake compiles with those -m flags.
#define NWGRAD_LVL_NAME lvl_avx512
#define NWGRAD_LVL_ENUM SimdLevel::Avx512
#include "level_common.inc"
