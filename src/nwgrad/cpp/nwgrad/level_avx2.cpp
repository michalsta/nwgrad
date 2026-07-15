// AVX2 + FMA, native double width = 4.  CMake compiles with -mavx2 -mfma.
#define NWGRAD_LVL_NAME lvl_avx2
#define NWGRAD_LVL_ENUM SimdLevel::Avx2
#include "level_common.inc"
