// The avx512 level's soft kernels, in their own TU (see level_common.inc, NWGRAD_LEVEL_SOFT_TU).
#define NWGRAD_LEVEL_SOFT_TU 1
#define NWGRAD_LVL_NAME lvl_avx512_soft
#define NWGRAD_LVL_ENUM SimdLevel::Avx512
#include "level_common.inc"
