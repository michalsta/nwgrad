// The avx2 level's inter-pair hard kernels, in their own TU (see level_common.inc, NWGRAD_LEVEL_INTER_TU).
#define NWGRAD_LEVEL_INTER_TU 1
#define NWGRAD_LVL_NAME lvl_avx2_inter
#define NWGRAD_LVL_ENUM SimdLevel::Avx2
#include "level_common.inc"
