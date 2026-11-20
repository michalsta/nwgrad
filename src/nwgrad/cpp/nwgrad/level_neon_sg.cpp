// The neon level's runtime-ends (semi-global) linear fills, in their own TU (see
// level_common.inc, NWGRAD_LEVEL_SG_TU).
#define NWGRAD_LEVEL_SG_TU 1
#define NWGRAD_LVL_NAME lvl_neon_sg
#define NWGRAD_LVL_ENUM SimdLevel::Neon
#include "level_common.inc"
