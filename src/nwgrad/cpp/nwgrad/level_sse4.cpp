// SSE4.2 (x86-64-v2 vector set), native double width = 2.  CMake compiles with -msse4.2.
// Same width as the sse2 level; what it buys is codegen: a stdx::where blend is one
// blendvpd/blendvps instead of SSE2's and/andn/or triple, any_of is ptest, and 64-bit
// lane compares are pcmpeqq.  For CPUs with SSE4 but no AVX2 (Nehalem..Ivy Bridge,
// Bulldozer/Piledriver, Jaguar, Atom, AVX-less Pentium/Celeron, VMs hiding AVX).
#define NWGRAD_LVL_NAME lvl_sse4
#define NWGRAD_LVL_ENUM SimdLevel::Sse4
#include "level_common.inc"
