// The engine's semi-global runtime-ends code (AlignMode::Glocal with free ends other than
// glocal's; the Aligner::sg_*_ entries): every SGR instantiation of the Aligner's fills,
// score paths and end searches is made here (the walks take every end set themselves), so
// the bindings' unit (py_exports.cpp, which defines NWGRAD_SG_OUT_OF_LINE and only calls
// these entries) instantiates none of it.  Why its own unit: py_exports sits at gcc's
// --param inline-unit-growth limit, so code instantiated there moves gcc's inlining of the
// Global / Local / glocal engine; see Aligner::sg_run_viterbi_.  Full band only (a guide
// band with these ends is refused).
#include "aligner.hpp"

#define NWGRAD_SG_INSTANTIATE(GM, T)                                                              \
    template void Aligner<GM, AlignMode::Glocal, AlignBand::Full, T>::sg_run_viterbi_(DpBufferT<T>&); \
    template double Aligner<GM, AlignMode::Glocal, AlignBand::Full, T>::sg_compute_score_(DpBufferT<T>&);

#define NWGRAD_SG_INSTANTIATE_T(T)                 \
    NWGRAD_SG_INSTANTIATE(GapModel::Linear, T)     \
    NWGRAD_SG_INSTANTIATE(GapModel::Affine, T)

NWGRAD_SG_INSTANTIATE_T(double)
NWGRAD_SG_INSTANTIATE_T(float)
NWGRAD_SG_INSTANTIATE_T(std::int16_t)
NWGRAD_SG_INSTANTIATE_T(std::int32_t)
NWGRAD_SG_INSTANTIATE_T(std::int64_t)
