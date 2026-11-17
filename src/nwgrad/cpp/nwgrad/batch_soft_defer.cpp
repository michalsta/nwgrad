// BatchEngine's soft full calls (batch_soft_defer.inl), compiled for the extension in a
// unit of their own: see the note there.  Every engine the bindings instantiate
// (SeqPairBatchT's six for each of the five scalar types).
#include "batch_engine.hpp"
#include "batch_soft_defer.inl"

#define NWGRAD_SOFT_DEFER_INST(T, G, A)                                              \
    template double BatchEngine<T, GapModel::G, AlignMode::A>::soft_score_and_grad_(bool); \
    template void BatchEngine<T, GapModel::G, AlignMode::A>::soft_one_(size_t, bool);      \
    template void BatchEngine<T, GapModel::G, AlignMode::A>::soft_banded_prep_();          \
    template void BatchEngine<T, GapModel::G, AlignMode::A>::resolve_caller_(size_t);
#define NWGRAD_SOFT_DEFER_INST_T(T)                                                   \
    NWGRAD_SOFT_DEFER_INST(T, Linear, Global) NWGRAD_SOFT_DEFER_INST(T, Linear, Local) \
    NWGRAD_SOFT_DEFER_INST(T, Affine, Global) NWGRAD_SOFT_DEFER_INST(T, Affine, Local) \
    NWGRAD_SOFT_DEFER_INST(T, Linear, Glocal) NWGRAD_SOFT_DEFER_INST(T, Affine, Glocal)
NWGRAD_SOFT_DEFER_INST_T(float)
NWGRAD_SOFT_DEFER_INST_T(double)
NWGRAD_SOFT_DEFER_INST_T(int16_t)
NWGRAD_SOFT_DEFER_INST_T(int32_t)
NWGRAD_SOFT_DEFER_INST_T(int64_t)
