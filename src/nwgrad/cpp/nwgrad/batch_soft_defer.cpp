// BatchEngine's soft full calls and the guide deferral (batch_soft_defer.inl), compiled
// for the extension in a unit of their own: see the note there and at the end of
// BatchEngine.  Every engine the extension instantiates (SeqPairBatchT's six for each of
// the five scalar types), and the bindings' soft entry points (seq_pair_batch.hpp).
#define NWGRAD_SOFT_DEFER_UNIT 1
#include "batch_engine.hpp"
#include "batch_soft_defer.inl"
#include "seq_pair_batch.hpp"

#define NWGRAD_SOFT_DEFER_INST(T, G, A)                                                          \
    template double BatchEngine<T, GapModel::G, AlignMode::A>::soft_score_and_grad(bool);              \
    template double BatchEngine<T, GapModel::G, AlignMode::A>::soft_banded_grad(int, bool);            \
    template void BatchEngine<T, GapModel::G, AlignMode::A>::soft_score_and_grad_one(size_t, bool);    \
    template void BatchEngine<T, GapModel::G, AlignMode::A>::soft_banded_one(size_t, int, bool);       \
    template std::vector<int> BatchEngine<T, GapModel::G, AlignMode::A>::soft_guide_j(size_t);         \
    template size_t BatchEngine<T, GapModel::G, AlignMode::A>::guides_pending() const noexcept;
#define NWGRAD_SOFT_DEFER_INST_T(T)                                                   \
    NWGRAD_SOFT_DEFER_INST(T, Linear, Global) NWGRAD_SOFT_DEFER_INST(T, Linear, Local) \
    NWGRAD_SOFT_DEFER_INST(T, Affine, Global) NWGRAD_SOFT_DEFER_INST(T, Affine, Local) \
    NWGRAD_SOFT_DEFER_INST(T, Linear, Glocal) NWGRAD_SOFT_DEFER_INST(T, Affine, Glocal) \
    template size_t SeqPairBatchT<T>::guides_pending() const noexcept;                  \
    template double nwgrad_soft_score_and_grad<T>(SeqPairBatchT<T>&, bool);              \
    template double nwgrad_soft_banded_grad<T>(SeqPairBatchT<T>&, int, bool);            \
    template double nwgrad_soft_align<T>(SeqPairT<T>&, bool);                            \
    template double nwgrad_soft_banded_align<T>(SeqPairT<T>&, int, bool);                \
    template std::vector<int> nwgrad_soft_guide_j<T>(SeqPairT<T>&);
NWGRAD_SOFT_DEFER_INST_T(float)
NWGRAD_SOFT_DEFER_INST_T(double)
NWGRAD_SOFT_DEFER_INST_T(int16_t)
NWGRAD_SOFT_DEFER_INST_T(int32_t)
NWGRAD_SOFT_DEFER_INST_T(int64_t)
