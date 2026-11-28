// BatchEngine's soft full calls and the guide deferral (batch_soft_defer.inl), compiled
// for the extension in a unit of their own: see the note there and at the end of
// BatchEngine.  The soft entry points for float and double only (soft batches are
// floating-point; an integer one cannot be constructed): instantiating them for the
// integer engines made this unit reference integer engine helpers that main's link drops
// as unused, and the engine units' copies of those then came back into their blocks.
// The test counter for every engine the extension instantiates (SeqPairBatchT's six for
// each of the five scalar types).
#define NWGRAD_SOFT_DEFER_UNIT 1
#include "batch_engine.hpp"
#include "batch_soft_defer.inl"
#include "seq_pair_batch.hpp"

#define NWGRAD_SOFT_DEFER_MODES(X, T)                                       \
    X(T, Linear, Global) X(T, Linear, Local) X(T, Affine, Global)           \
    X(T, Affine, Local) X(T, Linear, Glocal) X(T, Affine, Glocal)
#define NWGRAD_SOFT_DEFER_COUNT(T, G, A) \
    template size_t BatchEngine<T, GapModel::G, AlignMode::A>::guides_pending() const noexcept;
#define NWGRAD_SOFT_DEFER_ENTRY(T, G, A)                                                             \
    template double BatchEngine<T, GapModel::G, AlignMode::A>::soft_score_and_grad(bool);              \
    template double BatchEngine<T, GapModel::G, AlignMode::A>::soft_banded_grad(int, bool);            \
    template void BatchEngine<T, GapModel::G, AlignMode::A>::soft_score_and_grad_one(size_t, bool);    \
    template void BatchEngine<T, GapModel::G, AlignMode::A>::soft_banded_one(size_t, int, bool);       \
    template std::vector<int> BatchEngine<T, GapModel::G, AlignMode::A>::soft_guide_j(size_t);
#define NWGRAD_SOFT_DEFER_COUNT_T(T)                     \
    NWGRAD_SOFT_DEFER_MODES(NWGRAD_SOFT_DEFER_COUNT, T)   \
    template size_t SeqPairBatchT<T>::guides_pending() const noexcept;
#define NWGRAD_SOFT_DEFER_ENTRY_T(T)                                                  \
    NWGRAD_SOFT_DEFER_MODES(NWGRAD_SOFT_DEFER_ENTRY, T)                                \
    template double nwgrad_soft_score_and_grad<T>(SeqPairBatchT<T>&, bool);              \
    template double nwgrad_soft_banded_grad<T>(SeqPairBatchT<T>&, int, bool);            \
    template double nwgrad_soft_align<T>(SeqPairT<T>&, bool);                            \
    template double nwgrad_soft_banded_align<T>(SeqPairT<T>&, int, bool);                \
    template std::vector<int> nwgrad_soft_guide_j<T>(SeqPairT<T>&);
NWGRAD_SOFT_DEFER_COUNT_T(float)
NWGRAD_SOFT_DEFER_COUNT_T(double)
NWGRAD_SOFT_DEFER_COUNT_T(int16_t)
NWGRAD_SOFT_DEFER_COUNT_T(int32_t)
NWGRAD_SOFT_DEFER_COUNT_T(int64_t)
NWGRAD_SOFT_DEFER_ENTRY_T(float)
NWGRAD_SOFT_DEFER_ENTRY_T(double)
