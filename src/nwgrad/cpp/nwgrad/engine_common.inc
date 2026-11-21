// ── Shared body of an engine translation unit (engine_<type>.cpp) ─────────────────
//
// Each engine_<type>.cpp defines NWGRAD_ENGINE_T and #include's this.  It compiles, for
// that scalar type, everything of the engine the Python extension calls: SeqPairBatchT /
// SeqPairT (seq_pair_batch.hpp; through them every BatchEngine, Aligner, traceback and
// scalar fill) and the single-pair convenience functions (pair_entry.hpp).  The bindings
// (py_exports.cpp) only call these, so nothing they contain enters this unit's inlining
// decisions; and one unit per type keeps a change to one type's code from moving the
// inlining of the others.  See the note above SeqPairBatchT.

#include "pair_entry.hpp"
#include "seq_pair_batch.hpp"

template class SeqPairBatchT<NWGRAD_ENGINE_T>;
template class SeqPairT<NWGRAD_ENGINE_T>;
template double pair_score<NWGRAD_ENGINE_T>(GapModel, AlignMode, const PairProblem&);
template std::pair<double, AlignParams> pair_hard_grad<NWGRAD_ENGINE_T>(GapModel, AlignMode, const PairProblem&);
#if NWGRAD_ENGINE_FLOATING
template std::pair<double, AlignParams> pair_soft_grad<NWGRAD_ENGINE_T>(GapModel, AlignMode, const PairProblem&,
                                                                       SoftImpl, double);
#endif
