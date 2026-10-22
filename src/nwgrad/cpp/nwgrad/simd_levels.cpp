// ── register_level_*: the one place the LevelKernels struct is assembled ───────
//
// Compiled at the x86-64 baseline (no -mavx2/-mavx512), unlike the level_*.cpp TUs.
// This is deliberate and load-bearing: the level TUs' static registrars call this at
// load on EVERY CPU, so the struct build + store must be baseline-legal code.  Were
// this inline in the header, it would be compiled into each level TU with that TU's
// -march and could emit an AVX512 broadcast (vpbroadcastd) for the zero-init — which
// SIGILLs at load on a CPU without AVX512.  See the note on the declaration in
// simd_levels.hpp.

#include "simd_levels.hpp"

template <class T>
void register_level_typed(SimdLevel l,
                          typename TypedKernels<T>::viterbi_fn viterbi,
                          typename TypedKernels<T>::viterbi_fn viterbi_ptr,
                          banded_row_fn_t<T> banded_row_global, banded_row_fn_t<T> banded_row_local,
                          typename TypedKernels<T>::hb_fn hb_sweep,
                          typename TypedKernels<T>::hb_fn hb_sweep_pmax,
                          typename TypedKernels<T>::hbbase_fn hb_base,
                          typename TypedKernels<T>::hbscan_fn hb_scan,
                          typename TypedKernels<T>::score_fn score,
                          typename TypedKernels<T>::inter_fn inter_fill,
                          typename TypedKernels<T>::inter_fn inter_score, int inter_w) {
    TypedKernels<T>& k = level_table()[(int)l].template of<T>();
    k.viterbi           = viterbi;
    k.viterbi_ptr       = viterbi_ptr;
    k.banded_row_global = banded_row_global;
    k.banded_row_local  = banded_row_local;
    k.hb_sweep          = hb_sweep;
    k.hb_sweep_pmax     = hb_sweep_pmax;
    k.hb_base           = hb_base;
    k.hb_scan           = hb_scan;
    k.score             = score;
    k.inter_fill        = inter_fill;
    k.inter_score       = inter_score;
    k.inter_w           = inter_w;
}

#define NWGRAD_INSTANTIATE_REGISTER(T) \
    template void register_level_typed<T>(SimdLevel, TypedKernels<T>::viterbi_fn, \
        TypedKernels<T>::viterbi_fn, banded_row_fn_t<T>, banded_row_fn_t<T>, \
        TypedKernels<T>::hb_fn, TypedKernels<T>::hb_fn, TypedKernels<T>::hbbase_fn, \
        TypedKernels<T>::hbscan_fn, TypedKernels<T>::score_fn, TypedKernels<T>::inter_fn, \
        TypedKernels<T>::inter_fn, int);
NWGRAD_INSTANTIATE_REGISTER(double)
NWGRAD_INSTANTIATE_REGISTER(float)
NWGRAD_INSTANTIATE_REGISTER(int16_t)
NWGRAD_INSTANTIATE_REGISTER(int32_t)
NWGRAD_INSTANTIATE_REGISTER(int64_t)
#undef NWGRAD_INSTANTIATE_REGISTER

void register_level_common(SimdLevel l, int row_block, inter_soft_fn inter_soft,
                           soft_pair_fn soft_pair_linear, soft_pair_fn soft_pair_affine) {
    LevelKernels& k = level_table()[(int)l];
    k.row_block        = row_block;
    k.inter_soft       = inter_soft;
    k.soft_pair_linear = soft_pair_linear;
    k.soft_pair_affine = soft_pair_affine;
}
