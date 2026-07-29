// ── register_level: the one place the LevelKernels struct is assembled ─────────
//
// Compiled at the x86-64 baseline (no -mavx2/-mavx512), unlike the level_*.cpp TUs.
// This is deliberate and load-bearing: the level TUs' static registrars call this at
// load on EVERY CPU, so the 56-byte struct build + store must be baseline-legal code.
// Were this inline in the header, it would be compiled into each level TU with that
// TU's -march and could emit an AVX512 broadcast (vpbroadcastd) for the zero-init —
// which SIGILLs at load on a CPU without AVX512.  See the note on the declaration in
// simd_levels.hpp.

#include "simd_levels.hpp"

void register_level(SimdLevel l, viterbi_fn viterbi, viterbi_fn_f viterbi_f,
                    viterbi_fn viterbi_ptr, viterbi_fn_f viterbi_ptr_f,
                    banded_row_fn banded_row_global, banded_row_fn banded_row_local,
                    hb_fn hb_sweep, hb_fn_f hb_sweep_f,
                    hb_fn hb_sweep_pmax, hb_fn_f hb_sweep_pmax_f,
                    hbbase_fn hb_base, hbbase_fn_f hb_base_f,
                    hbscan_fn hb_scan, hbscan_fn_f hb_scan_f,
                    int row_block) {
    LevelKernels k;
    k.viterbi          = viterbi;
    k.viterbi_f        = viterbi_f;
    k.viterbi_ptr        = viterbi_ptr;
    k.viterbi_ptr_f      = viterbi_ptr_f;
    k.banded_row_global = banded_row_global;
    k.banded_row_local  = banded_row_local;
    k.hb_sweep          = hb_sweep;
    k.hb_sweep_f        = hb_sweep_f;
    k.hb_sweep_pmax     = hb_sweep_pmax;
    k.hb_sweep_pmax_f   = hb_sweep_pmax_f;
    k.hb_base           = hb_base;
    k.hb_base_f         = hb_base_f;
    k.hb_scan           = hb_scan;
    k.hb_scan_f         = hb_scan_f;
    k.row_block         = row_block;
    level_table()[(int)l] = k;
}
