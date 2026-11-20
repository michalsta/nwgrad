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
                          typename TypedKernels<T>::viterbi_fn linear,
                          typename TypedKernels<T>::viterbi_fn linear_ptr,
                          void (*lhb)(LhbJob<T>&),
                          banded_row_fn_t<T> banded_row_global, banded_row_fn_t<T> banded_row_local,
                          typename TypedKernels<T>::hb_fn hb_sweep,
                          typename TypedKernels<T>::hb_fn hb_sweep_pmax,
                          typename TypedKernels<T>::hbbase_fn hb_base,
                          typename TypedKernels<T>::hbscan_fn hb_scan,
                          typename TypedKernels<T>::score_fn score) {
    TypedKernels<T>& k = level_table()[(int)l].template of<T>();
    k.viterbi           = viterbi;
    k.viterbi_ptr       = viterbi_ptr;
    k.linear            = linear;
    k.linear_ptr        = linear_ptr;
    k.lhb               = lhb;
    k.banded_row_global = banded_row_global;
    k.banded_row_local  = banded_row_local;
    k.hb_sweep          = hb_sweep;
    k.hb_sweep_pmax     = hb_sweep_pmax;
    k.hb_base           = hb_base;
    k.hb_scan           = hb_scan;
    k.score             = score;
}

template <class T>
void register_level_inter(SimdLevel l, typename TypedKernels<T>::inter_fn inter_fill,
                          typename TypedKernels<T>::inter_fn inter_score, int inter_w) {
    TypedKernels<T>& k = level_table()[(int)l].template of<T>();
    k.inter_fill  = inter_fill;
    k.inter_score = inter_score;
    k.inter_w     = inter_w;
}

template <class T>
void register_level_grid(SimdLevel l, typename TypedKernels<T>::grid_fn grid_fwd,
                         typename TypedKernels<T>::gridbase_fn grid_base) {
    level_table()[(int)l].template of<T>().grid_fwd = grid_fwd;
    level_table()[(int)l].template of<T>().grid_base = grid_base;
}

template <class T>
void register_level_profile(SimdLevel l, typename TypedKernels<T>::profile_fn profile_build) {
    level_table()[(int)l].template of<T>().profile_build = profile_build;
}

#define NWGRAD_INSTANTIATE_REGISTER(T) \
    template void register_level_typed<T>(SimdLevel, TypedKernels<T>::viterbi_fn, \
        TypedKernels<T>::viterbi_fn, TypedKernels<T>::viterbi_fn, TypedKernels<T>::viterbi_fn, \
        void (*)(LhbJob<T>&), banded_row_fn_t<T>, banded_row_fn_t<T>, \
        TypedKernels<T>::hb_fn, TypedKernels<T>::hb_fn, TypedKernels<T>::hbbase_fn, \
        TypedKernels<T>::hbscan_fn, TypedKernels<T>::score_fn); \
    template void register_level_inter<T>(SimdLevel, TypedKernels<T>::inter_fn, \
        TypedKernels<T>::inter_fn, int); \
    template void register_level_grid<T>(SimdLevel, TypedKernels<T>::grid_fn, TypedKernels<T>::gridbase_fn); \
    template void register_level_profile<T>(SimdLevel, TypedKernels<T>::profile_fn);
NWGRAD_INSTANTIATE_REGISTER(double)
NWGRAD_INSTANTIATE_REGISTER(float)
NWGRAD_INSTANTIATE_REGISTER(int16_t)
NWGRAD_INSTANTIATE_REGISTER(int32_t)
NWGRAD_INSTANTIATE_REGISTER(int64_t)
#undef NWGRAD_INSTANTIATE_REGISTER

// ── The int8 score tier's gates (Aligner::score_i8_, BatchEngine::int8_own_) ─────
// Out of line, here, for the bindings / engine TU's inline budget (see Score8Req).  AlignMode
// values: 0 Global, 1 Local, 2 Glocal.
namespace {
constexpr int kScore8Local = 1;

// The predictor's cap: score8_try runs a Local problem in int8 only while the thread's previous
// problem of its type scored below it.  127 = "would have overflowed".  At 16 lanes (SSE2 /
// SSE4 / NEON) affine Local is capped at 64: there an int8 run pays only on low scores
// (wloczykij sse4, DNA L100 affine Local score, int8 run / int16 instructions: unrelated
// (scores ~12) 0.47, 40 % mutated (median 40) 0.63, 30 % (72) 0.83, 20 % (114) 1.00; an
// overflowing run costs 1.1x an int16 run before the int16 re-run), so the 127 cap cost 20 %
// mutated pairs 1.22x (median 114, 27 % >= 127; wall 1.25x) and 10 % mutated 1.04x.  Replaying
// those score sequences through that cost model: cap 64 -> 1.000 / 1.001, 30 % 0.94, unrelated
// 0.53; cap 96 still 1.046 at 20 %.  Linear's one-row int8 kernel gained at every rate there
// (0.69-0.86 at scores 20-90), and 32 / 64 lanes take L100 only past score_int8_min_segs: 127.
inline float score8_pred_cap(int lanes, bool linear) noexcept {
    return (lanes < 32 && !linear) ? 64.0f : 127.0f;
}

int score8_try_impl(const Score8Req& q, int& score) {
    const int mode = q.mode;
    if (q.m < 1 || q.n < 1) return 0;
    if (mode == 1) {   // the row length the level's lanes need (score_int8_min_segs), first:
        const int W = score_int8_lanes(q.backend);   // short pairs leave here, before any other test
        if ((q.n + W - 1) / W < score_int8_min_segs(W, q.linear != 0, q.nalpha)) return 0;
    }
    const score8_fn fn = level_kernels(q.backend).score_i8;
    if (!fn || q.smax_pos < 0) return 0;
    if (q.smax_pos > 127 || q.smax_neg > 127) return 0;
    const long long goa = q.linear ? 0 : static_cast<long long>(q.go_a);
    const long long gob = q.linear ? 0 : static_cast<long long>(q.go_b);
    const long long gea = static_cast<long long>(q.ge_a);
    const long long geb = static_cast<long long>(q.ge_b);
    if (goa < 0 || gob < 0 || gea < 0 || geb < 0 || goa + gea > 127 || gob + geb > 127) return 0;
    const long long gain = static_cast<long long>(std::min(q.m, q.n)) * q.smax_pos;
    if (q.am != kScore8Local && gain > 126) return 0;
    // Predictor: a Local run that overflows is wasted (up to the row that reached
    // 127), and related pairs overflow in runs -- protein L100 30 %-mutated Local
    // cost 2.0x the int16 instructions trying every pair (skynet sse4); a geometric
    // back-off (skip 1, 3, ... 63 after an overflow) still left 1.01-1.03.  The
    // wide kernel's score says exactly whether int8 would have held (Local: score
    // < 127), so the thread tries int8 only when its previous Local problem of this
    // type scored below 127: an all-overflowing batch pays one wasted run per
    // thread, an i.i.d. mix p(1-p) of its pairs.
    Score8Pred& pr = score8_pred();
    if (mode == 1 && pr.last[q.key] >= score8_pred_cap(score_int8_lanes(q.backend), q.linear != 0)) return -1;
    // Per-thread scratch: the int8 block (cast per problem) and the kernel's rows.
    struct Scratch { std::vector<int8_t> blk, rows; };
    static thread_local Scratch sc;
    const std::size_t nn = static_cast<std::size_t>(q.nalpha) * q.nalpha;
    sc.blk.resize(nn);
    switch (q.tbytes) {
    case 2: { const auto* s = static_cast<const int16_t*>(q.blk);
              for (std::size_t k = 0; k < nn; ++k) sc.blk[k] = static_cast<int8_t>(s[k]); } break;
    case 4: { const auto* s = static_cast<const int32_t*>(q.blk);
              for (std::size_t k = 0; k < nn; ++k) sc.blk[k] = static_cast<int8_t>(s[k]); } break;
    default: { const auto* s = static_cast<const int64_t*>(q.blk);
              for (std::size_t k = 0; k < nn; ++k) sc.blk[k] = static_cast<int8_t>(s[k]); } break;
    }
    Score8Job job{};
    job.a = q.a; job.m = q.m;
    job.b = q.b; job.n = q.n;
    job.blk = sc.blk.data(); job.nalpha = q.nalpha;
    job.go_a = static_cast<int8_t>(goa); job.ge_a = static_cast<int8_t>(gea);
    job.go_b = static_cast<int8_t>(gob); job.ge_b = static_cast<int8_t>(geb);
    job.local = q.am;
    job.linear = q.linear ? 1 : 0;
    job.gain = static_cast<int>(std::min<long long>(gain, 126));
    job.scratch = &sc.rows;
    fn(job);
    Score8Stats& st = score8_stats();
    if (st.on.load(std::memory_order_relaxed)) {
        st.runs.fetch_add(1, std::memory_order_relaxed);
        if (job.overflow) st.overflows.fetch_add(1, std::memory_order_relaxed);
    }
    if (job.overflow) return mode == 2 ? 0 : -1;   // gate 2: every problem tried
    pr.last[q.key] = static_cast<float>(job.score);
    score = job.score;
    return 1;
}

bool score8_own_impl(const Score8Req& q) {
    {   // the row lengths first: a group of short rows (the shared kernel's own case) leaves here
        const int W = score_int8_lanes(q.backend);
        const int segs = std::max(16, score_int8_min_segs(W, q.linear != 0, q.nalpha));
        for (int l = 0; l < q.real; ++l)
            if ((q.nb[l] + W - 1) / W < segs) return false;
    }
    if (score_int8_mode() != 1) return false;
    if (!level_kernels(q.backend).score_i8 ||
        score8_pred().last[q.key] >= score8_pred_cap(score_int8_lanes(q.backend), q.linear != 0)) return false;
    if (q.smax_pos < 0 || q.smax_pos > 127 || q.smax_neg > 127) return false;
    const double goa = q.linear ? 0.0 : q.go_a, gob = q.linear ? 0.0 : q.go_b;
    return goa >= 0 && gob >= 0 && q.ge_a >= 0 && q.ge_b >= 0 &&
           goa + q.ge_a <= 127 && gob + q.ge_b <= 127;
}
}   // namespace

void register_level_common(SimdLevel l, int row_block, score8_fn score_i8) {
    LevelKernels& k = level_table()[(int)l];
    k.score_i8  = score_i8;
    k.score8_try = &score8_try_impl;
    k.score8_own = &score8_own_impl;
    k.level     = static_cast<int>(l);
    k.row_block = row_block;
}
void register_level_soft(SimdLevel l, inter_soft_fn inter_soft,
                         soft_pair_fn soft_pair_linear, soft_pair_fn soft_pair_affine,
                         inter_soft_fn_f inter_soft_f, soft_pair_fn_f soft_pair_linear_f,
                         soft_pair_fn_f soft_pair_affine_f) {
    LevelKernels& k = level_table()[(int)l];
    k.inter_soft         = inter_soft;
    k.soft_pair_linear   = soft_pair_linear;
    k.soft_pair_affine   = soft_pair_affine;
    k.inter_soft_f       = inter_soft_f;
    k.soft_pair_linear_f = soft_pair_linear_f;
    k.soft_pair_affine_f = soft_pair_affine_f;
}
