// ── Per-pair scaled forward-backward kernels (SoftImpl::Scaled) ───────────────
//
// No include guard: included once at baseline inside aligner.hpp (namespace
// soft_base, which serves scalar_fallback, header-only builds and any level without its
// own copy) and once per ISA level inside level_common.inc (that level's namespace,
// compiled with its -march, registered as LevelKernels::soft_pair_*).  Separate
// namespaces make the copies distinct functions, so the linker never merges a
// wide-ISA copy into baseline code.
//
// The algorithm and its range argument are documented at Aligner::run_fwdbwd and in
// AGENTS.md ("The soft path").  Everything the kernel needs arrives in SoftPairJob
// (simd_levels.hpp): sequences, the exp'd query profile and gap weights (temperature
// applied), per-row band ranges, tables and scratch.  It writes log Z (of params/T),
// the expected counts into J.scnt and the gap fields (already negative), and ok = 0 if
// the lost-mass bound or an overflow check failed.
//
// Rows are split into carry-free passes the compiler vectorizes plus one gap carry
// stepped two cells at a time; reductions carry `omp simd` (the build sets
// -fopenmp-simd).  FMA contraction is enabled for these functions only — the soft path
// is tolerance-tested, the hard kernels keep -ffp-contract=off.

#ifndef NWGRAD_SOFT_PAIR_FMA_DEFINED
#  define NWGRAD_SOFT_PAIR_FMA_DEFINED
#  if defined(__clang__)
#    define NWGRAD_SP_FMA_FN
#    define NWGRAD_SP_FMA_BODY _Pragma("clang fp contract(fast)")
#  elif defined(__GNUC__) && defined(__aarch64__)
// gcc on AArch64 also unrolls these loops: it issues one vector iteration per loop trip
// where clang unrolls and interleaves (whole-TU -funroll-loops, NEON: 7-14 % on the soft
// kernels).
#    define NWGRAD_SP_FMA_FN __attribute__((optimize("fp-contract=fast", "unroll-loops")))
#    define NWGRAD_SP_FMA_BODY
#  elif defined(__GNUC__)
#    define NWGRAD_SP_FMA_FN __attribute__((optimize("fp-contract=fast")))
#    define NWGRAD_SP_FMA_BODY
#  else
#    define NWGRAD_SP_FMA_FN
#    define NWGRAD_SP_FMA_BODY
#  endif
#  define NWGRAD_SP_PRAGMA_(x) _Pragma(#x)
// Row maxima as reductions.  clang does not recognise `v > m ? v : m` as an FP max
// reduction without no-NaN/no-signed-zero flags, so every loop carrying one stayed
// scalar under clang (omp simd included: -Wpass-failed) — the fused backward pass
// among them.  IEEE `maximum` is a reduction LLVM vectorizes without fast-math, and
// equals the select here: the values are probability mass, never NaN or -0.  gcc
// vectorizes the select (and would call libm for fmax), so it keeps it.
// Not on x86 below SSE4.1: without blendvpd, clang lowers IEEE maximum (NaN and +-0
// ordering) to and/andn/or sequences that cost more than the scalar loop (measured
// 1.07-1.15x slower at sse2 / scalar_fallback); NEON's fmax is maximum itself.
#  if defined(__clang__) && defined(__has_builtin) && (!(defined(__x86_64__) || defined(__i386__)) || defined(__SSE4_1__))
#    if __has_builtin(__builtin_elementwise_maximum)
#      define NWGRAD_SP_VMAX(m, v) __builtin_elementwise_maximum((m), (v))
#    endif
#  endif
// gcc on AArch64: an `omp simd reduction(max:)` keeps its per-lane accumulator in a stack
// array, stored and reloaded every vector step (it is never promoted back to a register),
// while a plain fmax loop is vectorized as a register reduction (fmaxnm is one
// instruction there; on x86 fmax is a libm call, hence the select above).  So there the
// max is reduced WITHOUT the pragma, and the fused backward passes take it in a separate
// loop over the row they just wrote (NWGRAD_SP_SPLIT_MAX): the same values, the same
// order of additions.  The values are mass, never NaN or -0, so fmax equals the select.
#  if !defined(NWGRAD_SP_VMAX) && defined(__GNUC__) && !defined(__clang__) && defined(__aarch64__)
// Type-generic: __builtin_fmax on float operands promotes to double and the loop runs
// scalar (the float32 soft pass was 1.3-2x slower than double on the M1, 2026-10-31).
#    define NWGRAD_SP_VMAX(m, v) sp_fmax((m), (v))
#    define NWGRAD_SP_SPLIT_MAX 1
#  endif
#  ifndef NWGRAD_SP_VMAX
#    define NWGRAD_SP_VMAX(m, v) ((v) > (m) ? (v) : (m))
#  endif
#  ifndef NWGRAD_SP_SPLIT_MAX
#    define NWGRAD_SP_SPLIT_MAX 0
#  endif
#  define NWGRAD_SP_SUM(...) NWGRAD_SP_PRAGMA_(omp simd reduction(+:__VA_ARGS__))
#  if NWGRAD_SP_SPLIT_MAX
#    define NWGRAD_SP_MAX(v)
#  else
#    define NWGRAD_SP_MAX(v) NWGRAD_SP_PRAGMA_(omp simd reduction(max:v))
#  endif
#endif


// y[j] = u[j] + e·y[j-1], j = lo..hi (y[lo-1] given; u may alias y).  FOUR cells per
// carry step: only y[j+3] = (u3 + e·u2 + e²·u1 + e³·u0) + e⁴·c sits on the serial
// chain (one FMA latency per four cells); y[j..j+2] hang off it.  Reassociated — the
// soft path is tolerance-tested.
// sp_ldexp(x, e) == std::ldexp(x, e), bit for bit: a product with an exact power of two rounds
// once, as ldexp does (and flushes as it does under FTZ/DAZ), and the power is built from its
// bits -- libc's scalbn was a call per row (~1-3 % of the per-pair float32 linear pass at
// AVX2, 2026-11-02).  Outside the normal exponent range, std::ldexp itself.
template <class SF>
static inline SF sp_ldexp(SF x, int e) noexcept {
    if constexpr (std::is_same_v<SF, float>) {
        if (e >= -126 && e <= 127) return x * std::bit_cast<float>(static_cast<uint32_t>(e + 127) << 23);
    } else {
        if (e >= -1022 && e <= 1023) return x * std::bit_cast<double>(static_cast<uint64_t>(e + 1023) << 52);
    }
    return std::ldexp(x, e);
}
template <class SF>
static inline void sp_carry_fwd(SF* y, const SF* u, int lo, int hi, SF e) noexcept {
    NWGRAD_SP_FMA_BODY   // clang: a pragma reaches only its own scope, not the caller's
    SF c = y[lo - 1];
    const SF e2 = e * e, e3 = e2 * e, e4 = e2 * e2;
    int j = lo;
    for (; j + 3 <= hi; j += 4) {
        const SF u0 = u[j], u1 = u[j + 1], u2 = u[j + 2], u3 = u[j + 3];
        const SF p1 = u1 + e * u0, p2 = u2 + e * p1, p3 = u3 + e * p2;
        y[j] = u0 + e * c;
        y[j + 1] = p1 + e2 * c;
        y[j + 2] = p2 + e3 * c;
        c = p3 + e4 * c;
        y[j + 3] = c;
    }
    for (; j <= hi; ++j) { c = u[j] + e * c; y[j] = c; }
}
// y[j] += e·y[j+1], j = hi..lo, four cells per carry step.
template <class SF>
static inline void sp_carry_bwd(SF* y, int lo, int hi, SF e) noexcept {
    NWGRAD_SP_FMA_BODY   // clang: a pragma reaches only its own scope, not the caller's
    SF c = y[hi + 1];
    const SF e2 = e * e, e3 = e2 * e, e4 = e2 * e2;
    int j = hi;
    for (; j - 3 >= lo; j -= 4) {
        const SF u0 = y[j], u1 = y[j - 1], u2 = y[j - 2], u3 = y[j - 3];
        const SF p1 = u1 + e * u0, p2 = u2 + e * p1, p3 = u3 + e * p2;
        y[j] = u0 + e * c;
        y[j - 1] = p1 + e2 * c;
        y[j - 2] = p2 + e3 * c;
        c = p3 + e4 * c;
        y[j - 3] = c;
    }
    for (; j >= lo; --j) { c = y[j] + e * c; y[j] = c; }
}

// LAZY rescale: only when the row max mx leaves [2^-256, 2^256] (or overflows: bad).
// Scales [lo, hi] of the given rows (nullptr skips) and mx; returns the exponent removed.
template <class SF>
static inline int sp_lazy(SF& mx, int lo, int hi, bool& bad, SF* r0, SF* r1 = nullptr,
                          SF* r2 = nullptr, SF* r3 = nullptr) noexcept {
    NWGRAD_SP_FMA_BODY
    constexpr SF win = SoftRange<SF>::win;
    if (!(mx <= std::numeric_limits<SF>::max())) { bad = true; return 0; }
    if (mx == SF(0) || (mx <= win && mx >= SF(1) / win)) return 0;
    const int k = std::ilogb(mx);
    const SF sc = sp_ldexp(SF(1), -k);
    for (SF* r : {r0, r1, r2, r3})
        if (r) for (int j = lo; j <= hi; ++j) r[j] *= sc;
    mx *= sc;
    return k;
}
// Bound on the posterior damage of mass lost in one row pair (see AGENTS.md): lost terms
// are below DMIN in stored units (x2^-k where a rescale scaled UP), and act through the
// other direction's actual row max times the row's posterior factor g.  `terms`: the
// operations per cell and direction whose result can fall below DMIN, with margin —
// kSpTermsLinear (~7.5 here: the cell sum and the four-cell carry) and kSpTermsAffine,
// the same counts the inter-pair soft kernels (inter_soft_impl.inl) charge, so a pair
// passes or fails the range check whichever fill computes it.
template <class SF> static inline SF sp_fmax(SF m, SF v) noexcept {
    if constexpr (std::is_same_v<SF, float>) return __builtin_fmaxf(m, v); else return __builtin_fmax(m, v);
}
static constexpr double kSpTermsLinear = 16.0, kSpTermsAffine = 32.0;
template <class SF> static constexpr double kSpMin = static_cast<double>(std::numeric_limits<SF>::min());
template <class SF>
static inline double sp_loss(double terms, int n, int kf, int kb, SF mxf, SF mxb,
                             double g) noexcept {
    NWGRAD_SP_FMA_BODY
    const double lf = kf < 0 ? sp_ldexp(1.0, -kf) : 1.0;
    const double lb = kb < 0 ? sp_ldexp(1.0, -kb) : 1.0;
    // In units of SF's MIN: the callers accumulate it apart (lossu) and scale once.  With
    // MIN inside the per-row product it went SUBNORMAL in double whenever a row's
    // posterior factor was small (Z large: Local, Glocal's row-m sum, ~2^30 on miRNA
    // pairs) -- a microcode assist per row on x86, 13 % of a linear glocal soft pass.
    return terms * (n + 1) * (lf * mxb + lb * mxf) * g;
}
// The gradient pass multiplies table values -- F̂·P·B̂ for the match posteriors, F̂·B̂ for
// the gap sums -- with B̂ from the backward row BEFORE its rescale, in whatever stored
// units the two rows happen to be in.  Such a product can flush below MIN while the
// posterior it carries is O(1): a 31 x 4 Global pair at T = 0.05 lost 20 % of a count in
// float32 (a cell with F̂·P = 1.7e-20, B̂ = 3.6e-20, posterior 0.12) with the tables'
// bound at 5e-7.  The flushed products are COUNTED: lb sums B̂ over the cells whose
// intermediate F̂·P flushed with both factors nonzero (lost < MIN·B̂·gv0 each), lc counts
// the products that flushed as a whole with nonzero factors (lost < MIN·g each).  A bound
// through the row maxima instead refused every long diffuse pair (max·max·g reaches
// 2^185 where gaps cost 0.3 nats).  In double every term here is ~2^-700.
// (sp_loss_products, sp_loss_products_max and the counting budget: simd_levels.hpp, shared
// with the inter-pair kernels.)
// A product that flushed to zero although both factors were not.
#define NWGRAD_SP_FLUSHED(prod, a, b) (((prod) == SF(0) && (a) > SF(0) && (b) > SF(0)) ? SF(1) : SF(0))
// acc += a * b for the gap sums.  With a fused multiply-add the product is exact inside the
// operation and cannot flush (only the sum rounds), so FMA targets accumulate that way and
// count nothing; the others take the plain product and count its flush into lc.
// Double counts nothing anywhere: its terms are ~2^-700, and the tests cost it 1.2-1.9x
// (sse4, 2026-11-01).  kSpCount: the float32 instantiation only.
// NWGRAD_SP_NOFMA: a timing probe (the plain product on an FMA target).  The gradient
// pass itself never counts: the counting is a second pass over the finished row
// (sp_count_*, below), run only where sp_loss_products_max says it could matter.
// Not under clang: LLVM's vectorizer does not take an llvm.fma recurrence as a reduction,
// so every float32 gradient loop carrying one stayed scalar (-Wpass-failed at AVX2,
// AVX-512 and NEON; the wheels are gcc's, a clang build is not). clang takes the plain product and
// counts its flush, as the targets without FMA do (sound whether or not the backend
// fuses it: a fused product cannot flush, so the count can only overcharge).
#if (defined(__FMA__) || defined(__ARM_FEATURE_FMA)) && !defined(NWGRAD_SP_NOFMA) && !defined(__clang__)
#  define NWGRAD_SP_MAC(acc, a, b) do { if constexpr (kSpCountF) (acc) = std::fma((a), (b), (acc)); else (acc) += (a) * (b); } while (0)
#  define NWGRAD_SP_MAC_COUNTS 0
#else
#  define NWGRAD_SP_MAC(acc, a, b) do { (acc) += (a) * (b); } while (0)
#  define NWGRAD_SP_MAC_COUNTS 1
#endif
// The flushed products of a FINISHED row, counted from the stored rows and tables: lc the
// two-factor products a[j + AO] * b[j] (sp_count_prod; sp_count_prod2: (a1 + a2) * b),
// and for the match products q[j-1] * P[j] * b[j] (sp_count_match: one q row,
// sp_count_match3: the three affine rows summed) lb the b of a flushed intermediate and
// lc the flushed whole product.  Only the float32 kernels call these, and only for a
// row whose worst case (sp_loss_products_max) would eat into the tolerance.
template <class SF, int AO>
static inline void sp_count_prod(const SF* __restrict a, const SF* __restrict b, int j1, int hi0, SF& lc) noexcept {
    SF c = 0;
    NWGRAD_SP_SUM(c)
    for (int j = j1; j <= hi0; ++j) { const SF av = a[j + AO], p = av * b[j]; c += NWGRAD_SP_FLUSHED(p, av, b[j]); }
    lc += c;
}
template <class SF, int AO>
static inline void sp_count_prod2(const SF* __restrict a1, const SF* __restrict a2, const SF* __restrict b, int j1, int hi0, SF& lc) noexcept {
    SF c = 0;
    NWGRAD_SP_SUM(c)
    for (int j = j1; j <= hi0; ++j) { const SF av = a1[j + AO] + a2[j + AO], p = av * b[j]; c += NWGRAD_SP_FLUSHED(p, av, b[j]); }
    lc += c;
}
template <class SF>
static inline void sp_count_match(const SF* __restrict q, const SF* __restrict pr, const SF* __restrict b, int j1, int hi0, SF& lb, SF& lc) noexcept {
    SF cb = 0, cc = 0;
    NWGRAD_SP_SUM(cb, cc)
    for (int j = j1; j <= hi0; ++j) {
        const SF qs = q[j - 1], t1 = qs * pr[j], tt = t1 * b[j];
        cb += (qs > SF(0) && t1 == SF(0)) ? b[j] : SF(0);
        cc += NWGRAD_SP_FLUSHED(tt, t1, b[j]);
    }
    lb += cb; lc += cc;
}
template <class SF>
static inline void sp_count_match3(const SF* __restrict q1, const SF* __restrict q2, const SF* __restrict q3, const SF* __restrict pr,
                                   const SF* __restrict b, int j1, int hi0, SF& lb, SF& lc) noexcept {
    SF cb = 0, cc = 0;
    NWGRAD_SP_SUM(cb, cc)
    for (int j = j1; j <= hi0; ++j) {
        const SF qs = q1[j - 1] + q2[j - 1] + q3[j - 1], t1 = qs * pr[j], tt = t1 * b[j];
        cb += (qs > SF(0) && t1 == SF(0)) ? b[j] : SF(0);
        cc += NWGRAD_SP_FLUSHED(tt, t1, b[j]);
    }
    lb += cb; lc += cc;
}
static inline double sp_post(int e, double izr, bool& bad) noexcept {
    const double f = sp_ldexp(izr, e);
    if (!(f <= std::numeric_limits<double>::max())) bad = true;
    return f;
}
// Match posteriors tmp[j] of row i into count row `crow` (the row of a[i-1]; tracked,
// row i-1 itself), times g.
template <class SF>
static inline void sp_scatter(const SoftPairJobT<SF>& J, const SF* tmp, size_t crow, int lo, int hi,
                              double g) noexcept {
    NWGRAD_SP_FMA_BODY
    const int na = J.nalpha;
    SF* srow = J.srow;
    for (int c = 0; c < na; ++c) srow[c] = SF(0);
    for (int j = lo; j <= hi; ++j) srow[J.b[j - 1]] += tmp[j];
    double* gr = J.scnt + crow * na;
    for (int c = 0; c < na; ++c) gr[c] += srow[c] * g;
}
// The profile / count row of DP row i (1..m): its residue's, or (Tr) its own.
template <bool Tr, class SF>
static inline size_t sp_row(const SoftPairJobT<SF>& J, int i) noexcept {
    if constexpr (Tr) return static_cast<size_t>(i - 1);
    else              return static_cast<size_t>(J.a[i - 1]);
}
// Z = 2^ze·zr from per-row sums (Local), row m's sum (Glocal) or the (m, n) cell
// (Global), zr normalized into [1, 2).  Returns false if Z is not positive and finite (handled by caller).
template <class SF>
static inline bool sp_log_z(SoftPairJobT<SF>& J, const int* S, const SF* rowsum_or_null,
                            SF corner, int& ze, double& izr) noexcept {
    double zr;
    if (!J.local) { ze = S[J.m]; zr = corner; }
    else if (J.local == 2) { ze = S[J.m]; zr = rowsum_or_null[J.m]; }
    else {
        ze = S[0];
        for (int i = 1; i <= J.m; ++i) ze = std::max(ze, S[i]);
        zr = 0.0;
        for (int i = 0; i <= J.m; ++i) zr += sp_ldexp(rowsum_or_null[i], S[i] - ze);
    }
    if (!(zr > 0.0) || !(zr <= std::numeric_limits<double>::max())) return false;
    const int kz = std::ilogb(zr);
    zr = sp_ldexp(zr, -kz); ze += kz;
    izr = 1.0 / zr;
    J.log_z = std::log(zr) + ze * 0.69314718055994530942;
    return true;
}

// ── Checkpoints (J.ck > 0) ─────────────────────────────────────────────────────
// The full forward tables are 8 B (float32: 4) per cell and state; from L ~1000 they leave
// the cache and the backward, which reads rows i and i-1 of them once each, turns
// memory-bound (AGENTS.md, "The soft path").  Checkpointed, the forward keeps only the
// first row of every block of ck rows (cM..cT) and computes into a block buffer of ck+1
// rows; the backward walks the blocks from the top down, recomputing each block from its
// checkpoint (the last block is still in the buffer) and running unchanged over it.
// The recomputation is the SAME function over the SAME inputs at the SAME addresses
// (every row of block c lands in slot i - c*ck, the scratch rows are q0/q1 both times), so
// it reproduces the first pass bit for bit, exponents and maxima included -- the
// backward's S, fmax and Z, taken from the first pass, describe the recomputed rows
// exactly, and the lost-mass bound is charged once per row as before
// (NWGRAD_SOFT_CKPT_CHECK hashes every row both times and compares).  Banded rows are
// zeroed over their span right before they are computed (the full-table path zeroed every
// span up front: the same values, since a row writes only itself).
#if defined(__GNUC__) || defined(__clang__)
#  define NWGRAD_SP_NOINLINE __attribute__((noinline))
#else
#  define NWGRAD_SP_NOINLINE
#endif
template <class SF>
static inline void sp_copy_span(SF* __restrict dst, const SF* __restrict src, int lo, int hi) noexcept {
    for (int j = lo; j <= hi; ++j) dst[j] = src[j];
}
// FNV-1a over the bits of row i (up to three states, nullptr skips) over its span, with its
// exponent and max.
template <class SF>
static inline uint64_t sp_row_hash(const SoftPairJobT<SF>& J, int i, const SF* a, const SF* b, const SF* c) noexcept {
    using U = std::conditional_t<sizeof(SF) == 4, uint32_t, uint64_t>;
    uint64_t h = 1469598103934665603ull;
    auto mix = [&h](uint64_t v) { h = (h ^ v) * 1099511628211ull; };
    for (const SF* r : {a, b, c})
        if (r) for (int j = J.slo[i]; j <= J.shi[i]; ++j) mix(std::bit_cast<U>(r[j]));
    mix(static_cast<uint64_t>(static_cast<int64_t>(J.S[i])));
    mix(std::bit_cast<U>(J.fmax[i]));
    return h;
}
// Rows i0..i1 of the block whose slot 0 is row `base`: record (first pass) or check them.
template <class SF>
static inline void sp_ck_hash(SoftPairJobT<SF>& J, const SF* A, const SF* B, const SF* C, int base,
                              int i0, int i1, bool check) noexcept {
    const size_t st = J.stride;
    for (int i = i0; i <= i1; ++i) {
        const size_t o = static_cast<size_t>(i - base) * st;
        const uint64_t h = sp_row_hash(J, i, A + o, B ? B + o : nullptr, C ? C + o : nullptr);
        if (!check) J.ckh[i] = h;
        else if (J.ckh[i] != h) ++J.ck_bad;
    }
}

// Forward rows i0..i1 of the linear model into F, row i at F + (i - base)*stride (base 0:
// the full table; else the block buffer, slot 0 holding row base = i0 - 1).  Writes S, fmax
// and the row sums.  False: a row overflowed.
template <class SF, bool Tr>
NWGRAD_SP_FMA_FN NWGRAD_SP_NOINLINE static bool sp_linear_fwd(const SoftPairJobT<SF>& J, SF* F, int base,
                                                              int i0, int i1) noexcept {
    NWGRAD_SP_FMA_BODY
    const int m = J.m;
    const bool L = J.local == 1, G = J.local == 2;
    SF ea = J.ea, eb = J.eb;
    const SF* RW = J.rw;
    const size_t st = J.stride, w = static_cast<size_t>(J.n) + 2;
    int* S = J.S;
    SF* rs = J.rowsum;
    SF* mf = J.fmax;
    const SF* P = J.P;
    bool bad = false;
    for (int i = i0; i <= i1; ++i) {
        SF* __restrict r = F + static_cast<size_t>(i - base) * st;
        const SF* __restrict p = r - st;
        if (!J.full) for (int j = J.slo[i]; j <= J.shi[i]; ++j) r[j] = 0.0;
        if constexpr (Tr) { ea = RW[4 * i + 1]; eb = RW[4 * i + 3]; }
        SF fr = 0.0;
        if (L) { fr = sp_ldexp(SF(1), -S[i - 1]); r[0] = fr; }
        else   r[0] = (i <= J.bi) ? p[0] * eb : 0.0;
        const SF* __restrict pr = P + sp_row<Tr>(J, i) * w;
        const int lo = J.jlo[i], hi = J.jhi[i];
        for (int j = lo; j <= hi; ++j) r[j] = p[j - 1] * pr[j] + p[j] * eb + fr;
        if (lo <= hi) sp_carry_fwd(r, r, lo, hi, ea);
        SF mx = 0.0;
        NWGRAD_SP_MAX(mx)
        for (int j = J.slo[i]; j <= J.shi[i]; ++j) mx = NWGRAD_SP_VMAX(mx, r[j]);
        S[i] = S[i - 1] + sp_lazy(mx, J.slo[i], J.shi[i], bad, r);
        if (bad) return false;
        mf[i] = mx;
        if (L || (G && i == m)) {
            SF s = 0.0;
            NWGRAD_SP_SUM(s)
            for (int j = J.jlo0[i]; j <= J.jhi0[i]; ++j) s += r[j];
            rs[i] = s;
        }
    }
    return true;
}
// Block c of a checkpointed linear forward, recomputed into the buffer from its checkpoint.
template <class SF, bool Tr>
static inline bool sp_linear_reload(SoftPairJobT<SF>& J, int c) noexcept {
    const int k = J.ck, i0 = c * k, i1 = std::min(i0 + k, J.m);
    const size_t st = J.stride;
    sp_copy_span(J.FM, J.cM + static_cast<size_t>(c) * st, J.slo[i0], J.shi[i0]);
    if (!sp_linear_fwd<SF, Tr>(J, J.FM, i0, i0 + 1, i1)) return false;
    if (J.ckh) sp_ck_hash<SF>(J, J.FM, nullptr, nullptr, i0, i0 + 1, i1, true);
    return true;
}

// Tr: tracked (J.rw) — each row's weights, its own profile and count row, and the gap
// gradient per row into J.grow.
// Ck: checkpointed (J.ck > 0).  A template flag, not a runtime test: with the block
// offset a variable, the full-table path ran ~0.5 % more instructions on miRNA-sized pairs.
template <class SF, bool Tr, bool Ck>
NWGRAD_SP_FMA_FN static void soft_pair_linear_t(SoftPairJobT<SF>& J) noexcept {
    NWGRAD_SP_FMA_BODY
    constexpr bool kSpCountF = std::is_same_v<SF, float> && NWGRAD_SP_COUNT_F32;   // the float32 instantiation accounts for flushed products (sp_count_*)
    const int m = J.m, n = J.n;
    // Glocal (local == 2): row 0 a free start (no moves along it), column 0 Global's
    // chain, no free start inside, Z = row m's sum, the backward seeded along row m.
    const bool L = J.local == 1, G = J.local == 2;
    const int kmin = (L || G) ? 1 : 0;    // first row whose horizontal moves count
    const int kmin0 = L ? 1 : 0;          // first row whose column-0 vertical move counts
    SF ea = J.ea, eb = J.eb;
    const SF* RW = J.rw;
    if constexpr (Tr) {
        ea = RW[1];
        for (int i = 0; i <= m; ++i) for (int f = 0; f < 4; ++f) J.grow[4 * i + f] = 0.0;
    }
    const size_t st = J.stride, w = static_cast<size_t>(n) + 2;
    SF* F = J.FM;
    int* S = J.S;
    SF* rs = J.rowsum;
    SF* mf = J.fmax;
    const SF* P = J.P;
    bool bad = false;
    J.ok = 0;

    const int ck = Ck ? J.ck : 0;
    const int nb = (ck && m) ? (m + ck - 1) / ck : 1;   // blocks (one: the full table)
    if (!J.full) for (int j = J.slo[0]; j <= J.shi[0]; ++j) F[j] = 0.0;
    if (L || G) { for (int j = 0; j <= n; ++j) F[j] = 1.0; }
    else {
        if (J.full) for (int j = 0; j <= n; ++j) F[j] = 0.0;
        F[0] = 1.0;
        for (int j = 1; j <= J.bj; ++j) F[j] = F[j - 1] * ea;
    }
    S[0] = 0;
    {
        SF mx = 0.0;
        for (int j = J.slo[0]; j <= J.shi[0]; ++j) mx = NWGRAD_SP_VMAX(mx, F[j]);
        mf[0] = mx;
        if (L || (G && m == 0)) { SF s = 0.0; for (int j = J.jlo0[0]; j <= J.jhi0[0]; ++j) s += F[j]; rs[0] = s; }
    }
    if constexpr (!Ck) { if (!sp_linear_fwd<SF, Tr>(J, F, 0, 1, m)) return; }
    else
        for (int c = 0; c < nb; ++c) {
            const int i0 = c * ck, i1 = std::min(i0 + ck, m);
            if (c > 0) sp_copy_span(F, F + static_cast<size_t>(ck) * st, J.slo[i0], J.shi[i0]);
            sp_copy_span(J.cM + static_cast<size_t>(c) * st, F, J.slo[i0], J.shi[i0]);
            if (i0 < i1 && !sp_linear_fwd<SF, Tr>(J, F, i0, i0 + 1, i1)) return;
            if (J.ckh) sp_ck_hash<SF>(J, F, nullptr, nullptr, i0, i0 + 1, i1, false);
        }
    int base = (nb - 1) * ck, curb = nb - 1;   // the block in the buffer, and its slot-0 row
    const bool mn_in = n >= J.jlo0[m] && n <= J.jhi0[m];
    const SF corner = mn_in ? F[static_cast<size_t>(m - base) * st + n] : 0.0;
    int ze; double izr;
    if (!sp_log_z(J, S, rs, corner, ze, izr)) {
        if (!L && !G && !mn_in) {   // Global, (m, n) outside the band: log Z = -inf, genuinely
            J.log_z = -std::numeric_limits<SF>::infinity();
            J.g_oa = J.g_ea = J.g_ob = J.g_eb = 0.0;
            J.ok = 1;
        }
        return;
    }

    SF* cur = J.r0; SF* nxt = J.r1; SF* tmp = J.r2;
    for (size_t j = 0; j < w; ++j) { cur[j] = 0.0; nxt[j] = 0.0; }
    int clo = 0, chi = n, nlo = 0, nhi = n;
    int Tn = 0;
    double g_ea = 0.0, g_eb = 0.0, loss = 0.0, lossu = 0.0;   // lossu: sp_loss's, in units of MIN
    for (int i = m; i >= 0; --i) {
        if constexpr (Ck) {   // row i reads forward rows i and i-1: block (i-1)/ck (row 0: block 0)
            const int cb = i ? (i - 1) / ck : 0;
            if (cb != curb) {
                curb = cb; base = cb * ck;
                if (!sp_linear_reload<SF, Tr>(J, cb)) return;
            }
        }
        const int lo0 = J.jlo0[i], hi0 = J.jhi0[i];
        for (int j = clo; j <= chi; ++j) cur[j] = 0.0;
        const SF init = L ? sp_ldexp(SF(1), -Tn) : 0.0;
        SF ebn = eb;   // the move down into row i+1 (Tr: that row's)
        if constexpr (Tr) { ea = RW[4 * i + 1]; eb = RW[4 * i + 3]; if (i < m) ebn = RW[4 * (i + 1) + 3]; }
        if (i < m) {
            const SF* __restrict pn = P + sp_row<Tr>(J, i + 1) * w;
            SF* __restrict c = cur;
            const SF* __restrict nx = nxt;
            for (int j = lo0; j <= hi0; ++j) c[j] = pn[j + 1] * nx[j + 1] + nx[j] * ebn + init;
        } else {
            const SF e = G ? SF(1) : init;   // Glocal: every cell of row m an end
            for (int j = lo0; j <= hi0; ++j) cur[j] = e;
            if (!L && !G && n >= lo0 && n <= hi0) cur[n] = 1.0;
        }
        if (lo0 <= hi0) sp_carry_bwd(cur, lo0, hi0, ea);
        clo = lo0; chi = hi0;

        // One vector pass: row max and every gradient sum, on the pre-rescale values.
        const SF* __restrict Fi = F + static_cast<size_t>(i - base) * st;
        const SF* __restrict cb = cur;
        const int j1 = std::max(1, lo0);
        SF mx = 0.0, sa = 0.0, sb = 0.0;
        const double gh0 = sp_post(S[i] + Tn - ze, izr, bad);   // the pre-rescale units' posterior factor
        double lossp = 0.0;                                       // the gradient products' flushes (float32)
        if (lo0 == 0) { mx = cb[0]; if (i >= 1 && kmin0 == 0) sb += (Fi - st)[0] * cb[0]; }
        if (i >= 1) {
            const SF* __restrict Fp = Fi - st;
            const SF* __restrict pr = P + sp_row<Tr>(J, i) * w;
            SF* __restrict t = tmp;
#if NWGRAD_SP_SPLIT_MAX
            for (int j = j1; j <= hi0; ++j) mx = NWGRAD_SP_VMAX(mx, cb[j]);
            NWGRAD_SP_PRAGMA_(omp simd reduction(+:sa, sb))
#else
            NWGRAD_SP_PRAGMA_(omp simd reduction(+:sa, sb) reduction(max:mx))
#endif
            for (int j = j1; j <= hi0; ++j) {
                const SF b = cb[j];
#if !NWGRAD_SP_SPLIT_MAX
                mx = NWGRAD_SP_VMAX(mx, b);
#endif
                NWGRAD_SP_MAC(sa, Fi[j - 1], b);
                NWGRAD_SP_MAC(sb, Fp[j], b);
                t[j] = Fp[j - 1] * pr[j] * b;
            }
        } else {
#if NWGRAD_SP_SPLIT_MAX
            for (int j = j1; j <= hi0; ++j) mx = NWGRAD_SP_VMAX(mx, cb[j]);
            NWGRAD_SP_SUM(sa)
#else
            NWGRAD_SP_PRAGMA_(omp simd reduction(+:sa) reduction(max:mx))
#endif
            for (int j = j1; j <= hi0; ++j) {
                const SF b = cb[j];
#if !NWGRAD_SP_SPLIT_MAX
                mx = NWGRAD_SP_VMAX(mx, b);
#endif
                NWGRAD_SP_MAC(sa, Fi[j - 1], b);
            }
        }
        if constexpr (kSpCountF) {   // the flushed products: the worst case, or the count where it could matter
            const double gv0 = i >= 1 ? sp_post(S[i - 1] + Tn - ze, izr, bad) : 0.0;
            const int kinds = (i >= 1 ? 1 : 0) + (i >= 1 ? 2 : 1) * NWGRAD_SP_MAC_COUNTS;
            lossp = sp_loss_products_max<SF>(hi0 - j1 + 1, kinds, mx, gh0, gv0);
            if (!(loss + lossu * kSpMin<SF> + lossp <= kSpCountBudget * SoftRange<SF>::tol)) {
                SF lb = 0.0, lc = 0.0;
                if (i >= 1) {
                    const SF* Fp = Fi - st;
                    sp_count_match<SF>(Fp, P + sp_row<Tr>(J, i) * w, cb, j1, hi0, lb, lc);
                    if (NWGRAD_SP_MAC_COUNTS) { sp_count_prod<SF, -1>(Fi, cb, j1, hi0, lc); sp_count_prod<SF, 0>(Fp, cb, j1, hi0, lc); }
                } else if (NWGRAD_SP_MAC_COUNTS) sp_count_prod<SF, -1>(Fi, cb, j1, hi0, lc);
                lossp = sp_loss_products<SF>(lb, lc, gh0, gv0);
            }
        }
        if (J.gpost) {   // row argmax of the EXIT mass (see SoftPairJob::gpost)
            int bj = lo0; SF bv = -1.0;
            for (int j = std::max(lo0, 0); j <= hi0; ++j) {
                const SF nb = j < hi0 ? cb[j + 1] : 0.0;
                const SF v = Fi[j] * (cb[j] - ea * nb);
                if (v > bv) { bv = v; bj = j; }
            }
            J.gpost[i] = bj;
        }
        const int kb = sp_lazy(mx, lo0, hi0, bad, cur);
        if (bad) return;
        const int Ti = Tn + kb;
        // Sums were taken before any rescale: their factor uses Tn (gh0), the bound uses Ti.
        const double gh = sp_post(S[i] + Ti - ze, izr, bad);
        lossu += sp_loss(kSpTermsLinear, n, i > 0 ? S[i] - S[i - 1] : 0, kb, mf[i], mx, gh);
        if constexpr (kSpCountF) loss += lossp;
        if (i >= kmin) {
            g_ea += sa * ea * gh0;
            if constexpr (Tr) J.grow[4 * i + 1] = -(sa * ea * gh0);
        }
        if (i >= 1) {
            const double gv0 = sp_post(S[i - 1] + Tn - ze, izr, bad);
            g_eb += sb * eb * gv0;
            if constexpr (Tr) J.grow[4 * i + 3] = -(sb * eb * gv0);
            sp_scatter(J, tmp, sp_row<Tr>(J, i), j1, hi0, gv0);
        }
        if (bad) return;
        Tn = Ti;
        std::swap(cur, nxt); std::swap(clo, nlo); std::swap(chi, nhi);
    }
    loss += lossu * kSpMin<SF>;
    J.loss = loss;
    if (!(loss <= SoftRange<SF>::tol)) return;
    J.g_oa = 0.0; J.g_ea = -g_ea; J.g_ob = 0.0; J.g_eb = -g_eb;
    J.ok = 1;
}

// Forward rows i0..i1 of the affine model, as sp_linear_fwd (row i at FM/FX/FY + (i -
// base)*stride).  tp: the previous row's M+X+Y on entry (row i0-1), row i1's on exit; uy:
// the Y carry's input.
template <class SF, bool Tr>
NWGRAD_SP_FMA_FN NWGRAD_SP_NOINLINE static bool sp_affine_fwd(const SoftPairJobT<SF>& J, SF* FM, SF* FX, SF* FY,
                                                              int base, int i0, int i1, SF* tp, SF* uy) noexcept {
    NWGRAD_SP_FMA_BODY
    const int m = J.m;
    const bool L = J.local == 1, G = J.local == 2;
    SF oa = J.oa, ea = J.ea, ob = J.ob, eb = J.eb;
    const SF* RW = J.rw;
    const size_t st = J.stride, w = static_cast<size_t>(J.n) + 2;
    int* S = J.S;
    SF* rs = J.rowsum;
    SF* mf = J.fmax;
    const SF* P = J.P;
    bool bad = false;
    for (int i = i0; i <= i1; ++i) {
        const size_t ro = static_cast<size_t>(i - base) * st;
        SF* __restrict rM = FM + ro; SF* __restrict rX = FX + ro;
        SF* __restrict rY = FY + ro;
        const SF* __restrict pM = rM - st; const SF* __restrict pX = rX - st;
        const SF* __restrict pY = rY - st;
        if (!J.full) for (int j = J.slo[i]; j <= J.shi[i]; ++j) { rM[j] = 0.0; rX[j] = 0.0; rY[j] = 0.0; }
        if constexpr (Tr) { oa = RW[4 * i]; ea = RW[4 * i + 1]; ob = RW[4 * i + 2]; eb = RW[4 * i + 3]; }
        SF fr = 0.0;
        if (L) { fr = sp_ldexp(SF(1), -S[i - 1]); rM[0] = fr; rX[0] = 0.0; rY[0] = 0.0; }
        else {
            rM[0] = 0.0; rY[0] = 0.0;
            rX[0] = (i <= J.bi) ? (pM[0] + pY[0]) * ob + pX[0] * eb : 0.0;
        }
        const SF* __restrict pr = P + sp_row<Tr>(J, i) * w;
        const SF* __restrict tq = tp;
        SF* __restrict u = uy;
        const int lo = J.jlo[i], hi = J.jhi[i];
        if (lo <= hi) {
            u[lo] = (rM[lo - 1] + rX[lo - 1]) * oa;
            // M, X, and the Y carry's input for the NEXT column, in one vector pass.
            for (int j = lo; j <= hi; ++j) {
                const SF mv = tq[j - 1] * pr[j] + fr;
                const SF xv = (pM[j] + pY[j]) * ob + pX[j] * eb;
                rM[j] = mv; rX[j] = xv;
                u[j + 1] = (mv + xv) * oa;
            }
            sp_carry_fwd(rY, u, lo, hi, ea);
        }
        SF mx = 0.0;
        SF* __restrict t = tp;
        NWGRAD_SP_MAX(mx)
        for (int j = J.slo[i]; j <= J.shi[i]; ++j) {
            const SF v = rM[j] + rX[j] + rY[j];
            t[j] = v;
            mx = NWGRAD_SP_VMAX(mx, v);
        }
        S[i] = S[i - 1] + sp_lazy(mx, J.slo[i], J.shi[i], bad, rM, rX, rY, tp);
        if (bad) return false;
        mf[i] = mx;
        if (L || (G && i == m)) {
            SF s = 0.0;
            NWGRAD_SP_SUM(s)
            for (int j = J.jlo0[i]; j <= J.jhi0[i]; ++j) s += tp[j];
            rs[i] = s;
        }
    }
    return true;
}
// Block c of a checkpointed affine forward, recomputed into the buffer from its checkpoint.
template <class SF, bool Tr>
static inline bool sp_affine_reload(SoftPairJobT<SF>& J, int c) noexcept {
    const int k = J.ck, i0 = c * k, i1 = std::min(i0 + k, J.m);
    const size_t co = static_cast<size_t>(c) * J.stride;
    const int lo = J.slo[i0], hi = J.shi[i0];
    sp_copy_span(J.FM, J.cM + co, lo, hi); sp_copy_span(J.FX, J.cX + co, lo, hi);
    sp_copy_span(J.FY, J.cY + co, lo, hi); sp_copy_span(J.q0, J.cT + co, lo, hi);
    if (!sp_affine_fwd<SF, Tr>(J, J.FM, J.FX, J.FY, i0, i0 + 1, i1, J.q0, J.q1)) return false;
    if (J.ckh) sp_ck_hash<SF>(J, J.FM, J.FX, J.FY, i0, i0 + 1, i1, true);
    return true;
}

// Ck: checkpointed (J.ck > 0).  A template flag, not a runtime test: with the block
// offset a variable, the full-table path ran ~0.5 % more instructions on miRNA-sized pairs.
template <class SF, bool Tr, bool Ck>
NWGRAD_SP_FMA_FN static void soft_pair_affine_t(SoftPairJobT<SF>& J) noexcept {
    NWGRAD_SP_FMA_BODY
    constexpr bool kSpCountF = std::is_same_v<SF, float> && NWGRAD_SP_COUNT_F32;   // the float32 instantiation accounts for flushed products (sp_count_*)
    const int m = J.m, n = J.n;
    // Glocal (local == 2): row 0 a free start (no moves along it), column 0 Global's
    // chain, no free start inside, Z = row m's sum, the backward seeded along row m.
    const bool L = J.local == 1, G = J.local == 2;
    const int kmin = (L || G) ? 1 : 0;    // first row whose horizontal moves count
    const int kmin0 = L ? 1 : 0;          // first row whose column-0 vertical move counts
    SF oa = J.oa, ea = J.ea, ob = J.ob, eb = J.eb;
    const SF* RW = J.rw;
    if constexpr (Tr) {
        oa = RW[0]; ea = RW[1];
        for (int i = 0; i <= m; ++i) for (int f = 0; f < 4; ++f) J.grow[4 * i + f] = 0.0;
    }
    const size_t st = J.stride, w = static_cast<size_t>(n) + 2;
    SF* FM = J.FM; SF* FX = J.FX; SF* FY = J.FY;
    int* S = J.S;
    SF* rs = J.rowsum;
    SF* mf = J.fmax;
    const SF* P = J.P;
    bool bad = false;
    J.ok = 0;
    const int ck = Ck ? J.ck : 0;
    const int nb = (ck && m) ? (m + ck - 1) / ck : 1;   // blocks (one: the full tables)
    // tp: M+X+Y of the previous forward row; uy: the Y carry's carry-free input.  Full
    // tables: r5 / r4 (r5 later the match scratch); checkpointed: q0 / q1, which the
    // recomputations inside the backward use again.
    SF* tp = ck ? J.q0 : J.r5;
    SF* uy = ck ? J.q1 : J.r4;

    if (!J.full) for (int j = J.slo[0]; j <= J.shi[0]; ++j) { FM[j] = 0.0; FX[j] = 0.0; FY[j] = 0.0; }
    else for (int j = 0; j <= n; ++j) { FM[j] = 0.0; FX[j] = 0.0; FY[j] = 0.0; }
    if (L || G) { for (int j = 0; j <= n; ++j) FM[j] = 1.0; }
    else {
        FM[0] = 1.0;
        if (J.bj >= 1) FY[1] = oa;
        for (int j = 2; j <= J.bj; ++j) FY[j] = FY[j - 1] * ea;
    }
    {
        SF mx = 0.0;
        for (int j = J.slo[0]; j <= J.shi[0]; ++j) {
            tp[j] = FM[j] + FX[j] + FY[j];
            mx = NWGRAD_SP_VMAX(mx, tp[j]);
        }
        mf[0] = mx;
    }
    S[0] = 0;
    if (L || (G && m == 0)) { SF s = 0.0; for (int j = J.jlo0[0]; j <= J.jhi0[0]; ++j) s += tp[j]; rs[0] = s; }
    if constexpr (!Ck) { if (!sp_affine_fwd<SF, Tr>(J, FM, FX, FY, 0, 1, m, tp, uy)) return; }
    else
        for (int c = 0; c < nb; ++c) {
            const int i0 = c * ck, i1 = std::min(i0 + ck, m);
            const int lo = J.slo[i0], hi = J.shi[i0];
            const size_t ko = static_cast<size_t>(ck) * st, co = static_cast<size_t>(c) * st;
            if (c > 0) { sp_copy_span(FM, FM + ko, lo, hi); sp_copy_span(FX, FX + ko, lo, hi); sp_copy_span(FY, FY + ko, lo, hi); }
            sp_copy_span(J.cM + co, FM, lo, hi); sp_copy_span(J.cX + co, FX, lo, hi);
            sp_copy_span(J.cY + co, FY, lo, hi); sp_copy_span(J.cT + co, tp, lo, hi);
            if (i0 < i1 && !sp_affine_fwd<SF, Tr>(J, FM, FX, FY, i0, i0 + 1, i1, tp, uy)) return;
            if (J.ckh) sp_ck_hash<SF>(J, FM, FX, FY, i0, i0 + 1, i1, false);
        }
    int base = (nb - 1) * ck, curb = nb - 1;   // the block in the buffer, and its slot-0 row
    const bool mn_in = n >= J.jlo0[m] && n <= J.jhi0[m];
    const size_t cmn = static_cast<size_t>(m - base) * st + n;
    const SF corner = mn_in ? FM[cmn] + FX[cmn] + FY[cmn] : 0.0;
    int ze; double izr;
    if (!sp_log_z(J, S, rs, corner, ze, izr)) {
        if (!L && !G && !mn_in) {
            J.log_z = -std::numeric_limits<SF>::infinity();
            J.g_oa = J.g_ea = J.g_ob = J.g_eb = 0.0;
            J.ok = 1;
        }
        return;
    }

    SF *cM = J.r0, *cX = J.r1, *cY = J.r2, *nM = J.r3, *nX = J.r4;
    SF* tmp = J.r5;
    for (size_t j = 0; j < w; ++j) { cM[j] = cX[j] = cY[j] = nM[j] = nX[j] = 0.0; }
    int clo = 0, chi = n, nlo = 0, nhi = n;
    int Tn = 0;
    double g_ea = 0.0, g_eb = 0.0, g_oa = 0.0, g_ob = 0.0, loss = 0.0, lossu = 0.0;   // as soft_pair_linear_t
    for (int i = m; i >= 0; --i) {
        if constexpr (Ck) {   // row i reads forward rows i and i-1: block (i-1)/ck (row 0: block 0)
            const int cb = i ? (i - 1) / ck : 0;
            if (cb != curb) {
                curb = cb; base = cb * ck;
                if (!sp_affine_reload<SF, Tr>(J, cb)) return;
            }
        }
        const int lo0 = J.jlo0[i], hi0 = J.jhi0[i];
        for (int j = clo; j <= chi; ++j) { cM[j] = 0.0; cX[j] = 0.0; }
        // cY is not swapped (only M and X of the next row are read): it still holds
        // row i+1 over [nlo, nhi].
        for (int j = nlo; j <= nhi; ++j) cY[j] = 0.0;
        const SF init = L ? sp_ldexp(SF(1), -Tn) : 0.0;
        SF obn = ob, ebn = eb;   // the move down into row i+1 (Tr: that row's)
        if constexpr (Tr) {
            oa = RW[4 * i]; ea = RW[4 * i + 1]; ob = RW[4 * i + 2]; eb = RW[4 * i + 3];
            if (i < m) { obn = RW[4 * (i + 1) + 2]; ebn = RW[4 * (i + 1) + 3]; }
        }
        {
            SF* __restrict b1 = cM; SF* __restrict b2 = cX;
            SF* __restrict y = cY;
            const SF* __restrict nm = nM; const SF* __restrict nx = nX;
            if (i < m) {
                const SF* __restrict pn = P + sp_row<Tr>(J, i + 1) * w;
                for (int j = lo0; j <= hi0; ++j) {
                    const SF d = pn[j + 1] * nm[j + 1] + init, v = nx[j];
                    const SF v1 = d + v * obn;
                    b1[j] = v1; b2[j] = d + v * ebn; y[j] = v1;
                }
            } else {
                const SF e = G ? SF(1) : init;   // Glocal: every cell of row m an end
                for (int j = lo0; j <= hi0; ++j) { b1[j] = e; b2[j] = e; y[j] = e; }
                if (!L && !G && n >= lo0 && n <= hi0) { b1[n] = 1.0; b2[n] = 1.0; y[n] = 1.0; }
            }
        }
        if (lo0 <= hi0) sp_carry_bwd(cY, lo0, hi0, ea);
        clo = lo0; chi = hi0;

        // One vector pass: final M/X, the row max, every gradient sum and the match
        // posteriors, all on the pre-rescale values.
        const size_t ro = static_cast<size_t>(i - base) * st;
        const SF* __restrict fM = FM + ro; const SF* __restrict fX = FX + ro;
        const SF* __restrict fY = FY + ro;
        SF* __restrict bM = cM; SF* __restrict bX = cX;
        const SF* __restrict bY = cY;
        const int j1 = std::max(1, lo0);
        SF mx = 0.0, sx = 0.0, sy = 0.0, so = 0.0, sob = 0.0;
        const double gh0 = sp_post(S[i] + Tn - ze, izr, bad);   // the pre-rescale units' posterior factor
        double lossp = 0.0;                                       // the gradient products' flushes (float32)
        if (lo0 == 0) {
            const SF h = bY[1] * oa;
            const SF mv = bM[0] + h, xv = bX[0] + h;
            bM[0] = mv; bX[0] = xv;
            mx = mv + xv + bY[0];
            if (i >= 1) {
                sx += fX[0] * xv;
                if (kmin0 == 0) sob += ((fM - st)[0] + (fY - st)[0]) * xv;
            }
        }
        if (i >= 1) {
            const SF* __restrict qM = fM - st; const SF* __restrict qX = fX - st;
            const SF* __restrict qY = fY - st;
            const SF* __restrict pr = P + sp_row<Tr>(J, i) * w;
            SF* __restrict t = tmp;
#if NWGRAD_SP_SPLIT_MAX
            NWGRAD_SP_SUM(sx, sy, so, sob)
#else
            NWGRAD_SP_PRAGMA_(omp simd reduction(+:sx, sy, so, sob) reduction(max:mx))
#endif
            for (int j = j1; j <= hi0; ++j) {
                const SF y = bY[j];
                const SF h = bY[j + 1] * oa;
                const SF mv = bM[j] + h, xv = bX[j] + h;
                bM[j] = mv; bX[j] = xv;
#if !NWGRAD_SP_SPLIT_MAX
                const SF tot = mv + xv + y;
                mx = NWGRAD_SP_VMAX(mx, tot);
#endif
                NWGRAD_SP_MAC(sx, fX[j], xv);
                NWGRAD_SP_MAC(sy, fY[j], y);
                NWGRAD_SP_MAC(so, fM[j - 1] + fX[j - 1], y);
                NWGRAD_SP_MAC(sob, qM[j] + qY[j], xv);
                t[j] = (qM[j - 1] + qX[j - 1] + qY[j - 1]) * pr[j] * mv;
            }
#if NWGRAD_SP_SPLIT_MAX
            for (int j = j1; j <= hi0; ++j) mx = NWGRAD_SP_VMAX(mx, bM[j] + bX[j] + bY[j]);
#endif
        } else {
#if NWGRAD_SP_SPLIT_MAX
            NWGRAD_SP_SUM(sy, so)
#else
            NWGRAD_SP_PRAGMA_(omp simd reduction(+:sy, so) reduction(max:mx))
#endif
            for (int j = j1; j <= hi0; ++j) {
                const SF y = bY[j];
                const SF h = bY[j + 1] * oa;
                const SF mv = bM[j] + h, xv = bX[j] + h;
                bM[j] = mv; bX[j] = xv;
#if !NWGRAD_SP_SPLIT_MAX
                const SF tot = mv + xv + y;
                mx = NWGRAD_SP_VMAX(mx, tot);
#endif
                NWGRAD_SP_MAC(sy, fY[j], y);
                NWGRAD_SP_MAC(so, fM[j - 1] + fX[j - 1], y);
            }
#if NWGRAD_SP_SPLIT_MAX
            for (int j = j1; j <= hi0; ++j) mx = NWGRAD_SP_VMAX(mx, bM[j] + bX[j] + bY[j]);
#endif
        }
        if constexpr (kSpCountF) {   // the flushed products: the worst case, or the count where it could matter
            const double gv0 = i >= 1 ? sp_post(S[i - 1] + Tn - ze, izr, bad) : 0.0;
            const int kinds = (i >= 1 ? 1 : 0) + (i >= 1 ? 4 : 2) * NWGRAD_SP_MAC_COUNTS;
            lossp = sp_loss_products_max<SF>(hi0 - j1 + 1, kinds, mx, gh0, gv0);
            if (!(loss + lossu * kSpMin<SF> + lossp <= kSpCountBudget * SoftRange<SF>::tol)) {
                SF lb = 0.0, lc = 0.0;
                if (i >= 1) {
                    const SF* qM = fM - st; const SF* qX = fX - st; const SF* qY = fY - st;
                    sp_count_match3<SF>(qM, qX, qY, P + sp_row<Tr>(J, i) * w, bM, j1, hi0, lb, lc);
                    if (NWGRAD_SP_MAC_COUNTS) {
                        sp_count_prod<SF, 0>(fX, bX, j1, hi0, lc); sp_count_prod<SF, 0>(fY, bY, j1, hi0, lc);
                        sp_count_prod2<SF, -1>(fM, fX, bY, j1, hi0, lc); sp_count_prod2<SF, 0>(qM, qY, bX, j1, hi0, lc);
                    }
                } else if (NWGRAD_SP_MAC_COUNTS) { sp_count_prod<SF, 0>(fY, bY, j1, hi0, lc); sp_count_prod2<SF, -1>(fM, fX, bY, j1, hi0, lc); }
                lossp = sp_loss_products<SF>(lb, lc, gh0, gv0);
            }
        }
        if (J.gpost) {   // row argmax of the EXIT mass (see SoftPairJob::gpost)
            int bj = lo0; SF bv = -1.0;
            for (int j = std::max(lo0, 0); j <= hi0; ++j) {
                const SF ny = j < hi0 ? fY[j + 1] * bY[j + 1] : 0.0;
                const SF v = fM[j] * bM[j] + fX[j] * bX[j] + fY[j] * bY[j] - ny;
                if (v > bv) { bv = v; bj = j; }
            }
            J.gpost[i] = bj;
        }
        const int kb = sp_lazy(mx, lo0, hi0, bad, cM, cX, cY);
        if (bad) return;
        const int Ti = Tn + kb;
        const double gh = sp_post(S[i] + Ti - ze, izr, bad);
        lossu += sp_loss(kSpTermsAffine, n, i > 0 ? S[i] - S[i - 1] : 0, kb, mf[i], mx, gh);
        if constexpr (kSpCountF) loss += lossp;
        g_eb += sx * gh0;
        g_ea += sy * gh0;
        if (i >= kmin) g_oa += so * oa * gh0;
        if constexpr (Tr) {
            J.grow[4 * i + 3] = -(sx * gh0);
            J.grow[4 * i + 1] = -(sy * gh0);
            if (i >= kmin) J.grow[4 * i] = -(so * oa * gh0);
        }
        if (i >= 1) {
            const double gv0 = sp_post(S[i - 1] + Tn - ze, izr, bad);
            g_ob += sob * ob * gv0;
            if constexpr (Tr) J.grow[4 * i + 2] = -(sob * ob * gv0);
            sp_scatter(J, tmp, sp_row<Tr>(J, i), j1, hi0, gv0);
        }
        if (bad) return;
        Tn = Ti;
        std::swap(cM, nM); std::swap(cX, nX);
        std::swap(clo, nlo); std::swap(chi, nhi);
    }
    loss += lossu * kSpMin<SF>;
    J.loss = loss;
    if (!(loss <= SoftRange<SF>::tol)) return;
    J.g_oa = -g_oa; J.g_ea = -g_ea; J.g_ob = -g_ob; J.g_eb = -g_eb;
    J.ok = 1;
}

NWGRAD_SP_FMA_FN static void soft_pair_linear(SoftPairJob& J) noexcept {
    if (J.ck) { if (J.rw) soft_pair_linear_t<double, true, true>(J); else soft_pair_linear_t<double, false, true>(J); }
    else if (J.rw) soft_pair_linear_t<double, true, false>(J); else soft_pair_linear_t<double, false, false>(J);
}
NWGRAD_SP_FMA_FN static void soft_pair_affine(SoftPairJob& J) noexcept {
    if (J.ck) { if (J.rw) soft_pair_affine_t<double, true, true>(J); else soft_pair_affine_t<double, false, true>(J); }
    else if (J.rw) soft_pair_affine_t<double, true, false>(J); else soft_pair_affine_t<double, false, false>(J);
}
NWGRAD_SP_FMA_FN static void soft_pair_linear_f(SoftPairJobF& J) noexcept {
    if (J.ck) { if (J.rw) soft_pair_linear_t<float, true, true>(J); else soft_pair_linear_t<float, false, true>(J); }
    else if (J.rw) soft_pair_linear_t<float, true, false>(J); else soft_pair_linear_t<float, false, false>(J);
}
NWGRAD_SP_FMA_FN static void soft_pair_affine_f(SoftPairJobF& J) noexcept {
    if (J.ck) { if (J.rw) soft_pair_affine_t<float, true, true>(J); else soft_pair_affine_t<float, false, true>(J); }
    else if (J.rw) soft_pair_affine_t<float, true, false>(J); else soft_pair_affine_t<float, false, false>(J);
}
// The entry of soft type SF (the Aligner's baseline copy picks by type).
template <class SF>
static inline auto soft_pair_entry(bool linear) noexcept {
    if constexpr (std::is_same_v<SF, float>) return linear ? &soft_pair_linear_f : &soft_pair_affine_f;
    else                                     return linear ? &soft_pair_linear : &soft_pair_affine;
}
