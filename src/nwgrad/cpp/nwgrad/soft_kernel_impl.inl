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
    const SF sc = std::ldexp(SF(1), -k);
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
template <class SF>
static inline double sp_loss(double terms, int n, int kf, int kb, SF mxf, SF mxb,
                             double g) noexcept {
    NWGRAD_SP_FMA_BODY
    const double lf = kf < 0 ? std::ldexp(1.0, -kf) : 1.0;
    const double lb = kb < 0 ? std::ldexp(1.0, -kb) : 1.0;
    return terms * (n + 1) * static_cast<double>(std::numeric_limits<SF>::min()) * (lf * mxb + lb * mxf) * g;
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
// (sp_loss_products and the counting budget: simd_levels.hpp, shared with the inter-pair kernels.)
// A product that flushed to zero although both factors were not.
#define NWGRAD_SP_FLUSHED(prod, a, b) (((prod) == SF(0) && (a) > SF(0) && (b) > SF(0)) ? SF(1) : SF(0))
// acc += a * b for the gap sums.  With a fused multiply-add the product is exact inside the
// operation and cannot flush (only the sum rounds), so FMA targets accumulate that way and
// count nothing; the others take the plain product and count its flush into lc.
// Double counts nothing anywhere: its terms are ~2^-700, and the tests cost it 1.2-1.9x
// (sse4, 2026-11-01).  kSpCount: the float32 instantiation only.
// NWGRAD_SP_NOFMA: a timing probe (the plain product and its count on an FMA target).
// kSpCount names the pass's counting flag at each use (the gradient pass runs without the
// counters first; see sp_loss_products_max).
#if (defined(__FMA__) || defined(__ARM_FEATURE_FMA)) && !defined(NWGRAD_SP_NOFMA)
#  define NWGRAD_SP_MAC(acc, a, b) do { if constexpr (kSpCountF) (acc) = std::fma((a), (b), (acc)); else (acc) += (a) * (b); } while (0)
#  define NWGRAD_SP_MAC_COUNTS 0
#else
#  define NWGRAD_SP_MAC(acc, a, b) do { if constexpr (kSpCount) { const SF p_ = (a) * (b); (acc) += p_; lc += NWGRAD_SP_FLUSHED(p_, (a), (b)); } else (acc) += (a) * (b); } while (0)
#  define NWGRAD_SP_MAC_COUNTS 1
#endif
static inline double sp_post(int e, double izr, bool& bad) noexcept {
    const double f = std::ldexp(izr, e);
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
// Z = 2^ze·zr from per-row sums (Local) or the (m, n) cell (Global), zr normalized
// into [1, 2).  Returns false if Z is not positive and finite (handled by caller).
template <class SF>
static inline bool sp_log_z(SoftPairJobT<SF>& J, const int* S, const SF* rowsum_or_null,
                            SF corner, int& ze, double& izr) noexcept {
    double zr;
    if (!J.local) { ze = S[J.m]; zr = corner; }
    else {
        ze = S[0];
        for (int i = 1; i <= J.m; ++i) ze = std::max(ze, S[i]);
        zr = 0.0;
        for (int i = 0; i <= J.m; ++i) zr += std::ldexp(rowsum_or_null[i], S[i] - ze);
    }
    if (!(zr > 0.0) || !(zr <= std::numeric_limits<double>::max())) return false;
    const int kz = std::ilogb(zr);
    zr = std::ldexp(zr, -kz); ze += kz;
    izr = 1.0 / zr;
    J.log_z = std::log(zr) + ze * 0.69314718055994530942;
    return true;
}

// Tr: tracked (J.rw) — each row's weights, its own profile and count row, and the gap
// gradient per row into J.grow.
template <class SF, bool Tr>
NWGRAD_SP_FMA_FN static void soft_pair_linear_t(SoftPairJobT<SF>& J) noexcept {
    NWGRAD_SP_FMA_BODY
    constexpr bool kSpCountF = std::is_same_v<SF, float> && NWGRAD_SP_COUNT_F32;   // the float32 instantiation counts flushed products (sp_loss_products)
    const int m = J.m, n = J.n;
    const bool L = J.local != 0;
    const int kmin = L ? 1 : 0;
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

    if (!J.full)
        for (int i = 0; i <= m; ++i) {
            SF* r = F + static_cast<size_t>(i) * st;
            for (int j = J.slo[i]; j <= J.shi[i]; ++j) r[j] = 0.0;
        }
    if (L) { for (int j = 0; j <= n; ++j) F[j] = 1.0; }
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
        if (L) { SF s = 0.0; for (int j = J.jlo0[0]; j <= J.jhi0[0]; ++j) s += F[j]; rs[0] = s; }
    }
    for (int i = 1; i <= m; ++i) {
        SF* __restrict r = F + static_cast<size_t>(i) * st;
        const SF* __restrict p = r - st;
        if constexpr (Tr) { ea = RW[4 * i + 1]; eb = RW[4 * i + 3]; }
        SF fr = 0.0;
        if (L) { fr = std::ldexp(SF(1), -S[i - 1]); r[0] = fr; }
        else   r[0] = (i <= J.bi) ? p[0] * eb : 0.0;
        const SF* __restrict pr = P + sp_row<Tr>(J, i) * w;
        const int lo = J.jlo[i], hi = J.jhi[i];
        for (int j = lo; j <= hi; ++j) r[j] = p[j - 1] * pr[j] + p[j] * eb + fr;
        if (lo <= hi) sp_carry_fwd(r, r, lo, hi, ea);
        SF mx = 0.0;
        NWGRAD_SP_MAX(mx)
        for (int j = J.slo[i]; j <= J.shi[i]; ++j) mx = NWGRAD_SP_VMAX(mx, r[j]);
        S[i] = S[i - 1] + sp_lazy(mx, J.slo[i], J.shi[i], bad, r);
        if (bad) return;
        mf[i] = mx;
        if (L) {
            SF s = 0.0;
            NWGRAD_SP_SUM(s)
            for (int j = J.jlo0[i]; j <= J.jhi0[i]; ++j) s += r[j];
            rs[i] = s;
        }
    }
    const bool mn_in = n >= J.jlo0[m] && n <= J.jhi0[m];
    const SF corner = mn_in ? F[static_cast<size_t>(m) * st + n] : 0.0;
    int ze; double izr;
    if (!sp_log_z(J, S, rs, corner, ze, izr)) {
        if (!L && !mn_in) {   // Global, (m, n) outside the band: log Z = -inf, genuinely
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
    double g_ea = 0.0, g_eb = 0.0, loss = 0.0;
    for (int i = m; i >= 0; --i) {
        const int lo0 = J.jlo0[i], hi0 = J.jhi0[i];
        for (int j = clo; j <= chi; ++j) cur[j] = 0.0;
        const SF init = L ? std::ldexp(SF(1), -Tn) : 0.0;
        SF ebn = eb;   // the move down into row i+1 (Tr: that row's)
        if constexpr (Tr) { ea = RW[4 * i + 1]; eb = RW[4 * i + 3]; if (i < m) ebn = RW[4 * (i + 1) + 3]; }
        if (i < m) {
            const SF* __restrict pn = P + sp_row<Tr>(J, i + 1) * w;
            SF* __restrict c = cur;
            const SF* __restrict nx = nxt;
            for (int j = lo0; j <= hi0; ++j) c[j] = pn[j + 1] * nx[j + 1] + nx[j] * ebn + init;
        } else {
            for (int j = lo0; j <= hi0; ++j) cur[j] = init;
            if (!L && n >= lo0 && n <= hi0) cur[n] = 1.0;
        }
        if (lo0 <= hi0) sp_carry_bwd(cur, lo0, hi0, ea);
        clo = lo0; chi = hi0;

        // One vector pass: row max and every gradient sum, on the pre-rescale values.
        const SF* __restrict Fi = F + static_cast<size_t>(i) * st;
        const SF* __restrict cb = cur;
        const int j1 = std::max(1, lo0);
        SF mx = 0.0, sa = 0.0, sb = 0.0;
        double lossp = 0.0;   // the flushed products' term (sp_loss_products), float32 only
        const double gh0 = sp_post(S[i] + Tn - ze, izr, bad);   // the pre-rescale units' posterior factor
        if (lo0 == 0) { mx = cb[0]; if (i >= 1 && kmin == 0) sb += (Fi - st)[0] * cb[0]; }
        if (i >= 1) {
            const SF* __restrict Fp = Fi - st;
            const SF* __restrict pr = P + sp_row<Tr>(J, i) * w;
            SF* __restrict t = tmp;
            const SF mx0 = mx, sa0 = sa, sb0 = sb;
            SF lb = 0.0, lc = 0.0;
            // The pass, with or without the counters (sp_loss_products_max): the same
            // values either way, so the counted rerun is a plain recomputation.
            auto pass = [&](auto cnt) {
                constexpr bool kSpCount = kSpCountF && decltype(cnt)::value;
                SF mx_ = mx0, sa_ = sa0, sb_ = sb0, lb_ = 0.0, lc_ = 0.0;
                {
                SF& lc = lc_;   // NWGRAD_SP_MAC counts into `lc`
#if NWGRAD_SP_SPLIT_MAX
                for (int j = j1; j <= hi0; ++j) mx_ = NWGRAD_SP_VMAX(mx_, cb[j]);
                NWGRAD_SP_PRAGMA_(omp simd reduction(+:sa_, sb_, lb_, lc))
#else
                NWGRAD_SP_PRAGMA_(omp simd reduction(+:sa_, sb_, lb_, lc) reduction(max:mx_))
#endif
                for (int j = j1; j <= hi0; ++j) {
                    const SF b = cb[j];
#if !NWGRAD_SP_SPLIT_MAX
                    mx_ = NWGRAD_SP_VMAX(mx_, b);
#endif
                    NWGRAD_SP_MAC(sa_, Fi[j - 1], b);
                    NWGRAD_SP_MAC(sb_, Fp[j], b);
                    const SF fp = Fp[j - 1], t1 = fp * pr[j], tt = t1 * b;
                    t[j] = tt;
                    if constexpr (kSpCount) {   // flushed products (sp_loss_products)
                        lb_ += (fp > SF(0) && t1 == SF(0)) ? b : SF(0);
                        lc += NWGRAD_SP_FLUSHED(tt, t1, b);
                    }
                }
                }
                mx = mx_; sa = sa_; sb = sb_; lb = lb_; lc = lc_;
            };
            auto pass_counted = [&]() __attribute__((noinline)) { pass(std::true_type{}); };   // out of line (see inter_soft_impl.inl)
            pass(std::false_type{});
            if constexpr (kSpCountF) {
                const double gv0 = sp_post(S[i - 1] + Tn - ze, izr, bad);
                lossp = sp_loss_products_max<SF>(hi0 - j1 + 1, 1 + 2 * NWGRAD_SP_MAC_COUNTS, mx, gh0, gv0);
#ifdef NWGRAD_SP_RERUN_STATS
                { static std::atomic<long> rows_{0}, reruns_{0}; const long r = ++rows_; if (!(loss + lossp <= kSpCountBudget * SoftRange<SF>::tol)) ++reruns_;
                  if (r % 100000 == 0) std::fprintf(stderr, "pair rows %ld reruns %ld\n", r, reruns_.load()); }
#endif
                if (!(loss + lossp <= kSpCountBudget * SoftRange<SF>::tol)) {   // near rejection: count
                    pass_counted();
                    lossp = sp_loss_products<SF>(lb, lc, gh0, gv0);
                }
            }
        } else {
            constexpr bool kSpCount = kSpCountF;   // row 0: counted always (one row)
            SF lc = 0.0;
#if NWGRAD_SP_SPLIT_MAX
            for (int j = j1; j <= hi0; ++j) mx = NWGRAD_SP_VMAX(mx, cb[j]);
            NWGRAD_SP_SUM(sa, lc)
#else
            NWGRAD_SP_PRAGMA_(omp simd reduction(+:sa, lc) reduction(max:mx))
#endif
            for (int j = j1; j <= hi0; ++j) {
                const SF b = cb[j];
#if !NWGRAD_SP_SPLIT_MAX
                mx = NWGRAD_SP_VMAX(mx, b);
#endif
                NWGRAD_SP_MAC(sa, Fi[j - 1], b);
            }
            if constexpr (kSpCountF) lossp = sp_loss_products<SF>(SF(0), lc, gh0, 0.0);
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
        loss += sp_loss(kSpTermsLinear, n, i > 0 ? S[i] - S[i - 1] : 0, kb, mf[i], mx, gh);
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
    J.loss = loss;
    if (!(loss <= SoftRange<SF>::tol)) return;
    J.g_oa = 0.0; J.g_ea = -g_ea; J.g_ob = 0.0; J.g_eb = -g_eb;
    J.ok = 1;
}

template <class SF, bool Tr>
NWGRAD_SP_FMA_FN static void soft_pair_affine_t(SoftPairJobT<SF>& J) noexcept {
    NWGRAD_SP_FMA_BODY
    constexpr bool kSpCountF = std::is_same_v<SF, float> && NWGRAD_SP_COUNT_F32;   // the float32 instantiation counts flushed products (sp_loss_products)
    const int m = J.m, n = J.n;
    const bool L = J.local != 0;
    const int kmin = L ? 1 : 0;
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
    SF* tp = J.r5;   // M+X+Y of the previous forward row; later the match scratch
    SF* uy = J.r4;   // forward: the Y carry's carry-free input

    if (!J.full) {
        for (int i = 0; i <= m; ++i) {
            const size_t ro = static_cast<size_t>(i) * st;
            for (int j = J.slo[i]; j <= J.shi[i]; ++j) { FM[ro + j] = 0.0; FX[ro + j] = 0.0; FY[ro + j] = 0.0; }
        }
    } else {
        for (int j = 0; j <= n; ++j) { FM[j] = 0.0; FX[j] = 0.0; FY[j] = 0.0; }
    }
    if (L) { for (int j = 0; j <= n; ++j) FM[j] = 1.0; }
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
    if (L) { SF s = 0.0; for (int j = J.jlo0[0]; j <= J.jhi0[0]; ++j) s += tp[j]; rs[0] = s; }
    for (int i = 1; i <= m; ++i) {
        const size_t ro = static_cast<size_t>(i) * st;
        SF* __restrict rM = FM + ro; SF* __restrict rX = FX + ro;
        SF* __restrict rY = FY + ro;
        const SF* __restrict pM = rM - st; const SF* __restrict pX = rX - st;
        const SF* __restrict pY = rY - st;
        if constexpr (Tr) { oa = RW[4 * i]; ea = RW[4 * i + 1]; ob = RW[4 * i + 2]; eb = RW[4 * i + 3]; }
        SF fr = 0.0;
        if (L) { fr = std::ldexp(SF(1), -S[i - 1]); rM[0] = fr; rX[0] = 0.0; rY[0] = 0.0; }
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
        if (bad) return;
        mf[i] = mx;
        if (L) {
            SF s = 0.0;
            NWGRAD_SP_SUM(s)
            for (int j = J.jlo0[i]; j <= J.jhi0[i]; ++j) s += tp[j];
            rs[i] = s;
        }
    }
    const bool mn_in = n >= J.jlo0[m] && n <= J.jhi0[m];
    const size_t cmn = static_cast<size_t>(m) * st + n;
    const SF corner = mn_in ? FM[cmn] + FX[cmn] + FY[cmn] : 0.0;
    int ze; double izr;
    if (!sp_log_z(J, S, rs, corner, ze, izr)) {
        if (!L && !mn_in) {
            J.log_z = -std::numeric_limits<SF>::infinity();
            J.g_oa = J.g_ea = J.g_ob = J.g_eb = 0.0;
            J.ok = 1;
        }
        return;
    }

    SF *cM = J.r0, *cX = J.r1, *cY = J.r2, *nM = J.r3, *nX = J.r4;
    SF* tmp = tp;
    for (size_t j = 0; j < w; ++j) { cM[j] = cX[j] = cY[j] = nM[j] = nX[j] = 0.0; }
    int clo = 0, chi = n, nlo = 0, nhi = n;
    int Tn = 0;
    double g_ea = 0.0, g_eb = 0.0, g_oa = 0.0, g_ob = 0.0, loss = 0.0;
    for (int i = m; i >= 0; --i) {
        const int lo0 = J.jlo0[i], hi0 = J.jhi0[i];
        for (int j = clo; j <= chi; ++j) { cM[j] = 0.0; cX[j] = 0.0; }
        // cY is not swapped (only M and X of the next row are read): it still holds
        // row i+1 over [nlo, nhi].
        for (int j = nlo; j <= nhi; ++j) cY[j] = 0.0;
        const SF init = L ? std::ldexp(SF(1), -Tn) : 0.0;
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
                for (int j = lo0; j <= hi0; ++j) { b1[j] = init; b2[j] = init; y[j] = init; }
                if (!L && n >= lo0 && n <= hi0) { b1[n] = 1.0; b2[n] = 1.0; y[n] = 1.0; }
            }
        }
        if (lo0 <= hi0) sp_carry_bwd(cY, lo0, hi0, ea);
        clo = lo0; chi = hi0;

        // One vector pass: final M/X, the row max, every gradient sum and the match
        // posteriors, all on the pre-rescale values.
        const size_t ro = static_cast<size_t>(i) * st;
        const SF* __restrict fM = FM + ro; const SF* __restrict fX = FX + ro;
        const SF* __restrict fY = FY + ro;
        SF* __restrict bM = cM; SF* __restrict bX = cX;
        const SF* __restrict bY = cY;
        const int j1 = std::max(1, lo0);
        SF mx = 0.0, sx = 0.0, sy = 0.0, so = 0.0, sob = 0.0;
        double lossp = 0.0;   // the flushed products' term (sp_loss_products), float32 only
        const double gh0 = sp_post(S[i] + Tn - ze, izr, bad);   // the pre-rescale units' posterior factor
        if (lo0 == 0) {
            const SF h = bY[1] * oa;
            const SF mv = bM[0] + h, xv = bX[0] + h;
            bM[0] = mv; bX[0] = xv;
            mx = mv + xv + bY[0];
            if (i >= 1) {
                sx += fX[0] * xv;
                if (kmin == 0) sob += ((fM - st)[0] + (fY - st)[0]) * xv;
            }
        }
        if (i >= 1) {
            const SF* __restrict qM = fM - st; const SF* __restrict qX = fX - st;
            const SF* __restrict qY = fY - st;
            const SF* __restrict pr = P + sp_row<Tr>(J, i) * w;
            SF* __restrict t = tmp;
            const SF mx0 = mx, sx0 = sx, sy0 = sy, so0 = so, sob0 = sob;
            SF lb = 0.0, lc = 0.0;
            // The pass, with or without the counters (sp_loss_products_max).  The first
            // pass finishes the backward row's M / X in place (the Y-open term h); the
            // counted rerun reads them finished and recomputes everything else.
            auto pass = [&](auto cnt) {
                constexpr bool kSpCount = kSpCountF && decltype(cnt)::value;
                SF mx_ = mx0, sx_ = sx0, sy_ = sy0, so_ = so0, sob_ = sob0, lb_ = 0.0, lc_ = 0.0;
                {
                SF& lc = lc_;   // NWGRAD_SP_MAC counts into `lc`
#if NWGRAD_SP_SPLIT_MAX
                NWGRAD_SP_SUM(sx_, sy_, so_, sob_, lb_, lc)
#else
                NWGRAD_SP_PRAGMA_(omp simd reduction(+:sx_, sy_, so_, sob_, lb_, lc) reduction(max:mx_))
#endif
                for (int j = j1; j <= hi0; ++j) {
                    const SF y = bY[j];
                    SF mv, xv;
                    if constexpr (!decltype(cnt)::value) {
                        const SF h = bY[j + 1] * oa;
                        mv = bM[j] + h; xv = bX[j] + h;
                        bM[j] = mv; bX[j] = xv;
                    } else { mv = bM[j]; xv = bX[j]; }
#if !NWGRAD_SP_SPLIT_MAX
                    const SF tot = mv + xv + y;
                    mx_ = NWGRAD_SP_VMAX(mx_, tot);
#endif
                    NWGRAD_SP_MAC(sx_, fX[j], xv);
                    NWGRAD_SP_MAC(sy_, fY[j], y);
                    NWGRAD_SP_MAC(so_, fM[j - 1] + fX[j - 1], y);
                    NWGRAD_SP_MAC(sob_, qM[j] + qY[j], xv);
                    const SF qs = qM[j - 1] + qX[j - 1] + qY[j - 1], t1 = qs * pr[j], tt = t1 * mv;
                    t[j] = tt;
                    if constexpr (kSpCount) {   // flushed products (sp_loss_products)
                        lb_ += (qs > SF(0) && t1 == SF(0)) ? mv : SF(0);
                        lc += NWGRAD_SP_FLUSHED(tt, t1, mv);
                    }
                }
#if NWGRAD_SP_SPLIT_MAX
                for (int j = j1; j <= hi0; ++j) mx_ = NWGRAD_SP_VMAX(mx_, bM[j] + bX[j] + bY[j]);
#endif
                }
                mx = mx_; sx = sx_; sy = sy_; so = so_; sob = sob_; lb = lb_; lc = lc_;
            };
            auto pass_counted = [&]() __attribute__((noinline)) { pass(std::true_type{}); };   // out of line (see inter_soft_impl.inl)
            pass(std::false_type{});
            if constexpr (kSpCountF) {
                const double gv0 = sp_post(S[i - 1] + Tn - ze, izr, bad);
                lossp = sp_loss_products_max<SF>(hi0 - j1 + 1, 1 + 4 * NWGRAD_SP_MAC_COUNTS, mx, gh0, gv0);
#ifdef NWGRAD_SP_RERUN_STATS
                { static std::atomic<long> rows_{0}, reruns_{0}; const long r = ++rows_; if (!(loss + lossp <= kSpCountBudget * SoftRange<SF>::tol)) ++reruns_;
                  if (r % 100000 == 0) std::fprintf(stderr, "pair rows %ld reruns %ld\n", r, reruns_.load()); }
#endif
                if (!(loss + lossp <= kSpCountBudget * SoftRange<SF>::tol)) {   // near rejection: count
                    pass_counted();
                    lossp = sp_loss_products<SF>(lb, lc, gh0, gv0);
                }
            }
        } else {
            constexpr bool kSpCount = kSpCountF;   // row 0: counted always (one row)
            SF lc = 0.0;
#if NWGRAD_SP_SPLIT_MAX
            NWGRAD_SP_SUM(sy, so, lc)
#else
            NWGRAD_SP_PRAGMA_(omp simd reduction(+:sy, so, lc) reduction(max:mx))
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
            if constexpr (kSpCountF) lossp = sp_loss_products<SF>(SF(0), lc, gh0, 0.0);
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
        loss += sp_loss(kSpTermsAffine, n, i > 0 ? S[i] - S[i - 1] : 0, kb, mf[i], mx, gh);
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
    J.loss = loss;
    if (!(loss <= SoftRange<SF>::tol)) return;
    J.g_oa = -g_oa; J.g_ea = -g_ea; J.g_ob = -g_ob; J.g_eb = -g_eb;
    J.ok = 1;
}

NWGRAD_SP_FMA_FN static void soft_pair_linear(SoftPairJob& J) noexcept {
    if (J.rw) soft_pair_linear_t<double, true>(J); else soft_pair_linear_t<double, false>(J);
}
NWGRAD_SP_FMA_FN static void soft_pair_affine(SoftPairJob& J) noexcept {
    if (J.rw) soft_pair_affine_t<double, true>(J); else soft_pair_affine_t<double, false>(J);
}
NWGRAD_SP_FMA_FN static void soft_pair_linear_f(SoftPairJobF& J) noexcept {
    if (J.rw) soft_pair_linear_t<float, true>(J); else soft_pair_linear_t<float, false>(J);
}
NWGRAD_SP_FMA_FN static void soft_pair_affine_f(SoftPairJobF& J) noexcept {
    if (J.rw) soft_pair_affine_t<float, true>(J); else soft_pair_affine_t<float, false>(J);
}
// The entry of soft type SF (the Aligner's baseline copy picks by type).
template <class SF>
static inline auto soft_pair_entry(bool linear) noexcept {
    if constexpr (std::is_same_v<SF, float>) return linear ? &soft_pair_linear_f : &soft_pair_affine_f;
    else                                     return linear ? &soft_pair_linear : &soft_pair_affine;
}
