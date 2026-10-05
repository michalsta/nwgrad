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
#  elif defined(__GNUC__)
#    define NWGRAD_SP_FMA_FN __attribute__((optimize("fp-contract=fast")))
#    define NWGRAD_SP_FMA_BODY
#  else
#    define NWGRAD_SP_FMA_FN
#    define NWGRAD_SP_FMA_BODY
#  endif
#  define NWGRAD_SP_PRAGMA_(x) _Pragma(#x)
#  define NWGRAD_SP_SUM(...) NWGRAD_SP_PRAGMA_(omp simd reduction(+:__VA_ARGS__))
#  define NWGRAD_SP_MAX(v) NWGRAD_SP_PRAGMA_(omp simd reduction(max:v))
#endif

static constexpr double kSpLossTol = 0x1p-45;

// y[j] += e·y[j-1], j = lo..hi, two cells per carry step.
static inline void sp_carry_fwd(double* y, int lo, int hi, double e) noexcept {
    double c = y[lo - 1];
    const double e2 = e * e;
    int j = lo;
    for (; j < hi; j += 2) {
        const double u1 = y[j], u2 = y[j + 1];
        y[j] = u1 + e * c;
        c = (u2 + e * u1) + e2 * c;
        y[j + 1] = c;
    }
    if (j == hi) y[j] = y[j] + e * c;
}
// y[j] += e·y[j+1], j = hi..lo.
static inline void sp_carry_bwd(double* y, int lo, int hi, double e) noexcept {
    double c = y[hi + 1];
    const double e2 = e * e;
    int j = hi;
    for (; j > lo; j -= 2) {
        const double u1 = y[j], u2 = y[j - 1];
        y[j] = u1 + e * c;
        c = (u2 + e * u1) + e2 * c;
        y[j - 1] = c;
    }
    if (j == lo) y[j] = y[j] + e * c;
}

// Rescale [lo, hi] of one or three rows so the total's max lands in [1, 2); totals to
// tot if given.  Returns the exponent removed; sets bad on overflow.
static inline int sp_rescale(double* __restrict r0, double* __restrict r1, double* __restrict r2,
                             double* __restrict tot, int lo, int hi, bool& bad) noexcept {
    double mx = 0.0;
    if (r1 && tot) {
        NWGRAD_SP_MAX(mx)
        for (int j = lo; j <= hi; ++j) {
            const double t = r0[j] + r1[j] + r2[j];
            tot[j] = t;
            mx = t > mx ? t : mx;
        }
    } else if (r1) {
        NWGRAD_SP_MAX(mx)
        for (int j = lo; j <= hi; ++j) {
            const double t = r0[j] + r1[j] + r2[j];
            mx = t > mx ? t : mx;
        }
    } else {
        NWGRAD_SP_MAX(mx)
        for (int j = lo; j <= hi; ++j) mx = r0[j] > mx ? r0[j] : mx;
    }
    if (!(mx <= std::numeric_limits<double>::max())) { bad = true; return 0; }
    if (mx == 0.0) return 0;
    const int k = std::ilogb(mx);
    if (k == 0) return 0;
    const double sc = std::ldexp(1.0, -k);
    for (int j = lo; j <= hi; ++j) r0[j] *= sc;
    if (r1) for (int j = lo; j <= hi; ++j) { r1[j] *= sc; r2[j] *= sc; }
    if (tot) for (int j = lo; j <= hi; ++j) tot[j] *= sc;
    return k;
}

static inline double sp_post(int e, double izr, bool& bad) noexcept {
    const double f = std::ldexp(izr, e);
    if (!(f <= std::numeric_limits<double>::max())) bad = true;
    return f;
}
static inline double sp_row_loss(int n, int kf, int kb, double g) noexcept {
    const double lf = kf < 0 ? std::ldexp(1.0, -kf) : 1.0;
    const double lb = kb < 0 ? std::ldexp(1.0, -kb) : 1.0;
    return 32.0 * (n + 1) * std::numeric_limits<double>::min() * (lf + lb) * g;
}
// Match posteriors tmp[j] of row i into the count row of a[i-1], times g.
static inline void sp_scatter(const SoftPairJob& J, const double* tmp, int i, int lo, int hi,
                              double g) noexcept {
    const int na = J.nalpha;
    double* srow = J.srow;
    for (int c = 0; c < na; ++c) srow[c] = 0.0;
    for (int j = lo; j <= hi; ++j) srow[J.b[j - 1]] += tmp[j];
    double* gr = J.scnt + static_cast<size_t>(J.a[i - 1]) * na;
    for (int c = 0; c < na; ++c) gr[c] += srow[c] * g;
}
// Z = 2^ze·zr from per-row sums (Local) or the (m, n) cell (Global), zr normalized
// into [1, 2).  Returns false if Z is not positive and finite (handled by caller).
static inline bool sp_log_z(SoftPairJob& J, const int* S, const double* rowsum_or_null,
                            double corner, int& ze, double& izr) noexcept {
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

NWGRAD_SP_FMA_FN static void soft_pair_linear(SoftPairJob& J) noexcept {
    NWGRAD_SP_FMA_BODY
    const int m = J.m, n = J.n, na = J.nalpha;
    const bool L = J.local != 0;
    const int kmin = L ? 1 : 0;
    const double ea = J.ea, eb = J.eb;
    const size_t st = J.stride, w = static_cast<size_t>(n) + 2;
    double* F = J.FM;
    int* S = J.S;
    double* rs = J.rowsum;
    const double* P = J.P;
    bool bad = false;
    J.ok = 0;

    if (!J.full)
        for (int i = 0; i <= m; ++i) {
            double* r = F + static_cast<size_t>(i) * st;
            for (int j = J.slo[i]; j <= J.shi[i]; ++j) r[j] = 0.0;
        }
    if (L) { for (int j = 0; j <= n; ++j) F[j] = 1.0; }
    else {
        if (J.full) for (int j = 0; j <= n; ++j) F[j] = 0.0;
        F[0] = 1.0;
        for (int j = 1; j <= J.bj; ++j) F[j] = F[j - 1] * ea;
    }
    S[0] = 0;
    if (L) { double s = 0.0; for (int j = J.jlo0[0]; j <= J.jhi0[0]; ++j) s += F[j]; rs[0] = s; }
    for (int i = 1; i <= m; ++i) {
        double* __restrict r = F + static_cast<size_t>(i) * st;
        const double* __restrict p = r - st;
        double fr = 0.0;
        if (L) { fr = std::ldexp(1.0, -S[i - 1]); r[0] = fr; }
        else   r[0] = (i <= J.bi) ? p[0] * eb : 0.0;
        const double* __restrict pr = P + static_cast<size_t>(J.a[i - 1]) * w;
        const int lo = J.jlo[i], hi = J.jhi[i];
        for (int j = lo; j <= hi; ++j) r[j] = p[j - 1] * pr[j] + p[j] * eb + fr;
        if (lo <= hi) sp_carry_fwd(r, lo, hi, ea);
        S[i] = S[i - 1] + sp_rescale(r, nullptr, nullptr, nullptr, J.slo[i], J.shi[i], bad);
        if (bad) return;
        if (L) {
            double s = 0.0;
            NWGRAD_SP_SUM(s)
            for (int j = J.jlo0[i]; j <= J.jhi0[i]; ++j) s += r[j];
            rs[i] = s;
        }
    }
    const bool mn_in = n >= J.jlo0[m] && n <= J.jhi0[m];
    const double corner = mn_in ? F[static_cast<size_t>(m) * st + n] : 0.0;
    int ze; double izr;
    if (!sp_log_z(J, S, rs, corner, ze, izr)) {
        // Global with (m, n) outside the band: log Z = -inf and zero counts, genuinely.
        if (!L && !mn_in) {
            J.log_z = -std::numeric_limits<double>::infinity();
            J.g_oa = J.g_ea = J.g_ob = J.g_eb = 0.0;
            J.ok = 1;
        }
        return;
    }

    double* cur = J.r0; double* nxt = J.r1; double* tmp = J.r2;
    for (size_t j = 0; j < w; ++j) { cur[j] = 0.0; nxt[j] = 0.0; }
    int clo = 0, chi = n, nlo = 0, nhi = n;
    int Tn = 0;
    double g_ea = 0.0, g_eb = 0.0, loss = 0.0;
    for (int i = m; i >= 0; --i) {
        const int lo0 = J.jlo0[i], hi0 = J.jhi0[i];
        for (int j = clo; j <= chi; ++j) cur[j] = 0.0;
        const double init = L ? std::ldexp(1.0, -Tn) : 0.0;
        if (i < m) {
            const double* __restrict pn = P + static_cast<size_t>(J.a[i]) * w;
            double* __restrict c = cur;
            const double* __restrict nx = nxt;
            for (int j = lo0; j <= hi0; ++j) c[j] = pn[j + 1] * nx[j + 1] + nx[j] * eb + init;
        } else {
            for (int j = lo0; j <= hi0; ++j) cur[j] = init;
            if (!L && n >= lo0 && n <= hi0) cur[n] = 1.0;
        }
        if (lo0 <= hi0) sp_carry_bwd(cur, lo0, hi0, ea);
        clo = lo0; chi = hi0;
        const int Ti = Tn + sp_rescale(cur, nullptr, nullptr, nullptr, lo0, hi0, bad);
        if (bad) return;

        const double* __restrict Fi = F + static_cast<size_t>(i) * st;
        const double* __restrict cb = cur;
        const double gh = sp_post(S[i] + Ti - ze, izr, bad);
        loss += sp_row_loss(n, i > 0 ? S[i] - S[i - 1] : 0, Ti - Tn, gh);
        if (i >= kmin) {
            double s = 0.0;
            NWGRAD_SP_SUM(s)
            for (int j = std::max(1, lo0); j <= hi0; ++j) s += Fi[j - 1] * cb[j];
            g_ea += s * ea * gh;
        }
        if (i >= 1) {
            const double gv = sp_post(S[i - 1] + Ti - ze, izr, bad);
            const double* __restrict Fp = Fi - st;
            double s = 0.0;
            NWGRAD_SP_SUM(s)
            for (int j = std::max(kmin, lo0); j <= hi0; ++j) s += Fp[j] * cb[j];
            g_eb += s * eb * gv;
            const double* __restrict pr = P + static_cast<size_t>(J.a[i - 1]) * w;
            double* __restrict t = tmp;
            const int lo = J.jlo[i], hi = J.jhi[i];
            for (int j = lo; j <= hi; ++j) t[j] = Fp[j - 1] * pr[j] * cb[j];
            sp_scatter(J, tmp, i, lo, hi, gv);
        }
        if (bad) return;
        Tn = Ti;
        std::swap(cur, nxt); std::swap(clo, nlo); std::swap(chi, nhi);
    }
    if (!(loss <= kSpLossTol)) return;
    J.g_oa = 0.0; J.g_ea = -g_ea; J.g_ob = 0.0; J.g_eb = -g_eb;
    J.ok = 1;
    (void)na;
}

NWGRAD_SP_FMA_FN static void soft_pair_affine(SoftPairJob& J) noexcept {
    NWGRAD_SP_FMA_BODY
    const int m = J.m, n = J.n;
    const bool L = J.local != 0;
    const int kmin = L ? 1 : 0;
    const double oa = J.oa, ea = J.ea, ob = J.ob, eb = J.eb;
    const size_t st = J.stride, w = static_cast<size_t>(n) + 2;
    double* FM = J.FM; double* FX = J.FX; double* FY = J.FY;
    int* S = J.S;
    double* rs = J.rowsum;
    const double* P = J.P;
    bool bad = false;
    J.ok = 0;
    double* tp = J.r5;   // M+X+Y of the previous forward row; later the match scratch

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
    for (int j = J.slo[0]; j <= J.shi[0]; ++j) tp[j] = FM[j] + FX[j] + FY[j];
    S[0] = 0;
    if (L) { double s = 0.0; for (int j = J.jlo0[0]; j <= J.jhi0[0]; ++j) s += tp[j]; rs[0] = s; }
    for (int i = 1; i <= m; ++i) {
        const size_t ro = static_cast<size_t>(i) * st;
        double* __restrict rM = FM + ro; double* __restrict rX = FX + ro;
        double* __restrict rY = FY + ro;
        const double* __restrict pM = rM - st; const double* __restrict pX = rX - st;
        const double* __restrict pY = rY - st;
        double fr = 0.0;
        if (L) { fr = std::ldexp(1.0, -S[i - 1]); rM[0] = fr; rX[0] = 0.0; rY[0] = 0.0; }
        else {
            rM[0] = 0.0; rY[0] = 0.0;
            rX[0] = (i <= J.bi) ? (pM[0] + pY[0]) * ob + pX[0] * eb : 0.0;
        }
        const double* __restrict pr = P + static_cast<size_t>(J.a[i - 1]) * w;
        const double* __restrict tq = tp;
        const int lo = J.jlo[i], hi = J.jhi[i];
        for (int j = lo; j <= hi; ++j) {
            rM[j] = tq[j - 1] * pr[j] + fr;
            rX[j] = (pM[j] + pY[j]) * ob + pX[j] * eb;
        }
        for (int j = lo; j <= hi; ++j) rY[j] = (rM[j - 1] + rX[j - 1]) * oa;
        if (lo <= hi) sp_carry_fwd(rY, lo, hi, ea);
        S[i] = S[i - 1] + sp_rescale(rM, rX, rY, tp, J.slo[i], J.shi[i], bad);
        if (bad) return;
        if (L) {
            double s = 0.0;
            NWGRAD_SP_SUM(s)
            for (int j = J.jlo0[i]; j <= J.jhi0[i]; ++j) s += tp[j];
            rs[i] = s;
        }
    }
    const bool mn_in = n >= J.jlo0[m] && n <= J.jhi0[m];
    const size_t cmn = static_cast<size_t>(m) * st + n;
    const double corner = mn_in ? FM[cmn] + FX[cmn] + FY[cmn] : 0.0;
    int ze; double izr;
    if (!sp_log_z(J, S, rs, corner, ze, izr)) {
        if (!L && !mn_in) {
            J.log_z = -std::numeric_limits<double>::infinity();
            J.g_oa = J.g_ea = J.g_ob = J.g_eb = 0.0;
            J.ok = 1;
        }
        return;
    }

    double *cM = J.r0, *cX = J.r1, *cY = J.r2, *nM = J.r3, *nX = J.r4;
    double* tmp = tp;
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
        const double init = L ? std::ldexp(1.0, -Tn) : 0.0;
        {
            double* __restrict b1 = cM; double* __restrict b2 = cX;
            double* __restrict y = cY;
            const double* __restrict nm = nM; const double* __restrict nx = nX;
            if (i < m) {
                const double* __restrict pn = P + static_cast<size_t>(J.a[i]) * w;
                for (int j = lo0; j <= hi0; ++j) {
                    const double d = pn[j + 1] * nm[j + 1] + init, v = nx[j];
                    b1[j] = d + v * ob; b2[j] = d + v * eb; y[j] = b1[j];
                }
            } else {
                for (int j = lo0; j <= hi0; ++j) { b1[j] = init; b2[j] = init; y[j] = init; }
                if (!L && n >= lo0 && n <= hi0) { b1[n] = 1.0; b2[n] = 1.0; y[n] = 1.0; }
            }
        }
        if (lo0 <= hi0) sp_carry_bwd(cY, lo0, hi0, ea);
        {
            double* __restrict b1 = cM; double* __restrict b2 = cX;
            const double* __restrict y = cY;
            for (int j = lo0; j <= hi0; ++j) {
                const double h = y[j + 1] * oa;
                b1[j] += h; b2[j] += h;
            }
        }
        clo = lo0; chi = hi0;
        const int Ti = Tn + sp_rescale(cM, cX, cY, nullptr, lo0, hi0, bad);
        if (bad) return;

        const size_t ro = static_cast<size_t>(i) * st;
        const double* __restrict fM = FM + ro; const double* __restrict fX = FX + ro;
        const double* __restrict fY = FY + ro;
        const double* __restrict bM = cM; const double* __restrict bX = cX;
        const double* __restrict bY = cY;
        const double gh = sp_post(S[i] + Ti - ze, izr, bad);
        loss += sp_row_loss(n, i > 0 ? S[i] - S[i - 1] : 0, Ti - Tn, gh);
        {
            double sx = 0.0;
            if (i >= 1) {
                NWGRAD_SP_SUM(sx)
                for (int j = lo0; j <= hi0; ++j) sx += fX[j] * bX[j];
            }
            double sy = 0.0, so = 0.0;
            NWGRAD_SP_SUM(sy, so)
            for (int j = std::max(1, lo0); j <= hi0; ++j) {
                sy += fY[j] * bY[j];
                so += (fM[j - 1] + fX[j - 1]) * bY[j];
            }
            g_eb += sx * gh;
            g_ea += sy * gh;
            if (i >= kmin) g_oa += so * oa * gh;
        }
        if (i >= 1) {
            const double gv = sp_post(S[i - 1] + Ti - ze, izr, bad);
            const double* __restrict qM = fM - st; const double* __restrict qX = fX - st;
            const double* __restrict qY = fY - st;
            double s = 0.0;
            NWGRAD_SP_SUM(s)
            for (int j = std::max(kmin, lo0); j <= hi0; ++j) s += (qM[j] + qY[j]) * bX[j];
            g_ob += s * ob * gv;
            const double* __restrict pr = P + static_cast<size_t>(J.a[i - 1]) * w;
            double* __restrict t = tmp;
            const int lo = J.jlo[i], hi = J.jhi[i];
            for (int j = lo; j <= hi; ++j) t[j] = (qM[j - 1] + qX[j - 1] + qY[j - 1]) * pr[j] * bM[j];
            sp_scatter(J, tmp, i, lo, hi, gv);
        }
        if (bad) return;
        Tn = Ti;
        std::swap(cM, nM); std::swap(cX, nX);
        std::swap(clo, nlo); std::swap(chi, nhi);
    }
    if (!(loss <= kSpLossTol)) return;
    J.g_oa = -g_oa; J.g_ea = -g_ea; J.g_ob = -g_ob; J.g_eb = -g_eb;
    J.ok = 1;
}
