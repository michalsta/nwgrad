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

// y[j] = u[j] + e·y[j-1], j = lo..hi (y[lo-1] given; u may alias y).  FOUR cells per
// carry step: only y[j+3] = (u3 + e·u2 + e²·u1 + e³·u0) + e⁴·c sits on the serial
// chain (one FMA latency per four cells); y[j..j+2] hang off it.  Reassociated — the
// soft path is tolerance-tested.
static inline void sp_carry_fwd(double* y, const double* u, int lo, int hi, double e) noexcept {
    double c = y[lo - 1];
    const double e2 = e * e, e3 = e2 * e, e4 = e2 * e2;
    int j = lo;
    for (; j + 3 <= hi; j += 4) {
        const double u0 = u[j], u1 = u[j + 1], u2 = u[j + 2], u3 = u[j + 3];
        const double p1 = u1 + e * u0, p2 = u2 + e * p1, p3 = u3 + e * p2;
        y[j] = u0 + e * c;
        y[j + 1] = p1 + e2 * c;
        y[j + 2] = p2 + e3 * c;
        c = p3 + e4 * c;
        y[j + 3] = c;
    }
    for (; j <= hi; ++j) { c = u[j] + e * c; y[j] = c; }
}
// y[j] += e·y[j+1], j = hi..lo, four cells per carry step.
static inline void sp_carry_bwd(double* y, int lo, int hi, double e) noexcept {
    double c = y[hi + 1];
    const double e2 = e * e, e3 = e2 * e, e4 = e2 * e2;
    int j = hi;
    for (; j - 3 >= lo; j -= 4) {
        const double u0 = y[j], u1 = y[j - 1], u2 = y[j - 2], u3 = y[j - 3];
        const double p1 = u1 + e * u0, p2 = u2 + e * p1, p3 = u3 + e * p2;
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
static inline int sp_lazy(double& mx, int lo, int hi, bool& bad, double* r0, double* r1 = nullptr,
                          double* r2 = nullptr, double* r3 = nullptr) noexcept {
    if (!(mx <= std::numeric_limits<double>::max())) { bad = true; return 0; }
    if (mx == 0.0 || (mx <= 0x1p256 && mx >= 0x1p-256)) return 0;
    const int k = std::ilogb(mx);
    const double sc = std::ldexp(1.0, -k);
    for (double* r : {r0, r1, r2, r3})
        if (r) for (int j = lo; j <= hi; ++j) r[j] *= sc;
    mx *= sc;
    return k;
}
// Bound on the posterior damage of mass lost in one row pair (see AGENTS.md): lost terms
// are below DMIN in stored units (x2^-k where a rescale scaled UP), and act through the
// other direction's actual row max times the row's posterior factor g.
static inline double sp_loss(int n, int kf, int kb, double mxf, double mxb, double g) noexcept {
    const double lf = kf < 0 ? std::ldexp(1.0, -kf) : 1.0;
    const double lb = kb < 0 ? std::ldexp(1.0, -kb) : 1.0;
    return 32.0 * (n + 1) * std::numeric_limits<double>::min() * (lf * mxb + lb * mxf) * g;
}
static inline double sp_post(int e, double izr, bool& bad) noexcept {
    const double f = std::ldexp(izr, e);
    if (!(f <= std::numeric_limits<double>::max())) bad = true;
    return f;
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
    const int m = J.m, n = J.n;
    const bool L = J.local != 0;
    const int kmin = L ? 1 : 0;
    const double ea = J.ea, eb = J.eb;
    const size_t st = J.stride, w = static_cast<size_t>(n) + 2;
    double* F = J.FM;
    int* S = J.S;
    double* rs = J.rowsum;
    double* mf = J.fmax;
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
    {
        double mx = 0.0;
        for (int j = J.slo[0]; j <= J.shi[0]; ++j) mx = F[j] > mx ? F[j] : mx;
        mf[0] = mx;
        if (L) { double s = 0.0; for (int j = J.jlo0[0]; j <= J.jhi0[0]; ++j) s += F[j]; rs[0] = s; }
    }
    for (int i = 1; i <= m; ++i) {
        double* __restrict r = F + static_cast<size_t>(i) * st;
        const double* __restrict p = r - st;
        double fr = 0.0;
        if (L) { fr = std::ldexp(1.0, -S[i - 1]); r[0] = fr; }
        else   r[0] = (i <= J.bi) ? p[0] * eb : 0.0;
        const double* __restrict pr = P + static_cast<size_t>(J.a[i - 1]) * w;
        const int lo = J.jlo[i], hi = J.jhi[i];
        for (int j = lo; j <= hi; ++j) r[j] = p[j - 1] * pr[j] + p[j] * eb + fr;
        if (lo <= hi) sp_carry_fwd(r, r, lo, hi, ea);
        double mx = 0.0;
        NWGRAD_SP_MAX(mx)
        for (int j = J.slo[i]; j <= J.shi[i]; ++j) mx = r[j] > mx ? r[j] : mx;
        S[i] = S[i - 1] + sp_lazy(mx, J.slo[i], J.shi[i], bad, r);
        if (bad) return;
        mf[i] = mx;
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
        if (!L && !mn_in) {   // Global, (m, n) outside the band: log Z = -inf, genuinely
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

        // One vector pass: row max and every gradient sum, on the pre-rescale values.
        const double* __restrict Fi = F + static_cast<size_t>(i) * st;
        const double* __restrict cb = cur;
        const int j1 = std::max(1, lo0);
        double mx = 0.0, sa = 0.0, sb = 0.0;
        if (lo0 == 0) { mx = cb[0]; if (i >= 1 && kmin == 0) sb += (Fi - st)[0] * cb[0]; }
        if (i >= 1) {
            const double* __restrict Fp = Fi - st;
            const double* __restrict pr = P + static_cast<size_t>(J.a[i - 1]) * w;
            double* __restrict t = tmp;
            NWGRAD_SP_PRAGMA_(omp simd reduction(+:sa, sb) reduction(max:mx))
            for (int j = j1; j <= hi0; ++j) {
                const double b = cb[j];
                mx = b > mx ? b : mx;
                sa += Fi[j - 1] * b;
                sb += Fp[j] * b;
                t[j] = Fp[j - 1] * pr[j] * b;
            }
        } else {
            NWGRAD_SP_PRAGMA_(omp simd reduction(+:sa) reduction(max:mx))
            for (int j = j1; j <= hi0; ++j) {
                const double b = cb[j];
                mx = b > mx ? b : mx;
                sa += Fi[j - 1] * b;
            }
        }
        const int kb = sp_lazy(mx, lo0, hi0, bad, cur);
        if (bad) return;
        const int Ti = Tn + kb;
        // Sums were taken before any rescale: their factor uses Tn, the bound uses Ti.
        const double gh0 = sp_post(S[i] + Tn - ze, izr, bad);
        const double gh = sp_post(S[i] + Ti - ze, izr, bad);
        loss += sp_loss(n, i > 0 ? S[i] - S[i - 1] : 0, kb, mf[i], mx, gh);
        if (i >= kmin) g_ea += sa * ea * gh0;
        if (i >= 1) {
            const double gv0 = sp_post(S[i - 1] + Tn - ze, izr, bad);
            g_eb += sb * eb * gv0;
            sp_scatter(J, tmp, i, j1, hi0, gv0);
        }
        if (bad) return;
        Tn = Ti;
        std::swap(cur, nxt); std::swap(clo, nlo); std::swap(chi, nhi);
    }
    if (!(loss <= kSpLossTol)) return;
    J.g_oa = 0.0; J.g_ea = -g_ea; J.g_ob = 0.0; J.g_eb = -g_eb;
    J.ok = 1;
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
    double* mf = J.fmax;
    const double* P = J.P;
    bool bad = false;
    J.ok = 0;
    double* tp = J.r5;   // M+X+Y of the previous forward row; later the match scratch
    double* uy = J.r4;   // forward: the Y carry's carry-free input

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
        double mx = 0.0;
        for (int j = J.slo[0]; j <= J.shi[0]; ++j) {
            tp[j] = FM[j] + FX[j] + FY[j];
            mx = tp[j] > mx ? tp[j] : mx;
        }
        mf[0] = mx;
    }
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
        double* __restrict u = uy;
        const int lo = J.jlo[i], hi = J.jhi[i];
        if (lo <= hi) {
            u[lo] = (rM[lo - 1] + rX[lo - 1]) * oa;
            // M, X, and the Y carry's input for the NEXT column, in one vector pass.
            for (int j = lo; j <= hi; ++j) {
                const double mv = tq[j - 1] * pr[j] + fr;
                const double xv = (pM[j] + pY[j]) * ob + pX[j] * eb;
                rM[j] = mv; rX[j] = xv;
                u[j + 1] = (mv + xv) * oa;
            }
            sp_carry_fwd(rY, u, lo, hi, ea);
        }
        double mx = 0.0;
        double* __restrict t = tp;
        NWGRAD_SP_MAX(mx)
        for (int j = J.slo[i]; j <= J.shi[i]; ++j) {
            const double v = rM[j] + rX[j] + rY[j];
            t[j] = v;
            mx = v > mx ? v : mx;
        }
        S[i] = S[i - 1] + sp_lazy(mx, J.slo[i], J.shi[i], bad, rM, rX, rY, tp);
        if (bad) return;
        mf[i] = mx;
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
                    const double v1 = d + v * ob;
                    b1[j] = v1; b2[j] = d + v * eb; y[j] = v1;
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
        const double* __restrict fM = FM + ro; const double* __restrict fX = FX + ro;
        const double* __restrict fY = FY + ro;
        double* __restrict bM = cM; double* __restrict bX = cX;
        const double* __restrict bY = cY;
        const int j1 = std::max(1, lo0);
        double mx = 0.0, sx = 0.0, sy = 0.0, so = 0.0, sob = 0.0;
        if (lo0 == 0) {
            const double h = bY[1] * oa;
            const double mv = bM[0] + h, xv = bX[0] + h;
            bM[0] = mv; bX[0] = xv;
            mx = mv + xv + bY[0];
            if (i >= 1) {
                sx += fX[0] * xv;
                if (kmin == 0) sob += ((fM - st)[0] + (fY - st)[0]) * xv;
            }
        }
        if (i >= 1) {
            const double* __restrict qM = fM - st; const double* __restrict qX = fX - st;
            const double* __restrict qY = fY - st;
            const double* __restrict pr = P + static_cast<size_t>(J.a[i - 1]) * w;
            double* __restrict t = tmp;
            NWGRAD_SP_PRAGMA_(omp simd reduction(+:sx, sy, so, sob) reduction(max:mx))
            for (int j = j1; j <= hi0; ++j) {
                const double y = bY[j];
                const double h = bY[j + 1] * oa;
                const double mv = bM[j] + h, xv = bX[j] + h;
                bM[j] = mv; bX[j] = xv;
                const double tot = mv + xv + y;
                mx = tot > mx ? tot : mx;
                sx += fX[j] * xv;
                sy += fY[j] * y;
                so += (fM[j - 1] + fX[j - 1]) * y;
                sob += (qM[j] + qY[j]) * xv;
                t[j] = (qM[j - 1] + qX[j - 1] + qY[j - 1]) * pr[j] * mv;
            }
        } else {
            NWGRAD_SP_PRAGMA_(omp simd reduction(+:sy, so) reduction(max:mx))
            for (int j = j1; j <= hi0; ++j) {
                const double y = bY[j];
                const double h = bY[j + 1] * oa;
                const double mv = bM[j] + h, xv = bX[j] + h;
                bM[j] = mv; bX[j] = xv;
                const double tot = mv + xv + y;
                mx = tot > mx ? tot : mx;
                sy += fY[j] * y;
                so += (fM[j - 1] + fX[j - 1]) * y;
            }
        }
        const int kb = sp_lazy(mx, lo0, hi0, bad, cM, cX, cY);
        if (bad) return;
        const int Ti = Tn + kb;
        const double gh0 = sp_post(S[i] + Tn - ze, izr, bad);
        const double gh = sp_post(S[i] + Ti - ze, izr, bad);
        loss += sp_loss(n, i > 0 ? S[i] - S[i - 1] : 0, kb, mf[i], mx, gh);
        g_eb += sx * gh0;
        g_ea += sy * gh0;
        if (i >= kmin) g_oa += so * oa * gh0;
        if (i >= 1) {
            const double gv0 = sp_post(S[i - 1] + Tn - ze, izr, bad);
            g_ob += sob * ob * gv0;
            sp_scatter(J, tmp, i, j1, hi0, gv0);
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
