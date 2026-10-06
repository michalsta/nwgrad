// ── Inter-pair soft (scaled forward-backward), one instantiation per ISA level ──
//
// Included by level_common.inc after inter_kernel_impl.inl (uses IW, ivd, ivl, ivmax,
// ivsel).  See InterSoftJob in simd_levels.hpp for the contract.
//
// The soft counterpart of inter_fill_entry: W pairs at once, one per vector lane, all
// sharing B's length n, Full band.  Per lane it is exactly Aligner's scaled path
// (soft_pair_affine in soft_kernel_impl.inl: probability space, power-of-two row rescaling, gradient fused
// into the backward pass, the same lost-mass bound), so it is tolerance-equal to that
// path, not bit-equal: the soft path is not bit-exact by design.  Per-lane differences:
// each lane rescales by its own exponent (one scalar ilogb per lane per row), and a lane
// whose A is shorter than the group's longest carries zero backward mass in the rows
// past its end, so those rows contribute nothing.  The match counts are accumulated per
// B letter through lane masks (no scatter); nalpha <= 8, as for the Viterbi fill.

#ifndef NWGRAD_LEVEL_NS
#  error "inter_soft_impl.inl is included from a level TU (level_common.inc); not standalone"
#endif

// FMA contraction, for THESE functions only.  The build compiles everything with
// -ffp-contract=off because the Viterbi/hard kernels' bit-exactness across levels
// depends on it; the soft path is tolerance-tested and may fuse (so on avx2/avx512/neon
// each a*b + c here is one rounding, on sse2 two).  gcc: the optimize attribute on the
// function; clang: the scoped pragma at the top of the body (clang ignores the
// attribute).  Lambdas inside are separate functions and keep the global setting.
#if defined(__clang__)
#  define NWGRAD_SOFT_FMA_FN
#  define NWGRAD_SOFT_FMA_BODY _Pragma("clang fp contract(fast)")
#elif defined(__GNUC__)
#  define NWGRAD_SOFT_FMA_FN __attribute__((optimize("fp-contract=fast")))
#  define NWGRAD_SOFT_FMA_BODY
#else
#  define NWGRAD_SOFT_FMA_FN
#  define NWGRAD_SOFT_FMA_BODY
#endif

// Ragged B (J.nb, Rag): a lane's columns past its own length get no letter (weight 0)
// and every forward and backward cell there is masked to 0, so they add nothing to
// its row sums, maxima or gradient sums; Global starts its backward at (m, n_l) and
// reads Z there.  Rag = false compiles none of it.
template <bool Rag>
NWGRAD_SOFT_FMA_FN static void inter_soft_affine(InterSoftJob& J) noexcept {
    NWGRAD_SOFT_FMA_BODY
    const int n = J.n, M = J.M, st = n + 1, na = J.nalpha, w2 = n + 2;
    const bool local = J.align_mode == 1;
    const int kmin = local ? 1 : 0;   // Aligner::kGapTargetMin
    const ivd z = {}, one = z + 1.0;
    const ivd oa = z + J.oa, ea = z + J.ea, ob = z + J.ob, eb = z + J.eb;
    const double DMIN = std::numeric_limits<double>::min();
    const double DMAX = std::numeric_limits<double>::max();

    ivd* FM = reinterpret_cast<ivd*>(J.scratch);
    ivd* FX = FM + static_cast<size_t>(M + 1) * st;
    ivd* FY = FX + static_cast<size_t>(M + 1) * st;
    ivd* ER = FY + static_cast<size_t>(M + 1) * st;      // (M+1) rows of w2: weights per row
    ivd* cM = ER + static_cast<size_t>(M + 1) * w2;
    ivd* cX = cM + w2; ivd* cY = cX + w2; ivd* nM = cY + w2; ivd* nX = nM + w2;
    ivd* tq = nX + w2; ivd* tq2 = tq + w2;
    ivd* rs = tq2 + w2;                      // M+1 row sums (Local log Z)
    ivd* mf = rs + (M + 1);                  // M+1 forward row maxima (stored units)
    int* S = J.iscratch;                     // (M+1)*IW forward exponents

    // Lane masks: column j's B residue == c.
    // Alphabets over 8 letters (big): no letter masks — each row's weights are gathered
    // per lane (fill_E) and the match contributions kept per cell (trow), added into the
    // counts per lane after the row.  nm = the letters accumulated through masks.
    const bool big = na > 8;
    const int nm = big ? 0 : na;
    static thread_local std::vector<ivd> trowv;
    if (big && trowv.size() < static_cast<size_t>(w2)) trowv.resize(w2);
    ivd* trow = trowv.data();
    (void)trow;
    static thread_local std::vector<ivl> eqm;
    if (eqm.size() < static_cast<size_t>(w2) * nm) eqm.resize(static_cast<size_t>(w2) * nm);
    for (int j = 1; j <= n; ++j)
        for (int c = 0; c < nm; ++c) {
            ivl v;
            for (int l = 0; l < IW; ++l)
                v[l] = ((!Rag || j <= J.nb[l]) && J.b[l][j - 1] == c) ? -1 : 0;
            eqm[static_cast<size_t>(j) * na + c] = v;
        }
    // Ragged: cok[j] = lanes whose B reaches column j; endc[j] = lanes whose B ends there.
    static thread_local std::vector<ivl> cokv, endv;
    const ivl* cok = nullptr; const ivl* endc = nullptr;
    if constexpr (Rag) {
        if (cokv.size() < static_cast<size_t>(w2)) { cokv.resize(w2); endv.resize(w2); }
        for (int j = 0; j < w2; ++j)
            for (int l = 0; l < IW; ++l) {
                cokv[j][l] = j <= J.nb[l] ? -1 : 0;
                endv[j][l] = j == J.nb[l] ? -1 : 0;
            }
        cok = cokv.data(); endc = endv.data();
    }
    (void)cok; (void)endc;
    // ER[i][j] = exp(score(a_l[i-1], b_l[j-1]) / T) per lane; [0] = [n+1] = 0.  Built
    // once per row in the forward pass, read again by the backward pass.
    auto fill_E = [&](int i) {
        ivd* E = ER + static_cast<size_t>(i) * w2;
        E[0] = z; E[n + 1] = z;
        if (big) {
            for (int l = 0; l < IW; ++l) {
                const double* pr = J.es + static_cast<size_t>((i <= J.m[l]) ? J.a[l][i - 1] : 0) * na;
                const unsigned char* bl = J.b[l];
                const int nl = Rag ? J.nb[l] : n;
                for (int j = 1; j <= nl; ++j) E[j][l] = pr[bl[j - 1]];
                for (int j = nl + 1; j <= n; ++j) E[j][l] = 0.0;
            }
            return;
        }
        ivd P[8];
        for (int c = 0; c < na; ++c)
            for (int l = 0; l < IW; ++l) {
                const int a = (i <= J.m[l]) ? J.a[l][i - 1] : 0;
                P[c][l] = J.es[a * na + c];
            }
        for (int j = 1; j <= n; ++j) {
            ivd s = z;
            const ivl* mk = eqm.data() + static_cast<size_t>(j) * na;
            for (int c = 0; c < na; ++c) s = ivsel(mk[c], P[c], s);
            E[j] = s;
        }
    };
    // LAZY rescale.  A lane is rescaled only when its row max leaves [2^-256, 2^256]
    // (or is exactly 0: nothing to do); kv[l] receives the exponent removed (0 when
    // none), mx (in/out) the lane max in stored units after any rescale.  The lost-mass
    // bound below uses those actual maxima, so skipping rescales costs no rigour.
    int kv[IW];
    bool bad[IW];
    for (int l = 0; l < IW; ++l) bad[l] = false;
    auto rescale = [&](ivd* r0, ivd* r1, ivd* r2, ivd* tot, ivd& mx) {
        ivd sc;
        bool any = false;
        for (int l = 0; l < IW; ++l) {
            int k = 0;
            const double v = mx[l];
            if (!(v <= DMAX)) bad[l] = true;
            else if (v > 0.0 && (v > 0x1p256 || v < 0x1p-256)) k = std::ilogb(v);
            kv[l] = k;
            any |= (k != 0);
            sc[l] = std::ldexp(1.0, -k);
        }
        if (!any) return;
        for (int j = 0; j <= n; ++j) { r0[j] *= sc; r1[j] *= sc; r2[j] *= sc; }
        if (tot) for (int j = 0; j <= n; ++j) tot[j] *= sc;
        mx *= sc;
    };

    // ── Forward ──
    for (int j = 0; j <= n; ++j) { FM[j] = z; FX[j] = z; FY[j] = z; }
    if (local) { for (int j = 0; j <= n; ++j) FM[j] = one; }
    else {
        FM[0] = one;
        if (n >= 1) FY[1] = oa;
        for (int j = 2; j <= n; ++j) FY[j] = FY[j - 1] * ea;
    }
    if constexpr (Rag)
        for (int j = 0; j <= n; ++j) { FM[j] = ivsel(cok[j], FM[j], z); FY[j] = ivsel(cok[j], FY[j], z); }
    for (int l = 0; l < IW; ++l) S[l] = 0;
    {
        ivd s = z, mx = z;
        for (int j = 0; j <= n; ++j) { tq[j] = FM[j] + FX[j] + FY[j]; s += tq[j]; mx = ivmax(mx, tq[j]); }
        rs[0] = s; mf[0] = mx;
    }
    for (int i = 1; i <= M; ++i) {
        fill_E(i);
        const ivd* E = ER + static_cast<size_t>(i) * w2;
        const size_t ro = static_cast<size_t>(i) * st;
        ivd* rM = FM + ro; ivd* rX = FX + ro; ivd* rY = FY + ro;
        const ivd* pM = rM - st; const ivd* pX = rX - st; const ivd* pY = rY - st;
        ivd fr = z;
        if (local) for (int l = 0; l < IW; ++l) fr[l] = std::ldexp(1.0, -S[(i - 1) * IW + l]);
        ivd lm, lx, ly;
        if (local) { lm = fr; lx = z; ly = z; }
        else       { lm = z; ly = z; lx = (pM[0] + pY[0]) * ob + pX[0] * eb; }
        rM[0] = lm; rX[0] = lx; rY[0] = ly;
        ivd t0 = lm + lx + ly;
        tq2[0] = t0;
        ivd s = t0, mx = t0;
        // One fused column loop: the Y carry's latency hides under the M/X work.
        for (int j = 1; j <= n; ++j) {
            ivd mv = tq[j - 1] * E[j] + fr;
            ivd xv = (pM[j] + pY[j]) * ob + pX[j] * eb;
            ivd yv = (lm + lx) * oa + ly * ea;
            if constexpr (Rag) { mv = ivsel(cok[j], mv, z); xv = ivsel(cok[j], xv, z); yv = ivsel(cok[j], yv, z); }
            rM[j] = mv; rX[j] = xv; rY[j] = yv;
            lm = mv; lx = xv; ly = yv;
            const ivd t = mv + xv + yv;
            tq2[j] = t; s += t; mx = ivmax(mx, t);
        }
        std::swap(tq, tq2);
        rescale(rM, rX, rY, tq, mx);
        for (int l = 0; l < IW; ++l) {
            S[i * IW + l] = S[(i - 1) * IW + l] + kv[l];
            if (kv[l]) s[l] = std::ldexp(s[l], -kv[l]);
        }
        rs[i] = s; mf[i] = mx;
    }

    // ── Z = 2^ze · zr per lane ──
    int ze[IW];
    double izr[IW];
    for (int l = 0; l < IW; ++l) {
        const int ml = J.m[l];
        double zr;
        if (!local) {
            const size_t o = static_cast<size_t>(ml) * st + (Rag ? J.nb[l] : n);
            ze[l] = S[ml * IW + l];
            zr = FM[o][l] + FX[o][l] + FY[o][l];
        } else {
            int e = S[l];
            for (int i = 1; i <= ml; ++i) e = std::max(e, S[i * IW + l]);
            ze[l] = e;
            zr = 0.0;
            for (int i = 0; i <= ml; ++i) zr += std::ldexp(rs[i][l], S[i * IW + l] - e);
        }
        if (!(zr > 0.0) || !(zr <= DMAX)) { bad[l] = true; zr = 1.0; }
        // Normalize zr into [1, 2) so 1/zr and the ldexp factors stay in range.
        const int kz = std::ilogb(zr);
        zr = std::ldexp(zr, -kz); ze[l] += kz;
        izr[l] = 1.0 / zr;
        J.logz[l] = std::log(zr) + ze[l] * 0.69314718055994530942;
    }

    // ── Backward, gradient fused ──
    for (int j = 0; j < w2; ++j) { cM[j] = z; cX[j] = z; cY[j] = z; nM[j] = z; nX[j] = z; }
    int Tn[IW];
    double loss[IW], g_oa[IW], g_ea[IW], g_ob[IW], g_eb[IW];
    for (int l = 0; l < IW; ++l) { Tn[l] = 0; loss[l] = g_oa[l] = g_ea[l] = g_ob[l] = g_eb[l] = 0.0; }
    const size_t nn = static_cast<size_t>(na) * na;
    for (int l = 0; l < IW; ++l)
        for (size_t k = 0; k < nn; ++k) J.counts[l * nn + k] = 0.0;

    for (int i = M; i >= 0; --i) {
        ivl top;
        ivd init = z;
        for (int l = 0; l < IW; ++l) {
            top[l] = (i == J.m[l]) ? -1 : 0;
            if (local && i <= J.m[l]) init[l] = std::ldexp(1.0, -Tn[l]);
        }
        const ivd* En = (i < M) ? ER + static_cast<size_t>(i + 1) * w2 : nullptr;
        const size_t ro = static_cast<size_t>(i) * st;
        const ivd* fM = FM + ro; const ivd* fX = FX + ro; const ivd* fY = FY + ro;
        const ivd* qM = fM - st; const ivd* qX = fX - st; const ivd* qY = fY - st;
        const ivd* Ec = ER + static_cast<size_t>(i) * w2;
        // ONE fused descending pass per row: carry-free part, the Y carry, the M/X gap
        // terms, AND every gradient sum.  The sums are linear in this row's backward
        // values, so they are taken before any rescale and the (rare) rescale is
        // applied to them afterwards.
        ivd c = z, mx = z;
        ivd sx = z, sy = z, so = z, sob = z, acc[8];
        for (int cc = 0; cc < nm; ++cc) acc[cc] = z;
        // soft_guide="posterior": each lane's row argmax of the posterior; the loop runs
        // right to left, so >= keeps the leftmost (as the per-pair kernel's first >).
        const bool post = J.gpost != nullptr;
        ivd pbv = z - 1.0, pbj = z, pyn = z;   // pyn: the Y posterior at j+1 (exit mass)
        auto cell = [&](int j) {
            const ivd d = (En ? En[j + 1] * nM[j + 1] : z) + init;
            const ivd v = nX[j];
            ivd b1 = d + v * ob, b2 = d + v * eb;
            if constexpr (Rag) {
                if (!local) { const ivl e = top & endc[j]; b1 = ivsel(e, one, b1); b2 = ivsel(e, one, b2); }
            } else if (!local && j == n) { b1 = ivsel(top, one, b1); b2 = ivsel(top, one, b2); }
            const ivd h = c * oa;
            ivd y = b1 + ea * c;
            ivd mv = b1 + h, xv = b2 + h;
            if constexpr (Rag) { mv = ivsel(cok[j], mv, z); xv = ivsel(cok[j], xv, z); y = ivsel(cok[j], y, z); }
            cM[j] = mv; cX[j] = xv; cY[j] = y;
            c = y;
            mx = ivmax(mx, mv + xv + y);
            if (post) {
                const ivd py = fY[j] * y;
                const ivd pv = fM[j] * mv + fX[j] * xv + py - pyn;
                pyn = py;
                const ivl up = pv >= pbv;
                pbv = ivsel(up, pv, pbv);
                pbj = ivsel(up, z + static_cast<double>(j), pbj);
            }
            return std::make_pair(mv, xv);
        };
        if (i >= 1) {
            for (int j = n; j >= 1; --j) {
                const auto [mv, xv] = cell(j);
                const ivd y = c;
                sx += fX[j] * xv;
                sy += fY[j] * y;
                so += (fM[j - 1] + fX[j - 1]) * y;
                sob += (qM[j] + qY[j]) * xv;
                const ivd t = (qM[j - 1] + qX[j - 1] + qY[j - 1]) * Ec[j] * mv;
                if (big) trow[j] = t;
                const ivl* mk = eqm.data() + static_cast<size_t>(j) * nm;   // nm may be 0: no [] on an empty vector
                for (int cc = 0; cc < nm; ++cc) acc[cc] += ivsel(mk[cc], t, z);
            }
            const auto [mv0, xv0] = cell(0);
            (void)mv0;
            sx += fX[0] * xv0;
            if (kmin == 0) sob += (qM[0] + qY[0]) * xv0;
        } else {
            for (int j = n; j >= 1; --j) {
                cell(j);
                const ivd y = c;
                sy += fY[j] * y;
                so += (fM[j - 1] + fX[j - 1]) * y;
            }
            cell(0);
        }
        if (post)
            for (int l = 0; l < IW; ++l) J.gpost[static_cast<size_t>(i) * IW + l] = static_cast<int>(pbj[l]);
        rescale(cM, cX, cY, nullptr, mx);
        for (int l = 0; l < IW; ++l)
            if (kv[l]) {
                const double f = std::ldexp(1.0, -kv[l]);
                sx[l] *= f; sy[l] *= f; so[l] *= f; sob[l] *= f;
                for (int cc = 0; cc < nm; ++cc) acc[cc][l] *= f;
            }

        ivd gh, gv = z;
        int Ti[IW];
        for (int l = 0; l < IW; ++l) {
            Ti[l] = Tn[l] + kv[l];
            if (i > J.m[l]) { gh[l] = 0.0; continue; }
            gh[l] = std::ldexp(izr[l], S[i * IW + l] + Ti[l] - ze[l]);
            if (i >= 1) gv[l] = std::ldexp(izr[l], S[(i - 1) * IW + l] + Ti[l] - ze[l]);
            if (!(gh[l] <= DMAX) || !(gv[l] <= DMAX)) bad[l] = true;
            // Lost mass: below DMIN per term in stored units (times 2^-k where a
            // rescale scaled the row UP); its posterior effect is bounded by the other
            // direction's actual row max times the row's posterior factor.
            const int kf = i > 0 ? S[i * IW + l] - S[(i - 1) * IW + l] : 0, kb = kv[l];
            const double lf = kf < 0 ? std::ldexp(1.0, -kf) : 1.0;
            const double lb = kb < 0 ? std::ldexp(1.0, -kb) : 1.0;
            loss[l] += 32.0 * (n + 1) * DMIN * (lf * mx[l] + lb * mf[i][l]) * gh[l];
        }
        for (int l = 0; l < IW; ++l) {
            if (i > J.m[l]) continue;
            g_eb[l] += sx[l] * gh[l];
            g_ea[l] += sy[l] * gh[l];
            if (i >= kmin) g_oa[l] += so[l] * J.oa * gh[l];
            if (i >= 1) {
                g_ob[l] += sob[l] * J.ob * gv[l];
                double* gr = J.counts + l * nn + static_cast<size_t>(J.a[l][i - 1]) * na;
                if (big) {
                    // trow holds pre-rescale values: apply this row's rescale factor here.
                    const double f = std::ldexp(gv[l], -kv[l]);
                    const unsigned char* bl = J.b[l];
                    const int nl = Rag ? J.nb[l] : n;
                    for (int j = 1; j <= nl; ++j) gr[bl[j - 1]] += trow[j][l] * f;
                }
                for (int cc = 0; cc < nm; ++cc) gr[cc] += acc[cc][l] * gv[l];
            }
        }
        for (int l = 0; l < IW; ++l) Tn[l] = (i > J.m[l]) ? 0 : Ti[l];
        std::swap(cM, nM); std::swap(cX, nX);
    }
    for (int l = 0; l < IW; ++l) {
        J.ok[l] = (!bad[l] && loss[l] <= 0x1p-45) ? 1 : 0;
        J.gaps[l * 4 + 0] = -g_oa[l];
        J.gaps[l * 4 + 1] = -g_ea[l];
        J.gaps[l * 4 + 2] = -g_ob[l];
        J.gaps[l * 4 + 3] = -g_eb[l];
    }
}

// Linear gaps: one table, F(i,j) = F(i-1,j-1)·E + F(i-1,j)·eb + F(i,j-1)·ea (+ Local's
// free start).  Per lane soft_pair_linear (soft_kernel_impl.inl); the same fused column
// loops, lazy rescale and lost-mass bound as inter_soft_affine.  Gap opens are 0.
template <bool Rag>
NWGRAD_SOFT_FMA_FN static void inter_soft_linear(InterSoftJob& J) noexcept {
    NWGRAD_SOFT_FMA_BODY
    const int n = J.n, M = J.M, st = n + 1, na = J.nalpha, w2 = n + 2;
    const bool local = J.align_mode == 1;
    const int kmin = local ? 1 : 0;
    const ivd z = {}, one = z + 1.0;
    const ivd ea = z + J.ea, eb = z + J.eb;
    const double DMIN = std::numeric_limits<double>::min();
    const double DMAX = std::numeric_limits<double>::max();

    ivd* F = reinterpret_cast<ivd*>(J.scratch);
    ivd* ER = F + static_cast<size_t>(M + 1) * st;
    ivd* cur = ER + static_cast<size_t>(M + 1) * w2;
    ivd* nxt = cur + w2;
    ivd* rs = nxt + w2;
    ivd* mf = rs + (M + 1);
    int* S = J.iscratch;

    // Alphabets over 8 letters (big): no letter masks — each row's weights are gathered
    // per lane (fill_E) and the match contributions kept per cell (trow), added into the
    // counts per lane after the row.  nm = the letters accumulated through masks.
    const bool big = na > 8;
    const int nm = big ? 0 : na;
    static thread_local std::vector<ivd> trowv;
    if (big && trowv.size() < static_cast<size_t>(w2)) trowv.resize(w2);
    ivd* trow = trowv.data();
    (void)trow;
    static thread_local std::vector<ivl> eqm;
    if (eqm.size() < static_cast<size_t>(w2) * nm) eqm.resize(static_cast<size_t>(w2) * nm);
    for (int j = 1; j <= n; ++j)
        for (int c = 0; c < nm; ++c) {
            ivl v;
            for (int l = 0; l < IW; ++l)
                v[l] = ((!Rag || j <= J.nb[l]) && J.b[l][j - 1] == c) ? -1 : 0;
            eqm[static_cast<size_t>(j) * na + c] = v;
        }
    // Ragged: cok[j] = lanes whose B reaches column j; endc[j] = lanes whose B ends there.
    static thread_local std::vector<ivl> cokv, endv;
    const ivl* cok = nullptr; const ivl* endc = nullptr;
    if constexpr (Rag) {
        if (cokv.size() < static_cast<size_t>(w2)) { cokv.resize(w2); endv.resize(w2); }
        for (int j = 0; j < w2; ++j)
            for (int l = 0; l < IW; ++l) {
                cokv[j][l] = j <= J.nb[l] ? -1 : 0;
                endv[j][l] = j == J.nb[l] ? -1 : 0;
            }
        cok = cokv.data(); endc = endv.data();
    }
    (void)cok; (void)endc;
    auto fill_E = [&](int i) {
        ivd* E = ER + static_cast<size_t>(i) * w2;
        E[0] = z; E[n + 1] = z;
        if (big) {
            for (int l = 0; l < IW; ++l) {
                const double* pr = J.es + static_cast<size_t>((i <= J.m[l]) ? J.a[l][i - 1] : 0) * na;
                const unsigned char* bl = J.b[l];
                const int nl = Rag ? J.nb[l] : n;
                for (int j = 1; j <= nl; ++j) E[j][l] = pr[bl[j - 1]];
                for (int j = nl + 1; j <= n; ++j) E[j][l] = 0.0;
            }
            return;
        }
        ivd P[8];
        for (int c = 0; c < na; ++c)
            for (int l = 0; l < IW; ++l) {
                const int a = (i <= J.m[l]) ? J.a[l][i - 1] : 0;
                P[c][l] = J.es[a * na + c];
            }
        for (int j = 1; j <= n; ++j) {
            ivd sv = z;
            const ivl* mk = eqm.data() + static_cast<size_t>(j) * na;
            for (int c = 0; c < na; ++c) sv = ivsel(mk[c], P[c], sv);
            E[j] = sv;
        }
    };
    int kv[IW];
    bool bad[IW];
    for (int l = 0; l < IW; ++l) bad[l] = false;
    auto rescale = [&](ivd* r, ivd& mx) {
        ivd sc;
        bool any = false;
        for (int l = 0; l < IW; ++l) {
            int k = 0;
            const double v = mx[l];
            if (!(v <= DMAX)) bad[l] = true;
            else if (v > 0.0 && (v > 0x1p256 || v < 0x1p-256)) k = std::ilogb(v);
            kv[l] = k;
            any |= (k != 0);
            sc[l] = std::ldexp(1.0, -k);
        }
        if (!any) return;
        for (int j = 0; j <= n; ++j) r[j] *= sc;
        mx *= sc;
    };

    // ── Forward ──
    if (local) { for (int j = 0; j <= n; ++j) F[j] = one; }
    else { F[0] = one; for (int j = 1; j <= n; ++j) F[j] = F[j - 1] * ea; }
    if constexpr (Rag) for (int j = 0; j <= n; ++j) F[j] = ivsel(cok[j], F[j], z);
    for (int l = 0; l < IW; ++l) S[l] = 0;
    {
        ivd sv = z, mx = z;
        for (int j = 0; j <= n; ++j) { sv += F[j]; mx = ivmax(mx, F[j]); }
        rs[0] = sv; mf[0] = mx;
    }
    for (int i = 1; i <= M; ++i) {
        fill_E(i);
        const ivd* E = ER + static_cast<size_t>(i) * w2;
        ivd* r = F + static_cast<size_t>(i) * st;
        const ivd* p = r - st;
        ivd fr = z;
        if (local) for (int l = 0; l < IW; ++l) fr[l] = std::ldexp(1.0, -S[(i - 1) * IW + l]);
        ivd lv = local ? fr : p[0] * eb;
        r[0] = lv;
        ivd sv = lv, mx = lv;
        for (int j = 1; j <= n; ++j) {
            lv = p[j - 1] * E[j] + p[j] * eb + fr + lv * ea;
            if constexpr (Rag) lv = ivsel(cok[j], lv, z);
            r[j] = lv; sv += lv; mx = ivmax(mx, lv);
        }
        rescale(r, mx);
        for (int l = 0; l < IW; ++l) {
            S[i * IW + l] = S[(i - 1) * IW + l] + kv[l];
            if (kv[l]) sv[l] = std::ldexp(sv[l], -kv[l]);
        }
        rs[i] = sv; mf[i] = mx;
    }

    int ze[IW];
    double izr[IW];
    for (int l = 0; l < IW; ++l) {
        const int ml = J.m[l];
        double zr;
        if (!local) {
            ze[l] = S[ml * IW + l];
            zr = F[static_cast<size_t>(ml) * st + (Rag ? J.nb[l] : n)][l];
        } else {
            int e = S[l];
            for (int i = 1; i <= ml; ++i) e = std::max(e, S[i * IW + l]);
            ze[l] = e;
            zr = 0.0;
            for (int i = 0; i <= ml; ++i) zr += std::ldexp(rs[i][l], S[i * IW + l] - e);
        }
        if (!(zr > 0.0) || !(zr <= DMAX)) { bad[l] = true; zr = 1.0; }
        const int kz = std::ilogb(zr);
        zr = std::ldexp(zr, -kz); ze[l] += kz;
        izr[l] = 1.0 / zr;
        J.logz[l] = std::log(zr) + ze[l] * 0.69314718055994530942;
    }

    // ── Backward, gradient fused ──
    for (int j = 0; j < w2; ++j) { cur[j] = z; nxt[j] = z; }
    int Tn[IW];
    double loss[IW], g_ea[IW], g_eb[IW];
    for (int l = 0; l < IW; ++l) { Tn[l] = 0; loss[l] = g_ea[l] = g_eb[l] = 0.0; }
    const size_t nn = static_cast<size_t>(na) * na;
    for (int l = 0; l < IW; ++l)
        for (size_t k = 0; k < nn; ++k) J.counts[l * nn + k] = 0.0;

    for (int i = M; i >= 0; --i) {
        ivl top;
        ivd init = z;
        for (int l = 0; l < IW; ++l) {
            top[l] = (i == J.m[l]) ? -1 : 0;
            if (local && i <= J.m[l]) init[l] = std::ldexp(1.0, -Tn[l]);
        }
        const ivd* En = (i < M) ? ER + static_cast<size_t>(i + 1) * w2 : nullptr;
        const ivd* f = F + static_cast<size_t>(i) * st;
        const ivd* q = f - st;
        const ivd* Ec = ER + static_cast<size_t>(i) * w2;
        ivd c = z, mx = z, sa = z, sb = z, acc[8];
        for (int cc = 0; cc < nm; ++cc) acc[cc] = z;
        const bool post = J.gpost != nullptr;   // as in inter_soft_affine
        ivd pbv = z - 1.0, pbj = z;
        for (int j = n; j >= 0; --j) {
            const ivd bn = c;   // B at j+1 (exit mass)
            ivd b = (En ? En[j + 1] * nxt[j + 1] : z) + nxt[j] * eb + init + ea * c;
            if constexpr (Rag) {
                if (!local) b = ivsel(top & endc[j], one, b);
                b = ivsel(cok[j], b, z);
            } else if (!local && j == n) b = ivsel(top, one, b);
            cur[j] = b; c = b;
            mx = ivmax(mx, b);
            if (post) {
                const ivd pv = f[j] * (b - ea * bn);
                const ivl up = pv >= pbv;
                pbv = ivsel(up, pv, pbv);
                pbj = ivsel(up, z + static_cast<double>(j), pbj);
            }
            if (j >= 1) sa += f[j - 1] * b;
            if (i >= 1) {
                if (j >= kmin) sb += q[j] * b;
                if (j >= 1) {
                    const ivd t = q[j - 1] * Ec[j] * b;
                    if (big) trow[j] = t;
                    const ivl* mk = eqm.data() + static_cast<size_t>(j) * nm;   // nm may be 0: no [] on an empty vector
                    for (int cc = 0; cc < nm; ++cc) acc[cc] += ivsel(mk[cc], t, z);
                }
            }
        }
        if (post)
            for (int l = 0; l < IW; ++l) J.gpost[static_cast<size_t>(i) * IW + l] = static_cast<int>(pbj[l]);
        rescale(cur, mx);
        for (int l = 0; l < IW; ++l)
            if (kv[l]) {
                const double fct = std::ldexp(1.0, -kv[l]);
                sa[l] *= fct; sb[l] *= fct;
                for (int cc = 0; cc < nm; ++cc) acc[cc][l] *= fct;
            }
        for (int l = 0; l < IW; ++l) {
            const int Ti = Tn[l] + kv[l];
            if (i > J.m[l]) { Tn[l] = 0; continue; }
            const double gh = std::ldexp(izr[l], S[i * IW + l] + Ti - ze[l]);
            const double gv = i >= 1 ? std::ldexp(izr[l], S[(i - 1) * IW + l] + Ti - ze[l]) : 0.0;
            if (!(gh <= DMAX) || !(gv <= DMAX)) bad[l] = true;
            const int kf = i > 0 ? S[i * IW + l] - S[(i - 1) * IW + l] : 0, kb = kv[l];
            const double lf = kf < 0 ? std::ldexp(1.0, -kf) : 1.0;
            const double lb = kb < 0 ? std::ldexp(1.0, -kb) : 1.0;
            loss[l] += 16.0 * (n + 1) * DMIN * (lf * mx[l] + lb * mf[i][l]) * gh;
            if (i >= kmin) g_ea[l] += sa[l] * J.ea * gh;
            if (i >= 1) {
                g_eb[l] += sb[l] * J.eb * gv;
                double* gr = J.counts + l * nn + static_cast<size_t>(J.a[l][i - 1]) * na;
                if (big) {
                    const double f = std::ldexp(gv, -kv[l]);
                    const unsigned char* bl = J.b[l];
                    const int nl = Rag ? J.nb[l] : n;
                    for (int j = 1; j <= nl; ++j) gr[bl[j - 1]] += trow[j][l] * f;
                }
                for (int cc = 0; cc < nm; ++cc) gr[cc] += acc[cc][l] * gv;
            }
            Tn[l] = Ti;
        }
        std::swap(cur, nxt);
    }
    for (int l = 0; l < IW; ++l) {
        J.ok[l] = (!bad[l] && loss[l] <= 0x1p-45) ? 1 : 0;
        J.gaps[l * 4 + 0] = 0.0;
        J.gaps[l * 4 + 1] = -g_ea[l];
        J.gaps[l * 4 + 2] = 0.0;
        J.gaps[l * 4 + 3] = -g_eb[l];
    }
}

static void inter_soft_entry(InterSoftJob& J) noexcept {
    if (J.nb) { if (J.linear) inter_soft_linear<true>(J);  else inter_soft_affine<true>(J); }
    else      { if (J.linear) inter_soft_linear<false>(J); else inter_soft_affine<false>(J); }
}
