// ── Inter-pair soft (scaled forward-backward), one instantiation per ISA level ──
//
// Included by level_common.inc after inter_kernel_impl.inl (uses IW, ivd, ivl, ivmax,
// ivsel).  See InterSoftJob in simd_levels.hpp for the contract.
//
// The soft counterpart of inter_fill_entry: W pairs at once, one per vector lane, all
// sharing B's length n, Full band.  Per lane it is exactly Aligner's scaled path
// (fwdbwd_affine_scaled: probability space, power-of-two row rescaling, gradient fused
// into the backward pass, the same lost-mass bound), so it is tolerance-equal to that
// path, not bit-equal: the soft path is not bit-exact by design.  Per-lane differences:
// each lane rescales by its own exponent (one scalar ilogb per lane per row), and a lane
// whose A is shorter than the group's longest carries zero backward mass in the rows
// past its end, so those rows contribute nothing.  The match counts are accumulated per
// B letter through lane masks (no scatter); nalpha <= 8, as for the Viterbi fill.

#ifndef NWGRAD_LEVEL_NS
#  error "inter_soft_impl.inl is included from a level TU (level_common.inc); not standalone"
#endif

static void inter_soft_entry(InterSoftJob& J) noexcept {
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
    static thread_local std::vector<ivl> eqm;
    if (eqm.size() < static_cast<size_t>(w2) * na) eqm.resize(static_cast<size_t>(w2) * na);
    for (int j = 1; j <= n; ++j)
        for (int c = 0; c < na; ++c) {
            ivl v;
            for (int l = 0; l < IW; ++l) v[l] = (J.b[l][j - 1] == c) ? -1 : 0;
            eqm[static_cast<size_t>(j) * na + c] = v;
        }
    // ER[i][j] = exp(score(a_l[i-1], b_l[j-1]) / T) per lane; [0] = [n+1] = 0.  Built
    // once per row in the forward pass, read again by the backward pass.
    auto fill_E = [&](int i) {
        ivd* E = ER + static_cast<size_t>(i) * w2;
        ivd P[8];
        for (int c = 0; c < na; ++c)
            for (int l = 0; l < IW; ++l) {
                const int a = (i <= J.m[l]) ? J.a[l][i - 1] : 0;
                P[c][l] = J.es[a * na + c];
            }
        E[0] = z; E[n + 1] = z;
        for (int j = 1; j <= n; ++j) {
            ivd s = z;
            const ivl* mk = &eqm[static_cast<size_t>(j) * na];
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
            const ivd mv = tq[j - 1] * E[j] + fr;
            const ivd xv = (pM[j] + pY[j]) * ob + pX[j] * eb;
            const ivd yv = (lm + lx) * oa + ly * ea;
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
            const size_t o = static_cast<size_t>(ml) * st + n;
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
        // One fused descending loop: carry-free part, Y carry, and the M/X gap terms.
        ivd c = z, mx = z;
        for (int j = n; j >= 0; --j) {
            const ivd d = (En ? En[j + 1] * nM[j + 1] : z) + init;
            const ivd v = nX[j];
            ivd b1 = d + v * ob, b2 = d + v * eb;
            if (!local && j == n) { b1 = ivsel(top, one, b1); b2 = ivsel(top, one, b2); }
            const ivd h = c * oa;
            const ivd y = b1 + ea * c;
            const ivd mv = b1 + h, xv = b2 + h;
            cM[j] = mv; cX[j] = xv; cY[j] = y;
            c = y;
            mx = ivmax(mx, mv + xv + y);
        }
        rescale(cM, cX, cY, nullptr, mx);

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

        const size_t ro = static_cast<size_t>(i) * st;
        const ivd* fM = FM + ro; const ivd* fX = FX + ro; const ivd* fY = FY + ro;
        ivd sx = z, sy = z, so = z;
        if (i >= 1) for (int j = 0; j <= n; ++j) sx += fX[j] * cX[j];
        for (int j = 1; j <= n; ++j) {
            sy += fY[j] * cY[j];
            so += (fM[j - 1] + fX[j - 1]) * cY[j];
        }
        ivd sob = z, acc[8];
        if (i >= 1) {
            const ivd* Ec = ER + static_cast<size_t>(i) * w2;
            const ivd* qM = fM - st; const ivd* qX = fX - st; const ivd* qY = fY - st;
            for (int j = kmin; j <= n; ++j) sob += (qM[j] + qY[j]) * cX[j];
            for (int cc = 0; cc < na; ++cc) acc[cc] = z;
            for (int j = 1; j <= n; ++j) {
                const ivd t = (qM[j - 1] + qX[j - 1] + qY[j - 1]) * Ec[j] * cM[j];
                const ivl* mk = &eqm[static_cast<size_t>(j) * na];
                for (int cc = 0; cc < na; ++cc) acc[cc] += ivsel(mk[cc], t, z);
            }
        }
        for (int l = 0; l < IW; ++l) {
            if (i > J.m[l]) continue;
            g_eb[l] += sx[l] * gh[l];
            g_ea[l] += sy[l] * gh[l];
            if (i >= kmin) g_oa[l] += so[l] * J.oa * gh[l];
            if (i >= 1) {
                g_ob[l] += sob[l] * J.ob * gv[l];
                double* gr = J.counts + l * nn + static_cast<size_t>(J.a[l][i - 1]) * na;
                for (int cc = 0; cc < na; ++cc) gr[cc] += acc[cc][l] * gv[l];
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
