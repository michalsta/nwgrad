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
    ivd* cM = FY + static_cast<size_t>(M + 1) * st;
    ivd* cX = cM + w2; ivd* cY = cX + w2; ivd* nM = cY + w2; ivd* nX = nM + w2;
    ivd* tq = nX + w2; ivd* Ec = tq + w2; ivd* En = Ec + w2;
    ivd* rs = En + w2;                       // M+1 row sums (Local log Z)
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
    // E[j] = exp(score(a_l[i-1], b_l[j-1]) / T) per lane, for row i; E[0] = E[n+1] = 0.
    auto fill_E = [&](int i, ivd* E) {
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
    // Per-lane rescale of [0, n] of three rows so each lane's total max lands in [1, 2).
    int kv[IW];
    bool bad[IW];
    for (int l = 0; l < IW; ++l) bad[l] = false;
    auto rescale = [&](ivd* r0, ivd* r1, ivd* r2, ivd* tot) {
        ivd mx = z;
        for (int j = 0; j <= n; ++j) {
            const ivd t = r0[j] + r1[j] + r2[j];
            if (tot) tot[j] = t;
            mx = ivmax(mx, t);
        }
        ivd sc;
        bool any = false;
        for (int l = 0; l < IW; ++l) {
            int k = 0;
            if (!(mx[l] <= DMAX)) bad[l] = true;
            else if (mx[l] > 0.0) k = std::ilogb(mx[l]);
            kv[l] = k;
            any |= (k != 0);
            sc[l] = std::ldexp(1.0, -k);
        }
        if (!any) return;
        for (int j = 0; j <= n; ++j) { r0[j] *= sc; r1[j] *= sc; r2[j] *= sc; }
        if (tot) for (int j = 0; j <= n; ++j) tot[j] *= sc;
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
        ivd s = z;
        for (int j = 0; j <= n; ++j) { tq[j] = FM[j] + FX[j] + FY[j]; s += tq[j]; }
        rs[0] = s;
    }
    for (int i = 1; i <= M; ++i) {
        fill_E(i, Ec);
        const size_t ro = static_cast<size_t>(i) * st;
        ivd* rM = FM + ro; ivd* rX = FX + ro; ivd* rY = FY + ro;
        const ivd* pM = rM - st; const ivd* pX = rX - st; const ivd* pY = rY - st;
        ivd fr = z;
        if (local) for (int l = 0; l < IW; ++l) fr[l] = std::ldexp(1.0, -S[(i - 1) * IW + l]);
        if (local) { rM[0] = fr; rX[0] = z; rY[0] = z; }
        else       { rM[0] = z; rY[0] = z; rX[0] = (pM[0] + pY[0]) * ob + pX[0] * eb; }
        for (int j = 1; j <= n; ++j) {
            rM[j] = tq[j - 1] * Ec[j] + fr;
            rX[j] = (pM[j] + pY[j]) * ob + pX[j] * eb;
        }
        ivd c = rY[0];
        for (int j = 1; j <= n; ++j) {
            c = (rM[j - 1] + rX[j - 1]) * oa + c * ea;
            rY[j] = c;
        }
        rescale(rM, rX, rY, tq);
        ivd s = z;
        for (int j = 0; j <= n; ++j) s += tq[j];
        rs[i] = s;
        for (int l = 0; l < IW; ++l) S[i * IW + l] = S[(i - 1) * IW + l] + kv[l];
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
        if (!(zr > 0.0)) { bad[l] = true; zr = 1.0; }
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
        if (i < M) fill_E(i + 1, En);
        for (int j = 0; j <= n; ++j) {
            const ivd d = (i < M ? En[j + 1] * nM[j + 1] : z) + init;
            const ivd v = nX[j];
            const ivd b1 = d + v * ob;
            cM[j] = b1; cX[j] = d + v * eb; cY[j] = b1;
        }
        if (!local) { cM[n] = ivsel(top, one, cM[n]); cX[n] = ivsel(top, one, cX[n]); cY[n] = ivsel(top, one, cY[n]); }
        cY[n + 1] = z;
        {
            ivd c = z;
            for (int j = n; j >= 0; --j) { c = cY[j] + ea * c; cY[j] = c; }
        }
        for (int j = 0; j <= n; ++j) {
            const ivd h = cY[j + 1] * oa;
            cM[j] += h; cX[j] += h;
        }
        rescale(cM, cX, cY, nullptr);

        ivd gh, gv = z;
        int Ti[IW];
        for (int l = 0; l < IW; ++l) {
            Ti[l] = Tn[l] + kv[l];
            if (i > J.m[l]) { gh[l] = 0.0; continue; }
            gh[l] = std::ldexp(izr[l], S[i * IW + l] + Ti[l] - ze[l]);
            if (i >= 1) gv[l] = std::ldexp(izr[l], S[(i - 1) * IW + l] + Ti[l] - ze[l]);
            if (!(gh[l] <= DMAX) || !(gv[l] <= DMAX)) bad[l] = true;
            const int kf = i > 0 ? S[i * IW + l] - S[(i - 1) * IW + l] : 0, kb = kv[l];
            const double lf = kf < 0 ? std::ldexp(1.0, -kf) : 1.0;
            const double lb = kb < 0 ? std::ldexp(1.0, -kb) : 1.0;
            loss[l] += 32.0 * (n + 1) * DMIN * (lf + lb) * gh[l];
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
            fill_E(i, Ec);
            const ivd* qM = fM - st; const ivd* qX = fX - st; const ivd* qY = fY - st;
            for (int j = kmin; j <= n; ++j) sob += (qM[j] + qY[j]) * cX[j];
            for (int c = 0; c < na; ++c) acc[c] = z;
            for (int j = 1; j <= n; ++j) {
                const ivd t = (qM[j - 1] + qX[j - 1] + qY[j - 1]) * Ec[j] * cM[j];
                const ivl* mk = &eqm[static_cast<size_t>(j) * na];
                for (int c = 0; c < na; ++c) acc[c] += ivsel(mk[c], t, z);
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
                for (int c = 0; c < na; ++c) gr[c] += acc[c][l] * gv[l];
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
