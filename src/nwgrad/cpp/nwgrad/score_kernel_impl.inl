// ── Leveled score-only kernel: the optimal score, rolling rows, no table ─────────
//
// #include'd once per ISA level inside the level namespace, after kernels_impl.inl
// (whose AVX-512/clang mask helpers it reuses).  The striped Full recurrence of
// striped_affine_full — Farrar's layout and query profile, the same left-associated
// (v - go) - ge, the same lazy-F — so every cell it computes is bit-identical to the
// table-writing fill, and hence to the scalar viterbi_affine.  What it drops is
// everything a score does not need:
//
//   * the tables: six striped rows (previous / current VM, VX, VY) that stay in L1,
//     instead of 24 B (12 at float32) per cell streamed through the cache;
//   * the Local argmax: the fill scans every cell of every row in scalar code to keep
//     the M>X>Y position the traceback needs; a score needs only the value, and max is
//     exact in any order, so a per-lane vector max reduced once at the end gives the
//     identical number;
//   * (Fused) the separate VY pass: VY is computed in the same pass as VM/VX, so the
//     serial gap carry overlaps the carry-free work instead of running alone.
//
// Fused vs two-pass, and why both are exact.  Column j's gap-in-A value is the serial
// chain Y(j) = max(O(j-1), Y(j-1) - ge_a), O = (max(M, X) - go_a) - ge_a.  In striped
// order the predecessor of a lane's FIRST segment is the previous lane's LAST segment,
// not yet computed when the fused pass reaches it, so the fused pass seeds lanes 1..W-1
// with -inf there and lazy-F's first round carries both the open O and the extension Y
// across each lane boundary.  Lazy-F stops only when no lane improves, i.e. when every
// cell satisfies the recurrence exactly; the recurrence has one solution, reached cell
// by cell with the same floating-point expressions, so the result equals the scalar
// chain bit for bit — the same argument the two-pass form (and the fill) rests on.

#ifndef NWGRAD_LEVEL_NS
#  error "score_kernel_impl.inl must be included inside a level namespace by a level TU"
#endif

template <class T, bool Local, bool Fused>
static void score_affine_striped(ScoreJob<T>& job) {
    using vd = stdx::native_simd<T>;
    const int W = (int)vd::size();
    const T NINF = -std::numeric_limits<T>::infinity();

    const int m = job.m, n = job.n, nalpha = job.nalpha;
    const T go_a = job.go_a, ge_a = job.ge_a, go_b = job.go_b, ge_b = job.ge_b;
    const unsigned char* a = job.a;
    const unsigned char* b = job.b;
    DpBufferT<T>& buf = *job.buf;

    if (n == 0) {
        // No columns: Global ends in one gap run down column 0 (or is empty); Local is 0.
        // The fill's (m, 0) cell: VM = -inf (0 at m == 0), VX = -(go_b + m*ge_b), VY = -inf.
        if constexpr (Local) job.score = 0.0;
        else job.score = (m == 0) ? 0.0 : static_cast<double>(-(go_b + static_cast<T>(m) * ge_b));
        return;
    }

    const int seg = (n + W - 1) / W;
    const std::size_t sw = (std::size_t)seg * W;
    if (buf.sprof.size() < (std::size_t)nalpha * sw) buf.sprof.resize((std::size_t)nalpha * sw);
    for (auto* v : {&buf.rM, &buf.rX, &buf.rY, &buf.qM, &buf.qX, &buf.qY})
        if (v->size() < sw) v->resize(sw);
    if constexpr (!Fused) if (buf.sopenv.size() < sw) buf.sopenv.resize(sw);
    // Local: the padding slots (columns > n) must not reach the maximum.  With a negative
    // gap extend a gap run GAINS score along the padding, so they can exceed every real
    // cell (the fill's argmax scans real columns only).  Added, not selected: x + 0 = x
    // exactly, x + -inf = -inf, and no mask op (see kernels_impl.inl's AVX-512/clang note).
    const T* pad = nullptr;
    if constexpr (Local) {
        if (buf.spad.size() < sw) buf.spad.resize(sw);
        for (int l = 0; l < W; ++l)
            for (int s = 0; s < seg; ++s)
                buf.spad[(std::size_t)s * W + l] = (l * seg + s + 1 <= n) ? T(0) : NINF;
        pad = buf.spad.data();
    }
    auto padded = [&](const vd& v, int s) {
        vd p; p.copy_from(pad + (std::size_t)s * W, stdx::element_aligned);
        return v + p;
    };

    // striped query profile, padding columns -inf (as striped_affine_full)
    for (int c = 0; c < nalpha; ++c) {
        const T* row = job.blk + (std::size_t)c * nalpha;
        T* dst = buf.sprof.data() + (std::size_t)c * sw;
        for (int l = 0; l < W; ++l)
            for (int s = 0; s < seg; ++s) {
                const int j = l * seg + s + 1;
                dst[(std::size_t)s * W + l] = (j <= n) ? row[b[j - 1]] : NINF;
            }
    }

    T* pM = buf.qM.data(); T* pX = buf.qX.data(); T* pY = buf.qY.data();   // row i-1
    T* cM = buf.rM.data(); T* cX = buf.rX.data(); T* cY = buf.rY.data();   // row i

    // row 0 (columns 1..n, striped); padding -inf, as the fill leaves it
    for (std::size_t k = 0; k < sw; ++k) { pM[k] = NINF; pX[k] = NINF; pY[k] = NINF; }
    for (int l = 0; l < W; ++l)
        for (int s = 0; s < seg; ++s) {
            const int j = l * seg + s + 1;
            if (j <= n) {
                const std::size_t k = (std::size_t)s * W + l;
                if constexpr (Local) pM[k] = 0;
                else                 pY[k] = -(go_a + j * ge_a);
            }
        }

    const vd vgo_a(go_a), vge_a(ge_a), vgo_b(go_b), vge_b(ge_b);
    const vd vzero(static_cast<T>(0)), vninf(NINF);
    vd vbest = vzero;                       // Local: running per-lane max (best starts at 0)
    T bM = 0, bX = NINF, bY = NINF;         // column 0 of the previous row

    // Is any lane of a greater than b?  (The lazy-F early exit — the one mask op.)
    auto any_gt = [](const vd& x, const vd& y) -> bool {
#if defined(__clang__) && defined(__AVX512F__) && !defined(NWGRAD_STD_SIMD_AVX512_MASK_OK)
        if constexpr (std::is_same_v<T, double>) return avx512d_gt(x, y) != 0;
        else                                     return stdx::any_of(x > y);
#else
        return stdx::any_of(x > y);
#endif
    };

    for (int i = 1; i <= m; ++i) {
        const T nbM = Local ? T(0) : NINF;
        const T nbX = Local ? NINF : -(go_b + i * ge_b);
        const T nbOpen = (std::max(nbM, nbX) - go_a) - ge_a;
        const T* sub = buf.sprof.data() + (std::size_t)a[i - 1] * sw;

        vd Olast = vninf;                   // Fused: O of the last segment, for lazy-F
        if constexpr (Fused) {
            vd Oin([&](int q) { return q == 0 ? nbOpen : NINF; });
            vd Yin = vninf;                 // column 0's VY is -inf; lanes 1.. by lazy-F
            for (int s = 0; s < seg; ++s) {
                vd dM, dX, dY;
                if (s == 0) {
                    vd lM, lX, lY;
                    lM.copy_from(pM + (std::size_t)(seg - 1) * W, stdx::element_aligned);
                    lX.copy_from(pX + (std::size_t)(seg - 1) * W, stdx::element_aligned);
                    lY.copy_from(pY + (std::size_t)(seg - 1) * W, stdx::element_aligned);
                    dM = vd([&](int q) { return q == 0 ? bM : lM[q - 1]; });
                    dX = vd([&](int q) { return q == 0 ? bX : lX[q - 1]; });
                    dY = vd([&](int q) { return q == 0 ? bY : lY[q - 1]; });
                } else {
                    dM.copy_from(pM + (std::size_t)(s - 1) * W, stdx::element_aligned);
                    dX.copy_from(pX + (std::size_t)(s - 1) * W, stdx::element_aligned);
                    dY.copy_from(pY + (std::size_t)(s - 1) * W, stdx::element_aligned);
                }
                vd sb; sb.copy_from(sub + (std::size_t)s * W, stdx::element_aligned);
                vd vmv = stdx::max(stdx::max(dM, dX), dY) + sb;
                if constexpr (Local) vmv = stdx::max(vmv, vzero);
                vmv.copy_to(cM + (std::size_t)s * W, stdx::element_aligned);

                vd uM, uX, uY;
                uM.copy_from(pM + (std::size_t)s * W, stdx::element_aligned);
                uX.copy_from(pX + (std::size_t)s * W, stdx::element_aligned);
                uY.copy_from(pY + (std::size_t)s * W, stdx::element_aligned);
                vd vxv = stdx::max(stdx::max((uM - vgo_b) - vge_b, uX - vge_b), (uY - vgo_b) - vge_b);
                vxv.copy_to(cX + (std::size_t)s * W, stdx::element_aligned);

                vd vyv = stdx::max(Oin, Yin - vge_a);
                vyv.copy_to(cY + (std::size_t)s * W, stdx::element_aligned);
                if constexpr (Local) vbest = stdx::max(vbest, padded(stdx::max(stdx::max(vmv, vxv), vyv), s));
                Oin = (stdx::max(vmv, vxv) - vgo_a) - vge_a;
                Yin = vyv;
            }
            Olast = Oin;
        } else {
            T* ov = buf.sopenv.data();
            for (int s = 0; s < seg; ++s) {
                vd dM, dX, dY;
                if (s == 0) {
                    vd lM, lX, lY;
                    lM.copy_from(pM + (std::size_t)(seg - 1) * W, stdx::element_aligned);
                    lX.copy_from(pX + (std::size_t)(seg - 1) * W, stdx::element_aligned);
                    lY.copy_from(pY + (std::size_t)(seg - 1) * W, stdx::element_aligned);
                    dM = vd([&](int q) { return q == 0 ? bM : lM[q - 1]; });
                    dX = vd([&](int q) { return q == 0 ? bX : lX[q - 1]; });
                    dY = vd([&](int q) { return q == 0 ? bY : lY[q - 1]; });
                } else {
                    dM.copy_from(pM + (std::size_t)(s - 1) * W, stdx::element_aligned);
                    dX.copy_from(pX + (std::size_t)(s - 1) * W, stdx::element_aligned);
                    dY.copy_from(pY + (std::size_t)(s - 1) * W, stdx::element_aligned);
                }
                vd sb; sb.copy_from(sub + (std::size_t)s * W, stdx::element_aligned);
                vd vmv = stdx::max(stdx::max(dM, dX), dY) + sb;
                if constexpr (Local) vmv = stdx::max(vmv, vzero);
                vmv.copy_to(cM + (std::size_t)s * W, stdx::element_aligned);
                vd uM, uX, uY;
                uM.copy_from(pM + (std::size_t)s * W, stdx::element_aligned);
                uX.copy_from(pX + (std::size_t)s * W, stdx::element_aligned);
                uY.copy_from(pY + (std::size_t)s * W, stdx::element_aligned);
                vd vxv = stdx::max(stdx::max((uM - vgo_b) - vge_b, uX - vge_b), (uY - vgo_b) - vge_b);
                vxv.copy_to(cX + (std::size_t)s * W, stdx::element_aligned);
                if constexpr (Local) vbest = stdx::max(vbest, padded(stdx::max(vmv, vxv), s));
                ((stdx::max(vmv, vxv) - vgo_a) - vge_a).copy_to(ov + (std::size_t)s * W, stdx::element_aligned);
            }
            vd prev([&](int q) { return q == 0 ? bY : NINF; });
            for (int s = 0; s < seg; ++s) {
                vd O;
                if (s == 0) {
                    vd lo; lo.copy_from(ov + (std::size_t)(seg - 1) * W, stdx::element_aligned);
                    O = vd([&](int q) { return q == 0 ? nbOpen : lo[q - 1]; });
                } else {
                    O.copy_from(ov + (std::size_t)(s - 1) * W, stdx::element_aligned);
                }
                vd v = stdx::max(O, prev - vge_a);
                v.copy_to(cY + (std::size_t)s * W, stdx::element_aligned);
                if constexpr (Local) vbest = stdx::max(vbest, padded(v, s));
                prev = v;
            }
        }

        // ── lazy-F: carry the gap-in-A run across the lane boundaries ──
        // Round 0 of the fused form also carries the OPEN from the previous lane's last
        // column (seeded -inf above); later rounds, and every round of the two-pass form
        // (whose carry pass already saw the opens), carry the extension only.
        for (int r = 0; r < W; ++r) {
            vd last; last.copy_from(cY + (std::size_t)(seg - 1) * W, stdx::element_aligned);
            vd F([&](int q) { return q == 0 ? NINF : last[q - 1]; });
            F = F - vge_a;
            if constexpr (Fused) {
                if (r == 0) {
                    vd Os([&](int q) { return q == 0 ? NINF : Olast[q - 1]; });
                    F = stdx::max(Os, F);
                }
            }
            bool changed = false;
            for (int s = 0; s < seg; ++s) {
                vd v; v.copy_from(cY + (std::size_t)s * W, stdx::element_aligned);
                if (!any_gt(F, v)) break;
                v = stdx::max(v, F);
                v.copy_to(cY + (std::size_t)s * W, stdx::element_aligned);
                if constexpr (Local) vbest = stdx::max(vbest, padded(v, s));
                F = v - vge_a;
                changed = true;
            }
            if (!changed) break;
        }

        std::swap(pM, cM); std::swap(pX, cX); std::swap(pY, cY);
        bM = nbM; bX = nbX; bY = NINF;
    }

    if constexpr (Local) {
        T best = 0;
        for (int q = 0; q < W; ++q) best = std::max(best, static_cast<T>(vbest[q]));
        job.score = static_cast<double>(best);
    } else {
        // (m, n) in the last row computed (row 0 when m == 0), striped
        const std::size_t k = (std::size_t)((n - 1) % seg) * W + (std::size_t)((n - 1) / seg);
        job.score = static_cast<double>(std::max({pM[k], pX[k], pY[k]}));
    }
}

// The registered entry: Global or Local, fused (default) or two-pass (job.twopass, set
// from NWGRAD_SCORE_VARIANT=twopass by the caller, to measure the two in one binary).
template <class T>
static void score_entry(ScoreJob<T>& job) {
    if (job.twopass) {
        if (job.local) score_affine_striped<T, true, false>(job);
        else           score_affine_striped<T, false, false>(job);
    } else {
        if (job.local) score_affine_striped<T, true, true>(job);
        else           score_affine_striped<T, false, true>(job);
    }
}
