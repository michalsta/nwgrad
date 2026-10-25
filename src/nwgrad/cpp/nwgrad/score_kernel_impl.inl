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

// Tr: a tracked problem (job.tr; see striped_affine_full) — each row's penalties,
// border and profile row from the track arrays.
template <class T, bool Local, bool Fused, bool Tr = false>
static void score_affine_striped(ScoreJob<T>& job) {
    using vd = stdx::native_simd<T>;
    const int W = (int)vd::size();
    const T NINF = -std::numeric_limits<T>::infinity();

    const int m = job.m, n = job.n, nalpha = job.nalpha;
    T go_a = job.go_a, ge_a = job.ge_a, go_b = job.go_b, ge_b = job.ge_b;
    const TrackJob<T>& tr = job.tr;
    if constexpr (Tr) { go_a = tr.go_a[0]; ge_a = tr.ge_a[0]; }
    const unsigned char* a = job.a;
    const unsigned char* b = job.b;
    DpBufferT<T>& buf = *job.buf;

    if (n == 0) {
        // No columns: Global ends in one gap run down column 0 (or is empty); Local is 0.
        // The fill's (m, 0) cell: VM = -inf (0 at m == 0), VX = -(go_b + m*ge_b), VY = -inf.
        if constexpr (Local) job.score = 0.0;
        else if constexpr (Tr) job.score = (m == 0) ? 0.0 : static_cast<double>(tr.bx0[m]);
        else job.score = (m == 0) ? 0.0 : static_cast<double>(-(go_b + static_cast<T>(m) * ge_b));
        return;
    }

    const int seg = (n + W - 1) / W;
    const std::size_t sw = (std::size_t)seg * W;
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
    striped_profile_build<T, Tr>(buf.sprof, job.blk, nalpha, b, n, seg, W, tr);

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

    vd vgo_a(go_a), vge_a(ge_a), vgo_b(go_b), vge_b(ge_b);
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
        if constexpr (Tr) {
            go_a = tr.go_a[i]; ge_a = tr.ge_a[i]; go_b = tr.go_b[i]; ge_b = tr.ge_b[i];
            vgo_a = vd(go_a); vge_a = vd(ge_a); vgo_b = vd(go_b); vge_b = vd(ge_b);
        }
        const T nbM = Local ? T(0) : NINF;
        T nbX;
        if constexpr (Tr) nbX = Local ? NINF : tr.bx0[i];
        else              nbX = Local ? NINF : -(go_b + i * ge_b);
        const T nbOpen = (std::max(nbM, nbX) - go_a) - ge_a;
        const T* sub = striped_row_profile<T, Tr>(buf.sprof.data(), job.blk, nalpha, a, b, n, seg, W, i, tr);

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
                    dM = lane_shift_in(lM, bM);
                    dX = lane_shift_in(lX, bX);
                    dY = lane_shift_in(lY, bY);
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
                    dM = lane_shift_in(lM, bM);
                    dX = lane_shift_in(lX, bX);
                    dY = lane_shift_in(lY, bY);
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
                    O = lane_shift_in(lo, nbOpen);
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
            vd F = lane_shift_in(last, NINF);
            F = F - vge_a;
            if constexpr (Fused) {
                if (r == 0) {
                    vd Os = lane_shift_in(Olast, NINF);
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
                // F - g, not v - g: the same stored maxima and exit test, one subtraction
                // on the chain (see score_affine_striped_pos's lazy-F)
                F = F - vge_a;
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

// ── Fused, non-negative opens: the previous row as D = max(M, X, Y) and X only ──
// With go >= 0 an open from a cell is never above the extension from the same cell
// ((v - go) - ge <= v - ge, rounding being monotone), so X's open may be offered every
// state of the cell above: X = max((D_up - go_b) - ge_b, X_up - ge_b) — the same maximum
// as the three-operand form — and the diagonal is D(i-1, j-1).  The previous row needs D
// and X only; the current row still writes Y, which lazy-F corrects in place (each
// correction raises that cell's D too: Y only grows, so D = max(D, v) is exact).  Y's
// own open stays (max(M, X) - go_a) - ge_a, OFF the loop-carried chain: taking it from D
// put a max of three and two subtractions on the carry and made the inter-pair form
// slower than the fill (1.03x) before it was moved back.  MOnly (extends >= 0 too): no
// gap cell exceeds the cell it extends from, so the Local maximum is over M alone — and
// M's padding slots are clamped to exactly 0, so they need no mask.
// The row is stored as max(M, X), X and Y (D formed on reading); Y's carry steps two
// segments at a time and lazy-F's carry is a bare subtraction — see the notes in the body.
template <class T, bool Local, bool MOnly, bool Tr = false>
static void score_affine_striped_pos(ScoreJob<T>& job) {
    using vd = stdx::native_simd<T>;
    const int W = (int)vd::size();
    const T NINF = -std::numeric_limits<T>::infinity();

    const int m = job.m, n = job.n, nalpha = job.nalpha;
    T go_a = job.go_a, ge_a = job.ge_a, go_b = job.go_b, ge_b = job.ge_b;
    const TrackJob<T>& tr = job.tr;
    if constexpr (Tr) { go_a = tr.go_a[0]; ge_a = tr.ge_a[0]; }
    const unsigned char* a = job.a;
    const unsigned char* b = job.b;
    DpBufferT<T>& buf = *job.buf;

    if (n == 0) {
        if constexpr (Local) job.score = 0.0;
        else if constexpr (Tr) job.score = (m == 0) ? 0.0 : static_cast<double>(tr.bx0[m]);
        else job.score = (m == 0) ? 0.0 : static_cast<double>(-(go_b + static_cast<T>(m) * ge_b));
        return;
    }

    const int seg = (n + W - 1) / W;
    const std::size_t sw = (std::size_t)seg * W;
    for (auto* v : {&buf.rM, &buf.rX, &buf.rY, &buf.qM, &buf.qX, &buf.qY})
        if (v->size() < sw) v->resize(sw);
    const T* pad = nullptr;
    if constexpr (Local && !MOnly) {
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

    striped_profile_build<T, Tr>(buf.sprof, job.blk, nalpha, b, n, seg, W, tr);

    // The rows keep A = max(M, X), X and Y; D = max(A, Y) is formed where a row is read
    // (max is exact, so that is D bit for bit).  Lazy-F then raises Y alone: one load and
    // one store per visit instead of two each.
    T* pA = buf.qM.data(); T* pX = buf.qX.data(); T* pY = buf.qY.data();   // row i-1
    T* cA = buf.rM.data(); T* cX = buf.rX.data(); T* cY = buf.rY.data();   // row i

    // row 0 (three-state: Global M = X = -inf, Y = -(go + j*ge); Local M = 0); padding -inf
    for (std::size_t k = 0; k < sw; ++k) { pA[k] = NINF; pX[k] = NINF; pY[k] = NINF; }
    for (int l = 0; l < W; ++l)
        for (int s = 0; s < seg; ++s) {
            const int j = l * seg + s + 1;
            if (j <= n) {
                const std::size_t k = (std::size_t)s * W + l;
                if constexpr (Local) pA[k] = 0;
                else                 pY[k] = -(go_a + j * ge_a);
            }
        }

    vd vgo_a(go_a), vge_a(ge_a), vgo_b(go_b), vge_b(ge_b);
    const vd vzero(static_cast<T>(0)), vninf(NINF);
    vd vbest = vzero;
    T bD = 0;                               // D of column 0, previous row (M(0,0) = 0)

    auto any_gt = [](const vd& x, const vd& y) -> bool {
#if defined(__clang__) && defined(__AVX512F__) && !defined(NWGRAD_STD_SIMD_AVX512_MASK_OK)
        if constexpr (std::is_same_v<T, double>) return avx512d_gt(x, y) != 0;
        else                                     return stdx::any_of(x > y);
#else
        return stdx::any_of(x > y);
#endif
    };
    auto ld = [](const T* p) { vd v; v.copy_from(p, stdx::element_aligned); return v; };

    for (int i = 1; i <= m; ++i) {
        if constexpr (Tr) {
            go_a = tr.go_a[i]; ge_a = tr.ge_a[i]; go_b = tr.go_b[i]; ge_b = tr.ge_b[i];
            vgo_a = vd(go_a); vge_a = vd(ge_a); vgo_b = vd(go_b); vge_b = vd(ge_b);
        }
        const T nbM = Local ? T(0) : NINF;
        T nbX;
        if constexpr (Tr) nbX = Local ? NINF : tr.bx0[i];
        else              nbX = Local ? NINF : -(go_b + i * ge_b);
        const T nbD = std::max(nbM, nbX);   // column 0's VY is -inf
        const T nbOpen = (nbD - go_a) - ge_a;
        const T* sub = striped_row_profile<T, Tr>(buf.sprof.data(), job.blk, nalpha, a, b, n, seg, W, i, tr);

        vd Oin([&](int q) { return q == 0 ? nbOpen : NINF; });   // Y's open from the column before
        vd Yin = vninf;                                           // Y of the column before
        vd dD = lane_shift_in(stdx::max(ld(pA + (std::size_t)(seg - 1) * W),
                                        ld(pY + (std::size_t)(seg - 1) * W)), bD);
        // One segment's carry-free part: X stored, max(M, X) returned; dD moves on.
        auto cell = [&](int s, vd& vmv) NWGRAD_ALWAYS_INLINE_LAMBDA {
            const std::size_t o = (std::size_t)s * W;
            const vd uD = stdx::max(ld(pA + o), ld(pY + o));
            vmv = dD + ld(sub + o);
            if constexpr (Local) vmv = stdx::max(vmv, vzero);
            const vd vxv = stdx::max((uD - vgo_b) - vge_b, ld(pX + o) - vge_b);
            vxv.copy_to(cX + o, stdx::element_aligned);
            dD = uD;
            return stdx::max(vmv, vxv);
        };
        auto finish = [&](int s, const vd& vmv, const vd& mx, const vd& vyv) NWGRAD_ALWAYS_INLINE_LAMBDA {
            const std::size_t o = (std::size_t)s * W;
            vyv.copy_to(cY + o, stdx::element_aligned);
            mx.copy_to(cA + o, stdx::element_aligned);
            if constexpr (Local) {
                if constexpr (MOnly) vbest = stdx::max(vbest, vmv);
                else                 vbest = stdx::max(vbest, padded(stdx::max(mx, vyv), s));
            }
        };
        // Y(s) = max(O(s-1), Y(s-1) - ge): the loop-carried chain.  Two segments per step:
        // rounding is monotone, so max(a, b) - g == max(a - g, b - g) bit for bit, and
        //   Y(s+1) = max(O(s), Y(s) - g) = max(max(O(s), O(s-1) - g), (Y(s-1) - g) - g)
        // whose chain is two subtractions and ONE max per two segments (was two of each);
        // the inner max is off it (pinned: gcc re-associates it back otherwise).
        int s = 0;
        for (; s + 1 < seg; s += 2) {
            vd vm0, vm1;
            const vd mx0 = cell(s, vm0);
            const vd mx1 = cell(s + 1, vm1);
            const vd O0 = (mx0 - vgo_a) - vge_a;
            const vd t = Yin - vge_a;
            const vd y0 = stdx::max(Oin, t);
            const vd y1 = stdx::max(carry_pin(stdx::max(O0, Oin - vge_a)), t - vge_a);
            finish(s, vm0, mx0, y0);
            finish(s + 1, vm1, mx1, y1);
            Oin = (mx1 - vgo_a) - vge_a;
            Yin = y1;
        }
        if (s < seg) {
            vd vm0;
            const vd mx0 = cell(s, vm0);
            const vd y0 = stdx::max(Oin, Yin - vge_a);
            finish(s, vm0, mx0, y0);
            Oin = (mx0 - vgo_a) - vge_a;
            Yin = y0;
        }
        const vd Olast = Oin;

        // ── lazy-F: carry the gap-in-A run across the lane boundaries ──
        // The carry into segment s+1 is max(c(s), F) - g; it is taken as F - g.  The row
        // satisfies Y(s+1) >= Y(s) - g in every lane (the main pass computed it so, and a
        // raised cell is followed by a visit to the next), so where c(s) wins, c(s) - g <=
        // c(s+1) and neither form raises s+1: the stored maxima and the exit test are the
        // same, and the chain is one subtraction per segment, no max.
        for (int r = 0; r < W; ++r) {
            vd F = lane_shift_in(ld(cY + (std::size_t)(seg - 1) * W), NINF) - vge_a;
            if (r == 0) F = stdx::max(lane_shift_in(Olast, NINF), F);
            bool changed = false;
            for (int s2 = 0; s2 < seg; ++s2) {
                vd v = ld(cY + (std::size_t)s2 * W);
                if (!any_gt(F, v)) break;
                v = stdx::max(v, F);
                v.copy_to(cY + (std::size_t)s2 * W, stdx::element_aligned);
                if constexpr (Local && !MOnly) vbest = stdx::max(vbest, padded(v, s2));
                F = F - vge_a;
                changed = true;
            }
            if (!changed) break;
        }

        std::swap(pA, cA); std::swap(pX, cX); std::swap(pY, cY);
        bD = nbD;
    }

    if constexpr (Local) {
        T best = 0;
        for (int q = 0; q < W; ++q) best = std::max(best, static_cast<T>(vbest[q]));
        job.score = static_cast<double>(best);
    } else {
        const std::size_t k = (std::size_t)((n - 1) % seg) * W + (std::size_t)((n - 1) / seg);
        job.score = static_cast<double>(std::max(pA[k], pY[k]));
    }
}

// The registered entry: Global or Local; job.variant 0 = the D/X form when both opens
// are >= 0 (else fused three-state), 1 = two-pass, 2 = fused three-state (for measuring
// the forms in one binary; NWGRAD_SCORE_VARIANT=twopass / fused3).
template <class T, bool Tr>
static void score_entry_impl(ScoreJob<T>& job) {
    if (job.variant == 1) {
        if (job.local) score_affine_striped<T, true, false, Tr>(job);
        else           score_affine_striped<T, false, false, Tr>(job);
        return;
    }
    // Tracked: the forms' conditions must hold on every row (NaN: false).
    bool pos_open, pos_ext;
    if constexpr (Tr) {
        const TrackJob<T>& tr = job.tr;
        pos_open = pos_ext = true;
        for (int i = 0; i <= job.m; ++i) {
            pos_open &= tr.go_a[i] >= T(0);
            pos_ext  &= tr.ge_a[i] >= T(0);
            if (i > 0) { pos_open &= tr.go_b[i] >= T(0); pos_ext &= tr.ge_b[i] >= T(0); }
        }
    } else {
        pos_open = job.go_a >= T(0) && job.go_b >= T(0);   // NaN: false
        pos_ext  = job.ge_a >= T(0) && job.ge_b >= T(0);
    }
    if (job.variant == 0 && pos_open) {
        if (job.local) {
            if (pos_ext) score_affine_striped_pos<T, true, true, Tr>(job);
            else         score_affine_striped_pos<T, true, false, Tr>(job);
        } else {
            score_affine_striped_pos<T, false, false, Tr>(job);
        }
        return;
    }
    if (job.local) score_affine_striped<T, true, true, Tr>(job);
    else           score_affine_striped<T, false, true, Tr>(job);
}
template <class T>
static void score_entry(ScoreJob<T>& job) {
    if (job.tr.on()) score_entry_impl<T, true>(job);
    else             score_entry_impl<T, false>(job);
}
