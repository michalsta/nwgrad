// ── Grid forward pass (TracebackMode::Grid): the score recurrence over a rectangle ──
//
// #include'd once per ISA level inside that level's namespace, after kernels_impl.inl
// (lane_shift_in, carry_pin, spread_rows, the AVX-512/clang mask helpers).  The job and
// its output layout: GridFwdJob (simd_levels.hpp); the driver: Aligner::viterbi_affine_grid.
//
// score_affine_striped_pos_fp's recurrence (score_kernel_impl.inl) — the D/X form with the
// shortened gap carry: rows of A = max(M, X), X and Y; Y's carry two segments per dependent
// step; lazy-F carrying F - ge — generalized in two ways:
//   * the rectangle's top row and left column are given (all three states) instead of the
//     closed-form borders: the previous row's D = max(M, X, Y) and X come from the top line,
//     and column c0 of row i seeds lane 0 — the diagonal operand D(i-1, c0), the open into
//     column c0+1 (max(M, X)(i, c0) - go_a) - ge_a, and the extension Y(i, c0) - ge_a;
//   * every R-th row is saved dense, and q sets of SEGMENT-ALIGNED columns every row: set t
//     sits at segment st[t], so its W lanes are the columns c0 + l*seg + st[t] + 1 and one
//     vector store per state saves them.
// M is not kept by the D/X form; at a saved line it is formed again as D(i-1, j-1) + s,
// the sum the row pass formed.  Every value is therefore the full fill's, bit for bit
// (the D/X form's argument: opens >= 0), which is what makes the grid's re-filled blocks
// reproduce the Pointers fill's codes.
//
// Integer T (2026-11-05): lazy-F carrying F - ge would step an integer sentinel down by ge
// per segment (it could wrap), so Y takes score_affine_striped_pos's FUSED exact prefix-max
// carry from 4 lanes (the opens never reach memory, the ramp in registers, one pass after
// the row applies the lanes' carry-in and subtracts the ramp: exact, max being
// associative), and the plain exact lazy-F (F = max(v, F) - ge) below 4 lanes (int64 at
// 128 bits).  The left column's Y enters as one more open at column c0: max(open(c0),
// Y(c0) - ge).  Saturating int16 (Sat): every add / subtract saturates and the sentinel is
// lowest() -- the score kernel's regime, whose bound (nw_int_bound_check) keeps every
// sentinel-derived value below every reachable one, so the walk's codes (read on reachable
// cells only) are the full fill's.  An integer profile pads with 0 (nw_pad), as every
// striped kernel does.

#ifndef NWGRAD_LEVEL_NS
#  error "grid_kernel_impl.inl must be included inside a level namespace by a level TU"
#endif

template <class T, bool Local, bool Sat, bool Best = false>
static void grid_fwd_impl(GridFwdJob<T>& J) {
    using vd = stdx::native_simd<T>;
    const int W = (int)vd::size();
    constexpr bool kInt = nw_is_int<T>;
    const T NINF = nw_ninf_s<T, Sat>;
    const int h = J.r1 - J.r0, w = J.c1 - J.c0, nalpha = J.nalpha;
    const int seg = (w + W - 1) / W;
    const std::size_t sw = (std::size_t)seg * W;
    const T go_a = J.go_a, ge_a = J.ge_a, go_b = J.go_b, ge_b = J.ge_b;
    DpBufferT<T>& buf = *J.buf;

    std::size_t rs; T* const rb = spread_rows<T>(buf.grows, sw, 6, rs);
    T* pA = rb + 3 * rs; T* pX = rb + 4 * rs; T* pY = rb + 5 * rs;   // row i-1
    T* cA = rb;          T* cX = rb + rs;     T* cY = rb + 2 * rs;   // row i
    if (buf.gprof.size() < (std::size_t)nalpha * sw) buf.gprof.resize((std::size_t)nalpha * sw);
    buf.gbuilt.assign((std::size_t)nalpha, 0);
    const unsigned char* bb = J.b + J.c0;   // column c0 + j reads bb[j - 1]
    auto top  = [&](int s, int jj) { return J.top[(std::size_t)s * J.top_ss + (std::size_t)jj * J.top_es]; };
    auto left = [&](int s, int ii) { return J.left[(std::size_t)s * J.left_ss + (std::size_t)ii * J.left_es]; };

    // the top row (r0), striped: A = max(M, X), X, Y; padding -inf
    for (int l = 0; l < W; ++l)
        for (int s = 0; s < seg; ++s) {
            const int j = l * seg + s + 1;
            const std::size_t k = (std::size_t)s * W + l;
            if (j <= w) { pA[k] = std::max(top(0, j), top(1, j)); pX[k] = top(1, j); pY[k] = top(2, j); }
            else        { pA[k] = NINF; pX[k] = NINF; pY[k] = NINF; }
        }
    T bD = std::max({top(0, 0), top(1, 0), top(2, 0)});

    // the outputs: saved rows, column sets (set t at segment st[t]), their column ids
    const int q = std::max(1, std::min(seg, (seg + std::max(1, J.spacing) - 1) / std::max(1, J.spacing)));
    const int nrows = h >= 1 ? (h - 1) / J.R : 0;
    J.rows->resize((std::size_t)nrows * 3 * (w + 1));
    J.cols->resize((std::size_t)q * 3 * (h + 1) * W);
    J.colid->resize((std::size_t)q * W);
    T* const rows = J.rows->data(); T* const cols = J.cols->data(); int* const colid = J.colid->data();
    std::vector<int> st((std::size_t)q);
    for (int t = 0; t < q; ++t) {
        st[(std::size_t)t] = (int)(((long long)(t + 1) * seg) / q) - 1;
        for (int l = 0; l < W; ++l) {
            const int c = J.c0 + l * seg + st[(std::size_t)t] + 1;
            colid[t * W + l] = c < J.c1 ? c : -1;
            for (int s3 = 0; s3 < 3; ++s3)   // row r0 of each column line, from the top line
                cols[(((std::size_t)t * 3 + s3) * (h + 1)) * W + l] = c <= J.c1 ? top(s3, c - J.c0) : NINF;
        }
    }

    const vd vgo_a(go_a), vge_a(ge_a), vgo_b(go_b), vge_b(ge_b);
    const vd vgoe_a = vgo_a + vge_a, vgoe_b = vgo_b + vge_b;   // open_pen (integer T)
    const vd vninf(NINF);
    auto any_gt = [](const vd& x, const vd& y) -> bool {
#if defined(__clang__) && defined(__AVX512F__) && !defined(NWGRAD_STD_SIMD_AVX512_MASK_OK)
        if constexpr (sizeof(T) == 8) return avx512d_gt(x, y) != 0;
        else                          return stdx::any_of(x > y);
#else
        return stdx::any_of(x > y);
#endif
    };
    auto ld = [](const T* p) NWGRAD_ALWAYS_INLINE_LAMBDA { vd v; v.copy_from(p, stdx::element_aligned); return v; };
    auto stv = [](const vd& v, T* p) NWGRAD_ALWAYS_INLINE_LAMBDA { v.copy_to(p, stdx::element_aligned); };
    auto Dp = [&](int s) NWGRAD_ALWAYS_INLINE_LAMBDA {   // D of the previous row, segment s
        return stdx::max(ld(pA + (std::size_t)s * W), ld(pY + (std::size_t)s * W));
    };

    // Best (Local, the top level, every gap parameter >= 0): the best M cell in one pass,
    // as the Pointers kernel's `monly` -- per lane the largest M and the FIRST segment
    // reaching it (strict >), reduced after the row only when it beats the running best,
    // the first lane holding it being the leftmost column.  With the gap parameters >= 0
    // the first cell reaching a row's max(M, X, Y) is an M cell, so this is
    // local_row_best_'s cell, and it saves the separate end scan (hb_scan): skynet sse4,
    // int32 Local L1000 2.14 ns/cell against 4.56 with the scan (Pointers 3.57).  Padding
    // needs no mask: a padding M is D of the row above, at most the running best.
    // Best = false compiles all of it out (the Global instantiations are main's code).
    [[maybe_unused]] T best = T(0);
    [[maybe_unused]] int best_i = 0, best_j = 0;
    [[maybe_unused]] vd lmax, ls, vs;
    auto track = [&](const vd& vmv) NWGRAD_ALWAYS_INLINE_LAMBDA {
        if constexpr (Best) {
#if defined(__clang__) && defined(__AVX512F__) && !defined(NWGRAD_STD_SIMD_AVX512_MASK_OK)
            if constexpr (sizeof(T) == 8) ls = avx512d_blend(avx512d_gt(vmv, lmax), ls, vs);
            else                          stdx::where(vmv > lmax, ls) = vs;
#else
            stdx::where(vmv > lmax, ls) = vs;
#endif
            lmax = stdx::max(lmax, vmv);
            vs += vd(T(1));
        } else { (void)vmv; }
    };

    for (int ii = 1; ii <= h; ++ii) {
        if constexpr (Best) { lmax = vd(NINF); ls = vd(T(0)); vs = vd(T(0)); }
        const int i = J.r0 + ii;
        const T lM = left(0, ii), lX = left(1, ii), lY = left(2, ii);
        const T nbOpen = nw_sub_s<Sat>(nw_sub_s<Sat>(std::max(lM, lX), go_a), ge_a);
        const unsigned char ai = J.a[i - 1];
        T* const sub = buf.gprof.data() + (std::size_t)ai * sw;
        if (!buf.gbuilt[ai]) {   // the profile row of this letter, for this rectangle's columns
            buf.gbuilt[ai] = 1;
            const T* row = J.blk + (std::size_t)ai * nalpha;
            for (int l = 0; l < W; ++l)
                for (int s = 0; s < seg; ++s) {
                    const int j = l * seg + s + 1;
                    sub[(std::size_t)s * W + l] = (j <= w) ? row[bb[j - 1]] : nw_pad<T>;
                }
        }

        if constexpr (!kInt) {
        // ── the row pass: Y's carry two segments per dependent step (see
        // score_affine_striped_pos_fp) ──
        vd Oin([&](int qq) { return qq == 0 ? nbOpen : NINF; });   // Y's open from the column before
        vd Yin([&](int qq) { return qq == 0 ? lY : NINF; });       // Y of the column before
        vd dD = lane_shift_in(Dp(seg - 1), bD);
        auto cell = [&](int s, vd& mx) NWGRAD_ALWAYS_INLINE_LAMBDA {
            const vd uD = Dp(s);
            vd vmv = dD + ld(sub + (std::size_t)s * W);
            if constexpr (Local) vmv = stdx::max(vmv, vd(T(0)));   // viterbi_affine_ptr's clamp
            track(vmv);
            const vd vxv = stdx::max((uD - vgo_b) - vge_b, ld(pX + (std::size_t)s * W) - vge_b);
            mx = stdx::max(vmv, vxv);
            stv(vxv, cX + (std::size_t)s * W);
            dD = uD;
        };
        int s = 0;
        for (; s + 1 < seg; s += 2) {
            vd mx0, mx1;
            cell(s, mx0);
            cell(s + 1, mx1);
            const vd O0 = (mx0 - vgo_a) - vge_a;
            const vd t = Yin - vge_a;
            const vd y0 = stdx::max(Oin, t);
            const vd y1 = stdx::max(carry_pin(stdx::max(O0, Oin - vge_a)), t - vge_a);
            stv(y0, cY + (std::size_t)s * W);
            stv(y1, cY + (std::size_t)(s + 1) * W);
            stv(mx0, cA + (std::size_t)s * W);
            stv(mx1, cA + (std::size_t)(s + 1) * W);
            Oin = (mx1 - vgo_a) - vge_a;
            Yin = y1;
        }
        if (s < seg) {
            vd mx;
            cell(s, mx);
            const vd y = stdx::max(Oin, Yin - vge_a);
            stv(y, cY + (std::size_t)s * W);
            stv(mx, cA + (std::size_t)s * W);
            Oin = (mx - vgo_a) - vge_a;
            Yin = y;
        }
        const vd Olast = Oin;

        // ── lazy-F, carrying F - ge (lazy_f_next, kernels_impl.inl) ──
        for (int r = 0; r < W; ++r) {
            vd F = lane_shift_in(ld(cY + (std::size_t)(seg - 1) * W), NINF) - vge_a;
            if (r == 0) F = stdx::max(lane_shift_in(Olast, NINF), F);
            bool changed = false;
            for (int s2 = 0; s2 < seg; ++s2) {
                const vd v = ld(cY + (std::size_t)s2 * W);
                if (!any_gt(F, v)) break;
                stv(stdx::max(v, F), cY + (std::size_t)s2 * W);
                F = F - vge_a;
                changed = true;
            }
            if (!changed) break;
        }
        } else {
        // one cell of the integer row passes: M (clamped for Local), X, A = max(M, X) stored
        auto icell = [&](int s, const vd& dD, const vd& uD) NWGRAD_ALWAYS_INLINE_LAMBDA {
            vd vmv = vadd<Sat>(dD, ld(sub + (std::size_t)s * W));
            if constexpr (Local) vmv = stdx::max(vmv, vd(T(0)));
            track(vmv);
            const vd vxv = stdx::max(open_pen<Sat>(uD, vgo_b, vge_b, vgoe_b), vsub<Sat>(ld(pX + (std::size_t)s * W), vge_b));
            const vd mx = stdx::max(vmv, vxv);
            stv(vxv, cX + (std::size_t)s * W);
            stv(mx, cA + (std::size_t)s * W);
            return mx;
        };
        if (W >= 4) {
            // ── the fused prefix-max carry (score_affine_striped_pos): Q(s) = open(s-1) +
            // ramp(s) into a running prefix max per lane, stored to Y's row; the lanes'
            // carry-in and the ramp after the row ──
            const vd fBase([&](int l) { return static_cast<T>(static_cast<T>(l * seg) * ge_a); });
            vd fR = fBase, fP = vninf;
            vd fOprev = vsub<Sat>(vninf, fBase);   // segment 0 has no Q term here (see the score kernel)
            vd dD = lane_shift_in(Dp(seg - 1), bD);
            for (int s = 0; s < seg; ++s) {
                const vd uD = Dp(s);
                const vd mx = icell(s, dD, uD);
                dD = uD;
                fP = stdx::max(fP, vadd<Sat>(fOprev, fR));
                nv_store(cY + (std::size_t)s * W, fP);
                fOprev = open_pen<Sat>(mx, vgo_a, vge_a, vgoe_a);
                fR += vge_a;
            }
            // column c0 enters lane 0's segment 0 as one more open: its open and its Y extended
            const T seed = std::max(nbOpen, nw_sub_s<Sat>(lY, ge_a));
            const vd Q0 = vadd<Sat>(lane_shift_in(fOprev, seed), fBase);
            vd vcin;
            if constexpr (sizeof(T) == 2) {
                vcin = int_lane_carry_in<vd, (int)vd::size(), Sat>(stdx::max(Q0, fP), nw_sub<T>(NINF, ge_a));
            } else {
                alignas(64) T tot[64], cin[64];
                stdx::max(Q0, fP).copy_to(tot, stdx::element_aligned);
                T run = nw_sub<T>(NINF, ge_a);
                for (int l = 0; l < W; ++l) { cin[l] = run; run = std::max(run, tot[l]); }
                vcin.copy_from(cin, stdx::element_aligned);
            }
            vcin = stdx::max(vcin, Q0);
            vd rq = fBase;
            for (int s = 0; s < seg; ++s) {
                nv_store(cY + (std::size_t)s * W, vsub<Sat>(stdx::max(nv_load<vd>(cY + (std::size_t)s * W), vcin), rq));
                rq += vge_a;
            }
        } else {
            // ── below 4 lanes: the serial carry and the exact lazy-F (F = max(v, F) - ge) ──
            vd Oin([&](int qq) { return qq == 0 ? nbOpen : NINF; });
            vd Yin([&](int qq) { return qq == 0 ? lY : NINF; });
            vd dD = lane_shift_in(Dp(seg - 1), bD);
            for (int s = 0; s < seg; ++s) {
                const vd uD = Dp(s);
                const vd mx = icell(s, dD, uD);
                dD = uD;
                const vd y = stdx::max(Oin, vsub<Sat>(Yin, vge_a));
                stv(y, cY + (std::size_t)s * W);
                Oin = open_pen<Sat>(mx, vgo_a, vge_a, vgoe_a);
                Yin = y;
            }
            const vd Olast = Oin;
            for (int r = 0; r < W; ++r) {
                vd F = vsub<Sat>(lane_shift_in(ld(cY + (std::size_t)(seg - 1) * W), NINF), vge_a);
                if (r == 0) F = stdx::max(lane_shift_in(Olast, NINF), F);
                bool changed = false;
                for (int s2 = 0; s2 < seg; ++s2) {
                    vd v = ld(cY + (std::size_t)s2 * W);
                    if (!any_gt(F, v)) break;
                    v = stdx::max(v, F);
                    stv(v, cY + (std::size_t)s2 * W);
                    F = vsub<Sat>(v, vge_a);
                    changed = true;
                }
                if (!changed) break;
            }
        }
        }

        if constexpr (Best) {   // the row's best M cell, if it beats the running best
            T rmax = lmax[0];
            for (int qq = 1; qq < W; ++qq) rmax = std::max(rmax, T(lmax[qq]));
            if (rmax > best) {
                int l = 0;
                while (!(lmax[l] == rmax)) ++l;
                best = rmax; best_i = i; best_j = J.c0 + l * seg + (int)ls[l] + 1;
            }
        }

        // ── the saved lines ──
        auto Mat = [&](int s3) NWGRAD_ALWAYS_INLINE_LAMBDA {   // M of segment s3, as the row pass formed it
            const vd d = s3 == 0 ? lane_shift_in(Dp(seg - 1), bD) : Dp(s3 - 1);
            const vd mv = vadd<Sat>(d, ld(sub + (std::size_t)s3 * W));
            if constexpr (Local) return stdx::max(mv, vd(T(0)));
            else return mv;
        };
        for (int t = 0; t < q; ++t) {
            const int s3 = st[(std::size_t)t];
            T* base = cols + ((std::size_t)t * 3 * (h + 1) + ii) * W;
            const std::size_t ss = (std::size_t)(h + 1) * W;
            stv(Mat(s3), base);
            stv(ld(cX + (std::size_t)s3 * W), base + ss);
            stv(ld(cY + (std::size_t)s3 * W), base + 2 * ss);
        }
        if (ii % J.R == 0 && ii < h) {
            T* row = rows + (std::size_t)(ii / J.R - 1) * 3 * (w + 1);
            row[0] = lM; row[w + 1] = lX; row[2 * (w + 1)] = lY;
            for (int s3 = 0; s3 < seg; ++s3) {
                const vd M = Mat(s3);
                for (int l = 0; l < W; ++l) {
                    const int j = l * seg + s3 + 1;
                    if (j > w) continue;
                    const std::size_t k = (std::size_t)s3 * W + l;
                    row[j] = M[l]; row[w + 1 + j] = cX[k]; row[2 * (w + 1) + j] = cY[k];
                }
            }
        }
        std::swap(pA, cA); std::swap(pX, cX); std::swap(pY, cY);
        bD = std::max({lM, lX, lY});
    }
    if (J.last) {   // row r1 (pA after the last swap): D = max(A, Y) = max(M, X, Y)
        J.last->resize((std::size_t)w + 1);
        T* const lr = J.last->data();
        for (int s3 = 0; s3 < seg; ++s3)
            for (int l = 0; l < W; ++l) {
                const int j = l * seg + s3 + 1;
                if (j > w) continue;
                const std::size_t k = (std::size_t)s3 * W + l;
                lr[j] = std::max(pA[k], pY[k]);
            }
    }
    if constexpr (Best) { J.best = best; J.best_i = best_i; J.best_j = best_j; }
    J.W = W; J.q = q;
}

template <class T>
static void grid_fwd_entry(GridFwdJob<T>& J) {
    if constexpr (nw_sat<T>)
        if (J.sat) {
            if (J.local) { if (J.find_best) grid_fwd_impl<T, true, true, true>(J); else grid_fwd_impl<T, true, true>(J); }
            else         grid_fwd_impl<T, false, true>(J);
            return;
        }
    if (J.local) { if (J.find_best) grid_fwd_impl<T, true, false, true>(J); else grid_fwd_impl<T, true, false>(J); }
    else         grid_fwd_impl<T, false, false>(J);
}

// The registered pointer: every scalar type (the integers since 2026-11-05).
template <class T>
constexpr typename TypedKernels<T>::grid_fn grid_fwd_ptr() { return &grid_fwd_entry<T>; }
