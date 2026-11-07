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

// ── The base block: the striped Pointers fill over a sub-rectangle ────────────────
//
// Aligner::grid_base re-fills each base rectangle (up to kGridBase cells) and walks its
// codes.  Its scalar fill has a loop-carried Y and three code arrays: ~27 instructions a
// cell, as much as the whole forward pass at int16 L1000 (the base was 45 % of an int16
// Grid run there, perf).  This is striped_affine_full_ptr (kernels_pointers_impl.inl)
// restricted to the rectangle, as hb_base_striped is for Hirschberg, with Grid's borders:
//   * row r0 comes from the top line (three states per column), not row 0's closed form;
//   * column c0 of row i comes from the left line: its M / X / Y are the diagonal operand
//     of row i+1's first cell (shifted into lane 0, as the full kernel's bM / bX / bY),
//     its open (max(M, X) - go_a) - ge_a enters Y's carry with its code (M wins ties),
//     and its Y enters as Y(i, c0) - ge_a, the extension the scalar pick weighs.  (The
//     full kernel's column 0 has Y = -inf, so it passes the sentinel there.)
// Local's clamp (code 3) is the full kernel's.  Every other expression, comparison and
// association is the full kernel's, so each re-filled cell has the full fill's value and
// code -- the same argument as hb_base_striped and the Pointers kernel themselves (the
// scalar block in aligner.hpp stays the reference, and the tests compare both to
// Pointers).  Glocal's end column and the end cell's argmax read `last`.
template <class T, bool Local, bool Sat>
static void grid_base_striped(GridBaseJob<T>& J) {
    using vd = stdx::native_simd<T>;
    constexpr int W = (int)vd::size();
    constexpr bool kOpenRows = nw_is_int<T> && W >= 4;   // see striped_affine_full_ptr
    const T NINF = nw_ninf_s<T, Sat>;
    const int h = J.r1 - J.r0, w = J.c1 - J.c0, nalpha = J.nalpha;
    const int seg = (w + W - 1) / W;
    J.W = W; J.seg = seg;
    if (w <= 0 || h <= 0 || seg < J.min_seg) { J.done = 0; return; }
    J.done = 1;
    const std::size_t sw = (std::size_t)seg * W;
    const T go_a = J.go_a, ge_a = J.ge_a, go_b = J.go_b, ge_b = J.ge_b;
    DpBufferT<T>& buf = *J.buf;
    auto top  = [&](int s, int jj) { return J.top[(std::size_t)s * J.top_ss + (std::size_t)jj * J.top_es]; };
    auto left = [&](int s, int ii) { return J.left[(std::size_t)s * J.left_ss + (std::size_t)ii * J.left_es]; };

    if (buf.gD.size() < (std::size_t)h * sw) buf.gD.resize((std::size_t)h * sw);
    unsigned char* const D = buf.gD.data();
    // six rolling rows, ov | ok | Q (integer T), the y-code and M+X-code scratch
    std::size_t rs; T* const rows = spread_rows<T>(buf.prows, sw, 11, rs);
    T* pM = rows + 3 * rs; T* pX = rows + 4 * rs; T* pY = rows + 5 * rs;   // row i-1
    T* cM = rows;          T* cX = rows + rs;     T* cY = rows + 2 * rs;   // row i
    T* ov = rows + 6 * rs;
    T* ok = rows + 7 * rs;
    T* qb = rows + 8 * rs;
    T* yc = rows + 9 * rs;
    T* mx = rows + 10 * rs;

    // the profile rows of the letters this block meets, built on first use from the
    // block's striped column symbols (one gather a slot; int16 by byte shuffles)
    if (buf.gidx.size() < sw) buf.gidx.resize(sw);
    striped_profile_index(buf.gidx.data(), J.b + J.c0, w, seg, W);
    if (buf.gprof.size() < (std::size_t)nalpha * sw) buf.gprof.resize((std::size_t)nalpha * sw);
    buf.gbuilt.assign((std::size_t)nalpha, 0);

    // row r0 from the top line, striped; padding -inf
    for (int l = 0; l < W; ++l)
        for (int s = 0; s < seg; ++s) {
            const int c = l * seg + s + 1;
            const std::size_t k = (std::size_t)s * W + l;
            if (c <= w) { pM[k] = top(0, c); pX[k] = top(1, c); pY[k] = top(2, c); }
            else        { pM[k] = NINF; pX[k] = NINF; pY[k] = NINF; }
        }
    T bM = top(0, 0), bX = top(1, 0), bY = top(2, 0);   // column c0 of the row above

    const vd vgo_a(go_a), vge_a(ge_a), vgo_b(go_b), vge_b(ge_b);
    const vd vzero(T(0)), vone(T(1)), vtwo(T(2)), vthree(T(3));
    const vd vfour(T(4)), veight(T(8)), vsixteen(T(16)), vthirtytwo(T(32));

    for (int ii = 1; ii <= h; ++ii) {
        const unsigned char ai = J.a[J.r0 + ii - 1];
        T* const sub = buf.gprof.data() + (std::size_t)ai * sw;
        if (!buf.gbuilt[ai]) {
            buf.gbuilt[ai] = 1;
            striped_profile_row_idx(sub, J.blk + (std::size_t)ai * nalpha, buf.gidx.data(), w, seg, W, nalpha);
        }
        unsigned char* const dP = D + (std::size_t)(ii - 1) * sw;
        const T lM = left(0, ii), lX = left(1, ii), lY = left(2, ii);
        const T nbOpen = nw_sub(nw_sub(std::max(lM, lX), go_a), ge_a);
        const T nbOpenK = (nw_sub(nw_sub(lM, go_a), ge_a) >= nw_sub(nw_sub(lX, go_a), ge_a)) ? T(0) : T(16);

        // the diagonal predecessors, rotated in registers (striped_affine_full_ptr)
        vd dgM, dgX, dgY;
        {
            vd lMv, lXv, lYv;
            lMv.copy_from(pM + (std::size_t)(seg - 1) * W, stdx::element_aligned);
            lXv.copy_from(pX + (std::size_t)(seg - 1) * W, stdx::element_aligned);
            lYv.copy_from(pY + (std::size_t)(seg - 1) * W, stdx::element_aligned);
            dgM = lane_shift_in(lMv, bM);
            dgX = lane_shift_in(lXv, bX);
            dgY = lane_shift_in(lYv, bY);
        }
        for (int s = 0; s < seg; ++s) {
            vd sb; sb.copy_from(sub + (std::size_t)s * W, stdx::element_aligned);
            const vd mx3 = stdx::max(stdx::max(dgM, dgX), dgY);
            vd vmv = vadd<Sat>(mx3, sb);
            vd km = vtwo;
#if defined(__clang__) && defined(__AVX512F__) && !defined(NWGRAD_STD_SIMD_AVX512_MASK_OK)
            if constexpr (sizeof(T) == 8) {
                km = avx512d_blend(avx512d_ge(dgX, dgY), km, vone);
                km = avx512d_blend(avx512d_eq(mx3, dgM), km, vzero);
            } else {
                km = pick_code_mx(mx3, dgM, dgX, dgY, vzero, vone, vtwo);
            }
#else
            km = pick_code_mx(mx3, dgM, dgX, dgY, vzero, vone, vtwo);
#endif
            if constexpr (Local) {
                vmv = stdx::max(vmv, vzero);
#if defined(__clang__) && defined(__AVX512F__) && !defined(NWGRAD_STD_SIMD_AVX512_MASK_OK)
                if constexpr (sizeof(T) == 8) {
                    km = avx512d_blend(avx512d_ge(vzero, vmv), km, vthree);
                } else if constexpr (nw_is_int<T>) {
                    stdx::where(vmv == vzero, km) = vthree;
                } else {
                    stdx::where(vmv <= vzero, km) = vthree;
                }
#else
                if constexpr (nw_is_int<T>) stdx::where(vmv == vzero, km) = vthree;   // clamped >= 0
                else                        stdx::where(vmv <= vzero, km) = vthree;   // the walk stops here
#endif
            }
            vmv.copy_to(cM + (std::size_t)s * W, stdx::element_aligned);

            vd uM, uX, uY;
            uM.copy_from(pM + (std::size_t)s * W, stdx::element_aligned);
            uX.copy_from(pX + (std::size_t)s * W, stdx::element_aligned);
            uY.copy_from(pY + (std::size_t)s * W, stdx::element_aligned);
            dgM = uM; dgX = uX; dgY = uY;   // the next segment's diagonal
            const vd ax = vsub<Sat>(vsub<Sat>(uM, vgo_b), vge_b), bx = vsub<Sat>(uX, vge_b), cx = vsub<Sat>(vsub<Sat>(uY, vgo_b), vge_b);
            vd vxv = stdx::max(stdx::max(ax, bx), cx);
            vd kx = veight;                     // X's code, shifted: 0 / 4 / 8
#if defined(__clang__) && defined(__AVX512F__) && !defined(NWGRAD_STD_SIMD_AVX512_MASK_OK)
            if constexpr (sizeof(T) == 8) {
                kx = avx512d_blend(avx512d_ge(bx, cx), kx, vfour);
                kx = avx512d_blend(avx512d_eq(vxv, ax), kx, vzero);
            } else {
                kx = pick_code_mx(vxv, ax, bx, cx, vzero, vfour, veight);
            }
#else
            kx = pick_code_mx(vxv, ax, bx, cx, vzero, vfour, veight);
#endif
            vxv.copy_to(cX + (std::size_t)s * W, stdx::element_aligned);
            (km + kx).copy_to(mx + (std::size_t)s * W, stdx::element_aligned);
            if constexpr (kOpenRows) {   // the open on the ROUNDED candidates (see the full kernel)
                const vd om = vsub<Sat>(vsub<Sat>(vmv, vgo_a), vge_a), ox = vsub<Sat>(vsub<Sat>(vxv, vgo_a), vge_a);
                stdx::max(om, ox).copy_to(ov + (std::size_t)s * W, stdx::element_aligned);
                vd okv = vsixteen;
#if defined(__clang__) && defined(__AVX512F__) && !defined(NWGRAD_STD_SIMD_AVX512_MASK_OK)
                if constexpr (sizeof(T) == 8) okv = avx512d_blend(avx512d_ge(om, ox), okv, vzero);
                else                          okv = sel_ge(om, ox, vsixteen, vzero);
#else
                okv = sel_ge(om, ox, vsixteen, vzero);
#endif
                okv.copy_to(ok + (std::size_t)s * W, stdx::element_aligned);
            }
        }

        // ── Y's carry, codes tracked: column c0's Y enters as the extension seed ──
        if constexpr (kOpenRows) {
            int_pmax_carry<T, W, false, 32, Sat>(ov, ok, nbOpen, nbOpenK, lY, ge_a, seg, cY, yc, qb);
        } else {
            auto open_from = [&](int q, vd& O, vd& OK) __attribute__((always_inline)) {
                vd vm, vx;
                vm.copy_from(cM + (std::size_t)q * W, stdx::element_aligned);
                vx.copy_from(cX + (std::size_t)q * W, stdx::element_aligned);
                const vd om = vsub<Sat>(vsub<Sat>(vm, vgo_a), vge_a), ox = vsub<Sat>(vsub<Sat>(vx, vgo_a), vge_a);
                O = stdx::max(om, ox);
                OK = vsixteen;
#if defined(__clang__) && defined(__AVX512F__) && !defined(NWGRAD_STD_SIMD_AVX512_MASK_OK)
                if constexpr (sizeof(T) == 8) OK = avx512d_blend(avx512d_ge(om, ox), OK, vzero);
                else                          OK = sel_ge(om, ox, vsixteen, vzero);
#else
                OK = sel_ge(om, ox, vsixteen, vzero);
#endif
            };
            vd prev([&](int q) { return q == 0 ? lY : NINF; });
            vd OKs[2];
            prev = carry_pass<Sat>(seg, prev, vge_a,
                [&](int s) NWGRAD_ALWAYS_INLINE_LAMBDA {
                    const int q = s == 0 ? seg - 1 : s - 1;
                    vd O, OK;
                    open_from(q, O, OK);
                    OKs[s & 1] = s == 0 ? lane_shift_in(OK, nbOpenK) : OK;
                    return s == 0 ? lane_shift_in(O, nbOpen) : O;
                },
                [&](int s, const vd& O, const vd& ext, const vd& v) NWGRAD_ALWAYS_INLINE_LAMBDA {
                    vd ky = vthirtytwo;                       // extension
#if defined(__clang__) && defined(__AVX512F__) && !defined(NWGRAD_STD_SIMD_AVX512_MASK_OK)
                    if constexpr (sizeof(T) == 8) ky = avx512d_blend(avx512d_ge(O, ext), ky, OKs[s & 1]);
                    else                          ky = sel_ge(O, ext, vthirtytwo, OKs[s & 1]);
#else
                    ky = sel_ge(O, ext, vthirtytwo, OKs[s & 1]);   // open: M or X, ties go to open
#endif
                    v.copy_to(cY + (std::size_t)s * W, stdx::element_aligned);
                    ky.copy_to(yc + (std::size_t)s * W, stdx::element_aligned);
                });
            // lazy-F across the lanes (lane 0's column-c0 seed is in the carry already: the
            // sentinel here, as the full kernel's)
            for (int r = 0; r < W; ++r) {
                vd last; last.copy_from(cY + (std::size_t)(seg - 1) * W, stdx::element_aligned);
                vd F = vsub<Sat>(lane_shift_in(last, NINF), vge_a);
                bool changed = false;
                for (int s = 0; s < seg; ++s) {
                    vd v; v.copy_from(cY + (std::size_t)s * W, stdx::element_aligned);
#if defined(__clang__) && defined(__AVX512F__) && !defined(NWGRAD_STD_SIMD_AVX512_MASK_OK)
                    if constexpr (sizeof(T) == 8) { if (!avx512d_any_gt(F, v)) break; }
                    else { if (!stdx::any_of(F > v)) break; }
#else
                    if (!stdx::any_of(F > v)) break;
#endif
                    vd k; k.copy_from(yc + (std::size_t)s * W, stdx::element_aligned);
#if defined(__clang__) && defined(__AVX512F__) && !defined(NWGRAD_STD_SIMD_AVX512_MASK_OK)
                    if constexpr (sizeof(T) == 8) k = avx512d_blend(avx512d_gt(F, v), k, vthirtytwo);
                    else                          stdx::where(F > v, k) = vthirtytwo;
#else
                    stdx::where(F > v, k) = vthirtytwo;   // a raised cell is an extension
#endif
                    k.copy_to(yc + (std::size_t)s * W, stdx::element_aligned);
                    v = stdx::max(v, F);
                    v.copy_to(cY + (std::size_t)s * W, stdx::element_aligned);
                    F = lazy_f_next<Sat>(F, v, vge_a);
                    changed = true;
                }
                if (!changed) break;
            }
        }
        for (int s = 0; s < seg; ++s) {   // M | X << 2 | Y << 4
            vd k, kmx;
            k.copy_from(yc + (std::size_t)s * W, stdx::element_aligned);
            kmx.copy_from(mx + (std::size_t)s * W, stdx::element_aligned);
            store_codes<T, W>(kmx + k, dP + (std::size_t)s * W);
        }
        bM = lM; bX = lX; bY = lY;
        std::swap(pM, cM); std::swap(pX, cX); std::swap(pY, cY);
    }
    if (J.last) {   // row r1 (p* after the last swap), dense, column c0 from the left line
        J.last->resize(3 * ((std::size_t)w + 1));
        T* const lr = J.last->data();
        const std::size_t w1 = (std::size_t)w + 1;
        lr[0] = left(0, h); lr[w1] = left(1, h); lr[2 * w1] = left(2, h);
        for (int c = 1; c <= w; ++c) {
            const std::size_t k = (std::size_t)((c - 1) % seg) * W + (c - 1) / seg;
            lr[c] = pM[k]; lr[w1 + c] = pX[k]; lr[2 * w1 + c] = pY[k];
        }
    }
}

template <class T>
static void grid_base_entry(GridBaseJob<T>& J) {
    if constexpr (nw_sat<T>)
        if (J.sat) {
            if (J.local) grid_base_striped<T, true, true>(J); else grid_base_striped<T, false, true>(J);
            return;
        }
    if (J.local) grid_base_striped<T, true, false>(J); else grid_base_striped<T, false, false>(J);
}
