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
// reproduce the Pointers fill's codes.  Floating T only: lazy-F's F - ge would let an
// integer sentinel wrap.

#ifndef NWGRAD_LEVEL_NS
#  error "grid_kernel_impl.inl must be included inside a level namespace by a level TU"
#endif

template <class T>
static void grid_fwd_impl(GridFwdJob<T>& J) {
    using vd = stdx::native_simd<T>;
    const int W = (int)vd::size();
    const T NINF = nw_ninf<T>;
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

    for (int ii = 1; ii <= h; ++ii) {
        const int i = J.r0 + ii;
        const T lM = left(0, ii), lX = left(1, ii), lY = left(2, ii);
        const T nbOpen = (std::max(lM, lX) - go_a) - ge_a;
        const unsigned char ai = J.a[i - 1];
        T* const sub = buf.gprof.data() + (std::size_t)ai * sw;
        if (!buf.gbuilt[ai]) {   // the profile row of this letter, for this rectangle's columns
            buf.gbuilt[ai] = 1;
            const T* row = J.blk + (std::size_t)ai * nalpha;
            for (int l = 0; l < W; ++l)
                for (int s = 0; s < seg; ++s) {
                    const int j = l * seg + s + 1;
                    sub[(std::size_t)s * W + l] = (j <= w) ? row[bb[j - 1]] : NINF;
                }
        }

        // ── the row pass: Y's carry two segments per dependent step (see
        // score_affine_striped_pos_fp) ──
        vd Oin([&](int qq) { return qq == 0 ? nbOpen : NINF; });   // Y's open from the column before
        vd Yin([&](int qq) { return qq == 0 ? lY : NINF; });       // Y of the column before
        vd dD = lane_shift_in(Dp(seg - 1), bD);
        auto cell = [&](int s, vd& mx) NWGRAD_ALWAYS_INLINE_LAMBDA {
            const vd uD = Dp(s);
            const vd vmv = dD + ld(sub + (std::size_t)s * W);
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

        // ── the saved lines ──
        auto Mat = [&](int s3) NWGRAD_ALWAYS_INLINE_LAMBDA {   // M of segment s3, as the row pass formed it
            const vd d = s3 == 0 ? lane_shift_in(Dp(seg - 1), bD) : Dp(s3 - 1);
            return d + ld(sub + (std::size_t)s3 * W);
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
    J.W = W; J.q = q;
}

template <class T>
static void grid_fwd_entry(GridFwdJob<T>& J) { grid_fwd_impl<T>(J); }

// The registered pointer: floating T only (integer T runs as Pointers).
template <class T>
constexpr typename TypedKernels<T>::grid_fn grid_fwd_ptr() {
    if constexpr (std::is_floating_point_v<T>) return &grid_fwd_entry<T>;
    else                                       return nullptr;
}
