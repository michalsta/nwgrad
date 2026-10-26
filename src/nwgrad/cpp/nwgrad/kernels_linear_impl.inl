// ── Leveled LINEAR Viterbi, Full band ──────────────────────────────────────────
//
// Included once per ISA level like kernels_impl.inl, inside that level's namespace,
// compiled with that level's real -march.  The striped layout and the query profile of
// the affine kernels, applied to the one-table linear recurrence
//
//     H[i][j] = max( H[i-1][j-1] + s(a_i, b_j),  H[i-1][j] - ge_b,  H[i][j-1] - ge_a )
//
// (Local: then max(., 0)), in the same order and the same T as the scalar
// viterbi_linear / viterbi_linear_ptr in aligner.hpp, which these must equal bit for bit:
// the walker re-derives each step by exact equality on H (diagonal, then up, then left),
// and the byte form records the FIRST largest of the same three.
//
// Before this file the linear fills were header functions compiled at baseline: SSE2 on
// every x86 host, autovectorized scalar loops, no AVX2 / AVX-512 / NEON instantiation.
// Measured (nighthaven, AVX2, L200, hard, µs a pair): own Local linear fill 67.6 (int16),
// 118 (float32) against the affine striped Pointers kernel's 21.7 / 65.5 — 3× and 1.8×.
//
// The horizontal gap is the affine kernel's VY carry with a zero open.  Writing U[c] for
// the row's vertical-and-diagonal maximum (Local: clamped at 0),
//     H[c] = max(U[c], Y[c]),      Y[c] = H[c-1] - ge_a = max(U[c-1] - ge_a, Y[c-1] - ge_a)
// so Y is exactly VY with open[c] = U[c] - ge_a: the same-lane striped chain + lazy-F in
// floating point, the EXACT prefix max (int_pmax_carry) for an integer T from four lanes
// (kernels_impl.inl: the same gate as every other integer kernel).  Bit-exact with the
// scalar `left - ge_a` because rounding is monotone: max(a, b) - g == max(a - g, b - g).
// Local's clamp commutes with the chain as values (max(U, Y, 0) = max(max(U, 0), Y)), so
// the carry runs on the clamped U.  Direction codes: 0 diagonal, 1 up, 2 left, 3 stop —
// left only where Y STRICTLY exceeds U (the scalar's `t < l`), so the carry runs without
// its code pass; Local stops where the clamped value is <= 0.
//
//   Scores     one T table H, (m+1)*rowsz, striped (table_layout 1)
//   Pointers   one BYTE table DM, two rolling H rows

#ifndef NWGRAD_LEVEL_NS
#  error "kernels_linear_impl.inl must be included inside a level namespace by a level TU"
#endif

// Local: the best cell of one striped row of ONE table — striped_row_best over H alone
// (same contract: topmost row by strict `>`, leftmost column; padding masked by pad[]).
template <class T>
static inline void striped_row_best1(const T* cH, const T* pad, int seg, int i,
                                     T& best, int& bi, int& bj) {
    using vt = stdx::native_simd<T>;
    constexpr int W = (int)vt::size();
    vt lmax(nw_ninf<T>), ls(T(0));
    vt vs(T(0));
    const vt one(T(1));
    for (int s = 0; s < seg; ++s, vs += one) {
        const std::size_t o = (std::size_t)s * W;
        vt h, vp;
        h.copy_from(cH + o, stdx::element_aligned);
        vp.copy_from(pad + o, stdx::element_aligned);
        h = h + vp;
#if defined(__clang__) && defined(__AVX512F__) && !defined(NWGRAD_STD_SIMD_AVX512_MASK_OK)
        if constexpr (sizeof(T) == 8) ls = avx512d_blend(avx512d_gt(h, lmax), ls, vs);
        else                          stdx::where(h > lmax, ls) = vs;
#else
        stdx::where(h > lmax, ls) = vs;
#endif
        lmax = stdx::max(lmax, h);
    }
    T rmax;
    int l;
#if defined(__clang__) && defined(__AVX512F__) && !defined(NWGRAD_STD_SIMD_AVX512_MASK_OK)
    constexpr bool vec_reduce = nw_is_int<T> && sizeof(T) != 8;
#else
    constexpr bool vec_reduce = nw_is_int<T>;
#endif
    if constexpr (vec_reduce) {
        rmax = stdx::hmax(lmax);
        if (!(rmax > best)) return;
        l = stdx::find_first_set(lmax == rmax);
    } else {
        rmax = lmax[0];
        for (int q = 1; q < W; ++q) rmax = std::max(rmax, T(lmax[q]));
        if (!(rmax > best)) return;
        l = 0;
        while (!(lmax[l] == rmax)) ++l;
    }
    const int s = (int)ls[l];
    best = rmax; bi = i; bj = l * seg + s + 1;
}

// Ptr: record direction bytes (TracebackMode::Pointers) on rolling rows; else the H table.
// Tr: a tracked problem (job.tr): each row's extends, border and profile row from the
// track arrays, in the same expressions, so equal sets give the untracked table.
// Dense (Ptr only): the same codes, four segments' worth a byte (dense_codes_row_bytes,
// simd_levels.hpp: 0.25 B/cell), for the memory-bound regime -- each lane accumulates its
// 2-bit codes by Horner over a group of four segments and stores one byte per lane a group.
template <class T, bool Local, bool Ptr, bool Tr, bool Dense = false>
static void striped_linear_full(ViterbiJob<T>& job) {
    static_assert(!Dense || (Ptr && sizeof(T) >= 4), "dense codes: pointer fill, lanes >= 32 bits");
    using vd = stdx::native_simd<T>;
    constexpr int W = (int)vd::size();
    const T K_NINF = nw_ninf<T>;

    const int m = job.m, n = job.n;
    const int nalpha = job.nalpha;
    T ge_a = job.ge_a, ge_b = job.ge_b;
    const TrackJob<T>& tr = job.tr;
    if constexpr (Tr) ge_a = tr.ge_a[0];                 // row 0's gap in A
    const T* blk = job.blk;
    const unsigned char* a = job.a;
    const unsigned char* b = job.b;
    DpBufferT<T>& buf = *job.buf;

    const int seg = (n + W - 1) / W;
    const std::size_t sw = (std::size_t)seg * W;
    const std::size_t rowsz = (std::size_t)(seg + 1) * W;   // slot 0 = column 0; columns from W
    const std::size_t off   = (std::size_t)W;
    auto scol = [seg](int j) -> std::size_t {
        return (std::size_t)((j - 1) % seg) * W + (j - 1) / seg;
    };

    striped_profile_build<T, Tr>(buf.sprof, blk, nalpha, b, n, seg, W, tr);

    // Scratch: the opens (U - ge_a) and the carry Y, one striped row each; the byte
    // form keeps the vertical code (0 / 1) per slot in subbuf until the carry is known.
    // The byte form's two rolling rows ride in the same block (spread_rows, dp_buffer.hpp:
    // no two rows in one L1 set).
    // Rows: opens, carry, Local's padding mask, then (byte form) the vertical codes and the
    // two rolling rows.
    std::size_t rs; T* const rows = spread_rows<T>(buf.lrows, rowsz, Ptr ? 6 : 3, rs);
    T* ov = rows;
    T* yv = rows + rs;
    T* kc = nullptr;
    if constexpr (Ptr) kc = rows + 3 * rs;

    // The H table's row stride, on a narrow L1D (table_spread, dp_buffer.hpp) padded past
    // rowsz (spread_stride) so rows i and i-1 never share an L1 set or a 4 KB offset; it
    // reaches the aligner as job.rstride.  The byte form's D table keeps rowsz (bytes,
    // written once a cell).
    const std::size_t hstride = (Ptr || !table_spread()) ? rowsz : spread_stride<T>(rowsz, 2);
    T* pH; T* cH;
    // the direction table's row length in bytes: one per slot, or the dense groups
    const std::size_t drow = Dense ? dense_codes_row_bytes(seg, W, 1) : rowsz;
    if constexpr (Ptr) {
        const std::size_t dsz = (std::size_t)(m + 1) * drow;
        if (buf.DM.size() < dsz) buf.DM.resize(dsz);
        pH = rows + 4 * rs; cH = rows + 5 * rs;
    } else {
        const std::size_t hsz = (std::size_t)(m + 1) * hstride;
        if (buf.H.size() < hsz) buf.H.resize(hsz);
        pH = buf.H.data(); cH = pH + hstride;
    }

    // ── row 0 ──  H[0][0] = 0; H[0][j] = -(j * ge_a) (Global) or 0 (Local); pads NINF.
    pH[0] = static_cast<T>(0);
    for (int l = 0; l < W; ++l)
        for (int s = 0; s < seg; ++s) {
            const int j = l * seg + s + 1;
            const std::size_t k = off + (std::size_t)s * W + l;
            if (j <= n) pH[k] = Local ? static_cast<T>(0) : -static_cast<T>(j) * ge_a;
            else        pH[k] = K_NINF;
        }
    if constexpr (Ptr) {
        unsigned char* d0 = buf.DM.data();
        d0[0] = 3;
        if constexpr (Dense)   // every 2-bit field the same code: 3 = 0xFF, 2 = 0xAA
            std::memset(d0 + off, Local ? 0xFF : 0xAA, drow - off);
        else
            for (std::size_t k = 0; k < sw; ++k) d0[off + k] = Local ? 3 : 2;   // row 0: left (Global)
    }

    vd vge_a(ge_a), vge_b(ge_b);
    const vd vzero(static_cast<T>(0)), vone(static_cast<T>(1)), vtwo(static_cast<T>(2));
    const vd vthree(static_cast<T>(3));
    const vd vninf(K_NINF);
    T best_local = static_cast<T>(0); int best_i = 0, best_j = 0;
    const T* pad = Local ? striped_pad_into(rows + 2 * rs, n, seg, W) : nullptr;

    for (int i = 1; i <= m; ++i) {
        if constexpr (Tr) {
            ge_a = tr.ge_a[i]; ge_b = tr.ge_b[i];
            vge_a = vd(ge_a); vge_b = vd(ge_b);
        }
        // column 0 of this row, and the previous row's (the diagonal into column 1)
        T nb;
        if constexpr (Tr) nb = Local ? static_cast<T>(0) : tr.bx0[i];
        else              nb = Local ? static_cast<T>(0) : -static_cast<T>(i) * ge_b;
        const T bprev = pH[0];
        cH[0] = nb;
        const T nbOpen = nb - ge_a;                     // the open into column 1
        const T* sub = striped_row_profile<T, Tr>(buf.sprof.data(), blk, nalpha, a, b, n, seg, W, i, tr);
        unsigned char* dM = nullptr;
        if constexpr (Ptr) {
            dM = buf.DM.data() + (std::size_t)i * drow;
            dM[0] = Local ? 3 : 1;                       // column 0: up (Global) / stop
            dM += off;
        }

        if (seg > 0) {
            // ── pass 1: U = max(diag + s, up - ge_b) (Local: and 0), its code, the open ──
            // the diagonal: floating T, the previous iteration's up vector, rotated in a register
            // (each previous-row vector loaded once); at s = 0 the last segment, shifted one lane in
            vd dg(nw_ninf<T>);
            if (seg > 0) {
                vd last; last.copy_from(pH + off + (std::size_t)(seg - 1) * W, stdx::element_aligned);
                dg = lane_shift_in(last, bprev);
            }
            for (int s = 0; s < seg; ++s) {
                vd up; up.copy_from(pH + off + (std::size_t)s * W, stdx::element_aligned);
                vd sb; sb.copy_from(sub + (std::size_t)s * W, stdx::element_aligned);
                // Integer T reloads it: there the rotation cost NEON 2-5 % (int32, spot, every
                // length and thread count) and bought nothing on AVX2 (0.99-1.01, nighthaven).
                if constexpr (nw_is_int<T>)
                    if (s > 0) dg.copy_from(pH + off + (std::size_t)(s - 1) * W, stdx::element_aligned);
                const vd d0 = dg + sb;
                if constexpr (!nw_is_int<T>) dg = up;   // the next segment's diagonal
                const vd u  = up - vge_b;
                vd t = stdx::max(d0, u);
                if constexpr (Ptr) {
                    vd k = vzero;                        // diagonal unless up is STRICTLY larger
#if defined(__clang__) && defined(__AVX512F__) && !defined(NWGRAD_STD_SIMD_AVX512_MASK_OK)
                    if constexpr (sizeof(T) == 8) k = avx512d_blend(avx512d_gt(u, d0), k, vone);
                    else                          stdx::where(u > d0, k) = vone;
#else
                    stdx::where(u > d0, k) = vone;
#endif
                    k.copy_to(kc + (std::size_t)s * W, stdx::element_aligned);
                }
                if constexpr (Local) t = stdx::max(t, vzero);
                t.copy_to(cH + off + (std::size_t)s * W, stdx::element_aligned);
                if constexpr (nw_is_int<T> && W >= 4)   // the integer carry reads the opens; the floating one recomputes them
                    (t - vge_a).copy_to(ov + (std::size_t)s * W, stdx::element_aligned);
            }

            // ── the carry: Y[c] = max(open[c-1], Y[c-1] - ge_a) along each lane ──
            if constexpr (nw_is_int<T> && W >= 4) {
                int_pmax_carry<T, W>(ov, nullptr, nbOpen, static_cast<T>(0),
                                     K_NINF, ge_a, seg, yv, nullptr, nullptr);
            } else {
                // The open into Y at segment q: `t - ge_a`, t still in cH until pass 2 -- the same
                // bits pass 1 stores in `ov` for the integer carry; one stream fewer per segment.
                auto open_at = [&](int q) __attribute__((always_inline)) {
                    vd t; t.copy_from(cH + off + (std::size_t)q * W, stdx::element_aligned);
                    return vd(t - vge_a);
                };
                vd prev = vninf;                         // column 0 holds no Y
                for (int s = 0; s < seg; ++s) {
                    vd O;
                    if (s == 0) {
                        const vd lo = open_at(seg - 1);
                        O = lane_shift_in(lo, nbOpen);
                    } else {
                        O = open_at(s - 1);
                    }
                    const vd v = stdx::max(O, prev - vge_a);
                    v.copy_to(yv + (std::size_t)s * W, stdx::element_aligned);
                    prev = v;
                }
                for (int r = 0; r < W; ++r) {            // lazy-F: extend across the lane boundary
                    vd last; last.copy_from(yv + (std::size_t)(seg - 1) * W, stdx::element_aligned);
                    vd F = lane_shift_in(last, K_NINF) - vge_a;
                    bool changed = false;
                    for (int s = 0; s < seg; ++s) {
                        vd v; v.copy_from(yv + (std::size_t)s * W, stdx::element_aligned);
#if defined(__clang__) && defined(__AVX512F__) && !defined(NWGRAD_STD_SIMD_AVX512_MASK_OK)
                        if constexpr (sizeof(T) == 8) { if (!avx512d_any_gt(F, v)) break; }
                        else { if (!stdx::any_of(F > v)) break; }
#else
                        if (!stdx::any_of(F > v)) break;
#endif
                        v = stdx::max(v, F);
                        v.copy_to(yv + (std::size_t)s * W, stdx::element_aligned);
                        F = v - vge_a;
                        changed = true;
                    }
                    if (!changed) break;
                }
            }

            // ── pass 2: H = max(U, Y); codes: left where Y > U, stop where Local <= 0 ──
            vd acc = vzero;                              // Dense: this group's codes so far
            // Dense: in groups of four segments, a group's byte stored after it -- no branch in
            // the segment loop (one-byte codes and the H form: one group, the whole row).
            constexpr int G = Dense ? 4 : (1 << 30);
            for (int s0 = 0; s0 < seg; s0 += G) {
            const int s1 = std::min(seg - s0, G) + s0;
            for (int s = s0; s < s1; ++s) {
                vd u; u.copy_from(cH + off + (std::size_t)s * W, stdx::element_aligned);
                vd y; y.copy_from(yv + (std::size_t)s * W, stdx::element_aligned);
                const vd h = stdx::max(u, y);
                h.copy_to(cH + off + (std::size_t)s * W, stdx::element_aligned);
                if constexpr (Ptr) {
                    vd k; k.copy_from(kc + (std::size_t)s * W, stdx::element_aligned);
#if defined(__clang__) && defined(__AVX512F__) && !defined(NWGRAD_STD_SIMD_AVX512_MASK_OK)
                    if constexpr (sizeof(T) == 8) {
                        k = avx512d_blend(avx512d_gt(y, u), k, vtwo);
                        if constexpr (Local) k = avx512d_blend(avx512d_ge(vzero, h), k, vthree);
                    } else {
                        stdx::where(y > u, k) = vtwo;
                        if constexpr (Local) {
                            if constexpr (nw_is_int<T>) stdx::where(h == vzero, k) = vthree;   // clamped >= 0
                            else                        stdx::where(h <= vzero, k) = vthree;
                        }
                    }
#else
                    stdx::where(y > u, k) = vtwo;
                    if constexpr (Local) {
                        if constexpr (nw_is_int<T>) stdx::where(h == vzero, k) = vthree;   // clamped >= 0
                        else                        stdx::where(h <= vzero, k) = vthree;
                    }
#endif
                    if constexpr (Dense) {
                        acc = acc + acc; acc = acc + acc; acc = acc + k;   // 4*acc + k: exact in every T
                    } else {
                        store_codes<T, W>(k, dM + (std::size_t)s * W);
                    }
                }
            }
            if constexpr (Dense) {
                for (int s = s1; s < s0 + 4; ++s) { acc = acc + acc; acc = acc + acc; }   // short group: fixed shifts
                store_codes<T, W>(acc, dM + (std::size_t)(s0 >> 2) * W);   // <= 255
                acc = vzero;
            }
            }   // groups
            if constexpr (Local) striped_row_best1<T>(cH + off, pad, seg, i, best_local, best_i, best_j);
        }

        if constexpr (Ptr) std::swap(pH, cH);
        else { pH = cH; cH += hstride; }
    }

    if constexpr (Local) {
        job.score = best_local;
        job.best_i = best_i; job.best_j = best_j;
    } else {
        // pH is row m (the H table's last row, or the rolling row after the last swap)
        const std::size_t fn = (n == 0) ? 0 : off + scol(n);
        job.score = pH[fn];
        job.best_i = m; job.best_j = n;
    }
    job.best_tbl = 0;
    job.table_layout = 1;
    job.seg = seg;
    job.width = W;
    job.rstride = hstride;
}

// Whether this level registers the linear kernels for T.  int64 below eight lanes: no —
// AVX2 has no packed 64-bit max (vpcmpgtq + vpblendvb) and NEON / SSE run two lanes, while
// the scalar fill's 64-bit max is one cmov; measured striped / scalar 1.3-2.6x at every
// length on nighthaven (AVX2) and spot (NEON), DNA 40 x 200 and protein L128 included.
template <class T>
static constexpr bool linear_kernel_pays() {
    return !(nw_is_int<T> && sizeof(T) == 8 && stdx::native_simd<T>::size() < 8);
}

template <class T>
static void striped_linear_entry(ViterbiJob<T>& job) {
    if (job.tr.on()) {
        if (job.align_mode) striped_linear_full<T, true, false, true>(job);
        else                striped_linear_full<T, false, false, true>(job);
        return;
    }
    if (job.align_mode) striped_linear_full<T, true, false, false>(job);
    else                striped_linear_full<T, false, false, false>(job);
}
template <class T>
static void striped_linear_ptr_entry(ViterbiJob<T>& job) {
    if constexpr (sizeof(T) >= 4) {
        if (job.dense) {
            if (job.tr.on()) {
                if (job.align_mode) striped_linear_full<T, true, true, true, true>(job);
                else                striped_linear_full<T, false, true, true, true>(job);
            } else {
                if (job.align_mode) striped_linear_full<T, true, true, false, true>(job);
                else                striped_linear_full<T, false, true, false, true>(job);
            }
            return;
        }
    }
    job.dense = 0;
    if (job.tr.on()) {
        if (job.align_mode) striped_linear_full<T, true, true, true>(job);
        else                striped_linear_full<T, false, true, true>(job);
        return;
    }
    if (job.align_mode) striped_linear_full<T, true, true, false>(job);
    else                striped_linear_full<T, false, true, false>(job);
}

// ── Linear Hirschberg: the sweeps, the base case and the Local endpoint scans ──────
//
// One kernel for the four O(m*n) loops of Aligner::viterbi_linear_hirschberg (see LhbJob):
// the recurrence of striped_linear_full over rows the aligner lists in sweep order, with
// the borders it computed in scalar.  Row r: U = max(H[r-1][c-1] + s, H[r-1][c] - gb[r])
// (Local scan: max(U, 0)), the Y carry from column 0 with a zero open, H = max(U, Y) —
// bit for bit the scalar `v = d; if (v < u) v = u; if (v < l) v = l` (rounding is monotone),
// so the sweep's last row, hence the join's argmax and the path, cannot move with the
// level.  Codes as the base case records them: 0 diagonal, 1 up where u > d strictly,
// 2 left where Y strictly exceeds U.  Argmax as the scalar scans run it: strict `>`, rows
// in order, column 0 first (argmax == 2), then columns left to right.
template <class T>
static void lhb_striped(LhbJob<T>& job) {
    using vd = stdx::native_simd<T>;
    constexpr int W = (int)vd::size();
    const T K_NINF = nw_ninf<T>;
    const int n = job.n, h = job.h, nalpha = job.nalpha;
    const unsigned char* b = job.b;
    DpBufferT<T>& buf = *job.buf;

    const int seg = (n + W - 1) / W;
    const std::size_t sw = (std::size_t)seg * W;
    const std::size_t rowsz = (std::size_t)(seg + 1) * W;
    const std::size_t off = (std::size_t)W;
    auto scol = [seg](int j) -> std::size_t {
        return (std::size_t)((j - 1) % seg) * W + (j - 1) / seg;
    };

    // profile: one row per symbol (untracked, enough rows to pay for it) or per DP row
    const bool by_symbol = job.by_symbol != 0;
    {
        const std::size_t need = by_symbol ? (std::size_t)nalpha * sw : sw;
        if (buf.hprof.size() < need) buf.hprof.resize(need);
        if (by_symbol)
            for (int c = 0; c < nalpha; ++c)
                striped_profile_row(buf.hprof.data() + (std::size_t)c * sw, job.blk + (std::size_t)c * nalpha,
                                    b, n, seg, W);
    }
    // opens, carry, vertical codes, the two rolling rows and the argmax padding mask: one
    // block (spread_rows)
    std::size_t rs; T* const rows = spread_rows<T>(buf.lrows, rowsz, 6, rs);
    T* ov = rows;
    T* yv = rows + rs;
    T* kc = rows + 2 * rs;
    T* pH = rows + 3 * rs; T* cH = rows + 4 * rs;
    unsigned char* D = job.codes;
    const T* pad = job.argmax ? striped_pad_into(rows + 5 * rs, n, seg, W) : nullptr;

    // row 0
    pH[0] = job.row0[0];
    for (int l = 0; l < W; ++l)
        for (int s = 0; s < seg; ++s) {
            const int j = l * seg + s + 1;
            pH[off + (std::size_t)s * W + l] = (j <= n) ? job.row0[j] : K_NINF;
        }
    if (D) {
        D[0] = 3;
        for (std::size_t k = 0; k < sw; ++k) D[off + k] = 2;
    }

    const vd vzero(static_cast<T>(0)), vone(static_cast<T>(1)), vtwo(static_cast<T>(2));
    const vd vninf(K_NINF);
    T best = job.best; int bi = job.bi, bj = job.bj;

    for (int r = 1; r <= h; ++r) {
        const T ga = job.ga[r - 1], gb = job.gb[r - 1];
        const vd vga(ga), vgb(gb);
        const T c0 = job.col0[r];
        const T bprev = pH[0];
        cH[0] = c0;
        const T nbOpen = c0 - ga;
        const T* sub;
        if (by_symbol) {
            sub = buf.hprof.data() + (std::size_t)job.srow[r - 1] * sw;
        } else {
            striped_profile_row(buf.hprof.data(), job.blk + (std::size_t)job.srow[r - 1] * nalpha, b, n, seg, W);
            sub = buf.hprof.data();
        }
        unsigned char* dM = nullptr;
        if (D) { dM = D + (std::size_t)r * rowsz; dM[0] = 1; dM += off; }
        if (job.argmax == 2 && c0 > best) { best = c0; bi = r; bj = 0; }

        if (seg > 0) {
            // the diagonal: floating T, the previous iteration's up vector, rotated in a register
            // (each previous-row vector loaded once); at s = 0 the last segment, shifted one lane in
            vd dg(nw_ninf<T>);
            if (seg > 0) {
                vd last; last.copy_from(pH + off + (std::size_t)(seg - 1) * W, stdx::element_aligned);
                dg = lane_shift_in(last, bprev);
            }
            for (int s = 0; s < seg; ++s) {
                vd up; up.copy_from(pH + off + (std::size_t)s * W, stdx::element_aligned);
                vd sb; sb.copy_from(sub + (std::size_t)s * W, stdx::element_aligned);
                // Integer T reloads it: there the rotation cost NEON 2-5 % (int32, spot, every
                // length and thread count) and bought nothing on AVX2 (0.99-1.01, nighthaven).
                if constexpr (nw_is_int<T>)
                    if (s > 0) dg.copy_from(pH + off + (std::size_t)(s - 1) * W, stdx::element_aligned);
                const vd d0 = dg + sb;
                if constexpr (!nw_is_int<T>) dg = up;   // the next segment's diagonal
                const vd u = up - vgb;
                vd t = stdx::max(d0, u);
                if (D) {
                    vd k = vzero;
#if defined(__clang__) && defined(__AVX512F__) && !defined(NWGRAD_STD_SIMD_AVX512_MASK_OK)
                    if constexpr (sizeof(T) == 8) k = avx512d_blend(avx512d_gt(u, d0), k, vone);
                    else                          stdx::where(u > d0, k) = vone;
#else
                    stdx::where(u > d0, k) = vone;
#endif
                    k.copy_to(kc + (std::size_t)s * W, stdx::element_aligned);
                }
                if (job.clamp) t = stdx::max(t, vzero);
                t.copy_to(cH + off + (std::size_t)s * W, stdx::element_aligned);
                if constexpr (nw_is_int<T> && W >= 4)   // the integer carry reads the opens; the floating one recomputes them
                    (t - vga).copy_to(ov + (std::size_t)s * W, stdx::element_aligned);
            }

            if constexpr (nw_is_int<T> && W >= 4) {
                int_pmax_carry<T, W>(ov, nullptr, nbOpen, static_cast<T>(0),
                                     K_NINF, ga, seg, yv, nullptr, nullptr);
            } else {
                // The open into Y at segment q: `t - ge_a`, t still in cH until pass 2 -- the same
                // bits pass 1 stores in `ov` for the integer carry; one stream fewer per segment.
                auto open_at = [&](int q) __attribute__((always_inline)) {
                    vd t; t.copy_from(cH + off + (std::size_t)q * W, stdx::element_aligned);
                    return vd(t - vga);
                };
                vd prev = vninf;
                for (int s = 0; s < seg; ++s) {
                    vd O;
                    if (s == 0) {
                        const vd lo = open_at(seg - 1);
                        O = lane_shift_in(lo, nbOpen);
                    } else {
                        O = open_at(s - 1);
                    }
                    const vd v = stdx::max(O, prev - vga);
                    v.copy_to(yv + (std::size_t)s * W, stdx::element_aligned);
                    prev = v;
                }
                for (int rr = 0; rr < W; ++rr) {
                    vd last; last.copy_from(yv + (std::size_t)(seg - 1) * W, stdx::element_aligned);
                    vd F = lane_shift_in(last, K_NINF) - vga;
                    bool changed = false;
                    for (int s = 0; s < seg; ++s) {
                        vd v; v.copy_from(yv + (std::size_t)s * W, stdx::element_aligned);
#if defined(__clang__) && defined(__AVX512F__) && !defined(NWGRAD_STD_SIMD_AVX512_MASK_OK)
                        if constexpr (sizeof(T) == 8) { if (!avx512d_any_gt(F, v)) break; }
                        else { if (!stdx::any_of(F > v)) break; }
#else
                        if (!stdx::any_of(F > v)) break;
#endif
                        v = stdx::max(v, F);
                        v.copy_to(yv + (std::size_t)s * W, stdx::element_aligned);
                        F = v - vga;
                        changed = true;
                    }
                    if (!changed) break;
                }
            }

            for (int s = 0; s < seg; ++s) {
                vd u; u.copy_from(cH + off + (std::size_t)s * W, stdx::element_aligned);
                vd y; y.copy_from(yv + (std::size_t)s * W, stdx::element_aligned);
                stdx::max(u, y).copy_to(cH + off + (std::size_t)s * W, stdx::element_aligned);
                if (D) {
                    vd k; k.copy_from(kc + (std::size_t)s * W, stdx::element_aligned);
#if defined(__clang__) && defined(__AVX512F__) && !defined(NWGRAD_STD_SIMD_AVX512_MASK_OK)
                    if constexpr (sizeof(T) == 8) k = avx512d_blend(avx512d_gt(y, u), k, vtwo);
                    else                          stdx::where(y > u, k) = vtwo;
#else
                    stdx::where(y > u, k) = vtwo;
#endif
                    store_codes<T, W>(k, dM + (std::size_t)s * W);
                }
            }
            if (job.argmax) {
                int ri = 0, rj = 0;
                const T before = best;
                striped_row_best1<T>(cH + off, pad, seg, r, best, ri, rj);
                if (best > before) { bi = ri; bj = rj; }
            }
        }
        std::swap(pH, cH);
    }

    if (job.out) {
        job.out[0] = pH[0];
        for (int c = 1; c <= n; ++c) job.out[c] = pH[off + scol(c)];
    }
    job.best = best; job.bi = bi; job.bj = bj;
    job.seg = seg; job.width = W;
}
