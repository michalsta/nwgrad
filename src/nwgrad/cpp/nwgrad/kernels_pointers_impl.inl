// ── Leveled affine Viterbi, TracebackMode::Pointers ──────────────────────────
//
// Included once per ISA level exactly like kernels_impl.inl, inside that level's
// namespace, compiled with that level's real -march.  Same striped layout, same
// recurrence, same lazy-F.  The difference is what it retains:
//
//   Scores     three T score tables, (m+1)*rowsz each   12 B/cell (float32)
//   Pointers   three BYTE predecessor tables + two rolling score rows  3 B/cell
//
// The traceback then reads a recorded predecessor instead of re-deriving it from
// the tables.  Bit-exactness is preserved because the codes are produced by the
// *same* `>=` M>X>Y comparison chain the traceback would have run — see the
// scalar reference viterbi_affine_ptr() in aligner.hpp, which this must match.
//
// The subtle part is Y.  Its value is not final when first written: the lazy-F
// fixpoint can raise a cell after the base sweep.  A raised cell is by definition
// a gap EXTENSION, so its code must be rewritten to 2 at the moment the value is
// revised — recording the direction during the base sweep alone would be wrong for
// exactly the cells lazy-F exists to fix.

#ifndef NWGRAD_LEVEL_NS
#  error "kernels_pointers_impl.inl must be included inside a level namespace by a level TU"
#endif

// Narrow a T-typed lane-code vector (values 0..3) to one byte per lane.
// Deliberately a single static_simd_cast, not a per-lane loop: writing direction
// bytes scalar-per-lane measured ~2x slower than the vectorized narrow.
template <class T, int W>
static inline void store_codes(const stdx::native_simd<T>& code, unsigned char* dst) {
    using bvec = stdx::fixed_size_simd<unsigned char, W>;
    stdx::static_simd_cast<bvec>(code).copy_to(dst, stdx::element_aligned);
}

template <class T, bool Local>
static void striped_affine_full_ptr(ViterbiJob<T>& job) {
    using vd = stdx::native_simd<T>;
    constexpr int W = (int)stdx::native_simd<T>::size();
    const T K_NINF = -std::numeric_limits<T>::infinity();

    const int m = job.m, n = job.n;
    const int nalpha = job.nalpha;
    const T go_a = job.go_a, ge_a = job.ge_a, go_b = job.go_b, ge_b = job.ge_b;
    const T* blk = job.blk;
    const unsigned char* a = job.a;
    const unsigned char* b = job.b;
    DpBufferT<T>& buf = *job.buf;

    const int seg = (n + W - 1) / W;
    const std::size_t sw = (std::size_t)seg * W;
    const std::size_t rowsz = (std::size_t)(seg + 1) * W;
    const std::size_t off   = (std::size_t)W;

    if (buf.sopenv.size() < 2 * sw) buf.sopenv.resize(2 * sw);   // ov | openk
    if (buf.sprof.size()  < (std::size_t)nalpha * sw) buf.sprof.resize((std::size_t)nalpha * sw);
    const std::size_t dsz = (std::size_t)(m + 1) * rowsz;
    if (buf.DM.size() < dsz) { buf.DM.resize(dsz); buf.DX.resize(dsz); buf.DY.resize(dsz); }
    // rolling score rows: 2 x rowsz instead of (m+1) x rowsz -- the whole point
    if (buf.rM.size() < rowsz) {
        buf.rM.resize(rowsz); buf.rX.resize(rowsz); buf.rY.resize(rowsz);
        buf.qM.resize(rowsz); buf.qX.resize(rowsz); buf.qY.resize(rowsz);
    }
    if (buf.subbuf.size() < sw) buf.subbuf.resize(sw);           // y-code scratch

    auto scol = [seg](int j) -> std::size_t {
        return (std::size_t)((j - 1) % seg) * W + (j - 1) / seg;
    };

    for (int c = 0; c < nalpha; ++c) {
        const T* row = blk + (std::size_t)c * nalpha;
        T* dst = buf.sprof.data() + (std::size_t)c * sw;
        for (int l = 0; l < W; ++l)
            for (int s = 0; s < seg; ++s) {
                const int j = l * seg + s + 1;
                dst[(std::size_t)s * W + l] = (j <= n) ? row[b[j - 1]] : K_NINF;
            }
    }

    T* ov = buf.sopenv.data();
    T* ok = buf.sopenv.data() + sw;
    T* yc = buf.subbuf.data();

    T* pM = buf.qM.data(); T* pX = buf.qX.data(); T* pY = buf.qY.data();
    T* cM = buf.rM.data(); T* cX = buf.rX.data(); T* cY = buf.rY.data();

    // ── row 0: scores into the rolling "previous" row, codes into the tables ──
    {
        pM[0] = 0; pX[0] = K_NINF; pY[0] = K_NINF;
        for (std::size_t k = 0; k < sw; ++k) {
            pM[off + k] = K_NINF; pX[off + k] = K_NINF; pY[off + k] = K_NINF;
        }
        unsigned char* d0M = buf.DM.data(); unsigned char* d0X = buf.DX.data();
        unsigned char* d0Y = buf.DY.data();
        d0M[0] = 3; d0X[0] = 0; d0Y[0] = 0;
        for (int l = 0; l < W; ++l)
            for (int s = 0; s < seg; ++s) {
                const int j = l * seg + s + 1;
                const std::size_t k = (std::size_t)s * W + l;
                if (j <= n) {
                    if constexpr (Local) pM[off + k] = 0;
                    else                 pY[off + k] = -(go_a + j * ge_a);
                    d0M[off + k] = 3;                  // row 0 is a start
                    d0X[off + k] = 0;
                    d0Y[off + k] = (j <= 1) ? 0 : 2;   // (0,1) opens from M; rest extend
                } else {
                    d0M[off + k] = 3; d0X[off + k] = 0; d0Y[off + k] = 2;
                }
            }
    }

    const vd vgo_a(go_a), vge_a(ge_a), vgo_b(go_b), vge_b(ge_b);
    const vd vzero(static_cast<T>(0)), vone(static_cast<T>(1)), vtwo(static_cast<T>(2));
    const vd vthree(static_cast<T>(3));
    T bM = 0, bX = K_NINF, bY = K_NINF;
    T best_local = 0; int best_i = 0, best_j = 0, best_tbl = 0;

    for (int i = 1; i <= m; ++i) {
        const T nbM = Local ? T(0) : K_NINF;
        const T nbX = Local ? K_NINF : -(go_b + i * ge_b);
        const T nbOpen = (std::max(nbM, nbX) - go_a) - ge_a;
        const T nbOpenK = (nbM >= nbX) ? T(0) : T(1);
        const T* sub = buf.sprof.data() + (std::size_t)a[i - 1] * sw;

        unsigned char* dM = buf.DM.data() + (std::size_t)i * rowsz + off;
        unsigned char* dX = buf.DX.data() + (std::size_t)i * rowsz + off;
        unsigned char* dY = buf.DY.data() + (std::size_t)i * rowsz + off;
        cM[0] = nbM; cX[0] = nbX; cY[0] = K_NINF;
        buf.DM.data()[(std::size_t)i * rowsz] = 3;
        buf.DX.data()[(std::size_t)i * rowsz] = (i <= 1) ? 0 : 1;
        buf.DY.data()[(std::size_t)i * rowsz] = 0;

        if (seg > 0) {
            for (int s = 0; s < seg; ++s) {
                vd dgM, dgX, dgY;
                if (s == 0) {
                    vd lM, lX, lY;
                    lM.copy_from(pM + off + (std::size_t)(seg - 1) * W, stdx::element_aligned);
                    lX.copy_from(pX + off + (std::size_t)(seg - 1) * W, stdx::element_aligned);
                    lY.copy_from(pY + off + (std::size_t)(seg - 1) * W, stdx::element_aligned);
                    dgM = vd([&](int q) { return q == 0 ? bM : lM[q - 1]; });
                    dgX = vd([&](int q) { return q == 0 ? bX : lX[q - 1]; });
                    dgY = vd([&](int q) { return q == 0 ? bY : lY[q - 1]; });
                } else {
                    dgM.copy_from(pM + off + (std::size_t)(s - 1) * W, stdx::element_aligned);
                    dgX.copy_from(pX + off + (std::size_t)(s - 1) * W, stdx::element_aligned);
                    dgY.copy_from(pY + off + (std::size_t)(s - 1) * W, stdx::element_aligned);
                }
                vd sb; sb.copy_from(sub + (std::size_t)s * W, stdx::element_aligned);
                vd vmv = stdx::max(stdx::max(dgM, dgX), dgY) + sb;

                // M code: same >= chain as the scalar pick(), M last so it wins ties
                vd km = vtwo;
#if defined(__clang__) && defined(__AVX512F__) && !defined(NWGRAD_STD_SIMD_AVX512_MASK_OK)
                if constexpr (std::is_same_v<T, double>) {
                    km = avx512d_blend(avx512d_ge(dgX, dgY), km, vone);
                    km = avx512d_blend(avx512d_ge(dgM, dgX) & avx512d_ge(dgM, dgY), km, vzero);
                } else {
                    stdx::where(dgX >= dgY, km) = vone;
                    stdx::where((dgM >= dgX) && (dgM >= dgY), km) = vzero;
                }
#else
                stdx::where(dgX >= dgY, km) = vone;
                stdx::where((dgM >= dgX) && (dgM >= dgY), km) = vzero;
#endif
                if constexpr (Local) {
                    vmv = stdx::max(vmv, vzero);
#if defined(__clang__) && defined(__AVX512F__) && !defined(NWGRAD_STD_SIMD_AVX512_MASK_OK)
                    if constexpr (std::is_same_v<T, double>) {
                        // vmv <= vzero  <=>  vzero >= vmv
                        km = avx512d_blend(avx512d_ge(vzero, vmv), km, vthree);
                    } else {
                        stdx::where(vmv <= vzero, km) = vthree;   // traceback stops here
                    }
#else
                    stdx::where(vmv <= vzero, km) = vthree;   // traceback stops here
#endif
                }
                vmv.copy_to(cM + off + (std::size_t)s * W, stdx::element_aligned);
                store_codes<T, W>(km, dM + (std::size_t)s * W);

                vd uM, uX, uY;
                uM.copy_from(pM + off + (std::size_t)s * W, stdx::element_aligned);
                uX.copy_from(pX + off + (std::size_t)s * W, stdx::element_aligned);
                uY.copy_from(pY + off + (std::size_t)s * W, stdx::element_aligned);
                const vd ax = (uM - vgo_b) - vge_b, bx = uX - vge_b, cx = (uY - vgo_b) - vge_b;
                vd vxv = stdx::max(stdx::max(ax, bx), cx);
                vd kx = vtwo;
#if defined(__clang__) && defined(__AVX512F__) && !defined(NWGRAD_STD_SIMD_AVX512_MASK_OK)
                if constexpr (std::is_same_v<T, double>) {
                    kx = avx512d_blend(avx512d_ge(bx, cx), kx, vone);
                    kx = avx512d_blend(avx512d_ge(ax, bx) & avx512d_ge(ax, cx), kx, vzero);
                } else {
                    stdx::where(bx >= cx, kx) = vone;
                    stdx::where((ax >= bx) && (ax >= cx), kx) = vzero;
                }
#else
                stdx::where(bx >= cx, kx) = vone;
                stdx::where((ax >= bx) && (ax >= cx), kx) = vzero;
#endif
                vxv.copy_to(cX + off + (std::size_t)s * W, stdx::element_aligned);
                store_codes<T, W>(kx, dX + (std::size_t)s * W);

                ((stdx::max(vmv, vxv) - vgo_a) - vge_a)
                    .copy_to(ov + (std::size_t)s * W, stdx::element_aligned);
                vd okv = vone;
#if defined(__clang__) && defined(__AVX512F__) && !defined(NWGRAD_STD_SIMD_AVX512_MASK_OK)
                if constexpr (std::is_same_v<T, double>) {
                    okv = avx512d_blend(avx512d_ge(vmv, vxv), okv, vzero);
                } else {
                    stdx::where(vmv >= vxv, okv) = vzero;
                }
#else
                stdx::where(vmv >= vxv, okv) = vzero;
#endif
                okv.copy_to(ok + (std::size_t)s * W, stdx::element_aligned);
            }

            // ── the carry: VY, same-lane striped chain + lazy-F, codes tracked ──
            vd prev([&](int q) { return q == 0 ? bY : K_NINF; });
            for (int s = 0; s < seg; ++s) {
                vd O, OK;
                if (s == 0) {
                    vd lo, lk;
                    lo.copy_from(ov + (std::size_t)(seg - 1) * W, stdx::element_aligned);
                    lk.copy_from(ok + (std::size_t)(seg - 1) * W, stdx::element_aligned);
                    O  = vd([&](int q) { return q == 0 ? nbOpen  : lo[q - 1]; });
                    OK = vd([&](int q) { return q == 0 ? nbOpenK : lk[q - 1]; });
                } else {
                    O.copy_from(ov + (std::size_t)(s - 1) * W, stdx::element_aligned);
                    OK.copy_from(ok + (std::size_t)(s - 1) * W, stdx::element_aligned);
                }
                const vd ext = prev - vge_a;
                vd v = stdx::max(O, ext);
                vd ky = vtwo;                       // extension
#if defined(__clang__) && defined(__AVX512F__) && !defined(NWGRAD_STD_SIMD_AVX512_MASK_OK)
                if constexpr (std::is_same_v<T, double>) {
                    ky = avx512d_blend(avx512d_ge(O, ext), ky, OK);   // open: M or X, ties go to open
                } else {
                    stdx::where(O >= ext, ky) = OK;     // open: M or X, ties go to open
                }
#else
                stdx::where(O >= ext, ky) = OK;     // open: M or X, ties go to open
#endif
                v.copy_to(cY + off + (std::size_t)s * W, stdx::element_aligned);
                ky.copy_to(yc + (std::size_t)s * W, stdx::element_aligned);
                prev = v;
            }
            for (int r = 0; r < W; ++r) {
                vd last; last.copy_from(cY + off + (std::size_t)(seg - 1) * W, stdx::element_aligned);
                vd F([&](int q) { return q == 0 ? bY : last[q - 1]; });
                F = F - vge_a;
                bool changed = false;
                for (int s = 0; s < seg; ++s) {
                    vd v; v.copy_from(cY + off + (std::size_t)s * W, stdx::element_aligned);
#if defined(__clang__) && defined(__AVX512F__) && !defined(NWGRAD_STD_SIMD_AVX512_MASK_OK)
                    if constexpr (std::is_same_v<T, double>) {
                        alignas(64) double fa[8], va[8];
                        F.copy_to(fa, stdx::element_aligned);
                        v.copy_to(va, stdx::element_aligned);
                        if (_mm512_cmp_pd_mask(_mm512_load_pd(fa),
                                               _mm512_load_pd(va), _CMP_GT_OQ) == 0)
                            break;
                    } else {
                        if (!stdx::any_of(F > v)) break;
                    }
#else
                    if (!stdx::any_of(F > v)) break;
#endif
                    // A cell lazy-F raises came from an EXTENSION -- its recorded code
                    // must be revised, or B disagrees with A exactly on the cells this
                    // correction exists to fix.
                    vd k; k.copy_from(yc + (std::size_t)s * W, stdx::element_aligned);
#if defined(__clang__) && defined(__AVX512F__) && !defined(NWGRAD_STD_SIMD_AVX512_MASK_OK)
                    if constexpr (std::is_same_v<T, double>) {
                        k = avx512d_blend(avx512d_gt(F, v), k, vtwo);
                    } else {
                        stdx::where(F > v, k) = vtwo;
                    }
#else
                    stdx::where(F > v, k) = vtwo;
#endif
                    k.copy_to(yc + (std::size_t)s * W, stdx::element_aligned);
                    v = stdx::max(v, F);
                    v.copy_to(cY + off + (std::size_t)s * W, stdx::element_aligned);
                    F = v - vge_a;
                    changed = true;
                }
                if (!changed) break;
            }
            for (int s = 0; s < seg; ++s) {
                vd k; k.copy_from(yc + (std::size_t)s * W, stdx::element_aligned);
                store_codes<T, W>(k, dY + (std::size_t)s * W);
            }

            if constexpr (Local) {
                for (int j = 1; j <= n; ++j) {
                    const std::size_t k = scol(j);
                    const T mm = cM[off + k], xx = cX[off + k], yy = cY[off + k];
                    const T here = std::max({mm, xx, yy});
                    if (here > best_local) {
                        best_local = here; best_i = i; best_j = j;
                        best_tbl = (mm >= xx && mm >= yy) ? 0 : ((xx >= yy) ? 1 : 2);
                    }
                }
            }
        }
        bM = nbM; bX = nbX; bY = K_NINF;
        std::swap(pM, cM); std::swap(pX, cX); std::swap(pY, cY);
    }

    if constexpr (Local) {
        job.score = best_local;
        job.best_i = best_i; job.best_j = best_j; job.best_tbl = best_tbl;
    } else {
        const std::size_t fn = (n == 0) ? 0 : off + scol(n);
        const T fm = pM[fn], fx = pX[fn], fy = pY[fn];   // pM/pX/pY are row m after the swap
        job.score = std::max({fm, fx, fy});
        job.best_i = m; job.best_j = n;
        job.best_tbl = (fm >= fx && fm >= fy) ? 0 : ((fx >= fy) ? 1 : 2);
    }
    job.table_layout = 1;
    job.seg = seg;
    job.width = W;
}

template <class T>
static void striped_full_ptr_entry(ViterbiJob<T>& job) {
    if (job.align_mode) striped_affine_full_ptr<T, true>(job);
    else                striped_affine_full_ptr<T, false>(job);
}
