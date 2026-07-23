// ── Leveled affine Viterbi kernels ────────────────────────────────────────────
//
// This file is #include'd once per ISA level, inside that level's namespace, by a
// level TU that has already `#include <experimental/simd>` and the common headers
// (so `stdx`, DpBufferT, ViterbiJob are all visible in the global namespace).  Compiled
// with that level's real -march flag, so `stdx::native_simd<T>` here is the right
// width: for T=double 2 (sse2/neon), 4 (avx2), 8 (avx512); for T=float it is doubled
// (4 / 8 / 16), which is the whole point of the float32 precision — twice the lanes.
//
// THE ODR RULE: no std::simd type leaves this file.  Every entry takes plain data
// (ViterbiJob<T>) and uses std::simd on locals only.  Distinct native width per level
// keeps the std::simd instantiations from COMDAT-folding across the level TUs.
//
// The striped kernel covers affine Full (Global + Local — Local adds M-clamp-to-0 and
// an argmax) and is instantiated once per precision T (double and float).  GuideBanded
// still falls through to the row-wise kernel.  The striped forward writes VM/VX/VY in
// Farrar striped layout directly (no de-stripe copy); aligner.hpp's cell_index() reads
// the same layout, so the traceback/hard_grad see the tables the fill wrote.

#ifndef NWGRAD_LEVEL_NS
#  error "kernels_impl.inl must be included inside a level namespace by a level TU"
#endif

// ── AVX-512/clang std::simd mask workaround (T=double only), shared ──────────
// libstdc++'s <experimental/simd> mask path fails to COMPILE under clang at AVX-512
// width for T=double — its 512-bit mask reduction (_MaskImplX86Mixin::_S_to_bits)
// asserts the vector's 64-bit lane type is `long` (GCC's canonical 8-byte int), but
// clang canonicalizes it `long long`, so `static_assert(is_same_v<long long, long>)`
// fires (experimental/bits/simd_x86.h:4232) — a fixed mismatch in each compiler's
// type model, unaffected by clang or libstdc++ version (checked clang 18-22,
// libstdc++ 13-15 and current GCC trunk simd_x86.h — none guard this path). Below
// this file's own early-exit workaround (a reduction to bool) are two more general
// substitutes for `stdx::where(cond, dest) = value` compare-and-select, used by
// kernels_pointers_impl.inl and hb_kernel_impl.inl (both #include'd after this file,
// so these are visible there). T=float is untouched — its 32-bit lane type is `int`
// on both compilers, so the std::simd form still compiles. -DNWGRAD_STD_SIMD_AVX512_MASK_OK
// forces the std::simd form back if a future clang/libstdc++ pairing compiles it.
#if defined(__clang__) && defined(__AVX512F__) && !defined(NWGRAD_STD_SIMD_AVX512_MASK_OK)
static inline __mmask8 avx512d_ge(stdx::native_simd<double> a, stdx::native_simd<double> b) {
    alignas(64) double aa[8], bb[8];
    a.copy_to(aa, stdx::element_aligned);
    b.copy_to(bb, stdx::element_aligned);
    return _mm512_cmp_pd_mask(_mm512_load_pd(aa), _mm512_load_pd(bb), _CMP_GE_OQ);
}
static inline __mmask8 avx512d_gt(stdx::native_simd<double> a, stdx::native_simd<double> b) {
    alignas(64) double aa[8], bb[8];
    a.copy_to(aa, stdx::element_aligned);
    b.copy_to(bb, stdx::element_aligned);
    return _mm512_cmp_pd_mask(_mm512_load_pd(aa), _mm512_load_pd(bb), _CMP_GT_OQ);
}
// Same semantics as `stdx::where(mask, dest) = value`: lanes where the mask bit is
// set take `value`, the rest keep `dest`.
static inline stdx::native_simd<double> avx512d_blend(__mmask8 k, stdx::native_simd<double> dest,
                                                       stdx::native_simd<double> value) {
    alignas(64) double da[8], va[8], out[8];
    dest.copy_to(da, stdx::element_aligned);
    value.copy_to(va, stdx::element_aligned);
    _mm512_store_pd(out, _mm512_mask_blend_pd(k, _mm512_load_pd(da), _mm512_load_pd(va)));
    stdx::native_simd<double> res;
    res.copy_from(out, stdx::element_aligned);
    return res;
}
#endif

using vd = stdx::native_simd<double>;
static constexpr int KW = (int)vd::size();     // native double-lane count (row_kernel uses it)
static constexpr double K_NINF = -std::numeric_limits<double>::infinity();

// ── striped affine forward, Full (Global if !Local, else Local) ───────────────
// Templated on the Viterbi precision T (double or float32).  vd / W / the -inf
// sentinel are all local to T here, shadowing the file-scope double versions above.
template <class T, bool Local>
static void striped_affine_full(ViterbiJob<T>& job) {
    using vd = stdx::native_simd<T>;
    const int W = (int)vd::size();                       // native T-lane count for this level
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
    // VM/VX/VY hold the tables in STRIPED layout — one row of rowsz = (seg+1)*W scalars
    // per i, written in place, so the traceback/hard-gradient read them striped with no
    // per-row de-stripe copy.  Slot 0 is column 0 (the gap border); slots [W, W+sw) are
    // the striped columns 1..n (started at W so every W-wide access is aligned given the
    // 64-byte allocator — see dp_buffer.hpp), column j at W + ((j-1)%seg)*W + (j-1)/seg;
    // slots 1..W-1 pad.  aligner.hpp's cell_index() computes the identical index — the
    // two must stay in lockstep.
    const std::size_t rowsz = (std::size_t)(seg + 1) * W;
    const std::size_t off   = (std::size_t)W;   // striped columns start here (col 0 at 0)
    if (buf.sopenv.size() < sw)     buf.sopenv.resize(sw);
    if (buf.sprof.size()  < (std::size_t)nalpha * sw) buf.sprof.resize((std::size_t)nalpha * sw);
    const std::size_t vsz = (std::size_t)(m + 1) * rowsz;
    if (buf.VM.size() < vsz) { buf.VM.resize(vsz); buf.VX.resize(vsz); buf.VY.resize(vsz); }
    // striped slot within a row (offset past slot 0) for DP column j in 1..n
    auto scol = [seg](int j) -> std::size_t {
        return (std::size_t)((j - 1) % seg) * W + (j - 1) / seg;
    };

    // striped query profile: prof[c][s*W+l] = score(c, b[col(l,s)-1]), padding -> NINF
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

    // ── row 0, written striped straight into the tables (slot 0 = column 0) ──
    // This doubles as the previous row for i = 1 (read at buf.VM row 0 + off).  Global:
    // VM = NINF, VY = the Y-gap-open series; Local: VM = 0 everywhere.
    {
        T* z0M = buf.VM.data(); T* z0X = buf.VX.data(); T* z0Y = buf.VY.data();
        z0M[0] = 0; z0X[0] = K_NINF; z0Y[0] = K_NINF;                   // column 0
        for (std::size_t k = 0; k < sw; ++k) {                          // columns 1..n default NINF
            z0M[off + k] = K_NINF; z0X[off + k] = K_NINF; z0Y[off + k] = K_NINF;
        }
        for (int l = 0; l < W; ++l)
            for (int s = 0; s < seg; ++s) {
                const int j = l * seg + s + 1;
                if (j <= n) {
                    const std::size_t k = (std::size_t)s * W + l;
                    if constexpr (Local) z0M[off + k] = 0;
                    else                 z0Y[off + k] = -(go_a + j * ge_a);
                }
            }
    }

    const vd vgo_a(go_a), vge_a(ge_a), vgo_b(go_b), vge_b(ge_b);
    const vd vzero(static_cast<T>(0));
    // column-0 border of the previous row
    T bM = 0, bX = K_NINF, bY = K_NINF;

    T best_local = 0; int best_i = 0, best_j = 0, best_tbl = 0;

    for (int i = 1; i <= m; ++i) {
        // column-0 border of this row
        const T nbM = Local ? T(0) : K_NINF;
        const T nbX = Local ? K_NINF : -(go_b + i * ge_b);
        const T nbOpen = (std::max(nbM, nbX) - go_a) - ge_a;
        const T* sub = buf.sprof.data() + (std::size_t)a[i - 1] * sw;

        // this row and the previous row, striped, written in place in the tables
        T* cM = buf.VM.data() + (std::size_t)i * rowsz + off;
        T* cX = buf.VX.data() + (std::size_t)i * rowsz + off;
        T* cY = buf.VY.data() + (std::size_t)i * rowsz + off;
        const T* pM = buf.VM.data() + (std::size_t)(i - 1) * rowsz + off;
        const T* pX = buf.VX.data() + (std::size_t)(i - 1) * rowsz + off;
        const T* pY = buf.VY.data() + (std::size_t)(i - 1) * rowsz + off;
        // column 0 border of this row (slot 0)
        buf.VM.data()[(std::size_t)i * rowsz] = nbM;
        buf.VX.data()[(std::size_t)i * rowsz] = nbX;
        buf.VY.data()[(std::size_t)i * rowsz] = K_NINF;

        if (seg > 0) {
            // ── carry-free: VM (diagonal, clamped to 0 for Local), VX (same column) ──
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
                ((stdx::max(vmv, vxv) - vgo_a) - vge_a).copy_to(ov + (std::size_t)s * W, stdx::element_aligned);
            }

            // ── the carry: VY, same-lane striped chain + lazy-F ──
            vd prev([&](int q) { return q == 0 ? bY : K_NINF; });
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
                prev = v;
            }
            for (int r = 0; r < W; ++r) {
                vd last; last.copy_from(cY + (std::size_t)(seg - 1) * W, stdx::element_aligned);
                vd F([&](int q) { return q == 0 ? bY : last[q - 1]; });
                F = F - vge_a;
                bool changed = false;
                for (int s = 0; s < seg; ++s) {
                    vd v; v.copy_from(cY + (std::size_t)s * W, stdx::element_aligned);
                    // Lazy-F early-exit: stop once no lane of F exceeds v.  This is the
                    // ONLY std::simd *mask* operation in the whole kernel, and libstdc++'s
                    // <experimental/simd> fails to COMPILE it under clang at AVX-512 width
                    // *for T=double*.  Its 512-bit mask path (_MaskImplX86Mixin::_S_to_bits)
                    // asserts the vector's 64-bit integer lane type equals
                    // __int_for_sizeof_t<8> == `long` (GCC's canonical 8-byte int), but clang
                    // canonicalizes that lane as `long long`, so
                    // `static_assert(is_same_v<long long, long>)` fires
                    // (experimental/bits/simd_x86.h:4232).  The `long` vs `long long` choice
                    // is a fixed property of each compiler's type model, so it reproduces on
                    // clang 18-22 / libstdc++ 13-15 and no version bump or flag clears it.
                    // The T=float mask uses a 32-bit lane (__int_for_sizeof_t<4> == `int`,
                    // which both compilers agree on), so only the double instantiation needs
                    // the swap; float stays on the std::simd form.  Only this mask path is
                    // affected — every other op here compiles under clang.  Bit-exact:
                    // `_CMP_GT_OQ` matches std::simd's ordered `>` and this is only an
                    // early-exit gate.  -DNWGRAD_STD_SIMD_AVX512_MASK_OK forces the std::simd
                    // form back if a future clang compiles it.
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
                    v = stdx::max(v, F);
                    v.copy_to(cY + (std::size_t)s * W, stdx::element_aligned);
                    F = v - vge_a;
                    changed = true;
                }
                if (!changed) break;
            }

            // ── Local: argmax over this row, j ascending, reading the striped cells,
            // with the scalar kernel's M>X>Y tie-break and strict > (bit-exact path) ──
            if constexpr (Local) {
                for (int j = 1; j <= n; ++j) {
                    const std::size_t k = scol(j);
                    const T mm = cM[k], xx = cX[k], yy = cY[k];
                    const T here = std::max({mm, xx, yy});
                    if (here > best_local) {
                        best_local = here; best_i = i; best_j = j;
                        best_tbl = (mm >= xx && mm >= yy) ? 0 : ((xx >= yy) ? 1 : 2);
                    }
                }
            }
        }
        bM = nbM; bX = nbX; bY = K_NINF;
    }

    if constexpr (Local) {
        job.score = best_local;
        job.best_i = best_i; job.best_j = best_j; job.best_tbl = best_tbl;
    } else {
        // final cell (m, n), striped; column 0 (slot 0) when n == 0 (empty B)
        const std::size_t fn = (n == 0) ? 0 : off + scol(n);
        const T fm = buf.VM[(std::size_t)m * rowsz + fn];
        const T fx = buf.VX[(std::size_t)m * rowsz + fn];
        const T fy = buf.VY[(std::size_t)m * rowsz + fn];
        job.score = std::max({fm, fx, fy});
        job.best_i = m; job.best_j = n;
        job.best_tbl = (fm >= fx && fm >= fy) ? 0 : ((fx >= fy) ? 1 : 2);
    }
    job.table_layout = 1;   // striped (seg/width below tell the traceback how to index)
    job.seg = seg;
    job.width = W;
}

// The registered entry for a level, per precision: dispatch Full by mode.  GuideBanded
// is not handled here (the caller routes it to the row-wise kernel).
template <class T>
static void striped_full_entry(ViterbiJob<T>& job) {
    if (job.align_mode) striped_affine_full<T, true>(job);
    else                striped_affine_full<T, false>(job);
}
