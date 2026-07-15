// ── Leveled affine Viterbi kernels ────────────────────────────────────────────
//
// This file is #include'd once per ISA level, inside that level's namespace, by a
// level TU that has already `#include <experimental/simd>` and the common headers
// (so `stdx`, DpBuffer, ViterbiJob are all visible in the global namespace).  Compiled
// with that level's real -march flag, so `stdx::native_simd<double>` here is the right
// width: 2 (sse2/neon), 4 (avx2), 8 (avx512).
//
// THE ODR RULE: no std::simd type leaves this file.  Every entry takes plain data
// (ViterbiJob) and uses std::simd on locals only.  Distinct native width per level
// keeps the std::simd instantiations from COMDAT-folding across the level TUs.
//
// The striped kernel covers affine Full (Global + Local — Local adds M-clamp-to-0 and
// an argmax).  GuideBanded still falls through to the row-wise kernel.  The striped
// forward writes rolling striped rows and de-stripes each finished row into the
// row-major VM/VX/VY, so the traceback/hard_grad read the layout they already expect.

#ifndef NWGRAD_LEVEL_NS
#  error "kernels_impl.inl must be included inside a level namespace by a level TU"
#endif

using vd = stdx::native_simd<double>;
static constexpr int KW = (int)vd::size();     // native lane count for this level
static constexpr double K_NINF = -std::numeric_limits<double>::infinity();

// ── striped affine forward, Full (Global if !Local, else Local) ───────────────
template <bool Local>
static void striped_affine_full(ViterbiJob& job) {
    const int m = job.m, n = job.n, W = KW;
    const int nalpha = job.nalpha;
    const double go_a = job.go_a, ge_a = job.ge_a, go_b = job.go_b, ge_b = job.ge_b;
    const double* blk = job.blk;
    const unsigned char* a = job.a;
    const unsigned char* b = job.b;
    DpBuffer& buf = *job.buf;

    const int seg = (n + W - 1) / W;
    const std::size_t sw = (std::size_t)seg * W;
    const std::size_t stride = (std::size_t)n + 1;

    if (buf.srows.size()  < 6 * sw) buf.srows.resize(6 * sw);
    if (buf.sopenv.size() < sw)     buf.sopenv.resize(sw);
    if (buf.sprof.size()  < (std::size_t)nalpha * sw) buf.sprof.resize((std::size_t)nalpha * sw);
    const std::size_t vsz = (std::size_t)(m + 1) * stride;
    if (buf.VM.size() < vsz) { buf.VM.resize(vsz); buf.VX.resize(vsz); buf.VY.resize(vsz); }

    // striped query profile: prof[c][s*W+l] = score(c, b[col(l,s)-1]), padding -> NINF
    for (int c = 0; c < nalpha; ++c) {
        const double* row = blk + (std::size_t)c * nalpha;
        double* dst = buf.sprof.data() + (std::size_t)c * sw;
        for (int l = 0; l < W; ++l)
            for (int s = 0; s < seg; ++s) {
                const int j = l * seg + s + 1;
                dst[(std::size_t)s * W + l] = (j <= n) ? row[b[j - 1]] : K_NINF;
            }
    }

    double* pM = buf.srows.data() + 0 * sw; double* pX = buf.srows.data() + 1 * sw;
    double* pY = buf.srows.data() + 2 * sw; double* cM = buf.srows.data() + 3 * sw;
    double* cX = buf.srows.data() + 4 * sw; double* cY = buf.srows.data() + 5 * sw;
    double* ov = buf.sopenv.data();

    // row 0 (previous row for i=1), striped.  Global: VM=NINF, VY=Y-gap-open series.
    // Local: VM=0 everywhere (an alignment may start anywhere), VX=VY=NINF.
    for (std::size_t k = 0; k < sw; ++k) { pM[k] = K_NINF; pX[k] = K_NINF; pY[k] = K_NINF; }
    for (int l = 0; l < W; ++l)
        for (int s = 0; s < seg; ++s) {
            const int j = l * seg + s + 1;
            if (j <= n) {
                if constexpr (Local) pM[(std::size_t)s * W + l] = 0.0;
                else                 pY[(std::size_t)s * W + l] = -(go_a + j * ge_a);
            }
        }
    // row 0 in the row-major output
    {
        double* r0M = buf.VM.data(); double* r0X = buf.VX.data(); double* r0Y = buf.VY.data();
        if constexpr (Local) {
            for (int j = 0; j <= n; ++j) { r0M[j] = 0.0; r0X[j] = K_NINF; r0Y[j] = K_NINF; }
        } else {
            r0M[0] = 0.0; r0X[0] = K_NINF; r0Y[0] = K_NINF;
            for (int j = 1; j <= n; ++j) { r0M[j] = K_NINF; r0X[j] = K_NINF; r0Y[j] = -(go_a + j * ge_a); }
        }
    }

    const vd vgo_a(go_a), vge_a(ge_a), vgo_b(go_b), vge_b(ge_b);
    const vd vzero(0.0);
    // column-0 border of the previous row
    double bM = Local ? 0.0 : 0.0, bX = K_NINF, bY = K_NINF;

    double best_local = 0.0; int best_i = 0, best_j = 0, best_tbl = 0;

    for (int i = 1; i <= m; ++i) {
        // column-0 border of this row
        const double nbM = Local ? 0.0 : K_NINF;
        const double nbX = Local ? K_NINF : -(go_b + i * ge_b);
        const double nbOpen = (std::max(nbM, nbX) - go_a) - ge_a;
        const double* sub = buf.sprof.data() + (std::size_t)a[i - 1] * sw;

        double* rM = buf.VM.data() + (std::size_t)i * stride;
        double* rX = buf.VX.data() + (std::size_t)i * stride;
        double* rY = buf.VY.data() + (std::size_t)i * stride;
        rM[0] = nbM; rX[0] = nbX; rY[0] = K_NINF;

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
                    // <experimental/simd> fails to COMPILE it under clang at AVX-512 width.
                    // Its 512-bit mask path (_MaskImplX86Mixin::_S_to_bits) asserts the
                    // vector's 64-bit integer lane type equals __int_for_sizeof_t<8> ==
                    // `long` (GCC's canonical 8-byte int), but clang canonicalizes that
                    // lane as `long long`, so `static_assert(is_same_v<long long, long>)`
                    // fires (experimental/bits/simd_x86.h:4232).  The `long` vs `long long`
                    // choice is a fixed property of each compiler's type model, so it
                    // reproduces on clang 18-22 / libstdc++ 13-15 and no version bump or
                    // flag clears it (-fgnuc-version= only breaks the build elsewhere).
                    // Only this mask path is affected — every other op here (max/+/-/loads)
                    // compiles under clang — so we swap just this predicate for the
                    // equivalent AVX-512 intrinsic, and only in a clang AVX-512 TU.
                    // Bit-exact: `_CMP_GT_OQ` gives the same per-lane result as std::simd's
                    // ordered `>` (NaN compares false either way), and this is only an
                    // early-exit gate — `v = max(v, F)` below is unchanged, so at worst it
                    // iterates once more and re-applies an idempotent max.  If a future
                    // clang compiles the std::simd form, build with
                    // -DNWGRAD_STD_SIMD_AVX512_MASK_OK to force it back (or delete this #if).
#if defined(__clang__) && defined(__AVX512F__) && !defined(NWGRAD_STD_SIMD_AVX512_MASK_OK)
                    {
                        alignas(64) double fa[8], va[8];
                        F.copy_to(fa, stdx::element_aligned);
                        v.copy_to(va, stdx::element_aligned);
                        if (_mm512_cmp_pd_mask(_mm512_load_pd(fa),
                                               _mm512_load_pd(va), _CMP_GT_OQ) == 0)
                            break;
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

            // ── de-stripe this row into row-major VM/VX/VY ──
            for (int l = 0; l < W; ++l)
                for (int s = 0; s < seg; ++s) {
                    const int j = l * seg + s + 1;
                    if (j <= n) {
                        const std::size_t k = (std::size_t)s * W + l;
                        rM[j] = cM[k]; rX[j] = cX[k]; rY[j] = cY[k];
                    }
                }

            // ── Local: argmax over this row, in row-major (j ascending) order with the
            // scalar kernel's M>X>Y tie-break and strict > — reproduces its bit-exact path.
            if constexpr (Local) {
                for (int j = 1; j <= n; ++j) {
                    const double mm = rM[j], xx = rX[j], yy = rY[j];
                    const double here = std::max({mm, xx, yy});
                    if (here > best_local) {
                        best_local = here; best_i = i; best_j = j;
                        best_tbl = (mm >= xx && mm >= yy) ? 0 : ((xx >= yy) ? 1 : 2);
                    }
                }
            }

            std::swap(pM, cM); std::swap(pX, cX); std::swap(pY, cY);
        }
        bM = nbM; bX = nbX; bY = K_NINF;
    }

    if constexpr (Local) {
        job.score = best_local;
        job.best_i = best_i; job.best_j = best_j; job.best_tbl = best_tbl;
    } else {
        const double fm = buf.VM[(std::size_t)m * stride + n];
        const double fx = buf.VX[(std::size_t)m * stride + n];
        const double fy = buf.VY[(std::size_t)m * stride + n];
        job.score = std::max({fm, fx, fy});
        job.best_i = m; job.best_j = n;
        job.best_tbl = (fm >= fx && fm >= fy) ? 0 : ((fx >= fy) ? 1 : 2);
    }
    job.table_layout = 0;   // row-major (de-striped)
}

// The registered entry for a level: dispatch Full by mode.  GuideBanded is not handled
// here (the caller routes it to the row-wise kernel).
static void striped_full_entry(ViterbiJob& job) {
    if (job.align_mode) striped_affine_full<true>(job);
    else                striped_affine_full<false>(job);
}
