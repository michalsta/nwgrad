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
// First cut: the striped kernel covers affine Global + Full (the hot path).  Local-Full
// and GuideBanded fall through to the row-wise kernel (levelled the same way, added in
// kernels_rowwise below).  The striped forward writes rolling striped rows and
// de-stripes each finished row into the row-major VM/VX/VY, so the traceback/hard_grad
// read the layout they already expect.

#ifndef NWGRAD_LEVEL_NS
#  error "kernels_impl.inl must be included inside a level namespace by a level TU"
#endif

using vd = stdx::native_simd<double>;
static constexpr int KW = (int)vd::size();     // native lane count for this level
static constexpr double K_NINF = -std::numeric_limits<double>::infinity();

// ── striped affine forward, Global + Full ─────────────────────────────────────
static void striped_affine_global_full(ViterbiJob& job) {
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

    // row 0 (previous row for i=1): VM/VX = NINF, VY = the Y-gap-open series, striped.
    for (std::size_t k = 0; k < sw; ++k) { pM[k] = K_NINF; pX[k] = K_NINF; pY[k] = K_NINF; }
    for (int l = 0; l < W; ++l)
        for (int s = 0; s < seg; ++s) {
            const int j = l * seg + s + 1;
            if (j <= n) pY[(std::size_t)s * W + l] = -(go_a + j * ge_a);
        }
    // row 0 in the row-major output
    {
        double* r0M = buf.VM.data(); double* r0X = buf.VX.data(); double* r0Y = buf.VY.data();
        r0M[0] = 0.0; r0X[0] = K_NINF; r0Y[0] = K_NINF;
        for (int j = 1; j <= n; ++j) { r0M[j] = K_NINF; r0X[j] = K_NINF; r0Y[j] = -(go_a + j * ge_a); }
    }

    const vd vgo_a(go_a), vge_a(ge_a), vgo_b(go_b), vge_b(ge_b);
    double bM = 0.0, bX = K_NINF, bY = K_NINF;   // column-0 border of the previous row

    for (int i = 1; i <= m; ++i) {
        const double nbX = -(go_b + i * ge_b);   // column-0 border of this row
        const double nbOpen = (nbX - go_a) - ge_a;
        const double* sub = buf.sprof.data() + (std::size_t)a[i - 1] * sw;

        // ── carry-free: VM (diagonal), VX (same column) ──
        for (int s = 0; s < seg; ++s) {
            vd dM, dX, dY;
            if (s == 0) {                              // segment boundary: shift + border
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
                if (!stdx::any_of(F > v)) break;
                v = stdx::max(v, F);
                v.copy_to(cY + (std::size_t)s * W, stdx::element_aligned);
                F = v - vge_a;
                changed = true;
            }
            if (!changed) break;
        }

        // ── de-stripe this row into row-major VM/VX/VY ──
        double* rM = buf.VM.data() + (std::size_t)i * stride;
        double* rX = buf.VX.data() + (std::size_t)i * stride;
        double* rY = buf.VY.data() + (std::size_t)i * stride;
        rM[0] = K_NINF; rX[0] = nbX; rY[0] = K_NINF;
        for (int l = 0; l < W; ++l)
            for (int s = 0; s < seg; ++s) {
                const int j = l * seg + s + 1;
                if (j <= n) {
                    const std::size_t k = (std::size_t)s * W + l;
                    rM[j] = cM[k]; rX[j] = cX[k]; rY[j] = cY[k];
                }
            }

        std::swap(pM, cM); std::swap(pX, cX); std::swap(pY, cY);
        bM = K_NINF; bX = nbX; bY = K_NINF;
    }

    // Global result at (m, n), read from the row-major tables.
    const double fm = buf.VM[(std::size_t)m * stride + n];
    const double fx = buf.VX[(std::size_t)m * stride + n];
    const double fy = buf.VY[(std::size_t)m * stride + n];
    job.score = std::max({fm, fx, fy});
    job.best_i = m; job.best_j = n;
    job.best_tbl = (fm >= fx && fm >= fy) ? 0 : ((fx >= fy) ? 1 : 2);
    job.table_layout = 0;   // row-major (de-striped)
}
