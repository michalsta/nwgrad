// ── Leveled Hirschberg sweep: striped, std::simd, rolling rows only ───────────
//
// #include'd once per ISA level inside that level's namespace, same contract as
// kernels_impl.inl: no std::simd type leaves this file, every entry takes plain data
// (HbJob<T>) and uses std::simd on locals only.
//
// This is the striped affine recurrence with the tables deleted.  Only the final row
// survives, so there is nothing to write per cell and the whole sweep stays in L1/L2 —
// which is exactly why Hirschberg does not saturate memory bandwidth and the shipped
// Pointers kernel does.  Consequently this kernel is pure compute, and the thing that
// bounds it is the VY latency chain.  Striping is what breaks that chain: lane l owns
// columns l*seg+1 .. l*seg+seg, so VY's predecessor (column c-1) is the SAME lane at
// segment s-1 — a same-lane vector dependency with no shuffle, instead of a
// neighbouring-lane one.  That is Farrar's layout, used here for its dependency
// structure rather than for any table it produces.
//
// WHY STRIPED AND NOT THE ROW-WISE (contiguous) FORM.  The row-wise banded kernel does
// not vectorize the VY carry at all; it interleaves blocks so a later block's carry-free
// VM/VX can issue underneath an earlier block's serial VY.  That works when the carry is
// a small share of the work.  Here there is no table traffic to hide behind, so the
// carry IS the work, and it has to be vectorized rather than overlapped.  Contiguous
// lazy-F was measured a wash in this project precisely because its carry needs
// in-register shifts; striped needs none.
//
// BIT-EXACTNESS.  This kernel must agree with the scalar hb_fwd/hb_rev to the last bit,
// or scalar and simd Hirschberg would return different paths and "Hirschberg" would name
// two algorithms.  So: `(v - go) - ge` stays left-associated and is never folded to
// `v - (go+ge)`; `(max(M,X) - go_a) - ge_a` is the open value, computed as a max THEN
// two subtractions (identical bits to subtracting first, since a common subtrahend is
// order-preserving and deterministic); and the VY chain propagates one `- ge_a` per
// step rather than any closed form.
//
// NOT USED HERE: the closed-form prefix-max trick (VY[c] = prefixmax(open[k] + k*ge_a)
// - c*ge_a) would turn the carry into a max-only chain with no lazy-F, but it
// re-associates the gap arithmetic and is therefore NOT bit-exact with the scalar chain.
// It is implemented as an OPT-IN SIBLING further down this file
// (hb_sweep_striped_pmax, TracebackMode::HirschbergPmax) — never as a replacement.

#ifndef NWGRAD_LEVEL_NS
#  error "hb_kernel_impl.inl must be included inside a level namespace by a level TU"
#endif

// The AVX-512/clang std::simd mask workaround (avx512d_ge / avx512d_gt / avx512d_blend)
// used below is defined once in kernels_impl.inl, #include'd before this file — see
// the note there. Every masked `stdx::where(cond, dest) = value` site in this file
// goes through it for T=double; T=float is unaffected and keeps the std::simd form.

// One half-sweep of a Hirschberg block.  Rows are walked by (a_start, a_step) and
// columns by (b_start, b_step), so the forward and reverse halves are the same code
// with the steps negated — there is one carry implementation, not two.
template <class T>
static void hb_sweep_striped(HbJob<T>& job) {
    using vd = stdx::native_simd<T>;
    const int W = (int)vd::size();
    const T NINF = -std::numeric_limits<T>::infinity();

    const int H = job.H, NC = job.ncols;
    const T go_a = job.go_a, ge_a = job.ge_a, go_b = job.go_b, ge_b = job.ge_b;
    const unsigned char* a = job.a;
    const unsigned char* b = job.b;
    const T* blk = job.blk;
    const int nalpha = job.nalpha;
    DpBufferT<T>& buf = *job.buf;

    T* oM = job.outM; T* oX = job.outX; T* oY = job.outY;

    // ── row 0 of the block, scalar and contiguous ────────────────────────────
    //
    // Deliberately not vectorized: it is O(ncols) once per sweep against O(H*ncols) for
    // the body, and it is a pure serial VY chain.  Computing it with the same iterative
    // `- ge_a` the scalar path uses is what keeps the two bit-identical — the closed
    // form -(go_a + c*ge_a) rounds differently.
    // It is built in the caller's OUTPUT row, which is the right shape already and gets
    // overwritten by the de-stripe at the end — so the sweep needs no scratch of its own
    // for this, and the H == 0 case leaves the correct answer in place by construction.
    T* r0M = oM; T* r0X = oX; T* r0Y = oY;
    r0M[0] = job.in_x ? NINF : T(0);
    r0X[0] = job.in_x ? T(0) : NINF;
    r0Y[0] = NINF;
    for (int c = 1; c <= NC; ++c) {
        r0M[c] = NINF;
        r0X[c] = NINF;
        const T open = (std::max(r0M[c - 1], r0X[c - 1]) - go_a) - ge_a;
        r0Y[c] = std::max(open, r0Y[c - 1] - ge_a);
    }
    if (H == 0) return;                 // nothing to sweep: row 0 IS the answer, in place

    // ── striping geometry ────────────────────────────────────────────────────
    const int seg = (NC + W - 1) / W;
    const std::size_t sw = (std::size_t)seg * W;
    auto fit = [](typename DpBufferT<T>::TVec& v, std::size_t k) {
        if (v.size() < k) v.resize(k);
    };
    fit(buf.hfa, sw); fit(buf.hfb, sw); fit(buf.hfc, sw);
    fit(buf.hfd, sw); fit(buf.hfe, sw); fit(buf.hff, sw);
    fit(buf.hov, sw);
    fit(buf.hprof, (std::size_t)nalpha * sw);

    T* pM = buf.hfa.data(); T* pX = buf.hfb.data(); T* pY = buf.hfc.data();
    T* cM = buf.hfd.data(); T* cX = buf.hfe.data(); T* cY = buf.hff.data();
    T* ov = buf.hov.data();

    // Striped query profile over the block's column slice.  prof[sym][s*W+l] is the
    // score of `sym` against the residue in column l*seg+s+1.  Built per block because
    // the striping geometry depends on the block's own width — but it serves this
    // block's forward AND reverse sweeps, so it amortizes over H rows.  Padding columns
    // get -inf so they can never win a max.
    for (int sym = 0; sym < nalpha; ++sym) {
        const T* row = blk + (std::size_t)sym * nalpha;
        T* dst = buf.hprof.data() + (std::size_t)sym * sw;
        for (int l = 0; l < W; ++l)
            for (int s = 0; s < seg; ++s) {
                const int c = l * seg + s + 1;
                dst[(std::size_t)s * W + l] =
                    (c <= NC) ? row[b[job.b_start + (c - 1) * job.b_step]] : NINF;
            }
    }

    // Stripe row 0 into pM/pX/pY; its column 0 stays in the border scalars.
    for (int l = 0; l < W; ++l)
        for (int s = 0; s < seg; ++s) {
            const int c = l * seg + s + 1;
            const std::size_t k = (std::size_t)s * W + l;
            pM[k] = (c <= NC) ? r0M[c] : NINF;
            pX[k] = (c <= NC) ? r0X[c] : NINF;
            pY[k] = (c <= NC) ? r0Y[c] : NINF;
        }
    T bM = r0M[0], bX = r0X[0], bY = r0Y[0];

    const vd vgo_a(go_a), vge_a(ge_a), vgo_b(go_b), vge_b(ge_b);

    for (int t = 0; t < H; ++t) {
        const T* sub = buf.hprof.data() +
                       (std::size_t)a[job.a_start + t * job.a_step] * sw;

        // Column 0 of this row.  M cannot happen there (no column to pair with) and Y
        // cannot either (Y consumes a column); only X, extending downward.
        const T nbM = NINF;
        const T nbX = std::max(std::max((bM - go_b) - ge_b, bX - ge_b), (bY - go_b) - ge_b);
        const T nbY = NINF;
        const T nbOpen = (std::max(nbM, nbX) - go_a) - ge_a;

        // ── carry-free half: VM (diagonal) and VX (same column) ──────────────
        // Both read only the previous row, so every lane is independent.  The s == 0
        // segment is the one that reaches across a lane boundary: column c-1 for the
        // first column of each lane lives in the PREVIOUS lane's last segment, so the
        // vector is rebuilt shifted by one lane with the row border filling lane 0.
        // The generator constructor lowers to an unaligned load, not scalar inserts.
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
            const vd vmv = stdx::max(stdx::max(dM, dX), dY) + sb;
            vmv.copy_to(cM + (std::size_t)s * W, stdx::element_aligned);

            vd uM, uX, uY;
            uM.copy_from(pM + (std::size_t)s * W, stdx::element_aligned);
            uX.copy_from(pX + (std::size_t)s * W, stdx::element_aligned);
            uY.copy_from(pY + (std::size_t)s * W, stdx::element_aligned);
            const vd vxv = stdx::max(stdx::max((uM - vgo_b) - vge_b, uX - vge_b),
                                     (uY - vgo_b) - vge_b);
            vxv.copy_to(cX + (std::size_t)s * W, stdx::element_aligned);
            // The value a new Y-gap would open with, for the carry pass below.
            ((stdx::max(vmv, vxv) - vgo_a) - vge_a)
                .copy_to(ov + (std::size_t)s * W, stdx::element_aligned);
        }

        // ── the carry: VY, same-lane striped chain, then lazy-F ──────────────
        // The base sweep assumes each lane's chain starts fresh; a gap that runs past a
        // lane boundary is repaired by the correction rounds below.  Lazy-F here is
        // EXACT rather than approximate: it propagates one `- ge_a` at a time, which is
        // precisely the chain the scalar loop walks, so the fixpoint is identical
        // rather than merely close.
        vd prev([&](int q) { return q == 0 ? bY : NINF; });
        for (int s = 0; s < seg; ++s) {
            vd O;
            if (s == 0) {
                vd lo; lo.copy_from(ov + (std::size_t)(seg - 1) * W, stdx::element_aligned);
                O = vd([&](int q) { return q == 0 ? nbOpen : lo[q - 1]; });
            } else {
                O.copy_from(ov + (std::size_t)(s - 1) * W, stdx::element_aligned);
            }
            const vd v = stdx::max(O, prev - vge_a);
            v.copy_to(cY + (std::size_t)s * W, stdx::element_aligned);
            prev = v;
        }
        // seg == 0 means NC == 0 — a zero-width block, which hb_solve DOES produce when
        // the optimal path crosses at a column extreme.  The carry-free and carry loops
        // above are already `for s<seg` no-ops then, but the lazy-F sweep below runs W
        // times regardless and reads `cY + (seg-1)*W` = cY - W (size_t wrap) — 32 bytes
        // before the buffer.  Guard it; there is nothing to correct with no columns.
        if (seg > 0) for (int r = 0; r < W; ++r) {
            vd last; last.copy_from(cY + (std::size_t)(seg - 1) * W, stdx::element_aligned);
            vd F([&](int q) { return q == 0 ? bY : last[q - 1]; });
            F = F - vge_a;
            bool changed = false;
            for (int s = 0; s < seg; ++s) {
                vd v; v.copy_from(cY + (std::size_t)s * W, stdx::element_aligned);
                // clang + libstdc++ std::simd cannot compile the AVX-512 mask reduction
                // for T=double (its 64-bit lane asserts `long` but clang canonicalizes
                // `long long`).  Same swap as kernels_impl.inl — see the long note there.
#if defined(__clang__) && defined(__AVX512F__) && !defined(NWGRAD_STD_SIMD_AVX512_MASK_OK)
                if constexpr (std::is_same_v<T, double>) {
                    alignas(64) double fa[8], va[8];
                    F.copy_to(fa, stdx::element_aligned);
                    v.copy_to(va, stdx::element_aligned);
                    if (_mm512_cmp_pd_mask(_mm512_load_pd(fa), _mm512_load_pd(va), _CMP_GT_OQ) == 0) break;
                } else { if (!stdx::any_of(F > v)) break; }
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

        std::swap(pM, cM); std::swap(pX, cX); std::swap(pY, cY);
        bM = nbM; bX = nbX; bY = nbY;
    }

    // ── de-stripe the final row ──────────────────────────────────────────────
    // O(ncols) once per sweep.  The join that consumes this scans by column, and making
    // it stripe-aware would spread the layout into the recursion for no gain.
    oM[0] = bM; oX[0] = bX; oY[0] = bY;
    for (int l = 0; l < W; ++l)
        for (int s = 0; s < seg; ++s) {
            const int c = l * seg + s + 1;
            if (c <= NC) {
                const std::size_t k = (std::size_t)s * W + l;
                oM[c] = pM[k]; oX[c] = pX[k]; oY[c] = pY[k];
            }
        }
}

static void hb_sweep_entry_d(HbJob<double>& j) { hb_sweep_striped<double>(j); }
static void hb_sweep_entry_f(HbJob<float>&  j) { hb_sweep_striped<float>(j); }

// ── OPT-IN SIBLING: the same sweep with a CLOSED-FORM prefix-max carry ────────────
//
// Selected by TracebackMode::HirschbergPmax only.  It is NOT bit-exact with
// hb_sweep_striped above and never replaces it; it exists to answer whether the VY
// latency chain is the ceiling of the sweep, at a measured cost in exactness.
//
// THE ALGEBRA.  Write g[k] for the value a Y-gap opens with after column k, i.e.
// g[k] = (max(VM[k], VX[k]) - go_a) - ge_a (g[0] coming from the column-0 border).  The
// serial recurrence VY[c] = max(g[c-1], VY[c-1] - ge_a) unrolls to
//
//     VY[c] = max over k <= c-1 of ( g[k] - (c-1-k)*ge_a ),  and also  bY - c*ge_a
//           = ( max over k <= c-1 of ( g[k] + k*ge_a ) ) - (c-1)*ge_a
//
// with the border folded in as the seed g[-1] + (-1)*ge_a := bY - ge_a.  So the carry
// becomes a PREFIX MAX over Q[k] = g[k] + k*ge_a followed by one subtraction.  Two
// things follow, and they are the whole point:
//   * the chain is max-only (no `- ge_a` in it), roughly halving its latency; and
//   * lazy-F disappears entirely — a prefix max composes across lane boundaries by a
//     plain max of lane totals, so the cross-lane fixup is W scalar maxes per row
//     instead of up to W correction sweeps over the whole row.
//
// WHY IT IS INEXACT.  Q[k] adds k*ge_a and the last step subtracts it again.  The
// intermediate is O(k*ge_a) where the answer is O(g[k]), so the value carries a rounding
// proportional to the COLUMN INDEX, not to the gap-run length.  (The often-quoted
// L*eps model — L the winning run length — is what you would get from a per-run rebase,
// which is not what a single global prefix max does.  Measured, not assumed: see
// tests/Python/test_hirschberg_pmax.py.)  The path can therefore come back SUBOPTIMAL,
// a different kind of error from plain Hirschberg's tie-break (which is exactly optimal,
// just a different optimum).
//
// WHAT DOES SURVIVE: the family is self-consistent to the bit.  Every quantity above
// depends on the ABSOLUTE column index k and on nothing else — not on W, not on seg —
// and max is exact and associative, so the per-lane-then-across-lane prefix max equals a
// serial one exactly.  pmax-scalar == pmax-sse2 == pmax-avx2 == pmax-avx512 == pmax-neon,
// bit for bit, which is what keeps NWGRAD_ISA a speed knob here as everywhere else.
//
// The ramp k*ge_a is precomputed into buf.hramp by a scalar loop, once per sweep, for
// two reasons: it depends on the column and not the row, and a multiply adjacent to the
// add in the hot loop is exactly what a compiler contracts into an FMA — which would
// round differently in the -mfma level TUs than in the baseline one and shatter the
// bit-identity above.  Loading a precomputed ramp leaves nothing to contract.
template <class T>
static void hb_sweep_striped_pmax(HbJob<T>& job) {
    using vd = stdx::native_simd<T>;
    const int W = (int)vd::size();
    const T NINF = -std::numeric_limits<T>::infinity();

    const int H = job.H, NC = job.ncols;
    const T go_a = job.go_a, ge_a = job.ge_a, go_b = job.go_b, ge_b = job.ge_b;
    const unsigned char* a = job.a;
    const unsigned char* b = job.b;
    const T* blk = job.blk;
    const int nalpha = job.nalpha;
    DpBufferT<T>& buf = *job.buf;

    T* oM = job.outM; T* oX = job.outX; T* oY = job.outY;

    // ── row 0 of the block: kept EXACT (serial), deliberately ────────────────
    // It is O(ncols) once per sweep against O(H*ncols) for the body, so approximating it
    // would buy nothing measurable and would add error for free.  Identical code to
    // hb_sweep_striped's row 0, and the scalar pmax reference does the same — which is
    // what keeps the family bit-identical.
    T* r0M = oM; T* r0X = oX; T* r0Y = oY;
    r0M[0] = job.in_x ? NINF : T(0);
    r0X[0] = job.in_x ? T(0) : NINF;
    r0Y[0] = NINF;
    for (int c = 1; c <= NC; ++c) {
        r0M[c] = NINF;
        r0X[c] = NINF;
        const T open = (std::max(r0M[c - 1], r0X[c - 1]) - go_a) - ge_a;
        r0Y[c] = std::max(open, r0Y[c - 1] - ge_a);
    }
    if (H == 0) return;                 // nothing to sweep: row 0 IS the answer, in place

    const int seg = (NC + W - 1) / W;
    const std::size_t sw = (std::size_t)seg * W;
    auto fit = [](typename DpBufferT<T>::TVec& v, std::size_t k) {
        if (v.size() < k) v.resize(k);
    };
    fit(buf.hfa, sw); fit(buf.hfb, sw); fit(buf.hfc, sw);
    fit(buf.hfd, sw); fit(buf.hfe, sw); fit(buf.hff, sw);
    fit(buf.hov, sw); fit(buf.hramp, sw);
    fit(buf.hprof, (std::size_t)nalpha * sw);

    T* pM = buf.hfa.data(); T* pX = buf.hfb.data(); T* pY = buf.hfc.data();
    T* cM = buf.hfd.data(); T* cX = buf.hfe.data(); T* cY = buf.hff.data();
    T* ov = buf.hov.data();
    T* rmp = buf.hramp.data();

    // The gap ramp, striped: slot (s,l) carries k*ge_a for the ABSOLUTE column index
    // k = l*seg + s.  Scalar, so it is bit-identical to the scalar reference's
    // `static_cast<T>(k) * ge_a` by construction, and outside the row loop so it costs
    // O(ncols) per sweep rather than per row.  k is an exact integer well below 2^24
    // (float32's exact-integer bound) for any sequence this library can hold, so the
    // multiply is the only rounding here.
    for (int l = 0; l < W; ++l)
        for (int s = 0; s < seg; ++s)
            rmp[(std::size_t)s * W + l] = static_cast<T>(l * seg + s) * ge_a;

    // Striped query profile over the block's column slice — identical to the exact sweep.
    for (int sym = 0; sym < nalpha; ++sym) {
        const T* row = blk + (std::size_t)sym * nalpha;
        T* dst = buf.hprof.data() + (std::size_t)sym * sw;
        for (int l = 0; l < W; ++l)
            for (int s = 0; s < seg; ++s) {
                const int c = l * seg + s + 1;
                dst[(std::size_t)s * W + l] =
                    (c <= NC) ? row[b[job.b_start + (c - 1) * job.b_step]] : NINF;
            }
    }

    for (int l = 0; l < W; ++l)
        for (int s = 0; s < seg; ++s) {
            const int c = l * seg + s + 1;
            const std::size_t k = (std::size_t)s * W + l;
            pM[k] = (c <= NC) ? r0M[c] : NINF;
            pX[k] = (c <= NC) ? r0X[c] : NINF;
            pY[k] = (c <= NC) ? r0Y[c] : NINF;
        }
    T bM = r0M[0], bX = r0X[0], bY = r0Y[0];

    const vd vgo_a(go_a), vge_a(ge_a), vgo_b(go_b), vge_b(ge_b);

    for (int t = 0; t < H; ++t) {
        const T* sub = buf.hprof.data() +
                       (std::size_t)a[job.a_start + t * job.a_step] * sw;

        const T nbM = NINF;
        const T nbX = std::max(std::max((bM - go_b) - ge_b, bX - ge_b), (bY - go_b) - ge_b);
        const T nbY = NINF;
        const T nbOpen = (std::max(nbM, nbX) - go_a) - ge_a;

        // ── carry-free half: VM (diagonal) and VX (same column) ──────────────
        // Byte-for-byte the exact sweep's; pmax changes only what comes after.
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
            const vd vmv = stdx::max(stdx::max(dM, dX), dY) + sb;
            vmv.copy_to(cM + (std::size_t)s * W, stdx::element_aligned);

            vd uM, uX, uY;
            uM.copy_from(pM + (std::size_t)s * W, stdx::element_aligned);
            uX.copy_from(pX + (std::size_t)s * W, stdx::element_aligned);
            uY.copy_from(pY + (std::size_t)s * W, stdx::element_aligned);
            const vd vxv = stdx::max(stdx::max((uM - vgo_b) - vge_b, uX - vge_b),
                                     (uY - vgo_b) - vge_b);
            vxv.copy_to(cX + (std::size_t)s * W, stdx::element_aligned);
            ((stdx::max(vmv, vxv) - vgo_a) - vge_a)
                .copy_to(ov + (std::size_t)s * W, stdx::element_aligned);
        }

        // ── pass 1: Q[k] = g[k] + k*ge_a, then a LANE-LOCAL prefix max ───────
        // The `O` vector is built exactly as in the exact sweep (segment s of lane l
        // wants g[l*seg + s], which lives one segment back, or in the previous lane's
        // last segment when s == 0).  The chain here is a single max: no subtraction
        // sits in it, which is the latency the whole variant is trying to buy back.
        vd P(NINF);
        for (int s = 0; s < seg; ++s) {
            vd O;
            if (s == 0) {
                vd lo; lo.copy_from(ov + (std::size_t)(seg - 1) * W, stdx::element_aligned);
                O = vd([&](int q) { return q == 0 ? nbOpen : lo[q - 1]; });
            } else {
                O.copy_from(ov + (std::size_t)(s - 1) * W, stdx::element_aligned);
            }
            vd rq; rq.copy_from(rmp + (std::size_t)s * W, stdx::element_aligned);
            P = stdx::max(O + rq, P);
            P.copy_to(cY + (std::size_t)s * W, stdx::element_aligned);
        }

        // ── the cross-lane join: W scalar maxes, and EXACT ──────────────────
        // Lane order IS column order, so folding the running maximum through the lane
        // totals reproduces a serial prefix max over the whole row exactly — max neither
        // rounds nor cares about association.  This is what lazy-F cost W sweeps to do.
        T tot[64], cin[64];                     // 64 >= any native_simd width we build
        P.copy_to(tot, stdx::element_aligned);
        T run = bY - ge_a;                      // the border, folded in as the seed
        for (int l = 0; l < W; ++l) { cin[l] = run; run = std::max(run, tot[l]); }
        vd vcin; vcin.copy_from(cin, stdx::element_aligned);

        // ── pass 2: VY[c] = max(prefix, seed) - (c-1)*ge_a ───────────────────
        // Fully parallel: no lane depends on another, and no s depends on s-1.
        for (int s = 0; s < seg; ++s) {
            vd p; p.copy_from(cY + (std::size_t)s * W, stdx::element_aligned);
            vd rq; rq.copy_from(rmp + (std::size_t)s * W, stdx::element_aligned);
            (stdx::max(p, vcin) - rq).copy_to(cY + (std::size_t)s * W, stdx::element_aligned);
        }

        std::swap(pM, cM); std::swap(pX, cX); std::swap(pY, cY);
        bM = nbM; bX = nbX; bY = nbY;
    }

    oM[0] = bM; oX[0] = bX; oY[0] = bY;
    for (int l = 0; l < W; ++l)
        for (int s = 0; s < seg; ++s) {
            const int c = l * seg + s + 1;
            if (c <= NC) {
                const std::size_t k = (std::size_t)s * W + l;
                oM[c] = pM[k]; oX[c] = pX[k]; oY[c] = pY[k];
            }
        }
}

static void hb_sweep_pmax_entry_d(HbJob<double>& j) { hb_sweep_striped_pmax<double>(j); }
static void hb_sweep_pmax_entry_f(HbJob<float>&  j) { hb_sweep_striped_pmax<float>(j); }

// ── Hirschberg LOCAL endpoint scan: striped affine sweep that reports its argmax ──
//
// This is hb_sweep_striped's striped recurrence with the tables still deleted, but
// instead of a final row it keeps only the single best cell over the WHOLE block.  It
// is what Smith-Waterman needs before divide-and-conquer can start: the FORWARD pass
// (Local = clamped, local borders) finds the end cell (ie, je); the REVERSE pass
// (Local = false, hb_sweep's global borders, no clamp) computes Rev = the best global
// alignment of the suffixes and finds the start.  One recurrence, the template picks
// the borders and the clamp so nothing branches per cell.
//
// The argmax is done PER ROW, vectorized: each lane keeps its own leftmost best column
// (columns grow with the segment index within a lane, so the strict-`>` update keeps the
// smaller column on a tie); a short horizontal pass across lanes then takes the leftmost
// lane (lane order IS column order in the striped layout).  Rolled into a running global
// best with a strict-`>` topmost-row update, this reproduces exactly the scalar scan's
// topmost-row / leftmost-column tie-break — the property that keeps the scalar and simd
// scans bit-identical (forcing NWGRAD_ISA must never move the endpoint).
//
// Padding lanes need no mask.  In Local mode they clamp to 0, which can never beat a
// real positive value nor lift the strict-`>` global best above its initial 0; in global
// mode they stay -inf.  Either way they lose every comparison.  The one masked op is the
// leftmost-column blend, which takes the clang/AVX-512/double workaround like every other
// masked select in these kernels.
template <class T, bool Local>
static void hb_scan_impl(HbScanJob<T>& job) {
    using vd = stdx::native_simd<T>;
    constexpr int W = (int)vd::size();
    const T NINF = -std::numeric_limits<T>::infinity();

    const int H = job.H, NC = job.ncols;
    const T go_a = job.go_a, ge_a = job.ge_a, go_b = job.go_b, ge_b = job.ge_b;
    const unsigned char* a = job.a;
    const unsigned char* b = job.b;
    const T* blk = job.blk;
    const int nalpha = job.nalpha;
    DpBufferT<T>& buf = *job.buf;

    // The empty local alignment (score 0 at the origin) is the default answer.
    job.best = T(0); job.best_i = 0; job.best_j = 0;
    if (H == 0 || NC == 0) return;      // no interior cell to score

    const int seg = (NC + W - 1) / W;
    const std::size_t sw = (std::size_t)seg * W;
    auto fit = [](typename DpBufferT<T>::TVec& v, std::size_t k) {
        if (v.size() < k) v.resize(k);
    };
    fit(buf.hfa, sw); fit(buf.hfb, sw); fit(buf.hfc, sw);
    fit(buf.hfd, sw); fit(buf.hfe, sw); fit(buf.hff, sw);
    fit(buf.hov, sw);
    fit(buf.hprof, (std::size_t)nalpha * sw);

    T* pM = buf.hfa.data(); T* pX = buf.hfb.data(); T* pY = buf.hfc.data();
    T* cM = buf.hfd.data(); T* cX = buf.hfe.data(); T* cY = buf.hff.data();
    T* ov = buf.hov.data();

    // Striped query profile over the block's column slice (padding -> -inf).  Serves this
    // block's sweep exactly as in hb_sweep_striped.
    for (int sym = 0; sym < nalpha; ++sym) {
        const T* row = blk + (std::size_t)sym * nalpha;
        T* dst = buf.hprof.data() + (std::size_t)sym * sw;
        for (int l = 0; l < W; ++l)
            for (int s = 0; s < seg; ++s) {
                const int c = l * seg + s + 1;
                dst[(std::size_t)s * W + l] =
                    (c <= NC) ? row[b[job.b_start + (c - 1) * job.b_step]] : NINF;
            }
    }

    // ── row 0, striped ────────────────────────────────────────────────────────
    // Local: M = 0 for every real column (a fresh alignment may start anywhere), X = Y =
    // -inf.  Global: the same Y-gap-open series hb_sweep_striped builds, so the reverse
    // pass computes the suffixes' global alignment.  Column 0 is held in the border scalars.
    if constexpr (Local) {
        for (int l = 0; l < W; ++l)
            for (int s = 0; s < seg; ++s) {
                const int c = l * seg + s + 1;
                const std::size_t k = (std::size_t)s * W + l;
                pM[k] = (c <= NC) ? T(0) : NINF;
                pX[k] = NINF;
                pY[k] = NINF;
            }
    } else {
        // scalar Y-series in column order, then stripe it (identical to hb_sweep row 0).
        // Reuse cY as a contiguous scratch for the series (overwritten before use below).
        T prevY = NINF;                       // r0Y[0] = -inf (column 0 border)
        for (int c = 1; c <= NC; ++c) {
            // r0M[c-1]/r0X[c-1] are -inf for c-1>=1; for c==1 the predecessors are the
            // column-0 border (M=0, X=-inf), giving open = (max(0,-inf) - go_a) - ge_a.
            const T pm = (c == 1) ? T(0) : NINF;
            const T px = NINF;
            const T open = (std::max(pm, px) - go_a) - ge_a;
            prevY = std::max(open, prevY - ge_a);
            const int cc = c;
            const std::size_t k = (std::size_t)((cc - 1) % seg) * W + (cc - 1) / seg;
            pM[k] = NINF; pX[k] = NINF; pY[k] = prevY;
        }
        // pad slots (c > NC) to -inf
        for (int l = 0; l < W; ++l)
            for (int s = 0; s < seg; ++s) {
                const int c = l * seg + s + 1;
                if (c > NC) { const std::size_t k = (std::size_t)s * W + l; pM[k] = NINF; pX[k] = NINF; pY[k] = NINF; }
            }
    }
    // column-0 border of the previous row: M = 0 fresh start in both modes, X = Y = -inf.
    T bM = 0, bX = NINF, bY = NINF;

    const vd vgo_a(go_a), vge_a(ge_a), vgo_b(go_b), vge_b(ge_b);
    const vd vzero(T(0));
    const vd vlane([](int q) { return T(q); });   // [0,1,...,W-1] for per-lane column ids

    T gbest = T(0); int gi = 0, gj = 0;

    for (int t = 0; t < H; ++t) {
        const T* sub = buf.hprof.data() +
                       (std::size_t)a[job.a_start + t * job.a_step] * sw;

        // Column 0 of this row.  Local: M = 0 (fresh start).  Global: M = -inf.  X extends
        // downward from the previous row's column 0 in both modes; Y cannot occur (no
        // column to consume).
        const T nbM = Local ? T(0) : NINF;
        const T nbX = std::max(std::max((bM - go_b) - ge_b, bX - ge_b), (bY - go_b) - ge_b);
        const T nbY = NINF;
        const T nbOpen = (std::max(nbM, nbX) - go_a) - ge_a;

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
            const vd vxv = stdx::max(stdx::max((uM - vgo_b) - vge_b, uX - vge_b),
                                     (uY - vgo_b) - vge_b);
            vxv.copy_to(cX + (std::size_t)s * W, stdx::element_aligned);
            ((stdx::max(vmv, vxv) - vgo_a) - vge_a)
                .copy_to(ov + (std::size_t)s * W, stdx::element_aligned);
        }

        // ── the carry: VY, same-lane striped chain, then lazy-F (exact) ──────────
        vd prev([&](int q) { return q == 0 ? bY : NINF; });
        for (int s = 0; s < seg; ++s) {
            vd O;
            if (s == 0) {
                vd lo; lo.copy_from(ov + (std::size_t)(seg - 1) * W, stdx::element_aligned);
                O = vd([&](int q) { return q == 0 ? nbOpen : lo[q - 1]; });
            } else {
                O.copy_from(ov + (std::size_t)(s - 1) * W, stdx::element_aligned);
            }
            const vd v = stdx::max(O, prev - vge_a);
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
#if defined(__clang__) && defined(__AVX512F__) && !defined(NWGRAD_STD_SIMD_AVX512_MASK_OK)
                if constexpr (std::is_same_v<T, double>) {
                    alignas(64) double fa[8], va[8];
                    F.copy_to(fa, stdx::element_aligned);
                    v.copy_to(va, stdx::element_aligned);
                    if (_mm512_cmp_pd_mask(_mm512_load_pd(fa), _mm512_load_pd(va), _CMP_GT_OQ) == 0) break;
                } else { if (!stdx::any_of(F > v)) break; }
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

        // ── per-row argmax over the row's cells, leftmost-column per lane ────────
        vd vbest(NINF), vbcol(T(0));
        for (int s = 0; s < seg; ++s) {
            vd mm; mm.copy_from(cM + (std::size_t)s * W, stdx::element_aligned);
            vd xx; xx.copy_from(cX + (std::size_t)s * W, stdx::element_aligned);
            vd yy; yy.copy_from(cY + (std::size_t)s * W, stdx::element_aligned);
            const vd here = stdx::max(stdx::max(mm, xx), yy);
            const vd colv = vlane * T(seg) + T(s + 1);     // column l*seg + s + 1, per lane
#if defined(__clang__) && defined(__AVX512F__) && !defined(NWGRAD_STD_SIMD_AVX512_MASK_OK)
            if constexpr (std::is_same_v<T, double>) {
                vbcol = avx512d_blend(avx512d_gt(here, vbest), vbcol, colv);
            } else {
                stdx::where(here > vbest, vbcol) = colv;
            }
#else
            stdx::where(here > vbest, vbcol) = colv;
#endif
            vbest = stdx::max(vbest, here);
        }
        // horizontal reduce: leftmost lane wins ties (lane order = column order)
        alignas(sizeof(T) * W) T rb[W], cb[W];
        vbest.copy_to(rb, stdx::element_aligned);
        vbcol.copy_to(cb, stdx::element_aligned);
        T rowbest = NINF; T rowcol = 0;
        for (int l = 0; l < W; ++l)
            if (rb[l] > rowbest) { rowbest = rb[l]; rowcol = cb[l]; }
        // global update, strict > so the topmost row and (via rowcol) leftmost column win
        if (rowbest > gbest) { gbest = rowbest; gi = t + 1; gj = (int)rowcol; }

        std::swap(pM, cM); std::swap(pX, cX); std::swap(pY, cY);
        bM = nbM; bX = nbX; bY = nbY;
    }

    job.best = gbest; job.best_i = gi; job.best_j = gj;
}

static void hb_scan_entry_d(HbScanJob<double>& j) {
    if (j.local) hb_scan_impl<double, true>(j); else hb_scan_impl<double, false>(j);
}
static void hb_scan_entry_f(HbScanJob<float>& j) {
    if (j.local) hb_scan_impl<float, true>(j); else hb_scan_impl<float, false>(j);
}

// ── Hirschberg base case: striped fill that RECORDS direction bytes ───────────
//
// This is striped_affine_full_ptr (Global) restricted to a sub-rectangle with an affine
// boundary seed.  It records the whole block's direction tables (3 B/cell, striped, in
// buf.hbD) so the aligner walks the path back; that is what vectorizes the base case,
// which began life as the scalar fill in aligner.hpp::hb_base.  Bit-exact with that
// scalar fill by construction — same recurrence, same M>X>Y pick, same left-association,
// same exact lazy-F carry — which the pointer kernel already proves for the full problem.
//
// Two things differ from the full pointer kernel and both come from the sweep:
//   * column 0 is CARRIED, not a closed form — the border scalars bM/bX/bY roll down,
//     because a sub-block's left edge is wherever the parent split it, not the origin;
//   * the origin is seeded by in_x (M=0 for a fresh arrival, X=0 to continue a run).
template <class T>
static void hb_base_striped(HbBaseJob<T>& job) {
    using vd = stdx::native_simd<T>;
    const int W = (int)vd::size();
    const T NINF = -std::numeric_limits<T>::infinity();

    const int H = job.H, NC = job.ncols;
    const T go_a = job.go_a, ge_a = job.ge_a, go_b = job.go_b, ge_b = job.ge_b;
    const unsigned char* a = job.a;
    const unsigned char* b = job.b;
    const T* blk = job.blk;
    const int nalpha = job.nalpha;
    DpBufferT<T>& buf = *job.buf;

    const int seg = (NC + W - 1) / W;
    job.seg = seg;
    job.width = W;
    const std::size_t sw = (std::size_t)seg * W;
    const std::size_t rowsz = (std::size_t)(seg + 1) * W;
    const std::size_t off   = (std::size_t)W;
    const std::size_t plane = (std::size_t)(H + 1) * rowsz;

    auto fit = [](typename DpBufferT<T>::TVec& v, std::size_t k) { if (v.size() < k) v.resize(k); };
    if (buf.hbD.size() < 3 * plane) buf.hbD.resize(3 * plane);
    fit(buf.hprof, (std::size_t)nalpha * sw);
    fit(buf.hfa, rowsz); fit(buf.hfb, rowsz); fit(buf.hfc, rowsz);
    fit(buf.hfd, rowsz); fit(buf.hfe, rowsz); fit(buf.hff, rowsz);
    fit(buf.hov, 2 * sw);
    fit(buf.hsX, sw);                                   // y-code scratch

    auto scol = [seg](int c) -> std::size_t {
        return (std::size_t)((c - 1) % seg) * W + (c - 1) / seg;
    };

    // Striped query profile over the block's column slice.
    for (int sym = 0; sym < nalpha; ++sym) {
        const T* row = blk + (std::size_t)sym * nalpha;
        T* dst = buf.hprof.data() + (std::size_t)sym * sw;
        for (int l = 0; l < W; ++l)
            for (int s = 0; s < seg; ++s) {
                const int c = l * seg + s + 1;
                dst[(std::size_t)s * W + l] = (c <= NC) ? row[b[job.b_start + (c - 1) * job.b_step]] : NINF;
            }
    }

    unsigned char* D0 = buf.hbD.data();
    unsigned char* D1 = D0 + plane;
    unsigned char* D2 = D1 + plane;
    T* ov = buf.hov.data();
    T* ok = buf.hov.data() + sw;
    T* yc = buf.hsX.data();

    T* pM = buf.hfa.data(); T* pX = buf.hfb.data(); T* pY = buf.hfc.data();
    T* cM = buf.hfd.data(); T* cX = buf.hfe.data(); T* cY = buf.hff.data();

    // ── row 0: origin seed (in_x), Y series rightward ────────────────────────
    pM[0] = job.in_x ? NINF : T(0);
    pX[0] = job.in_x ? T(0) : NINF;
    pY[0] = NINF;
    D0[0] = 3; D1[0] = 3; D2[0] = 3;
    {
        // scalar row-0 Y series (serial, O(NC), once) — kept scalar to match hb_base bit
        // for bit; then stripe it into pY and record its codes.
        buf.hfd[0] = pM[0]; buf.hfe[0] = pX[0]; buf.hff[0] = pY[0];  // temp scalars via cM/cX/cY[0]
        T prevM = pM[0], prevX = pX[0], prevY = pY[0];
        for (int c = 1; c <= NC; ++c) {
            const T uy = prevM - go_a - ge_a, vy = prevX - go_a - ge_a, wy = prevY - ge_a;
            const T y = std::max({uy, vy, wy});
            const unsigned char code = (uy >= vy && uy >= wy) ? 0 : ((vy >= wy) ? 1 : 2);
            const std::size_t k = off + scol(c);
            pM[k] = NINF; pX[k] = NINF; pY[k] = y;
            D0[k] = 3; D1[k] = 3; D2[k] = code;
            prevM = NINF; prevX = NINF; prevY = y;
        }
    }
    // pad lanes past NC in row 0 with NINF so they never win a max
    for (int l = 0; l < W; ++l)
        for (int s = 0; s < seg; ++s) {
            const int c = l * seg + s + 1;
            if (c > NC) { const std::size_t k = (std::size_t)s * W + l; pM[off + k] = NINF; pX[off + k] = NINF; pY[off + k] = NINF; }
        }

    const vd vgo_a(go_a), vge_a(ge_a), vgo_b(go_b), vge_b(ge_b);
    const vd vzero(T(0)), vone(T(1)), vtwo(T(2)), vthree(T(3));
    T bM = pM[0], bX = pX[0], bY = pY[0];

    for (int r = 1; r <= H; ++r) {
        const T* sub = buf.hprof.data() + (std::size_t)a[job.a_start + (r - 1) * job.a_step] * sw;
        unsigned char* dM = D0 + (std::size_t)r * rowsz + off;
        unsigned char* dX = D1 + (std::size_t)r * rowsz + off;
        unsigned char* dY = D2 + (std::size_t)r * rowsz + off;

        // column 0 of this row: carried X gap down the left edge
        const T ax0 = (bM - go_b) - ge_b, bx0 = bX - ge_b, cx0 = (bY - go_b) - ge_b;
        const T nbM = NINF;
        const T nbX = std::max({ax0, bx0, cx0});
        const T nbY = NINF;
        const unsigned char cx0code = (ax0 >= bx0 && ax0 >= cx0) ? 0 : ((bx0 >= cx0) ? 1 : 2);
        cM[0] = nbM; cX[0] = nbX; cY[0] = nbY;
        D0[(std::size_t)r * rowsz] = 3;
        D1[(std::size_t)r * rowsz] = cx0code;
        D2[(std::size_t)r * rowsz] = 3;
        const T nbOpen = (std::max(nbM, nbX) - go_a) - ge_a;
        const T nbOpenK = (nbM >= nbX) ? T(0) : T(1);

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

            ((stdx::max(vmv, vxv) - vgo_a) - vge_a).copy_to(ov + (std::size_t)s * W, stdx::element_aligned);
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

        // ── VY carry: same-lane striped chain + exact lazy-F, codes tracked ──
        vd prev([&](int q) { return q == 0 ? bY : NINF; });
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
            vd ky = vtwo;
#if defined(__clang__) && defined(__AVX512F__) && !defined(NWGRAD_STD_SIMD_AVX512_MASK_OK)
            if constexpr (std::is_same_v<T, double>) {
                ky = avx512d_blend(avx512d_ge(O, ext), ky, OK);
            } else {
                stdx::where(O >= ext, ky) = OK;
            }
#else
            stdx::where(O >= ext, ky) = OK;
#endif
            v.copy_to(cY + off + (std::size_t)s * W, stdx::element_aligned);
            ky.copy_to(yc + (std::size_t)s * W, stdx::element_aligned);
            prev = v;
        }
        if (seg > 0) for (int rr = 0; rr < W; ++rr) {   // seg==0 (NC==0): nothing to correct
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
                    if (_mm512_cmp_pd_mask(_mm512_load_pd(fa), _mm512_load_pd(va), _CMP_GT_OQ) == 0) break;
                } else { if (!stdx::any_of(F > v)) break; }
#else
                if (!stdx::any_of(F > v)) break;
#endif
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

        bM = nbM; bX = nbX; bY = nbY;
        std::swap(pM, cM); std::swap(pX, cX); std::swap(pY, cY);
    }

    // Final cell (H, NC) scores, for the aligner to choose the walk-back start.
    if (NC == 0) { job.fM = bM; job.fX = bX; job.fY = bY; }
    else {
        const std::size_t fk = off + scol(NC);
        job.fM = pM[fk]; job.fX = pX[fk]; job.fY = pY[fk];   // p* is row H after the last swap
    }
}

static void hb_base_entry_d(HbBaseJob<double>& j) { hb_base_striped<double>(j); }
static void hb_base_entry_f(HbBaseJob<float>&  j) { hb_base_striped<float>(j); }
