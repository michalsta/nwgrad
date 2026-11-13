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
// Glocal (Local false): row 0 is a free start (M = 0 at every column, no Y series), column
// 0 Global's chain, and the score is the best cell of the last row in any state (the
// padding masked, as Local's maximum).
template <class T, bool Local, bool Fused, bool Tr = false, bool Sat = false, bool Glocal = false>
static void score_affine_striped(ScoreJob<T>& job) {
    using vd = stdx::native_simd<T>;
    const int W = (int)vd::size();
    const T NINF = nw_ninf_s<T, Sat>;

    const int m = job.m, n = job.n, nalpha = job.nalpha;
    T go_a = job.go_a, ge_a = job.ge_a, go_b = job.go_b, ge_b = job.ge_b;
    const TrackJob<T>& tr = job.tr;
    if constexpr (Tr) { go_a = tr.go_a[0]; ge_a = tr.ge_a[0]; }
    const unsigned char* a = job.a;
    const unsigned char* b = job.b;
    DpBufferT<T>& buf = *job.buf;
    [[maybe_unused]] IntRampRow<T> irr{buf.iramp};   // int_row_ramp's row (Global, int16 at 16 lanes)

    if (n == 0) {
        // No columns: Global ends in one gap run down column 0 (or is empty); Local is 0.
        // The fill's (m, 0) cell: VM = -inf (0 at m == 0), VX = -(go_b + m*ge_b), VY = -inf.
        if constexpr (Local) job.score = T(0);
        else if constexpr (Tr) job.score = (m == 0) ? T(0) : tr.bx0[m];
        else job.score = (m == 0) ? T(0) : static_cast<T>(-(go_b + static_cast<T>(m) * ge_b));
        return;
    }

    const int seg = (n + W - 1) / W;
    const std::size_t sw = (std::size_t)seg * W;
    // six rolling rows, the opens and Local's padding mask: one block (spread_rows,
    // dp_buffer.hpp)
    std::size_t rs; T* const rows = spread_rows<T>(buf.srows, sw, 8, rs);
    T* const opens = rows + 6 * rs;
    // Local: the padding slots (columns > n) must not reach the maximum.  With a negative
    // gap extend a gap run GAINS score along the padding, so they can exceed every real
    // cell (the fill's argmax scans real columns only).  Added, not selected: x + 0 = x
    // exactly, x + -inf = -inf, and no mask op (see kernels_impl.inl's AVX-512/clang note).
    const T* pad = nullptr;
    if constexpr (Local || Glocal) pad = striped_pad_into<T, Sat>(rows + 7 * rs, n, seg, W);
    auto padded = [&](const vd& v, int s) {
        vd p; p.copy_from(pad + (std::size_t)s * W, stdx::element_aligned);
        return vadd<Sat>(v, p);
    };

    // striped query profile, padding columns -inf (as striped_affine_full)
    striped_profile_build<T, Tr>(buf.sprof, job.blk, nalpha, b, n, seg, W, tr);

    T* pM = rows + 3 * rs; T* pX = rows + 4 * rs; T* pY = rows + 5 * rs;   // row i-1
    T* cM = rows;          T* cX = rows + rs;     T* cY = rows + 2 * rs;   // row i

    // row 0 (columns 1..n, striped); padding -inf, as the fill leaves it
    for (std::size_t k = 0; k < sw; ++k) { pM[k] = NINF; pX[k] = NINF; pY[k] = NINF; }
    for (int l = 0; l < W; ++l)
        for (int s = 0; s < seg; ++s) {
            const int j = l * seg + s + 1;
            if (j <= n) {
                const std::size_t k = (std::size_t)s * W + l;
                if constexpr (Local || Glocal) pM[k] = 0;
                else                           pY[k] = -(go_a + j * ge_a);
            }
        }

    vd vgo_a(go_a), vge_a(ge_a), vgo_b(go_b), vge_b(ge_b);
    vd vgoe_a = vgo_a + vge_a, vgoe_b = vgo_b + vge_b;   // open_pen (integer T)
    const vd vzero(static_cast<T>(0)), vninf(NINF);
    vd vbest = vzero;                       // Local: running per-lane max (best starts at 0)
    T bM = 0, bX = NINF, bY = NINF;         // column 0 of the previous row

    // Is any lane of a greater than b?  (The lazy-F early exit — the one mask op.)
    auto any_gt = [](const vd& x, const vd& y) -> bool {
#if defined(__clang__) && defined(__AVX512F__) && !defined(NWGRAD_STD_SIMD_AVX512_MASK_OK)
        if constexpr (sizeof(T) == 8) return avx512d_gt(x, y) != 0;
        else                                     return stdx::any_of(x > y);
#else
        return stdx::any_of(x > y);
#endif
    };

    for (int i = 1; i <= m; ++i) {
        if constexpr (Tr) {
            go_a = tr.go_a[i]; ge_a = tr.ge_a[i]; go_b = tr.go_b[i]; ge_b = tr.ge_b[i];
            vgo_a = vd(go_a); vge_a = vd(ge_a); vgo_b = vd(go_b); vge_b = vd(ge_b);
            vgoe_a = vgo_a + vge_a; vgoe_b = vgo_b + vge_b;
        }
        const T nbM = Local ? T(0) : NINF;
        T nbX;
        if constexpr (Tr) nbX = Local ? NINF : tr.bx0[i];
        else              nbX = Local ? NINF : -(go_b + i * ge_b);
        const T nbOpen = nw_sub(nw_sub(std::max(nbM, nbX), go_a), ge_a);
        const T* sub = striped_row_profile<T, Tr>(buf.sprof.data(), job.blk, nalpha, a, b, n, seg, W, i, tr);

        vd Olast = vninf;                   // Fused: O of the last segment, for lazy-F
        if constexpr (Fused) {
            vd Oin([&](int q) { return q == 0 ? nbOpen : NINF; });
            vd Yin = vninf;                 // column 0's VY is -inf; lanes 1.. by lazy-F
            // The diagonal predecessors: the previous iteration's up vectors, rotated in
            // registers (each previous-row vector loaded once, not twice); at s = 0 the last
            // segment, shifted one lane in.
            vd dM(NINF), dX(NINF), dY(NINF);
            if (seg > 0) {
                vd lM, lX, lY;
                lM.copy_from(pM + (std::size_t)(seg - 1) * W, stdx::element_aligned);
                lX.copy_from(pX + (std::size_t)(seg - 1) * W, stdx::element_aligned);
                lY.copy_from(pY + (std::size_t)(seg - 1) * W, stdx::element_aligned);
                dM = lane_shift_in(lM, bM);
                dX = lane_shift_in(lX, bX);
                dY = lane_shift_in(lY, bY);
            }
            for (int s = 0; s < seg; ++s) {
                vd sb; sb.copy_from(sub + (std::size_t)s * W, stdx::element_aligned);
                vd vmv = vadd<Sat>(stdx::max(stdx::max(dM, dX), dY), sb);
                if constexpr (Local) vmv = stdx::max(vmv, vzero);
                vmv.copy_to(cM + (std::size_t)s * W, stdx::element_aligned);

                vd uM, uX, uY;
                uM.copy_from(pM + (std::size_t)s * W, stdx::element_aligned);
                uX.copy_from(pX + (std::size_t)s * W, stdx::element_aligned);
                uY.copy_from(pY + (std::size_t)s * W, stdx::element_aligned);
                dM = uM; dX = uX; dY = uY;   // the next segment's diagonal
                vd vxv = stdx::max(stdx::max(open_pen<Sat>(uM, vgo_b, vge_b, vgoe_b), vsub<Sat>(uX, vge_b)), open_pen<Sat>(uY, vgo_b, vge_b, vgoe_b));
                vxv.copy_to(cX + (std::size_t)s * W, stdx::element_aligned);

                if constexpr (nw_is_int<T> && W >= 4) {
                    // the opens to ov (int_pmax_carry reads them one segment back)
                    (open_pen<Sat>(stdx::max(vmv, vxv), vgo_a, vge_a, vgoe_a)).copy_to(opens + (std::size_t)s * W, stdx::element_aligned);
                    if constexpr (Local) vbest = stdx::max(vbest, padded(stdx::max(vmv, vxv), s));
                } else {
                vd vyv = stdx::max(Oin, vsub<Sat>(Yin, vge_a));
                vyv.copy_to(cY + (std::size_t)s * W, stdx::element_aligned);
                if constexpr (Local) vbest = stdx::max(vbest, padded(stdx::max(stdx::max(vmv, vxv), vyv), s));
                Oin = open_pen<Sat>(stdx::max(vmv, vxv), vgo_a, vge_a, vgoe_a);
                Yin = vyv;
                }
            }
            Olast = Oin;
        } else {
            T* ov = opens;
            // The diagonal predecessors: the previous iteration's up vectors, rotated in
            // registers (each previous-row vector loaded once, not twice); at s = 0 the last
            // segment, shifted one lane in.
            vd dM(NINF), dX(NINF), dY(NINF);
            if (seg > 0) {
                vd lM, lX, lY;
                lM.copy_from(pM + (std::size_t)(seg - 1) * W, stdx::element_aligned);
                lX.copy_from(pX + (std::size_t)(seg - 1) * W, stdx::element_aligned);
                lY.copy_from(pY + (std::size_t)(seg - 1) * W, stdx::element_aligned);
                dM = lane_shift_in(lM, bM);
                dX = lane_shift_in(lX, bX);
                dY = lane_shift_in(lY, bY);
            }
            for (int s = 0; s < seg; ++s) {
                vd sb; sb.copy_from(sub + (std::size_t)s * W, stdx::element_aligned);
                vd vmv = vadd<Sat>(stdx::max(stdx::max(dM, dX), dY), sb);
                if constexpr (Local) vmv = stdx::max(vmv, vzero);
                vmv.copy_to(cM + (std::size_t)s * W, stdx::element_aligned);
                vd uM, uX, uY;
                uM.copy_from(pM + (std::size_t)s * W, stdx::element_aligned);
                uX.copy_from(pX + (std::size_t)s * W, stdx::element_aligned);
                uY.copy_from(pY + (std::size_t)s * W, stdx::element_aligned);
                dM = uM; dX = uX; dY = uY;   // the next segment's diagonal
                vd vxv = stdx::max(stdx::max(open_pen<Sat>(uM, vgo_b, vge_b, vgoe_b), vsub<Sat>(uX, vge_b)), open_pen<Sat>(uY, vgo_b, vge_b, vgoe_b));
                vxv.copy_to(cX + (std::size_t)s * W, stdx::element_aligned);
                if constexpr (Local) vbest = stdx::max(vbest, padded(stdx::max(vmv, vxv), s));
                if constexpr (nw_is_int<T> && W >= 4)   // the integer carry reads the opens; the floating one recomputes them
                    (open_pen<Sat>(stdx::max(vmv, vxv), vgo_a, vge_a, vgoe_a)).copy_to(ov + (std::size_t)s * W, stdx::element_aligned);
            }
            if constexpr (!(nw_is_int<T> && W >= 4)) {
            // The open into Y at segment q: what pass 1 stores in `ov` for the integer carry,
            // recomputed from the stored M and X with the same operations in the same order
            // (the same bits), one stream fewer per segment.
            auto open_at = [&](int q) __attribute__((always_inline)) {
                vd vm, vx;
                vm.copy_from(cM + (std::size_t)q * W, stdx::element_aligned);
                vx.copy_from(cX + (std::size_t)q * W, stdx::element_aligned);
                return vd(open_pen<Sat>(stdx::max(vm, vx), vgo_a, vge_a, vgoe_a));
            };
            vd prev([&](int q) { return q == 0 ? bY : NINF; });
            for (int s = 0; s < seg; ++s) {
                vd O;
                if (s == 0) {
                    const vd lo = open_at(seg - 1);
                    O = lane_shift_in(lo, nbOpen);
                } else {
                    O = open_at(s - 1);
                }
                vd v = stdx::max(O, vsub<Sat>(prev, vge_a));
                v.copy_to(cY + (std::size_t)s * W, stdx::element_aligned);
                if constexpr (Local) vbest = stdx::max(vbest, padded(v, s));
                prev = v;
            }
            }
        }

        // ── lazy-F: carry the gap-in-A run across the lane boundaries ──
        // Round 0 of the fused form also carries the OPEN from the previous lane's last
        // column (seeded -inf above); later rounds, and every round of the two-pass form
        // (whose carry pass already saw the opens), carry the extension only.
        // (Integer T: the exact prefix-max carry instead, then Local's max over Y.)
        if constexpr (nw_is_int<T> && W >= 4) {
            int_pmax_carry<T, W, int_row_ramp<T, W> && !Local, 2, Sat>(opens, nullptr, nbOpen, T(0), bY, ge_a,
                                 seg, cY, nullptr, nullptr, &irr);
            if constexpr (Local)
                for (int s = 0; s < seg; ++s) {
                    vd y; y.copy_from(cY + (std::size_t)s * W, stdx::element_aligned);
                    vbest = stdx::max(vbest, padded(y, s));
                }
        } else
        for (int r = 0; r < W; ++r) {
            vd last; last.copy_from(cY + (std::size_t)(seg - 1) * W, stdx::element_aligned);
            vd F = lane_shift_in(last, NINF);
            F = vsub<Sat>(F, vge_a);
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
                F = lazy_f_next<Sat>(F, v, vge_a);   // floating T: F - ge (kernels_impl.inl)
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
        job.score = best;
    } else if constexpr (Glocal) {
        // the best cell of the last row computed (row 0 when m == 0), any state
        vd vb = vninf;
        for (int s = 0; s < seg; ++s) {
            vd vm, vx, vy;
            vm.copy_from(pM + (std::size_t)s * W, stdx::element_aligned);
            vx.copy_from(pX + (std::size_t)s * W, stdx::element_aligned);
            vy.copy_from(pY + (std::size_t)s * W, stdx::element_aligned);
            vb = stdx::max(vb, padded(stdx::max(stdx::max(vm, vx), vy), s));
        }
        T best = std::max({bM, bX, bY});   // column 0 of the last row: an end too (A all gapped)
        for (int q = 0; q < W; ++q) best = std::max(best, static_cast<T>(vb[q]));
        job.score = best;
    } else {
        // (m, n) in the last row computed (row 0 when m == 0), striped
        const std::size_t k = (std::size_t)((n - 1) % seg) * W + (std::size_t)((n - 1) / seg);
        job.score = std::max({pM[k], pX[k], pY[k]});
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
// An integer T (from 4 lanes) takes Y by the exact prefix max, FUSED into the row pass:
// the opens never reach memory (the running prefix max of open(s-1) + ramp(s) is kept in
// a register and stored to Y's row; the per-lane ramp base l*seg*ge_a plus one add per
// segment replaces the ramp row), and after the row one pass applies the lanes' carry-in,
// subtracts the ramp and folds Y into D.  Two passes over the row, six rows of working set,
// where the separate int_pmax_carry took four passes and eight rows (opens and ramp
// scratch).  Measured 2026-10-24, score-only, int32 ns/cell, fused / separate: AVX2
// (nighthaven) 0.73-0.84 inside L1 and 0.5 at L1400-1600 (the separate form's L1 cliff:
// eight rows of 4 B cross 48 KB at ~L1500, six at ~L2000), 6 threads L1600 0.49; AVX-512
// (solace) 0.75-0.9 and 0.45 at L1600; NEON (spot) 0.70-0.79 at every length; sse4
// (skynet) 0.60-0.95; int64 (AVX2 / AVX-512) 0.56-0.82; int16 0.74-0.93.  Bit-identical:
// max is exact and associative, so Y is the separate carry's.  (A middle form, the ramp
// in registers but the opens row kept, moved the cliff only and lost 3-10 % on NEON.)
// U2 (integer T): the row pass two segments a trip (see the loop).  Since 2026-11-05 the
// prefix is not stored either: pass 2 re-forms Y from the D row (four rolling rows, not
// five), and Local masks the padding segments only (see the passes; AGENTS.md).
template <class T>
inline constexpr bool kIntCarryEarly = nw_is_int<T> && (int)stdx::native_simd<T>::size() >= 4;
// kMainForm: main's pass 2 (the stored prefix row) and main's Local mask (every segment,
// MOnly too) at every length, for the targets where the re-formed Y and the segment-limited
// mask measured slower (2026-11-06, score-only, branch / main): AArch64 every type (spot, M1
// NEON: int16 1.05-1.33, int32 not narrowed 1.10-1.21, short int16 1.18-1.27), int16 at 32
// lanes (solace AVX-512: Global 1.02-1.11, the rebuilt chain's latency) and int16 at 16 lanes
// (nighthaven AVX2, 2d0a158: L200 1.06-1.09, L300-L3000 1.05-1.11, cycles +7-15 % at FEWER
// instructions, 0.88-0.99).  Only x86 int16 at 8 lanes (sse2 / sse4, not a target) keeps the
// rebuild past kStoredPrefixSeg (skynet sse4 L200-L1000 0.82-0.94).
template <class T>
inline constexpr bool kMainForm =
#if defined(__aarch64__)
    true;
#else
    sizeof(T) == 2 && (int)stdx::native_simd<T>::size() >= 16;
#endif
// kMainDirect: the entry calls main's form itself, compiled as its own function inlined
// into the entry, as on main.  Through the SQ = false form (which tail-calls the stored-
// prefix instantiation) gcc inlined it into that function, and the combined body is
// the one nighthaven measured: its L200 cells run the stored-prefix form with main's
// operations, yet 6-9 % slower at 12 % fewer instructions -- the generated code, not the
// work.  Where kMainCopy, main's kernel itself (score_affine_striped_pos_main).  On
// AArch64 int32 only: the combined body measured FASTER than main for int16 there (spot
// int16 affine Global L100 0.82-0.86), so int16 keeps it, while int32 Global paid ~9 ns a
// row in it (spot, 2026-11-07: L1000 1.020 .. L4000 1.005, Local 1.000).
template <class T>
inline constexpr bool kMainDirect =
#if defined(__aarch64__)
    sizeof(T) == 4 && kMainForm<T> && kIntCarryEarly<T>;
#else
    kMainForm<T> && kIntCarryEarly<T>;
#endif
template <class T, bool Local, bool MOnly, bool Tr = false, bool Sat = false, bool U2 = false, bool Glocal = false, bool SQ = false>
static void score_affine_striped_pos(ScoreJob<T>& job) {
    using vd = stdx::native_simd<T>;
    const int W = (int)vd::size();
    const T NINF = nw_ninf_s<T, Sat>;

    const int m = job.m, n = job.n, nalpha = job.nalpha;
    T go_a = job.go_a, ge_a = job.ge_a, go_b = job.go_b, ge_b = job.ge_b;
    const TrackJob<T>& tr = job.tr;
    if constexpr (Tr) { go_a = tr.go_a[0]; ge_a = tr.ge_a[0]; }
    const unsigned char* a = job.a;
    const unsigned char* b = job.b;
    DpBufferT<T>& buf = *job.buf;

    if (n == 0) {
        if constexpr (Local) job.score = T(0);
        else if constexpr (Tr) job.score = (m == 0) ? T(0) : tr.bx0[m];
        else job.score = (m == 0) ? T(0) : static_cast<T>(-(go_b + static_cast<T>(m) * ge_b));
        return;
    }

    const int seg = (n + W - 1) / W;
    // SQ: the stored-prefix form (pass 1 stores the running prefix to a Y row, pass 2 reads
    // it back: main's form before 2026-11-05) for short rows.  The re-formed Y (below) puts a
    // dependent chain through pass 2 -- one sub + max per segment pair -- where the stored
    // form's pass 2 is independent per segment once the carry-in is known; at a few segments
    // a row that chain's latency is exposed and the saved row buys nothing (the rows sit in
    // L1 anyway).  Measured, int16 score-only, the re-formed form / main: skynet sse4 L100
    // (13 segments) Global 1.02, L200 (25) 0.93-0.94; nighthaven / oldsolace AVX2 L100-L200
    // (7-13 segments) 1.02-1.04.  So up to 16 segments a row the stored form runs.
    constexpr int kStoredPrefixSeg = 16;
    if constexpr (kIntCarryEarly<T> && !SQ) {
        if (kMainForm<T> || seg <= kStoredPrefixSeg) {
            score_affine_striped_pos<T, Local, MOnly, Tr, Sat, U2, Glocal, true>(job);
            return;
        }
    }
    const std::size_t sw = (std::size_t)seg * W;
    // the rolling rows and Local's padding mask: one block (spread_rows, dp_buffer.hpp).
    // An integer T (from 4 lanes) keeps no Y row: four rolling rows (D, X twice) and the
    // mask, where the lazy-F form keeps five (see the carry below).
    constexpr bool kIntCarry = kIntCarryEarly<T>;
    constexpr int kRolling = (kIntCarry && !SQ) ? 4 : 5;
    std::size_t rs; T* const rows = spread_rows<T>(buf.srows, sw, kRolling + 1, rs);
    const T* pad = nullptr;
    // MOnly needs no mask at an integer T either, though its padding profile is 0
    // (nw_pad) and a padding M is a real value: M(i, j > n) = max(D(i-1, j-1) + 0, 0),
    // and with every gap parameter >= 0 no D exceeds the best M at or before it (a gap
    // cell never exceeds the cell it extends from), real or padding alike -- padding
    // only follows real columns, never feeds them.  So the padding M never exceeds the
    // running best and the unmasked max is the same.  (The W < 4 form below always ran
    // it unmasked.)
    constexpr bool kMaskAll = kMainForm<T> && kIntCarryEarly<T>;   // main's mask (see kMainForm)
    if constexpr ((Local && (!MOnly || kMaskAll)) || Glocal) pad = striped_pad_into<T, Sat>(rows + kRolling * rs, n, seg, W);
    auto padded = [&](const vd& v, int s) {
        vd p; p.copy_from(pad + (std::size_t)s * W, stdx::element_aligned);
        return vadd<Sat>(v, p);
    };
    // The padding slots (column l*seg + s + 1 > n) are in segments s >= s0 only (lane W-1
    // reaches past n first; with seg >= W that is the last segment alone), so the row
    // passes add the mask from s0 on (a load and an add fewer a vector cell before it).
    // The mask's real slots are 0 and v + 0 == v at an integer T: the same maxima.
    const int s0 = kMaskAll ? 0 : std::max(0, n - (W - 1) * seg);
    auto mpad = [&]<bool Mk>(const vd& v, int s) __attribute__((always_inline)) {
        if constexpr (Mk) return padded(v, s);
        else              return v;
    };

    striped_profile_build<T, Tr>(buf.sprof, job.blk, nalpha, b, n, seg, W, tr);

    T* pD = rows + (kRolling - 2) * rs; T* pX = rows + (kRolling - 1) * rs;   // row i-1
    T* cD = rows;          T* cX = rows + rs;                              // row i
    [[maybe_unused]] T* cY = (kIntCarry && !SQ) ? nullptr : rows + 2 * rs;   // row i (lazy-F / SQ form)

    // row 0: D = max(M, X, Y) of the three-state row 0, X = -inf; padding -inf
    for (std::size_t k = 0; k < sw; ++k) { pD[k] = NINF; pX[k] = NINF; }
    for (int l = 0; l < W; ++l)
        for (int s = 0; s < seg; ++s) {
            const int j = l * seg + s + 1;
            if (j <= n) {
                const std::size_t k = (std::size_t)s * W + l;
                if constexpr (Local || Glocal) pD[k] = 0;
                else                           pD[k] = -(go_a + j * ge_a);
            }
        }

    vd vgo_a(go_a), vge_a(ge_a), vgo_b(go_b), vge_b(ge_b);
    vd vgoe_a = vgo_a + vge_a, vgoe_b = vgo_b + vge_b;   // open_pen (integer T)
    const vd vzero(static_cast<T>(0)), vninf(NINF);
    vd vbest = vzero;
    T bD = 0;                               // D of column 0, previous row (M(0,0) = 0)

    auto any_gt = [](const vd& x, const vd& y) -> bool {
#if defined(__clang__) && defined(__AVX512F__) && !defined(NWGRAD_STD_SIMD_AVX512_MASK_OK)
        if constexpr (sizeof(T) == 8) return avx512d_gt(x, y) != 0;
        else                                     return stdx::any_of(x > y);
#else
        return stdx::any_of(x > y);
#endif
    };

    for (int i = 1; i <= m; ++i) {
        if constexpr (Tr) {
            go_a = tr.go_a[i]; ge_a = tr.ge_a[i]; go_b = tr.go_b[i]; ge_b = tr.ge_b[i];
            vgo_a = vd(go_a); vge_a = vd(ge_a); vgo_b = vd(go_b); vge_b = vd(ge_b);
            vgoe_a = vgo_a + vge_a; vgoe_b = vgo_b + vge_b;
        }
        const T nbM = Local ? T(0) : NINF;
        T nbX;
        if constexpr (Tr) nbX = Local ? NINF : tr.bx0[i];
        else              nbX = Local ? NINF : -(go_b + i * ge_b);
        const T nbD = std::max(nbM, nbX);   // column 0's VY is -inf
        const T nbOpen = nw_sub(nw_sub(nbD, go_a), ge_a);
        const T* sub = striped_row_profile<T, Tr>(buf.sprof.data(), job.blk, nalpha, a, b, n, seg, W, i, tr);

        vd Oin([&](int q) { return q == 0 ? nbOpen : NINF; });
        vd Yin = vninf;
        // integer carry: running prefix max, the previous segment's opens, the ramp
        [[maybe_unused]] vd fP = vninf, fOprev = vninf, fR, fBase;
        if constexpr (nw_is_int<T> && W >= 4) {
            fBase = vd([&](int l) { return static_cast<T>(static_cast<T>(l * seg) * ge_a); });
            fR = fBase;
            // Segment 0 has no Q term; seeded so that its fOprev + fR is exactly -inf (the
            // lane arithmetic wraps), which leaves fP at -inf with no `s > 0` test in the
            // loop — gcc kept that test on every segment (sse4: 3 branches per segment).
            fOprev = vsub<Sat>(vninf, fBase);   // nw_sat: stays -inf; its Q is then -inf + base, a drifted sentinel below every real value
        }
        if constexpr (nw_is_int<T> && W >= 4 && U2) {
            // Two segments a trip: both segments' loads first, the two cells, then the two
            // carry steps.  The same operations in the same order per cell (bit-identical);
            // what changes is the instruction order gcc emits -- it schedules a loop body
            // as written and issued one segment's chain at a time, where clang unrolls the
            // rolled loop by two and interleaves the iterations itself.  The diagonal of
            // segment s is segment s-1's up vector, carried in a register (the rolled form
            // reloads it).
            auto cell = [&](const vd& dD, const vd& uD, const vd& uX, const vd& sb, int s,
                            vd& mx, vd& vmv) __attribute__((always_inline)) {
                vmv = vadd<Sat>(dD, sb);
                if constexpr (Local) vmv = stdx::max(vmv, vzero);
                const vd vxv = stdx::max(open_pen<Sat>(uD, vgo_b, vge_b, vgoe_b), vsub<Sat>(uX, vge_b));
                mx = stdx::max(vmv, vxv);
                vxv.copy_to(cX + (std::size_t)s * W, stdx::element_aligned);
                mx.copy_to(cD + (std::size_t)s * W, stdx::element_aligned);
            };
            auto carry = [&]<bool Mk>(const vd& mx, const vd& vmv, int s) __attribute__((always_inline)) {
                fP = stdx::max(fP, vadd<Sat>(fOprev, fR));
                if constexpr (SQ) nv_store(cY + (std::size_t)s * W, fP);
                fOprev = open_pen<Sat>(mx, vgo_a, vge_a, vgoe_a);
                fR += vge_a;
                if constexpr (Local) {
                    if constexpr (MOnly) vbest = stdx::max(vbest, kMaskAll ? padded(vmv, s) : vmv);   // no mask: see `pad`
                    else                 vbest = stdx::max(vbest, mpad.template operator()<Mk>(mx, s));
                }
            };
            vd dD;
            {
                vd lD; lD.copy_from(pD + (std::size_t)(seg - 1) * W, stdx::element_aligned);
                dD = lane_shift_in(lD, bD);
            }
            auto trip = [&]<bool Mk>(int s) __attribute__((always_inline)) {
                vd sb0, sb1, uD0, uX0, uD1, uX1;
                sb0.copy_from(sub + (std::size_t)s * W, stdx::element_aligned);
                sb1.copy_from(sub + (std::size_t)(s + 1) * W, stdx::element_aligned);
                uD0.copy_from(pD + (std::size_t)s * W, stdx::element_aligned);
                uX0.copy_from(pX + (std::size_t)s * W, stdx::element_aligned);
                uD1.copy_from(pD + (std::size_t)(s + 1) * W, stdx::element_aligned);
                uX1.copy_from(pX + (std::size_t)(s + 1) * W, stdx::element_aligned);
                vd mx0, vmv0, mx1, vmv1;
                cell(dD, uD0, uX0, sb0, s, mx0, vmv0);
                cell(uD0, uD1, uX1, sb1, s + 1, mx1, vmv1);
                carry.template operator()<Mk>(mx0, vmv0, s);
                carry.template operator()<Mk>(mx1, vmv1, s + 1);
                dD = uD1;
            };
            int s = 0;
            if constexpr (Local) {   // the trips before the padding, unmasked
                for (; s + 1 < s0; s += 2) trip.template operator()<false>(s);
            }
            for (; s + 1 < seg; s += 2) trip.template operator()<true>(s);
            if (s < seg) {
                vd sb0, uD0, uX0, mx0, vmv0;
                sb0.copy_from(sub + (std::size_t)s * W, stdx::element_aligned);
                uD0.copy_from(pD + (std::size_t)s * W, stdx::element_aligned);
                uX0.copy_from(pX + (std::size_t)s * W, stdx::element_aligned);
                cell(dD, uD0, uX0, sb0, s, mx0, vmv0);
                carry.template operator()<true>(mx0, vmv0, s);
            }
        } else {
        auto seg1 = [&]<bool Mk>(int s) __attribute__((always_inline)) {
            vd dD;
            if (s == 0) {
                vd lD; lD.copy_from(pD + (std::size_t)(seg - 1) * W, stdx::element_aligned);
                dD = lane_shift_in(lD, bD);
            } else {
                dD.copy_from(pD + (std::size_t)(s - 1) * W, stdx::element_aligned);
            }
            vd sb; sb.copy_from(sub + (std::size_t)s * W, stdx::element_aligned);
            vd vmv = vadd<Sat>(dD, sb);
            if constexpr (Local) vmv = stdx::max(vmv, vzero);
            vd uD, uX;
            uD.copy_from(pD + (std::size_t)s * W, stdx::element_aligned);
            uX.copy_from(pX + (std::size_t)s * W, stdx::element_aligned);
            const vd vxv = stdx::max(open_pen<Sat>(uD, vgo_b, vge_b, vgoe_b), vsub<Sat>(uX, vge_b));
            const vd mx = stdx::max(vmv, vxv);
            vxv.copy_to(cX + (std::size_t)s * W, stdx::element_aligned);
            if constexpr (nw_is_int<T> && W >= 4) {
                // D without Y for now (the carry raises it below).  Q(s) = open(s-1) +
                // ramp(s) into the running prefix max (s >= 1; segment 0's Q needs the
                // LAST segment's open, shifted a lane, and is folded in after the row).
                // Only the row's total is needed here (the lanes' carry-in): the second
                // pass forms the prefix again from the stored D.
                mx.copy_to(cD + (std::size_t)s * W, stdx::element_aligned);
                fP = stdx::max(fP, vadd<Sat>(fOprev, fR));
                if constexpr (SQ) nv_store(cY + (std::size_t)s * W, fP);
                fOprev = open_pen<Sat>(mx, vgo_a, vge_a, vgoe_a);
                fR += vge_a;
                if constexpr (Local) {
                    if constexpr (MOnly) vbest = stdx::max(vbest, kMaskAll ? padded(vmv, s) : vmv);   // no mask: see `pad`
                    else                 vbest = stdx::max(vbest, mpad.template operator()<Mk>(mx, s));
                }
            } else {
            const vd vyv = stdx::max(Oin, vsub<Sat>(Yin, vge_a));
            const vd dn = stdx::max(mx, vyv);
            vyv.copy_to(cY + (std::size_t)s * W, stdx::element_aligned);
            dn.copy_to(cD + (std::size_t)s * W, stdx::element_aligned);
            if constexpr (Local) {
                if constexpr (MOnly) vbest = stdx::max(vbest, vmv);
                else                 vbest = stdx::max(vbest, padded(dn, s));
            }
            Oin = open_pen<Sat>(mx, vgo_a, vge_a, vgoe_a);
            Yin = vyv;
            }
        };
        int s = 0;
        if constexpr (Local && !MOnly && nw_is_int<T> && W >= 4) {
            for (; s < s0; ++s) seg1.template operator()<false>(s);
        }
        for (; s < seg; ++s) seg1.template operator()<true>(s);
        }
        const vd Olast = Oin;

        if constexpr (nw_is_int<T> && W >= 4) {
            // segment 0's term, then the lane carry-in, then ONE pass: Y = max(P', cin') - ramp
            // folded straight into D.  max is exact and associative, so Y is int_pmax_carry's.
            const vd Q0 = vadd<Sat>(lane_shift_in(fOprev, nbOpen), fBase);
            // int16: the in-register lane scan (int_lane_carry_in), fused score-only time / the
            // scalar scan 0.83-0.94 v3, 0.78-0.80 AVX-512, 0.83-0.87 NEON.  Wider types keep the
            // scalar scan: int32 0.91-1.03 across the three (v3 Global L300 1.027).
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
            // The second pass forms Y again from the D row as pass 1 stored it (max(M, X),
            // read before this pass overwrites it with max(D, Y)), with the lanes' carry-in
            // known.  Segment 0 as the prefix form has it, y(0) = max(vcin, Q(0)) - ramp(0)
            // (Q(0) is pass 1's seed, exactly -inf + base); then the serial step
            // y(s) = max(y(s-1) - ge_a, open(s-1)) -- max(P(s-1), open(s-1) + ramp(s)) -
            // ramp(s) with the ramp cancelled, ramp(s) = ramp(s-1) + ge_a -- two segments
            // per dependent step: y(s+1) = max(max(open(s), open(s-1) - ge_a), y(s-1) -
            // 2 ge_a), one subtraction and one max on the chain per two segments (the max
            // off it pinned against gcc's re-association, as in carry_pass).  In the plain
            // regime nothing wraps (the quarter rule bounds every operand), so the integer
            // identities hold exactly and Y is the stored-prefix form's bit for bit.
            // Saturating (int16 past the quarter rule): with ge_a >= 0 nothing clamps but
            // at the bottom, and a bottom clamp commutes with max and composes over the
            // subtractions, so again the same Y; with ge_a < 0 a saturated open lifted by
            // the ramp (the prefix form) or not (this one) differs, but only
            // sentinel-derived values differ, and those stay below every reachable value
            // (nw_int_bound_check), so every reachable cell and the score are the same.
            // Measured (skynet sse4, instructions per vector cell, Global / Local):
            // stored prefix 31.5 / 36.0, recomputed prefix (ramp kept) 34.0 / 39.0,
            // serial one segment a step 31.0 / 33.9 but 1.02-1.08x SLOWER (sub + max on
            // the chain, 2-cycle latencies on Piledriver), this form see AGENTS.md.
            if constexpr (SQ) {   // the stored-prefix form's pass 2, as main had it
                vd rq = fBase;
                auto stepq = [&]<bool Mk>(int s) __attribute__((always_inline)) {
                    const vd yq = vsub<Sat>(stdx::max(nv_load<vd>(cY + (std::size_t)s * W), vcin), rq);
                    vd d; d.copy_from(cD + (std::size_t)s * W, stdx::element_aligned);
                    d = stdx::max(d, yq);
                    d.copy_to(cD + (std::size_t)s * W, stdx::element_aligned);
                    if constexpr (Local && !MOnly) vbest = stdx::max(vbest, mpad.template operator()<Mk>(yq, s));
                    rq += vge_a;
                };
                int s = 0;
                if constexpr (Local && !MOnly) {
                    for (; s < s0; ++s) stepq.template operator()<false>(s);
                }
                for (; s < seg; ++s) stepq.template operator()<true>(s);
            } else {
            vd Op, y;
            const vd vge2_a = vge_a + vge_a;
            auto fin = [&]<bool Mk>(vd d, const vd& yv, int s) __attribute__((always_inline)) {
                d = stdx::max(d, yv);
                d.copy_to(cD + (std::size_t)s * W, stdx::element_aligned);
                if constexpr (Local && !MOnly) vbest = stdx::max(vbest, mpad.template operator()<Mk>(yv, s));
            };
            {
                vd d; d.copy_from(cD, stdx::element_aligned);
                y = vsub<Sat>(stdx::max(vcin, vadd<Sat>(vsub<Sat>(vninf, fBase), fBase)), fBase);
                Op = open_pen<Sat>(d, vgo_a, vge_a, vgoe_a);
                if (s0 == 0) fin.template operator()<true>(d, y, 0);
                else         fin.template operator()<false>(d, y, 0);
            }
            auto trip2 = [&]<bool Mk>(int s) __attribute__((always_inline)) {
                vd d0, d1;
                d0.copy_from(cD + (std::size_t)s * W, stdx::element_aligned);
                d1.copy_from(cD + (std::size_t)(s + 1) * W, stdx::element_aligned);
                const vd O0 = open_pen<Sat>(d0, vgo_a, vge_a, vgoe_a);
                const vd O1 = open_pen<Sat>(d1, vgo_a, vge_a, vgoe_a);
                const vd y0 = stdx::max(vsub<Sat>(y, vge_a), Op);
                const vd y1 = stdx::max(carry_pin(stdx::max(O0, vsub<Sat>(Op, vge_a))), vsub<Sat>(y, vge2_a));   // chain
                fin.template operator()<Mk>(d0, y0, s);
                fin.template operator()<Mk>(d1, y1, s + 1);
                y = y1; Op = O1;
            };
            int s = 1;
            if constexpr (Local && !MOnly) {
                for (; s + 1 < s0; s += 2) trip2.template operator()<false>(s);
            }
            for (; s + 1 < seg; s += 2) trip2.template operator()<true>(s);
            if (s < seg) {
                vd d; d.copy_from(cD + (std::size_t)s * W, stdx::element_aligned);
                y = stdx::max(vsub<Sat>(y, vge_a), Op);
                fin.template operator()<true>(d, y, s);
            }
            }
        } else
        for (int r = 0; r < W; ++r) {
            vd last; last.copy_from(cY + (std::size_t)(seg - 1) * W, stdx::element_aligned);
            vd F = lane_shift_in(last, NINF);
            F = vsub<Sat>(F, vge_a);
            if (r == 0) {
                vd Os = lane_shift_in(Olast, NINF);
                F = stdx::max(Os, F);
            }
            bool changed = false;
            for (int s = 0; s < seg; ++s) {
                vd v; v.copy_from(cY + (std::size_t)s * W, stdx::element_aligned);
                if (!any_gt(F, v)) break;
                v = stdx::max(v, F);
                v.copy_to(cY + (std::size_t)s * W, stdx::element_aligned);
                vd d; d.copy_from(cD + (std::size_t)s * W, stdx::element_aligned);
                d = stdx::max(d, v);
                d.copy_to(cD + (std::size_t)s * W, stdx::element_aligned);
                if constexpr (Local && !MOnly) vbest = stdx::max(vbest, padded(v, s));
                F = vsub<Sat>(v, vge_a);
                changed = true;
            }
            if (!changed) break;
        }

        std::swap(pD, cD); std::swap(pX, cX);
        bD = nbD;
    }

    if constexpr (Local) {
        T best = 0;
        for (int q = 0; q < W; ++q) best = std::max(best, static_cast<T>(vbest[q]));
        job.score = best;
    } else if constexpr (Glocal) {
        vd vb = vninf;   // the best cell of the last row (D = every state), padding masked
        for (int s = 0; s < seg; ++s) {
            vd v; v.copy_from(pD + (std::size_t)s * W, stdx::element_aligned);
            vb = stdx::max(vb, padded(v, s));
        }
        T best = bD;   // column 0 of the last row: an end too (A all gapped)
        for (int q = 0; q < W; ++q) best = std::max(best, static_cast<T>(vb[q]));
        job.score = best;
    } else {
        const std::size_t k = (std::size_t)((n - 1) % seg) * W + (std::size_t)((n - 1) / seg);
        job.score = pD[k];
    }
}

// ── The D/X form at floating T, with the gap carry shortened ──────────────────────
// score_affine_striped_pos's recurrence (above) for float / double, three changes, all
// bit-identical: the rows keep A = max(M, X), X and Y (D = max(A, Y) formed where a row
// is read, so lazy-F raises Y alone); the main pass steps Y's carry two segments at a time;
// lazy-F carries F - ge instead of max(c, F) - ge.  Notes at each site; measurements in
// AGENTS.md ("The gap carry, shortened").  Floating T only: -inf - ge stays -inf, where an
// integer sentinel stepped by - ge per segment could wrap (the integer types take the exact
// prefix-max carry in score_affine_striped_pos instead).
// score_affine_striped_pos_main: main's integer score kernel (7c69934) verbatim, the form
// kMainDirect's targets run.  The stored-prefix instantiation of the re-formed kernel above
// (SQ) does main's operations, but written as lambdas around a second template parameter
// gcc compiles its row code differently: on AArch64 int32 Global a different once-per-row
// schedule (spill, reloads), ~9 ns a row more on spot (L1000 1.020, L3000 1.008, Local
// 1.000).  This copy compiles as main's does (object diff, lvl_neon int32: the entry and the
// kernels it calls instruction for instruction main's).
template <class T, bool Local, bool MOnly, bool Tr = false, bool Sat = false, bool U2 = false, bool Glocal = false>
static void score_affine_striped_pos_main(ScoreJob<T>& job) {
    using vd = stdx::native_simd<T>;
    const int W = (int)vd::size();
    const T NINF = nw_ninf_s<T, Sat>;

    const int m = job.m, n = job.n, nalpha = job.nalpha;
    T go_a = job.go_a, ge_a = job.ge_a, go_b = job.go_b, ge_b = job.ge_b;
    const TrackJob<T>& tr = job.tr;
    if constexpr (Tr) { go_a = tr.go_a[0]; ge_a = tr.ge_a[0]; }
    const unsigned char* a = job.a;
    const unsigned char* b = job.b;
    DpBufferT<T>& buf = *job.buf;

    if (n == 0) {
        if constexpr (Local) job.score = T(0);
        else if constexpr (Tr) job.score = (m == 0) ? T(0) : tr.bx0[m];
        else job.score = (m == 0) ? T(0) : static_cast<T>(-(go_b + static_cast<T>(m) * ge_b));
        return;
    }

    const int seg = (n + W - 1) / W;
    const std::size_t sw = (std::size_t)seg * W;
    // five rolling rows and Local's padding mask: one block (spread_rows, dp_buffer.hpp)
    std::size_t rs; T* const rows = spread_rows<T>(buf.srows, sw, 6, rs);
    const T* pad = nullptr;
    // (MOnly's M padding is clamped to exactly 0 for floating T — the padding profile is
    // -inf — but an integer T pads the profile with 0 (nw_pad), so its padding M is a
    // real value and needs the mask too.)
    if constexpr ((Local && (!MOnly || nw_is_int<T>)) || Glocal) pad = striped_pad_into<T, Sat>(rows + 5 * rs, n, seg, W);
    auto padded = [&](const vd& v, int s) {
        vd p; p.copy_from(pad + (std::size_t)s * W, stdx::element_aligned);
        return vadd<Sat>(v, p);
    };

    striped_profile_build<T, Tr>(buf.sprof, job.blk, nalpha, b, n, seg, W, tr);

    T* pD = rows + 3 * rs; T* pX = rows + 4 * rs;                          // row i-1
    T* cD = rows;          T* cX = rows + rs;     T* cY = rows + 2 * rs;   // row i

    // row 0: D = max(M, X, Y) of the three-state row 0, X = -inf; padding -inf
    for (std::size_t k = 0; k < sw; ++k) { pD[k] = NINF; pX[k] = NINF; }
    for (int l = 0; l < W; ++l)
        for (int s = 0; s < seg; ++s) {
            const int j = l * seg + s + 1;
            if (j <= n) {
                const std::size_t k = (std::size_t)s * W + l;
                if constexpr (Local || Glocal) pD[k] = 0;
                else                           pD[k] = -(go_a + j * ge_a);
            }
        }

    vd vgo_a(go_a), vge_a(ge_a), vgo_b(go_b), vge_b(ge_b);
    vd vgoe_a = vgo_a + vge_a, vgoe_b = vgo_b + vge_b;   // open_pen (integer T)
    const vd vzero(static_cast<T>(0)), vninf(NINF);
    vd vbest = vzero;
    T bD = 0;                               // D of column 0, previous row (M(0,0) = 0)

    auto any_gt = [](const vd& x, const vd& y) -> bool {
#if defined(__clang__) && defined(__AVX512F__) && !defined(NWGRAD_STD_SIMD_AVX512_MASK_OK)
        if constexpr (sizeof(T) == 8) return avx512d_gt(x, y) != 0;
        else                                     return stdx::any_of(x > y);
#else
        return stdx::any_of(x > y);
#endif
    };

    for (int i = 1; i <= m; ++i) {
        if constexpr (Tr) {
            go_a = tr.go_a[i]; ge_a = tr.ge_a[i]; go_b = tr.go_b[i]; ge_b = tr.ge_b[i];
            vgo_a = vd(go_a); vge_a = vd(ge_a); vgo_b = vd(go_b); vge_b = vd(ge_b);
            vgoe_a = vgo_a + vge_a; vgoe_b = vgo_b + vge_b;
        }
        const T nbM = Local ? T(0) : NINF;
        T nbX;
        if constexpr (Tr) nbX = Local ? NINF : tr.bx0[i];
        else              nbX = Local ? NINF : -(go_b + i * ge_b);
        const T nbD = std::max(nbM, nbX);   // column 0's VY is -inf
        const T nbOpen = nw_sub(nw_sub(nbD, go_a), ge_a);
        const T* sub = striped_row_profile<T, Tr>(buf.sprof.data(), job.blk, nalpha, a, b, n, seg, W, i, tr);

        vd Oin([&](int q) { return q == 0 ? nbOpen : NINF; });
        vd Yin = vninf;
        // integer carry: running prefix max, the previous segment's opens, the ramp
        [[maybe_unused]] vd fP = vninf, fOprev = vninf, fR, fBase;
        if constexpr (nw_is_int<T> && W >= 4) {
            fBase = vd([&](int l) { return static_cast<T>(static_cast<T>(l * seg) * ge_a); });
            fR = fBase;
            // Segment 0 has no Q term; seeded so that its fOprev + fR is exactly -inf (the
            // lane arithmetic wraps), which leaves fP at -inf with no `s > 0` test in the
            // loop — gcc kept that test on every segment (sse4: 3 branches per segment).
            fOprev = vsub<Sat>(vninf, fBase);   // nw_sat: stays -inf; its Q is then -inf + base, a drifted sentinel below every real value
        }
        if constexpr (nw_is_int<T> && W >= 4 && U2) {
            // Two segments a trip: both segments' loads first, the two cells, then the two
            // carry steps.  The same operations in the same order per cell (bit-identical);
            // what changes is the instruction order gcc emits -- it schedules a loop body
            // as written and issued one segment's chain at a time, where clang unrolls the
            // rolled loop by two and interleaves the iterations itself.  The diagonal of
            // segment s is segment s-1's up vector, carried in a register (the rolled form
            // reloads it).
            auto cell = [&](const vd& dD, const vd& uD, const vd& uX, const vd& sb, int s,
                            vd& mx, vd& vmv) __attribute__((always_inline)) {
                vmv = vadd<Sat>(dD, sb);
                if constexpr (Local) vmv = stdx::max(vmv, vzero);
                const vd vxv = stdx::max(open_pen<Sat>(uD, vgo_b, vge_b, vgoe_b), vsub<Sat>(uX, vge_b));
                mx = stdx::max(vmv, vxv);
                vxv.copy_to(cX + (std::size_t)s * W, stdx::element_aligned);
                mx.copy_to(cD + (std::size_t)s * W, stdx::element_aligned);
            };
            auto carry = [&](const vd& mx, const vd& vmv, int s) __attribute__((always_inline)) {
                fP = stdx::max(fP, vadd<Sat>(fOprev, fR));
                nv_store(cY + (std::size_t)s * W, fP);
                fOprev = open_pen<Sat>(mx, vgo_a, vge_a, vgoe_a);
                fR += vge_a;
                if constexpr (Local) {
                    if constexpr (MOnly) vbest = stdx::max(vbest, padded(vmv, s));
                    else                 vbest = stdx::max(vbest, padded(mx, s));
                }
            };
            vd dD;
            {
                vd lD; lD.copy_from(pD + (std::size_t)(seg - 1) * W, stdx::element_aligned);
                dD = lane_shift_in(lD, bD);
            }
            int s = 0;
            for (; s + 1 < seg; s += 2) {
                vd sb0, sb1, uD0, uX0, uD1, uX1;
                sb0.copy_from(sub + (std::size_t)s * W, stdx::element_aligned);
                sb1.copy_from(sub + (std::size_t)(s + 1) * W, stdx::element_aligned);
                uD0.copy_from(pD + (std::size_t)s * W, stdx::element_aligned);
                uX0.copy_from(pX + (std::size_t)s * W, stdx::element_aligned);
                uD1.copy_from(pD + (std::size_t)(s + 1) * W, stdx::element_aligned);
                uX1.copy_from(pX + (std::size_t)(s + 1) * W, stdx::element_aligned);
                vd mx0, vmv0, mx1, vmv1;
                cell(dD, uD0, uX0, sb0, s, mx0, vmv0);
                cell(uD0, uD1, uX1, sb1, s + 1, mx1, vmv1);
                carry(mx0, vmv0, s);
                carry(mx1, vmv1, s + 1);
                dD = uD1;
            }
            if (s < seg) {
                vd sb0, uD0, uX0, mx0, vmv0;
                sb0.copy_from(sub + (std::size_t)s * W, stdx::element_aligned);
                uD0.copy_from(pD + (std::size_t)s * W, stdx::element_aligned);
                uX0.copy_from(pX + (std::size_t)s * W, stdx::element_aligned);
                cell(dD, uD0, uX0, sb0, s, mx0, vmv0);
                carry(mx0, vmv0, s);
            }
        } else
        for (int s = 0; s < seg; ++s) {
            vd dD;
            if (s == 0) {
                vd lD; lD.copy_from(pD + (std::size_t)(seg - 1) * W, stdx::element_aligned);
                dD = lane_shift_in(lD, bD);
            } else {
                dD.copy_from(pD + (std::size_t)(s - 1) * W, stdx::element_aligned);
            }
            vd sb; sb.copy_from(sub + (std::size_t)s * W, stdx::element_aligned);
            vd vmv = vadd<Sat>(dD, sb);
            if constexpr (Local) vmv = stdx::max(vmv, vzero);
            vd uD, uX;
            uD.copy_from(pD + (std::size_t)s * W, stdx::element_aligned);
            uX.copy_from(pX + (std::size_t)s * W, stdx::element_aligned);
            const vd vxv = stdx::max(open_pen<Sat>(uD, vgo_b, vge_b, vgoe_b), vsub<Sat>(uX, vge_b));
            const vd mx = stdx::max(vmv, vxv);
            vxv.copy_to(cX + (std::size_t)s * W, stdx::element_aligned);
            if constexpr (nw_is_int<T> && W >= 4) {
                // D without Y for now (the carry raises it below).  Q(s) = open(s-1) +
                // ramp(s) into the running prefix max (s >= 1; segment 0's Q needs the
                // LAST segment's open, shifted a lane, and is folded in after the row)
                mx.copy_to(cD + (std::size_t)s * W, stdx::element_aligned);
                fP = stdx::max(fP, vadd<Sat>(fOprev, fR));
                nv_store(cY + (std::size_t)s * W, fP);
                fOprev = open_pen<Sat>(mx, vgo_a, vge_a, vgoe_a);
                fR += vge_a;
                if constexpr (Local) {
                    if constexpr (MOnly) vbest = stdx::max(vbest, padded(vmv, s));
                    else                 vbest = stdx::max(vbest, padded(mx, s));
                }
            } else {
            const vd vyv = stdx::max(Oin, vsub<Sat>(Yin, vge_a));
            const vd dn = stdx::max(mx, vyv);
            vyv.copy_to(cY + (std::size_t)s * W, stdx::element_aligned);
            dn.copy_to(cD + (std::size_t)s * W, stdx::element_aligned);
            if constexpr (Local) {
                if constexpr (MOnly) vbest = stdx::max(vbest, vmv);
                else                 vbest = stdx::max(vbest, padded(dn, s));
            }
            Oin = open_pen<Sat>(mx, vgo_a, vge_a, vgoe_a);
            Yin = vyv;
            }
        }
        const vd Olast = Oin;

        if constexpr (nw_is_int<T> && W >= 4) {
            // segment 0's term, then the lane carry-in, then ONE pass: Y = max(P', cin') - ramp
            // folded straight into D.  max is exact and associative, so Y is int_pmax_carry's.
            const vd Q0 = vadd<Sat>(lane_shift_in(fOprev, nbOpen), fBase);
            // int16: the in-register lane scan (int_lane_carry_in), fused score-only time / the
            // scalar scan 0.83-0.94 v3, 0.78-0.80 AVX-512, 0.83-0.87 NEON.  Wider types keep the
            // scalar scan: int32 0.91-1.03 across the three (v3 Global L300 1.027).
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
                const vd y = vsub<Sat>(stdx::max(nv_load<vd>(cY + (std::size_t)s * W), vcin), rq);
                vd d; d.copy_from(cD + (std::size_t)s * W, stdx::element_aligned);
                d = stdx::max(d, y);
                d.copy_to(cD + (std::size_t)s * W, stdx::element_aligned);
                if constexpr (Local && !MOnly) vbest = stdx::max(vbest, padded(y, s));
                rq += vge_a;
            }
        } else
        for (int r = 0; r < W; ++r) {
            vd last; last.copy_from(cY + (std::size_t)(seg - 1) * W, stdx::element_aligned);
            vd F = lane_shift_in(last, NINF);
            F = vsub<Sat>(F, vge_a);
            if (r == 0) {
                vd Os = lane_shift_in(Olast, NINF);
                F = stdx::max(Os, F);
            }
            bool changed = false;
            for (int s = 0; s < seg; ++s) {
                vd v; v.copy_from(cY + (std::size_t)s * W, stdx::element_aligned);
                if (!any_gt(F, v)) break;
                v = stdx::max(v, F);
                v.copy_to(cY + (std::size_t)s * W, stdx::element_aligned);
                vd d; d.copy_from(cD + (std::size_t)s * W, stdx::element_aligned);
                d = stdx::max(d, v);
                d.copy_to(cD + (std::size_t)s * W, stdx::element_aligned);
                if constexpr (Local && !MOnly) vbest = stdx::max(vbest, padded(v, s));
                F = vsub<Sat>(v, vge_a);
                changed = true;
            }
            if (!changed) break;
        }

        std::swap(pD, cD); std::swap(pX, cX);
        bD = nbD;
    }

    if constexpr (Local) {
        T best = 0;
        for (int q = 0; q < W; ++q) best = std::max(best, static_cast<T>(vbest[q]));
        job.score = best;
    } else if constexpr (Glocal) {
        vd vb = vninf;   // the best cell of the last row (D = every state), padding masked
        for (int s = 0; s < seg; ++s) {
            vd v; v.copy_from(pD + (std::size_t)s * W, stdx::element_aligned);
            vb = stdx::max(vb, padded(v, s));
        }
        T best = bD;   // column 0 of the last row: an end too (A all gapped)
        for (int q = 0; q < W; ++q) best = std::max(best, static_cast<T>(vb[q]));
        job.score = best;
    } else {
        const std::size_t k = (std::size_t)((n - 1) % seg) * W + (std::size_t)((n - 1) / seg);
        job.score = pD[k];
    }
}

template <class T, bool Local, bool MOnly, bool Tr = false, bool Glocal = false>
static void score_affine_striped_pos_fp(ScoreJob<T>& job) {
    using vd = stdx::native_simd<T>;
    const int W = (int)vd::size();
    const T NINF = nw_ninf<T>;

    const int m = job.m, n = job.n, nalpha = job.nalpha;
    T go_a = job.go_a, ge_a = job.ge_a, go_b = job.go_b, ge_b = job.ge_b;
    const TrackJob<T>& tr = job.tr;
    if constexpr (Tr) { go_a = tr.go_a[0]; ge_a = tr.ge_a[0]; }
    const unsigned char* a = job.a;
    const unsigned char* b = job.b;
    DpBufferT<T>& buf = *job.buf;

    if (n == 0) {
        if constexpr (Local) job.score = T(0);
        else if constexpr (Tr) job.score = (m == 0) ? T(0) : tr.bx0[m];
        else job.score = (m == 0) ? T(0) : static_cast<T>(-(go_b + static_cast<T>(m) * ge_b));
        return;
    }

    const int seg = (n + W - 1) / W;
    const std::size_t sw = (std::size_t)seg * W;
    // six rolling rows and Local's padding mask: one block (spread_rows, dp_buffer.hpp)
    std::size_t rs; T* const rows = spread_rows<T>(buf.srows, sw, 7, rs);
    const T* pad = nullptr;
    if constexpr ((Local && !MOnly) || Glocal) pad = striped_pad_into(rows + 6 * rs, n, seg, W);
    auto padded = [&](const vd& v, int s) {
        vd p; p.copy_from(pad + (std::size_t)s * W, stdx::element_aligned);
        return v + p;
    };

    striped_profile_build<T, Tr>(buf.sprof, job.blk, nalpha, b, n, seg, W, tr);

    // The rows keep A = max(M, X), X and Y; D = max(A, Y) is formed where a row is read
    // (max is exact, so that is D bit for bit).  Lazy-F then raises Y alone: one load and
    // one store per visit instead of two each.
    T* pA = rows + 3 * rs; T* pX = rows + 4 * rs; T* pY = rows + 5 * rs;   // row i-1
    T* cA = rows;          T* cX = rows + rs;     T* cY = rows + 2 * rs;   // row i

    // row 0 (three-state: Global M = X = -inf, Y = -(go + j*ge); Local M = 0); padding -inf
    for (std::size_t k = 0; k < sw; ++k) { pA[k] = NINF; pX[k] = NINF; pY[k] = NINF; }
    for (int l = 0; l < W; ++l)
        for (int s = 0; s < seg; ++s) {
            const int j = l * seg + s + 1;
            if (j <= n) {
                const std::size_t k = (std::size_t)s * W + l;
                if constexpr (Local || Glocal) pA[k] = 0;
                else                           pY[k] = -(go_a + j * ge_a);
            }
        }

    vd vgo_a(go_a), vge_a(ge_a), vgo_b(go_b), vge_b(ge_b);
    const vd vzero(static_cast<T>(0)), vninf(NINF);
    vd vbest = vzero;
    T bD = 0;                               // D of column 0, previous row (M(0,0) = 0)

    auto any_gt = [](const vd& x, const vd& y) -> bool {
#if defined(__clang__) && defined(__AVX512F__) && !defined(NWGRAD_STD_SIMD_AVX512_MASK_OK)
        if constexpr (sizeof(T) == 8) return avx512d_gt(x, y) != 0;
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
        job.score = best;
    } else if constexpr (Glocal) {
        vd vb(NINF);     // the best cell of the last row (A = max(M, X), Y), padding masked
        for (int s = 0; s < seg; ++s) {
            vd va, vy;
            va.copy_from(pA + (std::size_t)s * W, stdx::element_aligned);
            vy.copy_from(pY + (std::size_t)s * W, stdx::element_aligned);
            vb = stdx::max(vb, padded(stdx::max(va, vy), s));
        }
        T best = bD;   // column 0 of the last row: an end too (A all gapped)
        for (int q = 0; q < W; ++q) best = std::max(best, static_cast<T>(vb[q]));
        job.score = best;
    } else {
        const std::size_t k = (std::size_t)((n - 1) % seg) * W + (std::size_t)((n - 1) / seg);
        job.score = std::max(pA[k], pY[k]);
    }
}

// ── Linear gaps: one rolling row H ─────────────────────────────────────────────
// compute_score ran linear gaps through the D/X form above with zero opens (exact: see
// ScoreJob).  There the X state is dead -- X = max((D_up - 0) - ge_b, X_up - ge_b) is
// D_up - ge_b bit for bit, since D_up >= X_up and rounding is monotone -- and so is the
// distinction between D and max(M, X): this kernel keeps H = D alone.  Per cell
//   U = max(H(i-1, j-1) + s, H(i-1, j) - ge_b)     (Local: the diagonal clamped at 0 first)
//   H = max(U, H(i, j-1) - ge_a)
// the D/X kernel's values exactly: its Y(j) = max(O(j-1), Y(j-1) - ge_a) with O = mx - ge_a
// is max(mx, Y)(j-1) - ge_a = H(j-1) - ge_a (monotone rounding again), and its D is
// max(mx, Y) = H; the borders keep its expressions with go = 0 (job.go_a / go_b are 0 for a
// linear job) so even their zero signs are the same.  The previous row is one vector a
// segment (up, and in a register the next segment's diagonal), not two; the gap-in-B open,
// the X row and its loads and stores go.
//   Integer T from 4 lanes: score_linear_striped_int (the prefix-max carry, one pass a
// row).  Floating T (and integer T at fewer lanes: int64 at 128 bits), score_linear_
// striped_fp: rows H(i-1), H(i), the carry on H itself, two segments per dependent step
// as score_affine_striped_pos_fp's,
//   H(s+1) = max(max(U(s+1), U(s) - g), (H(s-1) - g) - g),
// and lazy-F on H (exit when no lane's F exceeds H: H >= Y, so no later than the D/X
// kernel's Y test; the values are the recurrence's unique solution either way).  Floating
// lazy-F carries F - ge (H(s+1) >= H(s) - ge holds after the main pass in every lane, the
// argument of score_affine_striped_pos_fp); integer T at 2 lanes keeps max(v, F) - ge, as
// a sentinel stepped by - ge per segment could wrap.
//   Local: the maximum over every cell; with both extends >= 0 (MOnly) over the clamped
// diagonal term alone, as the D/X kernel's MOnly (a gap cell never exceeds the cell it
// extends from).  The padding slots are masked wherever they can reach the maximum (a
// negative extend makes a run GAIN along them); MOnly's cannot (an integer profile pads
// with 0, not -inf, but then no padding value exceeds a real one: see the integer form).

// The integer form (from 4 lanes) of score_linear_striped: score_affine_striped_pos's exact
// prefix-max carry, with no second pass over the row.  The row pass stores U (the cell
// before the gap in A) and the running prefix max P of (U(s-1) - ge_a) + ramp(s); the row's
// H(s) = max(U(s), max(P(s), cin) - ramp(s)) -- the D/X kernel's pass 2, the same
// operations -- is formed where the NEXT row reads it as its up vector (and, in a
// register, as the diagonal of the segment after).  Both rows are overwritten in place:
// segment s of row i-1 is read before segment s of row i is stored.  So two rows (U, P) of
// working set where the D/X form keeps five, and one pass a row where it makes two: the
// carry-in, the ramp base and the ramp step of the row just finished ride along in
// registers (pcin, pbase, pge: row 0 is H = U, its P and carry-in -inf, its ramp 0).  The
// last segment's H (the next row's segment-0 diagonal) is formed at the end of the row
// from the registers, and the last row's H by one pass after the loop (the score reads it).
template <class T, bool Local, bool MOnly, bool Tr = false, bool Sat = false, bool Glocal = false>
static void score_linear_striped_int(ScoreJob<T>& job) {
    using vd = stdx::native_simd<T>;
    const int W = (int)vd::size();
    const T NINF = nw_ninf_s<T, Sat>;

    const int m = job.m, n = job.n, nalpha = job.nalpha;
    T go_a = job.go_a, ge_a = job.ge_a, go_b = job.go_b, ge_b = job.ge_b;   // go_a = go_b = 0
    const TrackJob<T>& tr = job.tr;
    if constexpr (Tr) { go_a = tr.go_a[0]; ge_a = tr.ge_a[0]; }
    const unsigned char* a = job.a;
    const unsigned char* b = job.b;
    DpBufferT<T>& buf = *job.buf;

    if (n == 0) {
        if constexpr (Local) job.score = T(0);
        else if constexpr (Tr) job.score = (m == 0) ? T(0) : tr.bx0[m];
        else job.score = (m == 0) ? T(0) : static_cast<T>(-(go_b + static_cast<T>(m) * ge_b));
        return;
    }

    const int seg = (n + W - 1) / W;
    const std::size_t sw = (std::size_t)seg * W;
    std::size_t rs; T* const rows = spread_rows<T>(buf.srows, sw, 3, rs);
    T* const rU = rows; T* const rP = rows + rs;
    // (MOnly needs no mask, even though an integer profile pads with 0: see the row pass)
    const T* pad = nullptr;
    if constexpr ((Local && !MOnly) || Glocal) pad = striped_pad_into<T, Sat>(rows + 2 * rs, n, seg, W);
    auto padded = [&](const vd& v, int s) NWGRAD_ALWAYS_INLINE_LAMBDA {
        return vadd<Sat>(v, nv_load<vd>(pad + (std::size_t)s * W));
    };

    striped_profile_build<T, Tr>(buf.sprof, job.blk, nalpha, b, n, seg, W, tr);

    // row 0 as U: Global -(go_a + j*ge_a), Local / Glocal 0; padding -inf; P -inf
    for (std::size_t k = 0; k < sw; ++k) { rU[k] = NINF; rP[k] = NINF; }
    for (int l = 0; l < W; ++l)
        for (int s = 0; s < seg; ++s) {
            const int j = l * seg + s + 1;
            if (j <= n) {
                const std::size_t k = (std::size_t)s * W + l;
                if constexpr (Local || Glocal) rU[k] = 0;
                else                           rU[k] = -(go_a + j * ge_a);
            }
        }

    vd vge_a(ge_a), vge_b(ge_b);
    const vd vzero(static_cast<T>(0)), vninf(NINF);
    vd vbest = vzero;
    T bH = 0;                                   // H of column 0, previous row
    // the previous row's carry-in, ramp base and step: max(P, -inf) - 0 = -inf <= U at row 0
    vd pcin = vninf, pbase = vzero, pge = vzero;
    vd hlast = nv_load<vd>(rU + (std::size_t)(seg - 1) * W);   // the previous row's last-segment H

    for (int i = 1; i <= m; ++i) {
        if constexpr (Tr) {
            ge_a = tr.ge_a[i]; ge_b = tr.ge_b[i];
            vge_a = vd(ge_a); vge_b = vd(ge_b);
        }
        T nbH;   // column 0: the D/X kernel's max(M, X) there (its Y is -inf), go_b = 0
        if constexpr (Local) nbH = T(0);
        else if constexpr (Tr) nbH = std::max(NINF, tr.bx0[i]);
        else nbH = std::max(NINF, static_cast<T>(-(go_b + i * ge_b)));
        const T nbOpen = nw_sub(nw_sub(nbH, go_a), ge_a);   // open_pen with go = 0
        const T* sub = striped_row_profile<T, Tr>(buf.sprof.data(), job.blk, nalpha, a, b, n, seg, W, i, tr);

        vd dH = lane_shift_in(hlast, bH);       // segment 0's diagonal
        vd prq = pbase;                         // the previous row's ramp at segment s
        const vd fBase([&](int l) { return static_cast<T>(static_cast<T>(l * seg) * ge_a); });
        // The prefix term of segment s is open(s-1) + ramp(s) = (U(s-1) - ge_a) + (fBase +
        // s*ge_a).  Plain regime: lane arithmetic wraps, so that is U(s-1) + (fBase + (s-1)*ge_a)
        // exactly, one subtraction fewer a segment (fR then runs one step behind and Uq holds
        // U(s-1), seeded so that segment 0's term is exactly -inf, as fOprev's seed below).
        // Saturating regime: the D/X kernel's form, as a clamped U - ge_a is not U - ge_a.
        vd fR = Sat ? fBase : vd(fBase - vge_a), fP = vninf;
        vd fOprev = vsub<Sat>(vninf, fBase);    // segment 0 has no Q term (see the D/X kernel)
        // (plain regime only: the saturating regime's sentinel is lowest() itself, where
        // - fR would overflow -- UB, caught by UBSan; the plain one is lowest()/2, and the
        // bound keeps -fR within its depth, as fOprev's seed)
        vd Uq = vninf;
        if constexpr (!Sat) Uq = vninf - fR;
        vd U = vninf;
        auto step = [&](int s) NWGRAD_ALWAYS_INLINE_LAMBDA {
            const std::size_t o = (std::size_t)s * W;
            const vd uH = stdx::max(nv_load<vd>(rU + o), vsub<Sat>(stdx::max(nv_load<vd>(rP + o), pcin), prq));
            prq += pge;
            if constexpr (Local && !MOnly) vbest = stdx::max(vbest, padded(uH, s));   // row i-1's H
            vd vmv = vadd<Sat>(dH, nv_load<vd>(sub + o));
            if constexpr (Local) vmv = stdx::max(vmv, vzero);
            // MOnly: unmasked.  The integer profile pads with 0, so a padding slot's diagonal
            // term is a real number, but with both extends >= 0 every padding value is at most
            // a real cell's or 0 (by induction over the cells: its diagonal + 0, its up - ge and
            // its left - ge are each at most a value already bounded so), and the maximum
            // starts at 0: the same maximum without the mask's load and add.
            if constexpr (Local && MOnly) vbest = stdx::max(vbest, vmv);
            dH = uH;
            U = stdx::max(vmv, vsub<Sat>(uH, vge_b));
            nv_store(rU + o, U);
            if constexpr (Sat) {
                fP = stdx::max(fP, vadd<Sat>(fOprev, fR));
                fOprev = vsub<Sat>(U, vge_a);
            } else {
                fP = stdx::max(fP, Uq + fR);
                Uq = U;
            }
            nv_store(rP + o, fP);
            fR += vge_a;
        };
        // two segments a trip (gcc does not unroll; the loop overhead is a sixth of the body)
        int s = 0;
        for (; s + 1 < seg; s += 2) { step(s); step(s + 1); }
        if (s < seg) step(s);
        if constexpr (!Sat) { fOprev = U - vge_a; fR += vge_a; }   // the saturating form's state
        const vd Q0 = vadd<Sat>(lane_shift_in(fOprev, nbOpen), fBase);
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
        // the last segment's H; its ramp fBase + (seg-1)*ge_a is fR - ge_a (wrapping, as += is)
        hlast = stdx::max(U, vsub<Sat>(stdx::max(fP, vcin), fR - vge_a));
        pcin = vcin; pbase = fBase; pge = vge_a;
        bH = nbH;
    }

    // the last row's H (row 0's when m == 0)
    {
        vd prq = pbase;
        for (int s = 0; s < seg; ++s) {
            const std::size_t o = (std::size_t)s * W;
            const vd h = stdx::max(nv_load<vd>(rU + o), vsub<Sat>(stdx::max(nv_load<vd>(rP + o), pcin), prq));
            nv_store(rU + o, h);
            if constexpr (Local && !MOnly) vbest = stdx::max(vbest, padded(h, s));
            prq += pge;
        }
    }

    if constexpr (Local) {
        T best = 0;
        for (int q = 0; q < W; ++q) best = std::max(best, static_cast<T>(vbest[q]));
        job.score = best;
    } else if constexpr (Glocal) {
        vd vb = vninf;   // the best cell of the last row, padding masked
        for (int s = 0; s < seg; ++s) vb = stdx::max(vb, padded(nv_load<vd>(rU + (std::size_t)s * W), s));
        T best = bH;     // column 0 of the last row: an end too (A all gapped)
        for (int q = 0; q < W; ++q) best = std::max(best, static_cast<T>(vb[q]));
        job.score = best;
    } else {
        const std::size_t k = (std::size_t)((n - 1) % seg) * W + (std::size_t)((n - 1) / seg);
        job.score = rU[k];
    }
}

template <class T, bool Local, bool MOnly, bool Tr = false, bool Sat = false, bool Glocal = false>
static void score_linear_striped_fp(ScoreJob<T>& job) {
    using vd = stdx::native_simd<T>;
    const int W = (int)vd::size();
    const T NINF = nw_ninf_s<T, Sat>;

    const int m = job.m, n = job.n, nalpha = job.nalpha;
    T go_a = job.go_a, ge_a = job.ge_a, go_b = job.go_b, ge_b = job.ge_b;   // go_a = go_b = 0
    const TrackJob<T>& tr = job.tr;
    if constexpr (Tr) { go_a = tr.go_a[0]; ge_a = tr.ge_a[0]; }
    const unsigned char* a = job.a;
    const unsigned char* b = job.b;
    DpBufferT<T>& buf = *job.buf;

    if (n == 0) {
        if constexpr (Local) job.score = T(0);
        else if constexpr (Tr) job.score = (m == 0) ? T(0) : tr.bx0[m];
        else job.score = (m == 0) ? T(0) : static_cast<T>(-(go_b + static_cast<T>(m) * ge_b));
        return;
    }

    const int seg = (n + W - 1) / W;
    const std::size_t sw = (std::size_t)seg * W;
    // H(i-1), H(i) and the padding mask: one block (spread_rows)
    std::size_t rs; T* const rows = spread_rows<T>(buf.srows, sw, 3, rs);
    constexpr bool NeedPad = (Local && !MOnly) || Glocal;   // (MOnly: see score_linear_striped_int)
    const T* pad = nullptr;
    if constexpr (NeedPad) pad = striped_pad_into<T, Sat>(rows + 2 * rs, n, seg, W);
    auto padded = [&](const vd& v, int s) NWGRAD_ALWAYS_INLINE_LAMBDA {
        vd p; p.copy_from(pad + (std::size_t)s * W, stdx::element_aligned);
        return vadd<Sat>(v, p);
    };

    striped_profile_build<T, Tr>(buf.sprof, job.blk, nalpha, b, n, seg, W, tr);

    T* pH = rows + rs; T* cH = rows;

    // row 0: Global -(go_a + j*ge_a) (the D/X kernel's D), Local / Glocal 0; padding -inf
    for (std::size_t k = 0; k < sw; ++k) pH[k] = NINF;
    for (int l = 0; l < W; ++l)
        for (int s = 0; s < seg; ++s) {
            const int j = l * seg + s + 1;
            if (j <= n) {
                const std::size_t k = (std::size_t)s * W + l;
                if constexpr (Local || Glocal) pH[k] = 0;
                else                           pH[k] = -(go_a + j * ge_a);
            }
        }

    vd vge_a(ge_a), vge_b(ge_b);
    const vd vzero(static_cast<T>(0)), vninf(NINF);
    vd vbest = vzero;
    T bH = 0;                               // H of column 0, previous row

    auto any_gt = [](const vd& x, const vd& y) -> bool {
#if defined(__clang__) && defined(__AVX512F__) && !defined(NWGRAD_STD_SIMD_AVX512_MASK_OK)
        if constexpr (sizeof(T) == 8) return avx512d_gt(x, y) != 0;
        else                                     return stdx::any_of(x > y);
#else
        return stdx::any_of(x > y);
#endif
    };
    auto ld = [](const T* p) NWGRAD_ALWAYS_INLINE_LAMBDA { vd v; v.copy_from(p, stdx::element_aligned); return v; };

    for (int i = 1; i <= m; ++i) {
        if constexpr (Tr) {
            ge_a = tr.ge_a[i]; ge_b = tr.ge_b[i];
            vge_a = vd(ge_a); vge_b = vd(ge_b);
        }
        // column 0: the D/X kernel's max(M, X) there (its Y is -inf), with go_b = 0
        T nbH;
        if constexpr (Local) nbH = T(0);
        else if constexpr (Tr) nbH = std::max(NINF, tr.bx0[i]);
        else nbH = std::max(NINF, static_cast<T>(-(go_b + i * ge_b)));
        const T* sub = striped_row_profile<T, Tr>(buf.sprof.data(), job.blk, nalpha, a, b, n, seg, W, i, tr);

        vd dH = lane_shift_in(ld(pH + (std::size_t)(seg - 1) * W), bH);   // segment 0's diagonal
        // One segment's carry-free part: U (and the clamped diagonal term, for MOnly).
        auto cell = [&](int s, vd& vmv) NWGRAD_ALWAYS_INLINE_LAMBDA {
            const std::size_t o = (std::size_t)s * W;
            const vd uH = ld(pH + o);
            vmv = vadd<Sat>(dH, ld(sub + o));
            if constexpr (Local) vmv = stdx::max(vmv, vzero);
            dH = uH;
            return stdx::max(vmv, vsub<Sat>(uH, vge_b));
        };

        auto fin = [&](int s, const vd& vmv, const vd& h) NWGRAD_ALWAYS_INLINE_LAMBDA {
            h.copy_to(cH + (std::size_t)s * W, stdx::element_aligned);
            if constexpr (Local) {
                if constexpr (MOnly) vbest = stdx::max(vbest, vmv);   // unmasked (padding M: 0, or bounded)
                else                 vbest = stdx::max(vbest, padded(h, s));
            }
        };
        // H of the column before, lane 0 column 0's; lanes 1.. by lazy-F.  (nbH - go_a)
        // - ge_a is the D/X kernel's open of column 0, nbH - ge_a with go_a = 0.
        vd Hin([&](int q) { return q == 0 ? nw_sub(nbH, go_a) : NINF; });
        int s = 0;
        if constexpr (std::is_floating_point_v<T>) {
            for (; s + 1 < seg; s += 2) {
                vd vm0, vm1;
                const vd U0 = cell(s, vm0);
                const vd U1 = cell(s + 1, vm1);
                const vd t = Hin - vge_a;
                const vd h0 = stdx::max(U0, t);
                const vd h1 = stdx::max(carry_pin(stdx::max(U1, U0 - vge_a)), t - vge_a);
                fin(s, vm0, h0);
                fin(s + 1, vm1, h1);
                Hin = h1;
            }
        }
        for (; s < seg; ++s) {
            vd vm0;
            const vd U0 = cell(s, vm0);
            const vd h0 = stdx::max(U0, vsub<Sat>(Hin, vge_a));
            fin(s, vm0, h0);
            Hin = h0;
        }

        // ── lazy-F on H ──
        for (int r = 0; r < W; ++r) {
            vd F = vsub<Sat>(lane_shift_in(ld(cH + (std::size_t)(seg - 1) * W), NINF), vge_a);
            bool changed = false;
            for (int s2 = 0; s2 < seg; ++s2) {
                vd v = ld(cH + (std::size_t)s2 * W);
                if (!any_gt(F, v)) break;
                v = stdx::max(v, F);
                v.copy_to(cH + (std::size_t)s2 * W, stdx::element_aligned);
                if constexpr (Local && !MOnly) vbest = stdx::max(vbest, padded(v, s2));
                if constexpr (std::is_floating_point_v<T>) F = F - vge_a;
                else                                       F = vsub<Sat>(v, vge_a);
                changed = true;
            }
            if (!changed) break;
        }

        std::swap(pH, cH);
        bH = nbH;
    }

    if constexpr (Local) {
        T best = 0;
        for (int q = 0; q < W; ++q) best = std::max(best, static_cast<T>(vbest[q]));
        job.score = best;
    } else if constexpr (Glocal) {
        vd vb = vninf;   // the best cell of the last row, padding masked
        for (int s = 0; s < seg; ++s) vb = stdx::max(vb, padded(ld(pH + (std::size_t)s * W), s));
        T best = bH;     // column 0 of the last row: an end too (A all gapped)
        for (int q = 0; q < W; ++q) best = std::max(best, static_cast<T>(vb[q]));
        job.score = best;
    } else {
        const std::size_t k = (std::size_t)((n - 1) % seg) * W + (std::size_t)((n - 1) / seg);
        job.score = pH[k];
    }
}
template <class T, bool Local, bool MOnly, bool Tr = false, bool Sat = false, bool Glocal = false>
static void score_linear_striped(ScoreJob<T>& job) {
    if constexpr (nw_is_int<T> && stdx::native_simd<T>::size() >= 4)
        score_linear_striped_int<T, Local, MOnly, Tr, Sat, Glocal>(job);
    else
        score_linear_striped_fp<T, Local, MOnly, Tr, Sat, Glocal>(job);
}

// The registered entry: Global or Local; job.variant 0 = the D/X form when both opens
// are >= 0 (else fused three-state; linear gaps, job.linear: score_linear_striped), 1 = two-pass, 2 = fused three-state (for measuring
// the forms in one binary; NWGRAD_SCORE_VARIANT=twopass / fused3).
// NWGRAD_SCORE_UNROLL=0 runs the rolled integer row pass (an A/B switch, one binary).
// Off on NEON by default: the M1 lost 25 % on int16 Global with it (spot, gcc 16, 2026-10-31);
// NWGRAD_SCORE_UNROLL=1 forces it on there.
static bool score_unroll2() noexcept {
    static const bool on = [] {
        const char* e = std::getenv("NWGRAD_SCORE_UNROLL");
#if defined(__aarch64__) || defined(_M_ARM64)
        return e && e[0] == '1';
#else
        return !(e && e[0] == '0');
#endif
    }();
    return on;
}
// kMainCopy: the kMainDirect targets that run main's kernel verbatim
// (score_affine_striped_pos_main): AArch64 int32 only (spot: its row code then compiles as
// main's).  On x86 kMainDirect calls the stored-prefix instantiation, main's form: the copy
// at 32 lanes (AVX-512) ran 27-322 fewer instructions a pair yet 1-2 % slower at protein
// L200-L300 (solace, 2026-11-07), and at 16 lanes (AVX2) it did not remove nighthaven's
// int16 Global L130-L170 residual.
template <class T>
inline constexpr bool kMainCopy =
#if defined(__aarch64__)
    kMainDirect<T>;
#else
    false;
#endif
// The integer positive-open kernel the entry runs: main's verbatim (kMainCopy), main's
// form as the stored-prefix instantiation (other kMainDirect targets), else the re-formed
// kernel (its stored-prefix form up to kStoredPrefixSeg segments).  On x86 spelled out at
// each call: through an always-inline wrapper gcc inlined a different set of forms into the
// avx2 int16 entry than with the direct calls (the objects are 2413307's this way).  On
// AArch64 through the wrapper, which is how the int32 direct call was measured (spot).
#if defined(__aarch64__) || defined(_M_ARM64)
template <class T, bool Local, bool MOnly, bool Tr, bool Sat, bool U2, bool Glocal>
[[gnu::always_inline]] static inline void score_pos_int(ScoreJob<T>& job) {
    if constexpr (kMainCopy<T>) score_affine_striped_pos_main<T, Local, MOnly, Tr, Sat, U2, Glocal>(job);
    else                        score_affine_striped_pos<T, Local, MOnly, Tr, Sat, U2, Glocal, kMainDirect<T>>(job);
}
#  define NWGRAD_SCORE_POS_INT(...) score_pos_int<__VA_ARGS__>(job)
#else
#  define NWGRAD_SCORE_POS_INT(...)                                                      \
    do {                                                                                  \
        if constexpr (kMainCopy<T>) score_affine_striped_pos_main<__VA_ARGS__>(job);      \
        else score_affine_striped_pos<__VA_ARGS__, kMainDirect<T>>(job);                  \
    } while (0)
#endif
template <class T, bool Tr, bool Sat = false>
static void score_entry_impl(ScoreJob<T>& job) {
    const bool glocal = job.local == 2;   // 0 Global, 1 Local, 2 Glocal (ScoreJob::local)
    if (job.variant == 1) {
        if (glocal)         score_affine_striped<T, false, false, Tr, Sat, true>(job);
        else if (job.local) score_affine_striped<T, true, false, Tr, Sat>(job);
        else                score_affine_striped<T, false, false, Tr, Sat>(job);
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
    if (job.variant == 0 && job.linear) {   // linear gaps (go = 0): one row H
        if (glocal) score_linear_striped<T, false, false, Tr, Sat, true>(job);
        else if (job.local) {
            if (pos_ext) score_linear_striped<T, true, true, Tr, Sat>(job);
            else         score_linear_striped<T, true, false, Tr, Sat>(job);
        } else {
            score_linear_striped<T, false, false, Tr, Sat>(job);
        }
        return;
    }
    if (job.variant == 0 && pos_open) {
        if constexpr (std::is_floating_point_v<T>) {
            if (glocal) score_affine_striped_pos_fp<T, false, false, Tr, true>(job);
            else if (job.local) {
                if (pos_ext) score_affine_striped_pos_fp<T, true, true, Tr>(job);
                else         score_affine_striped_pos_fp<T, true, false, Tr>(job);
            } else {
                score_affine_striped_pos_fp<T, false, false, Tr>(job);
            }
        } else if (score_unroll2()) {
            if (glocal) NWGRAD_SCORE_POS_INT(T, false, false, Tr, Sat, true, true);
            else if (job.local) {
                if (pos_ext) NWGRAD_SCORE_POS_INT(T, true, true, Tr, Sat, true, false);
                else         NWGRAD_SCORE_POS_INT(T, true, false, Tr, Sat, true, false);
            } else {
                NWGRAD_SCORE_POS_INT(T, false, false, Tr, Sat, true, false);
            }
        } else {
            if (glocal) NWGRAD_SCORE_POS_INT(T, false, false, Tr, Sat, false, true);
            else if (job.local) {
                if (pos_ext) NWGRAD_SCORE_POS_INT(T, true, true, Tr, Sat, false, false);
                else         NWGRAD_SCORE_POS_INT(T, true, false, Tr, Sat, false, false);
            } else {
                NWGRAD_SCORE_POS_INT(T, false, false, Tr, Sat, false, false);
            }
        }
        return;
    }
    if (glocal)         score_affine_striped<T, false, true, Tr, Sat, true>(job);
    else if (job.local) score_affine_striped<T, true, true, Tr, Sat>(job);
    else                score_affine_striped<T, false, true, Tr, Sat>(job);
}
// The registered entry is pinned to a page boundary (one per type and level; the default
// forms are inlined into it, so their row loops sit at fixed offsets from the page), as
// inter_fill_affine is (inter_kernel_impl.inl): otherwise their address mod 4096 is the
// size of everything linked before, and it moved int32 Global score-only cells on spot
// (L3000 0.999 -> 1.007 when fbd7bf9 relaid the inter units) with no change to the code.
// At most 4 KiB of padding per entry (5 per level).
#if defined(__GNUC__)
#  define NWGRAD_SCORE_ENTRY_ALIGNED __attribute__((aligned(4096), noinline))
#else
#  define NWGRAD_SCORE_ENTRY_ALIGNED
#endif
template <class T>
NWGRAD_SCORE_ENTRY_ALIGNED static void score_entry(ScoreJob<T>& job) {
    if constexpr (nw_sat<T>) {   // past the quarter-range bound: the saturating variant
        if (job.sat) {
            if (job.tr.on()) score_entry_impl<T, true, true>(job);
            else             score_entry_impl<T, false, true>(job);
            return;
        }
    }
    if (job.tr.on()) score_entry_impl<T, true>(job);
    else             score_entry_impl<T, false>(job);
}
