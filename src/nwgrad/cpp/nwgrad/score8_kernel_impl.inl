// ── int8 score-only: the saturating 8-bit tier ──────────────────────────────────
//
// #include'd once per ISA level inside the level namespace (level_common.inc, the typed
// unit), after kernels_impl.inl.  The score of an integer problem whose parameters fit
// int8, computed in saturating int8 lanes: twice the int16 kernel's lanes, half its rows'
// bytes.  The caller (Aligner::score_i8_, aligner.hpp) runs it first and falls back to the
// int16 / int32 kernels when it reports `overflow` (a cascade 8 -> 16 -> 32): a result
// that may have saturated is never returned.
//
// The kernel is Farrar's striped recurrence with lazy-F, its own (not the int16 kernel's
// D/X form): that form's stored prefix forms open + k * ge ramps, which int8 cannot hold.
// Gap parameters >= 0 only (the caller's precondition), so every transition but the
// diagonal SUBTRACTS, and with go >= 0 the three-state cell is the two-state Gotoh one:
// X(i,j) = max(max(M, Y)(i-1,j) - go_b - ge_b, X(i-1,j) - ge_b) equals
// max(H(i-1,j) - (go_b + ge_b), X(i-1,j) - ge_b) with H = max(M, X, Y), since the X term
// inside H loses to X - ge_b; likewise Y.  (v - go) - ge and v - (go + ge) are the same
// saturated value for go, ge >= 0 (both clamp at the same floor), so one subtraction opens.
//
// Why a saturated value is never returned (S: the stored value, V: the exact one; every
// operation is a saturating add / subtract or a max, all monotone):
//   * Local.  M = max(H(diag) + s, 0) >= 0, so H >= 0 is exact as long as nothing reaches
//     the top: an add that clamps at 127 makes that cell's H 127.  X / Y only subtract, so
//     each is exact or clamped at -128 with V <= -128, and a clamped one never wins
//     H = max(M >= 0, X, Y).  So: the maximum over every H (the score itself) < 127 => the
//     score is exact; == 127 => `overflow`.  Checked per row, so an overflowing problem
//     stops early.  Padding columns (substitution 0, row 0 at 0) never exceed the real
//     maximum (they only copy or lower real values), so no mask.
//   * Global / Glocal.  The bottom does clamp (the borders alone reach -(go + L * ge)).  A
//     clamped value is >= its V and climbs only by the gains after the clamp, so by
//     induction over the cells S(c) <= max(V(c), -128 + G) with G the most any path can
//     gain, at most min(m, n) * smax_pos (only diagonals gain), and S(c) >= V(c).  The
//     caller requires G <= 126, so no value ever reaches the top (V <= G and every
//     sentinel-derived value <= -128 + G).  Then a score S > G - 128 is exact (S <= max(V,
//     -128 + G) and S > -128 + G give S <= V); otherwise `overflow`.  Glocal takes the
//     row-m maximum: its argmax cell is exact by the same test and every other cell has
//     V <= S <= the maximum, so the maximum is the exact one.
// The score is then the integer the int16 / int32 kernels compute, bit for bit.
//
// Lazy-F stops when no lane's carried F, extended once more, beats the cell's H BEFORE this
// pass raised it, opened: what the main pass already handed the next segment.  (Against
// the raised H instead, a zero open stops one step early: the raised H equals
// F, and F - ge == H - (0 + ge) is not strictly greater.)

#ifndef NWGRAD_LEVEL_NS
#  error "score8_kernel_impl.inl must be included inside a level namespace by a level TU"
#endif

// Saturating int8 add / subtract on native_simd<int8_t>: one packed instruction per
// width (paddsb / psubsb, vpaddsb, sqadd / sqsub), lane by lane where there is none.
template <class V>
[[gnu::always_inline]] static inline V i8_adds(const V& a, const V& b) {
#if defined(__AVX512BW__)
    if constexpr (V::size() == 64) return V(_mm512_adds_epi8(static_cast<__m512i>(a), static_cast<__m512i>(b)));
#endif
#if defined(__AVX2__)
    if constexpr (V::size() == 32) return V(_mm256_adds_epi8(static_cast<__m256i>(a), static_cast<__m256i>(b)));
#endif
#if defined(__SSE2__)
    if constexpr (V::size() == 16) return V(_mm_adds_epi8(static_cast<__m128i>(a), static_cast<__m128i>(b)));
#endif
#if defined(__ARM_NEON)
    if constexpr (V::size() == 16) return V(vqaddq_s8(static_cast<int8x16_t>(a), static_cast<int8x16_t>(b)));
#endif
    return V([&](int q) { return static_cast<int8_t>(std::clamp(int(a[q]) + int(b[q]), -128, 127)); });
}
template <class V>
[[gnu::always_inline]] static inline V i8_subs(const V& a, const V& b) {
#if defined(__AVX512BW__)
    if constexpr (V::size() == 64) return V(_mm512_subs_epi8(static_cast<__m512i>(a), static_cast<__m512i>(b)));
#endif
#if defined(__AVX2__)
    if constexpr (V::size() == 32) return V(_mm256_subs_epi8(static_cast<__m256i>(a), static_cast<__m256i>(b)));
#endif
#if defined(__SSE2__)
    if constexpr (V::size() == 16) return V(_mm_subs_epi8(static_cast<__m128i>(a), static_cast<__m128i>(b)));
#endif
#if defined(__ARM_NEON)
    if constexpr (V::size() == 16) return V(vqsubq_s8(static_cast<int8x16_t>(a), static_cast<int8x16_t>(b)));
#endif
    return V([&](int q) { return static_cast<int8_t>(std::clamp(int(a[q]) - int(b[q]), -128, 127)); });
}
// [b, x0, ..., x_{W-2}]: the striped carry from the previous lane's last segment.
template <class V>
[[gnu::always_inline]] static inline V i8_shift_in(const V& x, int8_t b) {
#if defined(__AVX512BW__)
    if constexpr (V::size() == 64) {
        // 128-bit lane k of t = lane k-1 of x (lane 0 zero), then a byte alignr per lane.
        const __m512i xi = static_cast<__m512i>(x);
        const __m512i t = _mm512_alignr_epi64(xi, _mm512_setzero_si512(), 6);
        const __m512i r = _mm512_alignr_epi8(xi, t, 15);
        return V(_mm512_mask_blend_epi8(__mmask64(1), r, _mm512_set1_epi8(b)));
    }
#endif
#if defined(__AVX2__)
    if constexpr (V::size() == 32) {
        const __m256i xi = static_cast<__m256i>(x);
        const __m256i t = _mm256_permute2x128_si256(xi, xi, 0x08);   // [0 | x.low]
        const __m256i r = _mm256_alignr_epi8(xi, t, 15);              // byte 0 is 0
        return V(_mm256_or_si256(r, _mm256_zextsi128_si256(_mm_cvtsi32_si128(static_cast<uint8_t>(b)))));
    }
#endif
#if defined(__SSE2__)
    if constexpr (V::size() == 16)
        return V(_mm_or_si128(_mm_slli_si128(static_cast<__m128i>(x), 1), _mm_cvtsi32_si128(static_cast<uint8_t>(b))));
#endif
#if defined(__ARM_NEON)
    if constexpr (V::size() == 16) return V(vextq_s8(vdupq_n_s8(b), static_cast<int8x16_t>(x), 15));
#endif
    return V([&](int q) { return q == 0 ? b : x[q - 1]; });
}
// Any lane a > b.  AVX-512 through the mask intrinsic: clang cannot compile libstdc++'s
// 64-lane int8 compare (the _S_to_bits lane-type assertion kernels_impl.inl works around
// for 64-bit lanes trips on char vs signed char here).
template <class V>
[[gnu::always_inline]] static inline bool i8_any_gt(const V& a, const V& b) {
#if defined(__AVX512BW__)
    if constexpr (V::size() == 64) return _mm512_cmpgt_epi8_mask(static_cast<__m512i>(a), static_cast<__m512i>(b)) != 0;
    else
#endif
    return stdx::any_of(a > b);
}

// Local, Glocal (Local false), Linear (go = 0: one row H, no X row), Sym (go_a == go_b,
// ge_a == ge_b: one open per cell serves both directions).
template <bool Local, bool Glocal, bool Linear, bool Sym>
static void score8_striped(Score8Job& job) {
    using V = stdx::native_simd<int8_t>;
    constexpr int W = static_cast<int>(V::size());
    constexpr int8_t NINF = std::numeric_limits<int8_t>::lowest();
    const int m = job.m, n = job.n, nalpha = job.nalpha;
    const int seg = (n + W - 1) / W;
    const std::size_t sw = static_cast<std::size_t>(seg) * W;
    const unsigned char* a = job.a;
    const unsigned char* b = job.b;

    // Scratch, 64-byte aligned: the profile rows of the codes A uses, then H (previous,
    // current) and X.
    std::vector<int8_t>& sc = *job.scratch;
    const std::size_t need = static_cast<std::size_t>(nalpha + 3) * sw + 64;
    if (sc.size() < need) sc.resize(need);
    int8_t* base = reinterpret_cast<int8_t*>((reinterpret_cast<std::uintptr_t>(sc.data()) + 63) & ~std::uintptr_t(63));
    int8_t* prof = base;
    int8_t* Hp = prof + static_cast<std::size_t>(nalpha) * sw;
    int8_t* Hc = Hp + sw;
    int8_t* X = Hc + sw;

    bool used[256] = {};
    for (int i = 0; i < m; ++i) used[a[i]] = true;
    for (int c = 0; c < nalpha; ++c) {
        if (!used[c]) continue;
        const int8_t* row = job.blk + static_cast<std::size_t>(c) * nalpha;
        int8_t* p = prof + static_cast<std::size_t>(c) * sw;
        for (int s = 0; s < seg; ++s)
            for (int l = 0; l < W; ++l) {
                const int j = l * seg + s;
                p[static_cast<std::size_t>(s) * W + l] = j < n ? row[b[j]] : int8_t(0);
            }
    }
    const int goe_a = int(job.go_a) + job.ge_a, goe_b = int(job.go_b) + job.ge_b;
    auto sat8 = [](int v) { return static_cast<int8_t>(std::max(v, -128)); };
    // Column 0: H(i, 0).  Global / Glocal the gap run down column 0, Local 0.
    auto h0 = [&](int i) -> int8_t {
        if (Local || i == 0) return 0;
        return sat8(-(int(job.go_b) + i * int(job.ge_b)));
    };
    // Row 0: Global the gap run along it (padding: the sentinel -- it feeds no real cell),
    // Local / Glocal a free start (0).  The X row holds the NEXT row's X, so it starts as
    // X(1, j) = max(X(0, j) - ge_b, H(0, j) - goe_b) with X(0, j) the sentinel.
    for (int s = 0; s < seg; ++s)
        for (int l = 0; l < W; ++l) {
            const int j = l * seg + s;   // column j + 1
            const std::size_t k = static_cast<std::size_t>(s) * W + l;
            if constexpr (Local || Glocal) Hp[k] = 0;
            else Hp[k] = j < n ? sat8(-(int(job.go_a) + (j + 1) * int(job.ge_a))) : NINF;
            if constexpr (!Linear) X[k] = sat8(int(Hp[k]) - goe_b);
        }

    const V vgoeA(static_cast<int8_t>(goe_a)), vgeA(job.ge_a);
    const V vgoeB(static_cast<int8_t>(goe_b)), vgeB(job.ge_b);
    const V vzero(int8_t(0)), vninf(NINF), v126(int8_t(126));
    V vmax(int8_t(0));   // Local: the running maximum of H
    const auto ld = [](const int8_t* p) { V v; v.copy_from(p, stdx::vector_aligned); return v; };
    const auto st = [](int8_t* p, const V& v) { v.copy_to(p, stdx::vector_aligned); };

    for (int i = 1; i <= m; ++i) {
        const int8_t* P = prof + static_cast<std::size_t>(a[i - 1]) * sw;
        const int8_t left = h0(i);
        V vH = i8_shift_in(ld(Hp + static_cast<std::size_t>(seg - 1) * W), h0(i - 1));
        // F into column 1 from column 0; the other lanes' first segment waits for lazy-F.
        V vF = i8_shift_in(vninf, Linear ? sat8(int(left) - job.ge_a) : sat8(int(left) - goe_a));
        for (int s = 0; s < seg; ++s) {
            const std::size_t k = static_cast<std::size_t>(s) * W;
            const V up = ld(Hp + k);
            V h = i8_adds(vH, ld(P + k));
            if constexpr (Local) h = stdx::max(h, vzero);
            if constexpr (Linear) {
                h = stdx::max(h, i8_subs(up, vgeB));
                h = stdx::max(h, vF);
                st(Hc + k, h);
                if constexpr (Local) vmax = stdx::max(vmax, h);
                vF = i8_subs(h, vgeA);   // h >= vF, so max(vF - ge, h - ge) is h - ge
            } else {
                V x = ld(X + k);
                h = stdx::max(h, x);
                h = stdx::max(h, vF);
                st(Hc + k, h);
                if constexpr (Local) vmax = stdx::max(vmax, h);
                const V hoB = i8_subs(h, vgoeB);
                const V hoA = Sym ? hoB : i8_subs(h, vgoeA);
                st(X + k, stdx::max(i8_subs(x, vgeB), hoB));
                vF = stdx::max(i8_subs(vF, vgeA), hoA);
            }
            vH = up;
        }
        // Lazy-F: carry F across the lane boundaries until no lane improves.
        vF = i8_shift_in(vF, NINF);
        for (int pass = 0; pass < W; ++pass) {
            for (int s = 0; s < seg; ++s) {
                const std::size_t k = static_cast<std::size_t>(s) * W;
                const V hold = ld(Hc + k);
                const V h = stdx::max(hold, vF);
                st(Hc + k, h);
                if constexpr (Local) vmax = stdx::max(vmax, h);
                if constexpr (!Linear) st(X + k, stdx::max(ld(X + k), i8_subs(h, vgoeB)));
                const V fn = i8_subs(vF, vgeA);
                if (!i8_any_gt(fn, Linear ? i8_subs(hold, vgeA) : i8_subs(hold, vgoeA))) goto row_done;
                vF = fn;
            }
            vF = i8_shift_in(vF, NINF);
        }
    row_done:
        std::swap(Hp, Hc);
        if constexpr (Local) {
            if (i8_any_gt(vmax, v126)) { job.overflow = 1; job.score = 127; return; }
        }
    }

    int score;
    if constexpr (Local) {
        score = 0;
        for (int l = 0; l < W; ++l) score = std::max(score, int(vmax[l]));
        job.overflow = 0;
    } else {
        if constexpr (Glocal) {
            score = h0(m);
            for (int s = 0; s < seg; ++s)
                for (int l = 0; l < W; ++l)
                    if (l * seg + s < n) score = std::max(score, int(Hp[static_cast<std::size_t>(s) * W + l]));
        } else {
            const int j = n - 1;
            score = Hp[static_cast<std::size_t>(j % seg) * W + j / seg];
        }
        job.overflow = score > job.gain - 128 ? 0 : 1;
    }
    job.score = score;
}

template <bool Local, bool Glocal>
[[gnu::always_inline]] static inline void score8_gm(Score8Job& job) {
    if (job.linear)                                           score8_striped<Local, Glocal, true, false>(job);
    else if (job.go_a == job.go_b && job.ge_a == job.ge_b)   score8_striped<Local, Glocal, false, true>(job);
    else                                                      score8_striped<Local, Glocal, false, false>(job);
}
// The registered entry (LevelKernels::score_i8).  Preconditions (the caller's): m, n >= 1,
// every gap parameter >= 0 with go + ge <= 127 per side (linear: go = 0), every
// substitution score within [-127, 127], Global / Glocal job.gain = min(m, n) * smax_pos
// <= 126.
static void score8_entry(Score8Job& job) {
    if (job.local == 1)      score8_gm<true, false>(job);
    else if (job.local == 2) score8_gm<false, true>(job);
    else                     score8_gm<false, false>(job);
}
