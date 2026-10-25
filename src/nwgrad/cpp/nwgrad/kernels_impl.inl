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
// The int64 lane hits the same assertion (its mask is the same 64-bit lane type), so the
// same three helpers exist for it over the epi64 intrinsics.
static inline __mmask8 avx512d_ge(stdx::native_simd<int64_t> a, stdx::native_simd<int64_t> b) {
    alignas(64) int64_t aa[8], bb[8];
    a.copy_to(aa, stdx::element_aligned);
    b.copy_to(bb, stdx::element_aligned);
    return _mm512_cmpge_epi64_mask(_mm512_load_si512(aa), _mm512_load_si512(bb));
}
static inline __mmask8 avx512d_gt(stdx::native_simd<int64_t> a, stdx::native_simd<int64_t> b) {
    alignas(64) int64_t aa[8], bb[8];
    a.copy_to(aa, stdx::element_aligned);
    b.copy_to(bb, stdx::element_aligned);
    return _mm512_cmpgt_epi64_mask(_mm512_load_si512(aa), _mm512_load_si512(bb));
}
static inline stdx::native_simd<int64_t> avx512d_blend(__mmask8 k, stdx::native_simd<int64_t> dest,
                                                        stdx::native_simd<int64_t> value) {
    alignas(64) int64_t da[8], va[8], out[8];
    dest.copy_to(da, stdx::element_aligned);
    value.copy_to(va, stdx::element_aligned);
    _mm512_store_si512(out, _mm512_mask_blend_epi64(k, _mm512_load_si512(da), _mm512_load_si512(va)));
    stdx::native_simd<int64_t> res;
    res.copy_from(out, stdx::element_aligned);
    return res;
}
// The lazy-F early exit (any lane of F above v), for both 8-byte lane types.
template <class V>
static inline bool avx512d_any_gt(const V& F, const V& v) { return avx512d_gt(F, v) != 0; }
#endif

// ── Local: the best cell of one striped row, vectorized ──────────────────────
// Exactly the scalar scan in column order (max(M,X,Y), strict `>` against the running
// best, M>X>Y for the table): each lane keeps its largest value and the FIRST segment
// that reached it (strict `>`; columns grow with s within a lane), and the first lane
// holding the row's maximum is the leftmost column (lane order is column order).  The
// scalar form compiled under gcc to compare-and-branch on every cell, mispredicted
// (~30% of a Local Pointers fill).  pad[k] = 0 for a real column, -inf for padding: with
// a negative gap extend a Y run GAINS score into the padding, so it must be masked.
template <class T>
static inline void striped_row_best(const T* cM, const T* cX, const T* cY, const T* pad,
                                    int seg, int i, T& best, int& bi, int& bj, int& btbl) {
    using vt = stdx::native_simd<T>;
    constexpr int W = (int)vt::size();
    vt lmax(nw_ninf<T>), ls(T(0));
    // The segment index as a vector, carried and incremented — NOT `vt(T(s))` in the loop:
    // for 64-bit lanes at AVX-512 gcc lowered that broadcast to eight merge-masked
    // `vpbroadcastq`, one per lane and each dependent on the last, an 8-deep chain per
    // segment that the blend then waited on; the Local int64 Pointers fill spent 46 % of
    // its time here (perf, solace, gcc 15.2), 72 vs clang's 43 µs a pair at L200.
    vt vs(T(0));
    const vt one(T(1));
    for (int s = 0; s < seg; ++s, vs += one) {
        const std::size_t o = (std::size_t)s * W;
        vt vm, vx, vy, vp;
        vm.copy_from(cM + o, stdx::element_aligned);
        vx.copy_from(cX + o, stdx::element_aligned);
        vy.copy_from(cY + o, stdx::element_aligned);
        vp.copy_from(pad + o, stdx::element_aligned);
        const vt h = stdx::max(stdx::max(vm, vx), vy) + vp;
#if defined(__clang__) && defined(__AVX512F__) && !defined(NWGRAD_STD_SIMD_AVX512_MASK_OK)
        if constexpr (sizeof(T) == 8) ls = avx512d_blend(avx512d_gt(h, lmax), ls, vs);
        else                                      stdx::where(h > lmax, ls) = vs;
#else
        stdx::where(h > lmax, ls) = vs;
#endif
        lmax = stdx::max(lmax, h);
    }
    T rmax;
    int l;
    // (clang + AVX-512: the 64-bit lane's equality mask hits the libstdc++ assertion the
    // file's workaround describes; int64 keeps the scalar reduction there.)
#if defined(__clang__) && defined(__AVX512F__) && !defined(NWGRAD_STD_SIMD_AVX512_MASK_OK)
    constexpr bool vec_reduce = nw_is_int<T> && sizeof(T) != 8;
#else
    constexpr bool vec_reduce = nw_is_int<T>;
#endif
    if constexpr (vec_reduce) {
        // A vector reduction and a mask search (gcc extracted the 16 int16 lanes one by
        // one, ~5 % of the Pointers kernel's time).  Exact for integers; floating T keeps
        // the scalar loop, whose std::max keeps the FIRST of two equal values and so the
        // sign of a zero the reduction might swap.
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
    const std::size_t k = (std::size_t)s * W + (std::size_t)l;
    const T mm = cM[k], xx = cX[k], yy = cY[k];
    best = rmax; bi = i; bj = l * seg + s + 1;
    btbl = (mm >= xx && mm >= yy) ? 0 : ((xx >= yy) ? 1 : 2);
}

// pad[s*W + l] = 0 for a real column (l*seg + s + 1 <= n), -inf for padding.
// The same mask into a row the caller owns (a spread_rows row: dp_buffer.hpp).
template <class T>
static inline const T* striped_pad_into(T* v, int n, int seg, int W) {
    for (int l = 0; l < W; ++l)
        for (int s = 0; s < seg; ++s)
            v[(std::size_t)s * W + l] = (l * seg + s + 1 <= n) ? T(0) : nw_ninf<T>;
    return v;
}

template <class V>
static inline const typename V::value_type* striped_pad(V& v, int n, int seg, int W) {
    const std::size_t sw = (std::size_t)seg * W;
    if (v.size() < sw) v.resize(sw);
    return striped_pad_into(v.data(), n, seg, W);
}

using vd = stdx::native_simd<double>;
static constexpr int KW = (int)vd::size();     // native double-lane count (row_kernel uses it)
static constexpr double K_NINF = -std::numeric_limits<double>::infinity();

// The segment-boundary lane shift: [b, x0, x1, ..., x(W-2)].  Written as a std::simd
// generator, gcc lowers it at AVX2 width to partial reloads of x through memory at an
// 8-byte offset (vmovupd 8(%r), vmovhpd, vinsertf128) -- in the lazy-F rounds right
// after the carry stored that very vector -- where clang permutes it in a register
// (vpermpd + blend).  The intrinsic form gets gcc the same; at AVX-512 it is one valign
// (gcc 0.84-0.98 on the striped arms, clang unchanged).  A pure lane move, so the result
// is identical; SSE2/SSE4/NEON (W=2) keep the generator.
// The integer types (int16/int32/int64) take the same treatment: gcc assembled the
// generator form lane by lane (vpinsrw / vpinsrd / vpunpck sequences) at AVX2.
template <class V>
[[gnu::always_inline]] static inline V lane_shift_in(V x, typename V::value_type b) {
    using T [[maybe_unused]] = typename V::value_type;   // unused at the W=2 levels
#if defined(__AVX512F__)
    // One valignq / valignd: (x : splat(b)) shifted right by W-1 lanes is [b, x0..x_{W-2}].
    if constexpr (std::is_same_v<T, double> && V::size() == 8) {
        const __m512i r = _mm512_alignr_epi64(_mm512_castpd_si512(static_cast<__m512d>(x)),
                                              _mm512_castpd_si512(_mm512_set1_pd(b)), 7);
        return V(_mm512_castsi512_pd(r));
    } else if constexpr (std::is_same_v<T, float> && V::size() == 16) {
        const __m512i r = _mm512_alignr_epi32(_mm512_castps_si512(static_cast<__m512>(x)),
                                              _mm512_castps_si512(_mm512_set1_ps(b)), 15);
        return V(_mm512_castsi512_ps(r));
    } else if constexpr (std::is_same_v<T, int64_t> && V::size() == 8) {
        using NV = __v8di;
        const __m512i r = _mm512_alignr_epi64((__m512i)static_cast<NV>(x), _mm512_set1_epi64(b), 7);
        return V((NV)r);
    } else if constexpr (std::is_same_v<T, int32_t> && V::size() == 16) {
        using NV = __v16si;
        const __m512i r = _mm512_alignr_epi32((__m512i)static_cast<NV>(x), _mm512_set1_epi32(b), 15);
        return V((NV)r);
    } else if constexpr (std::is_same_v<T, int16_t> && V::size() == 32) {
#if defined(__AVX512BW__)
        using NV = __v32hi;
        const __m512i idx = _mm512_set_epi16(30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15,
                                             14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0);
        __m512i r = _mm512_permutexvar_epi16(idx, (__m512i)static_cast<NV>(x));
        r = _mm512_mask_blend_epi16(__mmask32(1), r, _mm512_set1_epi16(b));
        return V((NV)r);
#else
        return V([&](int q) { return q == 0 ? b : x[q - 1]; });
#endif
    } else
#elif defined(__AVX2__)
    if constexpr (std::is_same_v<T, double> && V::size() == 4) {
        const __m256d r = _mm256_permute4x64_pd(static_cast<__m256d>(x), 0x90);
        return V(_mm256_blend_pd(r, _mm256_set1_pd(b), 1));
    } else if constexpr (std::is_same_v<T, float> && V::size() == 8) {
        const __m256 r = _mm256_permutevar8x32_ps(static_cast<__m256>(x),
                                                  _mm256_setr_epi32(0, 0, 1, 2, 3, 4, 5, 6));
        return V(_mm256_blend_ps(r, _mm256_set1_ps(b), 1));
    } else if constexpr (std::is_same_v<T, int64_t> && V::size() == 4) {
        using NV = __v4di;
        const __m256i r = _mm256_permute4x64_epi64((__m256i)static_cast<NV>(x), 0x90);
        return V((NV)_mm256_blend_epi32(r, _mm256_set1_epi64x(b), 0x3));
    } else if constexpr (std::is_same_v<T, int32_t> && V::size() == 8) {
        using NV = __v8si;
        const __m256i r = _mm256_permutevar8x32_epi32((__m256i)static_cast<NV>(x),
                                                      _mm256_setr_epi32(0, 0, 1, 2, 3, 4, 5, 6));
        return V((NV)_mm256_blend_epi32(r, _mm256_set1_epi32(b), 1));
    } else if constexpr (std::is_same_v<T, int16_t> && V::size() == 16) {
        // A 2-byte shift across the whole register: alignr works within 128-bit halves,
        // so feed it the low half moved up (and zero below), then insert the border.
        using NV = __v16hi;
        const __m256i xi = (__m256i)static_cast<NV>(x);
        const __m256i t = _mm256_permute2x128_si256(xi, xi, 0x08);   // [0 | x.low]
        const __m256i r = _mm256_alignr_epi8(xi, t, 14);
        return V((NV)_mm256_insert_epi16(r, b, 0));
    } else
#endif
    return V([&](int q) { return q == 0 ? b : x[q - 1]; });
}

// ── The gap ramp k*ge_a, striped: slot (s, l) carries the ABSOLUTE column index k =
// l*seg + s times ge_a (the prefix-max carries below and in hb_kernel_impl.inl).  A
// scalar loop, loaded and never multiplied in a hot loop (no FMA for floating T; for an
// integer T the product is exact anyway).
template <class T>
static inline void hb_ramp(T* rmp, T ge_a, int seg, int W) {
    for (int l = 0; l < W; ++l)
        for (int s = 0; s < seg; ++s)
            rmp[(std::size_t)s * W + l] = static_cast<T>(l * seg + s) * ge_a;
}

// ── Integer T: the VY carry as an EXACT prefix max, no lazy-F ─────────────────────
//
// VY[c] = max over k <= c-1 of (open[k] + k*ge_a) minus (c-1)*ge_a (hb_sweep_striped_pmax,
// hb_kernel_impl.inl), with the column-0 border folded in as the seed bY - ge_a.  In
// floating point the ramp's round trip costs a rounding, which is why that carry is a
// sibling mode there; in integer arithmetic it is exact, so every striped integer kernel
// takes it in place of the serial chain + lazy-F fixpoint: fixed work per row (a lane-
// local prefix max, W scalar maxes across the lanes, one parallel pass) instead of up to W
// data-dependent correction rounds — which grow with the lane count, and int16 has 4x
// double's lanes.  Bit-identical to the chain by construction (tested against the double
// oracle at every level).  `ov` holds the opens per segment slot, read one segment back as
// the chain reads them (the lane shift at s == 0 with the border's open); the ramp k*ge_a
// is formed in a register (a lane base l*seg*ge_a, plus ge_a a segment: exact in integers,
// and never past the ramp's last value), not read from a row — one stream fewer, which
// moved the L1 capacity edge of the kernels that run it (nighthaven AVX2, int32: linear H
// fill 0.79-0.90 of the row-ramp time over L1600-2400, Scores L2030 0.75, Pointers L1000
// 0.90-0.93, 0.98-1.02 inside L1); Y lands in cY (+off), and P (the prefix maxima) rides in
// cY between the two passes.
// Codes (`ok`, `yc` non-null): the Y code is the open's (ok, shifted like ov) when the
// open at column c-1 is at least the prefix before it — Q[c-1] >= max(seed, Q[0..c-2]) —
// else 2 (extension): exactly `O >= ext` of the chain, since ext = that prefix - (c-1)*ge.
// Loads and stores of a std::simd through the GCC vector type of its width.  gcc lowered
// `copy_to` of an INTEGER simd (vector_aligned or not) to a memcpy through the value's
// address: the loop-carried prefix max below was kept on the stack and copied into the
// row through two general-purpose registers per vector (40 % of the sse4 int32 carry,
// the score kernel at 1.6x its lazy-F time).  A store through the native vector type is
// one instruction under both compilers.  (The types are spelled per width: gcc drops a
// dependent `vector_size` in an alias template to the scalar.)
template <class T, int Bytes> struct NVecOf;
#define NWGRAD_NV(T, B) template <> struct NVecOf<T, B> { typedef T type __attribute__((vector_size(B))); };
NWGRAD_NV(int16_t, 16) NWGRAD_NV(int16_t, 32) NWGRAD_NV(int16_t, 64)
NWGRAD_NV(int32_t, 16) NWGRAD_NV(int32_t, 32) NWGRAD_NV(int32_t, 64)
NWGRAD_NV(int64_t, 16) NWGRAD_NV(int64_t, 32) NWGRAD_NV(int64_t, 64)
#undef NWGRAD_NV
template <class V> [[gnu::always_inline]] static inline V nv_load(const typename V::value_type* p) {
    using NV = typename NVecOf<typename V::value_type, (int)(sizeof(typename V::value_type) * V::size())>::type;
    return V(*reinterpret_cast<const NV*>(p));
}
template <class V> [[gnu::always_inline]] static inline void nv_store(typename V::value_type* p, const V& x) {
    using NV = typename NVecOf<typename V::value_type, (int)(sizeof(typename V::value_type) * V::size())>::type;
    *reinterpret_cast<NV*>(p) = static_cast<NV>(x);
}

// x moved K lanes up, the low K lanes filled from f.  f is a SPLAT at every call, so the fill
// is taken from f's TOP lanes (f[W-K+I]), which spells the shuffle as one contiguous window
// of the pair (f, x): AArch64 `ext`, AVX2 `vperm2i128` + `vpalignr`.  Filling from f[I] (the
// same values) compiled under gcc 16 on NEON to a two-register `tbl` whose operands went
// through the stack, three round trips in the carry, and an index load per row (int16 1.015-
// 1.026 of the scalar scan on the M1; 0.906-0.950 this way).
template <class NV, int W, int K, std::size_t... I>
[[gnu::always_inline]] static inline NV nv_shift_up_(NV x, NV f, std::index_sequence<I...>) {
    return __builtin_shufflevector(x, f, (I < (std::size_t)K ? 2 * W - K + (int)I : (int)I - K)...);
}
template <class NV, int W, int K>
[[gnu::always_inline]] static inline NV nv_shift_up(NV x, NV f) {
    return nv_shift_up_<NV, W, K>(x, f, std::make_index_sequence<W>{});
}
// The carry-in of every lane: cin[l] = max(seed, P[0..l-1]), an exclusive prefix max across
// the lanes.  In registers, log2(W) shift + max steps, where the scalar form stored P,
// walked the lanes through a cmov chain and read cin back as one vector from W narrow
// stores (a load no store can forward).  Integer max is exact and associative, so the
// lanes carry the same values either way.  Time / scalar form (2026-10-25, medians of 9,
// controls 0.98-1.02): nighthaven AVX2 0.84-0.95 at L100-300 (int16 / int32 Scores,
// Pointers, linear), 0.97-1.01 at L1000; solace AVX-512 0.73-0.90, L1000 0.99; spot NEON
// (M1, g++-16) 0.85-0.98, int16 Scores 0.91-0.95.
// NWGRAD_SCALAR_LANE_SCAN forces the scalar form.
#if !defined(NWGRAD_SCALAR_LANE_SCAN) && defined(__has_builtin)
#if __has_builtin(__builtin_shufflevector)
#define NWGRAD_LANE_SCAN_REG 1
#endif
#endif
template <class vd, int W>
[[gnu::always_inline]] static inline vd int_lane_carry_in(const vd& P, typename vd::value_type seed) {
    using T = typename vd::value_type;
#ifdef NWGRAD_LANE_SCAN_REG
    using NV = typename NVecOf<T, (int)(sizeof(T) * W)>::type;
    const NV ninf = static_cast<NV>(vd(nw_ninf<T>));
    NV x = static_cast<NV>(P);
    auto mx = [](NV a, NV b) __attribute__((always_inline)) { return a > b ? a : b; };
    x = mx(x, nv_shift_up<NV, W, 1>(x, ninf));
    if constexpr (W > 2)  x = mx(x, nv_shift_up<NV, W, 2>(x, ninf));
    if constexpr (W > 4)  x = mx(x, nv_shift_up<NV, W, 4>(x, ninf));
    if constexpr (W > 8)  x = mx(x, nv_shift_up<NV, W, 8>(x, ninf));
    if constexpr (W > 16) x = mx(x, nv_shift_up<NV, W, 16>(x, ninf));
    static_assert(W <= 32, "int_lane_carry_in: more lanes than steps");
    const NV vs = static_cast<NV>(vd(seed));
    return vd(mx(nv_shift_up<NV, W, 1>(x, vs), vs));
#else
    alignas(64) T tot[64], cin[64];
    P.copy_to(tot, stdx::element_aligned);
    T run = seed;
    for (int l = 0; l < W; ++l) { cin[l] = run; run = std::max(run, tot[l]); }
    vd vcin; vcin.copy_from(cin, stdx::element_aligned);
    return vcin;
#endif
}

// Always inlined: gcc outlined it as a constant-propagated clone in every integer score
// kernel (a call, its own frame and the lane arrays per row), where it then cost more
// than the kernel body around it.
template <class T, int W>
[[gnu::always_inline]] static inline void int_pmax_carry(const T* ov, const T* ok, T nbOpen, T nbOpenK,
                                  T bY, T ge_a, int seg, T* cY, T* yc, T* qbuf) {
    using vd = stdx::native_simd<T>;
    const vd vtwo(T(2));
    const vd vge(ge_a);
    const vd rbase([&](int l) { return static_cast<T>(static_cast<T>(l * seg) * ge_a); });
    vd rq = rbase;                          // k*ge_a for k = l*seg + s
    vd P(nw_ninf<T>);
    for (int s = 0; s < seg; ++s) {
        vd O;
        if (s == 0) {
            vd lo; lo = nv_load<vd>(ov + (std::size_t)(seg - 1) * W);
            O = lane_shift_in(lo, nbOpen);
        } else {
            O = nv_load<vd>(ov + (std::size_t)(s - 1) * W);
        }
        const vd Q = O + rq;
        rq += vge;
        if (yc) nv_store(qbuf + (std::size_t)s * W, Q);
        P = stdx::max(Q, P);
        nv_store(cY + (std::size_t)s * W, P);
    }
    rq = rbase;
    const vd vcin = int_lane_carry_in<vd, W>(P, static_cast<T>(bY - ge_a));
    vd before = vcin;                       // the prefix before this segment's open
    // Two ramps a step apart, each advanced by 2*ge_a: this pass has no other loop-carried
    // value, and one ramp's add chain bound it on NEON (2-cycle integer add: 1.04-1.07x the
    // row ramp on the M1; an x86 add is 1 cycle, under the loop's throughput).
    const vd vge2 = vge + vge;
    vd rq1 = rq + vge;
    auto step = [&](int s, const vd& r) __attribute__((always_inline)) {
        vd p; p = nv_load<vd>(cY + (std::size_t)s * W);
        const vd pc = stdx::max(p, vcin);
        nv_store(cY + (std::size_t)s * W, pc - r);
        if (yc) {
            vd OK, q;
            if (s == 0) {
                vd lk; lk = nv_load<vd>(ok + (std::size_t)(seg - 1) * W);
                OK = lane_shift_in(lk, nbOpenK);
            } else {
                OK = nv_load<vd>(ok + (std::size_t)(s - 1) * W);
            }
            q = nv_load<vd>(qbuf + (std::size_t)s * W);
            vd ky = OK;                           // strict: q >= before keeps the open
#if defined(__clang__) && defined(__AVX512F__) && !defined(NWGRAD_STD_SIMD_AVX512_MASK_OK)
            if constexpr (sizeof(T) == 8) ky = avx512d_blend(avx512d_gt(before, q), ky, vtwo);
            else                          stdx::where(before > q, ky) = vtwo;
#else
            stdx::where(before > q, ky) = vtwo;
#endif
            nv_store(yc + (std::size_t)s * W, ky);
            before = pc;
        }
    };
    int s = 0;
    for (; s + 1 < seg; s += 2) { step(s, rq); step(s + 1, rq1); rq += vge2; rq1 += vge2; }
    if (s < seg) step(s, rq);
}

// ── Tracked rows: the striped query profile ───────────────────────────────────
// One striped profile row: prof[s*W + l] = row[b[col(l,s)-1]], padding -> nw_pad.
template <class T>
static inline void striped_profile_row(T* dst, const T* row, const unsigned char* b,
                                       int n, int seg, int W) {
    const T padv = nw_pad<T>;   // -inf, or 0 for an integer T (scalar_traits.hpp)
    for (int l = 0; l < W; ++l)
        for (int s = 0; s < seg; ++s) {
            const int j = l * seg + s + 1;
            dst[(std::size_t)s * W + l] = (j <= n) ? row[b[j - 1]] : padv;
        }
}
// The profile rows a striped fill reads (see TrackJob): Tr with tr.ra builds one per
// compact symbol; Tr without it builds none here (striped_row_profile builds each DP
// row's as the fill reaches it, into the single row reserved); untracked, one per
// alphabet symbol — exactly the loop the untracked kernels always ran.
template <class T, bool Tr>
static inline void striped_profile_build(typename DpBufferT<T>::TVec& prof, const T* blk,
                                         int nalpha, const unsigned char* b, int n, int seg, int W,
                                         const TrackJob<T>& tr) {
    const std::size_t sw = (std::size_t)seg * W;
    if constexpr (Tr) {
        if (tr.ra) {
            if (prof.size() < (std::size_t)tr.nrow * sw) prof.resize((std::size_t)tr.nrow * sw);
            for (int c = 0; c < tr.nrow; ++c)
                striped_profile_row(prof.data() + (std::size_t)c * sw, blk + (std::size_t)tr.crow[c] * nalpha,
                                    b, n, seg, W);
        } else if (prof.size() < sw) {
            prof.resize(sw);
        }
    } else {
        (void)tr;
        if (prof.size() < (std::size_t)nalpha * sw) prof.resize((std::size_t)nalpha * sw);
        for (int c = 0; c < nalpha; ++c)
            striped_profile_row(prof.data() + (std::size_t)c * sw, blk + (std::size_t)c * nalpha, b, n, seg, W);
    }
}
// DP row i's profile row (1-based i).
template <class T, bool Tr>
static inline const T* striped_row_profile(T* prof, const T* blk, int nalpha, const unsigned char* a,
                                           const unsigned char* b, int n, int seg, int W, int i,
                                           const TrackJob<T>& tr) {
    const std::size_t sw = (std::size_t)seg * W;
    if constexpr (Tr) {
        if (tr.ra) return prof + (std::size_t)tr.ra[i - 1] * sw;
        striped_profile_row(prof, blk + (std::size_t)tr.rs[i - 1] * nalpha, b, n, seg, W);
        return prof;
    } else {
        (void)blk; (void)nalpha; (void)b; (void)n; (void)W; (void)tr;
        return prof + (std::size_t)a[i - 1] * sw;
    }
}

// ── striped affine forward, Full (Global if !Local, else Local) ───────────────
// Templated on the Viterbi precision T (double or float32).  vd / W / the -inf
// sentinel are all local to T here, shadowing the file-scope double versions above.
// Tr: a tracked problem (job.tr) — each row's penalties, border and profile row from
// the track arrays, in the same expressions, so equal sets give the untracked tables.
template <class T, bool Local, bool Tr = false>
static void striped_affine_full(ViterbiJob<T>& job) {
    using vd = stdx::native_simd<T>;
    const int W = (int)vd::size();                       // native T-lane count for this level
    const T K_NINF = nw_ninf<T>;

    const int m = job.m, n = job.n;
    const int nalpha = job.nalpha;
    T go_a = job.go_a, ge_a = job.ge_a, go_b = job.go_b, ge_b = job.ge_b;
    const TrackJob<T>& tr = job.tr;
    if constexpr (Tr) { go_a = tr.go_a[0]; ge_a = tr.ge_a[0]; }   // row 0's gaps in A
    const T* blk = job.blk;
    const unsigned char* a = job.a;
    const unsigned char* b = job.b;
    DpBufferT<T>& buf = *job.buf;

    const int seg = (n + W - 1) / W;
    const std::size_t sw = (std::size_t)seg * W;
    // VM/VX/VY hold the tables in STRIPED layout — one row of (seg+1)*W scalars
    // per i, written in place, so the traceback/hard-gradient read them striped with no
    // per-row de-stripe copy.  Slot 0 is column 0 (the gap border); slots [W, W+sw) are
    // the striped columns 1..n (started at W so every W-wide access is aligned given the
    // 64-byte allocator — see dp_buffer.hpp), column j at W + ((j-1)%seg)*W + (j-1)/seg;
    // slots 1..W-1 pad.  aligner.hpp's cell_index() computes the identical index — the
    // two must stay in lockstep.
    // On a narrow L1D (table_spread, dp_buffer.hpp) the row STRIDE is padded past (seg+1)*W
    // (spread_stride) and the three tables start at chosen lines of a 4 KB page
    // (table_lines), so rows i and i-1 of VM, VX and VY sit in six distinct L1 sets:
    // unplaced, a 4-way L1D thrashed (wloczykij int32 Local L1020 3.5x its neighbours).  The
    // stride reaches the aligner as job.rstride.
    const bool spread = table_spread();
    const std::size_t rowsz = spread ? spread_stride<T>((std::size_t)(seg + 1) * W, 2)
                                     : (std::size_t)(seg + 1) * W;
    const std::size_t off   = (std::size_t)W;   // striped columns start here (col 0 at 0)
    if (buf.sopenv.size() < sw)     buf.sopenv.resize(sw);
    const std::size_t vsz = (std::size_t)(m + 1) * rowsz;
    {
        if (spread) {
            const TableLines tl = table_lines(rowsz * sizeof(T) / 64);
            buf.VM.place(0); buf.VX.place(tl.a); buf.VY.place(tl.b);
        } else {
            buf.VM.place(-1); buf.VX.place(-1); buf.VY.place(-1);
        }
    }
    // Per vector, never keyed on one (see kernels_pointers_impl.inl): a fill that grows
    // only part of a group must not leave the rest short for the next one.  resize() also
    // applies the placement, so it runs even when the size suffices.
    for (auto* v : {&buf.VM, &buf.VX, &buf.VY}) v->resize(std::max(v->size(), vsz));
    // striped slot within a row (offset past slot 0) for DP column j in 1..n
    auto scol = [seg](int j) -> std::size_t {
        return (std::size_t)((j - 1) % seg) * W + (j - 1) / seg;
    };

    // striped query profile: prof[c][s*W+l] = score(c, b[col(l,s)-1]), padding -> NINF
    striped_profile_build<T, Tr>(buf.sprof, blk, nalpha, b, n, seg, W, tr);

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

    vd vgo_a(go_a), vge_a(ge_a), vgo_b(go_b), vge_b(ge_b);
    const vd vzero(static_cast<T>(0));
    // column-0 border of the previous row
    T bM = 0, bX = K_NINF, bY = K_NINF;

    T best_local = 0; int best_i = 0, best_j = 0, best_tbl = 0;
    const T* pad = Local ? striped_pad(buf.spad, n, seg, W) : nullptr;

    for (int i = 1; i <= m; ++i) {
        if constexpr (Tr) {
            go_a = tr.go_a[i]; ge_a = tr.ge_a[i]; go_b = tr.go_b[i]; ge_b = tr.ge_b[i];
            vgo_a = vd(go_a); vge_a = vd(ge_a); vgo_b = vd(go_b); vge_b = vd(ge_b);
        }
        // column-0 border of this row
        const T nbM = Local ? T(0) : K_NINF;
        T nbX;
        if constexpr (Tr) nbX = Local ? K_NINF : tr.bx0[i];
        else              nbX = Local ? K_NINF : -(go_b + i * ge_b);
        const T nbOpen = (std::max(nbM, nbX) - go_a) - ge_a;
        const T* sub = striped_row_profile<T, Tr>(buf.sprof.data(), blk, nalpha, a, b, n, seg, W, i, tr);

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
                    dM = lane_shift_in(lM, bM);
                    dX = lane_shift_in(lX, bX);
                    dY = lane_shift_in(lY, bY);
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

            // ── the carry: VY, same-lane striped chain + lazy-F (integer T: the exact
            // prefix max, int_pmax_carry) ──
            if constexpr (nw_is_int<T> && W >= 4) {
                int_pmax_carry<T, W>(ov, nullptr, nbOpen, T(0), bY, ge_a, seg,
                                     cY, nullptr, nullptr);
            } else {
            vd prev([&](int q) { return q == 0 ? bY : K_NINF; });
            for (int s = 0; s < seg; ++s) {
                vd O;
                if (s == 0) {
                    vd lo; lo.copy_from(ov + (std::size_t)(seg - 1) * W, stdx::element_aligned);
                    O = lane_shift_in(lo, nbOpen);
                } else {
                    O.copy_from(ov + (std::size_t)(s - 1) * W, stdx::element_aligned);
                }
                vd v = stdx::max(O, prev - vge_a);
                v.copy_to(cY + (std::size_t)s * W, stdx::element_aligned);
                prev = v;
            }
            for (int r = 0; r < W; ++r) {
                vd last; last.copy_from(cY + (std::size_t)(seg - 1) * W, stdx::element_aligned);
                vd F = lane_shift_in(last, bY);
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
                    if constexpr (sizeof(T) == 8) { if (!avx512d_any_gt(F, v)) break; }
                    else { if (!stdx::any_of(F > v)) break; }
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
            }   // integer / floating carry

            // ── Local: argmax over this row, j ascending, reading the striped cells,
            // with the scalar kernel's M>X>Y tie-break and strict > (bit-exact path) ──
            if constexpr (Local)
                striped_row_best<T>(cM, cX, cY, pad, seg, i, best_local, best_i, best_j, best_tbl);
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
    job.rstride = rowsz;
}

// The registered entry for a level, per precision: dispatch Full by mode.  GuideBanded
// is not handled here (the caller routes it to the row-wise kernel).
template <class T>
static void striped_full_entry(ViterbiJob<T>& job) {
    if (job.tr.on()) {
        if (job.align_mode) striped_affine_full<T, true, true>(job);
        else                striped_affine_full<T, false, true>(job);
        return;
    }
    if (job.align_mode) striped_affine_full<T, true>(job);
    else                striped_affine_full<T, false>(job);
}
