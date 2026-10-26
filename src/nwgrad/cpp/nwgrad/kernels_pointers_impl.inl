// ── Leveled affine Viterbi, TracebackMode::Pointers ──────────────────────────
//
// Included once per ISA level exactly like kernels_impl.inl, inside that level's
// namespace, compiled with that level's real -march.  Same striped layout, same
// recurrence, same lazy-F.  The difference is what it retains:
//
//   Scores     three T score tables, (m+1)*rowsz each   12 B/cell (float32)
//   Pointers   one BYTE predecessor table + two rolling score rows     1 B/cell
//
// The byte packs the three states' codes, `M | X << 2 | Y << 4` (M 0..3, 3 = Local stop;
// X and Y 0..2).  Every code is produced already shifted into its field (X's picked as
// 0 / 4 / 8, Y's carried as 0 / 16 / 32), so packing is a sum.  Y's code is final only
// after the carry, so M's and X's wait for it in a T-typed scratch row (`mx`) and the last
// per-row loop stores `mx + y` -- one byte stream instead of three.  (Measured against
// storing M + X as bytes and OR-ing Y in: that narrows twice, and lost in cache.)
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

// Narrow a T-typed lane-code vector (values 0..63) to one byte per lane.
// Deliberately a single static_simd_cast, not a per-lane loop: writing direction
// bytes scalar-per-lane measured ~2x slower than the vectorized narrow.
// For the integer lane types the narrow is spelled in intrinsics: clang lowered the
// int64 cast to one byte store per lane (movb x4 per vector, 1.13-1.33x gcc's time on the
// Pointers kernel), gcc the int16/int32 ones to vpshufb sequences; a pack or a vpmov is
// one or two instructions.  The codes are 0..63 (a packed byte), so saturating packs are exact.
template <class T, int W>
[[gnu::always_inline]] static inline void store_codes(const stdx::native_simd<T>& code, unsigned char* dst) {
    using bvec = stdx::fixed_size_simd<unsigned char, W>;
#if defined(__AVX512F__) && defined(__AVX512BW__)
    if constexpr (std::is_same_v<T, int64_t> && W == 8) {
        _mm_storel_epi64((__m128i*)dst, _mm512_cvtepi64_epi8((__m512i)static_cast<__v8di>(code))); return;
    } else if constexpr (std::is_same_v<T, int32_t> && W == 16) {
        _mm_storeu_si128((__m128i*)dst, _mm512_cvtepi32_epi8((__m512i)static_cast<__v16si>(code))); return;
    } else if constexpr (std::is_same_v<T, int16_t> && W == 32) {
        _mm256_storeu_si256((__m256i*)dst, _mm512_cvtepi16_epi8((__m512i)static_cast<__v32hi>(code))); return;
    } else
#elif defined(__AVX2__)
    if constexpr (std::is_same_v<T, int64_t> && W == 4) {
        // the low dword of each qword, then two packs: 4 bytes
        const __m256i d = _mm256_permutevar8x32_epi32((__m256i)static_cast<__v4di>(code),
                                                      _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6));
        const __m128i w = _mm_packus_epi32(_mm256_castsi256_si128(d), _mm256_castsi256_si128(d));
        const __m128i b = _mm_packus_epi16(w, w);
        std::memcpy(dst, &b, 4); return;
    } else if constexpr (std::is_same_v<T, int32_t> && W == 8) {
        const __m256i w = _mm256_packus_epi32((__m256i)static_cast<__v8si>(code), (__m256i)static_cast<__v8si>(code));
        const __m256i p = _mm256_permute4x64_epi64(w, 0x08);   // [lo words | hi words] in the low half
        const __m128i b = _mm_packus_epi16(_mm256_castsi256_si128(p), _mm256_castsi256_si128(p));
        _mm_storel_epi64((__m128i*)dst, b); return;
    } else if constexpr (std::is_same_v<T, int16_t> && W == 16) {
        const __m256i b = _mm256_packus_epi16((__m256i)static_cast<__v16hi>(code), (__m256i)static_cast<__v16hi>(code));
        const __m256i p = _mm256_permute4x64_epi64(b, 0x08);
        _mm_storeu_si128((__m128i*)dst, _mm256_castsi256_si128(p)); return;
    } else
#elif defined(__SSE2__)
    if constexpr (std::is_same_v<T, int64_t> && W == 2) {
        const __m128i d = _mm_shuffle_epi32((__m128i)static_cast<__v2di>(code), 0x08);
        const __m128i b = _mm_packus_epi16(_mm_packs_epi32(d, d), _mm_packs_epi32(d, d));
        std::memcpy(dst, &b, 2); return;
    } else if constexpr (std::is_same_v<T, int32_t> && W == 4) {
        const __m128i w = _mm_packs_epi32((__m128i)static_cast<__v4si>(code), (__m128i)static_cast<__v4si>(code));
        const __m128i b = _mm_packus_epi16(w, w);
        std::memcpy(dst, &b, 4); return;
    } else if constexpr (std::is_same_v<T, int16_t> && W == 8) {
        const __m128i b = _mm_packus_epi16((__m128i)static_cast<__v8hi>(code), (__m128i)static_cast<__v8hi>(code));
        _mm_storel_epi64((__m128i*)dst, b); return;
    } else
#endif
    stdx::static_simd_cast<bvec>(code).copy_to(dst, stdx::element_aligned);
}

// store_codes, or (nt, x86) the same W bytes through non-temporal stores: narrowed into a
// scratch, then movnti in 4- / 8-byte words (W = 2 lanes: a plain 2-byte store).  The
// caller fences (_mm_sfence) once the table is complete.
template <class T, int W>
[[gnu::always_inline]] static inline void store_codes_nt(const stdx::native_simd<T>& code, unsigned char* dst, bool nt) {
#if defined(__SSE2__) && defined(__x86_64__)
    if (nt && W >= 4) {
        alignas(64) unsigned char t[W];
        store_codes<T, W>(code, t);
        if constexpr (W == 4) { int v; std::memcpy(&v, t, 4); _mm_stream_si32(reinterpret_cast<int*>(dst), v); }
        else for (int k = 0; k < W; k += 8) {
            long long v; std::memcpy(&v, t + k, 8);
            _mm_stream_si64(reinterpret_cast<long long*>(dst + k), v);
        }
        return;
    }
#endif
    (void)nt;
    store_codes<T, W>(code, dst);
}

// The three-way argmax code of the scalar pick(): 0 when m >= x && m >= y, else 1 when
// x >= y, else 2 -- read off the three-way max the kernel computes anyway,
// mx = max(max(m, x), y): m >= x && m >= y is exactly mx == m (max is exact, -inf
// equals itself, there is no NaN, and == ignores the sign of a zero, so the order the
// max ran in cannot matter).  One compare where the chain took two and an AND (floating)
// or two and an OR (integer).  For an integer T the x/y choice is STRICT with the blend
// inverted -- there is no packed integer >=, and gcc built each one from vpminsw +
// vpcmpeqw (clang swaps the operands instead): (y > x ? 2 : 1) is (x >= y ? 1 : 2).
// `vone` / `vtwo` are the codes of x / y (X's are passed pre-shifted, 4 and 8).
// (always_inline: at AVX-512 gcc called these out of line for the 32-lane int16 vectors,
// four and two calls per segment, each returning 64 bytes through memory.)
template <class V>
[[gnu::always_inline]] static inline V pick_code_mx(const V& mx, const V& m, const V& x, const V& y,
                                                   const V& vzero, const V& vone, const V& vtwo) {
    using T = typename V::value_type;
    V k;
    if constexpr (nw_is_int<T>) { k = vone; stdx::where(y > x, k) = vtwo; }
    else                        { k = vtwo; stdx::where(x >= y, k) = vone; }
    stdx::where(mx == m, k) = vzero;
    return k;
}
// `dest` where a >= b takes `value`: strict form for integer lanes (the default and the
// value swapped), the >= form for floating.
template <class V>
[[gnu::always_inline]] static inline V sel_ge(const V& a, const V& b, const V& dest, const V& value) {
    using T = typename V::value_type;
    if constexpr (nw_is_int<T>) { V r = value; stdx::where(b > a, r) = dest; return r; }
    else                        { V r = dest;  stdx::where(a >= b, r) = value; return r; }
}

// Tr: a tracked problem (job.tr), as striped_affine_full.
// Dense: the dense codes (dense_codes_row_bytes, simd_levels.hpp: 0.75 B/cell) -- three
// planes of 2-bit codes, M's and X's stored from pass 1 (final there), Y's after the
// carry; each lane accumulates a plane's codes over four segments by Horner and stores one
// byte per lane a group.  The codes are the same choices; only where their bits go differs.
template <class T, bool Local, bool Tr = false, bool Dense = false>
static void striped_affine_full_ptr(ViterbiJob<T>& job) {
    static_assert(!Dense || sizeof(T) >= 4, "dense codes: lanes >= 32 bits");
    const bool nt = Dense && job.nt;   // non-temporal code stores (store_codes_nt)
    using vd = stdx::native_simd<T>;
    constexpr int W = (int)stdx::native_simd<T>::size();
    // the integer carry (int_pmax_carry) reads the opens and their codes from rows; the
    // floating one recomputes them from cM / cX
    constexpr bool kOpenRows = nw_is_int<T> && W >= 4;
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
    const std::size_t rowsz = (std::size_t)(seg + 1) * W;
    const std::size_t off   = (std::size_t)W;

    const std::size_t drow = Dense ? dense_codes_row_bytes(seg, W, 3) : rowsz;   // code bytes a row
    const std::size_t dsz = (std::size_t)(m + 1) * drow;
    if (buf.DM.size() < dsz) buf.DM.resize(dsz);   // the packed codes (the linear fill's table too)
    // rolling score rows: 2 x rowsz instead of (m+1) x rowsz -- the whole point; with the
    // opens, open codes, Q (integer T), y-code and M+X-code scratch and Local's padding
    // mask, twelve rows of one block (spread_rows, dp_buffer.hpp: no two in one L1 set,
    // and none in the set of another stream read every row)
    std::size_t rs; T* const rows = spread_rows<T>(buf.prows, rowsz, 12, rs);

    auto scol = [seg](int j) -> std::size_t {
        return (std::size_t)((j - 1) % seg) * W + (j - 1) / seg;
    };

    striped_profile_build<T, Tr>(buf.sprof, blk, nalpha, b, n, seg, W, tr);

    T* ov = rows + 6 * rs;
    T* ok = rows + 7 * rs;
    T* qb = rows + 8 * rs;
    T* yc = rows + 9 * rs;
    T* mx = rows + 11 * rs;

    T* pM = rows + 3 * rs; T* pX = rows + 4 * rs; T* pY = rows + 5 * rs;
    T* cM = rows;          T* cX = rows + rs;     T* cY = rows + 2 * rs;

    // ── row 0: scores into the rolling "previous" row, codes into the tables ──
    {
        pM[0] = 0; pX[0] = K_NINF; pY[0] = K_NINF;
        for (std::size_t k = 0; k < sw; ++k) {
            pM[off + k] = K_NINF; pX[off + k] = K_NINF; pY[off + k] = K_NINF;
        }
        unsigned char* d0 = buf.DM.data();
        d0[0] = 3;                                     // M 3, X 0, Y 0
        if constexpr (Dense) {   // planes: M all 3, X all 0, Y 2 (extend) but (0,1), which opens
            for (int q = 0; q < (seg + 3) / 4; ++q) {
                unsigned char* g = d0 + off + (std::size_t)q * 3 * W;
                std::memset(g, 0xFF, W); std::memset(g + W, 0, W); std::memset(g + 2 * W, 0xAA, W);
            }
            if (seg > 0) d0[off + 2 * W] &= (unsigned char)~(3u << (2 * (std::min(4, seg) - 1)));
        }
        for (int l = 0; l < W; ++l)
            for (int s = 0; s < seg; ++s) {
                const int j = l * seg + s + 1;
                const std::size_t k = (std::size_t)s * W + l;
                if (j <= n) {
                    if constexpr (Local) pM[off + k] = 0;
                    else                 pY[off + k] = -(go_a + j * ge_a);
                    // row 0 is a start (M 3, X 0); Y: (0,1) opens from M, the rest extend
                    if constexpr (!Dense) d0[off + k] = (j <= 1) ? 3 : 3 | 2 << 4;
                } else {
                    if constexpr (!Dense) d0[off + k] = 3 | 2 << 4;
                }
            }
    }

    vd vgo_a(go_a), vge_a(ge_a), vgo_b(go_b), vge_b(ge_b);
    const vd vzero(static_cast<T>(0)), vone(static_cast<T>(1)), vtwo(static_cast<T>(2));
    const vd vthree(static_cast<T>(3));
    const vd vfour(static_cast<T>(4)), veight(static_cast<T>(8)), vsixteen(static_cast<T>(16)), vthirtytwo(static_cast<T>(32));
    // the codes X and Y record: pre-shifted into the one-byte packing, or plain 0..2 (Dense)
    const vd kX1 = Dense ? vone : vfour, kX2 = Dense ? vtwo : veight;
    const vd kYX = Dense ? vone : vsixteen, kYE = Dense ? vtwo : vthirtytwo;
    T bM = 0, bX = K_NINF, bY = K_NINF;
    T best_local = 0; int best_i = 0, best_j = 0, best_tbl = 0;
    const T* pad = Local ? striped_pad_into(rows + 10 * rs, n, seg, W) : nullptr;

    for (int i = 1; i <= m; ++i) {
        if constexpr (Tr) {
            go_a = tr.go_a[i]; ge_a = tr.ge_a[i]; go_b = tr.go_b[i]; ge_b = tr.ge_b[i];
            vgo_a = vd(go_a); vge_a = vd(ge_a); vgo_b = vd(go_b); vge_b = vd(ge_b);
        }
        const T nbM = Local ? T(0) : K_NINF;
        T nbX;
        if constexpr (Tr) nbX = Local ? K_NINF : tr.bx0[i];
        else              nbX = Local ? K_NINF : -(go_b + i * ge_b);
        const T nbOpen = (std::max(nbM, nbX) - go_a) - ge_a;
        const T nbOpenK = ((nbM - go_a) - ge_a >= (nbX - go_a) - ge_a) ? T(0) : T(Dense ? 1 : 16);   // Y's open codes
        const T* sub = striped_row_profile<T, Tr>(buf.sprof.data(), blk, nalpha, a, b, n, seg, W, i, tr);

        unsigned char* dP = buf.DM.data() + (std::size_t)i * drow + off;
        cM[0] = nbM; cX[0] = nbX; cY[0] = K_NINF;
        // column 0: M 3, Y 0; X opens from M(0,0) at (1,0), extends below
        buf.DM.data()[(std::size_t)i * drow] = (i <= 1) ? 3 : 3 | 1 << 2;

        if (seg > 0) {
            // The diagonal predecessors: segment s-1 of the previous row, i.e. the up vectors
            // of the previous iteration, rotated in registers (each previous-row vector is
            // loaded once, not twice); at s = 0 the last segment, shifted one lane in.
            vd dgM, dgX, dgY;
            {
                vd lM, lX, lY;
                lM.copy_from(pM + off + (std::size_t)(seg - 1) * W, stdx::element_aligned);
                lX.copy_from(pX + off + (std::size_t)(seg - 1) * W, stdx::element_aligned);
                lY.copy_from(pY + off + (std::size_t)(seg - 1) * W, stdx::element_aligned);
                dgM = lane_shift_in(lM, bM);
                dgX = lane_shift_in(lX, bX);
                dgY = lane_shift_in(lY, bY);
            }
            vd accM = vzero, accX = vzero;   // Dense: this group's M / X codes so far
            for (int s = 0; s < seg; ++s) {
                vd sb; sb.copy_from(sub + (std::size_t)s * W, stdx::element_aligned);
                const vd mx3 = stdx::max(stdx::max(dgM, dgX), dgY);
                vd vmv = mx3 + sb;

                // M code: same >= chain as the scalar pick(), M last so it wins ties
                vd km = vtwo;
#if defined(__clang__) && defined(__AVX512F__) && !defined(NWGRAD_STD_SIMD_AVX512_MASK_OK)
                if constexpr (sizeof(T) == 8) {
                    km = avx512d_blend(avx512d_ge(dgX, dgY), km, vone);
                    km = avx512d_blend(avx512d_eq(mx3, dgM), km, vzero);
                } else {
                    km = pick_code_mx(mx3, dgM, dgX, dgY, vzero, vone, vtwo);
                }
#else
                km = pick_code_mx(mx3, dgM, dgX, dgY, vzero, vone, vtwo);
#endif
                if constexpr (Local) {
                    vmv = stdx::max(vmv, vzero);
#if defined(__clang__) && defined(__AVX512F__) && !defined(NWGRAD_STD_SIMD_AVX512_MASK_OK)
                    if constexpr (sizeof(T) == 8) {
                        // vmv <= vzero  <=>  vzero >= vmv
                        km = avx512d_blend(avx512d_ge(vzero, vmv), km, vthree);
                    } else if constexpr (nw_is_int<T>) {
                        stdx::where(vmv == vzero, km) = vthree;   // clamped >= 0: <= 0 is == 0
                    } else {
                        stdx::where(vmv <= vzero, km) = vthree;   // traceback stops here
                    }
#else
                    if constexpr (nw_is_int<T>) stdx::where(vmv == vzero, km) = vthree;   // clamped >= 0
                    else                        stdx::where(vmv <= vzero, km) = vthree;   // traceback stops here
#endif
                }
                vmv.copy_to(cM + off + (std::size_t)s * W, stdx::element_aligned);

                vd uM, uX, uY;
                uM.copy_from(pM + off + (std::size_t)s * W, stdx::element_aligned);
                uX.copy_from(pX + off + (std::size_t)s * W, stdx::element_aligned);
                uY.copy_from(pY + off + (std::size_t)s * W, stdx::element_aligned);
                dgM = uM; dgX = uX; dgY = uY;   // the next segment's diagonal
                const vd ax = (uM - vgo_b) - vge_b, bx = uX - vge_b, cx = (uY - vgo_b) - vge_b;
                vd vxv = stdx::max(stdx::max(ax, bx), cx);
                vd kx = kX2;                        // X's code: 0 / 4 / 8 (Dense 0 / 1 / 2)
#if defined(__clang__) && defined(__AVX512F__) && !defined(NWGRAD_STD_SIMD_AVX512_MASK_OK)
                if constexpr (sizeof(T) == 8) {
                    kx = avx512d_blend(avx512d_ge(bx, cx), kx, kX1);
                    kx = avx512d_blend(avx512d_eq(vxv, ax), kx, vzero);
                } else {
                    kx = pick_code_mx(vxv, ax, bx, cx, vzero, kX1, kX2);
                }
#else
                kx = pick_code_mx(vxv, ax, bx, cx, vzero, kX1, kX2);
#endif
                vxv.copy_to(cX + off + (std::size_t)s * W, stdx::element_aligned);
                if constexpr (Dense) {   // M and X are final: their planes now (4*acc + k)
                    accM = accM + accM; accM = accM + accM; accM = accM + km;
                    accX = accX + accX; accX = accX + accX; accX = accX + kx;
                    if ((s & 3) == 3 || s == seg - 1) {
                        unsigned char* g = dP + (std::size_t)(s >> 2) * 3 * W;
                        store_codes_nt<T, W>(accM, g, nt); store_codes_nt<T, W>(accX, g + W, nt);
                        accM = vzero; accX = vzero;
                    }
                } else {
                    (km + kx).copy_to(mx + (std::size_t)s * W, stdx::element_aligned);
                }

                // The open's source is chosen on the ROUNDED candidates, as the scalar pick(uy, vy, wy)
                // does: M and X values that differ can round to the same open, and then M wins.
                // Comparing vmv >= vxv before subtracting picked X there (float32, a path moved).
                // max(om, ox) is bitwise max(vmv, vxv) - go - ge: rounding is monotone.
                // (Floating T: the carry recomputes both from cM / cX instead -- the same bits --
                // which saves two T-typed rows, a store and a load each, per segment.)
                if constexpr (kOpenRows) {
                    const vd om = (vmv - vgo_a) - vge_a, ox = (vxv - vgo_a) - vge_a;
                    stdx::max(om, ox).copy_to(ov + (std::size_t)s * W, stdx::element_aligned);
                    vd okv = kYX;                  // open from X: Y's code 1, shifted
#if defined(__clang__) && defined(__AVX512F__) && !defined(NWGRAD_STD_SIMD_AVX512_MASK_OK)
                    if constexpr (sizeof(T) == 8) {
                        okv = avx512d_blend(avx512d_ge(om, ox), okv, vzero);
                    } else {
                        okv = sel_ge(om, ox, kYX, vzero);
                    }
#else
                    okv = sel_ge(om, ox, kYX, vzero);
#endif
                    okv.copy_to(ok + (std::size_t)s * W, stdx::element_aligned);
                }
            }

            // ── the carry: VY, same-lane striped chain + lazy-F, codes tracked (integer
            // T: the exact prefix max with its codes, int_pmax_carry) ──
            if constexpr (kOpenRows) {
                int_pmax_carry<T, W, false, Dense ? 2 : 32>(ov, ok, nbOpen, nbOpenK, bY, ge_a, seg,
                                     cY + off, yc, qb);
            } else {
            // The open into Y and its source, from segment q of this row's M and X: what pass 1
            // once stored as ov / ok, recomputed with the same operations in the same order.
            // The open's source is chosen on the ROUNDED candidates (see pass 1).
            auto open_from = [&](int q, vd& O, vd& OK) __attribute__((always_inline)) {
                vd vm, vx;
                vm.copy_from(cM + off + (std::size_t)q * W, stdx::element_aligned);
                vx.copy_from(cX + off + (std::size_t)q * W, stdx::element_aligned);
                const vd om = (vm - vgo_a) - vge_a, ox = (vx - vgo_a) - vge_a;
                O = stdx::max(om, ox);
                OK = kYX;
#if defined(__clang__) && defined(__AVX512F__) && !defined(NWGRAD_STD_SIMD_AVX512_MASK_OK)
                if constexpr (sizeof(T) == 8) OK = avx512d_blend(avx512d_ge(om, ox), OK, vzero);
                else                          OK = sel_ge(om, ox, kYX, vzero);
#else
                OK = sel_ge(om, ox, kYX, vzero);
#endif
            };
            vd prev([&](int q) { return q == 0 ? bY : K_NINF; });
            for (int s = 0; s < seg; ++s) {
                vd O, OK;
                if (s == 0) {
                    vd lo, lk;
                    open_from(seg - 1, lo, lk);
                    O  = lane_shift_in(lo, nbOpen);
                    OK = lane_shift_in(lk, nbOpenK);
                } else {
                    open_from(s - 1, O, OK);
                }
                const vd ext = prev - vge_a;
                vd v = stdx::max(O, ext);
                vd ky = kYE;                       // extension
#if defined(__clang__) && defined(__AVX512F__) && !defined(NWGRAD_STD_SIMD_AVX512_MASK_OK)
                if constexpr (sizeof(T) == 8) {
                    ky = avx512d_blend(avx512d_ge(O, ext), ky, OK);   // open: M or X, ties go to open
                } else {
                    ky = sel_ge(O, ext, kYE, OK);      // open: M or X, ties go to open
                }
#else
                ky = sel_ge(O, ext, kYE, OK);      // open: M or X, ties go to open
#endif
                v.copy_to(cY + off + (std::size_t)s * W, stdx::element_aligned);
                ky.copy_to(yc + (std::size_t)s * W, stdx::element_aligned);
                prev = v;
            }
            for (int r = 0; r < W; ++r) {
                vd last; last.copy_from(cY + off + (std::size_t)(seg - 1) * W, stdx::element_aligned);
                vd F = lane_shift_in(last, bY);
                F = F - vge_a;
                bool changed = false;
                for (int s = 0; s < seg; ++s) {
                    vd v; v.copy_from(cY + off + (std::size_t)s * W, stdx::element_aligned);
#if defined(__clang__) && defined(__AVX512F__) && !defined(NWGRAD_STD_SIMD_AVX512_MASK_OK)
                    if constexpr (sizeof(T) == 8) { if (!avx512d_any_gt(F, v)) break; }
                    else { if (!stdx::any_of(F > v)) break; }
#else
                    if (!stdx::any_of(F > v)) break;
#endif
                    // A cell lazy-F raises came from an EXTENSION -- its recorded code
                    // must be revised, or B disagrees with A exactly on the cells this
                    // correction exists to fix.
                    vd k; k.copy_from(yc + (std::size_t)s * W, stdx::element_aligned);
#if defined(__clang__) && defined(__AVX512F__) && !defined(NWGRAD_STD_SIMD_AVX512_MASK_OK)
                    if constexpr (sizeof(T) == 8) {
                        k = avx512d_blend(avx512d_gt(F, v), k, kYE);
                    } else {
                        stdx::where(F > v, k) = kYE;
                    }
#else
                    stdx::where(F > v, k) = kYE;
#endif
                    k.copy_to(yc + (std::size_t)s * W, stdx::element_aligned);
                    v = stdx::max(v, F);
                    v.copy_to(cY + off + (std::size_t)s * W, stdx::element_aligned);
                    F = v - vge_a;
                    changed = true;
                }
                if (!changed) break;
            }
            }   // integer / floating carry
            if constexpr (Dense) {           // Y's plane
                vd accY = vzero;
                for (int s = 0; s < seg; ++s) {
                    vd k; k.copy_from(yc + (std::size_t)s * W, stdx::element_aligned);
                    accY = accY + accY; accY = accY + accY; accY = accY + k;
                    if ((s & 3) == 3 || s == seg - 1) {
                        store_codes_nt<T, W>(accY, dP + (std::size_t)(s >> 2) * 3 * W + 2 * W, nt);
                        accY = vzero;
                    }
                }
            } else
            for (int s = 0; s < seg; ++s) {   // M | X << 2 | Y << 4 (<= 43: the packs stay exact)
                vd k, kmx;
                k.copy_from(yc + (std::size_t)s * W, stdx::element_aligned);
                kmx.copy_from(mx + (std::size_t)s * W, stdx::element_aligned);
                store_codes<T, W>(kmx + k, dP + (std::size_t)s * W);
            }

            if constexpr (Local)
                striped_row_best<T>(cM + off, cX + off, cY + off, pad, seg, i,
                                    best_local, best_i, best_j, best_tbl);
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
#if defined(__SSE2__) && defined(__x86_64__)
    if (nt) _mm_sfence();   // the non-temporal code stores complete before the walk reads them
#endif
    job.table_layout = 1;
    job.seg = seg;
    job.width = W;
}

template <class T>
static void striped_full_ptr_entry(ViterbiJob<T>& job) {
    if constexpr (sizeof(T) >= 4) {
        if (job.dense) {
            if (job.tr.on()) {
                if (job.align_mode) striped_affine_full_ptr<T, true, true, true>(job);
                else                striped_affine_full_ptr<T, false, true, true>(job);
            } else {
                if (job.align_mode) striped_affine_full_ptr<T, true, false, true>(job);
                else                striped_affine_full_ptr<T, false, false, true>(job);
            }
            return;
        }
    }
    job.dense = 0;
    if (job.tr.on()) {
        if (job.align_mode) striped_affine_full_ptr<T, true, true>(job);
        else                striped_affine_full_ptr<T, false, true>(job);
        return;
    }
    if (job.align_mode) striped_affine_full_ptr<T, true>(job);
    else                striped_affine_full_ptr<T, false>(job);
}
