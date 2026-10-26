// Grid forward kernel: nwgrad's score_affine_striped_pos (Global, untracked) over a
// rectangle with seeded borders, saving grid rows and segment-aligned grid columns.
// #include'd inside a level namespace by gk_level_*.cpp (compiled with that level's -m flags).
//
// Every cell is computed by the same expressions as score_affine_striped_pos, hence (as
// that kernel argues) bit-identical to the scalar three-state fill.  The seeds:
//   * the previous row at r0: D = max(M, X, Y) and X of the top line;
//   * column c0 of row i: the diagonal operand of lane 0 is D(i-1, c0) (bD), the open
//     into column c0+1 is (max(M, X)(i, c0) - go_a) - ge_a, and the extension is
//     Y(i, c0) - ge_a (lane 0 of Yin).
// M is not kept by the D/X form; at a grid line it is recomputed as D(i-1, j-1) + s,
// the very sum the main pass formed.

template <class V>
static inline V gk_lane_shift_in(V x, typename V::value_type b) {
#if defined(__AVX512F__)
    using T = typename V::value_type;
    if constexpr (std::is_same_v<T, double> && V::size() == 8) {
        const __m512i r = _mm512_alignr_epi64(_mm512_castpd_si512(static_cast<__m512d>(x)),
                                              _mm512_castpd_si512(_mm512_set1_pd(b)), 7);
        return V(_mm512_castsi512_pd(r));
    } else if constexpr (std::is_same_v<T, float> && V::size() == 16) {
        const __m512i r = _mm512_alignr_epi32(_mm512_castps_si512(static_cast<__m512>(x)),
                                              _mm512_castps_si512(_mm512_set1_ps(b)), 15);
        return V(_mm512_castsi512_ps(r));
    } else
#elif defined(__AVX2__)
    using T = typename V::value_type;
    if constexpr (std::is_same_v<T, double> && V::size() == 4) {
        const __m256d r = _mm256_permute4x64_pd(static_cast<__m256d>(x), 0x90);
        return V(_mm256_blend_pd(r, _mm256_set1_pd(b), 1));
    } else if constexpr (std::is_same_v<T, float> && V::size() == 8) {
        const __m256 r = _mm256_permutevar8x32_ps(static_cast<__m256>(x),
                                                  _mm256_setr_epi32(0, 0, 1, 2, 3, 4, 5, 6));
        return V(_mm256_blend_ps(r, _mm256_set1_ps(b), 1));
    } else
#endif
    return V([&](int q) { return q == 0 ? b : x[q - 1]; });
}

#ifdef GK_COUNT
inline long long gk_rows = 0, gk_main = 0, gk_rounds = 0, gk_visits = 0;
#endif
// Variants (all bit-identical; see the notes at each use):
//   Two: the gap-in-A carry advances two segments per dependent step, in the main pass
//        and in lazy-F.  Rounding is monotone, so max(a, b) - g == max(a - g, b - g) bit
//        for bit; hence Y(s+1) = max(O(s), Y(s) - g) = max(O(s), O(s-1) - g, (Y(s-1) - g) - g),
//        whose chain is two subtractions and ONE max per two segments (was two of each).
//   MX:  the main pass stores max(M, X) instead of D = max(M, X, Y), keeps the previous
//        row's Y, and forms D = max(MX, Y) when it reads the row.  Lazy-F then touches only
//        the Y row (one load, one store per visit instead of two each).
// An empty asm on the value: gcc re-associates a max of maxes (max is exact, so it may),
// and in the Two carries it folded the off-chain max(P, chain) back into max(max(chain,
// a), b) -- two maxes on the dependency chain again, undoing the point of Two.  Pinning P
// keeps it one value.  clang keeps the order written; the barrier is gcc only.
template <class V>
static inline V gk_pin(V v) {
#if defined(__GNUC__) && !defined(__clang__) && (defined(__x86_64__) || defined(__aarch64__))
    using T = typename V::value_type;
    constexpr int B = (int)(V::size() * sizeof(T));
#  if defined(__x86_64__)
    if constexpr (B == 16 || B == 32 || B == 64) {
        using N = std::conditional_t<std::is_same_v<T, float>,
                    std::conditional_t<B == 16, __m128, std::conditional_t<B == 32, __m256, __m512>>,
                    std::conditional_t<B == 16, __m128d, std::conditional_t<B == 32, __m256d, __m512d>>>;
        N r = static_cast<N>(v);
        __asm__("" : "+v"(r));
        return V(r);
    }
#  endif
#endif
    return v;
}

// gcc does not inline these lambdas on its own here (it called cell/put/Dp out of line:
// 1.4-2.3x slower at AVX2); clang does.
#define GK_INL __attribute__((always_inline))
//   Sub: lazy-F carries F - g instead of max(c, F) - g (see the note there): one
//        subtraction per segment on the chain, no max.
template <class T, bool Two, bool MX, bool Sub = false>
static void grid_fwd(GridFwdJob<T>& J) {
    using vd = stdx::native_simd<T>;
    const int W = (int)vd::size();
    J.W = W;
    const T NINF = -std::numeric_limits<T>::infinity();
    const int h = J.r1 - J.r0, w = J.c1 - J.c0, nalpha = J.nalpha;
    const int seg = (w + W - 1) / W;
    const std::size_t sw = (std::size_t)seg * W;
    const T go_a = J.go_a, ge_a = J.ge_a, go_b = J.go_b, ge_b = J.ge_b;

    // 64-byte aligned, as nwgrad's DpBuffer: std::vector's 16-byte base made 32-byte
    // loads split cache lines (1.15-1.22x at float32, n >= 2500, nighthaven).
    using AV = std::vector<T, AlignedAllocator<T>>;
    thread_local AV v0, v1, v2, v3, v4, v5, prof;
    thread_local std::vector<unsigned char> built;
    for (auto* v : {&v0, &v1, &v2, &v3, &v4, &v5}) if (v->size() < sw) v->resize(sw);
    if (prof.size() < (std::size_t)nalpha * sw) prof.resize((std::size_t)nalpha * sw);
    built.assign((std::size_t)nalpha, 0);
    // A = D (or max(M, X) under MX); pY is read only under MX
    T* pA = v0.data(); T* pX = v1.data(); T* pY = v5.data();
    T* cA = v2.data(); T* cX = v3.data(); T* cY = v4.data();
    const unsigned char* bb = J.b + J.c0;   // column c0 + j reads bb[j - 1]

    // top row (r0), striped; padding -inf
    for (int l = 0; l < W; ++l)
        for (int s = 0; s < seg; ++s) {
            const int j = l * seg + s + 1;
            const std::size_t k = (std::size_t)s * W + l;
            if (j <= w) {
                pA[k] = MX ? std::max(J.top(0, j), J.top(1, j)) : std::max({J.top(0, j), J.top(1, j), J.top(2, j)});
                pX[k] = J.top(1, j);
                pY[k] = J.top(2, j);
            } else { pA[k] = NINF; pX[k] = NINF; pY[k] = NINF; }
        }
    T bD = std::max({J.top(0, 0), J.top(1, 0), J.top(2, 0)});

    // column sets: set t sits at segment st[t]; its lanes are columns c0 + l*seg + st + 1
    int q = std::max(1, std::min(J.q, seg));
    J.q = q;
    int st[256];
    for (int t = 0; t < q; ++t) {
        st[t] = (int)(((long long)(t + 1) * seg) / q) - 1;
        for (int l = 0; l < W; ++l) {
            const int c = J.c0 + l * seg + st[t] + 1;
            J.colid[t * W + l] = c < J.c1 ? c : -1;
            for (int s3 = 0; s3 < 3; ++s3)   // row r0 of each column line, from the top line
                J.cols[(((std::size_t)t * 3 + s3) * (h + 1)) * W + l] = c <= J.c1 ? J.top(s3, c - J.c0) : NINF;
        }
    }

    const vd vgo_a(go_a), vge_a(ge_a), vgo_b(go_b), vge_b(ge_b), vninf(NINF);
    // clang + libstdc++ std::simd cannot form an AVX-512 double mask (a static_assert in
    // simd_x86.h); compare through the intrinsic there, as nwgrad's avx512d_gt does.
    auto any_gt = [](const vd& x, const vd& y) GK_INL -> bool {
#if defined(__clang__) && defined(__AVX512F__)
        if constexpr (std::is_same_v<T, double>) {
            alignas(64) double xa[8], ya[8];
            x.copy_to(xa, stdx::element_aligned); y.copy_to(ya, stdx::element_aligned);
            return _mm512_cmp_pd_mask(_mm512_load_pd(xa), _mm512_load_pd(ya), _CMP_GT_OQ) != 0;
        } else
#endif
        return stdx::any_of(x > y);
    };
    auto ld = [](const T* p) GK_INL { vd v; v.copy_from(p, stdx::element_aligned); return v; };
    auto st_ = [](const vd& v, T* p) GK_INL { v.copy_to(p, stdx::element_aligned); };
    // D of the previous row, segment s
    auto Dp = [&](int s) GK_INL {
        if constexpr (MX) return stdx::max(ld(pA + (std::size_t)s * W), ld(pY + (std::size_t)s * W));
        else              return ld(pA + (std::size_t)s * W);
    };

    for (int ii = 1; ii <= h; ++ii) {
        const int i = J.r0 + ii;
        const T lM = J.left(0, ii), lX = J.left(1, ii), lY = J.left(2, ii);
        const T nbOpen = (std::max(lM, lX) - go_a) - ge_a;
        const unsigned char ai = J.a[i - 1];
        T* sub = prof.data() + (std::size_t)ai * sw;
        if (!built[ai]) {
            built[ai] = 1;
            const T* row = J.blk + (std::size_t)ai * nalpha;
            for (int l = 0; l < W; ++l)
                for (int s = 0; s < seg; ++s) {
                    const int j = l * seg + s + 1;
                    sub[(std::size_t)s * W + l] = (j <= w) ? row[bb[j - 1]] : NINF;
                }
        }

        // ── main pass ──
        vd Oin([&](int qq) { return qq == 0 ? nbOpen : NINF; });   // O of the segment before
        vd Yin([&](int qq) { return qq == 0 ? lY : NINF; });       // Y of the segment before
        vd dD = gk_lane_shift_in(Dp(seg - 1), bD);                 // diagonal operand of s
        // carry-free part of segment s: stores X and A, returns mx = max(M, X)
        auto cell = [&](int s, vd& mx) GK_INL {
            const vd uD = Dp(s);
            const vd vmv = dD + ld(sub + (std::size_t)s * W);
            const vd vxv = stdx::max((uD - vgo_b) - vge_b, ld(pX + (std::size_t)s * W) - vge_b);
            mx = stdx::max(vmv, vxv);
            st_(vxv, cX + (std::size_t)s * W);
            dD = uD;
        };
        int s = 0;
        if constexpr (Two) {
            for (; s + 1 < seg; s += 2) {
                vd mx0, mx1;
                cell(s, mx0);
                cell(s + 1, mx1);
                const vd O0 = (mx0 - vgo_a) - vge_a;               // Oin of s + 1
                const vd t = Yin - vge_a;                           // chain
                const vd y0 = stdx::max(Oin, t);
                const vd P = gk_pin(stdx::max(O0, Oin - vge_a));    // off the chain
                const vd y1 = stdx::max(P, t - vge_a);              // chain: sub, max
                st_(y0, cY + (std::size_t)s * W);
                st_(y1, cY + (std::size_t)(s + 1) * W);
                st_(MX ? mx0 : stdx::max(mx0, y0), cA + (std::size_t)s * W);
                st_(MX ? mx1 : stdx::max(mx1, y1), cA + (std::size_t)(s + 1) * W);
                Oin = (mx1 - vgo_a) - vge_a;
                Yin = y1;
            }
        }
        for (; s < seg; ++s) {
            vd mx;
            cell(s, mx);
            const vd vyv = stdx::max(Oin, Yin - vge_a);
            st_(vyv, cY + (std::size_t)s * W);
            st_(MX ? mx : stdx::max(mx, vyv), cA + (std::size_t)s * W);
            Oin = (mx - vgo_a) - vge_a;
            Yin = vyv;
        }
        const vd Olast = Oin;
#ifdef GK_COUNT
        gk_rows++; gk_main += seg;
#endif
        // ── lazy-F ──
        // One visit: v = max(c, F) stored (D too, without MX), F' = v - g.  Two: F' and F''
        // as max(c - g, F - g) and max(c1 - g, (c - g) - g, (F - g) - g), the same bits.
        auto put = [&](int s2, const vd& v) GK_INL {
            st_(v, cY + (std::size_t)s2 * W);
            if constexpr (!MX) st_(stdx::max(ld(cA + (std::size_t)s2 * W), v), cA + (std::size_t)s2 * W);
        };
        for (int r = 0; r < W; ++r) {
#ifdef GK_COUNT
            gk_rounds++;
#endif
            vd F = gk_lane_shift_in(ld(cY + (std::size_t)(seg - 1) * W), NINF) - vge_a;
            if (r == 0) F = stdx::max(gk_lane_shift_in(Olast, NINF), F);
            bool changed = false;
            int s2 = 0;
            if constexpr (Sub) {
                // The row already satisfies Y[s+1] >= Y[s] - g in every lane (the main pass
                // computed it so, and lazy-F only raises a cell when it then visits the next),
                // so the carry into s+1, max(c[s], F) - g, may be taken as F - g: where c[s]
                // wins, c[s] - g <= c[s+1] and neither can raise s+1.  The stored maxima and
                // the exit test come out the same; the chain is one subtraction per segment.
                for (; s2 < seg; ++s2) {
                    const vd v = ld(cY + (std::size_t)s2 * W);
#ifdef GK_COUNT
                    gk_visits++;
#endif
                    if (!any_gt(F, v)) goto done;
                    put(s2, stdx::max(v, F));
                    F = F - vge_a;
                    changed = true;
                }
            }
            if constexpr (Two) {
                for (; s2 + 1 < seg; s2 += 2) {
                    const vd c0 = ld(cY + (std::size_t)s2 * W);
#ifdef GK_COUNT
                    gk_visits++;
#endif
                    if (!any_gt(F, c0)) goto done;
                    changed = true;
                    const vd c1 = ld(cY + (std::size_t)(s2 + 1) * W);
                    const vd t = F - vge_a;                         // chain
                    const vd c0g = c0 - vge_a;
                    const vd F1 = stdx::max(c0g, t);                // = max(c0, F) - g
                    put(s2, stdx::max(c0, F));
#ifdef GK_COUNT
                    gk_visits++;
#endif
                    if (!any_gt(F1, c1)) goto done;
                    put(s2 + 1, stdx::max(c1, F1));
                    const vd P = gk_pin(stdx::max(c1 - vge_a, c0g - vge_a));       // off the chain
                    F = stdx::max(P, t - vge_a);                                    // chain: sub, max
                }
            }
            for (; s2 < seg; ++s2) {
                const vd v = ld(cY + (std::size_t)s2 * W);
#ifdef GK_COUNT
                gk_visits++;
#endif
                if (!any_gt(F, v)) break;
                const vd nv = stdx::max(v, F);
                put(s2, nv);
                F = nv - vge_a;
                changed = true;
            }
        done:
            if (!changed) break;
        }

        auto Mat = [&](int s3) GK_INL {   // M of segment s3 in this row, as the main pass formed it
            const vd d = s3 == 0 ? gk_lane_shift_in(Dp(seg - 1), bD) : Dp(s3 - 1);
            return d + ld(sub + (std::size_t)s3 * W);
        };
        for (int t = 0; t < q; ++t) {
            const int s3 = st[t];
            T* base = J.cols + ((std::size_t)t * 3 * (h + 1) + ii) * W;
            const std::size_t ss = (std::size_t)(h + 1) * W;
            st_(Mat(s3), base);
            st_(ld(cX + (std::size_t)s3 * W), base + ss);
            st_(ld(cY + (std::size_t)s3 * W), base + 2 * ss);
        }
        if (ii % J.R == 0 && ii < h) {
            T* row = J.rows + (std::size_t)(ii / J.R - 1) * 3 * (w + 1);
            row[0] = lM; row[w + 1] = lX; row[2 * (w + 1)] = lY;
            for (int s3 = 0; s3 < seg; ++s3) {
                const vd M = Mat(s3);
                for (int l = 0; l < W; ++l) {
                    const int j = l * seg + s3 + 1;
                    if (j > w) continue;
                    const std::size_t k = (std::size_t)s3 * W + l;
                    row[j] = M[l]; row[w + 1 + j] = cX[k]; row[2 * (w + 1) + j] = cY[k];
                }
            }
        }
        std::swap(pA, cA); std::swap(pX, cX); std::swap(pY, cY);
        bD = std::max({lM, lX, lY});
    }
}

template <class T, bool Two, bool MX, bool Sub = false> static void grid_fwd_e(GridFwdJob<T>& J) { grid_fwd<T, Two, MX, Sub>(J); }
#ifdef GK_COUNT
struct GkReport { ~GkReport() { if (gk_rows) std::fprintf(stderr, "lazy-F: %.2f rounds/row, %.3f segment visits per main-loop segment\n", (double)gk_rounds / gk_rows, (double)gk_visits / gk_main); } };
inline GkReport gk_report;
#endif
