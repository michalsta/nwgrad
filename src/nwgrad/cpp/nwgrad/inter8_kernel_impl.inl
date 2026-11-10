// ── Inter-pair SCORE ONLY in saturating int8 lanes ───────────────────────────────
//
// Included by level_common.inc in the level's inter TU, after inter_kernel_impl.inl (IW,
// ivmax, ivblend, kByteShuffle).  One pair per lane, as inter_score_affine_pos /
// inter_score_linear, in int8: 8 * IW lanes (16 at SSE and NEON, 32 AVX2, 64 AVX-512BW),
// twice int16's, four times int32's.  For the batch's short pairs (miRNA x target site,
// ~22 x 50), where the per-pair striped int8 kernel loses in Global / Glocal (lazy-F across
// every lane, every row) and the inter-pair kernel has no lazy-F at all.
//
// The caller (BatchEngine::score_only_) proves each lane's result as the striped int8 kernel
// does (score8_kernel_impl.inl, its head: the same argument cell by cell, per lane; S the
// stored value, V the exact one, every operation a saturating add / subtract or a max):
//   * Local.  M = max(D + s, 0) >= 0 and X / Y only subtract, so a lane's cells are exact
//     unless an add clamped at 127, which leaves that cell at 127.  The lane's maximum
//     over its own cells (rows <= m, columns <= n_l: `live` and the ragged mask; the cells
//     past them never feed its own, the DP flowing right and down) < 127 => exact,
//     == 127 => overflow, and the caller re-runs the pair wider.
//   * Global / Glocal.  S >= V and S(c) <= max(V(c), -128 + G), G = min(m, n) * smax_pos
//     <= 126 (the caller's a-priori precondition, per lane): the score is exact iff
//     S > G - 128.  The borders -(go + k * ge) are clamped to -128 when they pass it,
//     which is S = max(V, -128), inside the same bound.
// Gaps >= 0 only, with go + ge <= 127 per side, so (v - go) - ge saturates to the same
// value as v - (go + ge): one subtraction per open (as the striped int8 kernel).  With
// both >= 0 the affine form is inter_score_affine_pos's (two stored rows, D and X; the
// Local maximum over M alone).  Untracked, alphabets of at most 8 letters (a byte-shuffle
// table up to 4 letters at a level with a shuffle, else a blend tree).

#ifndef NWGRAD_LEVEL_NS
#  error "inter8_kernel_impl.inl is included from a level TU (level_common.inc); not standalone"
#endif

inline constexpr int I8W = 8 * IW;   // int8 lanes
typedef int8_t iv8 __attribute__((vector_size(I8W)));

template <class V> [[gnu::always_inline]] static inline V iv8_adds(V a, V b) noexcept {
#if defined(__AVX512BW__)
    if constexpr (sizeof(V) == 64) return (V)_mm512_adds_epi8((__m512i)a, (__m512i)b);
#endif
#if defined(__AVX2__)
    if constexpr (sizeof(V) == 32) return (V)_mm256_adds_epi8((__m256i)a, (__m256i)b);
#endif
#if defined(__SSE2__)
    if constexpr (sizeof(V) == 16) return (V)_mm_adds_epi8((__m128i)a, (__m128i)b);
#endif
#if defined(__ARM_NEON)
    if constexpr (sizeof(V) == 16) return (V)vqaddq_s8((int8x16_t)a, (int8x16_t)b);
#endif
    V r;
    for (unsigned k = 0; k < sizeof(V); ++k) r[k] = static_cast<int8_t>(std::clamp(int(a[k]) + int(b[k]), -128, 127));
    return r;
}
template <class V> [[gnu::always_inline]] static inline V iv8_subs(V a, V b) noexcept {
#if defined(__AVX512BW__)
    if constexpr (sizeof(V) == 64) return (V)_mm512_subs_epi8((__m512i)a, (__m512i)b);
#endif
#if defined(__AVX2__)
    if constexpr (sizeof(V) == 32) return (V)_mm256_subs_epi8((__m256i)a, (__m256i)b);
#endif
#if defined(__SSE2__)
    if constexpr (sizeof(V) == 16) return (V)_mm_subs_epi8((__m128i)a, (__m128i)b);
#endif
#if defined(__ARM_NEON)
    if constexpr (sizeof(V) == 16) return (V)vqsubq_s8((int8x16_t)a, (int8x16_t)b);
#endif
    V r;
    for (unsigned k = 0; k < sizeof(V); ++k) r[k] = static_cast<int8_t>(std::clamp(int(a[k]) - int(b[k]), -128, 127));
    return r;
}
// r[k] = t[idx[k]] for idx[k] in 0..15 (the 16-byte table repeated in every 128-bit lane).
// (A template: in a plain function the discarded widths' branches would still be checked.)
template <class V> [[gnu::always_inline]] static inline V iv8_shuf(isb16 t, V idx) noexcept {
#if defined(__AVX512BW__)
    if constexpr (sizeof(V) == 64)
        return (V)_mm512_shuffle_epi8(_mm512_broadcast_i32x4((__m128i)t), (__m512i)idx);
#endif
#if defined(__AVX2__)
    if constexpr (sizeof(V) == 32)
        return (V)_mm256_shuffle_epi8(_mm256_broadcastsi128_si256((__m128i)t), (__m256i)idx);
#endif
#if defined(__SSE4_1__)
    if constexpr (sizeof(V) == 16) return (V)_mm_shuffle_epi8((__m128i)t, (__m128i)idx);
#elif defined(__ARM_NEON)
    if constexpr (sizeof(V) == 16) return (V)vqtbl1q_s8((int8x16_t)t, (uint8x16_t)idx);
#endif
    V r{};   // levels without a shuffle never run this (kByteShuffle); it only compiles
    for (unsigned k = 0; k < sizeof(V); ++k) r[k] = t[idx[k] & 15];
    return r;
}
// Every lane = x, x already clamped to int8.
static inline iv8 iv8_splat(int x) noexcept {
    iv8 v;
    for (int l = 0; l < I8W; ++l) v[l] = static_cast<int8_t>(x);
    return v;
}
static inline int8_t sat8(long long v) noexcept { return static_cast<int8_t>(std::clamp<long long>(v, -128, 127)); }

// The lookup: Lv = kLvLut (one shuffle of the 4x4 table by 4a + b per vector cell) or a
// blend tree of 2 / 3 levels on the B residue's bits (inter_sub's shape, int8 profile).
template <int Lv>
struct Inter8Common {
    const InterJobT<int8_t>& J;
    std::vector<iv8>& colb;    // Lut: per column each lane's B code; tree: Lv bit masks per column
    std::vector<iv8>& rowa;    // per row each lane's A code (Lut: 4a), 0 past its A
    isb16 tbl{};               // Lut: t[4a + b]
    isb16 colt[8] = {};        // tree with a shuffle: colt[c][a] = score(a, c)
    iv8 mv{};                  // each lane's m (live = i <= m), J.M <= 127 checked by the caller
    iv8 nbv{};                 // ragged: each lane's B length
    int ord[I8W];              // lanes by m, stable (Global / Glocal capture)
    explicit Inter8Common(const InterJobT<int8_t>& job) : J(job), colb(buf<0>()), rowa(buf<1>()) {
        const int n = J.n, M = J.M, na = J.nalpha;
        const size_t nc = Lv == kLvLut ? 1 : static_cast<size_t>(Lv);
        if (colb.size() < nc * (n + 1)) colb.resize(nc * (n + 1));
        if (rowa.size() < static_cast<size_t>(M + 1)) rowa.resize(M + 1);
        // B codes transposed first (byte stores), then read back whole: the bit masks from
        // a whole vector (inter_kernel_impl.inl's InterCommon, the same store-forwarding
        // reason).
        for (int j = 1; j <= n; ++j) {
            iv8 v;
            for (int l = 0; l < I8W; ++l)
                v[l] = static_cast<int8_t>((!J.nb || j <= J.nb[l]) ? J.b[l][j - 1] : 0);   // ragged: pad
            colb[nc * j] = v;
        }
        if constexpr (Lv != kLvLut)
            for (int j = 1; j <= n; ++j) {
                const iv8 v = colb[nc * j];
                for (int k = Lv - 1; k >= 0; --k)   // bit k as 0 / -1 (k = 0 last: it overwrites v)
                    colb[nc * j + k] = -((v >> k) & 1);
            }
        const int mul = Lv == kLvLut ? 4 : 1;
        for (int i = 1; i <= M; ++i) {
            iv8 v;
            for (int l = 0; l < I8W; ++l) v[l] = static_cast<int8_t>(mul * ((i <= J.m[l]) ? J.a[l][i - 1] : 0));
            rowa[i] = v;
        }
        if constexpr (Lv == kLvLut) {
            for (int a = 0; a < na; ++a)
                for (int b = 0; b < na; ++b) tbl[4 * a + b] = J.blk[a * na + b];
        } else {
            for (int c = 0; c < (1 << Lv); ++c)
                for (int a = 0; a < na; ++a) colt[c][a] = J.blk[a * na + (c < na ? c : 0)];
        }
        for (int l = 0; l < I8W; ++l) {
            mv[l] = static_cast<int8_t>(J.m[l]);
            nbv[l] = static_cast<int8_t>(J.nb ? J.nb[l] : n);
            int k = l;
            while (k > 0 && J.m[ord[k - 1]] > J.m[l]) { ord[k] = ord[k - 1]; --k; }
            ord[k] = l;
        }
    }
    template <int K> static std::vector<iv8>& buf() { static thread_local std::vector<iv8> b; return b; }
    struct Prof { iv8 p[8]; iv8 ar; };
    // Row i's profile.
    [[gnu::always_inline]] void row(int i, Prof& P) const {
        const iv8 ai = rowa[i];
        if constexpr (Lv == kLvLut) { P.ar = ai; return; }
        else {
            constexpr int np = 1 << Lv;
            if constexpr (kByteShuffle) {
                for (int c = 0; c < np; ++c) P.p[c] = iv8_shuf(colt[c], ai);
            } else {
                for (int c = 0; c < np; ++c)
                    for (int l = 0; l < I8W; ++l) P.p[c][l] = colt[c][ai[l] & 15];
            }
        }
    }
    struct Sub {
        const iv8* cb; isb16 tbl;
        [[gnu::always_inline]] iv8 operator()(const Prof& P, int j) const {
            if constexpr (Lv == kLvLut) return iv8_shuf(tbl, P.ar + cb[j]);
            else return inter_sub<Lv>(P.p, cb, j);
        }
    };
    Sub sub() const { return Sub{colb.data(), tbl}; }
};

// Global / Glocal: the lanes whose A ends at row i, in order of m (ci: the next).
template <class C>
[[gnu::always_inline]] static inline void inter8_capture(const C& cm, const InterJobT<int8_t>& J, int i, int& ci,
                                                         bool glocal, const iv8* D) {
    for (int k = ci; k < I8W; ++k) {
        const int l = cm.ord[k];
        if (J.m[l] != i) break;
        ci = k + 1;
        const int nl = J.nb ? J.nb[l] : J.n;
        if (glocal) {
            int8_t b = D[0][l];
            for (int j = 1; j <= nl; ++j) b = std::max<int8_t>(b, D[j][l]);
            J.best[l] = b;
        } else {
            J.best[l] = D[nl][l];
        }
    }
}

// Affine, go, ge >= 0: inter_score_affine_pos in int8.  D = max(M, X, Y), X stored, Y in a
// register; the open is one saturating subtraction of go + ge (<= 127).
template <int Lv, bool Ragged>
static void inter8_score_affine(InterJobT<int8_t>& J) noexcept {
    using C = Inter8Common<Lv>;
    const C cm(J);
    const auto sub = cm.sub();
    const int n = J.n, M = J.M;
    const bool local = J.align_mode == 1, glocal = J.align_mode == 2;
    iv8* D = reinterpret_cast<iv8*>(J.VM);
    iv8* X = reinterpret_cast<iv8*>(J.VX);
    const iv8 z = {}, ninf = iv8_splat(-128);
    const iv8 goe_a = iv8_splat(J.go_a + J.ge_a), ge_a = iv8_splat(J.ge_a);
    const iv8 goe_b = iv8_splat(J.go_b + J.ge_b), ge_b = iv8_splat(J.ge_b);
    for (int j = 0; j <= n; ++j) {
        D[j] = (local || glocal || j == 0) ? z : iv8_splat(sat8(-(J.go_a + static_cast<long long>(j) * J.ge_a)));
        X[j] = ninf;
    }
    int ci = 0;
    if (!local) inter8_capture(cm, J, 0, ci, glocal, D);
    iv8 vbest = z;
    for (int i = 1; i <= M; ++i) {
        typename C::Prof P;
        cm.row(i, P);
        const iv8 live = iv8_splat(i <= 127 ? i : 127) <= cm.mv;
        iv8 dd = D[0];
        const iv8 x0 = local ? ninf : iv8_splat(sat8(-(J.go_b + static_cast<long long>(i) * J.ge_b)));
        iv8 lmx = local ? z : x0;
        iv8 ly = ninf;
        D[0] = lmx; X[0] = x0;
        iv8 rb = z;   // Local: this row's maximum over the lane's columns
        for (int j = 1; j <= n; ++j) {
            const iv8 s = sub(P, j);
            const iv8 du = D[j], xu = X[j];
            iv8 mv = iv8_adds(dd, s);
            if (local) mv = ivmax(mv, z);
            const iv8 x = ivmax(iv8_subs(du, goe_b), iv8_subs(xu, ge_b));
            const iv8 mx = ivmax(mv, x);
            const iv8 y = ivmax(iv8_subs(lmx, goe_a), iv8_subs(ly, ge_a));
            const iv8 dn = ivmax(mx, y);
            D[j] = dn; X[j] = x;
            dd = du; lmx = mx; ly = y;
            if (local) {
                if constexpr (Ragged) rb = ivmax(rb, ((iv8_splat(j) <= cm.nbv) ? mv : z));
                else rb = ivmax(rb, mv);
            }
        }
        if (local) vbest = ivmax(vbest, live ? rb : z);
        else inter8_capture(cm, J, i, ci, glocal, D);
    }
    if (local) for (int l = 0; l < I8W; ++l) J.best[l] = vbest[l];
}

// Linear: inter_score_linear in int8.
template <int Lv, bool Ragged>
static void inter8_score_linear(InterJobT<int8_t>& J) noexcept {
    using C = Inter8Common<Lv>;
    const C cm(J);
    const auto sub = cm.sub();
    const int n = J.n, M = J.M;
    const bool local = J.align_mode == 1, glocal = J.align_mode == 2;
    iv8* H = reinterpret_cast<iv8*>(J.VM);
    const iv8 z = {};
    const iv8 ge_a = iv8_splat(J.ge_a), ge_b = iv8_splat(J.ge_b);
    for (int j = 0; j <= n; ++j)
        H[j] = (local || glocal || j == 0) ? z : iv8_splat(sat8(-static_cast<long long>(j) * J.ge_a));
    int ci = 0;
    if (!local) inter8_capture(cm, J, 0, ci, glocal, H);
    iv8 vbest = z;
    for (int i = 1; i <= M; ++i) {
        typename C::Prof P;
        cm.row(i, P);
        const iv8 live = iv8_splat(i <= 127 ? i : 127) <= cm.mv;
        iv8 d = H[0];
        iv8 lh = local ? z : iv8_splat(sat8(-static_cast<long long>(i) * J.ge_b));
        H[0] = lh;
        iv8 rb = z;
        for (int j = 1; j <= n; ++j) {
            const iv8 up = H[j];
            iv8 v = iv8_adds(d, sub(P, j));
            v = ivmax(v, iv8_subs(up, ge_b));
            v = ivmax(v, iv8_subs(lh, ge_a));
            if (local) v = ivmax(v, z);
            H[j] = v;
            d = up; lh = v;
            if (local) {
                if constexpr (Ragged) rb = ivmax(rb, ((iv8_splat(j) <= cm.nbv) ? v : z));
                else rb = ivmax(rb, v);
            }
        }
        if (local) vbest = ivmax(vbest, live ? rb : z);
        else inter8_capture(cm, J, i, ci, glocal, H);
    }
    if (local) for (int l = 0; l < I8W; ++l) J.best[l] = vbest[l];
}

template <int Lv>
static void inter8_score_lv(InterJobT<int8_t>& J) noexcept {
    const bool rag = J.nb != nullptr;
    if (J.linear) { if (rag) inter8_score_linear<Lv, true>(J); else inter8_score_linear<Lv, false>(J); }
    else          { if (rag) inter8_score_affine<Lv, true>(J); else inter8_score_affine<Lv, false>(J); }
}
// Entry: the caller guarantees untracked, nalpha <= 8, J.M and J.n <= 127 (lane compares in
// int8), gaps >= 0, go + ge <= 127 per side, every substitution an int8.
static void inter8_score_entry(InterJobT<int8_t>& J) noexcept {
    if (J.nalpha <= 4 && kByteShuffle) inter8_score_lv<kLvLut>(J);
    else if (J.nalpha <= 4)            inter8_score_lv<2>(J);
    else                               inter8_score_lv<3>(J);
}
