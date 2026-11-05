// ── Inter-pair affine Full fill — one instantiation per ISA level ─────────────
//
// Included by level_common.inc inside the level namespace, after row_kernel_impl.inl,
// so `KW` (the level's native double-lane count) is in scope.  See InterJob in
// simd_levels.hpp for the contract.
//
// Why it exists.  On short pairs (miRNA x target site, ~22 x 50) neither per-pair
// kernel fills a vector well: the striped one pays a lazy-F fixpoint per row, the
// row-wise one a serial VY carry per row, and both pay per-row overhead on rows only
// ~50 cells long.  Putting one PAIR per lane removes all three: every lane runs the
// plain scalar recurrence, the VY carry is a lane-wise chain with no cross-lane fixup,
// and the substitution lookup is a short blend tree on the B residue's bits (DNA: two
// levels), not a gather.  Measured on 22x50 pairs: 4.1x the scalar fill at W=4 (AVX2).
//
// Bit-exactness.  Per lane, the operations and their order are those of the row-wise
// kernel (NWGRAD_BANDED_ROW_BODY) and of viterbi_affine_simd's Full borders: max, add
// and subtract on doubles only, the X subtractions left-associated, (v - go) - ge.
// Fusing the VY carry into the same column loop changes no value — each cell is still
// computed from the same operands in the same order.  The build disables FMA
// contraction (-ffp-contract=off), which the global borders' go + i*ge rely on.

#ifndef NWGRAD_LEVEL_NS
#  error "inter_kernel_impl.inl is included from a level TU (level_common.inc); not standalone"
#endif

inline constexpr int IW = KW;
typedef double ivd __attribute__((vector_size(8 * IW)));
typedef long long ivl __attribute__((vector_size(8 * IW)));
// float32: the same register holds twice the lanes; the integer types likewise by width.
typedef float ivf __attribute__((vector_size(8 * IW)));
typedef int ivi __attribute__((vector_size(8 * IW)));
typedef short ivs __attribute__((vector_size(8 * IW)));
typedef int16_t ivi16 __attribute__((vector_size(8 * IW)));
typedef int32_t ivi32 __attribute__((vector_size(8 * IW)));
typedef int64_t ivi64 __attribute__((vector_size(8 * IW)));

// Vector, mask (a signed integer vector of the same lane width) and lane count per
// scalar type.
template <class T> struct IVT;
template <> struct IVT<double>  { using v = ivd;   using l = ivl;   static constexpr int W = IW; };
template <> struct IVT<float>   { using v = ivf;   using l = ivi;   static constexpr int W = 2 * IW; };
template <> struct IVT<int16_t> { using v = ivi16; using l = ivi16; static constexpr int W = 4 * IW; };
template <> struct IVT<int32_t> { using v = ivi32; using l = ivi32; static constexpr int W = 2 * IW; };
template <> struct IVT<int64_t> { using v = ivi64; using l = ivi64; static constexpr int W = IW; };

// Lane-wise std::max(a, b), i.e. a < b ? b : a — the same choice on ties as the
// scalar and row-wise kernels.  On 128-bit x86 vectors it is MAXPD/MAXPS(b, a), which
// computes b > a ? b : a: the same lane on ties, ±0 and NaN included.  gcc emits that
// instruction for the select from SSE4.1 on, but at SSE2 it lowers it to cmpltpd +
// andpd + andnpd + orpd, four ops on the VY carry chain (0.49-0.70x the time of a
// training step on Manakov pairs, sse2 vs sse4 level).  clang emits maxpd at SSE2
// already.  Compilers without the builtins keep the select.
#if defined(__SSE2__) && defined(__has_builtin)
#  if __has_builtin(__builtin_ia32_maxpd) && __has_builtin(__builtin_ia32_maxps)
#    define NWGRAD_IVMAX_X86 1
#  endif
#endif
// always_inline: gcc left ivsel out of line once the soft kernels added call sites (2026-11-01),
// and every blend of the hard shared fill then paid a call and a mask normalisation (2.9x).
template <class V> [[gnu::always_inline]] static inline V ivmax(V a, V b) noexcept {
#ifdef NWGRAD_IVMAX_X86
    using E = std::remove_cvref_t<decltype(a[0])>;
    if constexpr (sizeof(V) == 16 && std::is_same_v<E, double>)
        return __builtin_ia32_maxpd(b, a);
    else if constexpr (sizeof(V) == 16 && std::is_same_v<E, float>)
        return __builtin_ia32_maxps(b, a);
    else
#endif
    return a < b ? b : a;
}
// ivblend: lane-wise m ? a : b for a mask m whose lanes are 0 or -1, taken as is -- for the
// blend tree, whose bit masks come from memory.  `m ? a : b` on a vector means m != 0,
// so where m is not a comparison in view -- a mask loaded from memory, as the blend tree's
// bits are -- gcc first compares it against zero (pcmpeq* before every pblendvb at sse4;
// perf annotate of inter_score_affine_pos<short> put that compare among the column
// loop's five hottest instructions).  A blend moves bits, so the float lanes stay exact.
// SSE2 has no blend: the and / andnot / or a select lowers to anyway, minus the compare.
// Only the tree takes it: where the mask is a comparison in view, `m ? a : b` lets gcc fold
// the pair into a max (rb = rb < v ? v : rb) or skip the compare by itself, and forcing the
// blend there cost the protein (gather) fills 2-6 % on nighthaven (AVX2, 2026-11-05).
template <class L, class V> [[gnu::always_inline]] static inline V ivsel(L m, V a, V b) noexcept { return m ? a : b; }
template <class L, class V> [[gnu::always_inline]] static inline V ivblend(L m, V a, V b) noexcept {
    static_assert(sizeof(L) == sizeof(V));
    using E [[maybe_unused]] = std::remove_cvref_t<decltype(a[0])>;
#if defined(__AVX512F__)
    if constexpr (sizeof(V) == 64)   // bitwise m ? a : b in one op (A ? B : C = 0xCA)
        return (V)_mm512_ternarylogic_epi64((__m512i)m, (__m512i)a, (__m512i)b, 0xCA);
    else
#endif
#if defined(__AVX2__)
    if constexpr (sizeof(V) == 32) {
        if constexpr (std::is_same_v<E, double>)     return (V)_mm256_blendv_pd((__m256d)b, (__m256d)a, (__m256d)m);
        else if constexpr (std::is_same_v<E, float>) return (V)_mm256_blendv_ps((__m256)b, (__m256)a, (__m256)m);
        else                                         return (V)_mm256_blendv_epi8((__m256i)b, (__m256i)a, (__m256i)m);
    } else
#endif
#if defined(__SSE4_1__)
    if constexpr (sizeof(V) == 16) {
        if constexpr (std::is_same_v<E, double>)     return (V)_mm_blendv_pd((__m128d)b, (__m128d)a, (__m128d)m);
        else if constexpr (std::is_same_v<E, float>) return (V)_mm_blendv_ps((__m128)b, (__m128)a, (__m128)m);
        else                                         return (V)_mm_blendv_epi8((__m128i)b, (__m128i)a, (__m128i)m);
    } else
#elif defined(__SSE2__)
    if constexpr (sizeof(V) == 16) {
        const __m128i mi = (__m128i)m;
        return (V)_mm_or_si128(_mm_and_si128(mi, (__m128i)a), _mm_andnot_si128(mi, (__m128i)b));
    } else
#elif defined(__ARM_NEON)
    if constexpr (sizeof(V) == 16)
        return (V)vbslq_u8((uint8x16_t)m, (uint8x16_t)a, (uint8x16_t)b);
    else
#endif
    return m ? a : b;
}
// Lane-wise add / subtract (plain: the inter-pair kernels never run a problem past the
// quarter-range bound -- the engine keeps such pairs on their own fill, whose kernels
// have a saturating variant; see nw_sat in scalar_traits.hpp).
template <class V> [[gnu::always_inline]] static inline V ivsub(V a, V b) noexcept { return a - b; }
template <class V> [[gnu::always_inline]] static inline V ivadd(V a, V b) noexcept { return a + b; }
// Every lane = x.  Not `ivd{} + x`: +0 + -0 is +0, so that idiom loses the sign of a
// zero border (-0 when a gap extend is 0), which the scalar fill keeps.
template <class V, class S> static inline V ivsplat(S x) noexcept {
    using E = std::remove_cvref_t<decltype(V{}[0])>;   // the lane type, not S's (an int
    V v;                                                // promotion would halve the count)
    for (unsigned l = 0; l < sizeof(V) / sizeof(E); ++l) v[l] = static_cast<E>(x);
    return v;
}

// A vector of f(0), f(1), ... f(W-1), as a braced initializer: built in registers (lane
// by lane into a local, then read back whole, it stalls on store forwarding).
// Each lane cast to the element type: an int16 lane's value arrives as an int (the
// promotion of -T(j) * ge), which a braced initializer narrows — clang refuses it.
template <class V, class F, std::size_t... I>
static inline V ivlanes_(F& f, std::index_sequence<I...>) noexcept {
    using E = std::remove_cvref_t<decltype(V{}[0])>;
    return V{static_cast<E>(f(static_cast<int>(I)))...};
}
template <class V, int W, class F>
static inline V ivlanes(F f) noexcept { return ivlanes_<V>(f, std::make_index_sequence<static_cast<std::size_t>(W)>{}); }

// Byte lookups (kLvLut below).  A level has them where a 16-entry byte shuffle exists:
// pshufb from SSSE3 (the sse4 level and up), tbl on NEON; the SSE2 baseline has none.
#if defined(__SSE4_1__) || defined(__ARM_NEON)
inline constexpr bool kByteShuffle = true;
#else
inline constexpr bool kByteShuffle = false;
#endif
typedef signed char isb16 __attribute__((vector_size(16)));
typedef signed char isb32 __attribute__((vector_size(32)));
// One byte per lane: 16 bytes, or 32 for the 32 int16 lanes of AVX-512.
template <int W> using isb = std::conditional_t<(W > 16), isb32, isb16>;
// r[k] = t[idx[k]] for idx[k] in 0..15, in each 16-byte half (pshufb's semantics).
template <class S> [[gnu::always_inline]] static inline S ibshuf(isb16 t, S idx) noexcept {
#if defined(__SSE4_1__)
    if constexpr (sizeof(S) == 16) return (S)_mm_shuffle_epi8((__m128i)t, (__m128i)idx);
#  if defined(__AVX2__)
    else return (S)_mm256_shuffle_epi8(_mm256_broadcastsi128_si256((__m128i)t), (__m256i)idx);
#  endif
#elif defined(__ARM_NEON)
    if constexpr (sizeof(S) == 16) return (S)vqtbl1q_s8((int8x16_t)t, (uint8x16_t)idx);
#endif
    S r{};   // levels without a shuffle never run this (kByteShuffle); it only compiles
    for (unsigned k = 0; k < sizeof(S); ++k) r[k] = t[idx[k] & 15];
    return r;
}
// The low W bytes of r, sign-extended (or converted) to V's W lanes: pmovsx* on x86.
template <class S, std::size_t... I>
[[gnu::always_inline]] static inline auto iblow_(S r, std::index_sequence<I...>) noexcept {
    return __builtin_shufflevector(r, r, static_cast<int>(I)...);
}
template <class V, int W, class S> [[gnu::always_inline]] static inline V ibwiden(S r) noexcept {
#if defined(__SSE4_1__)
    // 2 int64 lanes: gcc moved the 2-byte vector through a general register first
    using E = std::remove_cvref_t<decltype(V{}[0])>;
    if constexpr (W == 2 && sizeof(V) == 16 && std::is_integral_v<E>)
        return (V)_mm_cvtepi8_epi64((__m128i)r);
    else if constexpr (W == 2 && sizeof(V) == 16)   // 2 doubles: the same detour
        return (V)_mm_cvtepi32_pd(_mm_cvtepi8_epi32((__m128i)r));
    else
#endif
    return __builtin_convertvector(iblow_(r, std::make_index_sequence<static_cast<std::size_t>(W)>{}), V);
}
// v is exactly an int8 (as a bit pattern, for floats: -0 is not).
template <class T> static inline bool fits_i8(T v) noexcept {
    if constexpr (std::is_integral_v<T>) return v >= -128 && v <= 127;
    else {
        if (!(v >= T(-128) && v <= T(127))) return false;   // NaN too
        const T r = static_cast<T>(static_cast<int>(v));
        return std::memcmp(&r, &v, sizeof(T)) == 0;
    }
}


// The substitution lookup's form, a template parameter of every inter-pair kernel (Lv):
// kLvGather (-1) for alphabets over 8 letters (each row gathered into srow), 2 or 3 for
// a blend tree of that many levels (<= 4 / <= 8 letters).  Chosen once per job
// (inter_lv_dispatch).  It was a runtime `levels` tested at every column: perf annotate of
// inter_score_affine_pos<short> (sse4) put that compare and branch among the column loop's
// five hottest instructions.  Alphabets of 1 or 2 letters take the 2-level tree: their
// bit-1 masks are all 0 and P[2], P[3] repeat P[0], so it selects the same P[b].
inline constexpr int kLvGather = -1;
// kLvLut (4): alphabets of at most 4 letters, every score exactly an int8 (for float
// lanes as a bit pattern: an integral matrix without -0), at a level with a byte shuffle.  The 4x4 matrix is one 16-byte table, t[4a + b]; per row each
// lane's 4a as a byte vector, per column its b (the transposed B codes), and the score is
// one add, one shuffle and one widen per vector cell (float lanes: and a convert, exact
// for an int8), for the tree's 3 blends and 2 mask loads.  Values past int8 or
// non-integral, or tracked lanes (a block per row), take the tree.
inline constexpr int kLvLut = 4;

// Per lane, the score of this row's A residue against the column's B residue: a blend
// tree on the B residue's bits (bits[Lv*j + k] = bit k of each lane's b[j-1], as masks).
template <int Lv, class V, class L>
[[gnu::always_inline]] static inline V inter_sub(const V* P, const L* bits, int j) noexcept {
    static_assert(Lv == 2 || Lv == 3);
    const L b0 = bits[Lv * j];
    const V t0 = ivblend(b0, P[1], P[0]);
    const L b1 = bits[Lv * j + 1];
    const V u0 = ivblend(b1, ivblend(b0, P[3], P[2]), t0);
    if constexpr (Lv == 2) return u0;
    else {
        const V u1 = ivblend(b1, ivblend(b0, P[7], P[6]), ivblend(b0, P[5], P[4]));
        return ivblend(bits[Lv * j + 2], u1, u0);
    }
}

// The shared per-job setup: the B residues' bit masks and, per row, the profile.
// Alphabets over 8 letters (levels = -1) take no blend tree: each row's scores are
// GATHERED per lane into srow — one scalar load per lane-cell, outside the carry chain
// (Rognes' query profile cannot serve here: each lane has its own row residue).
template <class T, int Lv>
struct InterCommon {
    using ivd = typename IVT<T>::v; using ivl = typename IVT<T>::l;
    static constexpr int IW = IVT<T>::W;
#if defined(__ARM_NEON)
    static constexpr bool kMaskTranspose = IW >= 16;
#else
    static constexpr bool kMaskTranspose = IW >= 16 || (IW >= 8 && sizeof(T) == 2);
#endif
    const InterJobT<T>& J;
    static constexpr int levels = Lv;
    static constexpr bool kTree = Lv == 2 || Lv == 3;
    using sbv = isb<IW>;
    // A row's profile: the tree's 1 << Lv vectors, or (kLvLut) 4 * each lane's A code.
    struct Prof { ivd p[8]; sbv ar; };
    std::vector<ivl>& bits;
    std::vector<ivd>& srow;
    const sbv* btl = nullptr;   // kLvLut: per column, each lane's B code (ragged: pad 0)
    const sbv* atl = nullptr;   // kLvLut: per row, 4 * each lane's A code (0 past its A)
    isb16 tbl{};                // kLvLut: t[4a + b]
    // The per-row work, which scaled with W lane by lane (at W = 32, int16 AVX-512, the
    // tree's profile was 128 scalar loads and inserts a row): untracked tree rows build
    // P from the transposed A codes (atr), by byte shuffle where the scores are int8
    // (colt[c][a] = score(a, c); pshuf), `live` is one vector compare against the lanes'
    // m (mv; vlive: the row index fits the lane type), and Global's capture visits only
    // the lanes whose A ends at the row, in order of m (ord).
    const sbv* atr = nullptr;
    isb16 colt[8] = {};
    bool pshuf = false, vlive = false;
    ivl mv{};
    int ord[IW];
    explicit InterCommon(const InterJobT<T>& job) : J(job), bits(bits_buf()), srow(srow_buf()) {
        const int n = J.n;
        if (levels < 0 && srow.size() < static_cast<size_t>(n + 1)) srow.resize(n + 1);
        if constexpr (Lv == kLvLut) {
            const int na = std::min(J.nalpha, 4), M = J.M;   // <= 4 by dispatch; says so to gcc
            std::vector<sbv>& bb = sbv_buf<0>();
            std::vector<sbv>& ab = sbv_buf<1>();
            if (bb.size() < static_cast<size_t>(n + 1)) bb.resize(n + 1);
            if (ab.size() < static_cast<size_t>(M + 1)) ab.resize(M + 1);
            for (int j = 1; j <= n; ++j) {
                sbv v{};
                for (int l = 0; l < IW; ++l)
                    v[l] = static_cast<signed char>((!J.nb || j <= J.nb[l]) ? J.b[l][j - 1] : 0);   // ragged: pad
                bb[j] = v;
            }
            for (int i = 1; i <= M; ++i) {
                sbv v{};
                for (int l = 0; l < IW; ++l)
                    v[l] = static_cast<signed char>(4 * ((i <= J.m[l]) ? J.a[l][i - 1] : 0));
                ab[i] = v;
            }
            for (int a = 0; a < na; ++a)
                for (int b = 0; b < na; ++b) tbl[4 * a + b] = static_cast<signed char>(J.blk[a * na + b]);
            btl = bb.data(); atl = ab.data();
        }
        if constexpr (kTree && kByteShuffle) if (!J.tr) {   // atr only where a shuffle reads it
            const int na = J.nalpha, M = J.M;
            pshuf = true;
            for (int k = 0; k < na * na; ++k) pshuf &= fits_i8(J.blk[k]);
            if (pshuf) {
                for (int c = 0; c < (1 << Lv); ++c)
                    for (int a = 0; a < na; ++a)
                        colt[c][a] = static_cast<signed char>(J.blk[a * na + (c < na ? c : 0)]);
                std::vector<sbv>& ab = sbv_buf<1>();
                if (ab.size() < static_cast<size_t>(M + 1)) ab.resize(M + 1);
                for (int i = 1; i <= M; ++i) {
                    sbv v{};
                    for (int l = 0; l < IW; ++l) v[l] = static_cast<signed char>((i <= J.m[l]) ? J.a[l][i - 1] : 0);
                    ab[i] = v;
                }
                atr = ab.data();
            }
        }
        if constexpr (Lv >= 0) {   // the gather path keeps the lane-by-lane live and capture
            using E = std::remove_cvref_t<decltype(ivl{}[0])>;
            vlive = J.M <= static_cast<int>(std::numeric_limits<E>::max());
            for (int l = 0; l < IW; ++l) {
                mv[l] = static_cast<E>(std::min<long long>(J.m[l], std::numeric_limits<E>::max()));
                int k = l;   // insertion sort of the lanes by m, stable
                while (k > 0 && J.m[ord[k - 1]] > J.m[l]) { ord[k] = ord[k - 1]; --k; }
                ord[k] = l;
            }
        }
        if (levels > 0 && bits.size() < static_cast<size_t>(Lv * (n + 1))) bits.resize(Lv * (n + 1));
        // The masks from the lanes' residues: the W residues of column j gathered as bytes
        // (one byte load and store per lane), widened to the lane width in one convert, and
        // each bit's mask as an arithmetic shift pair -- where building each mask vector
        // lane by lane (a shift, a test and an insert per lane) was 14 % of a 22 x 50 DNA
        // score-only step at 16 lanes (nighthaven, perf, 2026-10-30).  Same masks.
        // Two loops, not one: the W byte stores of a column followed at once by one
        // W-byte load of the same bytes is a store-forwarding stall per column (3-6 % of
        // a 22 x 50 DNA step at 4-8 lanes, nighthaven 2026-10-30); transposed in full
        // first, the loads find their bytes in L1.  From 8 lanes: at 4 (double on AVX2,
        // float32 / int32 at 128 bits) the lane-by-lane build is as cheap, and this form
        // read 1.01-1.02 on the DNA double / float32 arms (5 rounds, nighthaven).  At 8
        // lanes it is a win only for 2-byte lanes on x86 (SSE int16: 4-6 % on nighthaven and
        // solace); NEON int16 LOST 2-6 % (spot, M1) and AVX2 float32 / int32 read a tie on
        // nighthaven and 3-8 % slower score-only on solace (Tiger Lake), so those keep the
        // lane-by-lane build (2026-10-31, the fleet round).
        if constexpr (!kTree) {
        } else if constexpr (kMaskTranspose) {
            using E = std::remove_cvref_t<decltype(ivl{}[0])>;
            typedef unsigned char u8v __attribute__((vector_size(IW)));
            constexpr int EB = 8 * sizeof(E);
            std::vector<unsigned char>& bt = bt_buf();
            if (bt.size() < static_cast<size_t>(n + 1) * IW) bt.resize(static_cast<size_t>(n + 1) * IW);
            unsigned char* btp = bt.data();
            for (int j = 1; j <= n; ++j)
                for (int l = 0; l < IW; ++l)
                    btp[static_cast<size_t>(j) * IW + l] = (!J.nb || j <= J.nb[l]) ? J.b[l][j - 1] : 0;   // ragged: pad
            for (int j = 1; j <= n; ++j) {
                u8v bv; std::memcpy(&bv, btp + static_cast<size_t>(j) * IW, IW);
                const ivl wv = __builtin_convertvector(bv, ivl);
                for (int k = 0; k < levels; ++k)
                    bits[Lv * j + k] = (wv << (EB - 1 - k)) >> (EB - 1);   // bit k, as 0 / -1
            }
        } else
        for (int j = 1; j <= n; ++j)
            for (int k = 0; k < levels; ++k) {
                bits[Lv * j + k] = ivlanes<ivl, IW>([&](int l) {
                    const int bl = (!J.nb || j <= J.nb[l]) ? J.b[l][j - 1] : 0;   // ragged: pad
                    return ((bl >> k) & 1) ? -1 : 0;
                });
            }
        if (J.nb)
            for (int l = 0; l < IW; ++l) nbv[l] = static_cast<T>(J.nb[l]);
    }
    typename IVT<T>::v nbv{};   // ragged: each lane's B length, as T
    static std::vector<ivl>& bits_buf() { static thread_local std::vector<ivl> b; return b; }
    static std::vector<unsigned char>& bt_buf() { static thread_local std::vector<unsigned char> b; return b; }
    static std::vector<ivd>& srow_buf() { static thread_local std::vector<ivd> b; return b; }
    template <int K> static std::vector<sbv>& sbv_buf() { static thread_local std::vector<sbv> b; return b; }
    // The row of `blk` lane l reads at DP row i: its residue's (tracked: its stacked row),
    // row 0 past the lane's own A (those cells are computed and ignored).
    template <bool Tr>
    size_t brow(int l, int i) const {
        if (i > J.m[l]) return 0;
        if constexpr (Tr) return static_cast<size_t>(J.tr[l].rs[i - 1]);
        else              return static_cast<size_t>(J.a[l][i - 1]);
    }
    // Tracked: lane l's value of a per-row array at DP row i, clamped to its own rows (the
    // rows past its A are computed and ignored, so any finite value serves there).
    static T lane_row(const T* arr, int i, int m) { return arr[i <= m ? i : m]; }
    // Tracked: row i's four penalties, per lane, each vector built as the untracked
    // kernels build theirs (z + value: the same rounding of a zero's sign).
    void row_gaps(int i, ivd& go_a, ivd& ge_a, ivd& go_b, ivd& ge_b) const {
        int r[IW];   // each lane's row, clamped
        for (int l = 0; l < IW; ++l) r[l] = i <= J.m[l] ? i : J.m[l];
        const ivd z = {};
        go_a = z + ivlanes<ivd, IW>([&](int l) { return J.tr[l].go_a[r[l]]; });
        ge_a = z + ivlanes<ivd, IW>([&](int l) { return J.tr[l].ge_a[r[l]]; });
        go_b = z + ivlanes<ivd, IW>([&](int l) { return J.tr[l].go_b[r[l]]; });
        ge_b = z + ivlanes<ivd, IW>([&](int l) { return J.tr[l].ge_b[r[l]]; });
    }
    // Tracked Global borders: column 0 of row i (bx0, -inf past the lane's A) and row 0
    // of column j (-(go_a + j*ge_a) of each lane's row 0; linear: -j*ge_a).
    ivd border_x(int i) const {
        return ivlanes<ivd, IW>([&](int l) {
            return (i <= J.m[l]) ? J.tr[l].bx0[i] : nw_ninf<T>;
        });
    }
    ivd border_y0(int j, bool linear) const {
        return ivlanes<ivd, IW>([&](int l) {
            const TrackJob<T>& t = J.tr[l];
            return linear ? -static_cast<T>(j) * t.ge_a[0] : -(t.go_a[0] + static_cast<T>(j) * t.ge_a[0]);
        });
    }
    // Row i's profile (P[c] = per lane, score of the lane's A residue i vs letter c) and
    // the lanes still inside their own A.
    // The gather path's row, out of line as main had it: inlined into the fill (or in the
    // unit of its own, level_<name>_inter.cpp) gcc compiled the lane loops below to 9
    // instructions an element where the out-of-line form takes ~3, +4.7 % instructions on
    // protein L100 float32 (skynet, sse4, 2026-11-05).
    template <bool Tr>
    [[gnu::noinline]] void row_gather(int i, ivl& live) const {
        const int na = J.nalpha;
        for (int l = 0; l < IW; ++l) live[l] = (i <= J.m[l]) ? -1 : 0;
        const int n = J.n;
        // A local pointer, not srow[j]: clang reloaded the vector's data pointer from
        // the object after every lane store (two dependent loads per element; this
        // gather is the larger half of a protein L100 inter-pair fill, clang 1.35x
        // gcc's samples here at sse4).
        ivd* const sr = srow.data();
        for (int l = 0; l < IW; ++l) {
            const T* pr;
            if constexpr (Tr) pr = J.blk + brow<Tr>(l, i) * na;
            else pr = J.blk + static_cast<size_t>((i <= J.m[l]) ? J.a[l][i - 1] : 0) * na;
            const unsigned char* bl = J.b[l];
            const int nl = J.nb ? J.nb[l] : n;
            for (int j = 1; j <= nl; ++j) sr[j][l] = pr[bl[j - 1]];
            for (int j = nl + 1; j <= n; ++j) sr[j][l] = pr[0];   // ragged: pad
        }
    }
    // The tree and table rows: always_inline (in the inter TU gcc otherwise left this out of
    // line in the fills).
    template <bool Tr = false>
    [[gnu::always_inline]] void row(int i, Prof& P, ivl& live) const {
        if constexpr (Lv < 0) { row_gather<Tr>(i, live); return; }
        else {
        const int na = J.nalpha;
        if (vlive) live = ivsplat<ivl>(i) <= mv;
        else for (int l = 0; l < IW; ++l) live[l] = (i <= J.m[l]) ? -1 : 0;
        if constexpr (Lv == kLvLut) {
            P.ar = atl[i];
        } else {
        if constexpr (Tr) {   // each lane's row once, then its letters (the P inter_sub reads)
            const T* rowp[IW];
            for (int l = 0; l < IW; ++l) rowp[l] = J.blk + brow<Tr>(l, i) * na;
            const int np = 1 << levels;
            for (int c = 0; c < np; ++c) {
                const int cc = c < na ? c : 0;
                P.p[c] = ivlanes<ivd, IW>([&](int l) { return rowp[l][cc]; });
            }
            return;
        }
        // The 1 << levels vectors inter_sub reads: by shuffle from the transposed row, or
        // each lane's row once, then the vectors built in registers.
        const int np = 1 << levels;
        if (pshuf) {
            const sbv ai = atr[i];
            for (int c = 0; c < np; ++c) P.p[c] = ibwiden<ivd, IW>(ibshuf(colt[c], ai));
            return;
        }
        size_t ro[IW];
        for (int l = 0; l < IW; ++l) ro[l] = static_cast<size_t>((i <= J.m[l]) ? J.a[l][i - 1] : 0) * na;
        for (int c = 0; c < np; ++c) {
            const int cc = c < na ? c : 0;
            P.p[c] = ivlanes<ivd, IW>([&](int l) { return J.blk[ro[l] + cc]; });
        }
        }
        }
    }
    // The lookup, over values copied out of the object, for the column loops.  `cm`
    // escapes (row() takes `this`), so through a member function clang reloaded the srow
    // vector's data pointer from memory at EVERY column — a store to the table through a
    // vector type may alias it as far as it can tell — and kept the loads on the critical
    // path (protein inter-pair fill, sse4).  A local copy whose address is never taken
    // stays in registers under either compiler.  Taken once per job: srow and bits are
    // sized in the constructor and never reallocated during the fill.
    struct Sub {
        const ivd* srow; const ivl* bits; const sbv* btl; isb16 tbl;
        [[gnu::always_inline]] ivd operator()(const Prof& P, int j) const {
            if constexpr (Lv < 0) return srow[j];
            else if constexpr (Lv == kLvLut) return ibwiden<ivd, IW>(ibshuf(tbl, P.ar + btl[j]));
            else return inter_sub<Lv>(P.p, bits, j);
        }
    };
    Sub sub() const { return Sub{srow.data(), bits.data(), btl, tbl}; }
    // Banded (InterJobT::blo != nullptr): row i's columns [j0, j1] to compute (the union
    // of the lanes' initialised spans) and the per-lane band [lo, hi] as T vectors.
    void band(int i, int& j0, int& j1, ivd& lo, ivd& hi) const {
        j0 = J.ulo[i]; j1 = J.uhi[i];
        for (int l = 0; l < IW; ++l) {
            lo[l] = static_cast<T>(J.blo[static_cast<size_t>(i) * IW + l]);
            hi[l] = static_cast<T>(J.bhi[static_cast<size_t>(i) * IW + l]);
        }
    }
    // Banded Global: a border cell is the closed form up to the lane's border_rows /
    // border_cols, -inf past it (as Aligner::viterbi_affine / viterbi_linear leave it).
    ivd border_mask(ivd v, int k, const int* lim) const {
        ivd kv, limv;
        for (int l = 0; l < IW; ++l) { kv[l] = static_cast<T>(k); limv[l] = static_cast<T>(lim[l]); }
        return ivsel(kv <= limv, v, ivd{} + nw_ninf<T>);
    }
    // Local: the row's first strict best folded into the running best (row-major order).
    void best_out(const ivd& best, const ivd& bi, const ivd& bj) const {
        for (int l = 0; l < IW; ++l) {
            J.best[l] = best[l];
            J.best_i[l] = static_cast<int>(bi[l]);
            J.best_j[l] = static_cast<int>(bj[l]);
        }
    }
};

// Linear gaps: one table H (in J.VM), per lane exactly Aligner::viterbi_linear —
// v = max(diag + s, up - ge_b, left - ge_a), Local clamped at 0, the same left-to-right
// max order (std::max over an initializer list keeps the FIRST largest, as the ivmax
// chain below does) and the same borders, so each lane's table is bit-identical to the
// pair's own fill.  Row-wise vectorization of linear was measured at 0.90x and deleted
// (the left carry is a pure latency chain); one PAIR per lane sidesteps that, since each
// lane's chain is its own and W of them run side by side.
//
// Banded: GuideBanded, each lane around its own guide.  Every row is computed over the
// union of the lanes' initialised spans, and a lane's cell outside its own band is set to
// -inf — which is what the pair's own banded fill leaves in every initialised cell it does
// not compute.  Its in-band cells read only cells of its own span, so they see exactly
// the operands of its own fill; the cells outside its span are never read by its own
// traceback or gradient.
// Tr: tracked lanes (J.tr) — each lane's borders, penalties and rows from its own arrays.
template <class T, int Lv, bool Banded, bool Ragged, bool Tr = false, bool Local = false>
static void inter_fill_linear(InterJobT<T>& J) noexcept {
    using C = InterCommon<T, Lv>;
    using ivd = typename C::ivd; using ivl = typename C::ivl;
    const C cm(J);
    const auto sub = cm.sub();
    const int n = J.n, M = J.M, st = n + 1;
    ivd* H = reinterpret_cast<ivd*>(J.VM);
    const ivd z = {}, ninf = z + nw_ninf<T>, one = z + T(1);

    if constexpr (Local) {
        for (int j = 0; j <= n; ++j) H[j] = z;
        for (int i = 1; i <= M; ++i) H[static_cast<size_t>(i) * st] = z;
    } else {
        H[0] = z;
        for (int i = 1; i <= M; ++i) {
            ivd v;
            if constexpr (Tr) v = cm.border_x(i);
            else              v = ivsplat<ivd>(-static_cast<T>(i) * J.ge_b);
            H[static_cast<size_t>(i) * st] = Banded ? cm.border_mask(v, i, J.bri) : v;
        }
        if (J.align_mode == 2) {   // Glocal: row 0 a free start at every column
            for (int j = 1; j <= n; ++j) H[j] = z;
        } else
        for (int j = 1; j <= n; ++j) {
            ivd v;
            if constexpr (Tr) v = cm.border_y0(j, true);
            else              v = ivsplat<ivd>(-static_cast<T>(j) * J.ge_a);
            H[j] = Banded ? cm.border_mask(v, j, J.brj) : v;
        }
    }

    ivd ge_a = z + J.ge_a, ge_b = z + J.ge_b;
    ivd best = z, bi = z, bj = z;

    for (int i = 1; i <= M; ++i) {
        typename C::Prof P;
        ivl live;
        cm.template row<Tr>(i, P, live);
        if constexpr (Tr) { ivd ga, gb; cm.row_gaps(i, ga, ge_a, gb, ge_b); }
        int j0 = 1, j1 = n;
        ivd lo = z, hi = z, jv = z;
        if constexpr (Banded) { cm.band(i, j0, j1, lo, hi); jv = z + static_cast<T>(j0); }

        ivd* h = H + static_cast<size_t>(i) * st;
        const ivd* p = h - st;
        ivd lh = (!Banded || j0 == 1) ? h[j0 - 1] : ninf;
        ivd rb = ninf, rj = z;
        ivd jc = z + static_cast<T>(j0);   // column j as T (exact): no per-column int->T convert
        for (int j = j0; j <= j1; ++j) {
            ivd v = ivadd(p[j - 1], sub(P, j));
            v = ivmax(v, ivsub(p[j], ge_b));
            v = ivmax(v, ivsub(lh, ge_a));
            if constexpr (Local) v = ivmax(v, z);
            if constexpr (Banded) { v = ivsel((jv >= lo) & (jv <= hi), v, ninf); jv += one; }
            h[j] = v;
            lh = v;
            if constexpr (Local) {
                ivl upd = rb < v;
                if constexpr (Ragged) upd &= jc <= cm.nbv;
                rb = ivsel(upd, v, rb);
                rj = ivsel(upd, jc, rj);
            }
            jc += one;
        }
        if constexpr (Local) {
            const ivl imp = live & (rb > best);
            best = ivsel(imp, rb, best);
            bi = ivsel(imp, z + static_cast<T>(i), bi);
            bj = ivsel(imp, rj, bj);
        }
    }
    if constexpr (Local) cm.best_out(best, bi, bj);
}

// Affine.  T = double or float32: the operations are the same in either precision, each
// in T — at float32 exactly the scalar viterbi_affine<float>, whose penalties are cast to
// T once and whose borders are -(go + T(i)*ge) in T.  Banded: as for linear above.
template <class T, int Lv, bool Banded, bool Ragged, bool Tr = false, bool Local = false>
static void inter_fill_affine(InterJobT<T>& J) noexcept {
    using C = InterCommon<T, Lv>;
    using ivd = typename C::ivd; using ivl = typename C::ivl;
    const C cm(J);
    const auto sub = cm.sub();
    const int n = J.n, M = J.M, st = n + 1;
    ivd* VM = reinterpret_cast<ivd*>(J.VM);
    ivd* VX = reinterpret_cast<ivd*>(J.VX);
    ivd* VY = reinterpret_cast<ivd*>(J.VY);
    const ivd z = {}, ninf = z + nw_ninf<T>, one = z + T(1);

    // Borders, as viterbi_affine_simd's Full path sets them.
    for (int j = 0; j <= n; ++j) { VM[j] = ninf; VX[j] = ninf; VY[j] = ninf; }
    for (int i = 1; i <= M; ++i) {
        const size_t o = static_cast<size_t>(i) * st;
        VM[o] = ninf; VX[o] = ninf; VY[o] = ninf;
    }
    if constexpr (Local) {
        for (int i = 0; i <= M; ++i) VM[static_cast<size_t>(i) * st] = z;
        for (int j = 0; j <= n; ++j) VM[j] = z;
    } else {
        VM[0] = z;
        for (int i = 1; i <= M; ++i) {
            ivd v;
            if constexpr (Tr) v = cm.border_x(i);
            else              v = ivsplat<ivd>(-(J.go_b + static_cast<T>(i) * J.ge_b));
            VX[static_cast<size_t>(i) * st] = Banded ? cm.border_mask(v, i, J.bri) : v;
        }
        if (J.align_mode == 2) {   // Glocal: row 0 a free start (M = 0) at every column, no Y
            for (int j = 1; j <= n; ++j) VM[j] = z;
        } else
        for (int j = 1; j <= n; ++j) {
            ivd v;
            if constexpr (Tr) v = cm.border_y0(j, false);
            else              v = ivsplat<ivd>(-(J.go_a + static_cast<T>(j) * J.ge_a));
            VY[j] = Banded ? cm.border_mask(v, j, J.brj) : v;
        }
    }

    ivd go_a = z + J.go_a, ge_a = z + J.ge_a, go_b = z + J.go_b, ge_b = z + J.ge_b;
    ivd best = z, bi = z, bj = z;   // Local: running best and its cell, per lane

    for (int i = 1; i <= M; ++i) {
        typename C::Prof P;
        ivl live;
        cm.template row<Tr>(i, P, live);
        if constexpr (Tr) cm.row_gaps(i, go_a, ge_a, go_b, ge_b);
        int j0 = 1, j1 = n;
        ivd lo = z, hi = z, jv = z;
        if constexpr (Banded) { cm.band(i, j0, j1, lo, hi); jv = z + static_cast<T>(j0); }

        ivd* vm = VM + static_cast<size_t>(i) * st;
        ivd* vx = VX + static_cast<size_t>(i) * st;
        ivd* vy = VY + static_cast<size_t>(i) * st;
        const ivd* pm = vm - st; const ivd* px = vx - st; const ivd* py = vy - st;
        ivd lm = ninf, lx = ninf, ly = ninf;   // column j0-1 of this row
        if (!Banded || j0 == 1) { lm = vm[j0 - 1]; lx = vx[j0 - 1]; ly = vy[j0 - 1]; }
        ivd rb = ninf, rj = z;
        ivd jc = z + static_cast<T>(j0);   // column j as T (exact): no per-column int->T convert
        for (int j = j0; j <= j1; ++j) {
            const ivd s = sub(P, j);
            ivd d = pm[j - 1];
            d = ivmax(d, px[j - 1]);
            d = ivmax(d, py[j - 1]);
            ivd mv = ivadd(d, s);
            if constexpr (Local) mv = ivmax(mv, z);
            ivd x = ivsub(ivsub(pm[j], go_b), ge_b);
            x = ivmax(x, ivsub(px[j], ge_b));
            x = ivmax(x, ivsub(ivsub(py[j], go_b), ge_b));
            ivd open = ivsub(ivsub(ivmax(lm, lx), go_a), ge_a);
            ivd y = ivmax(open, ivsub(ly, ge_a));
            if constexpr (Banded) {
                const ivl inb = (jv >= lo) & (jv <= hi);
                mv = ivsel(inb, mv, ninf); x = ivsel(inb, x, ninf); y = ivsel(inb, y, ninf);
                jv += one;
            }
            vm[j] = mv; vx[j] = x; vy[j] = y;
            lm = mv; lx = x; ly = y;
            if constexpr (Local) {
                const ivd cur = ivmax(mv, ivmax(x, y));
                ivl upd = rb < cur;                // first strict improvement in the row
                if constexpr (Ragged) upd &= jc <= cm.nbv;
                rb = ivsel(upd, cur, rb);
                rj = ivsel(upd, jc, rj);
            }
            jc += one;
        }
        if constexpr (Local) {
            const ivl imp = live & (rb > best);
            best = ivsel(imp, rb, best);
            bi = ivsel(imp, z + static_cast<T>(i), bi);
            bj = ivsel(imp, rj, bj);
        }
    }
    if constexpr (Local) cm.best_out(best, bi, bj);
}

// Banded needs no ragged variant: a lane's band never passes its own n (jhi <= n), and
// the masked cells cannot be its Local best.  Only the bits' padding (InterCommon) is
// needed there.  Full + ragged masks the Local best by column.
// Local is a template parameter, not a runtime flag in the column loop: clang (sse4,
// protein, 2026-10-27) tested it twice per column in the affine fill and kept the
// Local best-tracking out of line, where gcc unswitched it (clang 0.83x gcc).
template <class T, int Lv, bool Tr, bool Local>
static void inter_fill_tl(InterJobT<T>& J) noexcept {
    const bool lin = J.linear, rag = Local && J.nb != nullptr;
    if (J.blo) { if (lin) inter_fill_linear<T, Lv, true, false, Tr, Local>(J); else inter_fill_affine<T, Lv, true, false, Tr, Local>(J); }
    else if constexpr (Local) {
        if (rag) { if (lin) inter_fill_linear<T, Lv, false, true, Tr, Local>(J); else inter_fill_affine<T, Lv, false, true, Tr, Local>(J); }
        else     { if (lin) inter_fill_linear<T, Lv, false, false, Tr, Local>(J); else inter_fill_affine<T, Lv, false, false, Tr, Local>(J); }
    }
    else       { if (lin) inter_fill_linear<T, Lv, false, false, Tr, Local>(J); else inter_fill_affine<T, Lv, false, false, Tr, Local>(J); }
}
// The lookup form for a job (see kLvGather), as a compile-time constant for f.
template <class T, bool Tr, class F>
static void inter_lv_dispatch(const InterJobT<T>& J, F&& f) {
    const int na = J.nalpha;
    if (na > 8)      f(std::integral_constant<int, kLvGather>{});
    else if (na > 4) f(std::integral_constant<int, 3>{});
    else {
        if constexpr (kByteShuffle && !Tr) {
            bool fit = true;
            for (int k = 0; k < na * na; ++k) fit &= fits_i8(J.blk[k]);
            if (fit) { f(std::integral_constant<int, kLvLut>{}); return; }
        }
        f(std::integral_constant<int, 2>{});
    }
}
template <class T, bool Tr>
static void inter_fill_tr(InterJobT<T>& J) noexcept {
    inter_lv_dispatch<T, Tr>(J, [&](auto lv) {
        constexpr int Lv = decltype(lv)::value;
        if (J.align_mode == 1) inter_fill_tl<T, Lv, Tr, true>(J);
        else                   inter_fill_tl<T, Lv, Tr, false>(J);
    });
}
template <class T>
static void inter_fill_t(InterJobT<T>& J) noexcept {
    if (J.tr) inter_fill_tr<T, true>(J);
    else      inter_fill_tr<T, false>(J);
}

template <class T>
static void inter_fill_entry(InterJobT<T>& J) noexcept { inter_fill_t<T>(J); }

// ── Inter-pair SCORE ONLY (Full): the fills above on one row, updated in place ──
//
// Per lane exactly inter_fill_affine / inter_fill_linear (the same operands in the same
// order, so each lane's cells are bit-identical to its pair's own fill), but row i
// overwrites row i-1 left to right in J.VM/VX/VY ((n+1)*W each: one row, not a table).
// The diagonal operand of column j — row i-1, column j-1 — is overwritten one step
// earlier, so it is carried in registers.  Output, per lane, in J.best: Global the
// cell (m_l, n_l) — captured when the row loop passes that lane's last row — Glocal the
// best cell of that row (columns 0..n_l, every state), and Local
// the maximum over the lane's own cells (rows <= m_l, columns <= n_l), which is the
// fill's best value (max is exact in any order; the strict-first rule only places it).
template <class T, int Lv, bool Ragged, bool Tr = false>
static void inter_score_affine(InterJobT<T>& J) noexcept {
    using C = InterCommon<T, Lv>;
    using ivd = typename C::ivd; using ivl = typename C::ivl;
    constexpr int W = C::IW;
    const C cm(J);
    const auto sub = cm.sub();
    const int n = J.n, M = J.M;
    const bool local = J.align_mode == 1, glocal = J.align_mode == 2;
    ivd* VM = reinterpret_cast<ivd*>(J.VM);
    ivd* VX = reinterpret_cast<ivd*>(J.VX);
    ivd* VY = reinterpret_cast<ivd*>(J.VY);
    const ivd z = {}, ninf = z + nw_ninf<T>;
    ivd go_a = z + J.go_a, ge_a = z + J.ge_a, go_b = z + J.go_b, ge_b = z + J.ge_b;

    // row 0, as inter_fill_affine leaves it
    for (int j = 0; j <= n; ++j) {
        VM[j] = (local || glocal) ? z : (j == 0 ? z : ninf);
        VX[j] = ninf;
        if constexpr (Tr) VY[j] = (local || glocal || j == 0) ? ninf : cm.border_y0(j, false);
        else VY[j] = (local || glocal || j == 0) ? ninf : ivsplat<ivd>(-(J.go_a + static_cast<T>(j) * J.ge_a));
    }
    int ci = 0;                        // the next lane to capture, in order of m
    auto capture = [&](int i) {        // Global / Glocal: lanes whose A ends at row i
        for (int k = Lv < 0 ? 0 : ci; k < W; ++k) {
                const int l = Lv < 0 ? k : cm.ord[k];
                if (J.m[l] != i) { if (Lv < 0) continue; else break; }
                ci = k + 1;
                const int nl = Ragged ? J.nb[l] : n;
                if (glocal) {          // the best cell of the lane's last row (max is exact)
                    T b = std::max({VM[0][l], VX[0][l], VY[0][l]});
                    for (int j = 1; j <= nl; ++j) b = std::max(b, std::max({VM[j][l], VX[j][l], VY[j][l]}));
                    J.best[l] = b;
                } else {
                    J.best[l] = std::max({VM[nl][l], VX[nl][l], VY[nl][l]});
                }
            }
    };
    if (!local) capture(0);
    ivd vbest = z;

    for (int i = 1; i <= M; ++i) {
        typename C::Prof P;
        ivl live;
        cm.template row<Tr>(i, P, live);
        if constexpr (Tr) cm.row_gaps(i, go_a, ge_a, go_b, ge_b);
        ivd dm = VM[0], dx = VX[0], dy = VY[0];   // row i-1, column 0
        ivd lm = local ? z : ninf;
        ivd lx;
        if constexpr (Tr) lx = local ? ninf : cm.border_x(i);
        else lx = local ? ninf : ivsplat<ivd>(-(J.go_b + static_cast<T>(i) * J.ge_b));
        ivd ly = ninf;
        VM[0] = lm; VX[0] = lx; VY[0] = ly;
        for (int j = 1; j <= n; ++j) {
            const ivd s = sub(P, j);
            const ivd um = VM[j], ux = VX[j], uy = VY[j];   // row i-1, column j
            ivd d = dm;
            d = ivmax(d, dx);
            d = ivmax(d, dy);
            ivd mv = ivadd(d, s);
            if (local) mv = ivmax(mv, z);
            ivd x = ivsub(ivsub(um, go_b), ge_b);
            x = ivmax(x, ivsub(ux, ge_b));
            x = ivmax(x, ivsub(ivsub(uy, go_b), ge_b));
            const ivd open = ivsub(ivsub(ivmax(lm, lx), go_a), ge_a);
            const ivd y = ivmax(open, ivsub(ly, ge_a));
            VM[j] = mv; VX[j] = x; VY[j] = y;
            dm = um; dx = ux; dy = uy;
            lm = mv; lx = x; ly = y;
            if (local) {
                ivl keep = live;
                if constexpr (Ragged) keep &= (z + static_cast<T>(j)) <= cm.nbv;
                vbest = ivmax(vbest, ivsel(keep, ivmax(mv, ivmax(x, y)), z));
            }
        }
        if (!local) capture(i);
    }
    if (local) for (int l = 0; l < W; ++l) J.best[l] = vbest[l];
}

template <class T, int Lv, bool Ragged, bool Tr = false>
static void inter_score_linear(InterJobT<T>& J) noexcept {
    using C = InterCommon<T, Lv>;
    using ivd = typename C::ivd; using ivl = typename C::ivl;
    constexpr int W = C::IW;
    const C cm(J);
    const auto sub = cm.sub();
    const int n = J.n, M = J.M;
    const bool local = J.align_mode == 1, glocal = J.align_mode == 2;
    ivd* H = reinterpret_cast<ivd*>(J.VM);
    const ivd z = {};
    ivd ge_a = z + J.ge_a, ge_b = z + J.ge_b;

    for (int j = 0; j <= n; ++j) {
        if constexpr (Tr) H[j] = (local || glocal || j == 0) ? z : cm.border_y0(j, true);
        else H[j] = (local || glocal || j == 0) ? z : ivsplat<ivd>(-static_cast<T>(j) * J.ge_a);
    }
    int ci = 0;                        // the next lane to capture, in order of m
    auto capture = [&](int i) {
        for (int k = Lv < 0 ? 0 : ci; k < W; ++k) {
                const int l = Lv < 0 ? k : cm.ord[k];
                if (J.m[l] != i) { if (Lv < 0) continue; else break; }
                ci = k + 1;
                const int nl = Ragged ? J.nb[l] : n;
                if (glocal) {          // Glocal: the best cell of the lane's last row
                    T b = H[0][l];
                    for (int j = 1; j <= nl; ++j) b = std::max(b, static_cast<T>(H[j][l]));
                    J.best[l] = b;
                } else {
                    J.best[l] = H[nl][l];
                }
            }
    };
    if (!local) capture(0);
    ivd vbest = z;

    for (int i = 1; i <= M; ++i) {
        typename C::Prof P;
        ivl live;
        cm.template row<Tr>(i, P, live);
        if constexpr (Tr) { ivd ga, gb; cm.row_gaps(i, ga, ge_a, gb, ge_b); }
        ivd d = H[0];                                  // row i-1, column 0
        ivd lh;
        if constexpr (Tr) lh = local ? z : cm.border_x(i);
        else lh = local ? z : ivsplat<ivd>(-static_cast<T>(i) * J.ge_b);
        H[0] = lh;
        for (int j = 1; j <= n; ++j) {
            const ivd up = H[j];
            ivd v = ivadd(d, sub(P, j));
            v = ivmax(v, ivsub(up, ge_b));
            v = ivmax(v, ivsub(lh, ge_a));
            if (local) v = ivmax(v, z);
            H[j] = v;
            d = up;
            lh = v;
            if (local) {
                ivl keep = live;
                if constexpr (Ragged) keep &= (z + static_cast<T>(j)) <= cm.nbv;
                vbest = ivmax(vbest, ivsel(keep, v, z));
            }
        }
        if (!local) capture(i);
    }
    if (local) for (int l = 0; l < W; ++l) J.best[l] = vbest[l];
}

// Affine with non-negative OPEN penalties (the usual case): the same values from two
// stored rows instead of three.  With go >= 0, an open from a cell is never above the
// extension from it — (v - go) - ge <= v - ge, rounding being monotone — so offering the
// open every state of the cell (its D = max(M, X, Y)) adds only dominated candidates and
// drops none that could win:
//   X(i,j) = max((D(i-1,j) - go_b) - ge_b, X(i-1,j) - ge_b)
//   Y(i,j) = max((max(M,X)(i,j-1) - go_a) - ge_a, Y(i,j-1) - ge_a)   (as the fill: the
//            open must NOT come from D, which holds Y — that would put the max of three
//            and two subtractions on the loop-carried Y chain; measured 1.03x of the fill
//            with it, against 0.83x without)
//   M(i,j) = D(i-1,j-1) + s            (Local: clamped at 0)
// the same maxima, so the same values, as inter_score_affine's three-operand forms.  The
// row keeps D and X; M is never stored, Y lives in a register.  With the extends >= 0
// too, no gap cell exceeds the cell it extends from, so the Local maximum is the
// maximum over M alone (MOnly).
template <class T, int Lv, bool Ragged, bool MOnly, bool Tr = false>
static void inter_score_affine_pos(InterJobT<T>& J) noexcept {
    using C = InterCommon<T, Lv>;
    using ivd = typename C::ivd; using ivl = typename C::ivl;
    constexpr int W = C::IW;
    const C cm(J);
    const auto sub = cm.sub();
    const int n = J.n, M = J.M;
    const bool local = J.align_mode == 1, glocal = J.align_mode == 2;
    ivd* D = reinterpret_cast<ivd*>(J.VM);
    ivd* X = reinterpret_cast<ivd*>(J.VX);
    const ivd z = {}, ninf = z + nw_ninf<T>;
    ivd go_a = z + J.go_a, ge_a = z + J.ge_a, go_b = z + J.go_b, ge_b = z + J.ge_b;

    // row 0: D = max(M, X, Y) of inter_score_affine's row 0, X as there
    for (int j = 0; j <= n; ++j) {
        if constexpr (Tr) D[j] = (local || glocal) ? z : (j == 0 ? z : cm.border_y0(j, false));
        else D[j] = (local || glocal) ? z : (j == 0 ? z : ivsplat<ivd>(-(J.go_a + static_cast<T>(j) * J.ge_a)));
        X[j] = ninf;
    }
    int ci = 0;                        // the next lane to capture, in order of m
    auto capture = [&](int i) {
        for (int k = Lv < 0 ? 0 : ci; k < W; ++k) {
                const int l = Lv < 0 ? k : cm.ord[k];
                if (J.m[l] != i) { if (Lv < 0) continue; else break; }
                ci = k + 1;
                const int nl = Ragged ? J.nb[l] : n;
                if (glocal) {          // Glocal: the best cell of the lane's last row (D = every state)
                    T b = D[0][l];
                    for (int j = 1; j <= nl; ++j) b = std::max(b, static_cast<T>(D[j][l]));
                    J.best[l] = b;
                } else {
                    J.best[l] = D[nl][l];
                }
            }
    };
    if (!local) capture(0);
    ivd vbest = z;

    for (int i = 1; i <= M; ++i) {
        typename C::Prof P;
        ivl live;
        cm.template row<Tr>(i, P, live);
        if constexpr (Tr) cm.row_gaps(i, go_a, ge_a, go_b, ge_b);
        ivd dd = D[0];                                           // D(i-1, 0)
        ivd x0;
        if constexpr (Tr) x0 = local ? ninf : cm.border_x(i);
        else x0 = local ? ninf : ivsplat<ivd>(-(J.go_b + static_cast<T>(i) * J.ge_b));
        ivd lmx = local ? z : x0;                                // max(M, X)(i, 0)
        ivd ly = ninf;                                           // Y(i, 0)
        D[0] = lmx; X[0] = x0;                                   // D(i, 0): Y(i, 0) is -inf
        for (int j = 1; j <= n; ++j) {
            const ivd s = sub(P, j);
            const ivd du = D[j], xu = X[j];                      // row i-1, column j
            ivd mv = ivadd(dd, s);
            if (local) mv = ivmax(mv, z);
            const ivd x = ivmax(ivsub(ivsub(du, go_b), ge_b), ivsub(xu, ge_b));
            const ivd mx = ivmax(mv, x);
            const ivd y = ivmax(ivsub(ivsub(lmx, go_a), ge_a), ivsub(ly, ge_a));
            const ivd dn = ivmax(mx, y);
            D[j] = dn; X[j] = x;
            dd = du; lmx = mx; ly = y;
            if (local) {
                ivl keep = live;
                if constexpr (Ragged) keep &= (z + static_cast<T>(j)) <= cm.nbv;
                vbest = ivmax(vbest, ivsel(keep, MOnly ? mv : dn, z));
            }
        }
        if (!local) capture(i);
    }
    if (local) for (int l = 0; l < W; ++l) J.best[l] = vbest[l];
}

template <class T, int Lv, bool Tr>
static void inter_score_tr(InterJobT<T>& J) noexcept {
    const bool rag = J.nb != nullptr;
    if (J.linear) {
        if (rag) inter_score_linear<T, Lv, true, Tr>(J); else inter_score_linear<T, Lv, false, Tr>(J);
        return;
    }
    // NaN penalties compare false and take the general form.  Tracked: on every lane's
    // every row (the lanes' rows past their A included: they are computed too).
    bool pos_open, pos_ext;
    if constexpr (Tr) {
        pos_open = pos_ext = true;
        for (int l = 0; l < IVT<T>::W; ++l) {
            const TrackJob<T>& t = J.tr[l];
            for (int i = 0; i <= J.m[l]; ++i) {
                pos_open &= t.go_a[i] >= T(0);
                pos_ext  &= t.ge_a[i] >= T(0);
                if (i > 0) { pos_open &= t.go_b[i] >= T(0); pos_ext &= t.ge_b[i] >= T(0); }
            }
        }
    } else {
        pos_open = J.go_a >= T(0) && J.go_b >= T(0);
        pos_ext = J.ge_a >= T(0) && J.ge_b >= T(0);
    }
    if (pos_open && pos_ext) {
        if (rag) inter_score_affine_pos<T, Lv, true, true, Tr>(J);  else inter_score_affine_pos<T, Lv, false, true, Tr>(J);
    } else if (pos_open) {
        if (rag) inter_score_affine_pos<T, Lv, true, false, Tr>(J); else inter_score_affine_pos<T, Lv, false, false, Tr>(J);
    } else {
        if (rag) inter_score_affine<T, Lv, true, Tr>(J); else inter_score_affine<T, Lv, false, Tr>(J);
    }
}
template <class T, bool Tr>
static void inter_score_lv(InterJobT<T>& J) noexcept {
    inter_lv_dispatch<T, Tr>(J, [&](auto lv) { inter_score_tr<T, decltype(lv)::value, Tr>(J); });
}
template <class T>
static void inter_score_t(InterJobT<T>& J) noexcept {
    if (J.tr) inter_score_lv<T, true>(J);
    else      inter_score_lv<T, false>(J);
}
template <class T>
static void inter_score_entry(InterJobT<T>& J) noexcept { inter_score_t<T>(J); }
