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
// float32: the same register holds twice the lanes.
typedef float ivf __attribute__((vector_size(8 * IW)));
typedef int ivi __attribute__((vector_size(8 * IW)));

// Vector, mask and lane count per precision.
template <class T> struct IVT;
template <> struct IVT<double> { using v = ivd; using l = ivl; static constexpr int W = IW; };
template <> struct IVT<float>  { using v = ivf; using l = ivi; static constexpr int W = 2 * IW; };

// Lane-wise std::max(a, b), i.e. a < b ? b : a — the same choice on ties as the
// scalar and row-wise kernels.
template <class V> static inline V ivmax(V a, V b) noexcept { return a < b ? b : a; }
template <class L, class V> static inline V ivsel(L m, V a, V b) noexcept { return m ? a : b; }

// Linear gaps: one table H (in J.VM), per lane exactly Aligner::viterbi_linear —
// v = max(diag + s, up - ge_b, left - ge_a), Local clamped at 0, the same left-to-right
// max order (std::max over an initializer list keeps the FIRST largest, as the ivmax
// chain below does) and the same borders, so each lane's table is bit-identical to the
// pair's own fill.  Row-wise vectorization of linear was measured at 0.90x and deleted
// (the left carry is a pure latency chain); one PAIR per lane sidesteps that, since each
// lane's chain is its own and W of them run side by side.
template <class T>
static void inter_fill_linear(InterJobT<T>& J) noexcept {
    using ivd = typename IVT<T>::v; using ivl = typename IVT<T>::l;
    constexpr int IW = IVT<T>::W;
    const int n = J.n, M = J.M, st = n + 1;
    const bool local = J.align_mode == 1;
    ivd* H = reinterpret_cast<ivd*>(J.VM);
    const ivd z = {}, ninf = z + (-std::numeric_limits<T>::infinity());

    if (local) {
        for (int j = 0; j <= n; ++j) H[j] = z;
        for (int i = 1; i <= M; ++i) H[static_cast<size_t>(i) * st] = z;
    } else {
        H[0] = z;
        for (int i = 1; i <= M; ++i)
            H[static_cast<size_t>(i) * st] = z + (-static_cast<T>(i) * J.ge_b);
        for (int j = 1; j <= n; ++j) H[j] = z + (-static_cast<T>(j) * J.ge_a);
    }

    const int na = J.nalpha;
    const int levels = na <= 1 ? 0 : na <= 2 ? 1 : na <= 4 ? 2 : 3;
    static thread_local std::vector<ivl> bits;
    if (bits.size() < static_cast<size_t>(3 * (n + 1))) bits.resize(3 * (n + 1));
    for (int j = 1; j <= n; ++j)
        for (int k = 0; k < levels; ++k) {
            ivl v;
            for (int l = 0; l < IW; ++l) v[l] = ((J.b[l][j - 1] >> k) & 1) ? -1 : 0;
            bits[3 * j + k] = v;
        }

    const ivd ge_a = z + J.ge_a, ge_b = z + J.ge_b;
    ivd best = z, bi = z, bj = z;

    for (int i = 1; i <= M; ++i) {
        ivd P[8];
        ivl live;
        for (int c = 0; c < 8; ++c) {
            const int cc = c < na ? c : 0;
            for (int l = 0; l < IW; ++l) {
                const int a = (i <= J.m[l]) ? J.a[l][i - 1] : 0;
                P[c][l] = J.blk[a * na + cc];
            }
        }
        for (int l = 0; l < IW; ++l) live[l] = (i <= J.m[l]) ? -1 : 0;

        ivd* h = H + static_cast<size_t>(i) * st;
        const ivd* p = h - st;
        ivd lh = h[0];
        ivd rb = ninf, rj = z;
        for (int j = 1; j <= n; ++j) {
            ivd s;
            if (levels == 0) s = P[0];
            else {
                const ivl b0 = bits[3 * j];
                ivd t0 = ivsel(b0, P[1], P[0]), t1 = ivsel(b0, P[3], P[2]);
                if (levels == 1) s = t0;
                else {
                    const ivl b1 = bits[3 * j + 1];
                    ivd u0 = ivsel(b1, t1, t0);
                    if (levels == 2) s = u0;
                    else {
                        ivd t2 = ivsel(b0, P[5], P[4]), t3 = ivsel(b0, P[7], P[6]);
                        ivd u1 = ivsel(b1, t3, t2);
                        s = ivsel(bits[3 * j + 2], u1, u0);
                    }
                }
            }
            ivd v = p[j - 1] + s;
            v = ivmax(v, p[j] - ge_b);
            v = ivmax(v, lh - ge_a);
            if (local) v = ivmax(v, z);
            h[j] = v;
            lh = v;
            if (local) {
                const ivl upd = rb < v;
                rb = ivsel(upd, v, rb);
                rj = ivsel(upd, z + static_cast<T>(j), rj);
            }
        }
        if (local) {
            const ivl imp = live & (rb > best);
            best = ivsel(imp, rb, best);
            bi = ivsel(imp, z + static_cast<T>(i), bi);
            bj = ivsel(imp, rj, bj);
        }
    }
    if (local)
        for (int l = 0; l < IW; ++l) {
            J.best[l] = best[l];
            J.best_i[l] = static_cast<int>(bi[l]);
            J.best_j[l] = static_cast<int>(bj[l]);
        }
}

// Affine (and the linear dispatch).  T = double or float32: the operations are the same
// in either precision, each in T — at float32 exactly the scalar viterbi_affine<float>,
// whose penalties are cast to T once and whose borders are -(go + T(i)*ge) in T.
template <class T>
static void inter_fill_t(InterJobT<T>& J) noexcept {
    if (J.linear) { inter_fill_linear(J); return; }
    using ivd = typename IVT<T>::v; using ivl = typename IVT<T>::l;
    constexpr int IW = IVT<T>::W;
    const T NINF = -std::numeric_limits<T>::infinity();
    const int n = J.n, M = J.M, st = n + 1;
    const bool local = J.align_mode == 1;
    ivd* VM = reinterpret_cast<ivd*>(J.VM);
    ivd* VX = reinterpret_cast<ivd*>(J.VX);
    ivd* VY = reinterpret_cast<ivd*>(J.VY);
    const ivd z = {}, ninf = z + NINF;

    // Borders, as viterbi_affine_simd's Full path sets them.
    for (int j = 0; j <= n; ++j) { VM[j] = ninf; VX[j] = ninf; VY[j] = ninf; }
    for (int i = 1; i <= M; ++i) {
        const size_t o = static_cast<size_t>(i) * st;
        VM[o] = ninf; VX[o] = ninf; VY[o] = ninf;
    }
    if (local) {
        for (int i = 0; i <= M; ++i) VM[static_cast<size_t>(i) * st] = z;
        for (int j = 0; j <= n; ++j) VM[j] = z;
    } else {
        VM[0] = z;
        for (int i = 1; i <= M; ++i)
            VX[static_cast<size_t>(i) * st] = z + (-(J.go_b + static_cast<T>(i) * J.ge_b));
        for (int j = 1; j <= n; ++j)
            VY[j] = z + (-(J.go_a + static_cast<T>(j) * J.ge_a));
    }

    // Per column, the bits of each lane's B residue, as blend masks.
    const int na = J.nalpha;
    const int levels = na <= 1 ? 0 : na <= 2 ? 1 : na <= 4 ? 2 : 3;
    static thread_local std::vector<ivl> bits;
    if (bits.size() < static_cast<size_t>(3 * (n + 1))) bits.resize(3 * (n + 1));
    for (int j = 1; j <= n; ++j)
        for (int k = 0; k < levels; ++k) {
            ivl v;
            for (int l = 0; l < IW; ++l) v[l] = ((J.b[l][j - 1] >> k) & 1) ? -1 : 0;
            bits[3 * j + k] = v;
        }

    const ivd go_a = z + J.go_a, ge_a = z + J.ge_a, go_b = z + J.go_b, ge_b = z + J.ge_b;
    ivd best = z, bi = z, bj = z;   // Local: running best and its cell, per lane

    for (int i = 1; i <= M; ++i) {
        // Row profile: P[c] holds, per lane, the score of this row's A residue vs c.
        ivd P[8];
        ivl live;
        for (int c = 0; c < 8; ++c) {
            const int cc = c < na ? c : 0;
            for (int l = 0; l < IW; ++l) {
                const int a = (i <= J.m[l]) ? J.a[l][i - 1] : 0;
                P[c][l] = J.blk[a * na + cc];
            }
        }
        for (int l = 0; l < IW; ++l) live[l] = (i <= J.m[l]) ? -1 : 0;

        ivd* vm = VM + static_cast<size_t>(i) * st;
        ivd* vx = VX + static_cast<size_t>(i) * st;
        ivd* vy = VY + static_cast<size_t>(i) * st;
        const ivd* pm = vm - st; const ivd* px = vx - st; const ivd* py = vy - st;
        ivd lm = vm[0], lx = vx[0], ly = vy[0];   // column j-1 of this row
        ivd rb = ninf, rj = z;
        for (int j = 1; j <= n; ++j) {
            ivd s;
            if (levels == 0) s = P[0];
            else {
                const ivl b0 = bits[3 * j];
                ivd t0 = ivsel(b0, P[1], P[0]), t1 = ivsel(b0, P[3], P[2]);
                if (levels == 1) s = t0;
                else {
                    const ivl b1 = bits[3 * j + 1];
                    ivd u0 = ivsel(b1, t1, t0);
                    if (levels == 2) s = u0;
                    else {
                        ivd t2 = ivsel(b0, P[5], P[4]), t3 = ivsel(b0, P[7], P[6]);
                        ivd u1 = ivsel(b1, t3, t2);
                        s = ivsel(bits[3 * j + 2], u1, u0);
                    }
                }
            }
            ivd d = pm[j - 1];
            d = ivmax(d, px[j - 1]);
            d = ivmax(d, py[j - 1]);
            ivd mv = d + s;
            if (local) mv = ivmax(mv, z);
            ivd x = (pm[j] - go_b) - ge_b;
            x = ivmax(x, px[j] - ge_b);
            x = ivmax(x, (py[j] - go_b) - ge_b);
            ivd open = (ivmax(lm, lx) - go_a) - ge_a;
            ivd y = ivmax(open, ly - ge_a);
            vm[j] = mv; vx[j] = x; vy[j] = y;
            lm = mv; lx = x; ly = y;
            if (local) {
                const ivd cur = ivmax(mv, ivmax(x, y));
                const ivl upd = rb < cur;          // first strict improvement in the row
                rb = ivsel(upd, cur, rb);
                rj = ivsel(upd, z + static_cast<T>(j), rj);
            }
        }
        if (local) {
            const ivl imp = live & (rb > best);
            best = ivsel(imp, rb, best);
            bi = ivsel(imp, z + static_cast<T>(i), bi);
            bj = ivsel(imp, rj, bj);
        }
    }
    if (local)
        for (int l = 0; l < IW; ++l) {
            J.best[l] = best[l];
            J.best_i[l] = static_cast<int>(bi[l]);
            J.best_j[l] = static_cast<int>(bj[l]);
        }
}

static void inter_fill_entry(InterJob& J) noexcept { inter_fill_t<double>(J); }
static void inter_fill_entry_f(InterJobT<float>& J) noexcept { inter_fill_t<float>(J); }
