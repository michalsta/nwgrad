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
// Every lane = x.  Not `ivd{} + x`: +0 + -0 is +0, so that idiom loses the sign of a
// zero border (-0 when a gap extend is 0), which the scalar fill keeps.
template <class V, class S> static inline V ivsplat(S x) noexcept {
    V v;
    for (unsigned l = 0; l < sizeof(V) / sizeof(S); ++l) v[l] = x;
    return v;
}


// Per lane, the score of this row's A residue against the column's B residue: a blend
// tree on the B residue's bits (bits[3*j + k] = bit k of each lane's b[j-1], as masks).
template <class V, class L>
static inline V inter_sub(const V* P, const L* bits, int j, int levels) noexcept {
    if (levels == 0) return P[0];
    const L b0 = bits[3 * j];
    const V t0 = ivsel(b0, P[1], P[0]);
    if (levels == 1) return t0;
    const L b1 = bits[3 * j + 1];
    const V u0 = ivsel(b1, ivsel(b0, P[3], P[2]), t0);
    if (levels == 2) return u0;
    const V u1 = ivsel(b1, ivsel(b0, P[7], P[6]), ivsel(b0, P[5], P[4]));
    return ivsel(bits[3 * j + 2], u1, u0);
}

// The shared per-job setup: the B residues' bit masks and, per row, the profile.
// Alphabets over 8 letters (levels = -1) take no blend tree: each row's scores are
// GATHERED per lane into srow — one scalar load per lane-cell, outside the carry chain
// (Rognes' query profile cannot serve here: each lane has its own row residue).
template <class T>
struct InterCommon {
    using ivd = typename IVT<T>::v; using ivl = typename IVT<T>::l;
    static constexpr int IW = IVT<T>::W;
    const InterJobT<T>& J;
    int levels;
    std::vector<ivl>& bits;
    std::vector<ivd>& srow;
    explicit InterCommon(const InterJobT<T>& job) : J(job), bits(bits_buf()), srow(srow_buf()) {
        const int na = J.nalpha, n = J.n;
        levels = na <= 1 ? 0 : na <= 2 ? 1 : na <= 4 ? 2 : na <= 8 ? 3 : -1;
        if (levels < 0 && srow.size() < static_cast<size_t>(n + 1)) srow.resize(n + 1);
        if (bits.size() < static_cast<size_t>(3 * (n + 1))) bits.resize(3 * (n + 1));
        for (int j = 1; j <= n; ++j)
            for (int k = 0; k < levels; ++k) {
                ivl v;
                for (int l = 0; l < IW; ++l) {
                    const int bl = (!J.nb || j <= J.nb[l]) ? J.b[l][j - 1] : 0;   // ragged: pad
                    v[l] = ((bl >> k) & 1) ? -1 : 0;
                }
                bits[3 * j + k] = v;
            }
        if (J.nb)
            for (int l = 0; l < IW; ++l) nbv[l] = static_cast<T>(J.nb[l]);
    }
    typename IVT<T>::v nbv{};   // ragged: each lane's B length, as T
    static std::vector<ivl>& bits_buf() { static thread_local std::vector<ivl> b; return b; }
    static std::vector<ivd>& srow_buf() { static thread_local std::vector<ivd> b; return b; }
    // Row i's profile (P[c] = per lane, score of the lane's A residue i vs letter c) and
    // the lanes still inside their own A.
    void row(int i, ivd* P, ivl& live) const {
        const int na = J.nalpha;
        for (int l = 0; l < IW; ++l) live[l] = (i <= J.m[l]) ? -1 : 0;
        if (levels < 0) {
            const int n = J.n;
            for (int l = 0; l < IW; ++l) {
                const T* pr = J.blk + static_cast<size_t>((i <= J.m[l]) ? J.a[l][i - 1] : 0) * na;
                const unsigned char* bl = J.b[l];
                const int nl = J.nb ? J.nb[l] : n;
                for (int j = 1; j <= nl; ++j) srow[j][l] = pr[bl[j - 1]];
                for (int j = nl + 1; j <= n; ++j) srow[j][l] = pr[0];   // ragged: pad
            }
            return;
        }
        for (int c = 0; c < 8; ++c) {
            const int cc = c < na ? c : 0;
            for (int l = 0; l < IW; ++l) {
                const int a = (i <= J.m[l]) ? J.a[l][i - 1] : 0;
                P[c][l] = J.blk[a * na + cc];
            }
        }
    }
    ivd s(const ivd* P, int j) const {
        return levels < 0 ? srow[j] : inter_sub(P, bits.data(), j, levels);
    }
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
        return ivsel(kv <= limv, v, ivd{} + (-std::numeric_limits<T>::infinity()));
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
template <class T, bool Banded, bool Ragged>
static void inter_fill_linear(InterJobT<T>& J) noexcept {
    using C = InterCommon<T>;
    using ivd = typename C::ivd; using ivl = typename C::ivl;
    const C cm(J);
    const int n = J.n, M = J.M, st = n + 1;
    const bool local = J.align_mode == 1;
    ivd* H = reinterpret_cast<ivd*>(J.VM);
    const ivd z = {}, ninf = z + (-std::numeric_limits<T>::infinity()), one = z + T(1);

    if (local) {
        for (int j = 0; j <= n; ++j) H[j] = z;
        for (int i = 1; i <= M; ++i) H[static_cast<size_t>(i) * st] = z;
    } else {
        H[0] = z;
        for (int i = 1; i <= M; ++i) {
            const ivd v = ivsplat<ivd>(-static_cast<T>(i) * J.ge_b);
            H[static_cast<size_t>(i) * st] = Banded ? cm.border_mask(v, i, J.bri) : v;
        }
        for (int j = 1; j <= n; ++j) {
            const ivd v = ivsplat<ivd>(-static_cast<T>(j) * J.ge_a);
            H[j] = Banded ? cm.border_mask(v, j, J.brj) : v;
        }
    }

    const ivd ge_a = z + J.ge_a, ge_b = z + J.ge_b;
    ivd best = z, bi = z, bj = z;

    for (int i = 1; i <= M; ++i) {
        ivd P[8];
        ivl live;
        cm.row(i, P, live);
        int j0 = 1, j1 = n;
        ivd lo = z, hi = z, jv = z;
        if constexpr (Banded) { cm.band(i, j0, j1, lo, hi); jv = z + static_cast<T>(j0); }

        ivd* h = H + static_cast<size_t>(i) * st;
        const ivd* p = h - st;
        ivd lh = (!Banded || j0 == 1) ? h[j0 - 1] : ninf;
        ivd rb = ninf, rj = z;
        for (int j = j0; j <= j1; ++j) {
            ivd v = p[j - 1] + cm.s(P, j);
            v = ivmax(v, p[j] - ge_b);
            v = ivmax(v, lh - ge_a);
            if (local) v = ivmax(v, z);
            if constexpr (Banded) { v = ivsel((jv >= lo) & (jv <= hi), v, ninf); jv += one; }
            h[j] = v;
            lh = v;
            if (local) {
                ivl upd = rb < v;
                if constexpr (Ragged) upd &= (z + static_cast<T>(j)) <= cm.nbv;
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
    if (local) cm.best_out(best, bi, bj);
}

// Affine.  T = double or float32: the operations are the same in either precision, each
// in T — at float32 exactly the scalar viterbi_affine<float>, whose penalties are cast to
// T once and whose borders are -(go + T(i)*ge) in T.  Banded: as for linear above.
template <class T, bool Banded, bool Ragged>
static void inter_fill_affine(InterJobT<T>& J) noexcept {
    using C = InterCommon<T>;
    using ivd = typename C::ivd; using ivl = typename C::ivl;
    const C cm(J);
    const int n = J.n, M = J.M, st = n + 1;
    const bool local = J.align_mode == 1;
    ivd* VM = reinterpret_cast<ivd*>(J.VM);
    ivd* VX = reinterpret_cast<ivd*>(J.VX);
    ivd* VY = reinterpret_cast<ivd*>(J.VY);
    const ivd z = {}, ninf = z + (-std::numeric_limits<T>::infinity()), one = z + T(1);

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
        for (int i = 1; i <= M; ++i) {
            const ivd v = ivsplat<ivd>(-(J.go_b + static_cast<T>(i) * J.ge_b));
            VX[static_cast<size_t>(i) * st] = Banded ? cm.border_mask(v, i, J.bri) : v;
        }
        for (int j = 1; j <= n; ++j) {
            const ivd v = ivsplat<ivd>(-(J.go_a + static_cast<T>(j) * J.ge_a));
            VY[j] = Banded ? cm.border_mask(v, j, J.brj) : v;
        }
    }

    const ivd go_a = z + J.go_a, ge_a = z + J.ge_a, go_b = z + J.go_b, ge_b = z + J.ge_b;
    ivd best = z, bi = z, bj = z;   // Local: running best and its cell, per lane

    for (int i = 1; i <= M; ++i) {
        ivd P[8];
        ivl live;
        cm.row(i, P, live);
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
        for (int j = j0; j <= j1; ++j) {
            const ivd s = cm.s(P, j);
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
            if constexpr (Banded) {
                const ivl inb = (jv >= lo) & (jv <= hi);
                mv = ivsel(inb, mv, ninf); x = ivsel(inb, x, ninf); y = ivsel(inb, y, ninf);
                jv += one;
            }
            vm[j] = mv; vx[j] = x; vy[j] = y;
            lm = mv; lx = x; ly = y;
            if (local) {
                const ivd cur = ivmax(mv, ivmax(x, y));
                ivl upd = rb < cur;                // first strict improvement in the row
                if constexpr (Ragged) upd &= (z + static_cast<T>(j)) <= cm.nbv;
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
    if (local) cm.best_out(best, bi, bj);
}

// Banded needs no ragged variant: a lane's band never passes its own n (jhi <= n), and
// the masked cells cannot be its Local best.  Only the bits' padding (InterCommon) is
// needed there.  Full + ragged masks the Local best by column.
template <class T>
static void inter_fill_t(InterJobT<T>& J) noexcept {
    const bool lin = J.linear, rag = J.nb != nullptr && J.align_mode == 1;
    if (J.blo) { if (lin) inter_fill_linear<T, true, false>(J); else inter_fill_affine<T, true, false>(J); }
    else if (rag) { if (lin) inter_fill_linear<T, false, true>(J); else inter_fill_affine<T, false, true>(J); }
    else       { if (lin) inter_fill_linear<T, false, false>(J); else inter_fill_affine<T, false, false>(J); }
}

static void inter_fill_entry(InterJob& J) noexcept { inter_fill_t<double>(J); }
static void inter_fill_entry_f(InterJobT<float>& J) noexcept { inter_fill_t<float>(J); }

// ── Inter-pair SCORE ONLY (Full): the fills above on one row, updated in place ──
//
// Per lane exactly inter_fill_affine / inter_fill_linear (the same operands in the same
// order, so each lane's cells are bit-identical to its pair's own fill), but row i
// overwrites row i-1 left to right in J.VM/VX/VY ((n+1)*W each: one row, not a table).
// The diagonal operand of column j — row i-1, column j-1 — is overwritten one step
// earlier, so it is carried in registers.  Output, per lane, in J.best: Global the
// cell (m_l, n_l) — captured when the row loop passes that lane's last row — and Local
// the maximum over the lane's own cells (rows <= m_l, columns <= n_l), which is the
// fill's best value (max is exact in any order; the strict-first rule only places it).
template <class T, bool Ragged>
static void inter_score_affine(InterJobT<T>& J) noexcept {
    using C = InterCommon<T>;
    using ivd = typename C::ivd; using ivl = typename C::ivl;
    constexpr int W = C::IW;
    const C cm(J);
    const int n = J.n, M = J.M;
    const bool local = J.align_mode == 1;
    ivd* VM = reinterpret_cast<ivd*>(J.VM);
    ivd* VX = reinterpret_cast<ivd*>(J.VX);
    ivd* VY = reinterpret_cast<ivd*>(J.VY);
    const ivd z = {}, ninf = z + (-std::numeric_limits<T>::infinity());
    const ivd go_a = z + J.go_a, ge_a = z + J.ge_a, go_b = z + J.go_b, ge_b = z + J.ge_b;

    // row 0, as inter_fill_affine leaves it
    for (int j = 0; j <= n; ++j) {
        VM[j] = local ? z : (j == 0 ? z : ninf);
        VX[j] = ninf;
        VY[j] = (local || j == 0) ? ninf : ivsplat<ivd>(-(J.go_a + static_cast<T>(j) * J.ge_a));
    }
    auto capture = [&](int i) {        // Global: lanes whose A ends at row i
        for (int l = 0; l < W; ++l)
            if (J.m[l] == i) {
                const int nl = Ragged ? J.nb[l] : n;
                J.best[l] = std::max({VM[nl][l], VX[nl][l], VY[nl][l]});
            }
    };
    if (!local) capture(0);
    ivd vbest = z;

    for (int i = 1; i <= M; ++i) {
        ivd P[8];
        ivl live;
        cm.row(i, P, live);
        ivd dm = VM[0], dx = VX[0], dy = VY[0];   // row i-1, column 0
        ivd lm = local ? z : ninf;
        ivd lx = local ? ninf : ivsplat<ivd>(-(J.go_b + static_cast<T>(i) * J.ge_b));
        ivd ly = ninf;
        VM[0] = lm; VX[0] = lx; VY[0] = ly;
        for (int j = 1; j <= n; ++j) {
            const ivd s = cm.s(P, j);
            const ivd um = VM[j], ux = VX[j], uy = VY[j];   // row i-1, column j
            ivd d = dm;
            d = ivmax(d, dx);
            d = ivmax(d, dy);
            ivd mv = d + s;
            if (local) mv = ivmax(mv, z);
            ivd x = (um - go_b) - ge_b;
            x = ivmax(x, ux - ge_b);
            x = ivmax(x, (uy - go_b) - ge_b);
            const ivd open = (ivmax(lm, lx) - go_a) - ge_a;
            const ivd y = ivmax(open, ly - ge_a);
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

template <class T, bool Ragged>
static void inter_score_linear(InterJobT<T>& J) noexcept {
    using C = InterCommon<T>;
    using ivd = typename C::ivd; using ivl = typename C::ivl;
    constexpr int W = C::IW;
    const C cm(J);
    const int n = J.n, M = J.M;
    const bool local = J.align_mode == 1;
    ivd* H = reinterpret_cast<ivd*>(J.VM);
    const ivd z = {};
    const ivd ge_a = z + J.ge_a, ge_b = z + J.ge_b;

    for (int j = 0; j <= n; ++j)
        H[j] = (local || j == 0) ? z : ivsplat<ivd>(-static_cast<T>(j) * J.ge_a);
    auto capture = [&](int i) {
        for (int l = 0; l < W; ++l)
            if (J.m[l] == i) J.best[l] = H[Ragged ? J.nb[l] : n][l];
    };
    if (!local) capture(0);
    ivd vbest = z;

    for (int i = 1; i <= M; ++i) {
        ivd P[8];
        ivl live;
        cm.row(i, P, live);
        ivd d = H[0];                                  // row i-1, column 0
        ivd lh = local ? z : ivsplat<ivd>(-static_cast<T>(i) * J.ge_b);
        H[0] = lh;
        for (int j = 1; j <= n; ++j) {
            const ivd up = H[j];
            ivd v = d + cm.s(P, j);
            v = ivmax(v, up - ge_b);
            v = ivmax(v, lh - ge_a);
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

template <class T>
static void inter_score_t(InterJobT<T>& J) noexcept {
    if (J.nb) { if (J.linear) inter_score_linear<T, true>(J);  else inter_score_affine<T, true>(J); }
    else      { if (J.linear) inter_score_linear<T, false>(J); else inter_score_affine<T, false>(J); }
}
static void inter_score_entry(InterJob& J) noexcept { inter_score_t<double>(J); }
static void inter_score_entry_f(InterJobT<float>& J) noexcept { inter_score_t<float>(J); }
