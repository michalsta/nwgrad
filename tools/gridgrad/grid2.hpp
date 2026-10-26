// Grid-checkpoint hard gradient, affine Global, untracked, opens >= 0 — driver.
//
// solve(rect, entry cell) walks the optimal path back from the entry cell (i, j, state)
// until it reaches the rect's top row or left column.  A rect larger than the base size
// (and above max_depth levels) runs the vector grid kernel over [r0, i] x [c0, j] — the
// cells right of / below the entry cell cannot be on the path — then hands each block
// the path crosses to solve() one level down, with the saved lines as its borders.  A
// base rect is re-filled by the scalar pointer fill (codes as viterbi_affine_ptr).
//
//   max_depth = 1 : version A — one grid, every block re-filled scalar (block buffer
//                   ~ mn/k^2 cells).
//   max_depth = big: version B — recursion until blocks are <= base_cells: memory
//                   O(k (m + n)) for the grid lines, plus one base block.
#pragma once
#include "dp_buffer.hpp"
#include "gk.hpp"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

template <class T>
struct Grid2 {
    // problem
    const uint8_t* a; const uint8_t* b; int m, n, nalpha;
    const T* blk; T go_a, ge_a, go_b, ge_b;
    GridFwdFn<T> fwd;
    // tuning
    int k = 32;              // lines per dimension per level
    int max_depth = 1 << 20;
    long long base_cells = 64 * 64;
    int min_block = 8;       // rows / columns per block, at least
    // results
    double score = 0;
    std::vector<double> gmat;
    double g_go_a = 0, g_ge_a = 0, g_go_b = 0, g_ge_b = 0;
    long long cells_vec = 0, cells_scalar = 0;
    std::size_t peak_bytes = 0;

    static constexpr T NINF = -std::numeric_limits<T>::infinity();

    struct LevelBuf { std::vector<T, AlignedAllocator<T>> rows, cols; std::vector<int> colid; std::vector<std::pair<int, int>> gcols; };
    std::vector<LevelBuf> lv;   // reserved up front: solve() holds references into it
    std::vector<T> bM, bX, bY; std::vector<unsigned char> dM, dX, dY;
    std::vector<T> top0, left0;

    static unsigned char pick(T vm, T vx, T vy) {
        const unsigned char kxy = vx >= vy ? 1 : 2;
        return vm >= std::max(vx, vy) ? 0 : kxy;
    }

    std::size_t bytes_now() const {
        std::size_t s = (top0.size() + left0.size() + bM.size() * 3) * sizeof(T) + dM.size() * 3;
        for (auto& l : lv) s += (l.rows.size() + l.cols.size()) * sizeof(T) + l.colid.size() * sizeof(int);
        return s;
    }

    // Base case: scalar fill of [r0, r1] x [c0, c1] with pointer codes, then walk.
    void base(int r0, int r1, int c0, int c1, const Line<T>& top, const Line<T>& left, int& i, int& j, int& st) {
        const int bw = c1 - c0 + 1, h = r1 - r0 + 1;
        const std::size_t sz = (std::size_t)h * bw;
        if (bM.size() < sz) { bM.resize(sz); bX.resize(sz); bY.resize(sz); dM.resize(sz); dX.resize(sz); dY.resize(sz); }
        auto at = [&](int ii, int jj) { return (std::size_t)(ii - r0) * bw + (jj - c0); };
        for (int jj = c0; jj <= c1; ++jj) { const size_t q = at(r0, jj); bM[q] = top(0, jj - c0); bX[q] = top(1, jj - c0); bY[q] = top(2, jj - c0); }
        for (int ii = r0 + 1; ii <= r1; ++ii) {
            const size_t q0 = at(ii, c0);
            bM[q0] = left(0, ii - r0); bX[q0] = left(1, ii - r0); bY[q0] = left(2, ii - r0);
            const T* srow = blk + (size_t)a[ii - 1] * nalpha;
            for (int jj = c0 + 1; jj <= c1; ++jj) {
                const size_t q = at(ii, jj), up = q - bw, dg = up - 1, lf = q - 1;
                const T dgM = bM[dg], dgX = bX[dg], dgY = bY[dg];
                bM[q] = std::max({dgM, dgX, dgY}) + srow[b[jj - 1]];
                dM[q] = pick(dgM, dgX, dgY);
                const T ux = bM[up] - go_b - ge_b, vx = bX[up] - ge_b, wx = bY[up] - go_b - ge_b;
                bX[q] = std::max({ux, vx, wx}); dX[q] = pick(ux, vx, wx);
                const T uy = bM[lf] - go_a - ge_a, vy = bX[lf] - go_a - ge_a, wy = bY[lf] - ge_a;
                bY[q] = std::max({uy, vy, wy}); dY[q] = pick(uy, vy, wy);
            }
        }
        cells_scalar += (long long)(r1 - r0) * (c1 - c0);
        peak_bytes = std::max(peak_bytes, bytes_now());
        if (st < 0) {   // the end cell (m, n): its argmax, as best_tbl_
            const size_t q = at(i, j);
            const T vm = bM[q], vx = bX[q], vy = bY[q];
            st = (vm >= vx && vm >= vy) ? 0 : (vx >= vy ? 1 : 2);
            score = std::max({vm, vx, vy});
        }
        while (i > r0 && j > c0) {
            const size_t q = at(i, j);
            if (st == 0) { gmat[(size_t)a[i - 1] * nalpha + b[j - 1]] += 1; st = dM[q]; --i; --j; }
            else if (st == 1) { const int p = dX[q]; g_ge_b -= 1; if (p != 1) g_go_b -= 1; st = p; --i; }
            else { const int p = dY[q]; g_ge_a -= 1; if (p != 2) g_go_a -= 1; st = p; --j; }
        }
    }

    void solve(int r0, int c0, const Line<T>& top, const Line<T>& left, int depth, int& i, int& j, int& st) {
        const int r1 = i, c1 = j;            // nothing below / right of the entry cell is needed
        const int h = r1 - r0, w = c1 - c0;
        if (depth >= max_depth || (long long)h * w <= base_cells || h < 2 * min_block || w < 2 * min_block) {
            base(r0, r1, c0, c1, top, left, i, j, st);
            return;
        }
        if ((int)lv.size() <= depth) lv.resize(depth + 1);   // within the reserve: no move
        LevelBuf& L = lv[depth];
        const int R = std::max(min_block, (h + k - 1) / k);
        GridFwdJob<T> J{a, b, blk, nalpha, go_a, ge_a, go_b, ge_b, r0, r1, c0, c1, top, left, R, 1,
                        nullptr, nullptr, nullptr, 0};
        // q: column sets so that the lines are ~max(min_block, w/k) apart; W lanes per set
        const int Wl = (int)(sizeof(T) == 4 ? Wf : Wd);
        const int spacing = std::max(min_block, (w + k - 1) / k);
        const int seg = (w + Wl - 1) / Wl;
        J.q = std::max(1, std::min(seg, (seg + spacing - 1) / spacing));
        const std::size_t nrows = (std::size_t)std::max(0, (h - 1) / R);
        L.rows.resize(nrows * 3 * (w + 1));
        L.cols.resize((std::size_t)J.q * 3 * (h + 1) * Wl);
        L.colid.resize((std::size_t)J.q * Wl);
        J.rows = L.rows.data(); J.cols = L.cols.data(); J.colid = L.colid.data();
        fwd(J);
        cells_vec += (long long)h * w;
        peak_bytes = std::max(peak_bytes, bytes_now());
        const int W = J.W, q = J.q;
        // sorted grid columns: (column, set*W + lane); column c0 is the rect's left line (-1)
        L.gcols.clear();
        L.gcols.push_back({c0, -1});
        for (int t = 0; t < q * W; ++t) if (L.colid[t] >= 0) L.gcols.push_back({L.colid[t], t});
        std::sort(L.gcols.begin(), L.gcols.end());
        while (i > r0 && j > c0) {
            const int br0 = r0 + ((i - r0 - 1) / R) * R;
            auto it = std::lower_bound(L.gcols.begin(), L.gcols.end(), std::make_pair(j, -2));
            --it;                               // the largest grid column < j
            const int bc0 = it->first, tl = it->second;
            Line<T> ctop, cleft;
            if (br0 == r0) ctop = top.shifted(bc0 - c0);
            else ctop = Line<T>{L.rows.data() + (std::size_t)((br0 - r0) / R - 1) * 3 * (w + 1), (std::size_t)(w + 1), 1}.shifted(bc0 - c0);
            if (tl < 0) cleft = left.shifted(br0 - r0);
            else {
                const int t = tl / W, l = tl % W;
                cleft = Line<T>{L.cols.data() + (std::size_t)t * 3 * (h + 1) * W + l, (std::size_t)(h + 1) * W, (std::size_t)W}.shifted(br0 - r0);
            }
            solve(br0, bc0, ctop, cleft, depth + 1, i, j, st);
        }
    }

    int Wf = 1, Wd = 1;

    void run() {
        gmat.assign((size_t)nalpha * nalpha, 0.0);
        lv.reserve(64);
        top0.resize(3 * (size_t)(n + 1)); left0.resize(3 * (size_t)(m + 1));
        for (int jj = 0; jj <= n; ++jj) {
            top0[jj] = jj == 0 ? T(0) : NINF; top0[n + 1 + jj] = NINF;
            top0[2 * (n + 1) + jj] = jj == 0 ? NINF : -(go_a + static_cast<T>(jj) * ge_a);
        }
        for (int ii = 0; ii <= m; ++ii) {
            left0[ii] = ii == 0 ? T(0) : NINF;
            left0[m + 1 + ii] = ii == 0 ? NINF : -(go_b + static_cast<T>(ii) * ge_b);
            left0[2 * (m + 1) + ii] = NINF;
        }
        int i = m, j = n, st = -1;
        if (m == 0 || n == 0) {
            score = m == 0 ? (n == 0 ? 0.0 : (double)top0[2 * (n + 1) + n]) : (double)left0[m + 1 + m];
        } else {
            solve(0, 0, Line<T>{top0.data(), (std::size_t)(n + 1), 1}, Line<T>{left0.data(), (std::size_t)(m + 1), 1}, 0, i, j, st);
        }
        if (i == 0) for (; j > 0; --j) { g_ge_a -= 1; if (j == 1) g_go_a -= 1; }
        else if (j == 0) for (; i > 0; --i) { g_ge_b -= 1; if (i == 1) g_go_b -= 1; }
    }
};
