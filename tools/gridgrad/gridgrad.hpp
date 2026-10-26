// Grid-checkpoint hard gradient (FastLSA-style) — scalar prototype, affine Global, untracked.
//
// 1. One forward pass over rolling rows (the score-only recurrence), saving every R-th row
//    and every C-th column of all three states (M, X, Y).  Rows 0 and columns 0 are grid
//    lines too (the closed-form borders).
// 2. The grid cuts the DP into R x C blocks whose top row and left column are exact
//    forward values.  The traceback walks back from (m, n): for the block holding the
//    current cell it re-fills that block alone from its saved borders, recording the
//    predecessor codes (pointers), and walks inside the block until it leaves through
//    the top row or the left column -- into the next block.
// 3. The gradient is counted during the walk; the path itself is never stored.
//
// Bit-exactness with TracebackMode::Pointers: every re-filled cell is computed by the same
// expressions from the same predecessor values as the full fill, so it has the same value
// and the same code; the walk takes the same steps, ties included.
#pragma once
#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

template <class T>
struct GridGrad {
    // inputs
    const uint8_t* a; const uint8_t* b; int m, n, nalpha;
    const T* blk;                 // nalpha x nalpha, precision T
    T go_a, ge_a, go_b, ge_b;
    int R, C;                     // grid spacing (rows, columns)

    // outputs
    double score = 0;
    std::vector<double> gmat;     // nalpha * nalpha counts
    double g_go_a = 0, g_ge_a = 0, g_go_b = 0, g_ge_b = 0;
    long long cells_refilled = 0;
    int blocks_refilled = 0;

    static constexpr T NINF = -std::numeric_limits<T>::infinity();

    // grid storage: rows[r/R] = 3*(n+1) values (M, X, Y interleaved by state),
    // cols[c/C] = 3*(m+1) values
    int nr = 0, nc = 0;
    std::vector<T> grow, gcol;
    T& rowv(int gi, int s, int j) { return grow[((size_t)gi * 3 + s) * (n + 1) + j]; }
    T& colv(int gj, int s, int i) { return gcol[((size_t)gj * 3 + s) * (m + 1) + i]; }

    static unsigned char pick(T vm, T vx, T vy) {
        const unsigned char kxy = vx >= vy ? 1 : 2;
        return vm >= std::max(vx, vy) ? 0 : kxy;
    }

    void forward() {
        nr = m / R + 1; nc = n / C + 1;
        grow.assign((size_t)nr * 3 * (n + 1), NINF);
        gcol.assign((size_t)nc * 3 * (m + 1), NINF);
        std::vector<T> pM(n + 1), pX(n + 1), pY(n + 1), cM(n + 1), cX(n + 1), cY(n + 1);
        for (int j = 0; j <= n; ++j) {
            pM[j] = j == 0 ? T(0) : NINF; pX[j] = NINF;
            pY[j] = j == 0 ? NINF : -(go_a + static_cast<T>(j) * ge_a);
        }
        auto save = [&](int i, const T* M, const T* X, const T* Y) {
            if (i % R == 0)
                for (int j = 0; j <= n; ++j) {
                    rowv(i / R, 0, j) = M[j]; rowv(i / R, 1, j) = X[j]; rowv(i / R, 2, j) = Y[j];
                }
            for (int g = 0; g < nc; ++g) {
                const int j = g * C;
                colv(g, 0, i) = M[j]; colv(g, 1, i) = X[j]; colv(g, 2, i) = Y[j];
            }
        };
        save(0, pM.data(), pX.data(), pY.data());
        for (int i = 1; i <= m; ++i) {
            cM[0] = NINF; cX[0] = -(go_b + static_cast<T>(i) * ge_b); cY[0] = NINF;
            const T* srow = blk + (size_t)a[i - 1] * nalpha;
            for (int j = 1; j <= n; ++j) {
                cM[j] = std::max({pM[j - 1], pX[j - 1], pY[j - 1]}) + srow[b[j - 1]];
                cX[j] = std::max({pM[j] - go_b - ge_b, pX[j] - ge_b, pY[j] - go_b - ge_b});
                cY[j] = std::max({cM[j - 1] - go_a - ge_a, cX[j - 1] - go_a - ge_a, cY[j - 1] - ge_a});
            }
            save(i, cM.data(), cX.data(), cY.data());
            std::swap(pM, cM); std::swap(pX, cX); std::swap(pY, cY);
        }
        score = std::max({pM[n], pX[n], pY[n]});
    }

    // Re-fill block rows r0+1..r1, cols c0+1..c1 from row r0 / col c0; codes into d*.
    std::vector<T> bM, bX, bY;            // (R+1) x (C+1) values (only two rows needed; kept simple)
    std::vector<unsigned char> dM, dX, dY;
    int br0 = -1, bc0 = -1, bw = 0;
    void refill(int r0, int r1, int c0, int c1) {
        br0 = r0; bc0 = c0; bw = c1 - c0 + 1;
        const int h = r1 - r0 + 1;
        const size_t sz = (size_t)h * bw;
        if (bM.size() < sz) { bM.resize(sz); bX.resize(sz); bY.resize(sz); dM.resize(sz); dX.resize(sz); dY.resize(sz); }
        auto at = [&](int i, int j) { return (size_t)(i - r0) * bw + (j - c0); };
        for (int j = c0; j <= c1; ++j) {
            bM[at(r0, j)] = rowv(r0 / R, 0, j); bX[at(r0, j)] = rowv(r0 / R, 1, j); bY[at(r0, j)] = rowv(r0 / R, 2, j);
        }
        for (int i = r0 + 1; i <= r1; ++i) {
            bM[at(i, c0)] = colv(c0 / C, 0, i); bX[at(i, c0)] = colv(c0 / C, 1, i); bY[at(i, c0)] = colv(c0 / C, 2, i);
            const T* srow = blk + (size_t)a[i - 1] * nalpha;
            for (int j = c0 + 1; j <= c1; ++j) {
                const size_t k = at(i, j), up = at(i - 1, j), dg = at(i - 1, j - 1), lf = at(i, j - 1);
                const T dgM = bM[dg], dgX = bX[dg], dgY = bY[dg];
                bM[k] = std::max({dgM, dgX, dgY}) + srow[b[j - 1]];
                dM[k] = pick(dgM, dgX, dgY);
                const T ux = bM[up] - go_b - ge_b, vx = bX[up] - ge_b, wx = bY[up] - go_b - ge_b;
                bX[k] = std::max({ux, vx, wx}); dX[k] = pick(ux, vx, wx);
                const T uy = bM[lf] - go_a - ge_a, vy = bX[lf] - go_a - ge_a, wy = bY[lf] - ge_a;
                bY[k] = std::max({uy, vy, wy}); dY[k] = pick(uy, vy, wy);
            }
        }
        cells_refilled += (long long)(r1 - r0) * (c1 - c0);
        ++blocks_refilled;
    }

    void run() { forward(); walk(); }

    void walk() {
        gmat.assign((size_t)nalpha * nalpha, 0.0);
        int i = m, j = n;
        // 0 M, 1 X, 2 Y.  The end cell's state is the argmax at (m, n), read from the
        // first block re-filled (it holds (m, n)); -1 until then.
        int st = -1;
        while (true) {
            if (i == 0 && j == 0) break;
            if (i == 0) { for (; j > 0; --j) { g_ge_a -= 1; if (j == 1) g_go_a -= 1; } break; }
            if (j == 0) { for (; i > 0; --i) { g_ge_b -= 1; if (i == 1) g_go_b -= 1; } break; }
            const int r0 = (i - 1) / R * R, c0 = (j - 1) / C * C;
            refill(r0, std::min(r0 + R, m), c0, std::min(c0 + C, n));
            auto at = [&](int ii, int jj) { return (size_t)(ii - r0) * bw + (jj - c0); };
            if (st < 0) {   // end cell
                const size_t k = at(m, n);
                const T vm = bM[k], vx = bX[k], vy = bY[k];
                st = (vm >= vx && vm >= vy) ? 0 : (vx >= vy ? 1 : 2);
            }
            while (i > r0 && j > c0) {
                const size_t k = at(i, j);
                if (st == 0) {
                    gmat[(size_t)a[i - 1] * nalpha + b[j - 1]] += 1;
                    st = dM[k]; --i; --j;
                } else if (st == 1) {
                    const int p = dX[k];
                    g_ge_b -= 1; if (p != 1) g_go_b -= 1;
                    st = p; --i;
                } else {
                    const int p = dY[k];
                    g_ge_a -= 1; if (p != 2) g_go_a -= 1;
                    st = p; --j;
                }
            }
            // now on row r0 or column c0: continue in the next block (or the borders)
        }
    }
};
