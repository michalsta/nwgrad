// Grid forward kernel interface (plain data only; std::simd stays inside the level TUs).
#pragma once
#include <cstddef>

// A line of saved values: value(state, idx) = p[state * sstride + idx * estride].
// state 0 = M, 1 = X, 2 = Y.
template <class T>
struct Line {
    const T* p = nullptr;
    std::size_t sstride = 0, estride = 1;
    T operator()(int s, int idx) const { return p[(std::size_t)s * sstride + (std::size_t)idx * estride]; }
    Line shifted(int d) const { Line l = *this; l.p += (std::ptrdiff_t)d * (std::ptrdiff_t)estride; return l; }
};

// Forward fill of the rectangle rows r0..r1 x cols c0..c1 (affine Global recurrence,
// opens >= 0: the D/X form), from its top row and left column.  Saves:
//   rows: every R-th row r0 + t*R (t >= 1, < r1), dense [t-1][3][w+1] (index j - c0)
//   cols: q segment-aligned column sets, [t][3][h+1][W] (index i - r0, then lane);
//         colid[t*W + l] = the absolute column of set t, lane l (or -1 if >= c1)
template <class T>
struct GridFwdJob {
    const unsigned char* a; const unsigned char* b;
    const T* blk; int nalpha;
    T go_a, ge_a, go_b, ge_b;
    int r0, r1, c0, c1;
    Line<T> top, left;
    int R, q;
    T* rows; T* cols; int* colid;
    int W;   // out: lane count used
};

template <class T> using GridFwdFn = void (*)(GridFwdJob<T>&);
// Kernel variants, index Two * 2 + MX (see gk_impl.inl); 0 is the plain copy of nwgrad's kernel;
// 4 = Two + MX + Sub, 5 = Two + Sub.
struct GridLevel {
    const char* name;
    int Wf, Wd;
    GridFwdFn<float> f[6]; GridFwdFn<double> d[6];
};
