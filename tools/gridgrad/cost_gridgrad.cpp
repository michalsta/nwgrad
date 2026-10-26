// Cost model: share of cells re-filled, and time of forward vs re-fill (scalar, both).
#include "gridgrad.hpp"
#include <chrono>
#include <cstdio>
#include <random>
#include <string>
#include <array>

int main() {
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> aa(0, 19), coin(0, 99);
    // BLOSUM-like: diagonal 5, off-diagonal -1 .. +1
    std::vector<float> blk(400);
    for (int i = 0; i < 20; ++i) for (int j = 0; j < 20; ++j) blk[i * 20 + j] = i == j ? 5.f : (float)(coin(rng) % 3 - 2);
    for (int L : {300, 1000, 3000, 10000}) {
        std::vector<uint8_t> A, B;
        for (int i = 0; i < L; ++i) A.push_back(aa(rng));
        for (auto c : A) { int r = coin(rng); if (r < 3) continue; if (r < 6) B.push_back(aa(rng)); B.push_back(coin(rng) < 30 ? aa(rng) : c); }
        for (int k : {8, 16, 32, 64}) {
            const int R = std::max(1, L / k), C = std::max(1, (int)B.size() / k);
            GridGrad<float> g{A.data(), B.data(), (int)A.size(), (int)B.size(), 20, blk.data(), 11, 1, 11, 1, R, C};
            auto t0 = std::chrono::steady_clock::now();
            g.forward();
            auto t1 = std::chrono::steady_clock::now();
            g.walk();
            auto t2 = std::chrono::steady_clock::now();
            const double f = std::chrono::duration<double>(t1 - t0).count();
            const double w = std::chrono::duration<double>(t2 - t1).count();
            const double cells = (double)g.m * g.n;
            std::printf("L=%5d k=%2d  refilled %.3f of cells in %3d blocks  fwd %.2f ns/cell  walk %.3f x fwd  grid %.1f KB\n",
                        L, k, g.cells_refilled / cells, g.blocks_refilled, f / cells * 1e9, w / f,
                        (g.grow.size() + g.gcol.size()) * 4 / 1024.0);
        }
    }
}
