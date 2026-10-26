// Grid-checkpoint gradient: exactness test and benchmark against nwgrad's own modes,
// all in one binary (nwgrad's level TUs linked in, the same ISA level for every arm).
//
//   bench_grid test [N]            bit-exact vs scalar Pointers, versions A and B
//   bench_grid bench [reps]        ns/cell per arm, related protein pairs
#include "nwgrad/aligner.hpp"
#include "grid2.hpp"
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>

GridLevel gk_level_baseline(); GridLevel gk_level_sse4(); GridLevel gk_level_avx2(); GridLevel gk_level_avx512();

static GridLevel pick_level() {
    // NWGRAD_ISA (nwgrad's own override) picks the level of both kernels, so one CPU can
    // be measured at several widths
    const std::string nw = backend_name(global_default_backend());
    GridLevel g;
    if (nw == "avx512") g = gk_level_avx512();
    else if (nw == "avx2") g = gk_level_avx2();
    else if (nw == "sse4") g = gk_level_sse4();
    else g = gk_level_baseline();
    if (nw != g.name && !(nw == "sse2" && std::string(g.name) == "baseline")) { std::fprintf(stderr, "level mismatch %s vs %s\n", nw.c_str(), g.name); std::exit(2); }
    return g;
}

static const char* AA = "ACDEFGHIKLMNPQRSTVWY";

template <class T>
static Grid2<T> make_grid(const GridLevel& lev, const std::vector<uint8_t>& A, const std::vector<uint8_t>& B,
                          const std::vector<T>& blk, const AlignParams& p, int var = 3) {
    Grid2<T> g{A.data(), B.data(), (int)A.size(), (int)B.size(), 20, blk.data(),
               (T)p.gap_open_a, (T)p.gap_extend_a, (T)p.gap_open_b, (T)p.gap_extend_b, nullptr};
    if constexpr (sizeof(T) == 4) g.fwd = lev.f[var]; else g.fwd = lev.d[var];
    g.Wf = lev.Wf; g.Wd = lev.Wd;
    return g;
}

template <class T>
static int check(const GridLevel& lev, std::mt19937& rng, int trial) {
    std::uniform_int_distribution<int> L(0, 700), aa(0, 19), coin(0, 99);
    const int ma = L(rng);
    std::string A, B;
    for (int i = 0; i < ma; ++i) A += AA[aa(rng)];
    if (coin(rng) < 70) { for (char c : A) { int r = coin(rng); if (r < 5) continue; if (r < 10) B += AA[aa(rng)]; B += (coin(rng) < 30 ? AA[aa(rng)] : c); } }
    else { const int nb = L(rng); for (int i = 0; i < nb; ++i) B += AA[aa(rng)]; }
    std::array<double, 400> src{};
    const int scheme = trial % 3;
    std::uniform_int_distribution<int> sc(-4, 6);
    std::uniform_real_distribution<double> sr(-4, 6);
    for (int i = 0; i < 20; ++i) for (int j = 0; j < 20; ++j)
        src[i * 20 + j] = scheme == 2 ? sr(rng) : (scheme == 0 ? (i == j ? 2.0 : -1.0) : sc(rng));
    AlignParams p(SubstMatrix(src.data()));
    if (scheme == 2) { p.gap_open_a = 3.1; p.gap_extend_a = 0.1; p.gap_open_b = 2.7; p.gap_extend_b = 0.3; }
    else if (scheme == 1) { p.gap_open_a = 0; p.gap_extend_a = 1; p.gap_open_b = 2; p.gap_extend_b = 0; }
    else { p.gap_open_a = 3; p.gap_extend_a = 1; p.gap_open_b = 2; p.gap_extend_b = 1; }

    Aligner<GapModel::Affine, AlignMode::Global, AlignBand::Full, T> al;
    al.alloc_buf(); al.set_kernel(kBackendScalar); al.set_traceback(TracebackMode::Pointers);
    al.set_problem(A, B, p); al.compute_viterbi();
    AlignParams g = p * 0.0; al.hard_grad(g);

    const Alphabet& alpha = p.matrix.alphabet();
    auto ea = alpha.encode(A), eb = alpha.encode(B);
    std::vector<T> blk(400);
    for (int q = 0; q < 400; ++q) blk[q] = static_cast<T>(src[q]);
    int bad = 0;
    std::uniform_int_distribution<int> kk(2, 40), mb(1, 12);
    for (int version = 0; version < 12; ++version) {
        Grid2<T> gg = make_grid<T>(lev, ea, eb, blk, p, version / 2);
        gg.k = kk(rng); gg.min_block = mb(rng);
        if (version % 2 == 0) gg.max_depth = 1;
        else gg.base_cells = std::uniform_int_distribution<int>(4, 600)(rng);
        gg.run();
        bool ok = gg.score == al.score() && gg.g_go_a == g.gap_open_a && gg.g_ge_a == g.gap_extend_a &&
                  gg.g_go_b == g.gap_open_b && gg.g_ge_b == g.gap_extend_b;
        for (int q = 0; q < 400; ++q) ok = ok && gg.gmat[q] == g.matrix.data()[q];
        if (!ok) {
            ++bad;
            std::printf("MISMATCH var %d %s T=%s trial %d m=%d n=%d k=%d mb=%d base=%lld score %.9g vs %.9g\n",
                        version / 2, version % 2 ? "B" : "A", sizeof(T) == 4 ? "f32" : "f64", trial, gg.m, gg.n, gg.k,
                        gg.min_block, gg.base_cells, gg.score, al.score());
        }
    }
    return bad;
}

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

template <class T>
static void bench(const GridLevel& lev, int reps) {
    std::mt19937 rng(11);
    std::uniform_int_distribution<int> aa(0, 19), coin(0, 99);
    // BLOSUM62, gaps 11/1 (the README example)
    static const char* order = "ARNDCQEGHILKMFPSTWYV";
    static const int b62[20][20] = {
        {4,-1,-2,-2,0,-1,-1,0,-2,-1,-1,-1,-1,-2,-1,1,0,-3,-2,0},{-1,5,0,-2,-3,1,0,-2,0,-3,-2,2,-1,-3,-2,-1,-1,-3,-2,-3},
        {-2,0,6,1,-3,0,0,0,1,-3,-3,0,-2,-3,-2,1,0,-4,-2,-3},{-2,-2,1,6,-3,0,2,-1,-1,-3,-4,-1,-3,-3,-1,0,-1,-4,-3,-3},
        {0,-3,-3,-3,9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1},{-1,1,0,0,-3,5,2,-2,0,-3,-2,1,0,-3,-1,0,-1,-2,-1,-2},
        {-1,0,0,2,-4,2,5,-2,0,-3,-3,1,-2,-3,-1,0,-1,-3,-2,-2},{0,-2,0,-1,-3,-2,-2,6,-2,-4,-4,-2,-3,-3,-2,0,-2,-2,-3,-3},
        {-2,0,1,-1,-3,0,0,-2,8,-3,-3,-1,-2,-1,-2,-1,-2,-2,2,-3},{-1,-3,-3,-3,-1,-3,-3,-4,-3,4,2,-3,1,0,-3,-2,-1,-3,-1,3},
        {-1,-2,-3,-4,-1,-2,-3,-4,-3,2,4,-2,2,0,-3,-2,-1,-2,-1,1},{-1,2,0,-1,-3,1,1,-2,-1,-3,-2,5,-1,-3,-1,0,-1,-3,-2,-2},
        {-1,-1,-2,-3,-1,0,-2,-3,-2,1,2,-1,5,0,-2,-1,-1,-1,-1,1},{-2,-3,-3,-3,-2,-3,-3,-3,-1,0,0,-3,0,6,-4,-2,-2,1,3,-1},
        {-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4,7,-1,-1,-4,-3,-2},{1,-1,1,0,-1,0,0,0,-1,-2,-2,0,-1,-2,-1,4,1,-3,-2,-2},
        {0,-1,0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1,1,5,-2,-2,0},{-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1,1,-4,-3,-2,11,2,-3},
        {-2,-2,-2,-3,-2,-1,-2,-3,2,-1,-1,-2,-1,3,-3,-2,-2,2,7,-1},{0,-3,-3,-3,-1,-2,-2,-3,-3,3,1,-2,1,-1,-2,-2,0,-3,-1,4}};
    std::array<double, 400> src{};
    for (int x = 0; x < 20; ++x) for (int y = 0; y < 20; ++y) {
        const int ix = (int)(std::strchr(order, AA[x]) - order), iy = (int)(std::strchr(order, AA[y]) - order);
        src[x * 20 + y] = b62[ix][iy];
    }
    AlignParams p(SubstMatrix(src.data()));
    p.gap_open_a = p.gap_open_b = 11; p.gap_extend_a = p.gap_extend_b = 1;
    std::vector<T> blk(400);
    for (int q = 0; q < 400; ++q) blk[q] = (T)src[q];
    const Alphabet& alpha = p.matrix.alphabet();

    for (int Lq : {300, 1000, 3000, 10000}) {
        const int npairs = std::max(1, 3000000 / (Lq * Lq) * 4);
        std::vector<std::vector<uint8_t>> As, Bs;
        double cells = 0;
        for (int t = 0; t < npairs; ++t) {
            std::string A, B;
            for (int i = 0; i < Lq; ++i) A += AA[aa(rng)];
            for (char c : A) { int r = coin(rng); if (r < 3) continue; if (r < 6) B += AA[aa(rng)]; B += (coin(rng) < 30 ? AA[aa(rng)] : c); }
            As.push_back(alpha.encode(A)); Bs.push_back(alpha.encode(B));
            cells += (double)As.back().size() * Bs.back().size();
        }
        using Al = Aligner<GapModel::Affine, AlignMode::Global, AlignBand::Full, T>;
        auto time_arm = [&](auto&& fn) {
            double best = 1e30;
            for (int r = 0; r < reps; ++r) {
                const double t0 = now();
                for (int t = 0; t < npairs; ++t) fn(t);
                best = std::min(best, now() - t0);
            }
            return best / cells * 1e9;
        };
        double sink = 0;
        std::printf("%s %s L=%5d pairs=%d\n", lev.name, sizeof(T) == 4 ? "f32" : "f64", Lq, npairs);
        double base_ns = 0;
        {
            Al al; al.alloc_buf();
            base_ns = time_arm([&](int t) { al.set_problem(As[t], Bs[t], p); sink += al.compute_score(); });
            std::printf("  %-28s %6.3f ns/cell  1.00x\n", "nwgrad score-only", base_ns);
        }
        for (auto [name, tb] : {std::pair{"nwgrad pointers", TracebackMode::Pointers},
                                std::pair{"nwgrad hirschberg", TracebackMode::Hirschberg},
                                std::pair{"nwgrad hirschberg_pmax", TracebackMode::HirschbergPmax}}) {
            Al al; al.alloc_buf(); al.set_traceback(tb);
            AlignParams g = p * 0.0;
            const double ns = time_arm([&](int t) { al.set_problem(As[t], Bs[t], p); al.compute_viterbi(); al.hard_grad(g); sink += g.gap_open_a; });
            std::printf("  %-28s %6.3f ns/cell  %4.2fx\n", name, ns, ns / base_ns);
        }
        struct Cfg { const char* name; int k; int depth; long long base; };
        for (Cfg c : {Cfg{"grid B k=32 v0 (as before)", 32, 0, 0}, Cfg{"grid B k=32 v1 MX", 32, 1, 0},
                      Cfg{"grid B k=32 v2 Two", 32, 2, 0}, Cfg{"grid B k=32 v3 Two+MX", 32, 3, 0},
                      Cfg{"grid B k=32 v4 Two+MX+Sub", 32, 4, 0}, Cfg{"grid B k=32 v5 Two+Sub", 32, 5, 0}}) {
            long long cv = 0, cs = 0; std::size_t peak = 0;
            const double ns = time_arm([&](int t) {
                Grid2<T> gg = make_grid<T>(lev, As[t], Bs[t], blk, p, c.depth);
                gg.k = c.k;
                gg.run(); sink += gg.score;
                cv += gg.cells_vec; cs += gg.cells_scalar; peak = std::max(peak, gg.peak_bytes);
            });
            std::printf("  %-28s %6.3f ns/cell  %4.2fx   vec %.3f scalar %.3f of cells, peak %.0f KB\n", c.name, ns, ns / base_ns,
                        cv / (cells * reps), cs / (cells * reps), peak / 1024.0);
        }
        if (sink == 12345.678) std::printf("!");
        std::fflush(stdout);
    }
}

// Forward kernel alone over the whole matrix vs nwgrad's score-only, across lengths.
template <class T>
static void scan(const GridLevel& lev, int reps) {
    std::mt19937 rng(5);
    std::uniform_int_distribution<int> aa(0, 19), coin(0, 99);
    std::array<double, 400> src{};
    for (int x = 0; x < 20; ++x) for (int y = 0; y < 20; ++y) src[x * 20 + y] = x == y ? 5 : (x + y) % 3 - 2;
    AlignParams p(SubstMatrix(src.data()));
    p.gap_open_a = p.gap_open_b = 11; p.gap_extend_a = p.gap_extend_b = 1;
    std::vector<T> blk(400);
    for (int q = 0; q < 400; ++q) blk[q] = (T)src[q];
    const Alphabet& alpha = p.matrix.alphabet();
    for (int Lq : {1000, 1500, 2000, 2500, 3000, 3500, 4000, 6000, 8000}) {
        std::string A, B;
        for (int i = 0; i < Lq; ++i) A += AA[aa(rng)];
        for (char c : A) { int r = coin(rng); if (r < 3) continue; if (r < 6) B += AA[aa(rng)]; B += (coin(rng) < 30 ? AA[aa(rng)] : c); }
        auto ea = alpha.encode(A), eb = alpha.encode(B);
        const double cells = (double)ea.size() * eb.size();
        auto best = [&](auto&& fn) { double b = 1e30; for (int r = 0; r < reps; ++r) { double t0 = now(); fn(); b = std::min(b, now() - t0); } return b / cells * 1e9; };
        Aligner<GapModel::Affine, AlignMode::Global, AlignBand::Full, T> al; al.alloc_buf();
        double sink = 0;
        const double ns_score = best([&] { al.set_problem(ea, eb, p); sink += al.compute_score(); });
        std::printf("%s %s L=%5d score-only %.3f", lev.name, sizeof(T) == 4 ? "f32" : "f64", Lq, ns_score);
        for (int var = 0; var < 6; ++var) { const int qq = 4;
            Grid2<T> g = make_grid<T>(lev, ea, eb, blk, p, var);
            std::vector<T> top(3 * (eb.size() + 1)), left(3 * (ea.size() + 1));
            const int m = (int)ea.size(), n = (int)eb.size(), W = sizeof(T) == 4 ? lev.Wf : lev.Wd;
            for (int j = 0; j <= n; ++j) { top[j] = j ? -INFINITY : 0; top[n + 1 + j] = -INFINITY; top[2 * (n + 1) + j] = j ? -(p.gap_open_a + j * p.gap_extend_a) : -INFINITY; }
            for (int i = 0; i <= m; ++i) { left[i] = i ? -INFINITY : 0; left[m + 1 + i] = i ? -(p.gap_open_b + i * p.gap_extend_b) : -INFINITY; left[2 * (m + 1) + i] = -INFINITY; }
            const int R = qq ? m / 32 : m + 1;
            std::vector<T> rows((size_t)32 * 3 * (n + 1)), cols((size_t)std::max(qq, 1) * 3 * (m + 1) * W);
            std::vector<int> colid(std::max(qq, 1) * W);
            const double ns = best([&] {
                GridFwdJob<T> J{ea.data(), eb.data(), blk.data(), 20, (T)p.gap_open_a, (T)p.gap_extend_a, (T)p.gap_open_b, (T)p.gap_extend_b,
                                0, m, 0, n, Line<T>{top.data(), (size_t)(n + 1), 1}, Line<T>{left.data(), (size_t)(m + 1), 1},
                                R, std::max(qq, 1), rows.data(), cols.data(), colid.data(), 0};
                g.fwd(J); sink += cols[5];
            });
            std::printf("  v%d %.3f (%.2fx)", var, ns, ns / ns_score);
        }
        std::printf("\n");
        if (sink == 1.2345) std::printf("!");
        std::fflush(stdout);
    }
}

int main(int argc, char** argv) {
    const GridLevel lev = pick_level();
    const std::string mode = argc > 1 ? argv[1] : "test";
    if (mode == "prof") {   // prof L f32|f64 k reps: grid B only, one related pair, for perf
        const int Lq = std::atoi(argv[2]); const bool f32 = std::string(argv[3]) == "f32";
        const int k = std::atoi(argv[4]), reps = std::atoi(argv[5]), var = argc > 6 ? std::atoi(argv[6]) : 3;
        std::mt19937 rng(5);
        std::uniform_int_distribution<int> aa(0, 19), coin(0, 99);
        std::array<double, 400> src{};
        for (int x = 0; x < 20; ++x) for (int y = 0; y < 20; ++y) src[x * 20 + y] = x == y ? 5 : (x + y) % 3 - 2;
        AlignParams p(SubstMatrix(src.data()));
        p.gap_open_a = p.gap_open_b = 11; p.gap_extend_a = p.gap_extend_b = 1;
        std::string A, B;
        for (int i = 0; i < Lq; ++i) A += AA[aa(rng)];
        for (char c : A) { int r = coin(rng); if (r < 3) continue; if (r < 6) B += AA[aa(rng)]; B += (coin(rng) < 30 ? AA[aa(rng)] : c); }
        auto ea = p.matrix.alphabet().encode(A), eb = p.matrix.alphabet().encode(B);
        auto go = [&](auto tag) {
            using T = decltype(tag);
            std::vector<T> blk(400); for (int q = 0; q < 400; ++q) blk[q] = (T)src[q];
            double sink = 0; const double t0 = now();
            for (int r = 0; r < reps; ++r) { Grid2<T> g = make_grid<T>(lev, ea, eb, blk, p, var); g.k = k; g.run(); sink += g.score; }
            std::printf("%.3f ns/cell (%g)\n", (now() - t0) / reps / ((double)ea.size() * eb.size()) * 1e9, sink);
        };
        if (f32) go(float{}); else go(double{});
        return 0;
    }
    if (mode == "scan") { const int reps = argc > 2 ? std::atoi(argv[2]) : 5; scan<float>(lev, reps); scan<double>(lev, reps); return 0; }
    if (mode == "test") {
        const int N = argc > 2 ? std::atoi(argv[2]) : 500;
        std::mt19937 rng(3);
        int bad = 0;
        for (int t = 0; t < N; ++t) { bad += check<float>(lev, rng, t); bad += check<double>(lev, rng, t); }
        std::printf("[%s] %d mismatches in %d comparisons (6 variants x versions A, B)\n", lev.name, bad, 24 * N);
        return bad != 0;
    }
    const int reps = argc > 2 ? std::atoi(argv[2]) : 3;
    const std::string prec = argc > 3 ? argv[3] : "both";
    if (prec != "f64") bench<float>(lev, reps);
    if (prec != "f32") bench<double>(lev, reps);
}
