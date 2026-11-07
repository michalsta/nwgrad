// The persistent worker pool (parallel.hpp): run_workers_guarded hands multi-threaded
// calls to parked threads.  Here, unlike from Python (where the GIL serialises calls),
// calls really overlap: several threads dispatch at once, so all but one take the
// spawn fallback while the pool is busy, and a call nested inside a worker must not
// deadlock.  Results must equal the one-thread / spawn answers exactly.
#include "catch.hpp"
#include "align_params.hpp"
#include "seq_pair_batch.hpp"
#include <array>
#include <random>
#include <stdexcept>
#include <thread>

namespace {

long run_sum(int nt, size_t N) {
    std::atomic<size_t> idx{0};
    std::atomic<long> sum{0};
    run_workers_guarded(nt, [&] {
        for (size_t i; (i = idx.fetch_add(1)) < N;) sum += static_cast<long>(i);
    });
    return sum.load();
}

long tri(size_t N) { return N ? static_cast<long>(N * (N - 1) / 2) : 0; }

struct PoolSwitch {
    bool saved = thread_pool_enabled();
    explicit PoolSwitch(bool on) { set_thread_pool_enabled(on); }
    ~PoolSwitch() { set_thread_pool_enabled(saved); }
};

AlignParams unit_params() {
    std::array<double, 400> src{};
    for (int i = 0; i < 20; ++i) src[i * 20 + i] = 1.0;
    AlignParams p(SubstMatrix(src.data()));
    p.gap_extend_a = p.gap_extend_b = 1.0;
    p.gap_open_a = p.gap_open_b = 3.0;
    return p;
}

std::vector<std::string> random_seqs(size_t n, unsigned seed) {
    static const char aa[] = "ACDEFGHIKLMNPQRSTVWY";
    std::mt19937 rng(seed);
    std::vector<std::string> out(n);
    for (auto& s : out) {
        s.resize(5 + rng() % 40);
        for (auto& c : s) c = aa[rng() % 20];
    }
    return out;
}

std::vector<std::string_view> views(const std::vector<std::string>& v) {
    return {v.begin(), v.end()};
}

}  // namespace

TEST_CASE("thread pool: every item once, at every thread count", "[pool]") {
    PoolSwitch on(true);
    for (int nt : {1, 2, 3, 4, 8, 16})
        for (size_t N : {size_t(0), size_t(1), size_t(7), size_t(1000)})
            REQUIRE(run_sum(nt, N) == tri(N));
}

TEST_CASE("thread pool: overlapping calls from several threads, and nested calls", "[pool]") {
    PoolSwitch on(true);
    std::atomic<int> bad{0};
    std::vector<std::thread> ts;
    for (int t = 0; t < 8; ++t)
        ts.emplace_back([&, t] {
            for (int r = 0; r < 500; ++r) {
                const size_t N = 1 + static_cast<size_t>(r % 50);
                if (run_sum(2 + (t + r) % 7, N) != tri(N)) ++bad;
            }
        });
    for (auto& t : ts) t.join();
    REQUIRE(bad.load() == 0);
    std::atomic<long> nested{0};
    run_workers_guarded(4, [&] { nested += run_sum(3, 100); });
    REQUIRE(nested.load() % tri(100) == 0);
    REQUIRE(nested.load() >= tri(100));
}

TEST_CASE("thread pool: a worker's exception reaches the caller, and the pool survives it", "[pool]") {
    PoolSwitch on(true);
    REQUIRE_THROWS_AS(run_workers_guarded(4, [] { throw std::runtime_error("boom"); }),
                      std::runtime_error);
    REQUIRE(run_sum(4, 100) == tri(100));
}

TEST_CASE("thread pool: batch results identical pool vs spawn, with overlapping calls", "[pool]") {
    const AlignParams p = unit_params();
    const auto A = random_seqs(300, 1), B = random_seqs(300, 2);
    auto run = [&](int nt) {
        SeqPairBatch b(views(A), views(B), p, GapModel::Affine, AlignMode::Local, GradMode::Hard, nt);
        const double total = b.score_and_grad(true);
        std::vector<double> s = b.scores();
        s.push_back(total);
        return s;
    };
    std::vector<double> ref;
    {
        PoolSwitch off(false);
        ref = run(1);
        REQUIRE(run(8) == ref);
    }
    PoolSwitch on(true);
    for (int nt : {2, 4, 8, 16}) REQUIRE(run(nt) == ref);
    std::atomic<int> bad{0};
    std::vector<std::thread> ts;
    for (int t = 0; t < 4; ++t)
        ts.emplace_back([&, t] {
            for (int r = 0; r < 10; ++r)
                if (run(2 + (t + r) % 6) != ref) ++bad;
        });
    for (auto& t : ts) t.join();
    REQUIRE(bad.load() == 0);
}
