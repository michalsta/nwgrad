// Randomized batch regressions against direct public Aligner calls. The reference
// does not use BatchEngine, SeqPair, inter-pair scheduling or stored-path compression.
// Covers 192 stored-batch and 240 streaming configurations across precisions, types,
// parameter segments, schedules, fills, traceback modes, soft options and guide bands.
#include "catch.hpp"
#include "batch_engine.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {
struct Cfg {
    GradMode gd;
    TracebackMode tb;
    bool inter, rowwise, sorted;
    int soft_guide;        // 0 eager, 1 lazy, 2 posterior
    SoftImpl soft_impl;
    double temp;
    int hb_cutoff;
    int threads;
    bool protein;
    bool two_segments, scalar_second;
    int bw;
};

std::string rseq(std::mt19937& rng, const std::string& al, int len) {
    std::string s(static_cast<size_t>(len), al[0]);
    for (auto& c : s) c = al[rng() % al.size()];
    return s;
}
int rlen(std::mt19937& rng) {
    const unsigned r = rng() % 100;
    if (r < 4) return 0;
    if (r < 8) return 1;
    if (r < 90) return 2 + static_cast<int>(rng() % 50);
    return 60 + static_cast<int>(rng() % 120);
}

AlignParams rparams(std::mt19937& rng, const Alphabet& alpha, GapModel gm, int kind) {
    const int n = alpha.size();
    std::vector<double> m(static_cast<size_t>(n) * n);
    std::normal_distribution<double> nd(0.0, 1.0);
    for (int a = 0; a < n; ++a)
        for (int b = 0; b < n; ++b)
            m[static_cast<size_t>(a) * n + b] =
                kind == 0 ? (a == b ? 2.0 : -1.0)                       // ties
              : kind == 1 ? static_cast<double>(static_cast<int>(rng() % 9) - 3)
                          : nd(rng);
    const double go = gm == GapModel::Linear ? 0.0 : (kind == 0 ? 3.0 : 2.5);
    const double ge = kind == 2 ? 0.1 : 0.7;
    return AlignParams(SubstMatrix(m.data(), alpha), go, ge, go * 1.3, ge * 0.8);
}


bool same(double x, double y) { return x == y || (std::isnan(x) && std::isnan(y)); }

struct Expected {
    double score = 0;
    AlignParams grad;
    std::vector<int> guide;
    std::pair<std::string, std::string> aligned;
    std::pair<std::vector<int64_t>, std::vector<int64_t>> coordinates;
    explicit Expected(const Alphabet& a) : grad(a) {}
};

// Independent of batch storage, scheduling and path compression. Drive the public
// Aligner directly, then derive coordinates from its gapped strings and end cell.
template<class Al, class Buffer>
void capture_path(const Al& al, const Buffer& buf, Expected& out) {
    out.aligned = al.aligned(buf);
    const auto end = al.alignment_end();
    int64_t x = end.first, y = end.second;
    for (char c : out.aligned.first) x -= c != '-';
    for (char c : out.aligned.second) y -= c != '-';
    auto& [ci, cj] = out.coordinates;
    ci = {x}; cj = {y};
    int previous = -1;
    for (size_t k = 0; k < out.aligned.first.size(); ++k) {
        const int step = out.aligned.first[k] == '-' ? 2 : out.aligned.second[k] == '-' ? 1 : 0;
        if (previous != -1 && previous != step) { ci.push_back(x); cj.push_back(y); }
        x += step != 2; y += step != 1;
        previous = step;
    }
    ci.push_back(x); cj.push_back(y);
}

template<class T, GapModel GM, AlignMode AM, AlignBand AB>
Expected reference(std::string_view a, std::string_view b, const AlignParams& p,
                   const Cfg& c, int kernel, bool keep, int band = 0,
                   std::vector<int> guide = {}, bool streaming = false) {
    Aligner<GM, AM, AB, T> al;
    DpBufferT<T> buf;
    al.set_kernel(kernel);
    al.set_soft_impl(c.soft_impl); al.set_soft_temperature(c.temp);
    if constexpr (AB == AlignBand::Full) {
        al.set_traceback(c.tb); al.set_hb_cutoff(c.hb_cutoff); al.set_rowwise_full(c.rowwise);
    } else {
        al.set_traceback(TracebackMode::Scores);
    }
    al.set_problem(a, b, p, band, std::move(guide));
    Expected r(p.alphabet());
    const bool soft = c.gd == GradMode::Soft;
    const bool skip_viterbi = soft && (streaming || (!keep && AB == AlignBand::Full && c.soft_guide));
    if (c.gd == GradMode::None && !keep) {
        r.score = al.compute_score(buf);
        if (streaming) return r;
        al.compute_viterbi(buf);  // deferred guide, using the scoring parameters
        r.guide = al.guide_j_from_viterbi(buf);
        return r;
    }
    if (!skip_viterbi) {
        al.compute_viterbi(buf);
        r.score = al.score();
        r.guide = al.guide_j_from_viterbi(buf);
        if (keep) capture_path(al, buf, r);
        if (c.gd == GradMode::Hard) al.hard_grad(buf, r.grad);
    }
    if (soft) {
        al.set_posterior_guide(!streaming && !keep && AB == AlignBand::Full && c.soft_guide == 2);
        al.compute_forward_back(buf);
        r.score = al.log_z(); al.soft_grad(buf, r.grad);
        if (!streaming && !keep && AB == AlignBand::Full && c.soft_guide == 2)
            r.guide = al.posterior_guide();
    }
    return r;
}

void check_number(double expected, double actual, bool soft, double tolerance = 1e-10) {
    if (soft) CHECK(std::abs(expected - actual) <= tolerance * (1.0 + std::abs(expected)));
    else CHECK(same(expected, actual));
}
void check_grad(const AlignParams& expected, const AlignParams& actual, bool soft) {
    for (size_t k = 0; k < static_cast<size_t>(expected.matrix.size()) * expected.matrix.size(); ++k)
        check_number(expected.matrix.data()[k], actual.matrix.data()[k], soft, 1e-9);
    check_number(expected.gap_open_a, actual.gap_open_a, soft, 1e-9);
    check_number(expected.gap_extend_a, actual.gap_extend_a, soft, 1e-9);
    check_number(expected.gap_open_b, actual.gap_open_b, soft, 1e-9);
    check_number(expected.gap_extend_b, actual.gap_extend_b, soft, 1e-9);
}

// SIMD and per-pair soft passes may choose different posterior argmax columns
// on numerical ties. Validate the actual guide independently in log space before
// using that guide as the input to the banded reference (bands are different problems).
template<class T, GapModel GM, AlignMode AM>
void check_posterior_guide(std::string_view a, std::string_view b,
                           const AlignParams& p, const Cfg& c,
                           const std::vector<int>& guide) {
    Aligner<GM, AM, AlignBand::Full, T> al;
    DpBufferT<T> buf;
    al.set_soft_impl(SoftImpl::Log);
    al.set_soft_temperature(c.temp);
    al.set_problem(a, b, p);
    al.compute_forward_back(buf);
    const int m = static_cast<int>(a.size()), n = static_cast<int>(b.size());
    REQUIRE(guide.size() == static_cast<size_t>(m + 1));
    auto at = [n](const auto& table, int i, int j) {
        return table[static_cast<size_t>(i) * (n + 1) + j];
    };
    const double neg_inf = -std::numeric_limits<double>::infinity();
    auto logsum = [neg_inf](double x, double y, double z) {
        const double top = std::max({x, y, z});
        return top == neg_inf ? top : top + std::log(std::exp(x-top) + std::exp(y-top) + std::exp(z-top));
    };
    int previous = 0;
    for (int i = 0; i <= m; ++i) {
        REQUIRE(guide[i] >= previous);
        REQUIRE(guide[i] <= n);
        std::vector<double> cell(n+1), next_y(n+1, neg_inf), exit(n+1);
        for (int j = 0; j <= n; ++j) {
            if constexpr (GM == GapModel::Linear) {
                cell[j] = at(buf.F, i, j) + at(buf.B, i, j);
                if (j < n) next_y[j] = at(buf.F, i, j) - p.gap_extend_a/c.temp + at(buf.B, i, j+1);
            } else {
                cell[j] = logsum(at(buf.FM, i, j)+at(buf.BM, i, j),
                                 at(buf.FX, i, j)+at(buf.BX, i, j),
                                 at(buf.FY, i, j)+at(buf.BY, i, j));
                if (j < n) next_y[j] = at(buf.FY, i, j+1)+at(buf.BY, i, j+1);
            }
        }
        const double scale = *std::max_element(cell.begin(), cell.end());
        for (int j = 0; j <= n; ++j)
            exit[j] = std::exp(cell[j]-scale) - std::exp(next_y[j]-scale);
        const double best = *std::max_element(exit.begin(), exit.end());
        // If monotonicization kept the previous column, a row argmax at any
        // earlier column is sufficient. Otherwise this exact column must maximize.
        const double chosen = guide[i] == previous
            ? *std::max_element(exit.begin(), exit.begin()+previous+1) : exit[guide[i]];
        INFO("posterior row " << i << " chosen " << guide[i] << " mass " << chosen << " best " << best);
        CHECK(best - chosen <= 1e-10 * (1.0 + std::abs(best)));
        previous = guide[i];
    }
}

template <class T, GapModel GM, AlignMode AM>
void run_case(unsigned seed, const Cfg& c) {
    std::mt19937 rng(seed);
    const Alphabet& alpha = Alphabet::get(c.protein ? "ACDEFGHIKLMNPQRSTVWY" : "ACGT");
    const int kind = static_cast<int>(rng() % 3);
    const AlignParams p1 = rparams(rng, alpha, GM, kind);
    const AlignParams p1b = rparams(rng, alpha, GM, (kind + 1) % 3);
    const AlignParams p2 = rparams(rng, alpha, GM, (kind + 2) % 3);
    const int N1 = 1 + static_cast<int>(rng() % 30), N2 = c.two_segments ? 1 + static_cast<int>(rng() % 20) : 0;
    std::vector<std::string> A, B;
    const int lb = rlen(rng);
    for (int k = 0; k < N1 + N2; ++k) {
        A.push_back(rseq(rng, alpha.symbols(), rlen(rng)));
        B.push_back(rseq(rng, alpha.symbols(), rng() % 4 ? lb : rlen(rng)));
    }
    std::vector<std::string_view> A1(A.begin(), A.begin() + N1), B1(B.begin(), B.begin() + N1);
    std::vector<std::string_view> A2(A.begin() + N1, A.end()), B2(B.begin() + N1, B.end());
    const int k2 = c.scalar_second ? kBackendScalar : kBackendAuto;
    BatchEngine<T, GM, AM> eng(c.threads, c.gd, c.tb);
    eng.set_fill(c.rowwise, c.inter); eng.set_sorted_schedule(c.sorted);
    eng.set_soft_impl(c.soft_impl); eng.set_soft_temperature(c.temp);
    eng.set_soft_guide(c.soft_guide == 1, c.soft_guide == 2); eng.set_hb_cutoff(c.hb_cutoff);
    eng.add_many(A1, B1, p1);
    if (N2) eng.add_many(A2, B2, p1b, k2);
    const size_t N = A.size();
    REQUIRE(eng.size() == N);
    const bool soft = c.gd == GradMode::Soft;
    std::vector<Expected> expected;
    auto kernel = [&](size_t i) { return i < static_cast<size_t>(N1) ? kBackendAuto : k2; };
    double sum = 0;
    for (size_t i = 0; i < N; ++i) {
        expected.push_back(reference<T, GM, AM, AlignBand::Full>(A[i], B[i],
            i < static_cast<size_t>(N1) ? p1 : p1b, c, kernel(i), false));
        sum += expected.back().score;
    }
    check_number(sum, eng.score_and_grad(), soft);
    AlignParams plain(alpha), weighted(alpha);
    std::vector<double> weights(N);
    for (auto& w : weights) w = std::uniform_real_distribution<double>(-2, 2)(rng);
    for (size_t i = 0; i < N; ++i) {
        INFO("full pair " << i);
        REQUIRE(eng.score_valid(i)); REQUIRE(eng.path_valid(i)); REQUIRE_FALSE(eng.path_stored(i));
        REQUIRE(eng.grad_valid(i) == (c.gd != GradMode::None));
        check_number(expected[i].score, eng.score(i), soft);
        const bool pending = c.gd == GradMode::None || (soft && c.soft_guide == 1);
        REQUIRE(eng.guide_pending(i) == pending);
        if (!(soft && c.soft_guide == 1)) {
            const auto actual_guide = eng.guide_j(i);
            if (soft && c.soft_guide == 2) {
                check_posterior_guide<T, GM, AM>(A[i], B[i],
                    i < static_cast<size_t>(N1) ? p1 : p1b, c, actual_guide);
                expected[i].guide = actual_guide;
            } else CHECK(expected[i].guide == actual_guide);
        }
        if (c.gd != GradMode::None) {
            check_grad(expected[i].grad, eng.grad(i), soft);
            plain += expected[i].grad; weighted += weights[i] * expected[i].grad;
        }
    }
    if (c.gd != GradMode::None) {
        check_grad(plain, eng.compute_grad(), soft);
        check_grad(weighted, eng.weighted_grad(weights.data(), N), soft);
    }
    eng.set_params(p2);
    sum = 0;
    for (size_t i = 0; i < N; ++i) {
        REQUIRE_FALSE(eng.score_valid(i)); REQUIRE_FALSE(eng.grad_valid(i)); REQUIRE_FALSE(eng.path_stored(i));
        if (soft && c.soft_guide == 1)
            expected[i].guide = reference<T, GM, AM, AlignBand::Full>(A[i], B[i], p2, c, kernel(i), true).guide;
        expected[i] = reference<T, GM, AM, AlignBand::GuideBanded>(A[i], B[i], p2, c,
                            kernel(i), false, c.bw, expected[i].guide);
        sum += expected[i].score;
    }
    check_number(sum, eng.banded_grad(c.bw), soft);
    for (size_t i = 0; i < N; ++i) {
        INFO("banded pair " << i);
        check_number(expected[i].score, eng.score(i), soft);
        CHECK(expected[i].guide == eng.guide_raw(i));
        if (c.gd != GradMode::None) check_grad(expected[i].grad, eng.grad(i), soft);
    }
    for (bool banded : {false, true}) {
        sum = 0;
        for (size_t i = 0; i < N; ++i) {
            expected[i] = banded
                ? reference<T, GM, AM, AlignBand::GuideBanded>(A[i], B[i], p2, c, kernel(i), true, c.bw, expected[i].guide)
                : reference<T, GM, AM, AlignBand::Full>(A[i], B[i], p2, c, kernel(i), true);
            sum += expected[i].score;
        }
        check_number(sum, banded ? eng.banded_grad(c.bw, true) : eng.score_and_grad(true), soft);
        for (size_t i = 0; i < N; ++i) {
            INFO("stored paths pair " << i << " banded " << banded);
            REQUIRE(eng.path_stored(i));
            check_number(expected[i].score, eng.score(i), soft);
            CHECK(expected[i].aligned == eng.aligned(i));
            CHECK(expected[i].coordinates == eng.coordinates(i));
            CHECK(expected[i].guide == eng.guide_raw(i));
            if (c.gd != GradMode::None) check_grad(expected[i].grad, eng.grad(i), soft);
        }
    }
    eng.drop_paths();
    for (size_t i = 0; i < N; ++i) {
        REQUIRE_FALSE(eng.path_stored(i)); REQUIRE(eng.score_valid(i)); REQUIRE(eng.path_valid(i));
        REQUIRE(eng.grad_valid(i) == (c.gd != GradMode::None));
    }
}
template <class T, GapModel GM, AlignMode AM>
void sweep(int n) {
    std::mt19937 rng(1000u + static_cast<unsigned>(GM) * 10u + static_cast<unsigned>(AM) +
                     (std::is_same_v<T, float> ? 100u : 0u));
    const GradMode gds[] = {GradMode::Hard, GradMode::Soft, GradMode::None};
    const TracebackMode tbs[] = {TracebackMode::Default, TracebackMode::Pointers,
                                 TracebackMode::Scores, TracebackMode::Hirschberg};
    const SoftImpl sis[] = {SoftImpl::Scaled, SoftImpl::ScaledOrLog, SoftImpl::Log};
    for (int k = 0; k < n; ++k) {
        Cfg c;
        c.gd = gds[k % 3];
        c.tb = tbs[rng() % 4];
        c.inter = rng() % 3 != 0;
        c.rowwise = rng() % 3 == 0;
        c.sorted = rng() % 3 == 0;
        c.soft_guide = static_cast<int>(rng() % 3);
        c.soft_impl = sis[rng() % 3];
        c.temp = rng() % 3 == 0 ? 0.5 : 1.0;
        c.hb_cutoff = rng() % 2 ? 512 : 1 + static_cast<int>(rng() % 24);
        c.threads = 1 + static_cast<int>(rng() % 4);
        c.protein = rng() % 4 == 0;
        c.two_segments = rng() % 2;
        c.scalar_second = rng() % 2;
        c.bw = 1 + static_cast<int>(rng() % 8);
        const unsigned seed = static_cast<unsigned>(rng());
        INFO("case " << k << " seed " << seed << " gd " << static_cast<int>(c.gd) << " tb "
             << static_cast<int>(c.tb) << " inter " << c.inter << " rowwise " << c.rowwise
             << " sorted " << c.sorted << " guide " << c.soft_guide << " impl "
             << static_cast<int>(c.soft_impl) << " T " << c.temp << " cutoff " << c.hb_cutoff
             << " threads " << c.threads << " protein " << c.protein << " segs "
             << c.two_segments << " bw " << c.bw);
        run_case<T, GM, AM>(seed, c);
    }
}

} // namespace

TEST_CASE("BatchEngine == SeqPairBatch, double", "[batch_engine]") {
    sweep<double, GapModel::Affine, AlignMode::Global>(24);
    sweep<double, GapModel::Affine, AlignMode::Local>(24);
    sweep<double, GapModel::Linear, AlignMode::Global>(24);
    sweep<double, GapModel::Linear, AlignMode::Local>(24);
}

TEST_CASE("BatchEngine == SeqPairBatch, float32", "[batch_engine]") {
    sweep<float, GapModel::Affine, AlignMode::Global>(24);
    sweep<float, GapModel::Affine, AlignMode::Local>(24);
    sweep<float, GapModel::Linear, AlignMode::Global>(24);
    sweep<float, GapModel::Linear, AlignMode::Local>(24);
}

// Streaming regressions against direct Aligners, including guided band zero.
namespace {

template <class T, GapModel GM, AlignMode AM>
void stream_sweep(int n) {
    std::mt19937 rng(77u + static_cast<unsigned>(GM) * 10u + static_cast<unsigned>(AM) +
                     (std::is_same_v<T, float> ? 100u : 0u));
    const GradMode gds[] = {GradMode::Hard, GradMode::Soft, GradMode::None};
    for (int k = 0; k < n; ++k) {
        const int gi = k % 3;
        const bool protein = rng() % 4 == 0, inter = rng() % 3 != 0;
        const int band = rng() % 3 == 0 ? 1 + static_cast<int>(rng() % 6) : 0;
        const bool guides = rng() % 3 == 0;
        const int threads = 1 + static_cast<int>(rng() % 4);
        const double temp = rng() % 3 == 0 ? 0.5 : 1.0;
        const Alphabet& alpha = Alphabet::get(protein ? "ACDEFGHIKLMNPQRSTVWY" : "ACGT");
        const AlignParams p = rparams(rng, alpha, GM, static_cast<int>(rng() % 3));
        const int N = 1 + static_cast<int>(rng() % 40), lb = rlen(rng);
        std::vector<std::string> A, B;
        for (int i = 0; i < N; ++i) {
            A.push_back(rseq(rng, alpha.symbols(), rlen(rng)));
            B.push_back(rseq(rng, alpha.symbols(), rng() % 4 ? lb : rlen(rng)));
        }
        std::vector<ProblemInstance> probs;
        for (int i = 0; i < N; ++i) {
            ProblemInstance pi{A[i], B[i], {}};
            if (guides) {   // a diagonal guide, as a caller's aligned strings would give
                const int m = static_cast<int>(A[i].size()), nb = static_cast<int>(B[i].size());
                for (int r = 0; r <= m; ++r) pi.guide_j.push_back(m ? r * nb / m : 0);
                if (!pi.guide_j.empty()) pi.guide_j.back() = nb;
            }
            probs.push_back(pi);
        }
        INFO("stream case " << k << " gd " << gi << " band " << band << " guides " << guides
             << " inter " << inter << " threads " << threads << " protein " << protein);
        BatchEngine<T, GM, AM> eng(threads, gds[gi]);
        eng.set_fill(false, inter); eng.set_soft_temperature(temp);
        BatchResult ro(alpha);
        Cfg c{}; c.gd = gds[gi]; c.temp = temp; c.soft_impl = SoftImpl::Scaled;
        c.tb = TracebackMode::Default; c.hb_cutoff = 512;
        for (const auto& problem : probs) {
            auto r = band > 0 || guides
                ? reference<T, GM, AM, AlignBand::GuideBanded>(problem.seq_a, problem.seq_b,
                      p, c, kBackendAuto, false, band, problem.guide_j, true)
                : reference<T, GM, AM, AlignBand::Full>(problem.seq_a, problem.seq_b,
                      p, c, kBackendAuto, false, 0, {}, true);
            ro.scores.push_back(r.score);
            ro.grad += r.grad;
        }
        const BatchResult re = eng.align_stream(probs, p, band);
        REQUIRE(ro.scores.size() == re.scores.size());
        for (int i = 0; i < N; ++i) {
            INFO("problem " << i);
            if (gds[gi] == GradMode::Soft)
                CHECK(std::abs(ro.scores[i] - re.scores[i]) <= 1e-11 * (1 + std::abs(ro.scores[i])));
            else
                CHECK(same(ro.scores[i], re.scores[i]));
        }
        for (int q = 0; q < alpha.size() * alpha.size(); ++q) {
            const double x = ro.grad.matrix.data()[q], y = re.grad.matrix.data()[q];
            if (gds[gi] == GradMode::Soft) CHECK(std::abs(x - y) <= 1e-9 * (1 + std::abs(x)));
            else                           CHECK(same(x, y));
        }
    }
}

} // namespace

TEST_CASE("BatchEngine::align_stream matches direct Aligners", "[batch_engine]") {
    stream_sweep<double, GapModel::Affine, AlignMode::Global>(30);
    stream_sweep<double, GapModel::Affine, AlignMode::Local>(30);
    stream_sweep<double, GapModel::Linear, AlignMode::Global>(30);
    stream_sweep<double, GapModel::Linear, AlignMode::Local>(30);
    stream_sweep<float, GapModel::Affine, AlignMode::Global>(30);
    stream_sweep<float, GapModel::Affine, AlignMode::Local>(30);
    stream_sweep<float, GapModel::Linear, AlignMode::Global>(30);
    stream_sweep<float, GapModel::Linear, AlignMode::Local>(30);
}
