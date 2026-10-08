// fill = "interpair" against each pair's own fill, in C++ — so every sanitizer job
// (ASan+UBSan, TSan, gcc and clang, all with _GLIBCXX_ASSERTIONS) runs the inter-pair
// kernels: DNA (blend tree) and protein (gathered profile and gathered soft weights),
// hard and soft, both gap models and modes, both precisions, B lengths mixed within a
// group, banded affine, and the streaming SeqPairBatch::align().  Hard results must be bit-identical, soft ones
// tolerance-equal (the soft path is never bit-exact).
//
// Written after the Python sanitizer jobs caught `&eqm[j * nm]` on an EMPTY vector
// (nm = 0 for alphabets over 8) in the inter-pair soft pass — UB that release builds
// never notice, and that no C++ test reached: the C++ add_many test aligned protein
// pairs in hard mode only.

#include "catch.hpp"
#include "align_params.hpp"
#include "aligner.hpp"
#include "seq_pair.hpp"
#include "seq_pair_batch.hpp"

#include <cmath>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed ? seed : 1) {}
    uint32_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return static_cast<uint32_t>(s >> 32); }
    int in(int lo, int hi) { return lo + static_cast<int>(next() % static_cast<uint32_t>(hi - lo + 1)); }
};

AlignParams params_for(const std::string& alpha, bool linear, uint64_t seed) {
    const Alphabet& al = Alphabet::get(alpha);
    SubstMatrix M(al);
    Rng r(seed);
    for (int x = 0; x < al.size(); ++x)
        for (int y = 0; y < al.size(); ++y)
            M.at(x, y) = (x == y) ? 2.0 : -1.0 + r.in(0, 1000) / 1000.0;
    const double go = linear ? 0.0 : 2.0;
    return AlignParams(M, go, 0.7, go * 1.3, 0.9);
}

void make_seqs(const std::string& alpha, int n, int alo, int ahi, int blo, int bhi, uint64_t seed,
               std::vector<std::string>& A, std::vector<std::string>& B) {
    Rng r(seed);
    A.clear(); B.clear();
    for (int i = 0; i < n; ++i) {
        std::string a, b;
        for (int k = r.in(alo, ahi); k > 0; --k) a += alpha[r.in(0, (int)alpha.size() - 1)];
        for (int k = r.in(blo, bhi); k > 0; --k) b += alpha[r.in(0, (int)alpha.size() - 1)];
        A.push_back(a); B.push_back(b);
    }
}

std::vector<std::string_view> views(const std::vector<std::string>& v) {
    return std::vector<std::string_view>(v.begin(), v.end());
}

template <class T>
void batch_case(const std::string& alpha, GapModel gm, AlignMode am, GradMode gd, uint64_t seed,
                int band) {
    std::vector<std::string> A, B;
    make_seqs(alpha, 61, 1, 30, 30, 38, seed, A, B);   // B lengths mixed within the cap
    const AlignParams p = params_for(alpha, gm == GapModel::Linear, seed);
    SeqPairBatchT<T> own(3, TracebackMode::Pointers), inter(3, TracebackMode::Pointers);
    own.set_fill(false, false);
    inter.set_fill(false, true);
    own.add_many(views(A), views(B), p, gm, am, gd);
    inter.add_many(views(A), views(B), p, gm, am, gd);
    own.score_and_grad();
    inter.score_and_grad();
    if (band > 0) { own.banded_grad(band); inter.banded_grad(band); }
    const AlignParams g0 = own.compute_grad(), g1 = inter.compute_grad();
    const int N = p.matrix.alphabet().size();
    for (size_t i = 0; i < A.size(); ++i) {
        if (gd == GradMode::Hard) REQUIRE(own[i].score() == inter[i].score());
        else REQUIRE(own[i].score() == Approx(inter[i].score()).epsilon(1e-11).margin(1e-11));
    }
    for (int x = 0; x < N; ++x)
        for (int y = 0; y < N; ++y) {
            if (gd == GradMode::Hard) REQUIRE(g0.matrix.at(x, y) == g1.matrix.at(x, y));
            else REQUIRE(g0.matrix.at(x, y) == Approx(g1.matrix.at(x, y)).epsilon(1e-10).margin(1e-9));
        }
}

}  // namespace

TEST_CASE("interpair: SeqPairBatch matches each pair's own fill", "[interpair]") {
    const std::string DNA = "ACGT", AA = "ACDEFGHIKLMNPQRSTVWY";
    uint64_t seed = 101;
    for (const std::string* alpha : {&DNA, &AA})
        for (GapModel gm : {GapModel::Affine, GapModel::Linear})
            for (AlignMode am : {AlignMode::Global, AlignMode::Local})
                for (GradMode gd : {GradMode::Hard, GradMode::Soft}) {
                    INFO("alpha " << *alpha << " linear " << (gm == GapModel::Linear)
                         << " local " << (am == AlignMode::Local) << " soft " << (gd == GradMode::Soft));
                    batch_case<double>(*alpha, gm, am, gd, ++seed, 0);
                    batch_case<float>(*alpha, gm, am, gd, ++seed, 0);
                    if (gm == GapModel::Affine && gd == GradMode::Hard) {
                        batch_case<double>(*alpha, gm, am, gd, ++seed, 3);
                        batch_case<float>(*alpha, gm, am, gd, ++seed, 3);
                    }
                }
}

TEST_CASE("interpair: SeqPairBatch::align matches its own path", "[interpair]") {
    const std::string DNA = "ACGT", AA = "ACDEFGHIKLMNPQRSTVWY";
    uint64_t seed = 301;
    for (const std::string* alpha : {&DNA, &AA})
        for (GapModel gm : {GapModel::Affine, GapModel::Linear})
            for (AlignMode am : {AlignMode::Global, AlignMode::Local})
                for (auto gd : {GradMode::Hard, GradMode::Soft}) {
                    std::vector<std::string> A, B;
                    make_seqs(*alpha, 53, 0, 30, 30, 38, ++seed, A, B);
                    const AlignParams p = params_for(*alpha, gm == GapModel::Linear, seed);
                    std::vector<ProblemInstance> probs;
                    for (size_t i = 0; i < A.size(); ++i) probs.push_back({A[i], B[i], {}});
                    SeqPairBatch own(gm, am, gd, 3), inter(gm, am, gd, 3);
                    own.set_fill(false, false);
                    inter.set_fill(false, true);
                    const BatchResult r0 = own.align(probs, p), r1 = inter.align(probs, p);
                    for (size_t i = 0; i < A.size(); ++i) {
                        if (gd == GradMode::Hard) REQUIRE(r0.scores[i] == r1.scores[i]);
                        else REQUIRE(r0.scores[i] == Approx(r1.scores[i]).epsilon(1e-11).margin(1e-11));
                    }
                }
}
