// ── Viterbi, then forward-backward, then the Viterbi consumers ──────────────
//
// SeqPair's soft mode runs compute_viterbi() (for the guide and the alignment),
// then compute_forward_back() (for log Z and the soft gradient), and a caller may
// then ask for aligned() or hard_grad().  The forward-backward fill once cleared
// tables_striped_ / pointers_ — flags that describe how the RETAINED Viterbi state
// is laid out — so afterwards a Pointers traceback walked unallocated score tables
// (segfault) and a striped Scores traceback read its tables row-major (wrong
// alignment).  Each consumer here must return exactly what it returned before the
// forward-backward ran, and the soft results must be exactly those of an aligner
// that never ran Viterbi at all, through both the own-buffer and the
// external-buffer APIs.

#include "catch.hpp"

#include <string>
#include <vector>

#include "align_params.hpp"
#include "aligner.hpp"
#include "simd_levels.hpp"

namespace {

std::vector<int> backends_under_test() {
    std::vector<int> v{kBackendScalar};
    for (SimdLevel l : detect_available_levels()) v.push_back(static_cast<int>(l));
    return v;
}

bool same_grad(const AlignParams& x, const AlignParams& y) {
    const int N = x.matrix.alphabet().size();
    for (int a = 0; a < N; ++a)
        for (int b = 0; b < N; ++b)
            if (x.matrix.at(a, b) != y.matrix.at(a, b)) return false;
    return x.gap_open_a == y.gap_open_a && x.gap_extend_a == y.gap_extend_a &&
           x.gap_open_b == y.gap_open_b && x.gap_extend_b == y.gap_extend_b;
}

template <GapModel GM, AlignMode AM, AlignBand AB, class T>
void check(const std::string& A, const std::string& B, const AlignParams& p,
           int backend, TracebackMode tb, int hb_cutoff, bool external) {
    using Al = Aligner<GM, AM, AB, T>;
    const int band = (AB == AlignBand::GuideBanded) ? 3 : 0;
    const auto ea = p.matrix.alphabet().encode(A);
    const auto eb = p.matrix.alphabet().encode(B);

    Al al;
    al.set_kernel(backend);
    al.set_traceback(tb);
    al.set_hb_cutoff(hb_cutoff);
    DpBufferT<T> buf;
    if (!external) al.alloc_buf();
    al.set_problem(ea, eb, p, band);

    auto viterbi = [&] { external ? al.compute_viterbi(buf) : al.compute_viterbi(); };
    auto fwdbwd  = [&] { external ? al.compute_forward_back(buf) : al.compute_forward_back(); };
    auto aligned = [&] { return external ? al.aligned(buf) : al.aligned(); };
    auto guide   = [&] { return external ? al.guide_j_from_viterbi(buf) : al.guide_j_from_viterbi(); };
    auto hard    = [&] {
        AlignParams g(p.matrix.alphabet());
        external ? al.hard_grad(buf, g) : al.hard_grad(g);
        return g;
    };
    auto soft    = [&] {
        AlignParams g(p.matrix.alphabet());
        external ? al.soft_grad(buf, g) : al.soft_grad(g);
        return g;
    };

    viterbi();
    const double vscore = al.score();
    const auto aln0 = aligned();
    const auto gj0  = guide();
    const auto hg0  = hard();

    fwdbwd();
    REQUIRE(al.score() == al.log_z());
    CHECK(aligned() == aln0);
    CHECK(guide() == gj0);
    CHECK(same_grad(hard(), hg0));
    const auto sg = soft();

    // Reference: forward-backward alone, with no Viterbi state at all.
    Al ref;
    ref.set_kernel(backend);
    ref.alloc_buf();
    ref.set_problem(ea, eb, p, band);
    ref.compute_forward_back();
    AlignParams sref(p.matrix.alphabet());
    ref.soft_grad(sref);
    CHECK(al.log_z() == ref.log_z());
    CHECK(same_grad(sg, sref));

    // And Viterbi again on top: the hard path must not have been perturbed either.
    viterbi();
    CHECK(al.score() == vscore);
    CHECK(aligned() == aln0);
    CHECK(same_grad(hard(), hg0));
}

template <GapModel GM, AlignMode AM, class T>
void check_all(const std::string& A, const std::string& B, const AlignParams& p) {
    for (int backend : backends_under_test())
        for (bool external : {false, true}) {
            INFO("backend=" << backend_name(backend) << " external=" << external
                 << " A=" << A << " B=" << B);
            check<GM, AM, AlignBand::Full, T>(A, B, p, backend, TracebackMode::Scores, 512, external);
            check<GM, AM, AlignBand::GuideBanded, T>(A, B, p, backend, TracebackMode::Scores, 512, external);
            if constexpr (GM == GapModel::Affine) {
                check<GM, AM, AlignBand::Full, T>(A, B, p, backend, TracebackMode::Pointers, 512, external);
                // Above and below the cutoff: cutoff 512 runs these short pairs as
                // Pointers, cutoff 1 keeps the recursion under test.
                check<GM, AM, AlignBand::Full, T>(A, B, p, backend, TracebackMode::Hirschberg, 512, external);
                check<GM, AM, AlignBand::Full, T>(A, B, p, backend, TracebackMode::Hirschberg, 1, external);
            }
        }
}

}  // namespace

TEST_CASE("Viterbi consumers survive a subsequent forward-backward", "[soft][traceback]") {
    const Alphabet& al = Alphabet::get("ACGT");
    SubstMatrix M(al);
    for (int x = 0; x < al.size(); ++x)
        for (int y = 0; y < al.size(); ++y) M.at(x, y) = (x == y) ? 2.0 : -1.0;
    const AlignParams p(M, 2.0, 1.0, 2.5, 0.75);

    const std::vector<std::pair<std::string, std::string>> pairs{
        {"ACGT", "ACGT"},
        {"ACGTTGCAACGTAGCTAGCTAGGATCCAGT", "ACGTGCAACGTAGCTTAGCTAGGATCAGT"},
        {"GATTACAGATTACAGATTACA", "TTGATTACAGGATTCAGATTACATT"},
        {"A", "ACGT"},
        {"ACGT", "A"},
    };
    for (const auto& [A, B] : pairs) {
        check_all<GapModel::Affine, AlignMode::Global, double>(A, B, p);
        check_all<GapModel::Affine, AlignMode::Local,  double>(A, B, p);
        check_all<GapModel::Affine, AlignMode::Global, float >(A, B, p);
        check_all<GapModel::Affine, AlignMode::Local,  float >(A, B, p);
        check_all<GapModel::Linear, AlignMode::Global, double>(A, B, p);
        check_all<GapModel::Linear, AlignMode::Local,  double>(A, B, p);
        check_all<GapModel::Linear, AlignMode::Global, float >(A, B, p);
        check_all<GapModel::Linear, AlignMode::Local,  float >(A, B, p);
    }
}
