#include "catch.hpp"
#include "seq_pair_batch.hpp"

TEST_CASE("header-only consumer links no dispatched kernels", "[header-only]") {
    for (SimdLevel level : detect_available_levels()) {
        const auto& k = level_kernels(static_cast<int>(level));
        REQUIRE(k.viterbi == nullptr);
        REQUIRE(k.viterbi_f == nullptr);
        REQUIRE(k.score == nullptr);
        REQUIRE(k.score_f == nullptr);
        REQUIRE(k.hb_sweep == nullptr);
        REQUIRE(k.hb_sweep_f == nullptr);
        REQUIRE(k.inter_fill == nullptr);
        REQUIRE(k.inter_fill_f == nullptr);
        REQUIRE(k.inter_soft == nullptr);
    }
}

template<class T, GapModel GM, AlignMode AM, AlignBand AB>
static void check_alignment() {
    AlignParams params(Alphabet::dna());
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) params.matrix.at(i, j) = i == j ? 2 : -1;
    params.gap_open_a = params.gap_open_b = 2;
    params.gap_extend_a = params.gap_extend_b = 1;
    for (auto tb : {TracebackMode::Scores, TracebackMode::Pointers,
                    TracebackMode::Hirschberg, TracebackMode::HirschbergPmax}) {
        if constexpr (AB == AlignBand::GuideBanded) {
            if (is_hirschberg(tb)) continue;
        }
        Aligner<GM, AM, AB, T> al;
        DpBufferT<T> buf;
        al.set_traceback(tb);
        al.set_hb_cutoff(2); // Exercise recursion, not just its pointer base case.
        al.set_problem("ACGT", "ACGT", params, AB == AlignBand::Full ? 0 : 1);
        al.compute_viterbi(buf);
        REQUIRE(al.score() == 8);
        REQUIRE(al.aligned(buf) == std::make_pair(std::string("ACGT"), std::string("ACGT")));
        auto grad = AlignParams::zeros_like(params);
        al.hard_grad(buf, grad);
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) REQUIRE(grad.matrix.at(i, j) == (i == j ? 1 : 0));
        REQUIRE(grad.gap_extend_a == 0);
        REQUIRE(grad.gap_extend_b == 0);
    }
}

TEMPLATE_TEST_CASE("header-only fallback aligns across modes and precisions",
                   "[header-only]", float, double) {
    check_alignment<TestType, GapModel::Affine, AlignMode::Global, AlignBand::Full>();
    check_alignment<TestType, GapModel::Affine, AlignMode::Local, AlignBand::Full>();
    check_alignment<TestType, GapModel::Linear, AlignMode::Global, AlignBand::Full>();
    check_alignment<TestType, GapModel::Linear, AlignMode::Local, AlignBand::Full>();
    check_alignment<TestType, GapModel::Affine, AlignMode::Global, AlignBand::GuideBanded>();
    check_alignment<TestType, GapModel::Affine, AlignMode::Local, AlignBand::GuideBanded>();
}
