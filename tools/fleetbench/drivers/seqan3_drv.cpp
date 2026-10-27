// SeqAn3 align_pairwise.  One binary per (alphabet, score type, vectorised): compile with
//   -DALPH_DNA (else aa27 + BLOSUM62)  -DSCORE_T=int16_t|int32_t|int64_t|float|double
//   -DVECTORISED (inter-sequence SIMD; integer scores only)
// mode / output / threads are runtime choices.  Threads > 1 use SeqAn3's own
// align_cfg::parallel{n}; otherwise no parallel config at all.
//
// Gaps: SeqAn3 scores a gap of length k as open + k*extension -- nwgrad's go + ge*k --
// so open = -go, extension = -ge (linear: open 0).
// SeqAn3 3.4 walks the traceback only when begin positions are requested: with
// output_alignment alone it fills the trace matrix and returns an EMPTY alignment.
#include "drv_common.hpp"

#include <seqan3/alignment/configuration/all.hpp>
#include <seqan3/alignment/pairwise/align_pairwise.hpp>
#include <seqan3/alignment/scoring/aminoacid_scoring_scheme.hpp>
#include <seqan3/alignment/scoring/nucleotide_scoring_scheme.hpp>
#include <seqan3/alphabet/aminoacid/aa27.hpp>
#include <seqan3/alphabet/nucleotide/dna4.hpp>
#include <seqan3/utility/views/zip.hpp>

#ifndef SCORE_T
#define SCORE_T int32_t
#endif
#define STR2(x) #x
#define STR(x) STR2(x)

#ifdef ALPH_DNA
using alph_t = seqan3::dna4;
#else
using alph_t = seqan3::aa27;
#endif

template<bool Path, class Cfg>
static std::vector<double> run_cfg(const Cfg& cfg, std::vector<std::vector<alph_t>>& A,
                                   std::vector<std::vector<alph_t>>& B, long& sink) {
    std::vector<double> scores(A.size());
    auto pairs = seqan3::views::zip(A, B);
    for (auto&& res : seqan3::align_pairwise(pairs, cfg)) {
        scores[res.sequence1_id()] = res.score();
        if constexpr (Path) sink += std::get<0>(res.alignment()).size();
    }
    return scores;
}

template<class Base>
static Result run_out(const Args& a, const Base& base, std::vector<std::vector<alph_t>>& A,
                      std::vector<std::vector<alph_t>>& B) {
    long sink = 0;
    auto go = [&]<bool Path>(const auto& cfg, std::bool_constant<Path>) {
        if (a.threads > 1) {
            auto c = cfg | seqan3::align_cfg::parallel{(uint32_t)a.threads};
            return time_batch(a, [&] { return run_cfg<Path>(c, A, B, sink); });
        }
        return time_batch(a, [&] { return run_cfg<Path>(cfg, A, B, sink); });
    };
    if (a.out == "score")
        return go(base | seqan3::align_cfg::output_score{}, std::false_type{});
    return go(base | seqan3::align_cfg::output_score{} | seqan3::align_cfg::output_alignment{}
                   | seqan3::align_cfg::output_begin_position{}
                   | seqan3::align_cfg::output_end_position{}, std::true_type{});
}

template<class Method>
static Result run_method(const Args& a, const Method& method, const Problem& p,
                         std::vector<std::vector<alph_t>>& A,
                         std::vector<std::vector<alph_t>>& B) {
#ifdef ALPH_DNA
    seqan3::nucleotide_scoring_scheme<SCORE_T> scheme{
        seqan3::match_score{(SCORE_T)p.matrix[0][0]}, seqan3::mismatch_score{(SCORE_T)p.matrix[0][1]}};
#else
    seqan3::aminoacid_scoring_scheme<SCORE_T> scheme{seqan3::aminoacid_similarity_matrix::blosum62};
#endif
    int go = a.affine() ? p.go : 0;
    auto base = method
              | seqan3::align_cfg::scoring_scheme{scheme}
              | seqan3::align_cfg::gap_cost_affine{seqan3::align_cfg::open_score{-go},
                                                   seqan3::align_cfg::extension_score{-p.ge}}
              | seqan3::align_cfg::score_type<SCORE_T>{}
              | seqan3::align_cfg::output_sequence1_id{}
#ifdef VECTORISED
              | seqan3::align_cfg::vectorised{}
#endif
        ;
    return run_out(a, base, A, B);
}

static Result run(const Args& a, const std::vector<std::string>& SA,
                  const std::vector<std::string>& SB, const Problem& p) {
    if (a.out != "score" && a.out != "path") throw Unsupported("SeqAn3 has no gradient");
    if (a.dtype != STR(DTYPE_NAME)) throw Unsupported("this binary is " STR(DTYPE_NAME));
#ifdef ALPH_DNA
    if (!p.is_dna()) throw Unsupported("DNA binary");
#else
    if (!p.is_blosum62()) throw Unsupported("protein binary");
#endif
    std::vector<std::vector<alph_t>> A, B;
    for (auto& s : SA) { A.emplace_back(); for (char c : s) A.back().push_back(seqan3::assign_char_to(c, alph_t{})); }
    for (auto& s : SB) { B.emplace_back(); for (char c : s) B.back().push_back(seqan3::assign_char_to(c, alph_t{})); }
    Result r = a.global() ? run_method(a, seqan3::align_cfg::method_global{}, p, A, B)
                          : run_method(a, seqan3::align_cfg::method_local{}, p, A, B);
#ifdef VECTORISED
    r.info["vectorised"] = "true";
#else
    r.info["vectorised"] = "false";
#endif
    r.info["version"] = jstr("3.4.2");
    if (a.threads > 1) r.info["native_parallel"] = "true";
    return r;
}

DRIVER_MAIN(run)
