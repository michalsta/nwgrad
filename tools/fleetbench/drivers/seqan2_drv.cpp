// SeqAn 2.5 (seqan2 namespace).  One binary per (alphabet, score type):
//   -DALPH_DNA (else AminoAcid + BLOSUM62)  -DSCORE_T=short|int|long long|float|double
//   -DDTYPE_NAME=int16|int32|int64|float32|float64
// Variant exec=scalar (one pair at a time, globalAlignmentScore(seq, seq, score)) or
// exec=vectorial (ExecutionPolicy<Serial, Vectorial>: pairs in SIMD lanes, the score
// type deciding the lane width); with --threads > 1, ExecutionPolicy<Parallel, ...>
// with that many threads -- SeqAn2's own threading.
//
// Gaps: SeqAn2's Score(..., gapExtend, gapOpen) charges gapOpen for a gap's first
// position and gapExtend for each further one: nwgrad's go + ge*k is gapOpen = -(go+ge),
// gapExtend = -ge; linear is gapOpen == gapExtend (SeqAn2 then runs the linear DP).
#include "drv_common.hpp"

#include <seqan/align.h>
#include <seqan/align_parallel.h>

#define STR2(x) #x
#define STR(x) STR2(x)

using namespace seqan2;

#ifdef ALPH_DNA
using TAlph = Dna;
using TScore = Score<SCORE_T, Simple>;
#else
using TAlph = AminoAcid;
using TScore = Score<SCORE_T, ScoreMatrix<AminoAcid, Blosum62_>>;
#endif
using TSeq = String<TAlph>;

static TScore make_score(const Args& a, const Problem& p) {
    SCORE_T open = a.affine() ? -(p.go + p.ge) : -p.ge, ext = -p.ge;
#ifdef ALPH_DNA
    return TScore(p.matrix[0][0], p.matrix[0][1], ext, open);
#else
    return TScore(ext, open);
#endif
}

#ifdef SEQAN2_VEC
template<class Exec>
static Result run_vec(const Args& a, Exec exec, StringSet<TSeq>& A, StringSet<TSeq>& B,
                      const TScore& sc) {
    size_t n = length(A);
    if (a.out == "score") {
        return time_batch(a, [&] {
            String<SCORE_T> s = a.global() ? globalAlignmentScore(exec, A, B, sc)
                                           : localAlignmentScore(exec, A, B, sc);
            std::vector<double> v(n);
            for (size_t i = 0; i < n; ++i) v[i] = s[i];
            return v;
        });
    }
    return time_batch(a, [&] {
        StringSet<Gaps<TSeq>> GA, GB;
        resize(GA, n); resize(GB, n);
        for (size_t i = 0; i < n; ++i) { setSource(GA[i], A[i]); setSource(GB[i], B[i]); }
        String<SCORE_T> s = a.global() ? globalAlignment(exec, GA, GB, sc)
                                       : localAlignment(exec, GA, GB, sc);
        std::vector<double> v(n);
        for (size_t i = 0; i < n; ++i) v[i] = s[i];
        return v;
    });
}
#endif

static Result run(const Args& a, const std::vector<std::string>& SA,
                  const std::vector<std::string>& SB, const Problem& p) {
    if (a.out != "score" && a.out != "path") throw Unsupported("SeqAn2 has no gradient");
    if (a.dtype != STR(DTYPE_NAME)) throw Unsupported("this binary is " STR(DTYPE_NAME));
#ifdef ALPH_DNA
    if (!p.is_dna()) throw Unsupported("DNA binary");
#else
    if (!p.is_blosum62()) throw Unsupported("protein binary");
#endif
    auto var = parse_variant(a.variant);
    std::string exec = var.count("exec") ? var["exec"] : "vectorial";
    StringSet<TSeq> A, B;
    for (auto& s : SA) appendValue(A, TSeq(s));
    for (auto& s : SB) appendValue(B, TSeq(s));
    TScore sc = make_score(a, p);
    Result r;
    if (exec == "scalar") {
        r = time_pairs(a, SA.size(), [&](size_t i, int) -> double {
            if (a.out == "score")
                return a.global() ? globalAlignmentScore(A[i], B[i], sc)
                                  : localAlignmentScore(A[i], B[i], sc);
            Align<TSeq> al;
            resize(rows(al), 2);
            assignSource(row(al, 0), A[i]);
            assignSource(row(al, 1), B[i]);
            return a.global() ? globalAlignment(al, sc) : localAlignment(al, sc);
        });
    } else {
#ifndef SEQAN2_VEC
        throw Unsupported("SeqAn2 vectorises 8/16/32-bit integer scores only (x86)");
#else
      if (a.threads > 1) {
        ExecutionPolicy<Parallel, Vectorial> ep;
        setNumThreads(ep, a.threads);
        r = run_vec(a, ep, A, B, sc);
        r.info["native_parallel"] = "true";
      } else {
        r = run_vec(a, ExecutionPolicy<Serial, Vectorial>(), A, B, sc);
      }
#endif
    }
    r.info["exec"] = jstr(exec);
    r.info["version"] = jstr("2.5.3");
    return r;
}

DRIVER_MAIN(run)
