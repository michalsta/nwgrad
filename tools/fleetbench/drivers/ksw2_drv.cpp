// KSW2 ksw_extz2_sse (lh3/ksw2, minimap2's aligner): global alignment, full width (w < 0,
// no z-drop), Suzuki-Kasahara difference encoding in 8-bit lanes with 32-bit scores --
// filed under int32.  Generic substitution matrix (KSW_EZ_GENERIC_SC).  No local mode
// (its "extension" is not Smith-Waterman).
//
// Gaps: KSW2 charges q + k*e for a gap of length k -- nwgrad's go + ge*k (linear: q=0).
#include "drv_common.hpp"

#include <cstring>

extern "C" {
#include "ksw2.h"
}

static Result run(const Args& a, const std::vector<std::string>& A,
                  const std::vector<std::string>& B, const Problem& p) {
    if (!a.global()) throw Unsupported("KSW2 has no local (Smith-Waterman) mode");
    if (a.out != "score" && a.out != "path") throw Unsupported("KSW2 has no gradient");
    if (a.dtype != "int32") throw Unsupported("KSW2: 32-bit scores, filed under int32");
    // KSW2's SSE kernel caps a cell at mat[0] + 2(q+e) (ksw2_extz2_sse.c: max_sc_), i.e.
    // it assumes mat[0] is the largest score -- true for match/mismatch DNA, false for
    // BLOSUM62 (A-A = 4, W-W = 11), where it silently clips.  Renumber the alphabet so the
    // symbol holding the largest score comes first: the same problem, within KSW2's contract.
    int n = (int)p.alphabet.size();
    int top = 0, best = -1000;
    for (int i = 0; i < n; ++i) if (p.matrix[i][i] > best) best = p.matrix[i][i], top = i;
    std::vector<int> perm(n);                     // new index -> old index
    perm[0] = top;
    for (int i = 0, k = 1; i < n; ++i) if (i != top) perm[k++] = i;
    std::vector<int8_t> mat(n * n);
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) mat[i * n + j] = (int8_t)p.matrix[perm[i]][perm[j]];
    uint8_t code[256] = {0};
    for (int i = 0; i < n; ++i) code[(unsigned char)p.alphabet[perm[i]]] = (uint8_t)i;
    std::vector<std::vector<uint8_t>> EA(A.size()), EB(B.size());
    for (size_t i = 0; i < A.size(); ++i) {
        for (char c : A[i]) EA[i].push_back(code[(unsigned char)c]);
        for (char c : B[i]) EB[i].push_back(code[(unsigned char)c]);
    }
    int8_t q = a.affine() ? p.go : 0, e = p.ge;
    int flag = KSW_EZ_GENERIC_SC | (a.out == "score" ? KSW_EZ_SCORE_ONLY : 0);
    Result r = time_pairs(a, A.size(), [&](size_t i, int) -> double {
        ksw_extz_t ez;
        std::memset(&ez, 0, sizeof(ez));
        ksw_extz2_sse(nullptr, (int)EA[i].size(), EA[i].data(), (int)EB[i].size(), EB[i].data(),
                      (int8_t)n, mat.data(), q, e, -1, -1, 0, flag, &ez);
        double s = ez.score;
        std::free(ez.cigar);
        return s;
    });
    r.info["encoding"] = jstr("8-bit differences, 32-bit scores");
#if defined(__SSE4_1__)
    r.info["isa"] = jstr("sse4.1");
#else
    r.info["isa"] = jstr("sse2");
#endif
    return r;
}

DRIVER_MAIN(run)
