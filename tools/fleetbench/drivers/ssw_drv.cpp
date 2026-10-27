// SSW (Complete-Striped-Smith-Waterman-Library): striped local alignment, 8-bit lanes
// rerun at 16 bits on overflow (score_size=2), filed under int16 as "sat".  Local, affine
// (linear as open = extend).  The query profile is built per pair, inside the timing, as
// nwgrad builds its own.
//
// Gaps: SSW charges gapO for a gap's first position and gapE for each further one, so
// nwgrad's go + ge*k is gapO = go + ge, gapE = ge.
#include "drv_common.hpp"

extern "C" {
#include "ssw.h"
}

static Result run(const Args& a, const std::vector<std::string>& A,
                  const std::vector<std::string>& B, const Problem& p) {
    if (a.global()) throw Unsupported("SSW is local only");
    if (a.out != "score" && a.out != "path") throw Unsupported("SSW has no gradient");
    if (a.dtype != "int16") throw Unsupported("SSW: 8/16-bit saturating, filed under int16");
    int n = (int)p.alphabet.size();
    std::vector<int8_t> mat(n * n);
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) mat[i * n + j] = (int8_t)p.matrix[i][j];
    int8_t code[256];
    for (int c = 0; c < 256; ++c) code[c] = 0;
    for (int i = 0; i < n; ++i) code[(unsigned char)p.alphabet[i]] = (int8_t)i;
    std::vector<std::vector<int8_t>> EA(A.size()), EB(B.size());
    for (size_t i = 0; i < A.size(); ++i) {
        for (char c : A[i]) EA[i].push_back(code[(unsigned char)c]);
        for (char c : B[i]) EB[i].push_back(code[(unsigned char)c]);
    }
    uint8_t gO = a.affine() ? p.go + p.ge : p.ge, gE = p.ge;
    uint8_t flag = a.out == "path" ? 1 : 0;
    std::atomic<int> overflow{0};
    Result r = time_pairs(a, A.size(), [&](size_t i, int) -> double {
        s_profile* prof = ssw_init(EA[i].data(), (int32_t)EA[i].size(), mat.data(), n, 2);
        s_align* al = ssw_align(prof, EB[i].data(), (int32_t)EB[i].size(), gO, gE, flag, 0, 0, 15);
        double s = al->score1;
        if (al->score1 == 65535) overflow++;
        align_destroy(al);
        init_destroy(prof);
        return s;
    });
    if (overflow) { r.status = "error"; r.info["why"] = jstr("16-bit overflow"); }
    r.info["sat"] = "true";
    return r;
}

DRIVER_MAIN(run)
