// WFA2-lib (smarco/WFA2-lib): wavefront alignment, global (end-to-end), EXACT -- the
// default heuristic (wf_heuristic_wfadaptive) is switched off.  Match/mismatch scoring
// only (no substitution matrix), so DNA only.  Variant: memory=high|med|low|ultralow
// (ultralow = BiWFA), metric=edit (unit-cost edit distance; the linear global DNA cell).
//
// WFA2 speaks penalties: match M <= 0 (a bonus is a negative penalty), mismatch X,
// gap-affine o + e*k -- nwgrad's go + ge*k with the signs flipped.  One aligner per
// thread (an aligner is not thread-safe).
#include "drv_common.hpp"

extern "C" {
void* wfa_shim_new(int metric, int match, int mismatch, int go, int ge, int scope, int memory);
int wfa_shim_align(void* w, const char* a, int la, const char* b, int lb);
void wfa_shim_delete(void* w);
}

static Result run(const Args& a, const std::vector<std::string>& A,
                  const std::vector<std::string>& B, const Problem& p) {
    if (!p.is_dna()) throw Unsupported("WFA2: match/mismatch scoring only (no substitution matrix)");
    if (!a.global()) throw Unsupported("WFA2 has no local (Smith-Waterman) mode");
    if (a.out != "score" && a.out != "path") throw Unsupported("WFA2 has no gradient");
    if (a.dtype != "int32") throw Unsupported("WFA2: int scores, filed under int32");
    auto var = parse_variant(a.variant);
    int match = p.matrix[0][0], mismatch = p.matrix[0][1];
    std::string metric = var.count("metric") ? var["metric"] : (a.affine() ? "gap_affine" : "gap_linear");
    int m;
    if (metric == "edit") {
        if (a.affine() || match != 0 || mismatch != -1 || p.ge != 1)
            throw Unsupported("metric=edit needs unit costs");
        m = 0;
    } else if (metric == "gap_linear") {
        if (a.affine()) throw Unsupported("gap_linear on an affine cell");
        m = 1;
    } else {
        if (!a.affine()) throw Unsupported("gap_affine on a linear cell");
        m = 2;
    }
    std::string mem = var.count("memory") ? var["memory"] : "high";
    int mm = mem == "high" ? 0 : mem == "med" ? 1 : mem == "low" ? 2 : 3;
    std::vector<void*> wf(std::max(1, a.threads));
    for (auto& w : wf) w = wfa_shim_new(m, -match, -mismatch, p.go, p.ge, a.out == "path", mm);
    // WFA2 reports a weighted penalty as a negative score, but the edit metric's result
    // as the (positive) distance
    double sign = m == 0 ? -1.0 : 1.0;
    Result r = time_pairs(a, A.size(), [&](size_t i, int tid) -> double {
        return sign * wfa_shim_align(wf[tid], A[i].data(), (int)A[i].size(), B[i].data(), (int)B[i].size());
    });
    for (auto& w : wf) wfa_shim_delete(w);
    r.info["metric"] = jstr(metric);
    r.info["memory"] = jstr(mem);
    r.info["heuristic"] = jstr("none");
    return r;
}

DRIVER_MAIN(run)
