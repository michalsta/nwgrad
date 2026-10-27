// parasail through its C API, linked against the libparasail.so its Python wheel ships
// (the binary package; no Python per call).  Variant: impl=striped|scan|diag,
// width=16|32|64|sat.  sat (8 -> 16 -> 32 bits on overflow) is filed under the narrowest
// dtype it completed in: under int16 when no pair needed 32 bits, under int32 otherwise.
//
// Gaps: parasail charges open + (k-1)*extend for a gap of length k, so nwgrad's affine
// go + ge*k is open = go + ge, extend = ge, and linear ge*k is open = extend = ge.
#include "drv_common.hpp"

#include <parasail.h>

#include <algorithm>

static Result run(const Args& a, const std::vector<std::string>& A,
                  const std::vector<std::string>& B, const Problem& p) {
    auto var = parse_variant(a.variant);
    std::string impl = var.count("impl") ? var["impl"] : "striped";
    std::string width = var.count("width") ? var["width"] : "sat";
    if (a.out != "score" && a.out != "path") throw Unsupported("parasail has no gradient");
    if (a.dtype.rfind("int", 0) != 0) throw Unsupported("parasail is integer only");
    if (width != "sat" && "int" + width != a.dtype) throw Unsupported("width/dtype mismatch");
    if (width == "sat" && a.dtype != "int16" && a.dtype != "int32")
        throw Unsupported("sat is filed under int16 or int32");
    if (width == "64" && impl == "diag") { /* exists */ }
    std::string name = std::string(a.global() ? "nw" : "sw") +
                       (a.out == "path" ? "_trace_" : "_") + impl + "_" + width;
    parasail_function_t* f = parasail_lookup_function(name.c_str());
    if (!f) f = parasail_lookup_function(("parasail_" + name).c_str());
    if (!f) throw Unsupported("no parasail function " + name);
    const parasail_matrix_t* M;
    if (p.is_blosum62()) {
        M = parasail_matrix_lookup("blosum62");
    } else {
        M = parasail_matrix_create(p.alphabet.c_str(), p.matrix[0][0], p.matrix[0][1]);
    }
    int open = a.affine() ? p.go + p.ge : p.ge, ext = p.ge;
    bool trace = a.out == "path";
    std::atomic<int> maxbits{0}, saturated{0};
    Result r = time_pairs(a, A.size(), [&](size_t i, int) -> double {
        const std::string &x = A[i], &y = B[i];
        parasail_result_t* res = f(x.data(), (int)x.size(), y.data(), (int)y.size(), open, ext, M);
        double s = res->score;
        int bits = (res->flag & PARASAIL_FLAG_BITS_8) ? 8 : (res->flag & PARASAIL_FLAG_BITS_16) ? 16
                 : (res->flag & PARASAIL_FLAG_BITS_32) ? 32 : 64;
        int cur = maxbits.load();
        while (bits > cur && !maxbits.compare_exchange_weak(cur, bits)) {}
        if (parasail_result_is_saturated(res)) saturated++;
        if (trace) {
            parasail_cigar_t* c = parasail_result_get_cigar(res, x.data(), (int)x.size(), y.data(),
                                                            (int)y.size(), M);
            parasail_cigar_free(c);
        }
        parasail_result_free(res);
        return s;
    });
    // parasail's saturation flag, kept as information: the runner's score check decides.
    // (Its diag kernels raise it with linear gaps even at 32/64 bits, where the scores
    // come out right.)
    r.info["saturated_flags"] = std::to_string(saturated.load());
    if (width == "sat" && r.status == "ok") {
        if (a.dtype == "int16" && maxbits > 16)
            throw Unsupported("sat needed 32 bits: filed under int32");
        if (a.dtype == "int32" && maxbits <= 16)
            throw Unsupported("sat completed in <=16 bits: filed under int16");
    }
    r.info["function"] = jstr(name);
    r.info["max_bits"] = std::to_string(maxbits.load());
    r.info["sat"] = width == "sat" ? "true" : "false";
    r.info["version"] = jstr(std::to_string(PARASAIL_VERSION_MAJOR) + "." + std::to_string(PARASAIL_VERSION_MINOR) + "." + std::to_string(PARASAIL_VERSION_PATCH));
    return r;
}

DRIVER_MAIN(run)
