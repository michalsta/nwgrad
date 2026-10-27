// nwgrad through its C++ API: SeqPairBatchT<T>, the same engine the Python batch wraps,
// minus the binding layer.  Variants as arm.py's: traceback=, fill=, kernel=, schedule=,
// soft_impl=.
#include "drv_common.hpp"

#include "seq_pair_batch.hpp"

#include <cstdint>
#include <string_view>

static TracebackMode parse_tb(const std::string& s) {
    if (s == "pointers") return TracebackMode::Pointers;
    if (s == "scores") return TracebackMode::Scores;
    if (s == "hirschberg") return TracebackMode::Hirschberg;
    if (s == "hirschberg_pmax") return TracebackMode::HirschbergPmax;
    if (s == "grid") return TracebackMode::Grid;
    return TracebackMode::Default;
}

static int parse_kernel(const std::string& s) {
    if (s.empty() || s == "auto") return kBackendAuto;
    if (s == "scalar_fallback") return kBackendScalar;
    for (SimdLevel l : detect_available_levels())
        if (s == level_name(l)) return (int)l;
    throw Unsupported("kernel " + s + " not on this CPU");
}

template<class T>
static Result run_t(const Args& a, const std::vector<std::string>& A,
                    const std::vector<std::string>& B, const Problem& p) {
    auto var = parse_variant(a.variant);
    const Alphabet& alpha = Alphabet::get(p.alphabet);
    size_t n = p.alphabet.size();
    std::vector<double> m(n * n);
    for (size_t i = 0; i < n; ++i) for (size_t j = 0; j < n; ++j) m[i * n + j] = p.matrix[i][j];
    AlignParams P(SubstMatrix(m.data(), alpha), p.go, p.ge, p.go, p.ge);
    GradMode gd = a.out == "hard" ? GradMode::Hard : a.out == "soft" ? GradMode::Soft : GradMode::None;
    if (gd == GradMode::Soft && !std::is_same_v<T, double>)
        throw Unsupported("nwgrad's soft path is float64");
    std::vector<std::string_view> va(A.begin(), A.end()), vb(B.begin(), B.end());
    SeqPairBatchT<T> b(va, vb, P, a.affine() ? GapModel::Affine : GapModel::Linear,
                       a.global() ? AlignMode::Global : AlignMode::Local, gd, a.threads,
                       parse_tb(var["traceback"]), parse_kernel(var["kernel"]));
    if (var.count("fill")) {
        auto f = var["fill"];
        b.set_fill(f == "rowwise", f == "interpair");
    }
    if (var["schedule"] == "sorted") b.set_sorted_schedule(true);
    if (var["soft_impl"] == "log") b.set_soft_impl(SoftImpl::Log);
    bool keep = a.out == "path", grad = gd != GradMode::None;
    Result r = time_batch(a, [&] {
        b.set_params(P);
        b.score_and_grad(keep);
        if (grad) { volatile auto g = b.compute_grad().gap_open_a; (void)g; }
        return b.scores();
    });
    r.info["n_threads"] = std::to_string(b.n_threads());
    r.info["compiler"] = jstr(
#if defined(__clang__)
        std::string("clang ") + __clang_version__
#else
        std::string("gcc ") + __VERSION__
#endif
    );
    r.info["isa"] = jstr(level_name((SimdLevel)global_default_backend()));
    return r;
}

static Result run(const Args& a, const std::vector<std::string>& A,
                  const std::vector<std::string>& B, const Problem& p) {
    if (a.dtype == "float32") return run_t<float>(a, A, B, p);
    if (a.dtype == "float64") return run_t<double>(a, A, B, p);
    if (a.dtype == "int16") return run_t<int16_t>(a, A, B, p);
    if (a.dtype == "int32") return run_t<int32_t>(a, A, B, p);
    if (a.dtype == "int64") return run_t<int64_t>(a, A, B, p);
    throw Unsupported("dtype " + a.dtype);
}

DRIVER_MAIN(run)
