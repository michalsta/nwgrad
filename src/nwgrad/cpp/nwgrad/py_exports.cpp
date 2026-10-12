// nanobind (and so Python.h) first: pyconfig.h defines _POSIX_C_SOURCE/_XOPEN_SOURCE,
// which must be in place before any standard header is included.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "align_params.hpp"
#include "aligner.hpp"
#include "seq_pair_batch.hpp"

namespace nb = nanobind;

// Temporary 0.7 deprecations. Warn before mutation so warnings-as-errors is safe.
static void warn_pair_api(const char* old_name, const char* replacement) {
    const std::string message = std::string("SeqPair.") + old_name +
        " is deprecated in nwgrad 0.7.0 and will be removed in 0.8.0; use " + replacement;
    if (PyErr_WarnEx(PyExc_DeprecationWarning, message.c_str(), 1) < 0)
        throw nb::python_error();
}

void bind_logistic(nb::module_& parent);   // logistic/py_logistic.cpp
using nb_arr_f64    = nb::ndarray<double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using nb_arr_f64_3d = nb::ndarray<double, nb::ndim<3>, nb::c_contig, nb::device::cpu>;
using nb_arr_f64_1d = nb::ndarray<nb::numpy, double, nb::ndim<1>>;

// A new numpy float64 array of `shape` owning a copy of `n` doubles from `src`.
// Returned as an object, so a property getter's reference_internal policy never applies.
static nb::object numpy_copy(const double* src, size_t n, std::initializer_list<size_t> shape) {
    double* buf = new double[n ? n : 1];
    std::copy(src, src + n, buf);
    nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<double*>(p); });
    std::vector<size_t> sh(shape);
    return nb::cast(nb::ndarray<nb::numpy, double>(buf, sh.size(), sh.data(), owner));
}

// The compiler that built this extension, captured at compile time.  Exposed as
// nwgrad.compiled_with() — a reliable way to tell which toolchain produced the loaded
// binary (the .comment ELF section is stripped by some clang builds, and matters here
// because gcc and clang differ measurably on the std::simd kernels).
#if defined(__clang__)
#  if defined(__apple_build_version__)
#    define NWGRAD_COMPILER "Apple clang " __clang_version__
#  else
#    define NWGRAD_COMPILER "clang " __clang_version__
#  endif
#elif defined(__GNUC__)
#  define NWGRAD_COMPILER "gcc " __VERSION__
#elif defined(_MSC_VER)
#  define NWGRAD_STR2(x) #x
#  define NWGRAD_STR(x) NWGRAD_STR2(x)
#  define NWGRAD_COMPILER "msvc " NWGRAD_STR(_MSC_VER)
#else
#  define NWGRAD_COMPILER "unknown"
#endif

// ── Aligner factory: picks Full or GuideBanded at runtime. ───────────────────

// The `kernel=` string names the Viterbi backend: "auto" (default; best simd the CPU
// runs), "scalar_fallback", or a named simd level ("sse2"/"avx2"/"avx512"/"neon").  Every
// simd level is bit-exact with the scalar one, so it is a speed knob and never a
// correctness one.  simd_levels.hpp::parse_backend() turns the string into a backend int
// and throws on anything unrecognised or a level this CPU cannot run — a typo that
// silently gave you the wrong path would be undetectable.

// TYPE is the Viterbi precision (float32 by default in the Python surface, double via
// the _double-suffixed variant).  The DP buffer and Aligner take that precision; params
// and the returned gradient stay double.
#define WITH_ALIGNER_T(TYPE, GM, AM, band, guide_j, body)                                \
    do {                                                                                  \
        DpBufferT<TYPE> _buf;                                                             \
        const int _kern = parse_backend(kernel);                                          \
        if ((band) > 0 || !(guide_j).empty()) {                                          \
            Aligner<GapModel::GM, AlignMode::AM, AlignBand::GuideBanded, TYPE> al;       \
            al.set_kernel(_kern);                                                          \
            body                                                                           \
        } else {                                                                           \
            Aligner<GapModel::GM, AlignMode::AM, AlignBand::Full, TYPE> al;              \
            al.set_kernel(_kern);                                                          \
            body                                                                           \
        }                                                                                  \
    } while (0)

// Back-compat alias: the double-precision convenience functions still spell it this way.
#define WITH_ALIGNER(GM, AM, band, guide_j, body) \
    WITH_ALIGNER_T(double, GM, AM, band, guide_j, body)

// The convenience functions still take plain `str`: they validate and encode
// against the params' alphabet here, at the boundary.  An out-of-alphabet
// character throws.  The encoded vectors must outlive the set_problem() call,
// so they live in the caller's frame.
struct EncodedPair {
    std::vector<uint8_t> a, b;
    EncodedPair(std::string_view sa, std::string_view sb, const AlignParams& p)
        : a(p.matrix.alphabet().encode(sa)), b(p.matrix.alphabet().encode(sb)) {}
};

// A track argument (None, or a sequence of set indices) as uint16 set indices.  Range
// against the params' set count is checked by the Aligner / the batch, which know K.
static std::vector<uint16_t> to_track(const nb::handle& obj, const char* what) {
    std::vector<uint16_t> out;
    if (!obj.is_valid() || obj.is_none()) return out;
    std::vector<int64_t> v;
    try {
        v = nb::cast<std::vector<int64_t>>(obj);
    } catch (const nb::cast_error&) {
        throw nb::type_error((std::string("nwgrad: ") + what +
                              " must be a sequence of integers (parameter set indices) or None").c_str());
    }
    out.resize(v.size());
    for (size_t k = 0; k < v.size(); ++k) {
        if (v[k] < 0 || v[k] >= SubstMatrix::kMaxSets)
            throw nb::value_error((std::string("nwgrad: ") + what + "[" + std::to_string(k) + "] = " +
                                   std::to_string(v[k]) + " is not a parameter set index").c_str());
        out[k] = static_cast<uint16_t>(v[k]);
    }
    return out;
}

// Element q of a 1-D integer array of any width and signedness, widened.
static int64_t flat_at(const nb::ndarray<nb::ro, nb::ndim<1>, nb::device::cpu>& arr, size_t q) {
    const auto* base = static_cast<const char*>(arr.data()) + static_cast<int64_t>(q) * arr.stride(0) *
                       static_cast<int64_t>(arr.dtype().bits / 8);
    const bool sgn = arr.dtype().code == static_cast<uint8_t>(nb::dlpack::dtype_code::Int);
    switch (arr.dtype().bits) {
        case 8:  return sgn ? *reinterpret_cast<const int8_t*>(base)  : *reinterpret_cast<const uint8_t*>(base);
        case 16: return sgn ? *reinterpret_cast<const int16_t*>(base) : *reinterpret_cast<const uint16_t*>(base);
        case 32: return sgn ? *reinterpret_cast<const int32_t*>(base) : *reinterpret_cast<const uint32_t*>(base);
        default: {
            if (sgn) return *reinterpret_cast<const int64_t*>(base);
            const uint64_t u = *reinterpret_cast<const uint64_t*>(base);
            return u > static_cast<uint64_t>(INT64_MAX) ? -1 : static_cast<int64_t>(u);
        }
    }
}

// A tracks argument, converted: None; one sequence per pair (an entry may be None for
// that pair's default); or ONE flat 1-D integer numpy array, every pair's entries
// concatenated in pair order, pair i taking lens[i] of them (a flat array gives every
// pair a track).  A C-contiguous uint16 array is read in place (`keep` holds it);
// anything else is converted once into `own`.  view() is what the engine takes.
struct TracksArg {
    std::vector<std::vector<uint16_t>> nested;
    std::vector<uint16_t> own;
    std::vector<size_t> off;
    nb::ndarray<nb::ro, nb::ndim<1>, nb::device::cpu> keep;
    const uint16_t* flat = nullptr;
    bool given = false, is_flat = false;
    PairTracks view() const {
        if (!given) return {};
        if (is_flat) return PairTracks(flat, off.data(), off.size() - 1);
        return PairTracks(&nested);
    }
};

static TracksArg to_tracks(const nb::handle& obj, const std::vector<size_t>& lens, const char* what) {
    TracksArg out;
    if (!obj.is_valid() || obj.is_none()) return out;
    out.given = true;
    const size_t n = lens.size();
    if (nb::try_cast(obj, out.keep, false)) {
        const auto& arr = out.keep;
        const auto code = arr.dtype().code;
        if ((code != static_cast<uint8_t>(nb::dlpack::dtype_code::Int) &&
             code != static_cast<uint8_t>(nb::dlpack::dtype_code::UInt)) || arr.dtype().lanes != 1)
            throw nb::type_error((std::string("nwgrad: ") + what +
                                  " as one flat array must have an integer dtype").c_str());
        out.off.resize(n + 1);
        out.off[0] = 0;
        for (size_t i = 0; i < n; ++i) out.off[i + 1] = out.off[i] + lens[i];
        const size_t total = out.off[n];
        if (arr.shape(0) != total)
            throw nb::value_error((std::string("nwgrad: flat ") + what + " has " + std::to_string(arr.shape(0)) +
                                   " entries, expected " + std::to_string(total) +
                                   " (every pair's entries, concatenated)").c_str());
        out.is_flat = true;
        if (code == static_cast<uint8_t>(nb::dlpack::dtype_code::UInt) && arr.dtype().bits == 16 &&
            (arr.stride(0) == 1 || total <= 1)) {
            out.flat = static_cast<const uint16_t*>(arr.data());   // in place
            return out;
        }
        out.own.resize(total);
        for (size_t q = 0; q < total; ++q) {
            const int64_t v = flat_at(arr, q);
            if (v < 0 || v >= SubstMatrix::kMaxSets)
                throw nb::value_error((std::string("nwgrad: ") + what + "[" + std::to_string(q) + "] = " +
                                       std::to_string(v) + " is not a parameter set index").c_str());
            out.own[q] = static_cast<uint16_t>(v);
        }
        out.flat = out.own.data();
        return out;
    }
    const size_t len = nb::len(obj);
    if (len != n)
        throw nb::value_error((std::string("nwgrad: ") + what + " needs one entry per pair (got " +
                               std::to_string(len) + " for " + std::to_string(n) + " pairs)").c_str());
    out.nested.reserve(n);
    for (nb::handle h : obj) out.nested.push_back(to_track(h, what));
    return out;
}

// Per pair, the lengths of its track (len(A)) and gap track (len(A) + 1).
static std::pair<std::vector<size_t>, std::vector<size_t>> track_lengths(const std::vector<std::string_view>& seqs_a) {
    std::vector<size_t> la(seqs_a.size()), lg(seqs_a.size());
    for (size_t i = 0; i < seqs_a.size(); ++i) { la[i] = seqs_a[i].size(); lg[i] = la[i] + 1; }
    return {std::move(la), std::move(lg)};
}

static GapModel parse_gap_model(const std::string& name) {
    if (name == "linear") return GapModel::Linear;
    if (name == "affine") return GapModel::Affine;
    throw nb::value_error(
        ("nwgrad: unknown gap_model \"" + name +
         "\" (expected \"linear\" or \"affine\")").c_str());
}

static AlignMode parse_align_mode(const std::string& name) {
    if (name == "global") return AlignMode::Global;
    if (name == "local")  return AlignMode::Local;
    throw nb::value_error(
        ("nwgrad: unknown mode \"" + name +
         "\" (expected \"global\" or \"local\")").c_str());
}

template<class Mode>
static Mode parse_grad_mode(const std::string& name) {
    if (name == "hard") return Mode::Hard;
    if (name == "soft") return Mode::Soft;
    if (name == "none") return Mode::None;
    throw nb::value_error(
        ("nwgrad: unknown grad_mode \"" + name +
         "\" (expected \"hard\", \"soft\" or \"none\")").c_str());
}

// Traceback vocabulary, mirroring gap_model= / mode= / grad_mode= / kernel=.
static TracebackMode parse_traceback(const std::string& name) {
    if (name == "auto")       return TracebackMode::Default;
    if (name == "pointers")   return TracebackMode::Pointers;
    if (name == "scores")     return TracebackMode::Scores;
    if (name == "hirschberg") return TracebackMode::Hirschberg;
    if (name == "hirschberg_pmax") return TracebackMode::HirschbergPmax;
    throw nb::value_error(
        ("nwgrad: unknown traceback \"" + name +
         "\" (expected \"auto\", \"pointers\", \"scores\", \"hirschberg\" or "
         "\"hirschberg_pmax\")").c_str());
}
static bool parse_fill(const std::string& name) {
    if (name == "striped") return false;
    if (name == "rowwise") return true;
    throw nb::value_error(
        ("nwgrad: unknown fill \"" + name + "\" (expected \"striped\" or \"rowwise\")").c_str());
}
// The batch also has "interpair".  Returns {rowwise_full, inter_fill}.
static std::pair<bool, bool> parse_batch_fill(const std::string& name) {
    if (name == "interpair") return {false, true};
    if (name == "striped" || name == "rowwise") return {parse_fill(name), false};
    throw nb::value_error(
        ("nwgrad: unknown fill \"" + name +
         "\" (expected \"striped\", \"rowwise\" or \"interpair\")").c_str());
}
static SoftImpl parse_soft_impl(const std::string& name) {
    if (name == "scaled")        return SoftImpl::Scaled;
    if (name == "scaled_or_log") return SoftImpl::ScaledOrLog;
    if (name == "log")           return SoftImpl::Log;
    throw nb::value_error(
        ("nwgrad: unknown soft_impl \"" + name +
         "\" (expected \"scaled\", \"scaled_or_log\" or \"log\")").c_str());
}
static const char* soft_impl_name(SoftImpl s) {
    switch (s) {
        case SoftImpl::Scaled:      return "scaled";
        case SoftImpl::ScaledOrLog: return "scaled_or_log";
        default:                    return "log";
    }
}
#define NWGRAD_SOFT_IMPL_DOC \
    "How the soft path (grad_mode=\"soft\") evaluates forward-backward.\n" \
    "  \"scaled\" (default): probability space with exact power-of-two row\n" \
    "     rescaling, gradient fused into the backward pass; no exp/log per cell.\n" \
    "     Raises ValueError for a pair whose dynamic range does not fit a double\n" \
    "     (typically local log Z beyond ~700 nats, i.e. long high-scoring protein\n" \
    "     pairs, or a step score beyond +-34.6).\n" \
    "  \"scaled_or_log\": \"scaled\", silently falling back per pair to \"log\".\n" \
    "  \"log\": the log-space recurrences; unlimited range, far slower.\n" \
    "Results agree to ~1e-12 relative, not bit-for-bit; the hard path is unaffected."
#define NWGRAD_SOFT_TEMP_DOC \
    "Soft temperature T > 0 (default 1).  The soft score becomes T*log Z(params/T)\n" \
    "and the gradient its derivative w.r.t. params: the expected counts under\n" \
    "params/T.  T -> 0 approaches the Viterbi score and hard counts; lower T\n" \
    "widens the dynamic range, so the default soft_impl=\"scaled\" eventually\n" \
    "raises (use \"scaled_or_log\" when annealing towards 0)."
static const char* traceback_name(TracebackMode t) {
    switch (t) {
        case TracebackMode::Pointers:   return "pointers";
        case TracebackMode::Hirschberg: return "hirschberg";
        case TracebackMode::HirschbergPmax: return "hirschberg_pmax";
        case TracebackMode::Default:    return "auto";
        default:                        return "scores";
    }
}

static std::vector<int> make_guide(const std::string& aligned_a,
                                    const std::string& aligned_b) {
    if (aligned_a.empty() != aligned_b.empty())
        throw nb::value_error("nwgrad: a guide needs both aligned_a and aligned_b (got only one)");
    if (!aligned_a.empty())
        return guide_j_from_aligned(aligned_a, aligned_b);
    return {};
}

// Pretty-print two gapped, equal-length aligned strings as a 3-line block
// (sequence A / match line / sequence B), wrapped at `width` columns.
// The match line uses '|' for identity, '.' for a mismatch, ' ' for a gap.
static std::string format_alignment(const std::string& a, const std::string& b,
                                    int width) {
    std::string match;
    match.reserve(a.size());
    for (size_t k = 0; k < a.size(); ++k) {
        if      (a[k] == '-' || b[k] == '-') match.push_back(' ');
        else if (a[k] == b[k])               match.push_back('|');
        else                                  match.push_back('.');
    }
    if (width <= 0) width = static_cast<int>(a.size());
    std::string out;
    for (size_t off = 0; off < a.size(); off += static_cast<size_t>(width)) {
        size_t len = std::min(static_cast<size_t>(width), a.size() - off);
        out += a.substr(off, len);     out += '\n';
        out += match.substr(off, len); out += '\n';
        out += b.substr(off, len);
        if (off + len < a.size()) out += "\n\n";
    }
    return out;
}

// ── Templated registrars ─────────────────────────────────────────────────────
// The Python surface's default precision is float32 (the plain names); double is bound
// under _double-suffixed function names and *Double class names.  Each registrar is
// called once per precision below.  Note the soft-gradient path is double regardless of
// T (log-sum-exp stays double inside DpBufferT<T>), so the _double soft functions are
// numerically identical to the plain ones — bound for naming symmetry only.
template<class T>
static void bind_convenience(nb::module_& m, const std::string& sfx) {
#define NWG_BASE_ARGS \
        nb::arg("seq_a"), nb::arg("seq_b"), nb::arg("params"), nb::arg("band") = 0, \
        nb::arg("aligned_a") = "", nb::arg("aligned_b") = "", nb::arg("kernel") = "auto"
#define NWG_TRACK_ARGS \
        nb::kw_only(), nb::arg("track").none() = nb::none(), nb::arg("gap_track").none() = nb::none()
#define NWG_CONV_ARGS NWG_BASE_ARGS, NWG_TRACK_ARGS
#define NWG_TRACK_DOC \
    "\n\ntrack / gap_track (keyword, optional): the parameter set of each residue of seq_a\n" \
    "(len(seq_a) entries) and of each DP row 0..len(seq_a) for gaps in seq_a\n" \
    "(len(seq_a)+1 entries; default: row 0 -> track[0], row i -> track[i-1]).  See\n" \
    "SeqPairBatch.add_many.  With several sets the gradient has as many."
#define NWG_SCORE(FN, GM, AM, DOC) \
    m.def((std::string(FN) + sfx).c_str(), \
        [](const std::string& a, const std::string& b, const AlignParams& params, int band, \
           const std::string& aligned_a, const std::string& aligned_b, const std::string& kernel, \
           nb::handle track, nb::handle gap_track) { \
            auto gj = make_guide(aligned_a, aligned_b); EncodedPair enc(a, b, params); \
            const auto tk = to_track(track, "track"), gt = to_track(gap_track, "gap_track"); \
            WITH_ALIGNER_T(T, GM, AM, band, gj, { \
                al.set_problem(enc.a, enc.b, params, band, gj, tk, gt); \
                return al.compute_score(_buf); }); \
        }, NWG_CONV_ARGS, DOC NWG_TRACK_DOC)
#define NWG_HARD(FN, GM, AM, DOC) \
    m.def((std::string(FN) + sfx).c_str(), \
        [](const std::string& a, const std::string& b, const AlignParams& params, int band, \
           const std::string& aligned_a, const std::string& aligned_b, const std::string& kernel, \
           nb::handle track, nb::handle gap_track) { \
            auto gj = make_guide(aligned_a, aligned_b); EncodedPair enc(a, b, params); \
            const auto tk = to_track(track, "track"), gt = to_track(gap_track, "gap_track"); \
            WITH_ALIGNER_T(T, GM, AM, band, gj, { \
                al.set_problem(enc.a, enc.b, params, band, gj, tk, gt); al.compute_viterbi(_buf); \
                AlignParams grad = AlignParams::zeros_like(params); al.hard_grad(_buf, grad); \
                return nb::make_tuple(al.score(), grad); }); \
        }, NWG_CONV_ARGS, DOC NWG_TRACK_DOC)
#define NWG_SOFT(FN, GM, AM, DOC) \
    m.def((std::string(FN) + sfx).c_str(), \
        [](const std::string& a, const std::string& b, const AlignParams& params, int band, \
           const std::string& aligned_a, const std::string& aligned_b, const std::string& kernel, \
           const std::string& soft_impl, double temperature, \
           nb::handle track, nb::handle gap_track) { \
            auto gj = make_guide(aligned_a, aligned_b); EncodedPair enc(a, b, params); \
            const auto tk = to_track(track, "track"), gt = to_track(gap_track, "gap_track"); \
            const SoftImpl si = parse_soft_impl(soft_impl); \
            WITH_ALIGNER_T(T, GM, AM, band, gj, { \
                al.set_soft_impl(si); al.set_soft_temperature(temperature); \
                al.set_problem(enc.a, enc.b, params, band, gj, tk, gt); al.compute_forward_back(_buf); \
                AlignParams grad = AlignParams::zeros_like(params); al.soft_grad(_buf, grad); \
                return nb::make_tuple(al.log_z(), grad); }); \
        }, NWG_BASE_ARGS, nb::arg("soft_impl") = "scaled", nb::arg("temperature") = 1.0, NWG_TRACK_ARGS, \
        DOC "\n\nsoft_impl: \"scaled\" (default) | \"scaled_or_log\" | \"log\" -- " \
        "see SeqPairBatch.soft_impl.\ntemperature: T > 0; returns (T*log Z(params/T), " \
        "expected counts under params/T) -- see SeqPairBatch.soft_temperature." NWG_TRACK_DOC)

    NWG_SCORE("nw_score", Linear, Global, "Needleman-Wunsch global alignment score (linear gap penalty).");
    NWG_SCORE("sw_score", Linear, Local,  "Smith-Waterman local alignment score (linear gap penalty).");
    NWG_SCORE("nw_score_affine", Affine, Global, "Needleman-Wunsch global alignment score (affine gap penalty).");
    NWG_SCORE("sw_score_affine", Affine, Local,  "Smith-Waterman local alignment score (affine gap penalty).");
    NWG_HARD("nw_grad", Linear, Global, "NW global: returns (score, AlignParams grad) — hard subgradient (linear gap).");
    NWG_HARD("sw_grad", Linear, Local,  "SW local: returns (score, AlignParams grad) — hard subgradient (linear gap).");
    NWG_HARD("nw_affine_grad", Affine, Global, "NW global: returns (score, AlignParams grad) — hard subgradient (affine gap).");
    NWG_HARD("sw_affine_grad", Affine, Local,  "SW local: returns (score, AlignParams grad) — hard subgradient (affine gap).");
    NWG_SOFT("nw_soft_grad", Linear, Global, "NW global: returns (log_Z, AlignParams grad) — soft gradient (linear gap).");
    NWG_SOFT("sw_soft_grad", Linear, Local,  "SW local: returns (log_Z, AlignParams grad) — soft gradient (linear gap).");
    NWG_SOFT("nw_affine_soft_grad", Affine, Global, "NW global: returns (log_Z, AlignParams grad) — soft gradient (affine gap).");
    NWG_SOFT("sw_affine_soft_grad", Affine, Local,  "SW local: returns (log_Z, AlignParams grad) — soft gradient (affine gap).");
#undef NWG_SCORE
#undef NWG_HARD
#undef NWG_SOFT
#undef NWG_CONV_ARGS
#undef NWG_BASE_ARGS
#undef NWG_TRACK_ARGS
#undef NWG_TRACK_DOC
}

static nb::object numpy_1d(const std::vector<double>& v) {
    double* buf = new double[v.size() ? v.size() : 1];
    std::copy(v.begin(), v.end(), buf);
    nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<double*>(p); });
    size_t shape[1] = {v.size()};
    return nb::cast(nb::ndarray<nb::numpy, double>(buf, 1, shape, owner));
}

// ── SeqPair: a standalone pair, or a view of pair i of a batch ───────────────
template<class T>
static void bind_seq_pair(nb::module_& m, const char* name) {
    using SP = SeqPairT<T>;
    nb::class_<SP>(m, name, nb::dynamic_attr())
        .def(
            "__init__",
            [](SP* self, const std::string& seq_a, const std::string& seq_b,
               const AlignParams& params, const std::string& gap_model,
               const std::string& mode, const std::string& grad_mode,
               const std::string& kernel, const std::string& traceback,
               nb::handle track, nb::handle gap_track) {
                new (self) SP(seq_a, seq_b, params, parse_gap_model(gap_model),
                              parse_align_mode(mode), parse_grad_mode<GradMode>(grad_mode),
                              parse_backend(kernel), parse_traceback(traceback),
                              to_track(track, "track"), to_track(gap_track, "gap_track"));
            },
            nb::arg("seq_a"), nb::arg("seq_b"), nb::arg("params"),
            nb::arg("gap_model") = "affine", nb::arg("mode") = "global",
            nb::arg("grad_mode") = "hard", nb::arg("kernel") = "auto",
            nb::arg("traceback") = "auto", nb::kw_only(),
            nb::arg("track").none() = nb::none(), nb::arg("gap_track").none() = nb::none(),
            "One sequence pair, aligned on the calling thread.\n"
            "  gap_model : \"linear\" | \"affine\"\n"
            "  mode      : \"global\" | \"local\"\n"
            "  grad_mode : \"hard\" | \"soft\" | \"none\"\n"
            "  kernel    : \"auto\" (default) | \"scalar_fallback\" | \"sse2\" | \"avx2\" |\n"
            "              \"avx512\" | \"neon\" — bit-exact speed knob (Viterbi path).\n"
            "  traceback : \"auto\" (default) | \"pointers\" | \"scores\" | \"hirschberg\"\n"
            "              | \"hirschberg_pmax\" — see SeqPairBatch.\n"
            "  track / gap_track (keyword): parameter sets per residue of seq_a and per DP\n"
            "              row for gaps in seq_a — see SeqPairBatch.add_many.\n"
            "A pair taken from a batch (batch[i]) is a view of that batch's pair i: its\n"
            "results are the batch's, and its settings (params, fill, soft options,\n"
            "hb_cutoff) are the batch's too — set them on the batch.")
        .def(
            "set_params",
            [](SP& self, const AlignParams& params) { self.set_params(params); },
            nb::arg("params"),
            "Swap alignment parameters (a standalone pair; a batch's pair takes the\n"
            "batch's set_params).  Invalidates the score, gradient and stored path\n"
            "(aligned() raises until the next align); guide_j is kept for banded_align().")
        .def("align", &SP::align, nb::arg("keep_paths") = true,
             "Full computation. Returns the score and publishes the configured gradient\n"
             "immediately (grad is None in none mode). Stores the path by default.")
        .def("banded_align", &SP::banded_align, nb::arg("bandwidth"),
             nb::arg("keep_paths") = true,
             "Compute around the cached guide. Returns the score and publishes the\n"
             "configured gradient immediately. Requires an established guide.")
        .def("score_and_grad", &SP::score_and_grad, nb::arg("keep_paths") = true,
             "Full DP with immediate gradient. Returns (score, AlignParams grad).\n"
             "keep_paths=True preserves aligned()/coordinates() availability; False\n"
             "skips storing the path. Raises if grad_mode is \"none\".")
        .def("drop_paths", &SP::drop_paths,
             "Free the stored path and reusable scratch memory; scores, gradients and guides survive.")
        .def("align_full", [](SP& s) {
            warn_pair_api("align_full()", "align() or score_and_grad()");
            PairApiCompat::full(s);
        }, "Deprecated in 0.7; removed in 0.8. Stores paths and defers gradient visibility.")
        .def("realign_banded", [](SP& s, int bandwidth) {
            warn_pair_api("realign_banded()", "banded_align()");
            PairApiCompat::banded(s, bandwidth);
        }, nb::arg("bandwidth"), "Deprecated in 0.7; removed in 0.8. Defers gradient visibility.")
        .def("compute_grad", [](SP& s) {
            warn_pair_api("compute_grad()", "grad after align() or banded_align()");
            PairApiCompat::publish(s);
        }, "Deprecated in 0.7; removed in 0.8. Releases a deferred cached gradient.")
        .def("drop_dp", [](SP& s) {
            warn_pair_api("drop_dp()", "drop_paths()");
            s.drop_paths();
        }, "Deprecated in 0.7; removed in 0.8. Use drop_paths().")
        .def("aligned", &SP::aligned,
             "Return the alignment as a pair of gapped strings (seq_a, seq_b).")
        .def(
            "coordinates",
            [](const SP& s) {
                auto [ci, cj] = s.coordinates();
                const size_t k = ci.size();
                int64_t* buf = new int64_t[2 * k];
                std::copy(ci.begin(), ci.end(), buf);
                std::copy(cj.begin(), cj.end(), buf + k);
                nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<int64_t*>(p); });
                size_t shape[2] = {2, k};
                return nb::ndarray<nb::numpy, int64_t>(buf, 2, shape, owner);
            },
            "The alignment as Biopython-style coordinates: an int64 array of shape (2, k),\n"
            "row 0 positions in seq_a and row 1 in seq_b, with a column at the start, at\n"
            "each change between aligned and gap columns, and at the end.  Pass it to\n"
            "Bio.Align.Alignment([seq_a, seq_b], coordinates).  A local alignment starts\n"
            "where its path starts.  Same availability as aligned().")
        .def(
            "formatted",
            [](const SP& self, int width) {
                auto [a, b] = self.aligned();
                return format_alignment(a, b, width);
            },
            nb::arg("width") = 60,
            "Pretty-printed alignment block, wrapped at `width` columns (0 = no wrap).")
        .def_prop_ro(
            "score",
            [](const SP& self) -> nb::object {
                if (!self.score_valid()) return nb::none();
                return nb::cast(self.score());
            },
            "Alignment score, or None if not yet computed.")
        .def_prop_ro(
            "grad",
            [](const SP& self) -> nb::object {
                if (!self.grad_valid()) return nb::none();
                return nb::cast(self.grad());
            },
            "Gradient as an AlignParams object, or None if not computed.")
        .def_prop_ro(
            "guide_j",
            [](SP& self) -> nb::object {
                if (!self.path_valid()) return nb::none();
                return nb::cast(self.guide_j());
            },
            "Current alignment as a guide_j vector (length m+1), or None.")
        .def_prop_ro("traceback",
                     [](const SP& s) { return traceback_name(s.traceback()); },
                     "The resolved traceback mode (never \"auto\"): \"pointers\" | \"scores\" |\n"
                     "\"hirschberg\" | \"hirschberg_pmax\" — see SeqPairBatch.traceback.")
        .def_prop_ro("path_valid",  &SP::path_valid)
        .def_prop_ro("score_valid", &SP::score_valid)
        .def_prop_ro("grad_valid",  &SP::grad_valid)
        .def_prop_ro("stored_path_valid", &SP::stored_path_valid,
                     "Whether a stored path is available (aligned(), coordinates()).")
        .def_prop_ro("dp_valid", [](const SP& s) {
            warn_pair_api("dp_valid", "stored_path_valid");
            return s.stored_path_valid();
        }, "Deprecated in 0.7; removed in 0.8. Use stored_path_valid.")
        .def_prop_rw(
            "hb_cutoff",
            [](const SP& s) { return s.hb_cutoff(); },
            [](SP& s, int v) { s.set_hb_cutoff(v); },
            "Hirschberg base-case size in rows — see SeqPairBatch.hb_cutoff.")
        .def_prop_rw(
            "fill",
            [](const SP& s) { return s.rowwise_full() ? "rowwise" : "striped"; },
            [](SP& s, const std::string& v) { s.set_rowwise_full(parse_fill(v)); },
            "Full-DP simd fill at double precision — see SeqPairBatch.fill.")
        .def_prop_rw(
            "soft_impl",
            [](const SP& s) { return soft_impl_name(s.soft_impl()); },
            [](SP& s, const std::string& v) { s.set_soft_impl(parse_soft_impl(v)); },
            NWGRAD_SOFT_IMPL_DOC)
        .def_prop_rw(
            "soft_temperature",
            [](const SP& s) { return s.soft_temperature(); },
            [](SP& s, double v) { s.set_soft_temperature(v); },
            NWGRAD_SOFT_TEMP_DOC)
        .def_prop_ro("gap_model", [](const SP& s) { return gap_model_name(s.gap_model()); })
        .def_prop_ro("mode",      [](const SP& s) { return align_mode_name(s.align_mode()); })
        .def_prop_ro("grad_mode", [](const SP& s) { return grad_mode_name(s.grad_mode()); })
        .def_prop_ro("seq_a", [](const SP& s) { return s.seq_a(); })
        .def_prop_ro("seq_b", [](const SP& s) { return s.seq_b(); })
        .def_prop_ro("track", [](const SP& s) -> nb::object {
            if (!s.tracked()) return nb::none();
            return nb::cast(s.track());
        }, "The pair's parameter set per residue of seq_a (None when not tracked).")
        .def_prop_ro("gap_track", [](const SP& s) -> nb::object {
            if (!s.tracked()) return nb::none();
            return nb::cast(s.gap_track());
        }, "The pair's parameter set per DP row 0..len(seq_a) for gaps in seq_a (None when\n"
           "not tracked).");
}

// ── SeqPairBatch ─────────────────────────────────────────────────────────────
template<class T>
static void bind_seq_pair_batch(nb::module_& m, const char* name) {
    using SPB = SeqPairBatchT<T>;
    using SP  = SeqPairT<T>;
    nb::class_<SPB>(m, name, nb::dynamic_attr())
        .def(
            "__init__",
            [](SPB* self, int n_threads, const std::string& traceback,
               const std::string& gap_model, const std::string& mode,
               const std::string& grad_mode) {
                new (self) SPB(parse_gap_model(gap_model), parse_align_mode(mode),
                               parse_grad_mode<GradMode>(grad_mode), n_threads,
                               parse_traceback(traceback));
            },
            nb::arg("n_threads") = 0, nb::arg("traceback") = "auto", nb::kw_only(),
            nb::arg("gap_model") = "affine", nb::arg("mode") = "global",
            nb::arg("grad_mode") = "hard",
            "A batch of sequence pairs of ONE problem type, aligned in parallel.\n"
            "  gap_model : \"affine\" (default) | \"linear\"\n"
            "  mode      : \"global\" (default) | \"local\"\n"
            "  grad_mode : \"hard\" (default) | \"soft\" | \"none\"\n"
            "     — fixed at construction, including on an empty batch.\n"
            "  n_threads=0 (default) uses the PHYSICAL core count among the CPUs this\n"
            "  process may run on (its affinity mask; falling back to the allowed\n"
            "  logical count): this DP is stall-bound, so SMT siblings\n"
            "  contend and the logical count measured up to 1.44x slower.\n"
            "  traceback : \"auto\" (default) | \"pointers\" | \"scores\" | \"hirschberg\"\n"
            "              | \"hirschberg_pmax\"\n"
            "     — what the DP retains in order to recover predecessors.\n"
            "     \"auto\": affine Full at float32 uses \"hirschberg_pmax\" for both\n"
            "     global and local; at float64, global uses \"hirschberg\" and local\n"
            "     uses \"pointers\".  Linear uses pointers (small problems keep\n"
            "     scores); banded uses score tables.  The prefix-max carry is\n"
            "     1.5-3.9x faster on homologous data, with measured float32 shortfall\n"
            "     ~5e-4.  Ask for \"hirschberg\" to use the exact carry.\n"
            "     \"pointers\" records 1 byte/cell/state during the fill (3 B/cell);\n"
            "     \"scores\" keeps VM/VX/VY and re-derives the argmax (12 B/cell).  Those\n"
            "     two are bit-identical; pointers is smaller and 1.4-2.2x faster.\n"
            "     \"hirschberg\" is divide-and-conquer in O(n) memory at ~2x the cell\n"
            "     work; it never splits below hb_cutoff, so pairs that short run the\n"
            "     exact pointers fill and are bit-exact, and only longer pairs take a\n"
            "     valid-but-different subgradient.  Full DP only, affine or linear,\n"
            "     global or local; asking for it on a banded problem throws.\n"
            "     Above hb_cutoff it needs non-negative gap penalties (linear ignores\n"
            "     opens).  An explicit request raises on negative penalties; auto\n"
            "     uses pointers instead, increasing memory from linear to O(m*n).\n"
            "     \"hirschberg_pmax\" is the same algorithm with the VY gap carry\n"
            "     computed by a closed-form prefix max instead of the serial chain.\n"
            "     It is the ONE mode here that can return a SUBOPTIMAL path: the ramp\n"
            "     k*gap_extend_a is added and subtracted again, costing a rounding that\n"
            "     grows with the column index.  It is the float32 \"auto\" default and\n"
            "     opt-in at double; it stays bit-identical across ISA levels but not\n"
            "     with any other mode.\n"
            "Pairs live in the batch's own arrays (codes, scores, guides, gradients, a few\n"
            "hundred bytes per pair) and are aligned on per-thread buffers.")
        .def(
            "add_many",
            [](SPB& self, const std::vector<std::string_view>& seqs_a,
               const std::vector<std::string_view>& seqs_b, const AlignParams& params,
               const std::string& kernel, nb::handle tracks, nb::handle gap_tracks) {
                const auto [la, lg] = track_lengths(seqs_a);
                const auto tk = to_tracks(tracks, la, "tracks");
                const auto gk = to_tracks(gap_tracks, lg, "gap_tracks");
                self.add_many(seqs_a, seqs_b, params, parse_backend(kernel), tk.view(), gk.view());
            },
            nb::arg("seqs_a"), nb::arg("seqs_b"), nb::arg("params"), nb::arg("kernel") = "auto",
            nb::kw_only(), nb::arg("tracks").none() = nb::none(), nb::arg("gap_tracks").none() = nb::none(),
            "Append pairs under an owned copy of params (one segment). Sequences are\n"
            "validated and encoded in parallel; a bad character raises and adds nothing.\n"
            "The problem type is fixed by the constructor.\n"
            "\n"
            "Tracks.  params may hold K parameter sets (AlignParams.from_arrays /\n"
            "AlignParams.stack); each residue of sequence A is then scored by the set its\n"
            "track names.  tracks: one sequence per pair, len(seqs_a[i]) set indices in\n"
            "[0, K); gap_tracks: one per pair, len(seqs_a[i]) + 1 indices, the set of each\n"
            "DP row 0..len(A) for gaps in A.  Either may be None (every pair defaults), or an\n"
            "entry None (that pair defaults): the track to all set 0, the gap track to\n"
            "row 0 -> track[0] and row i -> track[i-1].  Within row i (1-based, after\n"
            "residue i-1 of A): a match uses set track[i-1]'s matrix; a gap in B there\n"
            "(A's residue against '-') costs track[i-1]'s gap_extend_b, plus its\n"
            "gap_open_b if the gap run starts in that row; a gap in A within row i costs\n"
            "gap_tracks[i]'s gap_open_a / gap_extend_a.  The gradient then has K sets\n"
            "too.  A batch with tracks (or K > 1) is TRACKED: every pair carries tracks,\n"
            "and every set_params() must keep K.  Its hard gradients are stored as paths\n"
            "(2 bits per step) and summed on demand, so memory does not grow with K.\n"
            "Either may also be ONE flat 1-D integer numpy array: every pair's entries\n"
            "concatenated in pair order (sum(len(a)) for tracks, sum(len(a) + 1) for\n"
            "gap_tracks) — no per-pair Python objects, much cheaper for many pairs.")
        .def("__len__", &SPB::size)
        .def(
            "__getitem__",
            [](nb::object self_obj, long i) -> nb::object {
                SPB& self = nb::cast<SPB&>(self_obj);
                const long n = static_cast<long>(self.size());
                if (i < 0) i += n;
                if (i < 0 || i >= n) throw nb::index_error("SeqPairBatch index out of range");
                nb::object w = nb::cast(SP(self, static_cast<size_t>(i)));
                w.attr("_owner") = self_obj;   // the view keeps its batch alive
                return w;
            },
            nb::arg("i"),
            "Pair i as a SeqPair view (score, grad, guide_j, aligned(), ...).")
        .def(
            "set_params",
            [](SPB& self, const AlignParams& params) { self.set_params(params); },
            nb::arg("params"),
            "Set the params of every pair (one alphabet).  Clears the cached scores,\n"
            "gradients and stored paths; guides stay, for banded_grad().")
        .def(
            "score_and_grad",
            [](SPB& self, bool keep_paths) { return self.score_and_grad(keep_paths); },
            nb::arg("keep_paths") = false,
            "Full DP on every pair: score, guide and (unless grad_mode \"none\") gradient,\n"
            "cached per pair.  keep_paths=True also stores each alignment path\n"
            "(batch[i].aligned() / .coordinates(); a few bytes per column).  Returns the\n"
            "sum of scores, in pair order.")
        .def(
            "banded_grad",
            [](SPB& self, int bandwidth, bool keep_paths) {
                return self.banded_grad(bandwidth, keep_paths);
            },
            nb::arg("bandwidth"), nb::arg("keep_paths") = false,
            "Banded re-align + gradient around each pair's cached guide path: run\n"
            "score_and_grad() once, then set_params() + banded_grad(bw) after each update;\n"
            "no full DP is run.  Returns the sum of scores.")
        .def(
            "align",
            [](const SPB& self, const std::vector<std::string_view>& seqs_a,
               const std::vector<std::string_view>& seqs_b, const AlignParams& params, int band,
               const std::vector<std::string>& aligned_a, const std::vector<std::string>& aligned_b,
               const std::string& kernel, nb::handle tracks, nb::handle gap_tracks) {
                if (seqs_a.size() != seqs_b.size())
                    throw nb::value_error("nwgrad: align() needs seqs_a and seqs_b of equal length");
                const auto [la, lg] = track_lengths(seqs_a);
                auto tk = to_tracks(tracks, la, "tracks");
                auto gk = to_tracks(gap_tracks, lg, "gap_tracks");
                const bool guides = !aligned_a.empty() || !aligned_b.empty();
                if (guides && (aligned_a.size() != seqs_a.size() || aligned_b.size() != seqs_a.size()))
                    throw nb::value_error("nwgrad: aligned_a / aligned_b need one entry per pair");
                std::vector<ProblemInstance> problems(seqs_a.size());
                for (size_t i = 0; i < seqs_a.size(); ++i) {
                    problems[i].seq_a = seqs_a[i];
                    problems[i].seq_b = seqs_b[i];
                    // Straight through guide_j_from_aligned, which validates the two
                    // strings (as BatchAligner.align did); make_guide would read an empty
                    // string as "no guide" and silently band around the diagonal.
                    if (guides) problems[i].guide_j = guide_j_from_aligned(aligned_a[i], aligned_b[i]);
                }
                StreamTracks st;
                st.tracks = tk.view();
                st.gap_tracks = gk.view();
                return self.align(problems, params, band, parse_backend(kernel), st);
            },
            nb::arg("seqs_a"), nb::arg("seqs_b"), nb::arg("params"), nb::arg("band") = 0,
            nb::arg("aligned_a") = std::vector<std::string>{},
            nb::arg("aligned_b") = std::vector<std::string>{}, nb::arg("kernel") = "auto",
            nb::kw_only(), nb::arg("tracks").none() = nb::none(), nb::arg("gap_tracks").none() = nb::none(),
            "Align (seqs_a[i], seqs_b[i]) under `params` WITHOUT adding them to the batch\n"
            "(what BatchAligner was): returns a BatchResult (scores, gradient summed over\n"
            "the pairs), storing nothing per pair, so memory stays O(threads).  Uses the\n"
            "batch's problem type, grad mode, threads and settings.  band > 0 bands every\n"
            "pair (aligned_a / aligned_b, one gapped string per pair, give the guides; the\n"
            "diagonal otherwise).  Soft: forward-backward only.  tracks / gap_tracks: as\n"
            "add_many's; the summed gradient then has params' K sets.")
        .def("compute_grad", [](SPB& self) { return self.compute_grad(); },
             nb::rv_policy::move,
             "The gradient summed over all pairs (cached; no DP).  Summed in fixed blocks\n"
             "of pairs, so it is bit-reproducible whatever n_threads.")
        .def("scores", [](const SPB& self) { return numpy_1d(self.scores()); },
             "The pairs' cached scores as a float64 array, in pair order.  Runs no\n"
             "alignment; raises if any pair has no valid score.")
        .def(
            "weighted_grad",
            [](const SPB& self,
               nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu> weights) {
                return self.weighted_grad(weights.data(), weights.shape(0));
            },
            nb::arg("weights"), nb::rv_policy::move,
            "sum_i weights[i] * grad_i over the pairs' CACHED gradients, as one\n"
            "AlignParams.  Runs no alignment: call score_and_grad() first, derive the\n"
            "weights from scores() if they depend on them, then call this.  Summed in\n"
            "fixed blocks of pairs (in parallel), then over the blocks in order, so the\n"
            "result does not depend on n_threads.  Raises on an empty batch, on\n"
            "len(weights) != len(batch), and on a pair without a valid gradient.")
        .def(
            "grads",
            [](const SPB& self) {
                const size_t N = self.size();
                const size_t n = static_cast<size_t>(self.alphabet().size());   // throws if empty
                const bool tr = self.tracked();
                const size_t K = tr ? static_cast<size_t>(self.num_sets()) : 1;
                double* mats = new double[std::max<size_t>(N * K * n * n, 1)];
                nb::capsule mats_owner(mats, [](void* p) noexcept { delete[] static_cast<double*>(p); });
                double* gaps = new double[std::max<size_t>(N * K * 4, 1)];
                nb::capsule gaps_owner(gaps, [](void* p) noexcept { delete[] static_cast<double*>(p); });
                self.grads_into(mats, gaps);
                if (tr) {
                    size_t mats_shape[4] = {N, K, n, n};
                    size_t gaps_shape[3] = {N, K, 4};
                    return nb::make_tuple(nb::ndarray<nb::numpy, double>(mats, 4, mats_shape, mats_owner),
                                          nb::ndarray<nb::numpy, double>(gaps, 3, gaps_shape, gaps_owner));
                }
                size_t mats_shape[3] = {N, n, n};
                size_t gaps_shape[2] = {N, 4};
                return nb::make_tuple(nb::ndarray<nb::numpy, double>(mats, 3, mats_shape, mats_owner),
                                      nb::ndarray<nb::numpy, double>(gaps, 2, gaps_shape, gaps_owner));
            },
            "The pairs' CACHED gradients as two float64 arrays, in pair order:\n"
            "(matrices, gaps).  matrices has shape (N, n, n), rows and columns in the\n"
            "order of the batch's alphabet; gaps has shape (N, 4), with columns\n"
            "gap_open_a, gap_extend_a, gap_open_b, gap_extend_b.  Runs no alignment.\n"
            "Raises on an empty batch and on a pair without a valid gradient.  A tracked\n"
            "batch returns every pair's K sets: shapes (N, K, n, n) and (N, K, 4) —\n"
            "N*K*(n*n+4) doubles, so for many pairs prefer weighted_grad().")
        .def_prop_ro("tracked", [](const SPB& s) { return s.tracked(); },
                     "Whether the pairs carry tracks (add_many(tracks=...) or K > 1 sets).")
        .def_prop_ro("num_sets", [](const SPB& s) { return s.num_sets(); },
                     "K, the number of parameter sets of the batch's params (0 when empty).")
        .def_prop_rw(
            "track_storage",
            [](const SPB& s) { return s.track_counts() ? "counts" : "paths"; },
            [](SPB& s, const std::string& v) {
                if      (v == "paths")  s.set_track_counts(false);
                else if (v == "counts") s.set_track_counts(true);
                else throw nb::value_error(
                    ("nwgrad: unknown track_storage \"" + v +
                     "\" (expected \"paths\" or \"counts\")").c_str());
            },
            "How a tracked batch keeps its hard gradients between score_and_grad() /\n"
            "banded_grad() and weighted_grad() / grads().\n"
            "  \"paths\" (default): each pair's alignment path, 2 bits a move, replayed\n"
            "     through the tracks on every read.  ~35 B a miRNA-sized pair.\n"
            "  \"counts\": each pair's per-set field counts, counted during the\n"
            "     traceback.  Reads are a plain sparse sum: measured 5-15% faster per\n"
            "     training step on linear gaps, about the same on affine, at ~10x the\n"
            "     memory (120-370 B a pair after a fill).\n"
            "Results are identical either way.  Changing it drops the stored hard\n"
            "gradients (scores and guides stay): run score_and_grad() or banded_grad()\n"
            "again.  No effect on an untracked batch or on soft gradients.")
        .def("drop_paths", &SPB::drop_paths,
             "Free the stored alignment paths (score_and_grad(keep_paths=True)).")
        .def_prop_ro(
            "alphabet",
            [](const SPB& self) { return self.alphabet().symbols(); },
            "The symbols of the alphabet every pair in the batch shares.  Raises on an\n"
            "empty batch.")
        .def_prop_ro("gap_model", [](const SPB& s) { return gap_model_name(s.gap_model()); })
        .def_prop_ro("mode", [](const SPB& s) { return align_mode_name(s.align_mode()); })
        .def_prop_ro("grad_mode", [](const SPB& s) { return grad_mode_name(s.grad_mode()); })
        .def_prop_rw(
            "schedule",
            [](const SPB& s) { return s.sorted_schedule() ? "sorted" : "dynamic"; },
            [](SPB& s, const std::string& v) {
                if      (v == "dynamic") s.set_sorted_schedule(false);
                else if (v == "sorted")  s.set_sorted_schedule(true);
                else throw nb::value_error(
                    ("nwgrad: unknown schedule \"" + v +
                     "\" (expected \"dynamic\" or \"sorted\")").c_str());
            },
            "Work-scheduling policy for score_and_grad() when fill is not \"interpair\".\n"
            "  \"dynamic\" (default): one atomic counter, tasks in insertion order.\n"
            "  \"sorted\": length-sorted equal-work chunks, one per thread, with a\n"
            "     reserve of the smallest tasks for threads that finish early.\n"
            "Bounds the DP memory high-water mark at sum-over-chunks rather than\n"
            "n_threads * the global maximum.  Results are identical either way.")
        .def_prop_rw(
            "fill",
            [](const SPB& s) {
                return s.inter_fill() ? "interpair" : s.rowwise_full() ? "rowwise" : "striped";
            },
            [](SPB& s, const std::string& v) {
                const auto [rowwise, inter] = parse_batch_fill(v);
                s.set_fill(rowwise, inter);
            },
            "How score_and_grad() and banded_grad() fill the DP.\n"
            "  \"interpair\" (default): several pairs at once, one per vector lane (W = 4\n"
            "     double / 8 float32 on AVX2), grouped by length (B lengths may differ by\n"
            "     up to 1.25x + 4 within a group).  Both precisions, both gap models,\n"
            "     any alphabet (over 8 letters: a per-row gathered profile), hard and\n"
            "     soft (the soft forward-backward is shared too), full DP and — affine,\n"
            "     hard — banded_grad().  Pairs it would slow down run their own fill in\n"
            "     the same pass: a group's tables past L2 (1.25 MiB; soft pass 512 KiB),\n"
            "     linear Global below 4 lanes or over 8 letters, Hirschberg pairs past\n"
            "     hb_cutoff, the scalar kernel, groups mixing params.  Measured on\n"
            "     nighthaven (AVX2): 0.2-0.5x striped's time on miRNA x site pairs.\n"
            "  \"striped\": every pair its own fill — the striped kernel at double, the\n"
            "     float32 / linear fills elsewhere.\n"
            "  \"rowwise\": as striped, but the full affine double fill is the row-wise\n"
            "     kernel (no lazy-F fixpoint; faster on short pairs than striped).\n"
            "Scores, paths and hard gradients are bit-identical whichever fill runs; soft\n"
            "results are tolerance-equal (the soft path is never bit-exact).\n"            "Applies to every pair in the batch.")
        .def_prop_rw(
            "soft_impl",
            [](const SPB& s) { return soft_impl_name(s.soft_impl()); },
            [](SPB& s, const std::string& v) { s.set_soft_impl(parse_soft_impl(v)); },
            NWGRAD_SOFT_IMPL_DOC "\nApplies to every pair in the batch.")
        .def_prop_rw(
            "soft_temperature",
            [](const SPB& s) { return s.soft_temperature(); },
            [](SPB& s, double v) { s.set_soft_temperature(v); },
            NWGRAD_SOFT_TEMP_DOC "\nApplies to every pair in the batch.")
        .def_prop_rw(
            "soft_guide",
            [](const SPB& s) {
                return s.soft_guide_lazy() ? "lazy" : s.soft_guide_posterior() ? "posterior" : "eager";
            },
            [](SPB& s, const std::string& v) {
                bool lazy = false, post = false;
                if      (v == "eager")     {}
                else if (v == "lazy")      lazy = true;
                else if (v == "posterior") post = true;
                else throw nb::value_error(("nwgrad: unknown soft_guide \"" + v +
                                            "\" (expected \"eager\", \"lazy\" or \"posterior\")").c_str());
                s.set_soft_guide(lazy, post);
            },
            "When soft pairs compute their guide path (the Viterbi alignment that\n"
            "realign_banded()/banded_grad() band around).\n"
            "  \"eager\" (default): score_and_grad() runs the guide Viterbi, as always.\n"
            "  \"lazy\": it does not; the guide is computed on first use (guide_j,\n"
            "     realign_banded, banded_grad) under the params CURRENT THEN — after\n"
            "     set_params() it follows the new params.\n"
            "     Saves the whole Viterbi for loops that rescore in full each step and\n"
            "     never band (soft-score continuation); use eager to band around the\n"
            "     scored path.\n"
            "  \"posterior\": it does not either; the guide comes from the forward-backward\n"
            "     itself — per row of A the column of greatest posterior mass, made\n"
            "     non-decreasing — under the params scored.  A band centre at no Viterbi\n"
            "     cost, but the posterior's centre line, not the Viterbi path.\n"
            "Applies to every pair in the batch.")
        .def_prop_rw(
            "hb_cutoff",
            [](const SPB& s) { return s.hb_cutoff(); },
            [](SPB& s, int v) { s.set_hb_cutoff(v); },
            "Rows per block at which Hirschberg stops splitting and solves the block\n"
            "outright with the (vectorized) pointers fill.  Applies to every pair; ignored\n"
            "unless the traceback resolves to Hirschberg.  Default 512.\n"
            "\n"
            "This is a THREE-way trade, not just a speed knob:\n"
            "  memory     the base case materializes 3*(rows+1)*(cols+1) direction\n"
            "             bytes, so the cutoff bounds the largest thing Hirschberg\n"
            "             ever allocates.  1 is pure Hirschberg (minimum memory).\n"
            "  overhead   smaller cutoff means more blocks and more recursion levels;\n"
            "             larger cutoff means bigger base cases that eventually re-enter\n"
            "             the memory wall on long pairs.\n"
            "  exactness  splits are the ONLY place the tie-break can diverge from\n"
            "             pointers, so a pair no longer than the cutoff never splits and\n"
            "             is bit-exact with pointers; only longer pairs take a\n"
            "             valid-but-different subgradient.  The cutoff sets that line.\n"
            "\n"
            "512 from a fleet sweep (sse2/avx2/avx512/neon, len 500-8000) with the base\n"
            "case vectorized: pairs <= 512 run at full pointer speed and bit-exact,\n"
            "longer pairs win 1.4-2.7x at high thread counts.")
        .def_prop_rw("reserve_frac", [](const SPB& s) { return s.reserve_frac(); },
                     [](SPB& s, double v) { s.set_reserve_frac(v); },
                "Fraction of total work held back as filler for early-finishing\n"
                "threads under schedule=\"sorted\".  Default 0.0 (no reserve): a\n"
                "300-arm sweep over 4 machines found no reserve was fastest on\n"
                "tailed length distributions and immaterial on flat ones.  Raise it\n"
                "only if your workload behaves unlike either.")
        .def_prop_rw("long_cost_ratio", [](const SPB& s) { return s.long_cost_ratio(); },
                     [](SPB& s, double v) { s.set_long_cost_ratio(v); },
                "schedule=\"sorted\" cost weight: how much more a cell costs once the DP\n"
                "tables no longer fit cache.  DEFAULT 1.0 = OFF: correcting the\n"
                "imbalance measured slower (it fixes balance but converts idle\n"
                "threads into memory contention).  Raise it to move the straggler\n"
                "toward the short-sequence chunks; expect to pay ~4-6%.")
        .def_prop_rw("weight_lo", [](const SPB& s) { return s.weight_lo(); },
                     [](SPB& s, double v) { s.set_weight_lo(v); },
                "Effective length below which cells are unweighted (default 500).")
        .def_prop_rw("weight_hi", [](const SPB& s) { return s.weight_hi(); },
                     [](SPB& s, double v) { s.set_weight_hi(v); },
                "Effective length at which the weight saturates (default 1700).")
        .def_prop_ro("traceback",
                     [](const SPB& s) { return traceback_name(s.traceback()); },
                     "The traceback mode the batch was constructed with — \"auto\" (the\n"
                     "default) is reported as-is (read batch[i].traceback for the mode it\n"
                     "resolves to).  See the constructor docstring for what each retains.")
        .def_prop_rw("profile", [](const SPB& s) { return s.profile(); },
                     [](SPB& s, bool v) { s.set_profile(v); },
                "Record per-thread phase timings during schedule=\"sorted\" runs "
                "(off by default).  Read them back with schedule_profile().")
        .def("schedule_profile",
             [](const SPB& self) {
                 nb::list out;
                 for (const auto& p : self.profile_out()) {
                     nb::dict d;
                     d["chunk_s"]       = p.chunk_s;
                     d["reserve_s"]     = p.reserve_s;
                     d["chunk_cells"]   = p.chunk_cells;
                     d["reserve_cells"] = p.reserve_cells;
                     d["chunk_tasks"]   = p.chunk_tasks;
                     d["reserve_tasks"] = p.reserve_tasks;
                     d["finish_s"]      = p.finish_s;
                     out.append(d);
                 }
                 return out;
             },
             "Per-thread phase timings from the last sorted-schedule run (needs "
             "profile=True).  One dict per worker: busy seconds, cells and task "
             "counts for the chunk and reserve phases, plus finish time.")
        .def_prop_ro("n_threads", [](const SPB& s) { return s.n_threads(); });
}

NB_MODULE(nwgrad_ext, m) {
    m.doc() = "nwgrad C++ nanobind module";

    // ── Alphabet ─────────────────────────────────────────────────────────────
    // Interned and immortal on the C++ side, so nanobind must never take
    // ownership: every binding hands back a reference to the one true instance.
    nb::class_<Alphabet>(m, "Alphabet")
        .def_static(
            "get",
            [](const std::string& symbols) -> const Alphabet& {
                return Alphabet::get(symbols);
            },
            nb::arg("symbols"),
            nb::rv_policy::reference,
            "Return the interned Alphabet for `symbols`, creating it if new.\n"
            "Calling this twice with the same symbols returns the same object.")
        .def_prop_ro("symbols", [](const Alphabet& a) { return a.symbols(); },
                     "The symbol string, in index order.")
        .def_prop_ro("size", &Alphabet::size, "Number of symbols, N.")
        .def("index_of", &Alphabet::index_of, nb::arg("c"),
             "Index of character `c`, or -1 if it is not in this alphabet.")
        .def("contains", &Alphabet::contains, nb::arg("c"))
        .def("encode",
             [](const Alphabet& self, const std::string& s) { return self.encode(s); },
             nb::arg("s"),
             "Validate and encode `s` to a list of alphabet indices.\n"
             "Raises ValueError on any character outside the alphabet.\n"
             "Case is significant: 'd' is not 'D'.")
        .def("decode",
             [](const Alphabet& self, const std::vector<uint8_t>& idx) {
                 return self.decode(idx);
             },
             nb::arg("indices"))
        .def("__len__", &Alphabet::size)
        .def("__eq__", [](const Alphabet& a, const Alphabet& b) { return &a == &b; },
             nb::is_operator())
        .def("__repr__", [](const Alphabet& a) {
            return "Alphabet(\"" + a.symbols() + "\")";
        });

    // Named alphabets.  Extensions append at the end, so the canonical 20 amino
    // acids keep indices 0-19 and an existing 20x20 matrix (BLOSUM62, PAM, ...)
    // embeds as the top-left block of any extended one.
    m.attr("DNA")         = nb::cast(&Alphabet::dna(),         nb::rv_policy::reference);
    m.attr("DNA_N")       = nb::cast(&Alphabet::dna_n(),       nb::rv_policy::reference);
    m.attr("RNA")         = nb::cast(&Alphabet::rna(),         nb::rv_policy::reference);
    m.attr("RNA_N")       = nb::cast(&Alphabet::rna_n(),       nb::rv_policy::reference);
    m.attr("PROTEIN")     = nb::cast(&Alphabet::protein(),     nb::rv_policy::reference);
    m.attr("PROTEIN_X")   = nb::cast(&Alphabet::protein_x(),   nb::rv_policy::reference);
    m.attr("PROTEIN_UO")  = nb::cast(&Alphabet::protein_uo(),  nb::rv_policy::reference);
    m.attr("PROTEIN_UOX") = nb::cast(&Alphabet::protein_uox(), nb::rv_policy::reference);

    // The alphabets nwgrad.matrices is expressed in.  Different orderings, not
    // extensions: NCBI columns run ARNDCQEG..., so BLOSUM62 does not embed in
    // PROTEIN_X.
    m.attr("NCBI_PROTEIN") = nb::cast(&Alphabet::ncbi_protein(), nb::rv_policy::reference);
    m.attr("IUPAC_DNA")    = nb::cast(&Alphabet::iupac_dna(),    nb::rv_policy::reference);

    // ── SubstMatrix ──────────────────────────────────────────────────────────
    nb::class_<SubstMatrix>(m, "SubstMatrix")
        .def(
            "__init__",
            [](SubstMatrix* self, nb_arr_f64 arr, const std::string& alphabet) {
                if (arr.shape(0) != arr.shape(1))
                    throw std::invalid_argument("matrix must be square");
                if (static_cast<size_t>(alphabet.size()) != arr.shape(0))
                    throw std::invalid_argument("alphabet length must match matrix size");
                new (self) SubstMatrix(arr.data(), Alphabet::get(alphabet));
            },
            nb::arg("matrix"),
            nb::arg("alphabet") = std::string(AA_ORDER),
            "Construct from an (N, N) float64 numpy array.\n"
            "alphabet: string of N symbols in row/column order "
            "(default: canonical AA order ACDEFGHIKLMNPQRSTVWY).")
        .def(
            "score",
            [](const SubstMatrix& self, const std::string& a, const std::string& b) {
                if (a.size() != 1 || b.size() != 1)
                    throw std::invalid_argument("score() expects single-character strings");
                return self.score(a[0], b[0]);
            },
            nb::arg("a"), nb::arg("b"),
            "Return substitution score for the two given single-character strings.")
        .def(
            "to_matrix",
            [](const SubstMatrix& self) {
                int n = self.size();
                double* buf = new double[n * n];
                self.to_array(buf);
                nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<double*>(p); });
                size_t shape[2] = {static_cast<size_t>(n), static_cast<size_t>(n)};
                return nb::ndarray<nb::numpy, double>(buf, 2, shape, owner);
            },
            "Export as an (N, N) float64 numpy array in alphabet order.")
        .def_prop_ro("size",     &SubstMatrix::size,  "Alphabet size N.")
        .def_prop_ro("alphabet", [](const SubstMatrix& s) { return s.order(); },
                     "Alphabet string (length N) defining row/column order.");

    // ── AlignParams ──────────────────────────────────────────────────────────
    nb::class_<AlignParams>(m, "AlignParams")
        .def(
            "__init__",
            [](AlignParams* self, nb_arr_f64 arr, const std::string& alphabet,
               double gap_open_a, double gap_extend_a,
               double gap_open_b, double gap_extend_b) {
                if (arr.shape(0) != arr.shape(1))
                    throw std::invalid_argument("matrix must be square");
                if (static_cast<size_t>(alphabet.size()) != arr.shape(0))
                    throw std::invalid_argument("alphabet length must match matrix size");
                new (self) AlignParams(SubstMatrix(arr.data(), Alphabet::get(alphabet)),
                                       gap_open_a, gap_extend_a, gap_open_b, gap_extend_b);
            },
            nb::arg("matrix"),
            nb::arg("alphabet")     = std::string(AA_ORDER),
            nb::arg("gap_open_a")   = 0.0,
            nb::arg("gap_extend_a") = 0.0,
            nb::arg("gap_open_b")   = 0.0,
            nb::arg("gap_extend_b") = 0.0,
            "Alignment parameters bundling a substitution matrix with asymmetric gap costs.\n"
            "  alphabet    : symbol order for the N×N matrix (default: canonical AA order)\n"
            "  gap_open_a / gap_extend_a : penalties for gaps in sequence A (Y state)\n"
            "  gap_open_b / gap_extend_b : penalties for gaps in sequence B (X state)\n"
            "All gap values default to 0.0.  An AlignParams always has an alphabet:\n"
            "to build a zero gradient accumulator, use one of the matrices you are\n"
            "already aligning with rather than a default-constructed AlignParams.")
        .def(
            "__init__",
            [](AlignParams* self, const SubstMatrix& matrix,
               double gap_open_a, double gap_extend_a,
               double gap_open_b, double gap_extend_b) {
                new (self) AlignParams(matrix, gap_open_a, gap_extend_a,
                                       gap_open_b, gap_extend_b);
            },
            nb::arg("matrix"),
            nb::arg("gap_open_a")   = 0.0,
            nb::arg("gap_extend_a") = 0.0,
            nb::arg("gap_open_b")   = 0.0,
            nb::arg("gap_extend_b") = 0.0,
            "Construct from a SubstMatrix (which carries its own alphabet) plus gap costs.\n"
            "Preferred over the (array, alphabet) form — no risk of an alphabet mismatch.")
        .def_static(
            "from_arrays",
            [](nb_arr_f64_3d mats, nb_arr_f64 gaps, const std::string& alphabet) {
                const size_t K = mats.shape(0), n = mats.shape(1);
                if (mats.shape(2) != n)
                    throw std::invalid_argument("nwgrad: matrices must have shape (K, N, N)");
                if (static_cast<size_t>(alphabet.size()) != n)
                    throw std::invalid_argument("nwgrad: alphabet length must match matrix size");
                if (gaps.shape(0) != K || gaps.shape(1) != 4)
                    throw std::invalid_argument(
                        "nwgrad: gaps must have shape (K, 4) — gap_open_a, gap_extend_a, "
                        "gap_open_b, gap_extend_b per set — for K = " + std::to_string(K) + " matrices");
                if (K == 0 || K > static_cast<size_t>(SubstMatrix::kMaxSets))
                    throw std::invalid_argument("nwgrad: the number of parameter sets must be in [1, " +
                                                std::to_string(SubstMatrix::kMaxSets) + "]");
                return AlignParams(SubstMatrix(mats.data(), Alphabet::get(alphabet), static_cast<int>(K)),
                                   gaps.data());
            },
            nb::arg("matrices"), nb::arg("gaps"), nb::arg("alphabet") = std::string(AA_ORDER),
            "K parameter sets from arrays: matrices (K, N, N) and gaps (K, 4), the gap\n"
            "columns being gap_open_a, gap_extend_a, gap_open_b, gap_extend_b.  Set k is\n"
            "what a track entry k selects (SeqPairBatch.add_many(tracks=...)).")
        .def_static(
            "stack", [](const std::vector<AlignParams>& parts) { return AlignParams::stack(parts); },
            nb::arg("sets"),
            "Stack params over one alphabet into one params with their sets in order\n"
            "(each part may hold several sets).")
        .def_prop_ro("num_sets", &AlignParams::num_sets,
                     "K, the number of parameter sets.  The plain fields (matrix, gap_open_a,\n"
                     "...) are set 0.")
        .def_prop_rw(
            "matrices",
            [](const AlignParams& self) {
                const size_t n = static_cast<size_t>(self.matrix.size()), K = static_cast<size_t>(self.num_sets());
                return numpy_copy(self.matrix.data(), K * n * n, {K, n, n});
            },
            [](AlignParams& self, nb_arr_f64_3d mats) {
                const size_t n = static_cast<size_t>(self.matrix.size()), K = static_cast<size_t>(self.num_sets());
                if (mats.shape(0) != K || mats.shape(1) != n || mats.shape(2) != n)
                    throw std::invalid_argument("nwgrad: matrices must have shape (K, N, N) = (" +
                                                std::to_string(K) + ", " + std::to_string(n) + ", " +
                                                std::to_string(n) + ")");
                std::copy(mats.data(), mats.data() + K * n * n, self.matrix.data());
            },
            "Every set's matrix as a (K, N, N) float64 array (a copy).  Assigning an array\n"
            "of the same shape replaces them.")
        .def_prop_rw(
            "gaps",
            [](const AlignParams& self) {
                const std::vector<double> g = self.all_gaps();
                const size_t K = static_cast<size_t>(self.num_sets());
                return numpy_copy(g.data(), g.size(), {K, 4});
            },
            [](AlignParams& self, nb_arr_f64 gaps) {
                const size_t K = static_cast<size_t>(self.num_sets());
                if (gaps.shape(0) != K || gaps.shape(1) != 4)
                    throw std::invalid_argument("nwgrad: gaps must have shape (K, 4) = (" +
                                                std::to_string(K) + ", 4)");
                for (size_t k = 0; k < K; ++k)
                    for (int f = 0; f < 4; ++f)
                        self.gap_ref(static_cast<int>(k), f) = gaps.data()[k * 4 + static_cast<size_t>(f)];
            },
            "Every set's gap fields as a (K, 4) float64 array (a copy): columns\n"
            "gap_open_a, gap_extend_a, gap_open_b, gap_extend_b.  Assignable.")
        .def(
            "__getitem__",
            [](const AlignParams& self, long k) {
                const long K = self.num_sets();
                if (k < 0) k += K;
                if (k < 0 || k >= K) throw nb::index_error("nwgrad: parameter set index out of range");
                return self.set(static_cast<int>(k));
            },
            nb::arg("k"), "Set k as a one-set AlignParams (a copy).")
        .def(
            "__setitem__",
            [](AlignParams& self, long k, const AlignParams& p) {
                const long K = self.num_sets();
                if (k < 0) k += K;
                if (k < 0 || k >= K) throw nb::index_error("nwgrad: parameter set index out of range");
                self.assign_set(static_cast<int>(k), p);
            },
            nb::arg("k"), nb::arg("params"), "Replace set k with a one-set AlignParams.")
        .def(
            "to_dict",
            [](const AlignParams& self) {
                const size_t n = static_cast<size_t>(self.matrix.size()), K = static_cast<size_t>(self.num_sets());
                nb::dict d;
                d["matrix"]       = numpy_copy(self.matrix.data(), n * n, {n, n});
                d["alphabet"]     = self.matrix.order();
                d["gap_open_a"]   = self.gap_open_a;
                d["gap_extend_a"] = self.gap_extend_a;
                d["gap_open_b"]   = self.gap_open_b;
                d["gap_extend_b"] = self.gap_extend_b;
                if (K > 1) {
                    d["num_sets"] = K;
                    d["matrices"] = numpy_copy(self.matrix.data(), K * n * n, {K, n, n});
                    const std::vector<double> g = self.all_gaps();
                    d["gaps"]     = numpy_copy(g.data(), g.size(), {K, 4});
                }
                return d;
            },
            "Return the parameters as a dict: 'matrix' (N×N array), 'alphabet', and the\n"
            "four gap fields — all of set 0 — and, when there are K > 1 sets, also\n"
            "'num_sets', 'matrices' (K×N×N) and 'gaps' (K×4) for every set.  Convenient\n"
            "for inspecting a gradient.")
        .def_prop_rw(
            "matrix",
            [](const AlignParams& self) { return self.matrix.slice(0); },
            [](AlignParams& self, const SubstMatrix& mat) {
                if (&mat.alphabet() != &self.matrix.alphabet())
                    throw std::invalid_argument(
                        "nwgrad: cannot assign a matrix over alphabet \"" +
                        mat.alphabet().symbols() + "\" to params over \"" +
                        self.matrix.alphabet().symbols() + "\"");
                self.matrix.assign_set(0, mat);
            },
            "The substitution matrix of set 0 (a copy; assign to replace it).")
        .def_rw("gap_open_a",   &AlignParams::gap_open_a)
        .def_rw("gap_extend_a", &AlignParams::gap_extend_a)
        .def_rw("gap_open_b",   &AlignParams::gap_open_b)
        .def_rw("gap_extend_b", &AlignParams::gap_extend_b)
        .def("__add__",  [](const AlignParams& a, const AlignParams& b) { return a + b; })
        .def("__iadd__", [](AlignParams& a, const AlignParams& b) -> AlignParams& { a += b; return a; },
             nb::rv_policy::reference)
        .def("__mul__",  [](const AlignParams& a, double s) { return a * s; })
        .def("__rmul__", [](const AlignParams& a, double s) { return a * s; })
        .def("__imul__", [](AlignParams& a, double s) -> AlignParams& { a *= s; return a; },
             nb::rv_policy::reference)
        .def("__neg__",  [](const AlignParams& a) { return -a; })
        .def("__sub__",  [](const AlignParams& a, const AlignParams& b) { return a - b; })
        .def("__isub__", [](AlignParams& a, const AlignParams& b) -> AlignParams& { a -= b; return a; },
             nb::rv_policy::reference);

    // ── Build provenance ──────────────────────────────────────────────────────
    m.def(
        "compiled_with",
        []() { return std::string(NWGRAD_COMPILER); },
        "The compiler and version that built this extension, e.g. \"gcc 15.2.1\" or\n"
        "\"clang 18.1.3\".  Captured at compile time; the reliable way to tell which\n"
        "toolchain produced the loaded binary (gcc and clang differ measurably on the\n"
        "std::simd kernels, and clang strips the ELF .comment that would otherwise say).");

    // ── SIMD ISA reporting ────────────────────────────────────────────────────
    // A stable alias for get_isa_level(): both report the one ISA level that now
    // drives every simd kernel (striped Full and row-wise banded alike).
    m.def(
        "simd_isa",
        []() { return get_isa_level(); },
        "Which instruction set the simd Viterbi kernels are dispatched to on this\n"
        "CPU: \"sse2\" | \"avx2\" | \"avx512\" (x86) or \"neon\" (AArch64).  An alias\n"
        "for get_isa_level().\n"
        "\n"
        "Worth checking before you conclude the simd kernel did not help.  Prebuilt\n"
        "wheels carry every x86 level and pick at load; \"sse2\" (two doubles per\n"
        "vector) is the fallback for CPUs without AVX2.  A modern CPU should report \"avx2\" or better.\n"
        "\n"
        "Plain \"avx\" is never selected: it is a measured regression on Bulldozer/\n"
        "Piledriver, whose FP unit splits every 256-bit operation into two 128-bit\n"
        "halves, so AVX-only CPUs run the sse2 level.  Set NWGRAD_ISA, or call\n"
        "set_isa_level(), to force a level (one the CPU cannot run falls back).");

    // ── Per-ISA-level dispatch for the striped affine kernel ──────────────────
    m.def("available_isa_levels", []() { return available_isa_levels(); },
          "The ISA levels this CPU can actually run, weakest first "
          "(e.g. [\"scalar_fallback\", \"sse2\", \"avx2\"]).");
    m.def("get_isa_level", []() { return get_isa_level(); },
          "The ISA level the striped affine kernel is currently dispatched to.");
    m.def("set_isa_level", [](const std::string& name) { set_isa_level(name); },
          nb::arg("level"),
          "Force the striped kernel's ISA level (for testing).  You may force any\n"
          "level the CPU supports — forcing *down* (e.g. \"sse2\" on an AVX2 box)\n"
          "is how a level's bit-exactness is checked on capable hardware.  Forcing a\n"
          "level the CPU cannot run raises ValueError (it would SIGILL).  NOT\n"
          "thread-safe to change while work is in flight — set it before dispatching.\n"
          "NWGRAD_ISA=<level> does the same before the module loads.");

    m.def(
        "guide_j_from_aligned",
        [](const std::string& a_aligned, const std::string& b_aligned) {
            return guide_j_from_aligned(a_aligned, b_aligned);
        },
        nb::arg("a_aligned"), nb::arg("b_aligned"),
        "Convert a pair of aligned strings (using '-' for gaps) to a guide_j vector.\n"
        "Returns a list of length m+1 where guide_j[i] = column j after consuming i\n"
        "characters of a.");

    // ── Single-pair convenience functions (float32 default; _double variants) ──
    bind_convenience<float >(m, "");
    bind_convenience<double>(m, "_double");

    // ── BatchResult ──────────────────────────────────────────────────────────
    nb::class_<BatchResult>(m, "BatchResult")
        .def_prop_ro(
            "scores",
            [](const BatchResult& self) {
                size_t N = self.scores.size();
                double* buf = new double[N ? N : 1];
                std::copy(self.scores.begin(), self.scores.end(), buf);
                nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<double*>(p); });
                return nb_arr_f64_1d(buf, {N}, owner);
            },
            nb::rv_policy::move,
            "Alignment scores, shape (N,).")
        .def_prop_ro(
            "grad",
            [](const BatchResult& self) -> const AlignParams& { return self.grad; },
            nb::rv_policy::reference_internal,
            "Summed gradient over the batch as an AlignParams object.");

    // ── SeqPair (float32 default) / SeqPairDouble ─────────────────────────────
    bind_seq_pair<float >(m, "SeqPair");
    bind_seq_pair<double>(m, "SeqPairDouble");

    // ── SeqPairBatch (float32 default) / SeqPairBatchDouble ───────────────────
    bind_seq_pair_batch<float >(m, "SeqPairBatch");
    bind_seq_pair_batch<double>(m, "SeqPairBatchDouble");

    // ── logistic: a binary logistic link over batch scores (submodule) ────────
    bind_logistic(m);
}
