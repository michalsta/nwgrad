#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>

#include "align_params.hpp"
#include "aligner.hpp"
#include "batch.hpp"
#include "seq_pair.hpp"
#include "seq_pair_batch.hpp"

namespace nb = nanobind;
using nb_arr_f64    = nb::ndarray<double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using nb_arr_f64_1d = nb::ndarray<nb::numpy, double, nb::ndim<1>>;

// ── Aligner factory: picks Full or GuideBanded at runtime. ───────────────────

// "scalar" (default) | "simd".  The two Viterbi kernels write bit-identical tables,
// so this is a speed knob and never a correctness one.  Anything unrecognised is a
// typo, and a typo that silently gave you the slow path would be undetectable — so
// it throws.
inline DpKernel parse_kernel(const std::string& k) {
    if (k == "scalar") return DpKernel::Scalar;
    if (k == "simd")   return DpKernel::Simd;
    throw std::invalid_argument(
        "nwgrad: kernel must be \"scalar\" or \"simd\", got \"" + k + "\"");
}

#define WITH_ALIGNER(GM, AM, band, guide_j, body)                                        \
    do {                                                                                  \
        DpBuffer _buf;                                                                    \
        const DpKernel _kern = parse_kernel(kernel);                                      \
        if ((band) > 0 || !(guide_j).empty()) {                                          \
            Aligner<GapModel::GM, AlignMode::AM, AlignBand::GuideBanded> al;             \
            al.set_kernel(_kern);                                                          \
            body                                                                           \
        } else {                                                                           \
            Aligner<GapModel::GM, AlignMode::AM, AlignBand::Full> al;                    \
            al.set_kernel(_kern);                                                          \
            body                                                                           \
        }                                                                                  \
    } while (0)

// The convenience functions still take plain `str`: they validate and encode
// against the params' alphabet here, at the boundary.  An out-of-alphabet
// character throws.  The encoded vectors must outlive the set_problem() call,
// so they live in the caller's frame.
struct EncodedPair {
    std::vector<uint8_t> a, b;
    EncodedPair(std::string_view sa, std::string_view sb, const AlignParams& p)
        : a(p.matrix.alphabet().encode(sa)), b(p.matrix.alphabet().encode(sb)) {}
};

// Hold a reference to the params a SeqPair / SeqPairBatch currently points at.
//
// C++ stores params as a bare pointer, so Python must keep the object alive.
// nb::keep_alive<1,2> is the obvious tool and the wrong one here, twice over:
//
//   - It appends to a linked list that nanobind walks on every call to
//     deduplicate (nb_type.cpp:1585), so K swaps cost O(K^2).
//   - It never releases.  A gradient-descent loop builds a fresh AlignParams
//     each step, so every superseded one stays pinned for the object's life.
//
// Only the *current* params needs to be alive — C++ has already dropped the
// pointer to the old one.  So overwrite a single attribute: O(1), and the
// superseded params is released on the spot.
static void keep_current_params(nb::object owner, nb::object params) {
    owner.attr("_params") = params;
}

static std::vector<int> make_guide(const std::string& aligned_a,
                                    const std::string& aligned_b) {
    if (!aligned_a.empty() && !aligned_b.empty())
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
        .def(
            "to_dict",
            [](const AlignParams& self) {
                int n = self.matrix.size();
                double* buf = new double[n * n];
                self.matrix.to_array(buf);
                nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<double*>(p); });
                size_t shape[2] = {static_cast<size_t>(n), static_cast<size_t>(n)};
                nb::dict d;
                d["matrix"]       = nb::ndarray<nb::numpy, double>(buf, 2, shape, owner);
                d["alphabet"]     = self.matrix.order();
                d["gap_open_a"]   = self.gap_open_a;
                d["gap_extend_a"] = self.gap_extend_a;
                d["gap_open_b"]   = self.gap_open_b;
                d["gap_extend_b"] = self.gap_extend_b;
                return d;
            },
            "Return the parameters as a dict: 'matrix' (N×N array), 'alphabet', and the\n"
            "four gap fields. Convenient for inspecting a gradient.")
        .def_prop_rw(
            "matrix",
            [](const AlignParams& self) { return self.matrix; },
            [](AlignParams& self, const SubstMatrix& mat) {
                if (&mat.alphabet() != &self.matrix.alphabet())
                    throw std::invalid_argument(
                        "nwgrad: cannot assign a matrix over alphabet \"" +
                        mat.alphabet().symbols() + "\" to params over \"" +
                        self.matrix.alphabet().symbols() + "\"");
                self.matrix = mat;
            })
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

    // ── Guide alignment utility ──────────────────────────────────────────────
    m.def(
        "simd_isa",
        []() { return std::string(nwgrad_simd::active_isa()); },
        "Which instruction set the simd Viterbi kernel selected on this CPU:\n"
        "\"baseline\" | \"avx\" | \"avx2\" | \"avx512\".\n"
        "\n"
        "Worth checking before you conclude the simd kernel did not help.  Prebuilt\n"
        "wheels are compiled for the x86-64 baseline, so \"baseline\" means SSE2 —\n"
        "two doubles per vector.  A modern CPU should report \"avx2\" or better.\n"
        "\n"
        "Plain \"avx\" is never selected automatically: it is a measured regression on\n"
        "Bulldozer/Piledriver, whose FP unit splits every 256-bit operation into two\n"
        "128-bit halves.  Set NWGRAD_ISA=baseline|avx|avx2|avx512 to override the\n"
        "probe (an ISA this CPU cannot run falls back rather than crashing).");

    // ── Per-ISA-level dispatch for the striped affine kernel ──────────────────
    m.def("available_isa_levels", []() { return available_isa_levels(); },
          "The ISA levels this CPU can actually run, weakest first "
          "(e.g. [\"baseline\", \"avx2\"]).");
    m.def("get_isa_level", []() { return get_isa_level(); },
          "The ISA level the striped affine kernel is currently dispatched to.");
    m.def("set_isa_level", [](const std::string& name) { set_isa_level(name); },
          nb::arg("level"),
          "Force the striped kernel's ISA level (for testing).  You may force any\n"
          "level the CPU supports — forcing *down* (e.g. \"baseline\" on an AVX2 box)\n"
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

    // ── Single-pair score functions ──────────────────────────────────────────

    m.def(
        "nw_score",
        [](const std::string& a, const std::string& b,
           const AlignParams& params, int band,
           const std::string& aligned_a, const std::string& aligned_b,
           const std::string& kernel) {
            auto gj = make_guide(aligned_a, aligned_b);
            EncodedPair enc(a, b, params);
            WITH_ALIGNER(Linear, Global, band, gj, {
                al.set_problem(enc.a, enc.b, params, band, gj);
                al.compute_viterbi(_buf);
                return al.score();
            });
        },
        nb::arg("seq_a"), nb::arg("seq_b"), nb::arg("params"),
        nb::arg("band") = 0, nb::arg("aligned_a") = "", nb::arg("aligned_b") = "",
        nb::arg("kernel") = "scalar",
        "Needleman-Wunsch global alignment score (linear gap penalty).");

    m.def(
        "sw_score",
        [](const std::string& a, const std::string& b,
           const AlignParams& params, int band,
           const std::string& aligned_a, const std::string& aligned_b,
           const std::string& kernel) {
            auto gj = make_guide(aligned_a, aligned_b);
            EncodedPair enc(a, b, params);
            WITH_ALIGNER(Linear, Local, band, gj, {
                al.set_problem(enc.a, enc.b, params, band, gj);
                al.compute_viterbi(_buf);
                return al.score();
            });
        },
        nb::arg("seq_a"), nb::arg("seq_b"), nb::arg("params"),
        nb::arg("band") = 0, nb::arg("aligned_a") = "", nb::arg("aligned_b") = "",
        nb::arg("kernel") = "scalar",
        "Smith-Waterman local alignment score (linear gap penalty).");

    m.def(
        "nw_score_affine",
        [](const std::string& a, const std::string& b,
           const AlignParams& params, int band,
           const std::string& aligned_a, const std::string& aligned_b,
           const std::string& kernel) {
            auto gj = make_guide(aligned_a, aligned_b);
            EncodedPair enc(a, b, params);
            WITH_ALIGNER(Affine, Global, band, gj, {
                al.set_problem(enc.a, enc.b, params, band, gj);
                al.compute_viterbi(_buf);
                return al.score();
            });
        },
        nb::arg("seq_a"), nb::arg("seq_b"), nb::arg("params"),
        nb::arg("band") = 0, nb::arg("aligned_a") = "", nb::arg("aligned_b") = "",
        nb::arg("kernel") = "scalar",
        "Needleman-Wunsch global alignment score (affine gap penalty).");

    m.def(
        "sw_score_affine",
        [](const std::string& a, const std::string& b,
           const AlignParams& params, int band,
           const std::string& aligned_a, const std::string& aligned_b,
           const std::string& kernel) {
            auto gj = make_guide(aligned_a, aligned_b);
            EncodedPair enc(a, b, params);
            WITH_ALIGNER(Affine, Local, band, gj, {
                al.set_problem(enc.a, enc.b, params, band, gj);
                al.compute_viterbi(_buf);
                return al.score();
            });
        },
        nb::arg("seq_a"), nb::arg("seq_b"), nb::arg("params"),
        nb::arg("band") = 0, nb::arg("aligned_a") = "", nb::arg("aligned_b") = "",
        nb::arg("kernel") = "scalar",
        "Smith-Waterman local alignment score (affine gap penalty).");

    // ── Single-pair hard gradient functions ─────────────────────────────────

    m.def(
        "nw_grad",
        [](const std::string& a, const std::string& b,
           const AlignParams& params, int band,
           const std::string& aligned_a, const std::string& aligned_b,
           const std::string& kernel) {
            auto gj = make_guide(aligned_a, aligned_b);
            EncodedPair enc(a, b, params);
            WITH_ALIGNER(Linear, Global, band, gj, {
                al.set_problem(enc.a, enc.b, params, band, gj);
                al.compute_viterbi(_buf);
                AlignParams grad = AlignParams::zeros_like(params);
                al.hard_grad(_buf, grad);
                return nb::make_tuple(al.score(), grad);
            });
        },
        nb::arg("seq_a"), nb::arg("seq_b"), nb::arg("params"),
        nb::arg("band") = 0, nb::arg("aligned_a") = "", nb::arg("aligned_b") = "",
        nb::arg("kernel") = "scalar",
        "NW global: returns (score, AlignParams grad) — hard subgradient (linear gap).");

    m.def(
        "sw_grad",
        [](const std::string& a, const std::string& b,
           const AlignParams& params, int band,
           const std::string& aligned_a, const std::string& aligned_b,
           const std::string& kernel) {
            auto gj = make_guide(aligned_a, aligned_b);
            EncodedPair enc(a, b, params);
            WITH_ALIGNER(Linear, Local, band, gj, {
                al.set_problem(enc.a, enc.b, params, band, gj);
                al.compute_viterbi(_buf);
                AlignParams grad = AlignParams::zeros_like(params);
                al.hard_grad(_buf, grad);
                return nb::make_tuple(al.score(), grad);
            });
        },
        nb::arg("seq_a"), nb::arg("seq_b"), nb::arg("params"),
        nb::arg("band") = 0, nb::arg("aligned_a") = "", nb::arg("aligned_b") = "",
        nb::arg("kernel") = "scalar",
        "SW local: returns (score, AlignParams grad) — hard subgradient (linear gap).");

    m.def(
        "nw_affine_grad",
        [](const std::string& a, const std::string& b,
           const AlignParams& params, int band,
           const std::string& aligned_a, const std::string& aligned_b,
           const std::string& kernel) {
            auto gj = make_guide(aligned_a, aligned_b);
            EncodedPair enc(a, b, params);
            WITH_ALIGNER(Affine, Global, band, gj, {
                al.set_problem(enc.a, enc.b, params, band, gj);
                al.compute_viterbi(_buf);
                AlignParams grad = AlignParams::zeros_like(params);
                al.hard_grad(_buf, grad);
                return nb::make_tuple(al.score(), grad);
            });
        },
        nb::arg("seq_a"), nb::arg("seq_b"), nb::arg("params"),
        nb::arg("band") = 0, nb::arg("aligned_a") = "", nb::arg("aligned_b") = "",
        nb::arg("kernel") = "scalar",
        "NW global: returns (score, AlignParams grad) — hard subgradient (affine gap).");

    m.def(
        "sw_affine_grad",
        [](const std::string& a, const std::string& b,
           const AlignParams& params, int band,
           const std::string& aligned_a, const std::string& aligned_b,
           const std::string& kernel) {
            auto gj = make_guide(aligned_a, aligned_b);
            EncodedPair enc(a, b, params);
            WITH_ALIGNER(Affine, Local, band, gj, {
                al.set_problem(enc.a, enc.b, params, band, gj);
                al.compute_viterbi(_buf);
                AlignParams grad = AlignParams::zeros_like(params);
                al.hard_grad(_buf, grad);
                return nb::make_tuple(al.score(), grad);
            });
        },
        nb::arg("seq_a"), nb::arg("seq_b"), nb::arg("params"),
        nb::arg("band") = 0, nb::arg("aligned_a") = "", nb::arg("aligned_b") = "",
        nb::arg("kernel") = "scalar",
        "SW local: returns (score, AlignParams grad) — hard subgradient (affine gap).");

    // ── Single-pair soft gradient functions ─────────────────────────────────

    m.def(
        "nw_soft_grad",
        [](const std::string& a, const std::string& b,
           const AlignParams& params, int band,
           const std::string& aligned_a, const std::string& aligned_b,
           const std::string& kernel) {
            auto gj = make_guide(aligned_a, aligned_b);
            EncodedPair enc(a, b, params);
            WITH_ALIGNER(Linear, Global, band, gj, {
                al.set_problem(enc.a, enc.b, params, band, gj);
                al.compute_forward_back(_buf);
                AlignParams grad = AlignParams::zeros_like(params);
                al.soft_grad(_buf, grad);
                return nb::make_tuple(al.log_z(), grad);
            });
        },
        nb::arg("seq_a"), nb::arg("seq_b"), nb::arg("params"),
        nb::arg("band") = 0, nb::arg("aligned_a") = "", nb::arg("aligned_b") = "",
        nb::arg("kernel") = "scalar",
        "NW global: returns (log_Z, AlignParams grad) — soft gradient (linear gap).");

    m.def(
        "sw_soft_grad",
        [](const std::string& a, const std::string& b,
           const AlignParams& params, int band,
           const std::string& aligned_a, const std::string& aligned_b,
           const std::string& kernel) {
            auto gj = make_guide(aligned_a, aligned_b);
            EncodedPair enc(a, b, params);
            WITH_ALIGNER(Linear, Local, band, gj, {
                al.set_problem(enc.a, enc.b, params, band, gj);
                al.compute_forward_back(_buf);
                AlignParams grad = AlignParams::zeros_like(params);
                al.soft_grad(_buf, grad);
                return nb::make_tuple(al.log_z(), grad);
            });
        },
        nb::arg("seq_a"), nb::arg("seq_b"), nb::arg("params"),
        nb::arg("band") = 0, nb::arg("aligned_a") = "", nb::arg("aligned_b") = "",
        nb::arg("kernel") = "scalar",
        "SW local: returns (log_Z, AlignParams grad) — soft gradient (linear gap).");

    m.def(
        "nw_affine_soft_grad",
        [](const std::string& a, const std::string& b,
           const AlignParams& params, int band,
           const std::string& aligned_a, const std::string& aligned_b,
           const std::string& kernel) {
            auto gj = make_guide(aligned_a, aligned_b);
            EncodedPair enc(a, b, params);
            WITH_ALIGNER(Affine, Global, band, gj, {
                al.set_problem(enc.a, enc.b, params, band, gj);
                al.compute_forward_back(_buf);
                AlignParams grad = AlignParams::zeros_like(params);
                al.soft_grad(_buf, grad);
                return nb::make_tuple(al.log_z(), grad);
            });
        },
        nb::arg("seq_a"), nb::arg("seq_b"), nb::arg("params"),
        nb::arg("band") = 0, nb::arg("aligned_a") = "", nb::arg("aligned_b") = "",
        nb::arg("kernel") = "scalar",
        "NW global: returns (log_Z, AlignParams grad) — soft gradient (affine gap).");

    m.def(
        "sw_affine_soft_grad",
        [](const std::string& a, const std::string& b,
           const AlignParams& params, int band,
           const std::string& aligned_a, const std::string& aligned_b,
           const std::string& kernel) {
            auto gj = make_guide(aligned_a, aligned_b);
            EncodedPair enc(a, b, params);
            WITH_ALIGNER(Affine, Local, band, gj, {
                al.set_problem(enc.a, enc.b, params, band, gj);
                al.compute_forward_back(_buf);
                AlignParams grad = AlignParams::zeros_like(params);
                al.soft_grad(_buf, grad);
                return nb::make_tuple(al.log_z(), grad);
            });
        },
        nb::arg("seq_a"), nb::arg("seq_b"), nb::arg("params"),
        nb::arg("band") = 0, nb::arg("aligned_a") = "", nb::arg("aligned_b") = "",
        nb::arg("kernel") = "scalar",
        "SW local: returns (log_Z, AlignParams grad) — soft gradient (affine gap).");

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

    // ── BatchAligner ─────────────────────────────────────────────────────────
    nb::class_<BatchAligner>(m, "BatchAligner")
        .def(
            "__init__",
            [](BatchAligner* self,
               const AlignParams& params,
               int    band,
               const std::string& gap_model,
               const std::string& mode,
               const std::string& grad_mode,
               int n_threads,
               const std::string& kernel) {
                GapModel  gm = (gap_model == "affine") ? GapModel::Affine  : GapModel::Linear;
                AlignMode am = (mode      == "local")  ? AlignMode::Local  : AlignMode::Global;
                BatchAligner::GradMode gd;
                if      (grad_mode == "hard") gd = BatchAligner::GradMode::Hard;
                else if (grad_mode == "soft") gd = BatchAligner::GradMode::Soft;
                else                          gd = BatchAligner::GradMode::None;
                new (self) BatchAligner(params, band, gm, am, gd, n_threads,
                                        parse_kernel(kernel));
            },
            nb::arg("params"),
            nb::arg("band")       = 0,
            nb::arg("gap_model")  = "affine",
            nb::arg("mode")       = "global",
            nb::arg("grad_mode")  = "hard",
            nb::arg("n_threads")  = 1,
            nb::arg("kernel")     = "scalar",
            "Create a BatchAligner.\n"
            "  gap_model : \"linear\" | \"affine\"\n"
            "  mode      : \"global\" | \"local\"\n"
            "  grad_mode : \"hard\" | \"soft\" | \"none\"\n"
            "  band      : 0 = full DP; >0 = banded half-width\n"
            "  kernel    : \"scalar\" | \"simd\" — which Viterbi fills the DP tables.\n"
            "              The two are bit-exact: identical tables, identical\n"
            "              alignments, identical gradients.  Purely a speed knob.\n"
            "              It affects the Viterbi (hard-gradient) path only; soft\n"
            "              gradients are computed by the same shared code either way,\n"
            "              and the linear gap model has no simd kernel (its scalar\n"
            "              loop is already at its latency floor), so \"simd\" is a\n"
            "              no-op there.")
        .def(
            "align",
            [](const BatchAligner& self,
               const std::vector<std::string>& seqs_a,
               const std::vector<std::string>& seqs_b,
               const std::vector<std::string>& aligned_a,
               const std::vector<std::string>& aligned_b) -> BatchResult {
                if (seqs_a.size() != seqs_b.size())
                    throw std::invalid_argument(
                        "sequences_a and sequences_b must have the same length");
                const bool has_guide = !aligned_a.empty();
                if (has_guide) {
                    if (aligned_a.size() != seqs_a.size() || aligned_b.size() != seqs_a.size())
                        throw std::invalid_argument(
                            "aligned_a and aligned_b must have the same length as sequences");
                }
                const size_t N = seqs_a.size();
                std::vector<ProblemInstance> problems;
                problems.reserve(N);
                for (size_t k = 0; k < N; ++k) {
                    ProblemInstance pi{seqs_a[k], seqs_b[k], {}};
                    if (has_guide)
                        pi.guide_j = guide_j_from_aligned(aligned_a[k], aligned_b[k]);
                    problems.push_back(std::move(pi));
                }
                return self.align(problems);
            },
            nb::arg("sequences_a"),
            nb::arg("sequences_b"),
            nb::arg("aligned_a") = std::vector<std::string>{},
            nb::arg("aligned_b") = std::vector<std::string>{},
            "Align N sequence pairs.  Returns a BatchResult with .scores and .grad.");

    // ── SeqPair ───────────────────────────────────────────────────────────────
    // dynamic_attr() so set_params() can park a *replaceable* reference to the
    // current params on the instance; see the comment there.  An instance with
    // no attribute set carries only a null dict pointer, so the 2.5M-SeqPair
    // case pays 8 bytes each and no dict.
    nb::class_<SeqPair>(m, "SeqPair", nb::dynamic_attr())
        .def(
            "__init__",
            [](SeqPair* self,
               const std::string& seq_a, const std::string& seq_b,
               const AlignParams& params,
               const std::string& gap_model,
               const std::string& mode,
               const std::string& grad_mode,
               const std::string& kernel) {
                GapModel  gm = (gap_model == "affine") ? GapModel::Affine  : GapModel::Linear;
                AlignMode am = (mode      == "local")  ? AlignMode::Local  : AlignMode::Global;
                GradMode  gd;
                if      (grad_mode == "hard") gd = GradMode::Hard;
                else if (grad_mode == "soft") gd = GradMode::Soft;
                else                          gd = GradMode::None;
                new (self) SeqPair(seq_a, seq_b, params, gm, am, gd,
                                   parse_kernel(kernel));
            },
            nb::arg("seq_a"), nb::arg("seq_b"), nb::arg("params"),
            nb::arg("gap_model") = "affine", nb::arg("mode") = "global",
            nb::arg("grad_mode") = "hard", nb::arg("kernel") = "scalar",
            // Keeps the initial `params` alive; a SeqPair whose params is never
            // swapped would otherwise hold a dangling pointer.  keep_alive is
            // fine *here* — one patient, registered once, so it stays O(1).  It
            // is set_params() that must not use it; see keep_current_params().
            nb::keep_alive<1, 4>(),
            "Persistent sequence pair.\n"
            "  gap_model : \"linear\" | \"affine\"\n"
            "  mode      : \"global\" | \"local\"\n"
            "  grad_mode : \"hard\" | \"soft\" | \"none\"\n"
            "  kernel    : \"scalar\" | \"simd\" — bit-exact alternatives; a speed\n"
            "              knob only.  Viterbi/hard-gradient path only; linear is\n"
            "              a no-op (see BatchAligner).")
        .def("alloc_dp", &SeqPair::alloc_dp,
             "Pre-allocate own DP tables for the fixed sequences.")
        .def(
            "set_params",
            [](nb::object self_obj, nb::object params_obj) {
                SeqPair& self = nb::cast<SeqPair&>(self_obj);
                // Validates the alphabet; throws before we take a reference.
                self.set_params(nb::cast<const AlignParams&>(params_obj));
                keep_current_params(self_obj, params_obj);
            },
            nb::arg("params"),
            "Swap alignment parameters.  Invalidates cached score and gradient.")
        .def("align_full",     &SeqPair::align_full,
             "Full DP alignment.  Updates score and alignment path; clears gradient cache.")
        .def("realign_banded", &SeqPair::realign_banded, nb::arg("bandwidth"),
             "Banded DP centred on the current alignment path.")
        .def("compute_grad",   &SeqPair::compute_grad,
             "Compute and cache the gradient from the current alignment.")
        .def("score_and_grad", &SeqPair::score_and_grad,
             "Convenience: allocate DP, align, and compute the gradient in one call.\n"
             "Returns (score, AlignParams grad).  Raises if grad_mode is \"none\".")
        .def("drop_dp", &SeqPair::drop_dp,
             "Free DP table memory.  Cached score, gradient, and guide_j remain valid.")
        .def("aligned", &SeqPair::aligned,
             "Return the alignment as a pair of gapped strings (seq_a, seq_b).\n"
             "Requires the DP tables (call align_full() first, before any drop_dp()).")
        .def(
            "formatted",
            [](const SeqPair& self, int width) {
                auto [a, b] = self.aligned();
                return format_alignment(a, b, width);
            },
            nb::arg("width") = 60,
            "Pretty-printed alignment block (seq_a / match line / seq_b),\n"
            "wrapped at `width` columns (0 = no wrapping).")
        .def_prop_ro(
            "score",
            [](const SeqPair& self) -> nb::object {
                if (!self.score_valid()) return nb::none();
                return nb::cast(self.score());
            },
            "Alignment score, or None if not yet computed.")
        .def_prop_ro(
            "grad",
            [](const SeqPair& self) -> nb::object {
                if (!self.grad_valid()) return nb::none();
                return nb::cast(self.grad());
            },
            nb::rv_policy::copy,
            "Gradient as an AlignParams object, or None if not computed.")
        .def_prop_ro(
            "guide_j",
            [](const SeqPair& self) -> nb::object {
                if (!self.path_valid()) return nb::none();
                return nb::cast(std::vector<int>(self.guide_j()));
            },
            "Current alignment as a guide_j vector (length m+1), or None.")
        .def_prop_ro("path_valid",  &SeqPair::path_valid)
        .def_prop_ro("score_valid", &SeqPair::score_valid)
        .def_prop_ro("grad_valid",  &SeqPair::grad_valid)
        .def_prop_ro("dp_valid",    &SeqPair::dp_valid)
        .def_prop_ro("seq_a", [](const SeqPair& s) { return s.seq_a(); })
        .def_prop_ro("seq_b", [](const SeqPair& s) { return s.seq_b(); });

    // ── SeqPairBatch ─────────────────────────────────────────────────────────
    nb::class_<SeqPairBatch>(m, "SeqPairBatch", nb::dynamic_attr())
        .def(
            "__init__",
            [](SeqPairBatch* self, int n_threads) {
                new (self) SeqPairBatch(n_threads);
            },
            nb::arg("n_threads") = 0,
            "Threaded batch of SeqPair objects.\n"
            "  n_threads=0 (default) uses hardware_concurrency.")
        .def(
            "add",
            [](nb::object self_obj, nb::object sp_obj) {
                SeqPairBatch& self = nb::cast<SeqPairBatch&>(self_obj);
                // Validates the alphabet against the batch's first pair; throws
                // before we take a reference to anything.
                self.add(nb::cast<SeqPair*>(sp_obj));

                // The batch holds SeqPairs by raw pointer, so each one must be
                // kept alive for as long as the batch is.  nb::keep_alive<1,2>
                // does that, but nanobind keeps a nurse's patients in a singly
                // linked list that it walks on every call to deduplicate — so
                // the k-th add() walks k nodes and N adds cost O(N^2).  At
                // Manakov's 2.5M pairs that is hours.  A Python list append is
                // O(1) and keeps the same reference for the same lifetime.
                nb::list refs;
                if (nb::hasattr(self_obj, "_keepalive"))
                    refs = nb::borrow<nb::list>(self_obj.attr("_keepalive"));
                else
                    self_obj.attr("_keepalive") = refs;
                refs.append(sp_obj);
            },
            nb::arg("seq_pair"),
            "Append a SeqPair to the batch.")
        .def(
            "add_many",
            [](nb::object self_obj,
               const std::vector<std::string_view>& seqs_a,
               const std::vector<std::string_view>& seqs_b,
               nb::object params_obj,
               const std::string& gap_model,
               const std::string& mode,
               const std::string& grad_mode,
               const std::string& kernel) {
                SeqPairBatch& self = nb::cast<SeqPairBatch&>(self_obj);
                const AlignParams& params = nb::cast<const AlignParams&>(params_obj);
                GapModel  gm = (gap_model == "affine") ? GapModel::Affine : GapModel::Linear;
                AlignMode am = (mode      == "local")  ? AlignMode::Local : AlignMode::Global;
                GradMode  gd;
                if      (grad_mode == "hard") gd = GradMode::Hard;
                else if (grad_mode == "soft") gd = GradMode::Soft;
                else                          gd = GradMode::None;
                DpKernel  kn = parse_kernel(kernel);

                // The string_views point into the argument lists' str objects.
                // We hold the GIL throughout — add_many()'s workers touch no
                // Python — so nothing can free or move them mid-call.
                self.add_many(seqs_a, seqs_b, params, gm, am, gd, kn);

                // These pairs are owned by C++ and hold a bare pointer to
                // `params`, and unlike a hand-built SeqPair none of them holds a
                // Python reference of its own.  keep_current_params() alone is
                // not enough: a second add_many() with different params would
                // overwrite the single `_params` slot and free the first, leaving
                // the first call's pairs pointing at a corpse.  So append to a
                // list — one entry per add_many() call, not per step of a
                // training loop, so this does not grow without bound.
                nb::list refs;
                if (nb::hasattr(self_obj, "_owned_params"))
                    refs = nb::borrow<nb::list>(self_obj.attr("_owned_params"));
                else
                    self_obj.attr("_owned_params") = refs;
                refs.append(params_obj);
            },
            nb::arg("seqs_a"), nb::arg("seqs_b"), nb::arg("params"),
            nb::arg("gap_model") = "affine",
            nb::arg("mode")      = "global",
            nb::arg("grad_mode") = "hard",
            nb::arg("kernel")    = "scalar",
            "Bulk-construct N SeqPairs in C++ and append them to the batch.\n"
            "\n"
            "Equivalent to constructing each SeqPair in Python and calling add()\n"
            "on it, but the pairs never become Python objects: for large N this\n"
            "is the difference between seconds and tens of seconds, and it drops\n"
            "the per-pair memory to what C++ actually needs.  Per-pair results\n"
            "remain fully available through batch[i].score / .grad / .aligned().\n"
            "\n"
            "  gap_model : \"linear\" | \"affine\"\n"
            "  mode      : \"global\" | \"local\"\n"
            "  grad_mode : \"hard\" | \"soft\" | \"none\"\n"
            "\n"
            "Raises if the two lists differ in length, if a sequence contains a\n"
            "character outside the params' alphabet, or if that alphabet differs\n"
            "from the batch's.  In every such case the batch is left unchanged.")
        .def("__len__", &SeqPairBatch::size)
        .def(
            "__getitem__",
            [](SeqPairBatch& self, int i) -> SeqPair& {
                if (i < 0) i += static_cast<int>(self.size());
                if (i < 0 || static_cast<size_t>(i) >= self.size())
                    throw nb::index_error("SeqPairBatch index out of range");
                return self[static_cast<size_t>(i)];
            },
            nb::rv_policy::reference,
            nb::arg("i"))
        .def("alloc_dp", [](SeqPairBatch& self) { self.alloc_dp(); },
             "Pre-allocate own DP tables on all pairs in parallel.")
        .def(
            "set_params",
            [](nb::object self_obj, nb::object params_obj) {
                SeqPairBatch& self = nb::cast<SeqPairBatch&>(self_obj);
                // Sets params on every pair in C++, so the pairs do not each
                // take their own Python reference — the batch holds the one that
                // keeps `params` alive on their behalf.
                self.set_params(nb::cast<const AlignParams&>(params_obj));
                keep_current_params(self_obj, params_obj);
            },
            nb::arg("params"),
            "Set alignment parameters on all pairs (clears score and grad caches).")
        .def(
            "align_full",
            [](SeqPairBatch& self) { return self.align_full(); },
            "Full DP alignment of all pairs in parallel.  Returns sum of scores.")
        .def(
            "realign_banded",
            [](SeqPairBatch& self, int bandwidth) { return self.realign_banded(bandwidth); },
            nb::arg("bandwidth"),
            "Banded realignment of all pairs in parallel.  Returns sum of scores.")
        .def(
            "compute_grad",
            [](SeqPairBatch& self) { return self.compute_grad(); },
            nb::rv_policy::move,
            "Compute gradient on all pairs in parallel.\n"
            "Returns summed gradient as an AlignParams object.")
        .def(
            "score_and_grad",
            [](SeqPairBatch& self, int bandwidth) { return self.score_and_grad(bandwidth); },
            nb::arg("bandwidth") = 0,
            "Full-pipeline batch operation using per-thread DP buffers.\n"
            "Returns sum of scores.")
        .def(
            "drop_dp",
            [](SeqPairBatch& self) { self.drop_dp(); },
            "Drop DP tables on all pairs in parallel.")
        .def_prop_ro("n_threads", [](const SeqPairBatch& s) { return s.n_threads; });

}
