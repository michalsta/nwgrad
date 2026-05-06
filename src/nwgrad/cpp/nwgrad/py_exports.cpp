#include <string>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
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

#define WITH_ALIGNER(GM, AM, band, guide_j, body)                                        \
    do {                                                                                  \
        DpBuffer _buf;                                                                    \
        if ((band) > 0 || !(guide_j).empty()) {                                          \
            Aligner<GapModel::GM, AlignMode::AM, AlignBand::GuideBanded> al;             \
            body                                                                           \
        } else {                                                                           \
            Aligner<GapModel::GM, AlignMode::AM, AlignBand::Full> al;                    \
            body                                                                           \
        }                                                                                  \
    } while (0)

static std::vector<int> make_guide(const std::string& aligned_a,
                                    const std::string& aligned_b) {
    if (!aligned_a.empty() && !aligned_b.empty())
        return guide_j_from_aligned(aligned_a, aligned_b);
    return {};
}

NB_MODULE(nwgrad_ext, m) {
    m.doc() = "nwgrad C++ nanobind module";

    // ── SubstMatrix ──────────────────────────────────────────────────────────
    nb::class_<SubstMatrix>(m, "SubstMatrix")
        .def(
            "__init__",
            [](SubstMatrix* self, nb_arr_f64 arr, const std::string& alphabet) {
                if (arr.shape(0) != arr.shape(1))
                    throw std::invalid_argument("matrix must be square");
                if (static_cast<size_t>(alphabet.size()) != arr.shape(0))
                    throw std::invalid_argument("alphabet length must match matrix size");
                new (self) SubstMatrix(arr.data(), alphabet);
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
                new (self) AlignParams();
                self->matrix       = SubstMatrix(arr.data(), alphabet);
                self->gap_open_a   = gap_open_a;
                self->gap_extend_a = gap_extend_a;
                self->gap_open_b   = gap_open_b;
                self->gap_extend_b = gap_extend_b;
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
            "All gap values default to 0.0 (usable as a zero gradient accumulator).")
        .def_prop_rw(
            "matrix",
            [](const AlignParams& self) { return self.matrix; },
            [](AlignParams& self, const SubstMatrix& mat) { self.matrix = mat; })
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
           const std::string& aligned_a, const std::string& aligned_b) {
            auto gj = make_guide(aligned_a, aligned_b);
            WITH_ALIGNER(Linear, Global, band, gj, {
                al.set_problem(a, b, params, band, gj);
                al.compute_viterbi(_buf);
                return al.score();
            });
        },
        nb::arg("seq_a"), nb::arg("seq_b"), nb::arg("params"),
        nb::arg("band") = 0, nb::arg("aligned_a") = "", nb::arg("aligned_b") = "",
        "Needleman-Wunsch global alignment score (linear gap penalty).");

    m.def(
        "sw_score",
        [](const std::string& a, const std::string& b,
           const AlignParams& params, int band,
           const std::string& aligned_a, const std::string& aligned_b) {
            auto gj = make_guide(aligned_a, aligned_b);
            WITH_ALIGNER(Linear, Local, band, gj, {
                al.set_problem(a, b, params, band, gj);
                al.compute_viterbi(_buf);
                return al.score();
            });
        },
        nb::arg("seq_a"), nb::arg("seq_b"), nb::arg("params"),
        nb::arg("band") = 0, nb::arg("aligned_a") = "", nb::arg("aligned_b") = "",
        "Smith-Waterman local alignment score (linear gap penalty).");

    m.def(
        "nw_score_affine",
        [](const std::string& a, const std::string& b,
           const AlignParams& params, int band,
           const std::string& aligned_a, const std::string& aligned_b) {
            auto gj = make_guide(aligned_a, aligned_b);
            WITH_ALIGNER(Affine, Global, band, gj, {
                al.set_problem(a, b, params, band, gj);
                al.compute_viterbi(_buf);
                return al.score();
            });
        },
        nb::arg("seq_a"), nb::arg("seq_b"), nb::arg("params"),
        nb::arg("band") = 0, nb::arg("aligned_a") = "", nb::arg("aligned_b") = "",
        "Needleman-Wunsch global alignment score (affine gap penalty).");

    m.def(
        "sw_score_affine",
        [](const std::string& a, const std::string& b,
           const AlignParams& params, int band,
           const std::string& aligned_a, const std::string& aligned_b) {
            auto gj = make_guide(aligned_a, aligned_b);
            WITH_ALIGNER(Affine, Local, band, gj, {
                al.set_problem(a, b, params, band, gj);
                al.compute_viterbi(_buf);
                return al.score();
            });
        },
        nb::arg("seq_a"), nb::arg("seq_b"), nb::arg("params"),
        nb::arg("band") = 0, nb::arg("aligned_a") = "", nb::arg("aligned_b") = "",
        "Smith-Waterman local alignment score (affine gap penalty).");

    // ── Single-pair hard gradient functions ─────────────────────────────────

    m.def(
        "nw_grad",
        [](const std::string& a, const std::string& b,
           const AlignParams& params, int band,
           const std::string& aligned_a, const std::string& aligned_b) {
            auto gj = make_guide(aligned_a, aligned_b);
            WITH_ALIGNER(Linear, Global, band, gj, {
                al.set_problem(a, b, params, band, gj);
                al.compute_viterbi(_buf);
                AlignParams grad;
                grad.matrix.order_ = params.matrix.order_;
                al.hard_grad(_buf, grad);
                return nb::make_tuple(al.score(), grad);
            });
        },
        nb::arg("seq_a"), nb::arg("seq_b"), nb::arg("params"),
        nb::arg("band") = 0, nb::arg("aligned_a") = "", nb::arg("aligned_b") = "",
        "NW global: returns (score, AlignParams grad) — hard subgradient (linear gap).");

    m.def(
        "sw_grad",
        [](const std::string& a, const std::string& b,
           const AlignParams& params, int band,
           const std::string& aligned_a, const std::string& aligned_b) {
            auto gj = make_guide(aligned_a, aligned_b);
            WITH_ALIGNER(Linear, Local, band, gj, {
                al.set_problem(a, b, params, band, gj);
                al.compute_viterbi(_buf);
                AlignParams grad;
                grad.matrix.order_ = params.matrix.order_;
                al.hard_grad(_buf, grad);
                return nb::make_tuple(al.score(), grad);
            });
        },
        nb::arg("seq_a"), nb::arg("seq_b"), nb::arg("params"),
        nb::arg("band") = 0, nb::arg("aligned_a") = "", nb::arg("aligned_b") = "",
        "SW local: returns (score, AlignParams grad) — hard subgradient (linear gap).");

    m.def(
        "nw_affine_grad",
        [](const std::string& a, const std::string& b,
           const AlignParams& params, int band,
           const std::string& aligned_a, const std::string& aligned_b) {
            auto gj = make_guide(aligned_a, aligned_b);
            WITH_ALIGNER(Affine, Global, band, gj, {
                al.set_problem(a, b, params, band, gj);
                al.compute_viterbi(_buf);
                AlignParams grad;
                grad.matrix.order_ = params.matrix.order_;
                al.hard_grad(_buf, grad);
                return nb::make_tuple(al.score(), grad);
            });
        },
        nb::arg("seq_a"), nb::arg("seq_b"), nb::arg("params"),
        nb::arg("band") = 0, nb::arg("aligned_a") = "", nb::arg("aligned_b") = "",
        "NW global: returns (score, AlignParams grad) — hard subgradient (affine gap).");

    m.def(
        "sw_affine_grad",
        [](const std::string& a, const std::string& b,
           const AlignParams& params, int band,
           const std::string& aligned_a, const std::string& aligned_b) {
            auto gj = make_guide(aligned_a, aligned_b);
            WITH_ALIGNER(Affine, Local, band, gj, {
                al.set_problem(a, b, params, band, gj);
                al.compute_viterbi(_buf);
                AlignParams grad;
                grad.matrix.order_ = params.matrix.order_;
                al.hard_grad(_buf, grad);
                return nb::make_tuple(al.score(), grad);
            });
        },
        nb::arg("seq_a"), nb::arg("seq_b"), nb::arg("params"),
        nb::arg("band") = 0, nb::arg("aligned_a") = "", nb::arg("aligned_b") = "",
        "SW local: returns (score, AlignParams grad) — hard subgradient (affine gap).");

    // ── Single-pair soft gradient functions ─────────────────────────────────

    m.def(
        "nw_soft_grad",
        [](const std::string& a, const std::string& b,
           const AlignParams& params, int band,
           const std::string& aligned_a, const std::string& aligned_b) {
            auto gj = make_guide(aligned_a, aligned_b);
            WITH_ALIGNER(Linear, Global, band, gj, {
                al.set_problem(a, b, params, band, gj);
                al.compute_forward_back(_buf);
                AlignParams grad;
                grad.matrix.order_ = params.matrix.order_;
                al.soft_grad(_buf, grad);
                return nb::make_tuple(al.log_z(), grad);
            });
        },
        nb::arg("seq_a"), nb::arg("seq_b"), nb::arg("params"),
        nb::arg("band") = 0, nb::arg("aligned_a") = "", nb::arg("aligned_b") = "",
        "NW global: returns (log_Z, AlignParams grad) — soft gradient (linear gap).");

    m.def(
        "sw_soft_grad",
        [](const std::string& a, const std::string& b,
           const AlignParams& params, int band,
           const std::string& aligned_a, const std::string& aligned_b) {
            auto gj = make_guide(aligned_a, aligned_b);
            WITH_ALIGNER(Linear, Local, band, gj, {
                al.set_problem(a, b, params, band, gj);
                al.compute_forward_back(_buf);
                AlignParams grad;
                grad.matrix.order_ = params.matrix.order_;
                al.soft_grad(_buf, grad);
                return nb::make_tuple(al.log_z(), grad);
            });
        },
        nb::arg("seq_a"), nb::arg("seq_b"), nb::arg("params"),
        nb::arg("band") = 0, nb::arg("aligned_a") = "", nb::arg("aligned_b") = "",
        "SW local: returns (log_Z, AlignParams grad) — soft gradient (linear gap).");

    m.def(
        "nw_affine_soft_grad",
        [](const std::string& a, const std::string& b,
           const AlignParams& params, int band,
           const std::string& aligned_a, const std::string& aligned_b) {
            auto gj = make_guide(aligned_a, aligned_b);
            WITH_ALIGNER(Affine, Global, band, gj, {
                al.set_problem(a, b, params, band, gj);
                al.compute_forward_back(_buf);
                AlignParams grad;
                grad.matrix.order_ = params.matrix.order_;
                al.soft_grad(_buf, grad);
                return nb::make_tuple(al.log_z(), grad);
            });
        },
        nb::arg("seq_a"), nb::arg("seq_b"), nb::arg("params"),
        nb::arg("band") = 0, nb::arg("aligned_a") = "", nb::arg("aligned_b") = "",
        "NW global: returns (log_Z, AlignParams grad) — soft gradient (affine gap).");

    m.def(
        "sw_affine_soft_grad",
        [](const std::string& a, const std::string& b,
           const AlignParams& params, int band,
           const std::string& aligned_a, const std::string& aligned_b) {
            auto gj = make_guide(aligned_a, aligned_b);
            WITH_ALIGNER(Affine, Local, band, gj, {
                al.set_problem(a, b, params, band, gj);
                al.compute_forward_back(_buf);
                AlignParams grad;
                grad.matrix.order_ = params.matrix.order_;
                al.soft_grad(_buf, grad);
                return nb::make_tuple(al.log_z(), grad);
            });
        },
        nb::arg("seq_a"), nb::arg("seq_b"), nb::arg("params"),
        nb::arg("band") = 0, nb::arg("aligned_a") = "", nb::arg("aligned_b") = "",
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
               int n_threads) {
                GapModel  gm = (gap_model == "affine") ? GapModel::Affine  : GapModel::Linear;
                AlignMode am = (mode      == "local")  ? AlignMode::Local  : AlignMode::Global;
                BatchAligner::GradMode gd;
                if      (grad_mode == "hard") gd = BatchAligner::GradMode::Hard;
                else if (grad_mode == "soft") gd = BatchAligner::GradMode::Soft;
                else                          gd = BatchAligner::GradMode::None;
                new (self) BatchAligner(params, band, gm, am, gd, n_threads);
            },
            nb::arg("params"),
            nb::arg("band")       = 0,
            nb::arg("gap_model")  = "affine",
            nb::arg("mode")       = "global",
            nb::arg("grad_mode")  = "hard",
            nb::arg("n_threads")  = 1,
            "Create a BatchAligner.\n"
            "  gap_model : \"linear\" | \"affine\"\n"
            "  mode      : \"global\" | \"local\"\n"
            "  grad_mode : \"hard\" | \"soft\" | \"none\"\n"
            "  band      : 0 = full DP; >0 = banded half-width")
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
    nb::class_<SeqPair>(m, "SeqPair")
        .def(
            "__init__",
            [](SeqPair* self,
               const std::string& seq_a, const std::string& seq_b,
               const AlignParams& params,
               const std::string& gap_model,
               const std::string& mode,
               const std::string& grad_mode) {
                GapModel  gm = (gap_model == "affine") ? GapModel::Affine  : GapModel::Linear;
                AlignMode am = (mode      == "local")  ? AlignMode::Local  : AlignMode::Global;
                GradMode  gd;
                if      (grad_mode == "hard") gd = GradMode::Hard;
                else if (grad_mode == "soft") gd = GradMode::Soft;
                else                          gd = GradMode::None;
                new (self) SeqPair(seq_a, seq_b, params, gm, am, gd);
            },
            nb::arg("seq_a"), nb::arg("seq_b"), nb::arg("params"),
            nb::arg("gap_model") = "affine", nb::arg("mode") = "global",
            nb::arg("grad_mode") = "hard",
            "Persistent sequence pair.\n"
            "  gap_model : \"linear\" | \"affine\"\n"
            "  mode      : \"global\" | \"local\"\n"
            "  grad_mode : \"hard\" | \"soft\" | \"none\"")
        .def("alloc_dp", &SeqPair::alloc_dp,
             "Pre-allocate own DP tables for the fixed sequences.")
        .def(
            "set_params",
            [](SeqPair& self, const AlignParams& params) { self.set_params(params); },
            nb::arg("params"),
            "Swap alignment parameters.  Invalidates cached score and gradient.\n"
            "The params object must remain alive as long as this SeqPair uses it.")
        .def("align_full",     &SeqPair::align_full,
             "Full DP alignment.  Updates score and alignment path; clears gradient cache.")
        .def("realign_banded", &SeqPair::realign_banded, nb::arg("bandwidth"),
             "Banded DP centred on the current alignment path.")
        .def("compute_grad",   &SeqPair::compute_grad,
             "Compute and cache the gradient from the current alignment.")
        .def("drop_dp", &SeqPair::drop_dp,
             "Free DP table memory.  Cached score, gradient, and guide_j remain valid.")
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
    nb::class_<SeqPairBatch>(m, "SeqPairBatch")
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
            [](SeqPairBatch& self, SeqPair* sp) { self.add(sp); },
            nb::arg("seq_pair"),
            nb::keep_alive<1, 2>(),
            "Append a SeqPair to the batch.")
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
            [](SeqPairBatch& self, const AlignParams& params) { self.set_params(params); },
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
