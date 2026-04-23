#include <string>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "subst_matrix.hpp"
#include "aligner.hpp"
#include "batch.hpp"
#include "seq_pair.hpp"
#include "seq_pair_batch.hpp"

namespace nb = nanobind;
using nb_arr_f64    = nb::ndarray<double, nb::shape<20, 20>, nb::c_contig, nb::device::cpu>;
using nb_arr_f64_1d = nb::ndarray<nb::numpy, double, nb::ndim<1>>;
using nb_arr_f64_2d = nb::ndarray<nb::numpy, double, nb::shape<20, 20>>;

static nb_arr_f64_2d grad_to_numpy(const double g[256][256]) {
    double* buf = new double[400];
    for (int i = 0; i < 20; ++i)
        for (int j = 0; j < 20; ++j)
            buf[i * 20 + j] =
                g[static_cast<unsigned char>(AA_ORDER[i])]
                 [static_cast<unsigned char>(AA_ORDER[j])];
    nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<double*>(p); });
    return nb_arr_f64_2d(buf, {20, 20}, owner);
}

// ── Aligner factory: picks Full or GuideBanded at runtime. ───────────────────
// guide_j may be empty (→ trivial diagonal guide) or provided by the caller.

#define WITH_ALIGNER(GM, AM, band, guide_j, body)                                         \
    do {                                                                                   \
        DpBuffer _buf;                                                                     \
        if ((band) > 0 || !(guide_j).empty()) {                                           \
            Aligner<GapModel::GM, AlignMode::AM, AlignBand::GuideBanded> al;              \
            body                                                                            \
        } else {                                                                            \
            Aligner<GapModel::GM, AlignMode::AM, AlignBand::Full> al;                     \
            body                                                                            \
        }                                                                                   \
    } while (0)

// Helper: compute guide_j from optional aligned strings.
static std::vector<int> make_guide(const std::string& aligned_a,
                                    const std::string& aligned_b) {
    if (!aligned_a.empty() && !aligned_b.empty())
        return guide_j_from_aligned(aligned_a, aligned_b);
    return {};
}

NB_MODULE(nwgrad_ext, m) {
    m.doc() = "nwgrad C++ nanobind module";

    // ── SubstMatrix ─────────────────────────────────────────────────────────
    nb::class_<SubstMatrix>(m, "SubstMatrix")
        .def(
            "__init__",
            [](SubstMatrix* self, nb_arr_f64 arr) {
                new (self) SubstMatrix(arr.data());
            },
            nb::arg("matrix"),
            "Construct from a (20, 20) float64 numpy array in canonical AA order "
            "(ACDEFGHIKLMNPQRSTVWY).")
        .def(
            "score",
            [](const SubstMatrix& self, const std::string& a, const std::string& b) {
                if (a.size() != 1 || b.size() != 1)
                    throw std::invalid_argument("score() expects single-character strings");
                return self.score(a[0], b[0]);
            },
            nb::arg("a"), nb::arg("b"),
            "Return substitution score for amino acids a and b.")
        .def(
            "to_matrix",
            [](const SubstMatrix& self) {
                double* buf = new double[400];
                self.to_array(buf);
                nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<double*>(p); });
                return nb::ndarray<nb::numpy, double, nb::shape<20, 20>>(buf, {20, 20}, owner);
            },
            "Export as a (20, 20) float64 numpy array in canonical AA order.");

    // ── Guide alignment utility ───────────────────────────────────────────────
    m.def(
        "guide_j_from_aligned",
        [](const std::string& a_aligned, const std::string& b_aligned) {
            return guide_j_from_aligned(a_aligned, b_aligned);
        },
        nb::arg("a_aligned"), nb::arg("b_aligned"),
        "Convert a pair of aligned strings (using '-' for gaps) to a guide_j vector.\n"
        "Returns a list of length m+1 where guide_j[i] = column j after consuming i\n"
        "characters of a.  Pass to single-pair functions as aligned_a/aligned_b or\n"
        "to BatchAligner.align as aligned_a/aligned_b lists.");

    // ── Single-pair score functions ───────────────────────────────────────────

    m.def(
        "nw_score",
        [](const std::string& a, const std::string& b,
           const SubstMatrix& mat, double gap_extend, int band,
           const std::string& aligned_a, const std::string& aligned_b) {
            auto gj = make_guide(aligned_a, aligned_b);
            WITH_ALIGNER(Linear, Global, band, gj, {
                al.set_problem(a, b, mat, gap_extend, 0.0, band, gj);
                al.compute_viterbi(_buf);
                return al.score();
            });
        },
        nb::arg("seq_a"), nb::arg("seq_b"), nb::arg("matrix"), nb::arg("gap_extend"),
        nb::arg("band") = 0, nb::arg("aligned_a") = "", nb::arg("aligned_b") = "",
        "Needleman-Wunsch global alignment score (linear gap penalty).\n"
        "band=0 and no aligned strings → full DP.\n"
        "band>0 → diagonal banded DP.  Supply aligned_a/aligned_b for guide-banded DP.");

    m.def(
        "sw_score",
        [](const std::string& a, const std::string& b,
           const SubstMatrix& mat, double gap_extend, int band,
           const std::string& aligned_a, const std::string& aligned_b) {
            auto gj = make_guide(aligned_a, aligned_b);
            WITH_ALIGNER(Linear, Local, band, gj, {
                al.set_problem(a, b, mat, gap_extend, 0.0, band, gj);
                al.compute_viterbi(_buf);
                return al.score();
            });
        },
        nb::arg("seq_a"), nb::arg("seq_b"), nb::arg("matrix"), nb::arg("gap_extend"),
        nb::arg("band") = 0, nb::arg("aligned_a") = "", nb::arg("aligned_b") = "",
        "Smith-Waterman local alignment score (linear gap penalty).");

    m.def(
        "nw_score_affine",
        [](const std::string& a, const std::string& b,
           const SubstMatrix& mat, double gap_open, double gap_extend, int band,
           const std::string& aligned_a, const std::string& aligned_b) {
            auto gj = make_guide(aligned_a, aligned_b);
            WITH_ALIGNER(Affine, Global, band, gj, {
                al.set_problem(a, b, mat, gap_extend, gap_open, band, gj);
                al.compute_viterbi(_buf);
                return al.score();
            });
        },
        nb::arg("seq_a"), nb::arg("seq_b"), nb::arg("matrix"),
        nb::arg("gap_open"), nb::arg("gap_extend"), nb::arg("band") = 0,
        nb::arg("aligned_a") = "", nb::arg("aligned_b") = "",
        "Needleman-Wunsch global alignment score (affine gap penalty).");

    m.def(
        "sw_score_affine",
        [](const std::string& a, const std::string& b,
           const SubstMatrix& mat, double gap_open, double gap_extend, int band,
           const std::string& aligned_a, const std::string& aligned_b) {
            auto gj = make_guide(aligned_a, aligned_b);
            WITH_ALIGNER(Affine, Local, band, gj, {
                al.set_problem(a, b, mat, gap_extend, gap_open, band, gj);
                al.compute_viterbi(_buf);
                return al.score();
            });
        },
        nb::arg("seq_a"), nb::arg("seq_b"), nb::arg("matrix"),
        nb::arg("gap_open"), nb::arg("gap_extend"), nb::arg("band") = 0,
        nb::arg("aligned_a") = "", nb::arg("aligned_b") = "",
        "Smith-Waterman local alignment score (affine gap penalty).");

    // ── Single-pair hard gradient functions ───────────────────────────────────

    m.def(
        "nw_grad",
        [](const std::string& a, const std::string& b,
           const SubstMatrix& mat, double gap_extend, int band,
           const std::string& aligned_a, const std::string& aligned_b) {
            auto gj = make_guide(aligned_a, aligned_b);
            WITH_ALIGNER(Linear, Global, band, gj, {
                al.set_problem(a, b, mat, gap_extend, 0.0, band, gj);
                al.compute_viterbi(_buf);
                double grad[256][256]{};
                al.hard_grad(_buf, grad);
                return nb::make_tuple(al.score(), grad_to_numpy(grad));
            });
        },
        nb::arg("seq_a"), nb::arg("seq_b"), nb::arg("matrix"), nb::arg("gap_extend"),
        nb::arg("band") = 0, nb::arg("aligned_a") = "", nb::arg("aligned_b") = "",
        "NW global alignment: returns (score, grad[20,20]) — hard subgradient (linear gap).");

    m.def(
        "sw_grad",
        [](const std::string& a, const std::string& b,
           const SubstMatrix& mat, double gap_extend, int band,
           const std::string& aligned_a, const std::string& aligned_b) {
            auto gj = make_guide(aligned_a, aligned_b);
            WITH_ALIGNER(Linear, Local, band, gj, {
                al.set_problem(a, b, mat, gap_extend, 0.0, band, gj);
                al.compute_viterbi(_buf);
                double grad[256][256]{};
                al.hard_grad(_buf, grad);
                return nb::make_tuple(al.score(), grad_to_numpy(grad));
            });
        },
        nb::arg("seq_a"), nb::arg("seq_b"), nb::arg("matrix"), nb::arg("gap_extend"),
        nb::arg("band") = 0, nb::arg("aligned_a") = "", nb::arg("aligned_b") = "",
        "SW local alignment: returns (score, grad[20,20]) — hard subgradient (linear gap).");

    m.def(
        "nw_affine_grad",
        [](const std::string& a, const std::string& b,
           const SubstMatrix& mat, double gap_open, double gap_extend, int band,
           const std::string& aligned_a, const std::string& aligned_b) {
            auto gj = make_guide(aligned_a, aligned_b);
            WITH_ALIGNER(Affine, Global, band, gj, {
                al.set_problem(a, b, mat, gap_extend, gap_open, band, gj);
                al.compute_viterbi(_buf);
                double grad[256][256]{};
                al.hard_grad(_buf, grad);
                return nb::make_tuple(al.score(), grad_to_numpy(grad));
            });
        },
        nb::arg("seq_a"), nb::arg("seq_b"), nb::arg("matrix"),
        nb::arg("gap_open"), nb::arg("gap_extend"), nb::arg("band") = 0,
        nb::arg("aligned_a") = "", nb::arg("aligned_b") = "",
        "NW global alignment: returns (score, grad[20,20]) — hard subgradient (affine gap).");

    m.def(
        "sw_affine_grad",
        [](const std::string& a, const std::string& b,
           const SubstMatrix& mat, double gap_open, double gap_extend, int band,
           const std::string& aligned_a, const std::string& aligned_b) {
            auto gj = make_guide(aligned_a, aligned_b);
            WITH_ALIGNER(Affine, Local, band, gj, {
                al.set_problem(a, b, mat, gap_extend, gap_open, band, gj);
                al.compute_viterbi(_buf);
                double grad[256][256]{};
                al.hard_grad(_buf, grad);
                return nb::make_tuple(al.score(), grad_to_numpy(grad));
            });
        },
        nb::arg("seq_a"), nb::arg("seq_b"), nb::arg("matrix"),
        nb::arg("gap_open"), nb::arg("gap_extend"), nb::arg("band") = 0,
        nb::arg("aligned_a") = "", nb::arg("aligned_b") = "",
        "SW local alignment: returns (score, grad[20,20]) — hard subgradient (affine gap).");

    // ── Single-pair soft gradient functions ───────────────────────────────────

    m.def(
        "nw_soft_grad",
        [](const std::string& a, const std::string& b,
           const SubstMatrix& mat, double gap_extend, int band,
           const std::string& aligned_a, const std::string& aligned_b) {
            auto gj = make_guide(aligned_a, aligned_b);
            WITH_ALIGNER(Linear, Global, band, gj, {
                al.set_problem(a, b, mat, gap_extend, 0.0, band, gj);
                al.compute_forward_back(_buf);
                double grad[256][256]{};
                al.soft_grad(_buf, grad);
                return nb::make_tuple(al.log_z(), grad_to_numpy(grad));
            });
        },
        nb::arg("seq_a"), nb::arg("seq_b"), nb::arg("matrix"), nb::arg("gap_extend"),
        nb::arg("band") = 0, nb::arg("aligned_a") = "", nb::arg("aligned_b") = "",
        "NW global: returns (log_Z, expected_counts[20,20]) — soft gradient (linear gap).");

    m.def(
        "sw_soft_grad",
        [](const std::string& a, const std::string& b,
           const SubstMatrix& mat, double gap_extend, int band,
           const std::string& aligned_a, const std::string& aligned_b) {
            auto gj = make_guide(aligned_a, aligned_b);
            WITH_ALIGNER(Linear, Local, band, gj, {
                al.set_problem(a, b, mat, gap_extend, 0.0, band, gj);
                al.compute_forward_back(_buf);
                double grad[256][256]{};
                al.soft_grad(_buf, grad);
                return nb::make_tuple(al.log_z(), grad_to_numpy(grad));
            });
        },
        nb::arg("seq_a"), nb::arg("seq_b"), nb::arg("matrix"), nb::arg("gap_extend"),
        nb::arg("band") = 0, nb::arg("aligned_a") = "", nb::arg("aligned_b") = "",
        "SW local: returns (log_Z, expected_counts[20,20]) — soft gradient (linear gap).");

    m.def(
        "nw_affine_soft_grad",
        [](const std::string& a, const std::string& b,
           const SubstMatrix& mat, double gap_open, double gap_extend, int band,
           const std::string& aligned_a, const std::string& aligned_b) {
            auto gj = make_guide(aligned_a, aligned_b);
            WITH_ALIGNER(Affine, Global, band, gj, {
                al.set_problem(a, b, mat, gap_extend, gap_open, band, gj);
                al.compute_forward_back(_buf);
                double grad[256][256]{};
                al.soft_grad(_buf, grad);
                return nb::make_tuple(al.log_z(), grad_to_numpy(grad));
            });
        },
        nb::arg("seq_a"), nb::arg("seq_b"), nb::arg("matrix"),
        nb::arg("gap_open"), nb::arg("gap_extend"), nb::arg("band") = 0,
        nb::arg("aligned_a") = "", nb::arg("aligned_b") = "",
        "NW global: returns (log_Z, expected_counts[20,20]) — soft gradient (affine gap).");

    m.def(
        "sw_affine_soft_grad",
        [](const std::string& a, const std::string& b,
           const SubstMatrix& mat, double gap_open, double gap_extend, int band,
           const std::string& aligned_a, const std::string& aligned_b) {
            auto gj = make_guide(aligned_a, aligned_b);
            WITH_ALIGNER(Affine, Local, band, gj, {
                al.set_problem(a, b, mat, gap_extend, gap_open, band, gj);
                al.compute_forward_back(_buf);
                double grad[256][256]{};
                al.soft_grad(_buf, grad);
                return nb::make_tuple(al.log_z(), grad_to_numpy(grad));
            });
        },
        nb::arg("seq_a"), nb::arg("seq_b"), nb::arg("matrix"),
        nb::arg("gap_open"), nb::arg("gap_extend"), nb::arg("band") = 0,
        nb::arg("aligned_a") = "", nb::arg("aligned_b") = "",
        "SW local: returns (log_Z, expected_counts[20,20]) — soft gradient (affine gap).");

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
            [](const BatchResult& self) { return grad_to_numpy(self.grad); },
            nb::rv_policy::move,
            "Summed gradient over the batch, shape (20, 20).");

    // ── BatchAligner ─────────────────────────────────────────────────────────
    nb::class_<BatchAligner>(m, "BatchAligner")
        .def(
            "__init__",
            [](BatchAligner* self,
               const SubstMatrix& matrix,
               double gap_open,
               double gap_extend,
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
                new (self) BatchAligner(matrix, gap_open, gap_extend, band, gm, am, gd, n_threads);
            },
            nb::arg("matrix"),
            nb::arg("gap_open")   = 11.0,
            nb::arg("gap_extend") = 1.0,
            nb::arg("band")       = 0,
            nb::arg("gap_model")  = "affine",
            nb::arg("mode")       = "global",
            nb::arg("grad_mode")  = "hard",
            nb::arg("n_threads")  = 1,
            "Create a BatchAligner.\n"
            "  gap_model : \"linear\" | \"affine\"\n"
            "  mode      : \"global\" | \"local\"\n"
            "  grad_mode : \"hard\" | \"soft\" | \"none\"\n"
            "  band      : 0 = full DP (default); >0 = banded half-width\n"
            "              (diagonal band when no guide supplied; guide-centered otherwise)")
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
            "Align N sequence pairs.  Returns a BatchResult with .scores and .grad.\n"
            "Supply aligned_a/aligned_b (lists of gapped strings) for guide-banded DP.");

    // ── SeqPair ───────────────────────────────────────────────────────────────
    nb::class_<SeqPair>(m, "SeqPair")
        .def(
            "__init__",
            [](SeqPair* self,
               const std::string& seq_a, const std::string& seq_b,
               const SubstMatrix& mat,
               double gap_open, double gap_extend,
               const std::string& gap_model,
               const std::string& mode,
               const std::string& grad_mode) {
                GapModel  gm = (gap_model == "affine") ? GapModel::Affine  : GapModel::Linear;
                AlignMode am = (mode      == "local")  ? AlignMode::Local  : AlignMode::Global;
                GradMode  gd;
                if      (grad_mode == "hard") gd = GradMode::Hard;
                else if (grad_mode == "soft") gd = GradMode::Soft;
                else                          gd = GradMode::None;
                new (self) SeqPair(seq_a, seq_b, mat, gap_open, gap_extend, gm, am, gd);
            },
            nb::arg("seq_a"), nb::arg("seq_b"), nb::arg("matrix"),
            nb::arg("gap_open") = 11.0, nb::arg("gap_extend") = 1.0,
            nb::arg("gap_model") = "affine", nb::arg("mode") = "global",
            nb::arg("grad_mode") = "hard",
            "Persistent sequence pair.  Call align_full() to compute the initial\n"
            "alignment, then realign_banded(bw) for cheap re-alignment, and\n"
            "compute_grad() to populate the gradient cache.\n"
            "  gap_model : \"linear\" | \"affine\"\n"
            "  mode      : \"global\" | \"local\"\n"
            "  grad_mode : \"hard\" (Viterbi, subgradient)\n"
            "            | \"soft\" (Viterbi + forward-backward, expected counts)\n"
            "            | \"none\" (score only, compute_grad() unavailable)")
        .def("alloc_dp", &SeqPair::alloc_dp,
             "Pre-allocate own DP tables for the fixed sequences.  Must be called\n"
             "before align_full() / realign_banded().  score_and_grad_with_dp() uses\n"
             "caller-supplied buffers and does not require this.")
        .def(
            "set_matrix",
            [](SeqPair& self, const SubstMatrix& mat) { self.set_matrix(mat); },
            nb::arg("matrix"),
            "Swap the substitution matrix.  Invalidates cached score and gradient.\n"
            "The matrix object must remain alive as long as this SeqPair uses it.")
        .def("align_full",    &SeqPair::align_full,
             "Full DP alignment.  Updates score and alignment path; clears gradient cache.")
        .def("realign_banded", &SeqPair::realign_banded, nb::arg("bandwidth"),
             "Banded DP centred on the current alignment path.  Requires align_full()\n"
             "to have been called first.  Clears gradient cache.")
        .def("compute_grad",  &SeqPair::compute_grad,
             "Compute and cache the hard subgradient from the current alignment.")
        .def("drop_dp", &SeqPair::drop_dp,
             "Free DP table memory (O(mn) per pair).  Cached score, gradient, and\n"
             "guide_j remain valid.  compute_grad() will throw until the next\n"
             "align_full() or realign_banded() call.")
        .def_prop_ro(
            "score",
            [](const SeqPair& self) -> nb::object {
                if (!self.score_valid()) return nb::none();
                return nb::cast(self.score());
            },
            "Alignment score under the current matrix, or None if not yet computed.")
        .def_prop_ro(
            "grad",
            [](const SeqPair& self) -> nb::object {
                if (!self.grad_valid()) return nb::none();
                return nb::cast(grad_to_numpy(self.grad()));
            },
            nb::rv_policy::move,
            "Gradient as a (20, 20) float64 numpy array, or None if not computed.")
        .def_prop_ro(
            "guide_j",
            [](const SeqPair& self) -> nb::object {
                if (!self.path_valid()) return nb::none();
                const auto& gj = self.guide_j();
                return nb::cast(std::vector<int>(gj));
            },
            "Current alignment as a guide_j vector (length m+1), or None.")
        .def_prop_ro("path_valid",  &SeqPair::path_valid,
                     "True if guide_j exists (realign_banded() is callable).")
        .def_prop_ro("score_valid", &SeqPair::score_valid,
                     "True if score matches the current matrix and alignment path.")
        .def_prop_ro("grad_valid",  &SeqPair::grad_valid,
                     "True if the gradient cache is valid.")
        .def_prop_ro("dp_valid",    &SeqPair::dp_valid,
                     "True if DP tables are in memory (compute_grad() is callable).")
        .def_prop_ro("seq_a", [](const SeqPair& s) { return s.seq_a(); })
        .def_prop_ro("seq_b", [](const SeqPair& s) { return s.seq_b(); });

    // ── SeqPairBatch ──────────────────────────────────────────────────────────
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
            nb::keep_alive<1, 2>(),   // keep seq_pair alive as long as batch is alive
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
             "Pre-allocate own DP tables on all pairs in parallel.\n"
             "Required before align_full() / realign_banded(); not needed for score_and_grad().")
        .def(
            "set_matrix",
            [](SeqPairBatch& self, const SubstMatrix& mat) {
                self.set_matrix(mat);
            },
            nb::arg("matrix"),
            "Set the substitution matrix on all pairs (clears score and grad caches).")
        .def(
            "align_full",
            [](SeqPairBatch& self) { return self.align_full(); },
            "Full DP alignment of all pairs in parallel.  Returns sum of scores.")
        .def(
            "realign_banded",
            [](SeqPairBatch& self, int bandwidth) {
                return self.realign_banded(bandwidth);
            },
            nb::arg("bandwidth"),
            "Banded realignment of all pairs in parallel.  Returns sum of scores.")
        .def(
            "compute_grad",
            [](SeqPairBatch& self) {
                double grad[256][256]{};
                self.compute_grad(grad);
                return grad_to_numpy(grad);
            },
            nb::rv_policy::move,
            "Compute gradient on all pairs in parallel.\n"
            "Returns summed gradient as a (20, 20) float64 numpy array.")
        .def(
            "score_and_grad",
            [](SeqPairBatch& self, int bandwidth) { return self.score_and_grad(bandwidth); },
            nb::arg("bandwidth") = 0,
            "Full-pipeline batch operation using per-thread DP buffers.\n"
            "For each pair: runs full DP (+ banded DP if bandwidth > 0), computes\n"
            "grad, saves score + guide_j + grad to the SeqPair.  The pairs' own DP\n"
            "tables are never allocated; dp_valid stays False after this call.\n"
            "Returns sum of scores.  Call compute_grad() afterwards to get the\n"
            "summed gradient.")
        .def(
            "drop_dp",
            [](SeqPairBatch& self) { self.drop_dp(); },
            "Drop DP tables on all pairs in parallel.  Frees O(mn) memory per pair.\n"
            "Cached scores, gradients, and guide_j vectors are preserved.")
        .def_prop_ro("n_threads", [](const SeqPairBatch& s) { return s.n_threads; });
}
