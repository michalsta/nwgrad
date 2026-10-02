// Python bindings for nwgrad::logistic, as the submodule nwgrad_ext.logistic
// (re-exported as nwgrad.logistic).  Registered from py_exports.cpp.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "logistic.hpp"

namespace nb = nanobind;
using vec_f64 = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

namespace {

void check_same_length(const vec_f64& scores, const vec_f64& labels) {
    if (scores.shape(0) != labels.shape(0))
        throw std::invalid_argument("nwgrad.logistic: scores and labels must have the same length (" +
                                    std::to_string(scores.shape(0)) + " vs " +
                                    std::to_string(labels.shape(0)) + ")");
}

template<class T>
void bind_step(nb::module_& m) {
    m.def(
        "step",
        [](const SeqPairBatchT<T>& batch, vec_f64 labels, double alpha0) {
            return nwgrad::logistic::step(batch, labels.data(), labels.shape(0), alpha0);
        },
        nb::arg("batch"), nb::arg("labels"), nb::arg("alpha0"),
        "One iteration's logistic work over a batch whose scores and gradients are\n"
        "cached (call score_and_grad() first).  Returns a Step: the log-likelihood\n"
        "at alpha0 (loglik_at_alpha0), the fitted intercept (alpha, as fit_alpha()),\n"
        "and grad = sum_i (labels[i] - expit(alpha + score_i)) * grad_i, as\n"
        "weighted_grad() would return it.  labels: float64 array of 0s and 1s, one\n"
        "per pair, both classes present.  Uses the batch's n_threads; the result\n"
        "does not depend on it.");
}

}  // namespace

void bind_logistic(nb::module_& parent) {
    nb::module_ m = parent.def_submodule(
        "logistic",
        "A binary logistic link over per-pair alignment scores: P(y = 1) =\n"
        "expit(alpha + score).  The likelihood DiscrimAlign maximises.  Every sum\n"
        "is taken over fixed blocks of elements, so results do not depend on\n"
        "n_threads (0 = nwgrad's default thread count).");

    nb::class_<nwgrad::logistic::Step>(m, "Step")
        .def_ro("alpha", &nwgrad::logistic::Step::alpha, "The fitted intercept.")
        .def_ro("loglik_at_alpha0", &nwgrad::logistic::Step::loglik_at_alpha0,
                "The log-likelihood at the starting intercept alpha0.")
        .def_ro("grad", &nwgrad::logistic::Step::grad,
                "sum_i (y_i - p_i) grad_i at the fitted alpha, as an AlignParams.")
        .def("__repr__", [](const nwgrad::logistic::Step& st) {
            return "Step(alpha=" + std::to_string(st.alpha) +
                   ", loglik_at_alpha0=" + std::to_string(st.loglik_at_alpha0) + ")";
        });

    m.def(
        "probabilities",
        [](vec_f64 scores, double alpha, int n_threads) {
            const size_t n = scores.shape(0);
            double* buf = new double[n];
            nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<double*>(p); });
            nwgrad::logistic::probabilities(scores.data(), n, alpha, buf, n_threads);
            size_t shape[1] = {n};
            return nb::ndarray<nb::numpy, double>(buf, 1, shape, owner);
        },
        nb::arg("scores"), nb::arg("alpha"), nb::arg("n_threads") = 0,
        "expit(alpha + scores) as a float64 array.");

    m.def(
        "log_likelihood",
        [](vec_f64 scores, vec_f64 labels, double alpha, int n_threads) {
            check_same_length(scores, labels);
            nwgrad::logistic::check_labels(labels.data(), labels.shape(0));
            return nwgrad::logistic::log_likelihood(scores.data(), labels.data(),
                                                    scores.shape(0), alpha, n_threads);
        },
        nb::arg("scores"), nb::arg("labels"), nb::arg("alpha"), nb::arg("n_threads") = 0,
        "sum_i y_i log(c_i) + (1 - y_i) log1p(-c_i), c_i = expit(alpha + score_i)\n"
        "clipped to [eps, 1 - eps].  labels: float64 0s and 1s, both classes present.");

    m.def(
        "fit_alpha",
        [](vec_f64 scores, vec_f64 labels, double alpha0, int n_threads, double tol,
           int max_newton, int maxiter) {
            check_same_length(scores, labels);
            return nwgrad::logistic::fit_alpha(scores.data(), labels.data(), scores.shape(0),
                                               alpha0, n_threads, tol, max_newton, maxiter);
        },
        nb::arg("scores"), nb::arg("labels"), nb::arg("alpha0"), nb::arg("n_threads") = 0,
        nb::arg("tol") = 1e-12, nb::arg("max_newton") = 8, nb::arg("maxiter") = 200,
        "The intercept maximising the likelihood, from alpha0: plain Newton steps\n"
        "while |step| <= 1, otherwise a bracketed safeguarded Newton (rtsafe).\n"
        "Exact to rounding from any start.  Raises ValueError unless labels are 0/1\n"
        "with both classes present.");

    bind_step<float>(m);
    bind_step<double>(m);
}
