#pragma once
// nwgrad::logistic -- a binary logistic link over per-pair alignment scores.
//
// Model: P(y_i = 1) = p_i = expit(alpha + s_i), with s_i a pair's alignment
// score, y_i its label and alpha an intercept.  This is the likelihood
// DiscrimAlign maximises.
//
// Labels are the caller's responsibility and are not checked here: each y_i
// must lie in [0, 1], either a 0/1 class label or a soft label (a target
// probability, giving the Bernoulli cross-entropy), and 0 < sum_i y_i < n,
// i.e. not all 0 and not all 1.  Only then does the likelihood have a finite
// maximum in alpha.  Otherwise fit_alpha() and step() return a meaningless
// alpha instead of raising (with all labels 1, an alpha past which every
// expit rounds to 1).  Nothing else here depends on the labels being 0 or 1:
// they enter only through y_i - p_i.  Kept apart from the alignment code: it
// only uses SeqPairBatchT's public interface (scores(), size(), weighted_grad()).
//
// Determinism: every sum is taken over fixed blocks of BLOCK elements, each in
// index order, and the block sums are added in block order, so results do not
// depend on the thread count.  As elsewhere in nwgrad, no product is formed
// next to the add that consumes it: products are stored by out-of-line helpers
// and summed afterwards (the build also passes -ffp-contract=off).

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "../align_params.hpp"
#include "../parallel.hpp"
#include "../seq_pair_batch.hpp"

namespace nwgrad::logistic {

// Elements per block in every reduction.  Part of the results' definition: a
// different size sums in a different order and can change the last bits.
inline constexpr size_t BLOCK = 4096;

// The logistic function, in scipy.special.expit's form.
inline double expit(double x) noexcept { return 1.0 / (1.0 + std::exp(-x)); }

inline int resolve_threads(int n_threads) noexcept {
    return n_threads > 0 ? n_threads : default_thread_count();
}

// Sums over all elements at one alpha.  The log-likelihood itself is not
// computed: the alpha fit needs only its derivatives, and a caller that wants
// L evaluates it in a stable form from the logits (e.g. DiscrimAlign's
// logit_logL) rather than from clipped probabilities.
struct Sums {
    double g  = 0.0;   // dL/dalpha = sum (y - p)
    double h  = 0.0;   // -d2L/dalpha2 = sum p (1 - p)
};

namespace detail {

// Per-element terms of one block, stored out of line (no product beside an add).
[[gnu::noinline]] inline void block_terms(const double* s, const double* y, size_t n,
                                          double alpha, double* tg, double* th) {
    for (size_t i = 0; i < n; ++i) {
        const double p = expit(alpha + s[i]);
        tg[i] = y[i] - p;
        th[i] = p * (1.0 - p);
    }
}

// Sums of one block's stored terms, in index order.
[[gnu::noinline]] inline Sums block_sums(size_t n, const double* tg, const double* th) {
    Sums out;
    for (size_t i = 0; i < n; ++i) {
        out.g += tg[i];
        out.h += th[i];
    }
    return out;
}

// Run fn(block_index, lo, hi) for every block, in parallel.
template<class Fn>
void for_blocks(size_t n, int n_threads, Fn&& fn) {
    const size_t nblocks = (n + BLOCK - 1) / BLOCK;
    if (nblocks == 0) return;
    std::atomic<size_t> next{0};
    auto worker = [&]() {
        for (size_t b; (b = next.fetch_add(1, std::memory_order_relaxed)) < nblocks; )
            fn(b, b * BLOCK, std::min(n, (b + 1) * BLOCK));
    };
    run_workers_guarded(std::min<int>(resolve_threads(n_threads), static_cast<int>(nblocks)),
                        worker);
}

}  // namespace detail

// g and h at alpha.
inline Sums evaluate(const double* s, const double* y, size_t n, double alpha, int n_threads) {
    const size_t nblocks = (n + BLOCK - 1) / BLOCK;
    std::vector<Sums> partial(nblocks);
    detail::for_blocks(n, n_threads, [&](size_t b, size_t lo, size_t hi) {
        thread_local std::vector<double> tg, th;
        tg.resize(BLOCK); th.resize(BLOCK);
        detail::block_terms(s + lo, y + lo, hi - lo, alpha, tg.data(), th.data());
        partial[b] = detail::block_sums(hi - lo, tg.data(), th.data());
    });
    Sums out;
    for (const Sums& p : partial) {
        out.g += p.g;
        out.h += p.h;
    }
    return out;
}

// expit(alpha + s_i) for every i, into out.
inline void probabilities(const double* s, size_t n, double alpha, double* out, int n_threads) {
    detail::for_blocks(n, n_threads, [&](size_t, size_t lo, size_t hi) {
        for (size_t i = lo; i < hi; ++i) out[i] = expit(alpha + s[i]);
    });
}

// The intercept maximising the likelihood, from alpha0: the root of g, which
// is strictly decreasing in alpha.  Plain Newton steps while |step| <= 1 (the
// warm-start case); otherwise bracket the root and run Numerical Recipes'
// rtsafe (Newton inside the bracket, bisection when Newton would leave it or
// converge too slowly).  The same algorithm and tolerances as DiscrimAlign's
// logit_link.fit_alpha().  `at_alpha0`, if given, must be evaluate() at alpha0.
// The labels are not checked: see the precondition at the top of this file.
inline double fit_alpha(const double* s, const double* y, size_t n, double alpha0,
                        int n_threads, double tol = 1e-12, int max_newton = 8,
                        int maxiter = 200, const Sums* at_alpha0 = nullptr) {
    if (!std::isfinite(alpha0) || !std::isfinite(tol) || tol < 0)
        throw std::invalid_argument("nwgrad.logistic.fit_alpha: alpha0 and tol must be finite, tol >= 0");
    // A non-finite score makes g NaN or infinite, and both bracket tests are false for
    // NaN: without this check the bisection "converges" on its step size alone.
    auto finite = [](double g, double h) {
        if (!std::isfinite(g) || !std::isfinite(h))
            throw std::invalid_argument("nwgrad.logistic.fit_alpha: non-finite derivative (a score is NaN or infinite)");
        return std::pair<double, double>(g, h);
    };
    auto derivatives = [&](double alpha) {
        const Sums d = evaluate(s, y, n, alpha, n_threads);
        return finite(d.g, d.h);
    };

    double alpha = alpha0;
    for (int k = 0; k < max_newton; ++k) {
        double g, h;
        std::tie(g, h) = (k == 0 && at_alpha0) ? finite(at_alpha0->g, at_alpha0->h)
                                               : derivatives(alpha);
        const double step = h > 0 ? g / h : std::numeric_limits<double>::infinity();
        if (!std::isfinite(step) || std::abs(step) > 1.0) break;
        alpha += step;
        if (std::abs(step) <= tol * std::max(1.0, std::abs(alpha))) return alpha;
    }

    // Bracket the root: g(lo) >= 0 >= g(hi).
    double lo = alpha - 1.0, hi = alpha + 1.0, width = 1.0;
    int k = 0;
    for (; k < maxiter && derivatives(lo).first < 0; ++k) { lo -= width; width *= 2; }
    if (k == maxiter) throw std::runtime_error("nwgrad.logistic.fit_alpha: could not bracket the root");
    width = 1.0;
    for (k = 0; k < maxiter && derivatives(hi).first > 0; ++k) { hi += width; width *= 2; }
    if (k == maxiter) throw std::runtime_error("nwgrad.logistic.fit_alpha: could not bracket the root");

    alpha = std::min(std::max(alpha, lo), hi);
    double dx_old = hi - lo, dx = dx_old;
    double g, h;
    std::tie(g, h) = derivatives(alpha);
    for (k = 0; k < maxiter; ++k) {
        if (g == 0.0) return alpha;
        if (h > 0 && lo <= alpha + g / h && alpha + g / h <= hi && std::abs(2 * g) <= std::abs(dx_old * h)) {
            dx_old = dx; dx = g / h; alpha += dx;
        } else {
            dx_old = dx; dx = 0.5 * (hi - lo); alpha = lo + dx;
        }
        if (std::abs(dx) <= tol * std::max(1.0, std::abs(alpha))) return alpha;
        std::tie(g, h) = derivatives(alpha);
        if (g > 0) lo = alpha;
        else       hi = alpha;
    }
    throw std::runtime_error("nwgrad.logistic.fit_alpha: did not converge");
}

// One iteration's logistic work over a batch whose scores and gradients are
// cached (after score_and_grad()): the fitted alpha, and
// sum_i (y_i - p_i) grad_i at that alpha.  The labels are not checked: see the
// precondition at the top of this file.
struct Step {
    double alpha;
    AlignParams grad;
};

template<class T>
Step step(const SeqPairBatchT<T>& batch, const double* y, size_t n, double alpha0) {
    if (n != batch.size())
        throw std::invalid_argument("nwgrad.logistic.step: needs one label per pair (got " +
                                    std::to_string(n) + " labels for " +
                                    std::to_string(batch.size()) + " pairs)");
    const int threads = batch.n_threads();
    const std::vector<double> s = batch.scores();
    const Sums at0 = evaluate(s.data(), y, n, alpha0, threads);
    const double alpha = fit_alpha(s.data(), y, n, alpha0, threads, 1e-12, 8, 200, &at0);
    std::vector<double> w(n);
    detail::for_blocks(n, threads, [&](size_t, size_t lo, size_t hi) {
        for (size_t i = lo; i < hi; ++i) w[i] = y[i] - expit(alpha + s[i]);
    });
    return Step{alpha, batch.weighted_grad(w.data(), n)};
}

}  // namespace nwgrad::logistic
