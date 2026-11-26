#pragma once

// ── The single-pair convenience functions' engine side ───────────────────────────
//
// nwgrad.nw_score / sw_grad / glocal_soft_grad / ... (py_exports.cpp, bind_convenience):
// one Aligner on the calling thread, Full or GuideBanded, for one (gap model, mode).
// The bindings encode the strings, parse the guide, the tracks and the kernel name, then
// call these.  Like SeqPairBatchT's members (seq_pair_batch.hpp), the bodies are not
// compiled in the bindings unit, which defines NWGRAD_ENGINE_EXTERN, but in
// engine_<type>.cpp: keep the engine out of the bindings' inlining budget.

#include <cstdint>
#include <utility>
#include <vector>

#include "aligner.hpp"

// One problem, as the bindings hand it over: codes of A and B (the params' alphabet),
// band (0 = none), guide (empty = none), per-residue track and per-row gap track (empty =
// defaults), and the Viterbi backend (parse_backend()).
struct PairProblem {
    const std::vector<uint8_t>& a;
    const std::vector<uint8_t>& b;
    const AlignParams& params;
    int band;
    const std::vector<int>& guide_j;
    const std::vector<uint16_t>& track;
    const std::vector<uint16_t>& gap_track;
    int kernel;
};

// The Viterbi score.
template<class T> double pair_score(GapModel gm, AlignMode am, const PairProblem& p);
// (score, hard subgradient).
template<class T> std::pair<double, AlignParams> pair_hard_grad(GapModel gm, AlignMode am, const PairProblem& p);
// (T*log Z(params/T), soft gradient): floating T only.
template<class T>
std::pair<double, AlignParams> pair_soft_grad(GapModel gm, AlignMode am, const PairProblem& p, SoftImpl si,
                                              double temperature);

#if !defined(NWGRAD_ENGINE_EXTERN)
namespace pair_entry_detail {
// `body(al, buf)` on a set-up Aligner<GM, AM, Full or GuideBanded, T>: GuideBanded when
// there is a band or a guide.  The kernel is set before set_problem, as it always was.
template<class T, GapModel GM, AlignMode AM, class Body>
inline auto with_aligner(const PairProblem& p, Body&& body) {
    DpBufferT<T> buf;
    if (p.band > 0 || !p.guide_j.empty()) {
        Aligner<GM, AM, AlignBand::GuideBanded, T> al;
        al.set_kernel(p.kernel);
        return body(al, buf);
    }
    Aligner<GM, AM, AlignBand::Full, T> al;
    al.set_kernel(p.kernel);
    return body(al, buf);
}
template<class T, class Body>
inline auto dispatch(GapModel gm, AlignMode am, const PairProblem& p, Body&& body) {
    if (gm == GapModel::Linear) {
        if (am == AlignMode::Global) return with_aligner<T, GapModel::Linear, AlignMode::Global>(p, body);
        if (am == AlignMode::Local) return with_aligner<T, GapModel::Linear, AlignMode::Local>(p, body);
        return with_aligner<T, GapModel::Linear, AlignMode::Glocal>(p, body);
    }
    if (am == AlignMode::Global) return with_aligner<T, GapModel::Affine, AlignMode::Global>(p, body);
    if (am == AlignMode::Local) return with_aligner<T, GapModel::Affine, AlignMode::Local>(p, body);
    return with_aligner<T, GapModel::Affine, AlignMode::Glocal>(p, body);
}
template<class AL>
inline void set_problem(AL& al, const PairProblem& p) {
    al.set_problem(p.a, p.b, p.params, p.band, p.guide_j, p.track, p.gap_track);
}
}  // namespace pair_entry_detail

template<class T> inline double pair_score(GapModel gm, AlignMode am, const PairProblem& p) {
    return pair_entry_detail::dispatch<T>(gm, am, p, [&](auto& al, DpBufferT<T>& buf) {
        pair_entry_detail::set_problem(al, p);
        return al.compute_score(buf);
    });
}
template<class T>
inline std::pair<double, AlignParams> pair_hard_grad(GapModel gm, AlignMode am, const PairProblem& p) {
    return pair_entry_detail::dispatch<T>(gm, am, p, [&](auto& al, DpBufferT<T>& buf) {
        pair_entry_detail::set_problem(al, p);
        al.compute_viterbi(buf);
        AlignParams grad = AlignParams::zeros_like(p.params);
        al.hard_grad(buf, grad);
        return std::pair<double, AlignParams>(al.score(), std::move(grad));
    });
}
template<class T>
inline std::pair<double, AlignParams> pair_soft_grad(GapModel gm, AlignMode am, const PairProblem& p,
                                                     SoftImpl si, double temperature) {
    static_assert(std::is_floating_point_v<T>, "the soft path has no integer form");
    return pair_entry_detail::dispatch<T>(gm, am, p, [&](auto& al, DpBufferT<T>& buf) {
        al.set_soft_impl(si);
        al.set_soft_temperature(temperature);
        pair_entry_detail::set_problem(al, p);
        al.compute_forward_back(buf);
        AlignParams grad = AlignParams::zeros_like(p.params);
        al.soft_grad(buf, grad);
        return std::pair<double, AlignParams>(al.log_z(), std::move(grad));
    });
}
#endif  // NWGRAD_ENGINE_EXTERN
