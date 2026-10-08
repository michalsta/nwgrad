#pragma once

// ── SeqPairBatchT<T> and SeqPairT<T> ────────────────────────────────────────
//
// The runtime-typed face of BatchEngine<T, GapModel, AlignMode> (batch_engine.hpp).
//
// A batch holds pairs of ONE problem type — one gap model, one alignment mode, one
// grad mode — fixed at construction:
//
//     SeqPairBatchT<double> b(GapModel::Affine, AlignMode::Local, GradMode::Hard);
//     b.add_many(seqs_a, seqs_b, params);              // params per call (a segment)
//
// or, for code written against the pre-0.6 API, by the FIRST add_many() of a batch
// built without a type (later calls must agree; the Python binding warns).  Underneath
// is one BatchEngine of that type, selected once: a dispatch per batch call, never per
// pair.  Settings live here, so they can be set before the type is known, and are
// pushed into the engine when it exists.
//
// SeqPairT<T> is a handle: either a standalone single pair (it owns a one-pair batch)
// or a view of pair i of a batch (b[i]).  Its operations are the engine's per-pair
// ones, on the calling thread.

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "batch_engine.hpp"

template<class T> class SeqPairT;

inline const char* gap_model_name(GapModel g) { return g == GapModel::Linear ? "linear" : "affine"; }
inline const char* align_mode_name(AlignMode a) { return a == AlignMode::Local ? "local" : "global"; }
inline const char* grad_mode_name(GradMode g) {
    return g == GradMode::Hard ? "hard" : g == GradMode::Soft ? "soft" : "none";
}

template<class T = double>
class SeqPairBatchT {
public:
    template<GapModel G, AlignMode A> using Engine = BatchEngine<T, G, A>;
    using Var = std::variant<std::monostate,
                             Engine<GapModel::Linear, AlignMode::Global>,
                             Engine<GapModel::Linear, AlignMode::Local>,
                             Engine<GapModel::Affine, AlignMode::Global>,
                             Engine<GapModel::Affine, AlignMode::Local>>;
    using SeqPair = SeqPairT<T>;

    // Typed: the batch's problem type is fixed here.
    SeqPairBatchT(GapModel gm, AlignMode am, GradMode gd, int n_threads = 0,
                  TracebackMode tb = TracebackMode::Default) {
        pending_.set_n_threads(n_threads);
        pending_.traceback = tb;
        make_engine_(gm, am, gd);
    }
    // Untyped (the pre-0.6 form): the first add_many() that adds pairs fixes the type.
    explicit SeqPairBatchT(int n_threads = 0, TracebackMode tb = TracebackMode::Default) {
        pending_.set_n_threads(n_threads);
        pending_.traceback = tb;
    }

    SeqPairBatchT(const SeqPairBatchT&)            = delete;
    SeqPairBatchT& operator=(const SeqPairBatchT&) = delete;
    SeqPairBatchT(SeqPairBatchT&&)                 = default;
    SeqPairBatchT& operator=(SeqPairBatchT&&)      = default;

    bool typed() const noexcept { return !std::holds_alternative<std::monostate>(e_); }
    GapModel  gap_model()  const { require_typed_(); return gm_; }
    AlignMode align_mode() const { require_typed_(); return am_; }
    GradMode  grad_mode()  const { require_typed_(); return gd_; }

    // Visit the engine (throws on an untyped batch).  One switch per batch call.
    template<class F> decltype(auto) visit(F&& f) {
        require_typed_();
        switch (e_.index()) {
            case 1:  return f(std::get<1>(e_));
            case 2:  return f(std::get<2>(e_));
            case 3:  return f(std::get<3>(e_));
            default: return f(std::get<4>(e_));
        }
    }
    template<class F> decltype(auto) visit(F&& f) const {
        require_typed_();
        switch (e_.index()) {
            case 1:  return f(std::get<1>(e_));
            case 2:  return f(std::get<2>(e_));
            case 3:  return f(std::get<3>(e_));
            default: return f(std::get<4>(e_));
        }
    }

    // ── Adding pairs ─────────────────────────────────────────────────────────

    // Append a segment of pairs under a copy of `params` (the batch owns it; a later
    // in-place change to the caller's object does not reach the batch).  gm/am/gd: fix the type of an untyped batch; on a typed one,
    // they must agree with it (one batch, one problem type).
    void add_many(const std::vector<std::string_view>& seqs_a,
                  const std::vector<std::string_view>& seqs_b,
                  const AlignParams& params,
                  std::optional<GapModel> gm, std::optional<AlignMode> am,
                  std::optional<GradMode> gd, int kernel = kBackendAuto) {
        if (!typed()) {
            if (!gm || !am || !gd)
                throw std::invalid_argument(
                    "nwgrad: this batch has no problem type yet; construct it with gap_model, "
                    "mode and grad_mode (or pass all three to its first add_many())");
            // Provisional: an untyped batch takes its type from the first add_many() that
            // ADDS pairs.  One that adds nothing, or throws, leaves it untyped (as before
            // 0.6, when an empty add_many() was a no-op).
            make_engine_(*gm, *am, *gd);
            try {
                visit([&](auto& e) { e.add_many(seqs_a, seqs_b, params, kernel); });
            } catch (...) {
                e_.template emplace<0>();   // pending_ still holds the settings
                throw;
            }
            if (size() == 0) e_.template emplace<0>();
            return;
        } else {
            auto clash = [&](const char* what, const char* have, const char* want) {
                throw std::invalid_argument(
                    std::string("nwgrad: this batch holds ") + gap_model_name(gm_) + " " +
                    align_mode_name(am_) + " " + grad_mode_name(gd_) + " pairs; add_many(" + what +
                    "=\"" + want + "\") cannot add " + want + " ones to it (have \"" + have +
                    "\").  One batch holds one problem type: use a separate batch.");
            };
            if (gm && *gm != gm_) clash("gap_model", gap_model_name(gm_), gap_model_name(*gm));
            if (am && *am != am_) clash("mode", align_mode_name(am_), align_mode_name(*am));
            if (gd && *gd != gd_) clash("grad_mode", grad_mode_name(gd_), grad_mode_name(*gd));
        }
        visit([&](auto& e) { e.add_many(seqs_a, seqs_b, params, kernel); });
    }
    // Typed form (C++ convenience): the batch must already have its type.
    void add_many(const std::vector<std::string_view>& seqs_a,
                  const std::vector<std::string_view>& seqs_b,
                  const AlignParams& params, int kernel = kBackendAuto) {
        add_many(seqs_a, seqs_b, params, std::nullopt, std::nullopt, std::nullopt, kernel);
    }
    // The pre-0.6 C++ signature.
    void add_many(const std::vector<std::string_view>& seqs_a,
                  const std::vector<std::string_view>& seqs_b,
                  const AlignParams& params, GapModel gm, AlignMode am, GradMode gd,
                  int kernel = kBackendAuto) {
        add_many(seqs_a, seqs_b, params, std::optional<GapModel>(gm), std::optional<AlignMode>(am),
                 std::optional<GradMode>(gd), kernel);
    }

    size_t size() const noexcept {
        return typed() ? visit([](const auto& e) { return e.size(); }) : 0;
    }
    const Alphabet& alphabet() const {
        if (!typed()) throw std::logic_error("nwgrad: empty batch has no alphabet");
        return visit([](const auto& e) -> const Alphabet& { return e.alphabet(); });
    }
    SeqPair operator[](size_t i);

    // ── Batch operations (an untyped batch is an empty one) ──────────────────

    void set_params(const AlignParams& p) { if (typed()) visit([&](auto& e) { e.set_params(p); }); }
    double score_and_grad(bool keep_paths = false, bool hold_grads = false) {
        if (!typed()) return 0.0;
        return visit([&](auto& e) { return e.score_and_grad(keep_paths, hold_grads); });
    }
    double banded_grad(int bandwidth, bool keep_paths = false, bool hold_grads = false) {
        if (!typed()) {
            if (bandwidth <= 0)
                throw std::invalid_argument("nwgrad: banded_grad() needs bandwidth > 0 (got " +
                                            std::to_string(bandwidth) + "); use score_and_grad() for full DP");
            return 0.0;
        }
        return visit([&](auto& e) { return e.banded_grad(bandwidth, keep_paths, hold_grads); });
    }
    void drop_paths() { if (typed()) visit([](auto& e) { e.drop_paths(); }); }
    AlignParams compute_grad() {
        if (!typed()) throw std::logic_error("nwgrad: empty batch has no alphabet");
        return visit([](auto& e) { return e.compute_grad(); });
    }
    std::vector<double> scores() const {
        if (!typed()) return {};
        return visit([](const auto& e) { return e.scores(); });
    }
    AlignParams weighted_grad(const double* w, size_t n) const {
        if (!typed()) throw std::logic_error("nwgrad: empty batch has no alphabet");
        return visit([&](const auto& e) { return e.weighted_grad(w, n); });
    }
    void grads_into(double* matrices, double* gaps) const {
        if (!typed()) throw std::logic_error("nwgrad: empty batch has no alphabet");
        visit([&](const auto& e) { e.grads_into(matrices, gaps); });
    }
    BatchResult align(const std::vector<ProblemInstance>& problems, const AlignParams& params,
                      int band = 0, int kernel = kBackendAuto) const {
        if (!typed())
            throw std::logic_error(
                "nwgrad: align() needs the batch's problem type; construct the batch with "
                "gap_model, mode and grad_mode");
        return visit([&](const auto& e) { return e.align_stream(problems, params, band, kernel); });
    }

    // ── Settings ─────────────────────────────────────────────────────────────
    // One copy of each: the engine's once the batch is typed, pending_ before (the engine
    // is built from it).  Read through settings(); every setter goes to that one copy,
    // through the same validating setters (BatchSettings / BatchEngine).

    const BatchSettings& settings() const {
        return typed() ? visit([](const auto& e) -> const BatchSettings& { return e.settings(); })
                       : pending_;
    }
    int n_threads() const { return settings().n_threads; }
    TracebackMode traceback() const { return settings().traceback; }
    TracebackMode traceback_resolved() const {
        return visit([](const auto& e) { return e.traceback_resolved(); });
    }
    int hb_cutoff() const { return settings().hb_cutoff; }
    bool rowwise_full() const { return settings().rowwise_full; }
    bool inter_fill() const { return settings().inter_fill; }
    SoftImpl soft_impl() const { return settings().soft_impl; }
    double soft_temperature() const { return settings().soft_temperature; }
    bool soft_guide_lazy() const { return settings().soft_guide_lazy; }
    bool soft_guide_posterior() const { return settings().soft_guide_posterior; }
    bool sorted_schedule() const { return settings().sorted_schedule; }
    double reserve_frac() const { return settings().reserve_frac; }
    double long_cost_ratio() const { return settings().long_cost_ratio; }
    double weight_lo() const { return settings().weight_lo; }
    double weight_hi() const { return settings().weight_hi; }
    bool profile() const { return settings().profile; }

#define NWGRAD_BATCH_SETTER(NAME, TYPE)                                               \
    void NAME(TYPE v) {                                                               \
        if (typed()) visit([&](auto& e) { e.NAME(v); }); else pending_.NAME(v);         \
    }
    NWGRAD_BATCH_SETTER(set_n_threads, int)
    NWGRAD_BATCH_SETTER(set_hb_cutoff, int)
    NWGRAD_BATCH_SETTER(set_soft_temperature, double)
#undef NWGRAD_BATCH_SETTER
    void set_fill(bool rowwise, bool inter) {
        if (typed()) visit([&](auto& e) { e.set_fill(rowwise, inter); });
        else pending_.set_fill(rowwise, inter);
    }
    void set_soft_guide(bool lazy, bool posterior) {
        if (typed()) visit([&](auto& e) { e.set_soft_guide(lazy, posterior); });
        else pending_.set_soft_guide(lazy, posterior);
    }
    // Plain fields, no validation, no side effects.
#define NWGRAD_BATCH_FIELD(NAME, FIELD, TYPE)                                         \
    void NAME(TYPE v) {                                                               \
        if (typed()) visit([&](auto& e) { e.NAME(v); }); else pending_.FIELD = v;       \
    }
    NWGRAD_BATCH_FIELD(set_soft_impl, soft_impl, SoftImpl)
    NWGRAD_BATCH_FIELD(set_sorted_schedule, sorted_schedule, bool)
    NWGRAD_BATCH_FIELD(set_reserve_frac, reserve_frac, double)
    NWGRAD_BATCH_FIELD(set_long_cost_ratio, long_cost_ratio, double)
    NWGRAD_BATCH_FIELD(set_weight_lo, weight_lo, double)
    NWGRAD_BATCH_FIELD(set_weight_hi, weight_hi, double)
    NWGRAD_BATCH_FIELD(set_profile, profile, bool)
#undef NWGRAD_BATCH_FIELD

    using PhaseProfile = typename Engine<GapModel::Affine, AlignMode::Global>::PhaseProfile;
    std::vector<PhaseProfile> profile_out() const {
        if (!typed()) return {};
        return visit([](const auto& e) {
            std::vector<PhaseProfile> v;
            for (const auto& p : e.profile_out) v.push_back({p.chunk_s, p.reserve_s, p.chunk_cells,
                                                              p.reserve_cells, p.chunk_tasks,
                                                              p.reserve_tasks, p.finish_s});
            return v;
        });
    }

private:
    Var e_;
    GapModel gm_ = GapModel::Affine;
    AlignMode am_ = AlignMode::Global;
    GradMode gd_ = GradMode::Hard;
    BatchSettings pending_;   // the settings until an engine exists (then it holds them)

    void require_typed_() const {
        if (!typed())
            throw std::logic_error(
                "nwgrad: this batch has no problem type yet (no pairs); construct it with "
                "gap_model, mode and grad_mode");
    }
    void make_engine_(GapModel gm, AlignMode am, GradMode gd) {
        gm_ = gm; am_ = am; gd_ = gd;
        if      (gm == GapModel::Linear && am == AlignMode::Global) e_.template emplace<1>(pending_, gd);
        else if (gm == GapModel::Linear && am == AlignMode::Local)  e_.template emplace<2>(pending_, gd);
        else if (gm == GapModel::Affine && am == AlignMode::Global) e_.template emplace<3>(pending_, gd);
        else                                                        e_.template emplace<4>(pending_, gd);
    }
};

// ── SeqPairT: one pair — standalone, or a view of pair i of a batch ──────────

template<class T = double>
class SeqPairT {
public:
    using Batch = SeqPairBatchT<T>;

    // Standalone: a one-pair batch of its own, holding a copy of `params`.
    SeqPairT(std::string_view a, std::string_view b, const AlignParams& params,
             GapModel gm, AlignMode am, GradMode gd = GradMode::Hard,
             int kernel = kBackendAuto, TracebackMode tb = TracebackMode::Default)
        : own_(std::make_unique<Batch>(gm, am, gd, 1, tb)), b_(own_.get()), i_(0) {
        own_->set_fill(false, false);   // one pair: nothing to share a vector with
        own_->add_many({a}, {b}, params, kernel);
    }
    // View of pair i of `batch` (which must outlive the view).
    SeqPairT(Batch& batch, size_t i) : b_(&batch), i_(i) {}

    bool is_view() const noexcept { return !own_; }
    Batch& batch() noexcept { return *b_; }
    size_t index() const noexcept { return i_; }

    // ── Alignment (on the calling thread) ────────────────────────────────────
    // align_full(): full DP, score and stored path; the gradient is held until
    // compute_grad() (as the pre-0.6 SeqPair, which derived it from retained tables).
    void align_full() { e([&](auto& x) { x.score_and_grad_one(i_, true, true); }); }
    void realign_banded(int bandwidth) { e([&](auto& x) { x.banded_one(i_, bandwidth, true, true); }); }
    void compute_grad() { e([&](auto& x) { x.compute_grad_one(i_); }); }
    std::pair<double, AlignParams> score_and_grad() {
        align_full();
        compute_grad();
        return {score(), grad()};
    }
    void alloc_dp() noexcept {}   // nothing to allocate: the DP runs on a worker buffer
    void drop_dp() { e([&](auto& x) { x.drop_path(i_); }); }

    // ── Results ──────────────────────────────────────────────────────────────
    double score() const { return ce([&](const auto& x) { return x.score(i_); }); }
    AlignParams grad() const { return ce([&](const auto& x) { return x.grad(i_); }); }
    std::pair<std::string, std::string> aligned() const {
        return ce([&](const auto& x) { return x.aligned(i_); });
    }
    std::pair<std::vector<int64_t>, std::vector<int64_t>> coordinates() const {
        return ce([&](const auto& x) { return x.coordinates(i_); });
    }
    std::vector<int> guide_j() { return e([&](auto& x) { return x.guide_j(i_); }); }
    bool path_valid()  const { return ce([&](const auto& x) { return x.path_valid(i_); }); }
    bool score_valid() const { return ce([&](const auto& x) { return x.score_valid(i_); }); }
    bool grad_valid()  const { return ce([&](const auto& x) { return x.grad_valid(i_); }); }
    bool dp_valid()    const { return ce([&](const auto& x) { return x.path_stored(i_); }); }
    std::string seq_a() const { return ce([&](const auto& x) { return x.seq_a(i_); }); }
    std::string seq_b() const { return ce([&](const auto& x) { return x.seq_b(i_); }); }
    size_t len_a() const { return ce([&](const auto& x) { return x.len_a(i_); }); }
    size_t len_b() const { return ce([&](const auto& x) { return x.len_b(i_); }); }
    int kernel() const { return ce([&](const auto& x) { return x.kernel(i_); }); }
    GapModel gap_model() const { return b_->gap_model(); }
    AlignMode align_mode() const { return b_->align_mode(); }
    GradMode grad_mode() const { return b_->grad_mode(); }
    TracebackMode traceback() const { return b_->traceback_resolved(); }
    const Alphabet& alphabet() const { return b_->alphabet(); }

    // ── Settings: a standalone pair's own; a view's belong to its batch ──────
    void set_params(const AlignParams& p) {
        standalone_("set_params");
        own_->visit([&](auto& x) { x.set_params_segment(0, p); });
    }
    int hb_cutoff() const { return b_->hb_cutoff(); }
    void set_hb_cutoff(int v) { standalone_("hb_cutoff"); own_->set_hb_cutoff(v); }
    bool rowwise_full() const { return b_->rowwise_full(); }
    void set_rowwise_full(bool on) { standalone_("fill"); own_->set_fill(on, false); }
    SoftImpl soft_impl() const { return b_->soft_impl(); }
    void set_soft_impl(SoftImpl s) { standalone_("soft_impl"); own_->set_soft_impl(s); }
    double soft_temperature() const { return b_->soft_temperature(); }
    void set_soft_temperature(double t) { standalone_("soft_temperature"); own_->set_soft_temperature(t); }

private:
    std::unique_ptr<Batch> own_;
    Batch* b_;
    size_t i_;

    template<class F> decltype(auto) e(F&& f) { return b_->visit(std::forward<F>(f)); }
    template<class F> decltype(auto) ce(F&& f) const {
        return std::as_const(*b_).visit(std::forward<F>(f));
    }
    void standalone_(const char* what) const {
        if (own_) return;
        throw std::logic_error(
            std::string("nwgrad: ") + what + " of a pair in a batch is the batch's: set it on "
            "the batch (batch." + what + "), which applies to every pair");
    }
};

template<class T>
SeqPairT<T> SeqPairBatchT<T>::operator[](size_t i) {
    if (i >= size()) throw std::out_of_range("nwgrad: SeqPairBatch index out of range");
    return SeqPairT<T>(*this, i);
}

// Default (double) aliases; Python binds the float32 classes as SeqPair / SeqPairBatch.
using SeqPairBatch = SeqPairBatchT<double>;
using SeqPair      = SeqPairT<double>;
