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
// One BatchEngine is selected at construction: one dispatch per batch call, never
// per pair. Settings live in that engine.
//
// SeqPairT<T> is a handle: either a standalone single pair (it owns a one-pair batch)
// or a view of pair i of a batch (b[i]).  Its operations are the engine's per-pair
// ones, on the calling thread.

#include <memory>
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
    using Var = std::variant<Engine<GapModel::Linear, AlignMode::Global>,
                             Engine<GapModel::Linear, AlignMode::Local>,
                             Engine<GapModel::Affine, AlignMode::Global>,
                             Engine<GapModel::Affine, AlignMode::Local>>;
    using SeqPair = SeqPairT<T>;

    SeqPairBatchT(GapModel gm, AlignMode am, GradMode gd, int n_threads = 0,
                  TracebackMode tb = TracebackMode::Default)
        : e_(make_engine_(gm, am, gd, n_threads, tb)), gm_(gm), am_(am), gd_(gd) {}

    SeqPairBatchT(const SeqPairBatchT&)            = delete;
    SeqPairBatchT& operator=(const SeqPairBatchT&) = delete;
    SeqPairBatchT(SeqPairBatchT&&)                 = default;
    SeqPairBatchT& operator=(SeqPairBatchT&&)      = default;

    GapModel gap_model() const noexcept { return gm_; }
    AlignMode align_mode() const noexcept { return am_; }
    GradMode grad_mode() const noexcept { return gd_; }

    // One switch per batch call; every batch has an engine, including an empty one.
    template<class F> decltype(auto) visit(F&& f) {
        return std::visit(std::forward<F>(f), e_);
    }
    template<class F> decltype(auto) visit(F&& f) const {
        return std::visit(std::forward<F>(f), e_);
    }

    // ── Adding pairs ─────────────────────────────────────────────────────────

    // Append a segment under an owned copy of params (tracks: see BatchEngine::add_many).
    void add_many(const std::vector<std::string_view>& seqs_a,
                  const std::vector<std::string_view>& seqs_b,
                  const AlignParams& params, int kernel = kBackendAuto,
                  const std::vector<std::vector<uint16_t>>* tracks = nullptr,
                  const std::vector<std::vector<uint16_t>>* gap_tracks = nullptr) {
        visit([&](auto& e) { e.add_many(seqs_a, seqs_b, params, kernel, tracks, gap_tracks); });
    }

    size_t size() const noexcept { return visit([](const auto& e) { return e.size(); }); }
    bool tracked() const noexcept { return visit([](const auto& e) { return e.tracked(); }); }
    int num_sets() const noexcept { return visit([](const auto& e) { return e.num_sets(); }); }
    const Alphabet& alphabet() const {
        return visit([](const auto& e) -> const Alphabet& { return e.alphabet(); });
    }
    SeqPair operator[](size_t i);

    // ── Batch operations  ──────────────────

    void set_params(const AlignParams& p) { visit([&](auto& e) { e.set_params(p); }); }
    double score_and_grad(bool keep_paths = false) {
        return visit([&](auto& e) { return e.score_and_grad(keep_paths); });
    }
    double banded_grad(int bandwidth, bool keep_paths = false) {
        return visit([&](auto& e) { return e.banded_grad(bandwidth, keep_paths); });
    }
    void drop_paths() { visit([](auto& e) { e.drop_paths(); }); }
    AlignParams compute_grad() {
        return visit([](auto& e) { return e.compute_grad(); });
    }
    std::vector<double> scores() const {
        return visit([](const auto& e) { return e.scores(); });
    }
    AlignParams weighted_grad(const double* w, size_t n) const {
        return visit([&](const auto& e) { return e.weighted_grad(w, n); });
    }
    void grads_into(double* matrices, double* gaps) const {
        visit([&](const auto& e) { e.grads_into(matrices, gaps); });
    }
    BatchResult align(const std::vector<ProblemInstance>& problems, const AlignParams& params,
                      int band = 0, int kernel = kBackendAuto, const StreamTracks& st = {}) const {
        return visit([&](const auto& e) { return e.align_stream(problems, params, band, kernel, st); });
    }

    // ── Settings ─────────────────────────────────────────────────────────────
    const BatchSettings& settings() const {
        return visit([](const auto& e) -> const BatchSettings& { return e.settings(); });
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
    bool track_counts() const { return settings().track_counts; }

#define NWGRAD_BATCH_SETTER(NAME, TYPE) \
    void NAME(TYPE v) { visit([&](auto& e) { e.NAME(v); }); }
    NWGRAD_BATCH_SETTER(set_n_threads, int)
    NWGRAD_BATCH_SETTER(set_hb_cutoff, int)
    NWGRAD_BATCH_SETTER(set_soft_temperature, double)
    NWGRAD_BATCH_SETTER(set_soft_impl, SoftImpl)
    NWGRAD_BATCH_SETTER(set_sorted_schedule, bool)
    NWGRAD_BATCH_SETTER(set_reserve_frac, double)
    NWGRAD_BATCH_SETTER(set_long_cost_ratio, double)
    NWGRAD_BATCH_SETTER(set_weight_lo, double)
    NWGRAD_BATCH_SETTER(set_weight_hi, double)
    NWGRAD_BATCH_SETTER(set_profile, bool)
    NWGRAD_BATCH_SETTER(set_track_counts, bool)
#undef NWGRAD_BATCH_SETTER
    void set_fill(bool rowwise, bool inter) {
        visit([&](auto& e) { e.set_fill(rowwise, inter); });
    }
    void set_soft_guide(bool lazy, bool posterior) {
        visit([&](auto& e) { e.set_soft_guide(lazy, posterior); });
    }

    using PhaseProfile = typename Engine<GapModel::Affine, AlignMode::Global>::PhaseProfile;
    std::vector<PhaseProfile> profile_out() const {
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
    static Var make_engine_(GapModel gm, AlignMode am, GradMode gd,
                            int n_threads, TracebackMode tb) {
        BatchSettings settings;
        settings.set_n_threads(n_threads);
        settings.traceback = tb;
        if (gm == GapModel::Linear) {
            if (am == AlignMode::Global) return Var(std::in_place_index<0>, settings, gd);
            return Var(std::in_place_index<1>, settings, gd);
        }
        if (am == AlignMode::Global) return Var(std::in_place_index<2>, settings, gd);
        return Var(std::in_place_index<3>, settings, gd);
    }

};

// ── SeqPairT: one pair — standalone, or a view of pair i of a batch ──────────

template<class T = double>
class SeqPairT {
public:
    using Batch = SeqPairBatchT<T>;

    // Standalone: a one-pair batch of its own, holding a copy of `params`; optionally
    // tracked (empty: the default track / gap track).
    SeqPairT(std::string_view a, std::string_view b, const AlignParams& params,
             GapModel gm, AlignMode am, GradMode gd = GradMode::Hard,
             int kernel = kBackendAuto, TracebackMode tb = TracebackMode::Default,
             std::vector<uint16_t> track = {}, std::vector<uint16_t> gap_track = {})
        : own_(std::make_unique<Batch>(gm, am, gd, 1, tb)), b_(own_.get()), i_(0) {
        own_->set_fill(false, false);   // one pair: nothing to share a vector with
        const bool tr = !track.empty() || !gap_track.empty();
        std::vector<std::vector<uint16_t>> tk{std::move(track)}, gk{std::move(gap_track)};
        own_->add_many({a}, {b}, params, kernel, tr ? &tk : nullptr, tr ? &gk : nullptr);
    }
    // View of pair i of `batch` (which must outlive the view).
    SeqPairT(Batch& batch, size_t i) : b_(&batch), i_(i) {}

    bool is_view() const noexcept { return !own_; }
    Batch& batch() noexcept { return *b_; }
    size_t index() const noexcept { return i_; }

    // ── Alignment (on the calling thread) ────────────────────────────────────
    double align(bool keep_paths = true) {
        e([&](auto& x) { x.score_and_grad_one(i_, keep_paths); });
        return score();
    }
    double banded_align(int bandwidth, bool keep_paths = true) {
        e([&](auto& x) { x.banded_one(i_, bandwidth, keep_paths); });
        return score();
    }
    std::pair<double, AlignParams> score_and_grad(bool keep_paths = true) {
        align(keep_paths);
        if (grad_mode() == GradMode::None)
            throw std::logic_error("nwgrad: grad_mode is None; construct with Hard or Soft to enable gradients");
        return {score(), grad()};
    }
    void drop_paths() { e([&](auto& x) { x.drop_path(i_); }); }

    [[deprecated("use align(); removed in nwgrad 0.8.0")]]
    void align_full() { PairApiCompat::full(*this); }
    [[deprecated("use banded_align(); removed in nwgrad 0.8.0")]]
    void realign_banded(int bandwidth) { PairApiCompat::banded(*this, bandwidth); }
    [[deprecated("read grad after align(); removed in nwgrad 0.8.0")]]
    void compute_grad() { PairApiCompat::publish(*this); }
    [[deprecated("use drop_paths(); removed in nwgrad 0.8.0")]]
    void drop_dp() { drop_paths(); }

    // ── Results ──────────────────────────────────────────────────────────────
    double score() const { return ce([&](const auto& x) { return x.score(i_); }); }
    AlignParams grad() const { return ce([&](const auto& x) { return x.grad(i_); }); }
    std::vector<uint16_t> track() const {
        return ce([&](const auto& x) { auto t = x.track(i_); return std::vector<uint16_t>(t.begin(), t.end()); });
    }
    std::vector<uint16_t> gap_track() const {
        return ce([&](const auto& x) { auto t = x.gap_track(i_); return std::vector<uint16_t>(t.begin(), t.end()); });
    }
    bool tracked() const { return b_->tracked(); }
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
    bool stored_path_valid() const { return ce([&](const auto& x) { return x.path_stored(i_); }); }
    [[deprecated("use stored_path_valid(); removed in nwgrad 0.8.0")]]
    bool dp_valid() const { return stored_path_valid(); }
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
