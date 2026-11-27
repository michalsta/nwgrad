#pragma once

// ── SeqPairBatchT<T> and SeqPairT<T> ────────────────────────────────────────
//
// The runtime-typed face of BatchEngine<T, GapModel, AlignMode> (batch_engine.hpp).
//
// A batch holds pairs of ONE problem type — one gap model, one alignment mode, one
// grad mode — fixed at construction:
//
//     SeqPairBatchT<double> b(seqs_a, seqs_b, params,  // the pairs, given once
//                             GapModel::Affine, AlignMode::Local, GradMode::Hard);
//
// The pairs are given once, to the constructor.  The pairless constructor makes a batch
// for streaming align() only (nothing kept per pair).
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
inline const char* align_mode_name(AlignMode a) {
    return a == AlignMode::Local ? "local" : a == AlignMode::Glocal ? "glocal" : "global";
}
inline const char* grad_mode_name(GradMode g) {
    return g == GradMode::Hard ? "hard" : g == GradMode::Soft ? "soft" : "none";
}

// ── Where the member bodies are compiled ──────────────────────────────────────
// Every member that reaches the engine (BatchEngine, Aligner) is only DECLARED in the
// classes below; its body is in the second half of this header.  A plain include sees
// both, so a header-only consumer is unaffected.  The Python extension's bindings unit
// (py_exports.cpp) defines NWGRAD_ENGINE_EXTERN: it sees the declarations only and
// calls the bodies that engine_<type>.cpp compiles, one explicit instantiation per
// scalar type (NWGRAD_ENGINE_INSTANTIATE below).
//
// Why: gcc decides inlining against a whole-unit budget (--param inline-unit-growth), and
// the bindings unit used to hold the engine too, at that limit.  Any code a change added
// to the bindings made gcc outline hot engine calls elsewhere in the unit (Aligner::rat()
// in the inter-pair tracebacks, +0.5-1 % instructions on cells the change never touched;
// #43, #50, #52, #53 each hit it).  With the engine in units of its own, the bindings'
// code no longer enters the engine's inlining decisions at all.

template<class T = double>
class SeqPairBatchT {
public:
    template<GapModel G, AlignMode A> using Engine = BatchEngine<T, G, A>;
    using Var = std::variant<Engine<GapModel::Linear, AlignMode::Global>,
                             Engine<GapModel::Linear, AlignMode::Local>,
                             Engine<GapModel::Affine, AlignMode::Global>,
                             Engine<GapModel::Affine, AlignMode::Local>,
                             Engine<GapModel::Linear, AlignMode::Glocal>,
                             Engine<GapModel::Affine, AlignMode::Glocal>>;
    using SeqPair = SeqPairT<T>;

    // The pairs (seqs_a[i], seqs_b[i]) under an owned copy of params, validated and
    // encoded in parallel (a bad character throws).  tracks / gap_tracks: one entry per
    // pair, or empty (see BatchEngine's constructor); a batch with tracks or K > 1
    // parameter sets is tracked.  kernel: the Viterbi backend (kBackendAuto,
    // kBackendScalar or a SimdLevel).
    SeqPairBatchT(const std::vector<std::string_view>& seqs_a,
                  const std::vector<std::string_view>& seqs_b, const AlignParams& params,
                  GapModel gm, AlignMode am, GradMode gd, int n_threads = 0,
                  TracebackMode tb = TracebackMode::Default, int kernel = kBackendAuto,
                  const PairTracks& tracks = {}, const PairTracks& gap_tracks = {});
    // No pairs: a batch for streaming align().
    SeqPairBatchT(GapModel gm, AlignMode am, GradMode gd, int n_threads = 0,
                  TracebackMode tb = TracebackMode::Default);

    SeqPairBatchT(const SeqPairBatchT&)            = delete;
    SeqPairBatchT& operator=(const SeqPairBatchT&) = delete;
    SeqPairBatchT(SeqPairBatchT&&);
    SeqPairBatchT& operator=(SeqPairBatchT&&);
    ~SeqPairBatchT();

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

    // ── Pairs ────────────────────────────────────────────────────────────────

    size_t size() const noexcept;
    std::array<size_t, 3> inter_plan_counts(bool score_only = false);
    std::vector<std::vector<size_t>> inter_plan_groups(bool score_only = false);
    bool tracked() const noexcept;
    int num_sets() const noexcept;
    const Alphabet& alphabet() const;
    // How many pairs' guides are pending (computed on first use); for tests.  Compiled in
    // the soft-defer unit (see below its definition), not in the engine units.
    size_t guides_pending() const noexcept;
    SeqPair operator[](size_t i);

    // ── Batch operations  ──────────────────

    void set_params(const AlignParams& p);
    // The semi-global free ends (kSgFree* bits; mode glocal only, kSgGlocal its default).
    void set_free_ends(int f);
    int free_ends() const;
    double score_and_grad(bool keep_paths = false);
    double banded_grad(int bandwidth, bool keep_paths = false);
    void drop_paths();
    AlignParams compute_grad() const;
    std::vector<double> scores() const;
    std::vector<std::string> cigars(bool extended = false, bool soft_clip = false) const;
    AlignParams weighted_grad(const double* w, size_t n) const;
    void grads_into(double* matrices, double* gaps) const;
    BatchResult align(const std::vector<ProblemInstance>& problems, const AlignParams& params,
                      int band = 0, int kernel = kBackendAuto, const StreamTracks& st = {}) const;

    // ── Settings ─────────────────────────────────────────────────────────────
    const BatchSettings& settings() const;
    int n_threads() const { return settings().n_threads; }
    TracebackMode traceback() const { return settings().traceback; }
    TracebackMode traceback_resolved() const;
    int hb_cutoff() const { return settings().hb_cutoff; }
    bool rowwise_full() const { return settings().rowwise_full; }
    bool inter_fill() const { return settings().inter_fill; }
    SoftImpl soft_impl() const { return settings().soft_impl; }
    bool soft_f32() const { return settings().soft_f32; }
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

#define NWGRAD_BATCH_SETTERS(X) \
    X(set_n_threads, int) X(set_hb_cutoff, int) X(set_soft_temperature, double) \
    X(set_soft_impl, SoftImpl) X(set_soft_f32, bool) X(set_sorted_schedule, bool) \
    X(set_reserve_frac, double) X(set_long_cost_ratio, double) X(set_weight_lo, double) \
    X(set_weight_hi, double) X(set_profile, bool) X(set_track_counts, bool)
#define NWGRAD_BATCH_SETTER_DECL(NAME, TYPE) void NAME(TYPE v);
    NWGRAD_BATCH_SETTERS(NWGRAD_BATCH_SETTER_DECL)
#undef NWGRAD_BATCH_SETTER_DECL
    void set_fill(bool rowwise, bool inter);
    void set_soft_guide(bool lazy, bool posterior);

    using PhaseProfile = typename Engine<GapModel::Affine, AlignMode::Global>::PhaseProfile;
    std::vector<PhaseProfile> profile_out() const;

private:
    Var e_;
    GapModel gm_ = GapModel::Affine;
    AlignMode am_ = AlignMode::Global;
    GradMode gd_ = GradMode::Hard;
    static BatchSettings settings_(int n_threads, TracebackMode tb) {
        BatchSettings settings;
        settings.set_n_threads(n_threads);
        settings.traceback = tb;
        return settings;
    }
    // The engine of this problem type, built from `args` (its constructor's, after the
    // grad mode).
    template<class... Args>
    static Var make_engine_(GapModel gm, AlignMode am, GradMode gd, BatchSettings settings,
                            const Args&... args) {
        if (gm == GapModel::Linear) {
            if (am == AlignMode::Global) return Var(std::in_place_index<0>, settings, gd, args...);
            if (am == AlignMode::Glocal) return Var(std::in_place_index<4>, settings, gd, args...);
            return Var(std::in_place_index<1>, settings, gd, args...);
        }
        if (am == AlignMode::Global) return Var(std::in_place_index<2>, settings, gd, args...);
        if (am == AlignMode::Glocal) return Var(std::in_place_index<5>, settings, gd, args...);
        return Var(std::in_place_index<3>, settings, gd, args...);
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
        : own_(make_own_(a, b, params, gm, am, gd, kernel, tb, std::move(track), std::move(gap_track))),
          b_(own_.get()), i_(0) {}
    // View of pair i of `batch` (which must outlive the view).
    SeqPairT(Batch& batch, size_t i) : b_(&batch), i_(i) {}

    bool is_view() const noexcept { return !own_; }
    Batch& batch() noexcept { return *b_; }
    size_t index() const noexcept { return i_; }

    // ── Alignment (on the calling thread) ────────────────────────────────────
    double align(bool keep_paths = true);
    double banded_align(int bandwidth, bool keep_paths = true);
    std::pair<double, AlignParams> score_and_grad(bool keep_paths = true) {
        align(keep_paths);
        if (grad_mode() == GradMode::None)
            throw std::logic_error("nwgrad: grad_mode is None; construct with Hard or Soft to enable gradients");
        return {score(), grad()};
    }
    void drop_paths();

    // ── Results ──────────────────────────────────────────────────────────────
    double score() const;
    AlignParams grad() const;
    std::vector<uint16_t> track() const;
    std::vector<uint16_t> gap_track() const;
    bool tracked() const { return b_->tracked(); }
    std::pair<std::string, std::string> aligned() const;
    std::pair<std::vector<int64_t>, std::vector<int64_t>> coordinates() const;
    std::string cigar(bool extended = false, bool soft_clip = false) const;
    std::vector<int> guide_j();
    bool path_valid()  const;
    bool score_valid() const;
    bool grad_valid()  const;
    bool stored_path_valid() const;
    std::string seq_a() const;
    std::string seq_b() const;
    size_t len_a() const;
    size_t len_b() const;
    int kernel() const;
    GapModel gap_model() const { return b_->gap_model(); }
    AlignMode align_mode() const { return b_->align_mode(); }
    int free_ends() const { return b_->free_ends(); }
    void set_free_ends(int f) { standalone_("free_ends"); own_->set_free_ends(f); }
    GradMode grad_mode() const { return b_->grad_mode(); }
    TracebackMode traceback() const { return b_->traceback_resolved(); }
    const Alphabet& alphabet() const { return b_->alphabet(); }

    // ── Settings: a standalone pair's own; a view's belong to its batch ──────
    void set_params(const AlignParams& p) {
        standalone_("set_params");
        own_->set_params(p);
    }
    int hb_cutoff() const { return b_->hb_cutoff(); }
    void set_hb_cutoff(int v) { standalone_("hb_cutoff"); own_->set_hb_cutoff(v); }
    bool rowwise_full() const { return b_->rowwise_full(); }
    void set_rowwise_full(bool on) { standalone_("fill"); own_->set_fill(on, false); }
    SoftImpl soft_impl() const { return b_->soft_impl(); }
    void set_soft_impl(SoftImpl s) { standalone_("soft_impl"); own_->set_soft_impl(s); }
    bool soft_f32() const { return b_->soft_f32(); }
    void set_soft_f32(bool v) { standalone_("soft_precision"); own_->set_soft_f32(v); }
    double soft_temperature() const { return b_->soft_temperature(); }
    void set_soft_temperature(double t) { standalone_("soft_temperature"); own_->set_soft_temperature(t); }

private:
    static std::unique_ptr<Batch> make_own_(std::string_view a, std::string_view b, const AlignParams& params,
                                            GapModel gm, AlignMode am, GradMode gd, int kernel,
                                            TracebackMode tb, std::vector<uint16_t> track,
                                            std::vector<uint16_t> gap_track) {
        const bool tr = !track.empty() || !gap_track.empty();
        std::vector<std::vector<uint16_t>> tk{std::move(track)}, gk{std::move(gap_track)};
        auto own = std::make_unique<Batch>(std::vector<std::string_view>{a}, std::vector<std::string_view>{b},
                                           params, gm, am, gd, 1, tb, kernel, tr ? &tk : nullptr,
                                           tr ? &gk : nullptr);
        own->set_fill(false, false);   // one pair: nothing to share a vector with
        return own;
    }

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

#if defined(NWGRAD_ENGINE_EXTERN)
// Compiled in engine_<type>.cpp (see the note above SeqPairBatchT).
#define NWGRAD_ENGINE_EXTERN_TYPE(T) \
    extern template class SeqPairBatchT<T>; \
    extern template class SeqPairT<T>;
NWGRAD_ENGINE_EXTERN_TYPE(float)
NWGRAD_ENGINE_EXTERN_TYPE(double)
NWGRAD_ENGINE_EXTERN_TYPE(int16_t)
NWGRAD_ENGINE_EXTERN_TYPE(int32_t)
NWGRAD_ENGINE_EXTERN_TYPE(int64_t)
#undef NWGRAD_ENGINE_EXTERN_TYPE
#else
// ── The engine-facing member bodies ──────────────────────────────────────────
// `inline`, as they were when written in the class: gcc's inliner treats a function
// declared inline differently (max-inline-insns-single, not -auto).

template<class T>
inline SeqPairBatchT<T>::SeqPairBatchT(const std::vector<std::string_view>& seqs_a,
                                       const std::vector<std::string_view>& seqs_b,
                                       const AlignParams& params, GapModel gm, AlignMode am,
                                       GradMode gd, int n_threads, TracebackMode tb, int kernel,
                                       const PairTracks& tracks, const PairTracks& gap_tracks)
    : e_(make_engine_(gm, am, gd, settings_(n_threads, tb), seqs_a, seqs_b, params, kernel,
                      tracks, gap_tracks)),
      gm_(gm), am_(am), gd_(gd) {}
template<class T>
inline SeqPairBatchT<T>::SeqPairBatchT(GapModel gm, AlignMode am, GradMode gd, int n_threads,
                                       TracebackMode tb)
    : e_(make_engine_(gm, am, gd, settings_(n_threads, tb))), gm_(gm), am_(am), gd_(gd) {}
template<class T> inline SeqPairBatchT<T>::SeqPairBatchT(SeqPairBatchT&&) = default;
template<class T> inline SeqPairBatchT<T>& SeqPairBatchT<T>::operator=(SeqPairBatchT&&) = default;
template<class T> inline SeqPairBatchT<T>::~SeqPairBatchT() = default;

template<class T> inline size_t SeqPairBatchT<T>::size() const noexcept {
    return visit([](const auto& e) { return e.size(); });
}
template<class T> inline std::array<size_t, 3> SeqPairBatchT<T>::inter_plan_counts(bool score_only) {
    return visit([score_only](auto& e) { return e.inter_plan_counts(score_only); });
}
template<class T>
inline std::vector<std::vector<size_t>> SeqPairBatchT<T>::inter_plan_groups(bool score_only) {
    return visit([score_only](auto& e) { return e.inter_plan_groups(score_only); });
}
template<class T> inline bool SeqPairBatchT<T>::tracked() const noexcept {
    return visit([](const auto& e) { return e.tracked(); });
}
template<class T> inline int SeqPairBatchT<T>::num_sets() const noexcept {
    return visit([](const auto& e) { return e.num_sets(); });
}
template<class T> inline const Alphabet& SeqPairBatchT<T>::alphabet() const {
    return visit([](const auto& e) -> const Alphabet& { return e.alphabet(); });
}

template<class T> inline void SeqPairBatchT<T>::set_params(const AlignParams& p) {
    visit([&](auto& e) { e.set_params(p); });
}
template<class T> inline void SeqPairBatchT<T>::set_free_ends(int f) {
    visit([&](auto& e) { e.set_free_ends(f); });
}
template<class T> inline int SeqPairBatchT<T>::free_ends() const {
    return visit([](const auto& e) { return e.free_ends(); });
}
template<class T> inline double SeqPairBatchT<T>::score_and_grad(bool keep_paths) {
    return visit([&](auto& e) { return e.score_and_grad(keep_paths); });
}
template<class T> inline double SeqPairBatchT<T>::banded_grad(int bandwidth, bool keep_paths) {
    return visit([&](auto& e) { return e.banded_grad(bandwidth, keep_paths); });
}
template<class T> inline void SeqPairBatchT<T>::drop_paths() { visit([](auto& e) { e.drop_paths(); }); }
template<class T> inline AlignParams SeqPairBatchT<T>::compute_grad() const {
    return visit([](auto& e) { return e.compute_grad(); });
}
template<class T> inline std::vector<double> SeqPairBatchT<T>::scores() const {
    return visit([](const auto& e) { return e.scores(); });
}
template<class T>
inline std::vector<std::string> SeqPairBatchT<T>::cigars(bool extended, bool soft_clip) const {
    return visit([&](const auto& e) { return e.cigars(extended, soft_clip); });
}
template<class T> inline AlignParams SeqPairBatchT<T>::weighted_grad(const double* w, size_t n) const {
    return visit([&](const auto& e) { return e.weighted_grad(w, n); });
}
template<class T> inline void SeqPairBatchT<T>::grads_into(double* matrices, double* gaps) const {
    visit([&](const auto& e) { e.grads_into(matrices, gaps); });
}
template<class T>
inline BatchResult SeqPairBatchT<T>::align(const std::vector<ProblemInstance>& problems,
                                           const AlignParams& params, int band, int kernel,
                                           const StreamTracks& st) const {
    return visit([&](const auto& e) { return e.align_stream(problems, params, band, kernel, st); });
}

template<class T> inline const BatchSettings& SeqPairBatchT<T>::settings() const {
    return visit([](const auto& e) -> const BatchSettings& { return e.settings(); });
}
template<class T> inline TracebackMode SeqPairBatchT<T>::traceback_resolved() const {
    return visit([](const auto& e) { return e.traceback_resolved(); });
}
#define NWGRAD_BATCH_SETTER_DEF(NAME, TYPE) \
    template<class T> inline void SeqPairBatchT<T>::NAME(TYPE v) { visit([&](auto& e) { e.NAME(v); }); }
NWGRAD_BATCH_SETTERS(NWGRAD_BATCH_SETTER_DEF)
#undef NWGRAD_BATCH_SETTER_DEF
template<class T> inline void SeqPairBatchT<T>::set_fill(bool rowwise, bool inter) {
    visit([&](auto& e) { e.set_fill(rowwise, inter); });
}
template<class T> inline void SeqPairBatchT<T>::set_soft_guide(bool lazy, bool posterior) {
    visit([&](auto& e) { e.set_soft_guide(lazy, posterior); });
}
template<class T>
inline std::vector<typename SeqPairBatchT<T>::PhaseProfile> SeqPairBatchT<T>::profile_out() const {
    return visit([](const auto& e) {
        std::vector<PhaseProfile> v;
        for (const auto& p : e.profile_out) v.push_back({p.chunk_s, p.reserve_s, p.chunk_cells,
                                                          p.reserve_cells, p.chunk_tasks,
                                                          p.reserve_tasks, p.finish_s});
        return v;
    });
}

template<class T> inline double SeqPairT<T>::align(bool keep_paths) {
    e([&](auto& x) { x.score_and_grad_one(i_, keep_paths); });
    return score();
}
template<class T> inline double SeqPairT<T>::banded_align(int bandwidth, bool keep_paths) {
    e([&](auto& x) { x.banded_one(i_, bandwidth, keep_paths); });
    return score();
}
template<class T> inline void SeqPairT<T>::drop_paths() { e([&](auto& x) { x.drop_path(i_); }); }
template<class T> inline double SeqPairT<T>::score() const {
    return ce([&](const auto& x) { return x.score(i_); });
}
template<class T> inline AlignParams SeqPairT<T>::grad() const {
    return ce([&](const auto& x) { return x.grad(i_); });
}
template<class T> inline std::vector<uint16_t> SeqPairT<T>::track() const {
    return ce([&](const auto& x) { auto t = x.track(i_); return std::vector<uint16_t>(t.begin(), t.end()); });
}
template<class T> inline std::vector<uint16_t> SeqPairT<T>::gap_track() const {
    return ce([&](const auto& x) { auto t = x.gap_track(i_); return std::vector<uint16_t>(t.begin(), t.end()); });
}
template<class T> inline std::pair<std::string, std::string> SeqPairT<T>::aligned() const {
    return ce([&](const auto& x) { return x.aligned(i_); });
}
template<class T>
inline std::pair<std::vector<int64_t>, std::vector<int64_t>> SeqPairT<T>::coordinates() const {
    return ce([&](const auto& x) { return x.coordinates(i_); });
}
template<class T> inline std::string SeqPairT<T>::cigar(bool extended, bool soft_clip) const {
    return ce([&](const auto& x) { return x.cigar(i_, extended, soft_clip); });
}
template<class T> inline std::vector<int> SeqPairT<T>::guide_j() {
    return e([&](auto& x) { return x.guide_j(i_); });
}
template<class T> inline bool SeqPairT<T>::path_valid() const {
    return ce([&](const auto& x) { return x.path_valid(i_); });
}
template<class T> inline bool SeqPairT<T>::score_valid() const {
    return ce([&](const auto& x) { return x.score_valid(i_); });
}
template<class T> inline bool SeqPairT<T>::grad_valid() const {
    return ce([&](const auto& x) { return x.grad_valid(i_); });
}
template<class T> inline bool SeqPairT<T>::stored_path_valid() const {
    return ce([&](const auto& x) { return x.path_stored(i_); });
}
template<class T> inline std::string SeqPairT<T>::seq_a() const {
    return ce([&](const auto& x) { return x.seq_a(i_); });
}
template<class T> inline std::string SeqPairT<T>::seq_b() const {
    return ce([&](const auto& x) { return x.seq_b(i_); });
}
template<class T> inline size_t SeqPairT<T>::len_a() const {
    return ce([&](const auto& x) { return x.len_a(i_); });
}
template<class T> inline size_t SeqPairT<T>::len_b() const {
    return ce([&](const auto& x) { return x.len_b(i_); });
}
template<class T> inline int SeqPairT<T>::kernel() const {
    return ce([](const auto& x) { return x.kernel(); });
}
#endif  // NWGRAD_ENGINE_EXTERN

// The soft guide deferral's test counter: with the soft-defer code (batch_soft_defer.inl)
// in a header-only build, and in that unit itself (batch_soft_defer.cpp); the extension's
// engine units only declare it (NWGRAD_SOFT_DEFER_EXTERN), so their code stays main's.
#if !defined(NWGRAD_SOFT_DEFER_EXTERN) || defined(NWGRAD_SOFT_DEFER_UNIT)
template<class T> inline size_t SeqPairBatchT<T>::guides_pending() const noexcept {
    return visit([](const auto& e) { return e.guides_pending(); });
}
#endif
#undef NWGRAD_BATCH_SETTERS

// Default (double) aliases; Python binds the float32 classes as SeqPair / SeqPairBatch.
using SeqPairBatch = SeqPairBatchT<double>;
using SeqPair      = SeqPairT<double>;
