#pragma once

// ── Soft guide deferral: BatchEngine's soft full calls ─────────────────────
//
// Included at the end of batch_engine.hpp (header-only use), or compiled into its own
// unit, batch_soft_defer.cpp, for the Python extension (NWGRAD_SOFT_DEFER_EXTERN).  Why
// a unit of its own: on Alder Lake a kernel's speed depends on where the engine code
// that drives it sits mod 4 KiB (AGENTS.md, "Code placement"), and any new code in the
// bindings unit changes gcc's inlining there and moves that engine code.  With the soft
// calls' bookkeeping here, the bindings unit's hard, path and score-only engine code is
// main's.  The engine's own fills know nothing of it: a soft call reaches them with
// settings().soft_guide_lazy set when it defers (their "no Viterbi, guide pending" path)
// and fixes the per-pair state up afterwards (SoftCall).
//
// soft_guide "eager" with the guide deferred (the soft_guide_defer gate): a soft call
// leaves every pair's guide pending under a snapshot of the params it scored (gparams_),
// computed on first use (guide_j, banded_grad, banded_align) by the same hard fill the
// eager path runs, so it is the eager guide, bit for bit; a call that never asks for it
// skips the hard fill, 24-26 % of a double soft step's instructions on skynet (AGENTS.md,
// "The soft path").  Adaptive (gate 1, the default): when the batch's guides were read
// (at least half of them: guide_j, banded_grad) after each of the last kReadStreak full
// soft calls, the call runs the Viterbi itself, as eager did; any step whose guides go
// unread drops back to deferring.  Why: resolving deferred guides after the call is a
// second parallel pass over the batch, 4-9 % slower per step than the in-call fill when
// every guide is read (solace AVX-512, 8 threads, 79b72ad), while deferring wins 21-30 %
// when none is.  A streak of 2, not 1: with 1 (the gate-1 rule until 2026-11-13) a loop
// reading its guides every OTHER step filled them in the call exactly on the steps that
// did not read them (nighthaven: 0.98-1.02 of main, against 0.87-0.94 deferring).

#include <climits>

// A soft call's guides: computed in the call (0: keep_paths, posterior, or the deferral
// switched off or not wanted), lazy (kPending | kLazy: "lazy"), or deferred under the
// params scored (kPending: "eager").  The value is the pending state a pair the call
// finishes ends in.
template<class T, GapModel GM, AlignMode AM>
uint8_t BatchEngine<T, GM, AM>::soft_pend_(bool keep, bool read) const noexcept {
    if (grad_mode_ != GradMode::Soft || keep || s_.soft_guide_posterior) return 0;
    if (s_.soft_guide_lazy) return static_cast<uint8_t>(kPending | kLazy);
    const int d = soft_guide_defer();
    return d == 2 || (d == 1 && !read) ? static_cast<uint8_t>(kPending) : uint8_t(0);
}

// The bookkeeping around one soft call over pairs [lo, hi), which the engine's fills run
// unchanged:
// - while it runs, soft_guide_lazy is set when it defers (then they skip the Viterbi and
//   leave the guide pending), and every pair carries kLazy as a marker: finish_ drops it,
//   so afterwards a pair without it is one the call finished;
// - a finished pair's pending guide is lazy or deferred (pend); an unfinished one (the
//   call threw) keeps its state;
// - a call that computes its guides and throws: an unfinished pair whose guide it had
//   already stored (its first entry no longer the sentinel written here) keeps that
//   guide and is no longer pending -- else a later read would overwrite it with the
//   older snapshot's;
// - a deferring call that throws after replacing the snapshot (prev): its unfinished pairs
//   still pending under prev are resolved under it now, so each keeps exactly the guide it
//   had.  The exception path only.
template<class T, GapModel GM, AlignMode AM>
struct BatchEngine<T, GM, AM>::SoftCall {
    BatchEngine& e;
    const size_t lo, hi;
    const uint8_t pend;
    const bool lazy0;     // the setting, restored after the call
    const bool watch;     // the call stores guides: watch the pending ones
    std::vector<uint8_t> old;
    std::vector<int> g0;  // watched pairs' first guide entries

    SoftCall(BatchEngine& eng, size_t a, size_t b, uint8_t p, bool keep)
        : e(eng), lo(a), hi(b), pend(p), lazy0(eng.s_.soft_guide_lazy),
          watch(!p && !(eng.s_.soft_guide_posterior && !keep)), old(b - a) {
        if (pend) e.s_.soft_guide_lazy = true;
        for (size_t i = lo; i < hi; ++i) {
            uint8_t& f = e.flags_[i];
            old[i - lo] = f;
            if (watch && (f & kPending)) {
                int* g = e.guide_ptr_(i);
                g0.push_back(*g);
                *g = INT_MIN;
            }
            f |= kLazy;
        }
    }
    void done(bool threw, const std::shared_ptr<const OwnedParams>& prev) {
        e.s_.soft_guide_lazy = lazy0;
        std::vector<size_t> settle;
        size_t k = 0;
        for (size_t i = lo; i < hi; ++i) {
            uint8_t f = e.flags_[i];
            const uint8_t o = old[i - lo];
            if (!(f & kLazy)) {   // finished
                if (f & kPending) f = static_cast<uint8_t>((f & ~kLazy) | pend);
            } else {
                f = static_cast<uint8_t>((f & ~kLazy) | (o & kLazy));
                if (watch && (o & kPending)) {
                    int* g = e.guide_ptr_(i);
                    if (*g != INT_MIN) f &= static_cast<uint8_t>(~(kPending | kLazy));
                    else *g = g0[k];
                    ++k;
                }
                if (threw && (f & (kPending | kLazy)) == kPending) settle.push_back(i);
            }
            e.flags_[i] = f;
        }
        if (settle.empty() || prev == e.gparams_) return;
        auto cur = e.gparams_;
        e.gparams_ = prev;
        for (size_t i : settle) {
            if (prev) e.soft_resolve_(i, e.solo_work_());
            else e.flags_[i] &= static_cast<uint8_t>(~(kPath | kPending));
        }
        e.gparams_ = std::move(cur);
    }
};

template<class T, GapModel GM, AlignMode AM>
double BatchEngine<T, GM, AM>::soft_score_and_grad_(bool keep) {
    const size_t N = size();
    prepare_paths_(keep);
    ++full_calls_;   // guides may have moved: the banded grouping is stale
    tc_begin_full_();
    read_streak_ = guide_reads_ * 2 >= N ? read_streak_ + 1 : 0;
    guide_reads_ = 0;
    const uint8_t pend = soft_pend_(keep, read_streak_ >= kReadStreak);
    // Every pair this call finishes defers its guide under the params scored now.  A pair
    // it does not finish (it threw) keeps the state it had (SoftCall::done).
    auto prev = gparams_;
    if (pend == kPending) gparams_ = params_;
    SoftCall c(*this, 0, N, pend, keep);
    try {
        if (s_.inter_fill)           score_and_grad_inter_(keep);
        else if (s_.sorted_schedule) score_and_grad_sorted_(keep);
        else                         score_and_grad_dynamic_(keep);
    } catch (...) {
        c.done(true, prev);
        throw;
    }
    c.done(false, prev);
    // Every pair finished: no guide is pending under an older snapshot.
    if (pend != kPending) gparams_.reset();
    return sum_scores_();
}

// One soft pair, on the calling thread (SeqPair).  gparams_ is one snapshot for the whole
// batch: one pair may defer its guide only when that does not move the other pairs'
// pending guides to new params (a standalone pair, or a snapshot that IS the current
// params); otherwise its guide is computed now, as eagerly as before.
template<class T, GapModel GM, AlignMode AM>
void BatchEngine<T, GM, AM>::soft_one_(size_t i, bool keep) {
    prepare_paths_(keep, /*all=*/false);
    uint8_t pend = soft_pend_(keep, false);
    const bool own = size() == 1 || !gparams_ || gparams_ == params_;
    if (pend == kPending && !own) pend = 0;
    SoftCall c(*this, i, i + 1, pend, keep);
    try {
        run_full_(i, solo_work_(), keep);
    } catch (...) {
        c.done(true, gparams_);
        throw;
    }
    c.done(false, gparams_);
    if (pend == kPending) gparams_ = params_;
}

// banded_grad of a soft batch: its guides count as read, the deferred ones are resolved
// per inter-pair group before the pass (resolve_groups_), and lazy ones too while a
// snapshot is set (the pass's own resolve_guide_ takes gparams_ when there is one).
template<class T, GapModel GM, AlignMode AM>
void BatchEngine<T, GM, AM>::soft_banded_prep_() {
    guide_reads_ += size();
    resolve_groups_();
    if (!gparams_) return;
    std::vector<size_t> lazy;
    for (size_t i = 0; i < size(); ++i)
        if ((flags_[i] & (kPending | kLazy)) == (kPending | kLazy)) lazy.push_back(i);
    if (lazy.empty()) return;
    std::atomic<size_t> idx{0};
    auto worker = [&]() {
        WorkLease lease(*this);
        for (size_t k; (k = idx.fetch_add(1, std::memory_order_relaxed)) < lazy.size(); )
            soft_resolve_(lazy[k], *lease);
    };
    run_workers_(lazy.size(), worker);
}

// resolve_guide_ by the pair's own kind: a lazy guide (kLazy) under the params current
// now, a deferred one under gparams_.  The same hard fill the eager path runs, so a
// deferred guide is the eager one, bit for bit.  Only the worker that owns pair i
// resolves it, so no lock is needed.
template<class T, GapModel GM, AlignMode AM>
void BatchEngine<T, GM, AM>::soft_resolve_(size_t i, Work& w) {
    if (!guide_pending(i)) return;
    w.full.set_kernel(kernel_);
    problem_(w.full, i, (flags_[i] & kLazy) || !gparams_ ? *params_ : *gparams_, 0, {});
    w.full.compute_viterbi(w.buf);
    store_guide_(i, w.full.guide_j_from_viterbi(w.buf));
    flags_[i] &= static_cast<uint8_t>(~(kPending | kLazy | kStored));
}

// Deferred guides in bulk.  An eager soft call under fill "interpair" took its guides
// from the shared inter-pair fill, cheaper per cell than each pair's own fill (skynet
// sse4, one soft step with every guide used: 21-33 % more instructions resolving pair
// by pair, see AGENTS.md).  So a deferred guide is resolved with its whole plan group:
// one inter-pair fill under the snapshot, every snapshot-pending lane's guide from it --
// the eager call's fill and walk, so the eager guides, bit for bit.  Untracked batches
// with the shared fill on only (tracked lanes and own fills resolve per pair, which is
// what the eager call ran for them too).
template<class T, GapModel GM, AlignMode AM>
bool BatchEngine<T, GM, AM>::group_resolve_ok_() const noexcept {
    return s_.inter_fill && !tracked_ && gparams_ && grad_mode_ == GradMode::Soft;
}

// Pair i's pending guide, on the calling thread.  A snapshot guide resolves EVERY
// snapshot-pending guide of the batch at once, in parallel (resolve_all_snap_): a caller
// reading guide_j pair by pair (Python) otherwise computed them one at a time on its
// own thread, while main's eager call had filled them on n_threads -- solace AVX-512,
// 8 threads, every guide read after each step: 1.6-2.2x main's time per step.  Any
// one read thus pays for the batch's guides (what the eager call always paid); a step
// whose guides are never read still pays nothing.  A lazy guide follows the params
// current at its own first use, so it stays per pair.
template<class T, GapModel GM, AlignMode AM>
void BatchEngine<T, GM, AM>::resolve_caller_(size_t i) {
    if (snap_pending_(i)) {
        resolve_all_snap_();
        if (!guide_pending(i)) return;
    }
    soft_resolve_(i, solo_work_());
}

// Every snapshot-pending guide: the inter-pair groups (resolve_groups_), then the rest
// per pair, each on one worker (dynamic, as score_and_grad's own fills).
template<class T, GapModel GM, AlignMode AM>
void BatchEngine<T, GM, AM>::resolve_all_snap_() {
    resolve_groups_();
    std::vector<size_t> rest;
    for (size_t i = 0; i < size(); ++i)
        if (snap_pending_(i)) rest.push_back(i);
    if (rest.empty()) return;
    std::atomic<size_t> idx{0};
    auto worker = [&]() {
        WorkLease lease(*this);
        for (size_t k; (k = idx.fetch_add(1, std::memory_order_relaxed)) < rest.size(); )
            soft_resolve_(rest[k], *lease);
    };
    run_workers_(rest.size(), worker);
}

// Before a banded pass: every group with a snapshot-pending lane, in parallel (one worker
// a group, so each pair is written by one thread); the rest resolve per pair inside the
// pass as before.
template<class T, GapModel GM, AlignMode AM>
void BatchEngine<T, GM, AM>::resolve_groups_() {
    if (!group_resolve_ok_()) return;
    bool any = false;
    for (size_t i = 0; i < size(); ++i) any |= snap_pending_(i);
    if (!any) return;
    ensure_plan_();
    const size_t G = plan_.groups.size();
    std::atomic<size_t> idx{0};
    auto worker = [&]() {
        WorkLease lease(*this);
        for (size_t t; (t = idx.fetch_add(1, std::memory_order_relaxed)) < G; ) resolve_group_(t, *lease);
    };
    run_workers_(G, worker);
}

// Group t's snapshot-pending lanes from one inter-pair fill under gparams_; false (nothing
// done) when no lane is pending or the level does not share this fill.  The job is
// score_and_grad_inter_'s for the group, under the snapshot's params.
template<class T, GapModel GM, AlignMode AM>
bool BatchEngine<T, GM, AM>::resolve_group_(size_t t, Work& w) {
    const auto& elig = plan_.elig;
    const size_t s = plan_.groups[t].first, real = plan_.groups[t].second - s;
    bool any = false;
    for (size_t l = 0; l < real; ++l) any |= snap_pending_(elig[s + l]);
    const LevelKernels& K = level_kernels(plan_.backend);
    if (!any || !linear_fill_ok_(K)) return false;
    const int W = inter_w_(K);
    const OwnedParams& o = *gparams_;
    std::vector<const unsigned char*> a(W), b(W);
    std::vector<int> m(W), nb(W), bi(W), bj(W);
    std::vector<T> best(W), blkT;
    int M = 0, n = 0;
    bool ragged = false;
    for (int l = 0; l < W; ++l) {   // as score_and_grad_inter_: spare lanes repeat the last pair
        const size_t i = elig[s + std::min<size_t>(l, real - 1)];
        a[l] = codes_a(i).data(); b[l] = codes_b(i).data();
        m[l] = static_cast<int>(len_a(i)); nb[l] = static_cast<int>(len_b(i));
        M = std::max(M, m[l]); n = std::max(n, nb[l]);
        ragged |= nb[l] != nb[0];
    }
    InterJobT<T> job{};
    job.a = a.data(); job.m = m.data(); job.b = b.data(); job.n = n; job.M = M;
    job.align_mode = (AM == AlignMode::Local) ? 1 : (AM == AlignMode::Glocal ? 2 : 0);
    if (ragged) job.nb = nb.data();
    // block_for_, under the snapshot's params
    const T* pp = o.stack_or_null();
    if (!pp) cast_block_(o.p, blkT);
    set_inter_params_(job, o.p, blkT);
    if (pp) job.blk = pp;
    wire_inter_tables_(job, w.buf, static_cast<size_t>(M + 1) * (n + 1) * W);
    job.best = best.data(); job.best_i = bi.data(); job.best_j = bj.data();
    run_inter_fill_(K, job);
    const size_t stride = ragged ? static_cast<size_t>(n) + 1 : 0;
    for (size_t l = 0; l < real; ++l) {
        const size_t i = elig[s + l];
        if (!snap_pending_(i)) continue;
        w.full.set_kernel(kernel_);
        problem_(w.full, i, o, 0, {});
        w.full.adopt_interleaved(w.buf, W, static_cast<int>(l), best[l], bi[l], bj[l], stride);
        store_guide_(i, w.full.guide_j_from_viterbi(w.buf));
        flags_[i] &= static_cast<uint8_t>(~(kPending | kLazy | kStored));
    }
    return true;
}
