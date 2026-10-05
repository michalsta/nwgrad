#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "align_params.hpp"
#include "aligner.hpp"
#include "parallel.hpp"
#include "simd_levels.hpp"

struct ProblemInstance {
    std::string_view seq_a;
    std::string_view seq_b;
    std::vector<int> guide_j;  // empty → trivial diagonal guide (band around main diagonal)
};

struct BatchResult {
    std::vector<double> scores;
    AlignParams grad;

    explicit BatchResult(const Alphabet& alpha) : grad(alpha) {}
};

template<class T = double>
struct BatchAlignerT {
    // Viterbi precision T (float32 default in Python, double via BatchAlignerDouble).
    // Shadows the global ::DpBuffer; AlignParams/scores stay double.
    using DpBuffer = DpBufferT<T>;

    AlignParams params;
    int    band;       // 0 = full DP; > 0 = banded with this half-width
    GapModel  gap_model;
    AlignMode align_mode;
    enum class GradMode { None, Hard, Soft };
    GradMode grad_mode;
    int n_threads;
    // Which Viterbi fills the DP tables.  A runtime field, exactly like the three
    // above — the two kernels are bit-exact, so this never changes a result, only a
    // duration.  Keeping it off the template parameter list is what leaves the
    // DISPATCH macro below at four arms rather than eight.
    int kernel;   // Viterbi backend (kBackendAuto/kBackendScalar or a SimdLevel index)
    SoftImpl soft_impl = SoftImpl::Scaled;   // soft-path evaluation, see SoftImpl
    double soft_temperature = 1.0;           // see Aligner::set_soft_temperature
    // fill = "interpair": W problems per vector lane at once, as SeqPairBatch.fill —
    // bit-identical scores and hard gradients; see align_inter_().  A speed knob only.
    bool inter_fill = false;

    BatchAlignerT(AlignParams p, int band,
                 GapModel gm, AlignMode am, GradMode gd, int nt,
                 int k = kBackendAuto)
        : params(std::move(p)), band(band),
          gap_model(gm), align_mode(am), grad_mode(gd), n_threads(nt), kernel(k) {}

    BatchResult align(const std::vector<ProblemInstance>& problems) const {
        const size_t N = problems.size();
        BatchResult result(params.matrix.alphabet());
        result.scores.resize(N, 0.0);

        if (N == 0) return result;

        // dispatch_worker picks one AlignBand for the whole batch by looking at
        // problems[0].guide_j — AlignBand is a compile-time Aligner parameter,
        // so one worker loop can't switch between Full and GuideBanded per
        // problem. That is fine when band > 0 (every problem gets banded, with
        // an auto diagonal guide filling in for any empty guide_j) or when no
        // problem carries a guide_j at all. It silently drops every guide past
        // problems[0] when band == 0 and only *some* problems carry one — the
        // dropped problems still get the correct (full-DP) score, just not the
        // banding they were given, with no diagnostic. Reject that case here,
        // on the caller's thread, before any work is dispatched.
        if (band == 0) {
            bool any_guided = false, any_unguided = false;
            for (const auto& p : problems) {
                if (p.guide_j.empty()) any_unguided = true;
                else                   any_guided   = true;
            }
            if (any_guided && any_unguided)
                throw std::invalid_argument(
                    "nwgrad: mixed batch with band == 0 — some problems carry a "
                    "guide_j and others don't. All problems in one align() call "
                    "must either all supply a guide_j or none of them; split "
                    "into separate align() calls, or set band > 0 so an unguided "
                    "problem gets an automatic diagonal guide instead.");
        }

        if (inter_fill) {
            const int be = inter_backend_();
            if (be >= 0) { align_inter_(problems, result, be); return result; }
        }

        std::atomic<size_t> work_idx{0};
        std::mutex grad_mutex;

        auto worker = [&]() {
            AlignParams local_grad = AlignParams::zeros_like(params);
            dispatch_worker(problems, N, work_idx, result.scores, local_grad);
            if (grad_mode != GradMode::None) {
                std::lock_guard<std::mutex> lock(grad_mutex);
                result.grad += local_grad;
            }
        };

        int actual_threads = std::min<int>(n_threads, static_cast<int>(N));
        run_workers_guarded(actual_threads, worker);

        return result;
    }

private:
    // ── fill = "interpair" ──────────────────────────────────────────────────────
    //
    // The level a shared pass would run on, or -1 when this batch takes its own path
    // whatever the problems: an alphabet over 8 letters, the scalar backend, soft_impl
    // "log" or a banded soft batch (no shared pass for either), linear Global Viterbi
    // below 4 lanes (a measured loss at W=2, as in SeqPairBatch), linear banded (its own
    // fill is cheaper — see SeqPairBatch::banded_grad_inter_).
    bool banded_(const std::vector<ProblemInstance>& problems) const {
        return band > 0 || (!problems.empty() && !problems[0].guide_j.empty());
    }
    static int inter_w_(const LevelKernels& K) {
        if constexpr (std::is_same_v<T, double>) return K.inter_fill ? K.inter_w : 0;
        else                                     return K.inter_fill_f ? K.inter_w_f : 0;
    }
    int inter_backend_() const {
        const int backend = (kernel == kBackendAuto) ? global_default_backend() : kernel;
        if (backend < 0) return -1;
        const LevelKernels& K = level_kernels(backend);
        if (grad_mode == GradMode::Soft)
            return (K.inter_soft && K.inter_w > 0 && soft_impl != SoftImpl::Log && band == 0 &&
                    params.matrix.size() <= 8) ? backend : -1;
        if (inter_w_(K) <= 0) return -1;
        if (gap_model == GapModel::Linear && (band > 0 ||
            (align_mode == AlignMode::Global && inter_w_(K) < 4))) return -1;
        return backend;
    }

    // Problems grouped W at a time, sorted by len B then len A (a counting sort per
    // key — linear, since each align() brings new problems and nothing is cached).  The
    // rest run their own path in the same pass: an empty sequence, and an affine Global
    // Full pair longer than the Hirschberg cutoff (the default traceback splits those).
    // A guided batch with band 0 is GuideBanded too (dispatch_worker) and must stay so.
    void align_inter_(const std::vector<ProblemInstance>& problems, BatchResult& result,
                      int backend) const {
        const size_t N = problems.size();
        const LevelKernels& K = level_kernels(backend);
        const bool soft = grad_mode == GradMode::Soft;
        const int W = soft ? K.inter_w : inter_w_(K);
        const bool banded = banded_(problems);
        const int hb_cut = Aligner<GapModel::Affine, AlignMode::Global, AlignBand::Full, T>{}.hb_cutoff();
        const bool hb_case = !soft && !banded && gap_model == GapModel::Affine &&
                             align_mode == AlignMode::Global;
        std::vector<size_t> other, elig;
        size_t maxa = 0, maxb = 0;
        for (size_t i = 0; i < N; ++i) {
            const size_t la = problems[i].seq_a.size(), lb = problems[i].seq_b.size();
            if (la == 0 || lb == 0 || (hb_case && la > static_cast<size_t>(hb_cut))) {
                other.push_back(i); continue;
            }
            elig.push_back(i);
            maxa = std::max(maxa, la); maxb = std::max(maxb, lb);
        }
        // LSD counting sort: by len A, then (stable) by len B.
        auto csort = [&](auto key, size_t kmax) {
            std::vector<size_t> cnt(kmax + 2, 0), out(elig.size());
            for (size_t i : elig) ++cnt[key(i) + 1];
            for (size_t k = 1; k < cnt.size(); ++k) cnt[k] += cnt[k - 1];
            for (size_t i : elig) out[cnt[key(i)]++] = i;
            elig.swap(out);
        };
        csort([&](size_t i) { return problems[i].seq_a.size(); }, maxa);
        csort([&](size_t i) { return problems[i].seq_b.size(); }, maxb);
        std::vector<std::pair<size_t, size_t>> groups;
        // Cut every W, and where len B outgrows the group's shortest by more than the
        // padding SeqPairBatch allows (ragged B, InterJobT::nb).
        for (size_t s = 0; s < elig.size();) {
            const size_t lb = problems[elig[s]].seq_b.size();
            size_t e = s + 1;
            while (e < elig.size() && e - s < static_cast<size_t>(W) &&
                   problems[elig[e]].seq_b.size() * 4 <= lb * 5 + 16) ++e;
            groups.emplace_back(s, e);
            s = e;
        }

        std::atomic<size_t> idx{0};
        std::mutex grad_mutex;
        auto worker = [&]() {
            AlignParams local_grad = AlignParams::zeros_like(params);
#define DISPATCH(GM, AM) \
            if (gap_model == GapModel::GM && align_mode == AlignMode::AM) { \
                if (banded) inter_loop<GapModel::GM, AlignMode::AM, AlignBand::GuideBanded>( \
                    problems, elig, groups, other, idx, result.scores, local_grad, K, W); \
                else        inter_loop<GapModel::GM, AlignMode::AM, AlignBand::Full>( \
                    problems, elig, groups, other, idx, result.scores, local_grad, K, W); \
            }
            DISPATCH(Linear, Global)
            DISPATCH(Linear, Local)
            DISPATCH(Affine, Global)
            DISPATCH(Affine, Local)
#undef DISPATCH
            if (grad_mode != GradMode::None) {
                std::lock_guard<std::mutex> lock(grad_mutex);
                result.grad += local_grad;
            }
        };
        const size_t tasks = groups.size() + other.size();
        run_workers_guarded(std::min<int>(n_threads, static_cast<int>(std::max<size_t>(tasks, 1))),
                            worker);
    }

    template<GapModel GM, AlignMode AM, AlignBand AB>
    void inter_loop(const std::vector<ProblemInstance>& problems,
                    const std::vector<size_t>& elig,
                    const std::vector<std::pair<size_t, size_t>>& groups,
                    const std::vector<size_t>& other, std::atomic<size_t>& idx,
                    std::vector<double>& scores, AlignParams& local_grad,
                    const LevelKernels& K, int W) const {
        using Al = Aligner<GM, AM, AB, T>;
        std::vector<Al> al(W);
        for (Al& x : al) {
            x.set_kernel(kernel);
            x.set_soft_impl(soft_impl);
            x.set_soft_temperature(soft_temperature);
        }
        DpBuffer buf;
        const Alphabet& alpha = params.matrix.alphabet();
        std::vector<std::vector<uint8_t>> ae(W), be(W);
        std::vector<const unsigned char*> a(W), b(W);
        std::vector<int> m(W), bi(W), bj(W), blo, bhi, ulo, uhi, bri(W), brj(W), nb(W);
        std::vector<T> best(W), blkT;
        const bool lin = GM == GapModel::Linear;
        const bool soft = grad_mode == GradMode::Soft;
        // Soft: the weights, once per worker (params are fixed for the call).
        std::vector<double> es, slogz, scnt, sgap;
        DVec sscr;
        std::vector<int> siscr, sok;
        InterSoftJob sj{};
        bool soft_ok = soft && inter_soft_weights(params, soft_temperature, lin, es, sj);
        const size_t nn = static_cast<size_t>(params.matrix.size()) * params.matrix.size();
        if (!soft) {
            blkT.resize(nn);
            for (size_t k = 0; k < nn; ++k) blkT[k] = static_cast<T>(params.matrix.data()[k]);
        }
        // One problem on its own path — exactly work_loop's body.
        auto own = [&](size_t i) {
            Al& x = al[0];
            const auto& p = problems[i];
            encode_into(alpha, p.seq_a, ae[0], i, 'a');
            encode_into(alpha, p.seq_b, be[0], i, 'b');
            x.set_problem(ae[0], be[0], params, band, p.guide_j);
            if (grad_mode == GradMode::Hard) {
                x.compute_viterbi(buf); scores[i] = x.score(); x.hard_grad(buf, local_grad);
            } else if (grad_mode == GradMode::Soft) {
                x.compute_forward_back(buf); scores[i] = x.log_z(); x.soft_grad(buf, local_grad);
            } else {
                x.compute_viterbi(buf); scores[i] = x.score();
            }
        };
        const size_t G = groups.size(), tasks = G + other.size();
        while (true) {
            const size_t t = idx.fetch_add(1, std::memory_order_relaxed);
            if (t >= tasks) break;
            if (t >= G) { own(other[t - G]); continue; }
            const auto [s, e] = groups[t];
            const size_t real = e - s;
            if (soft && !soft_ok) { for (size_t k = s; k < e; ++k) own(elig[k]); continue; }
            int M = 0, n = 0;
            bool ragged = false;
            for (int l = 0; l < W; ++l) {
                // Spare lanes repeat the last problem; their results are dropped.
                const size_t lr = std::min<size_t>(l, real - 1);
                if (static_cast<size_t>(l) == lr) {
                    const size_t i = elig[s + lr];
                    encode_into(alpha, problems[i].seq_a, ae[l], i, 'a');
                    encode_into(alpha, problems[i].seq_b, be[l], i, 'b');
                }
                a[l] = ae[lr].data(); b[l] = be[lr].data();
                m[l] = static_cast<int>(ae[lr].size());
                nb[l] = static_cast<int>(be[lr].size());
                M = std::max(M, m[l]);
                n = std::max(n, nb[l]);
                ragged |= nb[l] != nb[0];
            }
            const size_t gstride = ragged ? static_cast<size_t>(n) + 1 : 0;
            if (soft) {
                const size_t need = inter_soft_scratch(n, M, W);
                if (sscr.size() < need) sscr.resize(need);
                siscr.resize(static_cast<size_t>(M + 1) * W);
                slogz.resize(W); scnt.resize(W * nn); sgap.resize(W * 4); sok.resize(W);
                sj.a = a.data(); sj.m = m.data(); sj.b = b.data(); sj.n = n; sj.M = M;
                sj.nb = ragged ? nb.data() : nullptr;
                sj.align_mode = (AM == AlignMode::Local) ? 1 : 0;
                sj.scratch = sscr.data(); sj.iscratch = siscr.data();
                sj.logz = slogz.data(); sj.counts = scnt.data(); sj.gaps = sgap.data();
                sj.ok = sok.data();
                K.inter_soft(sj);
                for (size_t l = 0; l < real; ++l) {
                    const size_t i = elig[s + l];
                    if (!sok[l]) { own(i); continue; }
                    scores[i] = slogz[l] * soft_temperature;
                    double* g = local_grad.matrix.data();
                    for (size_t k = 0; k < nn; ++k) g[k] += scnt[l * nn + k];
                    local_grad.gap_open_a += sgap[l * 4 + 0]; local_grad.gap_extend_a += sgap[l * 4 + 1];
                    local_grad.gap_open_b += sgap[l * 4 + 2]; local_grad.gap_extend_b += sgap[l * 4 + 3];
                }
                continue;
            }
            // Viterbi: each lane's problem on its own aligner (banded: its band rows).
            if constexpr (AB == AlignBand::GuideBanded) {
                blo.assign(static_cast<size_t>(M + 1) * W, 1);
                bhi.assign(static_cast<size_t>(M + 1) * W, 0);
                ulo.assign(M + 1, n + 1); uhi.assign(M + 1, 0);
                std::fill(bri.begin(), bri.end(), 0); std::fill(brj.begin(), brj.end(), 0);
            }
            for (size_t l = 0; l < real; ++l) {
                al[l].set_problem(ae[l], be[l], params, band, problems[elig[s + l]].guide_j);
                if constexpr (AB == AlignBand::GuideBanded)
                    al[l].banded_lane_rows(W, static_cast<int>(l), M, blo.data(), bhi.data(),
                                           ulo.data(), uhi.data(), bri[l], brj[l]);
            }
            const size_t sz = static_cast<size_t>(M + 1) * (n + 1) * W;
            InterJobT<T> job{};
            job.a = a.data(); job.m = m.data(); job.b = b.data(); job.n = n; job.M = M;
            job.align_mode = (AM == AlignMode::Local) ? 1 : 0;
            if (ragged) job.nb = nb.data();
            job.blk = blkT.data(); job.nalpha = params.matrix.size();
            job.go_a = static_cast<T>(params.gap_open_a); job.ge_a = static_cast<T>(params.gap_extend_a);
            job.go_b = static_cast<T>(params.gap_open_b); job.ge_b = static_cast<T>(params.gap_extend_b);
            job.linear = lin ? 1 : 0;
            if (lin) { if (buf.H.size() < sz) buf.H.resize(sz); job.VM = buf.H.data(); }
            else {
                if (buf.VM.size() < sz) { buf.VM.resize(sz); buf.VX.resize(sz); buf.VY.resize(sz); }
                job.VM = buf.VM.data(); job.VX = buf.VX.data(); job.VY = buf.VY.data();
            }
            job.best = best.data(); job.best_i = bi.data(); job.best_j = bj.data();
            if constexpr (AB == AlignBand::GuideBanded) {
                job.blo = blo.data(); job.bhi = bhi.data();
                job.ulo = ulo.data(); job.uhi = uhi.data();
                job.bri = bri.data(); job.brj = brj.data();
            }
            if constexpr (std::is_same_v<T, double>) K.inter_fill(job);
            else                                     K.inter_fill_f(job);
            for (size_t l = 0; l < real; ++l) {
                const size_t i = elig[s + l];
                al[l].adopt_interleaved(buf, W, static_cast<int>(l), best[l], bi[l], bj[l], gstride);
                scores[i] = al[l].score();
                if (grad_mode == GradMode::Hard) al[l].hard_grad(buf, local_grad);
            }
        }
    }

    // Validate and encode one sequence into `out`.  An out-of-alphabet character
    // throws, naming the pair and which of the two sequences it was in — the
    // aligner's own message could only name a position in an anonymous string.
    // The throw happens on a worker thread and is rethrown to the caller by
    // run_workers_guarded().
    static void encode_into(const Alphabet& alpha, std::string_view s,
                            std::vector<uint8_t>& out, size_t pair_idx, char which) {
        out.clear();
        out.reserve(s.size());
        for (size_t k = 0; k < s.size(); ++k) {
            int i = alpha.index_of(s[k]);
            if (i < 0) {
                std::string msg = "nwgrad: pair ";
                msg += std::to_string(pair_idx);
                msg += ", sequence ";
                msg += which;
                msg += ": character '";
                msg += s[k];
                msg += "' at position " + std::to_string(k) +
                       " is not in alphabet \"" + alpha.symbols() + "\"";
                throw std::invalid_argument(msg);
            }
            out.push_back(static_cast<uint8_t>(i));
        }
    }

    template<GapModel GM, AlignMode AM, AlignBand AB>
    void work_loop(
        const std::vector<ProblemInstance>& problems,
        size_t N,
        std::atomic<size_t>& work_idx,
        std::vector<double>& scores,
        AlignParams& local_grad) const
    {
        Aligner<GM, AM, AB, T> al;
        al.set_kernel(kernel);
        al.set_soft_impl(soft_impl);
        al.set_soft_temperature(soft_temperature);
        DpBuffer buf;  // reused across iterations; grows to the largest pair seen
        const Alphabet& alpha = params.matrix.alphabet();
        // Encoding buffers, reused across iterations: the batch never
        // materialises all N encoded sequences at once.
        std::vector<uint8_t> a_enc, b_enc;
        while (true) {
            size_t idx = work_idx.fetch_add(1, std::memory_order_relaxed);
            if (idx >= N) break;
            const auto& p = problems[idx];
            encode_into(alpha, p.seq_a, a_enc, idx, 'a');
            encode_into(alpha, p.seq_b, b_enc, idx, 'b');
            al.set_problem(a_enc, b_enc, params, band, p.guide_j);

            if (grad_mode == GradMode::Hard) {
                al.compute_viterbi(buf);
                scores[idx] = al.score();
                al.hard_grad(buf, local_grad);
            } else if (grad_mode == GradMode::Soft) {
                al.compute_forward_back(buf);
                scores[idx] = al.log_z();
                al.soft_grad(buf, local_grad);
            } else {
                al.compute_viterbi(buf);
                scores[idx] = al.score();
            }
        }
    }

    void dispatch_worker(
        const std::vector<ProblemInstance>& problems,
        size_t N,
        std::atomic<size_t>& work_idx,
        std::vector<double>& scores,
        AlignParams& local_grad) const
    {
        // Safe to decide from problems[0] alone: align() has already rejected any
        // batch that mixes guided and unguided problems under band == 0, so every
        // problem here agrees with problems[0] on whether it carries a guide_j.
        const bool banded = (band > 0) || (!problems.empty() && !problems[0].guide_j.empty());
#define DISPATCH(GM, AM) \
        if (gap_model == GapModel::GM && align_mode == AlignMode::AM) { \
            if (banded) work_loop<GapModel::GM, AlignMode::AM, AlignBand::GuideBanded>(problems, N, work_idx, scores, local_grad); \
            else        work_loop<GapModel::GM, AlignMode::AM, AlignBand::Full>       (problems, N, work_idx, scores, local_grad); \
            return; \
        }
        DISPATCH(Linear, Global)
        DISPATCH(Linear, Local)
        DISPATCH(Affine, Global)
        DISPATCH(Affine, Local)
#undef DISPATCH
    }
};

// Default (double) alias; Python binds BatchAlignerT<float> as `BatchAligner`.
using BatchAligner = BatchAlignerT<double>;
