#pragma once

#include <algorithm>
#include <atomic>
#include <cfenv>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(__APPLE__)
#  include <sys/sysctl.h>
#endif
#if defined(__linux__)
#  include <sched.h>
#endif
#if !defined(_WIN32)
#  include <pthread.h>
#endif
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64)
#  include <immintrin.h>
#endif
#if defined(_WIN32)
// Lean and without the min/max macros, which would break every std::min / std::max in
// a TU that includes this header.  Both are restored to the includer's own setting.
#  if !defined(NOMINMAX)
#    define NOMINMAX
#    define NWGRAD_UNDEF_NOMINMAX
#  endif
#  if !defined(WIN32_LEAN_AND_MEAN)
#    define WIN32_LEAN_AND_MEAN
#    define NWGRAD_UNDEF_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#  if defined(NWGRAD_UNDEF_NOMINMAX)
#    undef NOMINMAX
#    undef NWGRAD_UNDEF_NOMINMAX
#  endif
#  if defined(NWGRAD_UNDEF_LEAN_AND_MEAN)
#    undef WIN32_LEAN_AND_MEAN
#    undef NWGRAD_UNDEF_LEAN_AND_MEAN
#  endif

// The process affinity mask, when the machine has a single processor group (<= 64
// logical CPUs) — the only case GetProcessAffinityMask describes fully.  0 when it is
// unknown or there is more than one group.
inline DWORD_PTR nwgrad_win_affinity_mask() noexcept {
    if (GetActiveProcessorGroupCount() != 1) return 0;
    DWORD_PTR proc = 0, sys = 0;
    if (!GetProcessAffinityMask(GetCurrentProcess(), &proc, &sys)) return 0;
    return proc;
}
#endif

// The CPU ids this process may run on (Linux: its affinity mask, so taskset, cpusets
// and container pinning are respected, and ids need not be contiguous); empty when
// unknown, and always empty elsewhere.  Windows: the process affinity mask, on a
// machine with one processor group.
inline std::vector<int> allowed_cpus() noexcept {
    std::vector<int> ids;
#if defined(_WIN32)
    try {
        const DWORD_PTR mask = nwgrad_win_affinity_mask();
        for (int i = 0; i < static_cast<int>(8 * sizeof mask); ++i)
            if ((mask >> i) & 1) ids.push_back(i);
    } catch (...) {
        ids.clear();
    }
#elif defined(__linux__)
    try {
        cpu_set_t set;
        CPU_ZERO(&set);
        if (sched_getaffinity(0, sizeof set, &set) == 0)
            for (int i = 0; i < CPU_SETSIZE; ++i)
                if (CPU_ISSET(i, &set)) ids.push_back(i);
    } catch (...) {
        ids.clear();
    }
#endif
    return ids;
}

// Number of PHYSICAL cores, or 0 if it cannot be determined.
//
// std::thread::hardware_concurrency() reports LOGICAL processors, and for this DP
// that is the wrong number.  The kernel is stall-bound, not compute-bound — perf
// counters show IPC collapsing 2.41 -> 0.38 at constant instruction count once the
// tables stop fitting cache — so two SMT siblings on one core halve each other's
// L1/L2 and both lose.  Measured on the human proteome (23.9 Gcells, sorted
// schedule, min of 4 runs with the thread list walked forward and reversed):
//
//   host        topology        optimum   physical   logical (hardware_concurrency)
//   spot-2      8 core, no SMT     8       1.000x     1.000x
//   solace      8 phys / 16 log    8       1.000x     1.125x
//   nighthaven  6 phys / 12 log    4       1.060x     1.439x
//   skynet(VM)  60 vCPU           45       1.045x     1.045x
//
// So physical count is within 6% of the measured optimum everywhere, while the
// logical count costs up to 44%.  It is NOT always exactly optimal — nighthaven
// peaks at 4 of its 6 cores, because a 2-channel desktop part saturates DRAM
// before it runs out of cores — but the remaining error sits on a plateau flat
// enough not to chase.  Auto-tuning was tried and rejected: probing the real
// kernel needs a multi-Gcell sample to stop favouring low thread counts, cost
// ~49s, and still only landed somewhere in the plateau.  Estimating from system
// info is worse: DIMM/channel data needs root, VMs fabricate the topology
// (skynet advertises 60 single-core sockets each with a private 16 MB L3), and
// nighthaven and solace are indistinguishable in every unprivileged field yet
// want 4 and 8 threads.  Callers who care can always pass n_threads explicitly.
inline int physical_cores() noexcept {
    try {
#if defined(__linux__)
        // Count distinct (package, core) pairs.  Unprivileged; absent in some
        // containers and on some VMs, in which case we fall through to 0.
        // Only CPUs in the affinity mask count: under `taskset -c 0` this is 1, not the
        // host's core count.
        std::set<std::pair<int, int>> cores;
        std::vector<int> ids = allowed_cpus();
        if (ids.empty())
            for (unsigned int i = 0; i < std::thread::hardware_concurrency(); ++i)
                ids.push_back(static_cast<int>(i));
        for (int i : ids) {
            const std::string base =
                "/sys/devices/system/cpu/cpu" + std::to_string(i) + "/topology/";
            std::ifstream pkg(base + "physical_package_id");
            std::ifstream core(base + "core_id");
            int p = 0, c = 0;
            if ((pkg >> p) && (core >> c)) cores.emplace(p, c);
        }
        if (!cores.empty()) return static_cast<int>(cores.size());
#elif defined(__APPLE__)
        int n = 0;
        size_t sz = sizeof(n);
        if (sysctlbyname("hw.physicalcpu", &n, &sz, nullptr, 0) == 0 && n > 0)
            return n;
#elif defined(_WIN32)
        // One RelationProcessorCore record per physical core; on a single-group machine
        // only cores with a logical CPU in the process affinity mask count (the Linux
        // branch's taskset rule, for `start /affinity`).
        DWORD len = 0;
        GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
        if (len > 0 && GetLastError() == ERROR_INSUFFICIENT_BUFFER) {
            std::vector<unsigned char> buf(len);
            if (GetLogicalProcessorInformationEx(
                    RelationProcessorCore,
                    reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data()), &len)) {
                const DWORD_PTR mask = nwgrad_win_affinity_mask();
                int n = 0;
                for (DWORD off = 0; off < len;) {
                    const auto* rec =
                        reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data() + off);
                    if (rec->Size == 0) break;
                    if (rec->Relationship == RelationProcessorCore &&
                        (mask == 0 || (rec->Processor.GroupMask[0].Mask & mask) != 0))
                        ++n;
                    off += rec->Size;
                }
                if (n > 0) return n;
            }
        }
#endif
    } catch (...) {
        // Topology discovery must never be the thing that breaks a batch.
    }
    return 0;
}

// Physical cores (among the CPUs this process may use) if discoverable, else the
// allowed logical count, else hardware_concurrency(), else 1.  Computed once: a later
// change of affinity is not seen.
inline int default_thread_count() noexcept {
    static const int cached = [] {
        int p = physical_cores();
        if (p > 0) return p;
        const size_t allowed = allowed_cpus().size();
        if (allowed > 0) return static_cast<int>(allowed);
        unsigned int hw = std::thread::hardware_concurrency();
        return hw > 0 ? static_cast<int>(hw) : 1;
    }();
    return cached;
}

// Whether n_threads threads must share physical cores (hyperthreads): more threads than
// physical_cores(), computed once.  False where the topology is unknown.
inline bool threads_share_cores(int n_threads) noexcept {
    static const int p = physical_cores();
    return p > 0 && n_threads > p;
}

// ── The worker pool ───────────────────────────────────────────────────────────
//
// A process-wide set of parked std::threads that run_workers_guarded() hands each
// multi-threaded call to, instead of spawning and joining n_threads - 1 threads per
// call.  Spawn + join is what a call of a few hundred short pairs paid most of its time
// for: skynet (VM), 256 miRNA pairs, 16 threads: 1178 us a call against 159 us on one
// thread; a standalone spawn+join loop costs 87 / 184 / 488 / 1105 us at 2 / 4 / 8 / 16
// threads there, a condition-variable pool waking the same threads 28 / 38 / 63 / 98 us
// (review-proposals-r2.md, item 4).  The work itself does not change: every batch worker
// already drains shared atomic counters (or claims chunks by an atomic index), so which
// OS thread runs a worker, and how many do, never changes a result.
//
// Design, and why:
// - Lazy: the first call with n_threads > 1 creates it; it grows to the largest
//   n_threads - 1 ever asked for (the caller is always one of the workers) and never
//   shrinks.  A failed thread start stops the growth and the call runs on the workers it
//   has (fewer threads, the same work: run_workers_guarded's old degrade-on-failure).
// - One condition variable PER WORKER: a call wakes exactly the k - 1 workers it uses.
//   One shared notify_all woke every parked worker of a 60-thread pool for a 4-thread
//   call.  Workers block between calls, never spin (a spinning pool steals the cores
//   other processes and the user's own threads need: the OpenBLAS lesson, benchmarks.md).
// - The caller joins on a counter: a short spin first (the last workers usually finish
//   within microseconds of the caller's own share), then a condition variable.
// - ONE call at a time: the pool is taken with try_lock for the whole call.  A second
//   concurrent call (another C++ or Python thread, or a nested call from inside a worker)
//   does not queue behind it; it runs on spawned threads exactly as before the pool, so
//   two concurrent calls still get 2 x their threads, never a deadlock, never a wait.
// - The floating-point environment travels with the job: a spawned thread inherited the
//   caller's at spawn, so each job carries the caller's (fegetenv) and the worker
//   installs it (fesetenv) before running, and the result cannot depend on which thread
//   ran it.  (FlushDenormalsScope inside a kernel restores on exit, so no state leaks
//   between jobs that way either.)
// - fork(): a child has only the forking thread, so the parked workers do not exist
//   there.  A pthread_atfork child handler drops (leaks, never destroys: its mutexes may
//   have been held at the fork) the pool, and the child's first call builds a new one.
// - Exit: the workers are detached and the pool is never destroyed, so nothing joins at
//   exit and nothing can hang there; parked workers die with the process.  (Joining in a
//   static destructor would hang an exit taken while a call is in flight on another
//   thread, and would run after the Python runtime is gone.)
// - Thread-locals in the kernels (inter_kernel's scratch vectors, the batch's Touched)
//   now live as long as the worker: their capacity is kept between calls, as the
//   calling thread's always was.
// - Switch: NWGRAD_THREAD_POOL=0 (read once, at the first multi-threaded call) or
//   nwgrad._tuning.set_thread_pool(False) restores spawn-per-call, for A/B.  n_threads = 1
//   never touches the pool.
namespace thread_pool_detail {

struct Slot {
    std::mutex m;
    std::condition_variable cv;
    void (*fn)(void*) = nullptr;   // non-null: a job is posted
    void* ctx = nullptr;
};

struct Pool {
    std::mutex call_mu;                         // held (try_lock) for a whole call
    std::vector<std::unique_ptr<Slot>> slots;   // grows under call_mu; a slot is never freed
    std::atomic<int> remaining{0};              // workers of the current call not yet done
    std::mutex done_m;
    std::condition_variable done_cv;
    std::fenv_t env{};                          // the current call's FP environment
};

// The pool (null until the first multi-threaded call, and again in a forked child).
inline std::atomic<Pool*> g_pool{nullptr};
// -1: not read yet (the environment decides at the first call), 0 off, 1 on.
inline std::atomic<int> g_enabled{-1};
// Set on a thread while it runs a pool call (caller or worker): a nested call from
// inside it must not try_lock a mutex its own thread may hold.
inline thread_local bool tl_in_pool = false;

inline void cpu_relax() noexcept {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64)
    _mm_pause();
#elif defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
    __asm__ __volatile__("yield");
#endif
}

inline void worker_main(Pool* p, Slot* s) noexcept {
    tl_in_pool = true;
    for (;;) {
        void (*fn)(void*);
        void* ctx;
        {
            std::unique_lock<std::mutex> lk(s->m);
            s->cv.wait(lk, [s] { return s->fn != nullptr; });
            fn = s->fn;
            ctx = s->ctx;
            s->fn = nullptr;
        }
        std::fesetenv(&p->env);
        fn(ctx);   // noexcept: run_workers_guarded's guarded wrapper
        if (p->remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            // Under done_m, so a caller between its predicate check and its wait cannot
            // miss it.  The pool outlives every call, so touching it here is safe even
            // when the caller has already returned (it saw 0 while spinning).
            std::lock_guard<std::mutex> g(p->done_m);
            p->done_cv.notify_one();
        }
    }
}

inline void after_fork_child() noexcept {
    // Leaked on purpose: its mutexes may have been held by threads that no longer exist.
    g_pool.store(nullptr, std::memory_order_relaxed);
}

inline Pool* get_pool() {
    Pool* p = g_pool.load(std::memory_order_acquire);
    if (p) return p;
#if !defined(_WIN32)
    static const bool registered = (pthread_atfork(nullptr, nullptr, &after_fork_child), true);
    (void)registered;
#endif
    auto fresh = std::make_unique<Pool>();
    if (g_pool.compare_exchange_strong(p, fresh.get(), std::memory_order_acq_rel))
        return fresh.release();   // never destroyed (see above)
    return p;                     // another thread's pool won
}

inline bool enabled() noexcept {
    int e = g_enabled.load(std::memory_order_relaxed);
    if (e < 0) {
        const char* v = std::getenv("NWGRAD_THREAD_POOL");
        e = (v && std::strcmp(v, "0") == 0) ? 0 : 1;
        int expected = -1;
        if (!g_enabled.compare_exchange_strong(expected, e, std::memory_order_relaxed)) e = expected;
    }
    return e != 0;
}

// Run fn(ctx) on n_threads threads, the caller included, on the pool.  False (nothing
// run) when the pool is off, busy with another call, or this thread is inside one: the
// caller then spawns threads as before.  fn must not throw.
inline bool run(int n_threads, void (*fn)(void*), void* ctx) {
    if (n_threads < 2 || tl_in_pool || !enabled()) return false;
    Pool* p;
    try { p = get_pool(); } catch (...) { return false; }
    std::unique_lock<std::mutex> call(p->call_mu, std::try_to_lock);
    if (!call.owns_lock()) return false;
    tl_in_pool = true;
    struct Clear { ~Clear() { tl_in_pool = false; } } clear_flag;

    const size_t want = static_cast<size_t>(n_threads - 1);
    try {
        // reserve first: once a thread runs on a slot, recording the slot must not throw
        // (a throwing push_back would free the slot under the live worker).
        p->slots.reserve(want);
        while (p->slots.size() < want) {
            auto s = std::make_unique<Slot>();
            std::thread(worker_main, p, s.get()).detach();
            p->slots.push_back(std::move(s));
        }
    } catch (...) {
        // Out of threads (or memory): run on the workers that exist, as the spawn path
        // runs on the threads that started.  Every worker drains shared counters, so
        // fewer threads still do all the work.
    }
    const size_t k = std::min(want, p->slots.size());
    std::fegetenv(&p->env);
    p->remaining.store(static_cast<int>(k), std::memory_order_relaxed);
    for (size_t i = 0; i < k; ++i) {
        Slot& s = *p->slots[i];
        {
            std::lock_guard<std::mutex> g(s.m);
            s.fn = fn;
            s.ctx = ctx;
        }
        s.cv.notify_one();
    }
    fn(ctx);
    if (k == 0) return true;
    for (int spin = 0; spin < 4096; ++spin) {
        if (p->remaining.load(std::memory_order_acquire) == 0) return true;
        cpu_relax();
    }
    std::unique_lock<std::mutex> lk(p->done_m);
    p->done_cv.wait(lk, [p] { return p->remaining.load(std::memory_order_acquire) == 0; });
    return true;
}

}  // namespace thread_pool_detail

// The pool's switch (nwgrad._tuning.set_thread_pool): false = spawn threads per call,
// as before the pool.  Not for flipping while a call is in flight on another thread
// (harmless, but that call keeps the mode it started with).
inline void set_thread_pool_enabled(bool on) noexcept {
    thread_pool_detail::g_enabled.store(on ? 1 : 0, std::memory_order_relaxed);
}
inline bool thread_pool_enabled() noexcept { return thread_pool_detail::enabled(); }
// Parked workers in this process's pool (0 before the first multi-threaded call, and in
// a forked child until its first).  For tests.
inline int thread_pool_size() noexcept {
    thread_pool_detail::Pool* p = thread_pool_detail::g_pool.load(std::memory_order_acquire);
    if (!p) return 0;
    std::lock_guard<std::mutex> g(p->call_mu);
    return static_cast<int>(p->slots.size());
}

// Run `worker` on `n_threads` threads, one of which is the calling thread.
//
// An exception escaping a std::thread's callable calls std::terminate, which
// aborts the process with no diagnostic — from Python, the interpreter dies
// without a traceback.  The workers here can throw: a SeqPair whose
// precondition is unmet, an AlignParams accumulated across mismatched
// alphabets.  So each worker body is guarded, the first exception is captured,
// and it is rethrown on the calling thread once every worker has joined.
//
// A worker that throws stops consuming work; the others drain the remaining
// items and may throw in turn.  Only the first exception is kept, and any
// results the workers produced must be treated as incomplete once one is
// rethrown.
// Spawn n_threads - 1 threads running `guarded`, run it on the caller too, join: the path
// before the pool, and the fallback when the pool is off, busy or nested.
template<typename Guarded>
inline void spawn_workers_(int n_threads, Guarded& guarded) {
    // A launch can fail (std::system_error when the process or user is out of
    // threads, bad_alloc).  Unguarded, that exception unwound through `threads`
    // while earlier workers were still joinable, and a joinable std::thread's
    // destructor calls std::terminate — the interpreter died with no traceback.
    // Instead, stop launching and carry on with the workers that did start: every
    // worker drains a shared queue (the sorted scheduler sweeps up chunks whose
    // owner never started), so fewer threads still means all the work, done
    // correctly.  The reserve() is inside the try for the same reason — it
    // allocates, and before it no thread exists to orphan.
    std::vector<std::thread> threads;
    try {
        threads.reserve(static_cast<size_t>(n_threads - 1));
        for (int t = 0; t < n_threads - 1; ++t)
            threads.emplace_back(guarded);
    } catch (...) {
        // Fall through with however many started — possibly none, in which case
        // the calling thread below does all of it.
    }
    guarded();
    for (auto& t : threads) t.join();
}

template<typename Worker>
inline void run_workers_guarded(int n_threads, Worker&& worker) {
    if (n_threads < 1) n_threads = 1;

    std::mutex         err_mutex;
    std::exception_ptr first_err;

    auto guarded = [&]() noexcept {
        try {
            worker();
        } catch (...) {
            std::lock_guard<std::mutex> lock(err_mutex);
            if (!first_err) first_err = std::current_exception();
        }
    };

    if (n_threads == 1) {
        guarded();
    } else if (!thread_pool_detail::run(
                   n_threads, [](void* c) { (*static_cast<decltype(guarded)*>(c))(); }, &guarded)) {
        spawn_workers_(n_threads, guarded);
    }
    if (first_err) std::rethrow_exception(first_err);
}

