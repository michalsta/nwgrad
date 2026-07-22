#pragma once

#include <exception>
#include <fstream>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(__APPLE__)
#  include <sys/sysctl.h>
#endif

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
        std::set<std::pair<int, int>> cores;
        unsigned int hw = std::thread::hardware_concurrency();
        for (unsigned int i = 0; i < hw; ++i) {
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
#endif
    } catch (...) {
        // Topology discovery must never be the thing that breaks a batch.
    }
    return 0;
}

// Physical cores if discoverable, else the logical count, else 1.
inline int default_thread_count() noexcept {
    static const int cached = [] {
        int p = physical_cores();
        if (p > 0) return p;
        unsigned int hw = std::thread::hardware_concurrency();
        return hw > 0 ? static_cast<int>(hw) : 1;
    }();
    return cached;
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

    std::vector<std::thread> threads;
    threads.reserve(static_cast<size_t>(n_threads - 1));
    for (int t = 0; t < n_threads - 1; ++t)
        threads.emplace_back(guarded);
    guarded();
    for (auto& t : threads) t.join();

    if (first_err) std::rethrow_exception(first_err);
}
