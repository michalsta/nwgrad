#pragma once

#include <exception>
#include <mutex>
#include <thread>
#include <vector>

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
