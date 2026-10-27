// Shared scaffolding of the C/C++ drivers: the same CLI and protocol as bench/arm.py.
//
//   driver --fixture F.tsv --problem P.txt --gap affine|linear --mode global|local
//          --out score|path|hard|soft --dtype int16|... --threads N --reps R
//          --deadline S --scores-out file --variant V
//
// Setup is untimed; one warm-up pass, then R timed passes, best reported.  A per-pair
// tool stops a pass at the first pair past --deadline ("timeout").  With --threads > 1
// and no threading of its own, pairs are handed out to N std::threads in chunks from an
// atomic counter ("harness_parallel").  Prints one JSON line.
#pragma once
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

struct Unsupported : std::runtime_error { using std::runtime_error::runtime_error; };

struct Problem {
    std::string alphabet;
    int go = 0, ge = 0;
    std::vector<std::vector<int>> matrix;
    int score(char a, char b) const {
        return matrix[alphabet.find(a)][alphabet.find(b)];
    }
    bool is_dna() const { return alphabet == "ACGT"; }
    bool is_blosum62() const { return alphabet.size() == 23; }
};

struct Args {
    std::map<std::string, std::string> kv;
    std::string get(const std::string& k, const std::string& d = "") const {
        auto it = kv.find(k); return it == kv.end() ? d : it->second;
    }
    std::string fixture, problem, gap, mode, out, dtype, scores_out, variant;
    int threads = 1, reps = 3;
    double deadline = 0;
    double warmup_slack = 3.0;   // the warm-up pass's allowance for one-time costs
    bool affine() const { return gap == "affine"; }
    bool global() const { return mode == "global"; }
};

inline Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string k = argv[i];
        if (k.rfind("--", 0) == 0) k = k.substr(2);
        a.kv[k] = argv[i + 1];
    }
    a.fixture = a.get("fixture"); a.problem = a.get("problem");
    a.gap = a.get("gap"); a.mode = a.get("mode"); a.out = a.get("out");
    a.dtype = a.get("dtype"); a.scores_out = a.get("scores-out");
    a.variant = a.get("variant", "default");
    a.threads = std::stoi(a.get("threads", "1"));
    a.reps = std::stoi(a.get("reps", "3"));
    a.deadline = std::stod(a.get("deadline", "0"));
    a.warmup_slack = std::stod(a.get("warmup-slack", "3"));
    return a;
}

// "key=value,key=value" -> map; "default" -> empty
inline std::map<std::string, std::string> parse_variant(const std::string& v) {
    std::map<std::string, std::string> m;
    if (v.empty() || v == "default") return m;
    std::stringstream ss(v); std::string item;
    while (std::getline(ss, item, ',')) {
        auto eq = item.find('=');
        m[item.substr(0, eq)] = eq == std::string::npos ? "" : item.substr(eq + 1);
    }
    return m;
}

inline Problem read_problem(const std::string& path) {
    Problem p; std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot read problem " + path);
    in >> p.alphabet >> p.go >> p.ge;
    size_t n = p.alphabet.size();
    p.matrix.assign(n, std::vector<int>(n));
    for (auto& r : p.matrix) for (auto& v : r) in >> v;
    return p;
}

inline void read_pairs(const std::string& path, std::vector<std::string>& A,
                       std::vector<std::string>& B) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot read fixture " + path);
    std::string line;
    while (std::getline(in, line)) {
        auto t = line.find('\t');
        A.push_back(line.substr(0, t)); B.push_back(line.substr(t + 1));
    }
}

inline double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

inline std::string jstr(const std::string& s) {
    std::string o = "\"";
    for (char c : s) { if (c == '"' || c == '\\') o += '\\'; if (c == '\n') { o += "\\n"; continue; } o += c; }
    return o + "\"";
}

struct Result {
    std::string status = "ok";
    std::vector<double> times;
    std::vector<double> scores;
    std::map<std::string, std::string> info;   // values are raw JSON
};

inline void emit(const Result& r, const Args& a, size_t n_pairs, double cells) {
    if (r.status == "ok" && !a.scores_out.empty()) {
        FILE* f = std::fopen(a.scores_out.c_str(), "w");
        for (double s : r.scores) std::fprintf(f, "%.17g\n", s);
        std::fclose(f);
    }
    std::printf("{\"status\": %s, \"times\": [", jstr(r.status).c_str());
    double best = 0;
    for (size_t i = 0; i < r.times.size(); ++i) {
        std::printf("%s%.9g", i ? ", " : "", r.times[i]);
        if (i == 0 || r.times[i] < best) best = r.times[i];
    }
    std::printf("], \"best_s\": ");
    if (r.times.empty()) std::printf("null"); else std::printf("%.9g", best);
    std::printf(", \"n_pairs\": %zu, \"cells\": %.0f, \"info\": {", n_pairs, cells);
    bool first = true;
    for (auto& [k, v] : r.info) { std::printf("%s%s: %s", first ? "" : ", ", jstr(k).c_str(), v.c_str()); first = false; }
    std::printf("}}\n");
    std::fflush(stdout);
}

inline void emit_status(const std::string& status, const std::string& why) {
    std::printf("{\"status\": %s, \"why\": %s}\n", jstr(status).c_str(), jstr(why).c_str());
    std::fflush(stdout);
}

// Per-pair timing loop.  f(i) aligns pair i and returns its score; it must be callable
// from several threads at once when threads > 1 (per-thread state indexed by `tid`).
template<class F>
Result time_pairs(const Args& a, size_t n, F&& f) {
    Result r;
    std::vector<double> sc(n);
    double deadline_timed = a.deadline > 0 ? a.deadline : 1e300;
    for (int rep = 0; rep <= a.reps; ++rep) {
        double deadline = rep ? deadline_timed : deadline_timed + a.warmup_slack;
        double t0 = now();
        bool late = false;
        if (a.threads <= 1) {
            for (size_t i = 0; i < n; ++i) {
                sc[i] = f(i, 0);
                if (now() - t0 > deadline) { late = true; break; }
            }
        } else {
            std::atomic<size_t> next{0};
            std::atomic<bool> stop{false};
            size_t chunk = std::max<size_t>(1, n / (size_t(a.threads) * 64));
            std::vector<std::thread> th;
            for (int t = 0; t < a.threads; ++t)
                th.emplace_back([&, t] {
                    for (;;) {
                        size_t s = next.fetch_add(chunk);
                        if (s >= n || stop) return;
                        for (size_t i = s; i < std::min(n, s + chunk); ++i) sc[i] = f(i, t);
                        if (now() - t0 > deadline) { stop = true; return; }
                    }
                });
            for (auto& x : th) x.join();
            late = stop;
            r.info["harness_parallel"] = "true";
        }
        double dt = now() - t0;
        if (late) { r.status = "timeout"; r.times.push_back(dt); return r; }
        if (rep) r.times.push_back(dt);
    }
    r.scores = sc;
    return r;
}

// One-call batch timing (the tool runs all pairs itself and cannot be interrupted;
// the parent's wall-clock kill bounds it).  run() returns the scores.
template<class F>
Result time_batch(const Args& a, F&& run) {
    Result r;
    double deadline_timed = a.deadline > 0 ? a.deadline : 1e300;
    for (int rep = 0; rep <= a.reps; ++rep) {
        double deadline = rep ? deadline_timed : deadline_timed + a.warmup_slack;
        double t0 = now();
        auto sc = run();
        double dt = now() - t0;
        if (rep) r.times.push_back(dt);
        r.scores = std::move(sc);
        if (dt > deadline) { r.status = "timeout"; if (!rep) r.times.push_back(dt); return r; }
    }
    return r;
}

inline double total_cells(const std::vector<std::string>& A, const std::vector<std::string>& B) {
    double c = 0;
    for (size_t i = 0; i < A.size(); ++i) c += double(A[i].size()) * double(B[i].size());
    return c;
}

// main() of every driver: run(args, A, B, problem) -> Result
#define DRIVER_MAIN(RUN)                                                          \
    int main(int argc, char** argv) {                                             \
        Args a = parse_args(argc, argv);                                          \
        try {                                                                     \
            std::vector<std::string> A, B;                                        \
            read_pairs(a.fixture, A, B);                                          \
            Problem p = read_problem(a.problem);                                  \
            Result r = RUN(a, A, B, p);                                           \
            emit(r, a, A.size(), total_cells(A, B));                              \
        } catch (const Unsupported& e) { emit_status("unsupported", e.what()); } \
        catch (const std::exception& e) { emit_status("error", e.what()); }      \
        return 0;                                                                 \
    }
