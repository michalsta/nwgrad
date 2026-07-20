// stress_batch.cpp — single-batch CLI benchmark for BatchAligner
//
// Usage:
//   ./stress_batch
//   ./stress_batch --n 5000 --seq-len 100
//   ./stress_batch --n 1000 --seq-len 20 200 --n-threads 1 4 8 16
//   ./stress_batch --gap-model affine --mode global --grad-mode soft
//   ./stress_batch --help

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "batch.hpp"

// ── BLOSUM62 (canonical AA order: ACDEFGHIKLMNPQRSTVWY) ──────────────────────

// Row = first AA, col = second AA; AA_ORDER is from subst_matrix.hpp.
static const double BLOSUM62[20][20] = {
    { 4,  0, -2, -1, -2,  0, -2, -1, -1, -1, -1, -2, -1, -1, -1,  1,  0,  0, -3, -2},
    { 0,  9, -3, -4, -2, -3, -3, -1, -3, -1, -1, -3, -3, -3, -3, -1, -1, -1, -2, -2},
    {-2, -3,  6,  2, -3, -1, -1, -3, -1, -4, -3,  1, -1,  0, -2,  0, -1, -3, -4, -3},
    {-1, -4,  2,  5, -3, -2,  0, -3,  1, -3, -2,  0, -1,  2,  0,  0, -1, -2, -3, -2},
    {-2, -2, -3, -3,  6, -3, -1,  0, -3,  0,  0, -3, -4, -3, -3, -2, -2, -1,  1,  3},
    { 0, -3, -1, -2, -3,  6, -2, -4, -2, -4, -3,  0, -2, -2, -2,  0, -2, -3, -2, -3},
    {-2, -3, -1,  0, -1, -2,  8, -3, -1, -3, -2,  1, -2,  0,  0, -1, -2, -3, -2,  2},
    {-1, -1, -3, -3,  0, -4, -3,  4, -3,  2,  1, -3, -3, -3, -3, -2, -1,  3, -3, -1},
    {-1, -3, -1,  1, -3, -2, -1, -3,  5, -2, -1,  0, -1,  1,  2,  0, -1, -2, -3, -2},
    {-1, -1, -4, -3,  0, -4, -3,  2, -2,  4,  2, -3, -3, -2, -2, -2, -1,  1, -2, -1},
    {-1, -1, -3, -2,  0, -3, -2,  1, -1,  2,  5, -2, -2,  0, -1, -1, -1,  1, -1, -1},
    {-2, -3,  1,  0, -3,  0,  1, -3,  0, -3, -2,  6, -2,  0,  0,  1,  0, -3, -4, -2},
    {-1, -3, -1, -1, -4, -2, -2, -3, -1, -3, -2, -2,  7, -1, -2, -1, -1, -2, -4, -3},
    {-1, -3,  0,  2, -3, -2,  0, -3,  1, -2,  0,  0, -1,  5,  1,  0, -1, -2, -2, -1},
    {-1, -3, -2,  0, -3, -2,  0, -3,  2, -2, -1,  0, -2,  1,  5, -1, -1, -3, -3, -2},
    { 1, -1,  0,  0, -2,  0, -1, -2,  0, -2, -1,  1, -1,  0, -1,  4,  1, -2, -3, -2},
    { 0, -1, -1, -1, -2, -2, -2, -1, -1, -1, -1,  0, -1, -1, -1,  1,  5,  0, -2, -2},
    { 0, -1, -3, -2, -1, -3, -3,  3, -2,  1,  1, -3, -2, -2, -3, -2,  0,  4, -3, -1},
    {-3, -2, -4, -3,  1, -2, -2, -3, -3, -2, -1, -4, -4, -2, -3, -3, -2, -3, 11,  2},
    {-2, -2, -3, -2,  3, -3,  2, -1, -2, -1, -1, -2, -3, -1, -2, -2, -2, -1,  2,  7},
};

static SubstMatrix make_blosum62() {
    double flat[400];
    for (int i = 0; i < 20; ++i)
        for (int j = 0; j < 20; ++j)
            flat[i * 20 + j] = BLOSUM62[i][j];
    return SubstMatrix(flat);
}

// ── argument parsing ──────────────────────────────────────────────────────────

struct Args {
    int         n           = 1000;
    int         seq_len_lo  = 50;
    int         seq_len_hi  = 50;       // == lo means fixed length
    GapModel    gap_model   = GapModel::Affine;
    AlignMode   align_mode  = AlignMode::Global;
    BatchAligner::GradMode grad_mode = BatchAligner::GradMode::Hard;
    double      gap_open    = 11.0;
    double      gap_extend  = 1.0;
    std::vector<int> n_threads = {4};
    uint64_t    seed        = 0;        // 0 = random
    bool        warmup      = false;
    int         kernel      = kBackendScalar;  // parse_backend vocabulary
};

[[noreturn]] static void usage(const char* prog, int exit_code = 0) {
    std::cout <<
        "Usage: " << prog << " [options]\n"
        "\n"
        "Options:\n"
        "  --n N                  number of sequence pairs (default 1000)\n"
        "  --seq-len LEN          fixed sequence length (default 50)\n"
        "  --seq-len LO HI        random length in [LO, HI]\n"
        "  --gap-model linear|affine  (default affine)\n"
        "  --mode global|local        (default global)\n"
        "  --grad-mode none|hard|soft (default hard)\n"
        "  --gap-open F           gap-open penalty (default 11.0)\n"
        "  --gap-extend F         gap-extend penalty (default 1.0)\n"
        "  --n-threads T...       one or more thread counts (default 4)\n"
        "  --kernel BACKEND       Viterbi backend: scalar_fallback|auto|sse2|avx2|avx512|neon\n"
        "                         (default scalar_fallback); every simd level is bit-exact\n"
        "  --seed N               RNG seed (default: random)\n"
        "  --warmup               run a silent warmup pass before timing\n"
        "  --help                 show this message\n";
    std::exit(exit_code);
}

static double parse_double(const char* s) {
    char* end;
    double v = std::strtod(s, &end);
    if (end == s || *end != '\0')
        throw std::invalid_argument(std::string("bad float: ") + s);
    return v;
}

static int parse_int(const char* s) {
    int v;
    auto [ptr, ec] = std::from_chars(s, s + std::strlen(s), v);
    if (ec != std::errc{} || ptr != s + std::strlen(s))
        throw std::invalid_argument(std::string("bad int: ") + s);
    return v;
}

static bool is_flag(const char* s) {
    return s[0] == '-' && s[1] == '-';
}

static Args parse_args(int argc, char** argv) {
    Args a;
    bool seed_set = false;

    for (int i = 1; i < argc; ++i) {
        auto need = [&](int k) {
            if (i + k >= argc)
                throw std::invalid_argument(std::string(argv[i]) + " needs an argument");
        };

        if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
        } else if (std::strcmp(argv[i], "--n") == 0) {
            need(1); a.n = parse_int(argv[++i]);
        } else if (std::strcmp(argv[i], "--seq-len") == 0) {
            need(1);
            a.seq_len_lo = parse_int(argv[++i]);
            // optional second value
            if (i + 1 < argc && !is_flag(argv[i + 1]))
                a.seq_len_hi = parse_int(argv[++i]);
            else
                a.seq_len_hi = a.seq_len_lo;
        } else if (std::strcmp(argv[i], "--gap-model") == 0) {
            need(1); ++i;
            if (std::strcmp(argv[i], "linear") == 0)       a.gap_model = GapModel::Linear;
            else if (std::strcmp(argv[i], "affine") == 0)  a.gap_model = GapModel::Affine;
            else throw std::invalid_argument("unknown gap-model: " + std::string(argv[i]));
        } else if (std::strcmp(argv[i], "--mode") == 0) {
            need(1); ++i;
            if (std::strcmp(argv[i], "global") == 0)      a.align_mode = AlignMode::Global;
            else if (std::strcmp(argv[i], "local") == 0)  a.align_mode = AlignMode::Local;
            else throw std::invalid_argument("unknown mode: " + std::string(argv[i]));
        } else if (std::strcmp(argv[i], "--grad-mode") == 0) {
            need(1); ++i;
            if      (std::strcmp(argv[i], "none") == 0) a.grad_mode = BatchAligner::GradMode::None;
            else if (std::strcmp(argv[i], "hard") == 0) a.grad_mode = BatchAligner::GradMode::Hard;
            else if (std::strcmp(argv[i], "soft") == 0) a.grad_mode = BatchAligner::GradMode::Soft;
            else throw std::invalid_argument("unknown grad-mode: " + std::string(argv[i]));
        } else if (std::strcmp(argv[i], "--gap-open") == 0) {
            need(1); a.gap_open = parse_double(argv[++i]);
        } else if (std::strcmp(argv[i], "--gap-extend") == 0) {
            need(1); a.gap_extend = parse_double(argv[++i]);
        } else if (std::strcmp(argv[i], "--n-threads") == 0) {
            need(1);
            a.n_threads.clear();
            while (i + 1 < argc && !is_flag(argv[i + 1]))
                a.n_threads.push_back(parse_int(argv[++i]));
            if (a.n_threads.empty())
                throw std::invalid_argument("--n-threads needs at least one value");
        } else if (std::strcmp(argv[i], "--seed") == 0) {
            need(1);
            uint64_t v;
            auto s = argv[++i];
            auto len = std::strlen(s);
            auto [ptr, ec] = std::from_chars(s, s + len, v);
            if (ec != std::errc{} || ptr != s + len)
                throw std::invalid_argument(std::string("bad seed: ") + s);
            a.seed = v;
            seed_set = true;
        } else if (std::strcmp(argv[i], "--kernel") == 0) {
            need(1); ++i;
            a.kernel = parse_backend(argv[i]);  // scalar_fallback|auto|sse2|avx2|avx512|neon
        } else if (std::strcmp(argv[i], "--warmup") == 0) {
            a.warmup = true;
        } else {
            throw std::invalid_argument("unknown option: " + std::string(argv[i]));
        }
    }

    if (!seed_set) {
        std::random_device rd;
        a.seed = (static_cast<uint64_t>(rd()) << 32) | rd();
    }
    return a;
}

// ── sequence generation ───────────────────────────────────────────────────────

static std::vector<std::string> random_seqs(std::mt19937_64& rng, int n, int lo, int hi) {
    std::uniform_int_distribution<int> len_dist(lo, hi);
    std::uniform_int_distribution<int> aa_dist(0, 19);
    std::vector<std::string> seqs(static_cast<size_t>(n));
    for (auto& s : seqs) {
        int len = len_dist(rng);
        s.resize(static_cast<size_t>(len));
        for (char& c : s)
            c = AA_ORDER[aa_dist(rng)];
    }
    return seqs;
}

// ── reporting ─────────────────────────────────────────────────────────────────

static void print_header(const Args& a) {
    std::string len_str;
    if (a.seq_len_lo == a.seq_len_hi)
        len_str = std::to_string(a.seq_len_lo);
    else
        len_str = std::to_string(a.seq_len_lo) + "-" + std::to_string(a.seq_len_hi);

    const char* gm  = (a.gap_model  == GapModel::Affine)  ? "affine"  : "linear";
    const char* mo  = (a.align_mode == AlignMode::Global)  ? "global"  : "local";
    const char* grd = (a.grad_mode  == BatchAligner::GradMode::Hard) ? "hard"
                    : (a.grad_mode  == BatchAligner::GradMode::Soft) ? "soft" : "none";

    std::string kn = backend_name(a.kernel);

    std::printf("n=%d  seq_len=%s  gap_model=%s  mode=%s  grad_mode=%s  kernel=%s"
                "  gap_open=%.1f  gap_extend=%.1f  seed=0x%016lx\n",
                a.n, len_str.c_str(), gm, mo, grd, kn.c_str(),
                a.gap_open, a.gap_extend, (unsigned long)a.seed);
}

static void print_seq_stats(const std::vector<std::string>& seqs) {
    double sum = 0;
    size_t lo = SIZE_MAX, hi = 0;
    for (const auto& s : seqs) {
        sum += static_cast<double>(s.size());
        lo = std::min(lo, s.size());
        hi = std::max(hi, s.size());
    }
    std::printf("actual seq_len: mean=%.0f  min=%zu  max=%zu\n",
                sum / static_cast<double>(seqs.size()), lo, hi);
    std::printf("\n");
}

static void print_run(int n_threads, double elapsed, const std::vector<double>& scores,
                      const AlignParams& grad, bool has_grad) {
    double mn = *std::min_element(scores.begin(), scores.end());
    double mx = *std::max_element(scores.begin(), scores.end());
    double mean = std::accumulate(scores.begin(), scores.end(), 0.0)
                  / static_cast<double>(scores.size());
    double pairs_per_sec = static_cast<double>(scores.size()) / elapsed;

    std::printf("  n_threads=%2d  time=%.3fs  pairs/s=%'.0f"
                "  score min=%.1f mean=%.1f max=%.1f",
                n_threads, elapsed, pairs_per_sec, mn, mean, mx);

    if (has_grad) {
        double grad_sum = 0.0;
        for (int i = 0; i < 20; ++i)
            for (int j = 0; j < 20; ++j)
                grad_sum += grad.matrix.at(i, j);
        std::printf("  grad_sum=%.0f", grad_sum);
    }
    std::printf("\n");
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    Args a;
    try {
        a = parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        usage(argv[0], 1);
    }

    // Enable thousands separator in printf.
    std::setlocale(LC_ALL, "");

    std::mt19937_64 rng(a.seed);
    auto seqs_a = random_seqs(rng, a.n, a.seq_len_lo, a.seq_len_hi);
    auto seqs_b = random_seqs(rng, a.n, a.seq_len_lo, a.seq_len_hi);

    // Build problem list (string_views into the owned strings).
    std::vector<ProblemInstance> problems;
    problems.reserve(static_cast<size_t>(a.n));
    for (int i = 0; i < a.n; ++i)
        problems.push_back({seqs_a[static_cast<size_t>(i)],
                            seqs_b[static_cast<size_t>(i)]});

    SubstMatrix blosum = make_blosum62();
    double go = (a.gap_model == GapModel::Affine) ? a.gap_open : 0.0;

    AlignParams params(Alphabet::protein());
    params.matrix       = blosum;
    params.gap_open_a   = params.gap_open_b   = go;
    params.gap_extend_a = params.gap_extend_b = a.gap_extend;

    print_header(a);
    print_seq_stats(seqs_a);

    bool has_grad = (a.grad_mode != BatchAligner::GradMode::None);

    for (int n_threads : a.n_threads) {
        BatchAligner aligner(params, /*band=*/0,
                             a.gap_model, a.align_mode, a.grad_mode, n_threads,
                             a.kernel);

        if (a.warmup) {
            size_t warmup_n = std::min<size_t>(16, static_cast<size_t>(a.n));
            std::vector<ProblemInstance> wp(problems.begin(),
                                           problems.begin() + static_cast<ptrdiff_t>(warmup_n));
            aligner.align(wp);
        }

        auto t0 = std::chrono::steady_clock::now();
        BatchResult result = aligner.align(problems);
        auto t1 = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();

        print_run(n_threads, elapsed, result.scores, result.grad, has_grad);
    }

    return 0;
}
