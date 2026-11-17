// ── The striped query profile: every level's build is the scalar lookup, byte for byte ──
//
// striped_profile_build (kernels_impl.inl) fills prof[c*sw + s*W + l] = row_c[b[col(l,s)-1]]
// (padding columns past n -> nw_pad) with vector permutes where a level has them (byte
// shuffles over byte planes, AVX-512 two-table permutes) and a scalar loop elsewhere.  The
// profile is data the kernels read, so its identity is checked directly, on bit patterns,
// against the definition: random bits for every type (floats include NaNs and -0), every
// alphabet size 1..70 (across the 16 / 32 / 64-symbol table limits), lengths 1..300 and a
// few long ones, lane counts beyond the level's own (the build takes W as a parameter, so
// every tail length is reached), the untracked build (every row) and the tracked one (a
// random subset of rows, in random order).  NWGRAD_PROFILE_FUZZ=<n> scales the lengths
// sampled per configuration.

#include "catch.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "simd_levels.hpp"
#include "scalar_traits.hpp"

namespace {

template <class T>
T random_bits(std::mt19937_64& rng) {
    std::uint64_t r = rng();
    switch (rng() % 8) {      // a few special values among the random patterns
        case 0: r = 0; break;
        case 1: r = ~std::uint64_t{0}; break;
        case 2: r = std::uint64_t{1} << (8 * sizeof(T) - 1); break;   // -0 / INT_MIN
        default: break;
    }
    T v;
    std::memcpy(&v, &r, sizeof v);
    return v;
}

template <class T>
void check_level(int level, std::mt19937_64& rng, long& n_checked) {
    const auto fn = level_kernels(level).template of<T>().profile_build;
    REQUIRE(fn != nullptr);
    const long fuzz = std::getenv("NWGRAD_PROFILE_FUZZ") ? std::atol(std::getenv("NWGRAD_PROFILE_FUZZ")) : 1;
    std::vector<int> lens;
    for (int n = 1; n <= 300; ++n) lens.push_back(n);
    for (int n : {511, 512, 513, 1000, 2047, 4099}) lens.push_back(n);
    const int nalphas[] = {1, 2, 4, 5, 15, 16, 17, 20, 21, 24, 25, 31, 32, 33, 48, 63, 64, 65, 70};
    for (int nalpha : nalphas) {
        std::vector<T> blk((std::size_t)nalpha * nalpha);
        for (T& v : blk) v = random_bits<T>(rng);
        for (int n : lens) {
            for (long rep = 0; rep < fuzz; ++rep) {
                // a lane count: the level's own widths and odd ones (seg = ceil(n / W), as the
                // kernels stripe; W > n leaves whole lanes of padding)
                const int Ws[] = {2, 4, 8, 16, 32, 64, 3, 7};
                const int W = Ws[rng() % 8];
                const int seg = (n + W - 1) / W;
                const std::size_t sw = (std::size_t)seg * W;
                std::vector<unsigned char> b((std::size_t)n);
                for (auto& c : b) c = (unsigned char)(rng() % (unsigned)nalpha);
                // tracked: a random subset of the rows, in random order (repeats allowed)
                const bool tracked = rng() % 2;
                std::vector<std::uint32_t> crow;
                if (tracked) {
                    const int nrow = 1 + (int)(rng() % (unsigned)(2 * nalpha));
                    for (int c = 0; c < nrow; ++c) crow.push_back((std::uint32_t)(rng() % (unsigned)nalpha));
                }
                const int nrow = tracked ? (int)crow.size() : nalpha;
                std::vector<T> got((std::size_t)nrow * sw), want((std::size_t)nrow * sw);
                fn(got.data(), blk.data(), nalpha, b.data(), n, seg, W,
                   tracked ? crow.data() : nullptr, nrow);
                for (int c = 0; c < nrow; ++c) {
                    const T* row = blk.data() + (std::size_t)(tracked ? crow[c] : c) * nalpha;
                    for (int l = 0; l < W; ++l)
                        for (int s = 0; s < seg; ++s) {
                            const int j = l * seg + s + 1;
                            want[(std::size_t)c * sw + (std::size_t)s * W + l] = j <= n ? row[b[j - 1]] : nw_pad<T>;
                        }
                }
                if (std::memcmp(got.data(), want.data(), got.size() * sizeof(T)) != 0) {
                    std::size_t k = 0;
                    while (std::memcmp(&got[k], &want[k], sizeof(T)) == 0) ++k;
                    FAIL("level " << level << " sizeof(T)=" << sizeof(T) << " nalpha=" << nalpha
                         << " n=" << n << " W=" << W << " tracked=" << tracked << ": first difference at "
                         << k << " (row " << k / sw << ", slot " << k % sw << ")");
                }
                ++n_checked;
            }
        }
    }
}

}  // namespace

TEMPLATE_TEST_CASE("striped profile build: every level is the scalar lookup, bit for bit",
                   "[profile]", double, float, std::int16_t, std::int32_t, std::int64_t) {
    std::mt19937_64 rng(0x5eed0f11e + sizeof(TestType));
    long n_checked = 0;
    for (SimdLevel l : detect_available_levels())
        check_level<TestType>(static_cast<int>(l), rng, n_checked);
    INFO("profiles checked: " << n_checked);
    CHECK((n_checked > 0 || detect_available_levels().empty()));
}
