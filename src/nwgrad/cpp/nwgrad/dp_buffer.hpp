#pragma once

// DpBuffer — the DP table storage, split out of aligner.hpp so the leveled kernel
// TUs (which are compiled per ISA level) can see it without dragging in the whole
// Aligner template.  It is plain data: the only currency that crosses a level
// boundary (see simd_levels.hpp).

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <new>
#include <vector>

#include "scalar_traits.hpp"
#if __has_include(<unistd.h>)
#  include <unistd.h>
#endif
#if defined(__linux__)
#  include <sys/mman.h>
#endif

// THP hinting is on by default; NWGRAD_HUGEPAGE=0 disables it (escape hatch, and the
// A/B knob for measuring its effect — resolved once).
inline bool nwgrad_hugepage_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("NWGRAD_HUGEPAGE");
        return !(e && e[0] == '0');
    }();
    return on;
}

// ── aligned, huge-page-hinting allocator for the DP tables ────────────────────
//
// Two things it does, both measured:
//
// (1) 64-byte base alignment.  std::vector guarantees only 16-byte, so on AVX2 every
//     256-bit (32-byte) W-wide load/store into VM/VX/VY is misaligned and some split
//     cache lines — a ~15-18% loss on AVX2 (nothing on SSE2/NEON, where W=2 is 16
//     bytes and 16-byte alignment already suffices).  A 64-byte base plus per-row
//     padding to a multiple of W (see rowsz in kernels_impl.inl / cell_index) keeps
//     every striped row's W-wide accesses off cache-line boundaries.
//
// (2) Transparent huge pages.  Our Linux boxes default THP to [madvise], so a plain
//     std::vector never gets huge pages and the DP walk eats TLB misses.  An
//     allocation >= the 2 MiB huge-page size is aligned to 2 MiB and MADV_HUGEPAGE'd,
//     so it can be backed by huge pages.  The >= 2 MiB gate is not a tunable: a table
//     smaller than a huge page CANNOT be backed by one (the page would run past the
//     allocation), so a table only qualifies once it is that big (len ~512+).  Best-
//     effort — madvise failing is ignored; Linux-only; a no-op elsewhere.
//
// Uses C++17 aligned operator new, so it is portable (no posix_memalign / _aligned_malloc
// split).  deallocate recomputes the alignment from the size, so it always matches.
template <class T, std::size_t Align = 64>
struct AlignedAllocator {
    using value_type = T;
    static constexpr std::size_t kHugePage = std::size_t(2) << 20;   // 2 MiB (x86 THP)

    static std::align_val_t align_of(std::size_t bytes) noexcept {
        return bytes >= kHugePage ? std::align_val_t(kHugePage) : std::align_val_t(Align);
    }

    AlignedAllocator() noexcept = default;
    template <class U> AlignedAllocator(const AlignedAllocator<U, Align>&) noexcept {}

    T* allocate(std::size_t n) {
        if (n == 0) return nullptr;
        const std::size_t bytes = n * sizeof(T);
        void* p = ::operator new(bytes, align_of(bytes));
#if defined(__linux__) && defined(MADV_HUGEPAGE)
        if (bytes >= kHugePage && nwgrad_hugepage_enabled())
            ::madvise(p, bytes, MADV_HUGEPAGE);                       // best-effort
#endif
        return static_cast<T*>(p);
    }
    void deallocate(T* p, std::size_t n) noexcept {
        ::operator delete(p, align_of(n * sizeof(T)));
    }

    template <class U> struct rebind { using other = AlignedAllocator<U, Align>; };
    template <class U> bool operator==(const AlignedAllocator<U, Align>&) const noexcept { return true; }
    template <class U> bool operator!=(const AlignedAllocator<U, Align>&) const noexcept { return false; }
};

// The always-double DP storage (forward-backward / soft path) uses it.
using DVec = std::vector<double, AlignedAllocator<double>>;

// Direction-pointer storage for the variant-B Viterbi (see aligner.hpp).  One byte
// per cell per state instead of a retained score table: 3 B/cell against 12 (float32)
// or 24 (double), which is the whole point of B.
using BVec = std::vector<unsigned char, AlignedAllocator<unsigned char>>;

// A vector whose data() starts at a chosen cache line of its 4 KB page (address mod 4096 =
// line * 64), for the striped Scores tables VM / VX / VY: they are separate allocations, so
// without it row i of the three shares whatever offsets the allocator gave them (above 2 MiB,
// 2 MiB-aligned: one offset), and a 4-way L1D (Piledriver) thrashed on them.  place(line)
// takes effect at the next resize(); -1 (the default) is plain placement, no slack.  The
// contents are preserved across resize() and placement changes, as a vector's are.
template <class T>
class PlacedVec {
public:
    using value_type = T;
    void place(int line) noexcept { line_ = line; }
    void resize(std::size_t n) {
        const std::size_t slack = line_ < 0 ? 0 : 4096 / sizeof(T);
        const std::size_t old_off = off_, old_n = n_;
        if (v_.size() < n + slack) v_.resize(n + slack);
        std::size_t off = 0;
        if (line_ >= 0) {
            const std::size_t at = reinterpret_cast<std::uintptr_t>(v_.data()) % 4096;
            off = ((static_cast<std::size_t>(line_) * 64 + 4096 - at) % 4096) / sizeof(T);
        }
        if (off != old_off && old_n)
            std::memmove(v_.data() + off, v_.data() + old_off, old_n * sizeof(T));
        off_ = off; n_ = n;
    }
    std::size_t size() const noexcept { return n_; }
    bool empty() const noexcept { return n_ == 0; }
    T* data() noexcept { return v_.data() + off_; }
    const T* data() const noexcept { return v_.data() + off_; }
    T& operator[](std::size_t k) noexcept { return v_[off_ + k]; }
    const T& operator[](std::size_t k) const noexcept { return v_[off_ + k]; }
    T* begin() noexcept { return data(); }
    T* end() noexcept { return data() + n_; }
    const T* begin() const noexcept { return data(); }
    const T* end() const noexcept { return data() + n_; }
    void clear() noexcept { v_.clear(); off_ = 0; n_ = 0; }
    void shrink_to_fit() { v_.shrink_to_fit(); }
private:
    std::vector<T, AlignedAllocator<T>> v_;
    std::size_t off_ = 0, n_ = 0;
    int line_ = -1;
};


// ── DpBuffer ──────────────────────────────────────────────────────────────────
//
// Holds all DP table vectors for one Aligner computation.
// Lives either inside the Aligner (own_buf_) or externally (e.g. per-thread).
// Vectors grow on demand and are never implicitly freed; call clear() to release.
//
// Templated on the *Viterbi* scalar type T (float32 or the default double).  The
// membrane between the two precisions runs right through this struct: the Viterbi
// tables + query profiles are T (so a float32 kernel halves their footprint and
// doubles its SIMD lane count), while the forward-backward / soft-gradient tables
// are ALWAYS double — that path is log-sum-exp and stays in double regardless of the
// Viterbi precision.  T=double gives back the original struct verbatim (TVec == DVec),
// which is why `using DpBuffer = DpBufferT<double>` below leaves every existing
// caller — and the on-disk double build — byte-for-byte unchanged.
template <class T = double>
struct DpBufferT {
    using TVec = std::vector<T, AlignedAllocator<T>>;

    TVec H;                        // linear viterbi
    PlacedVec<T> VM, VX, VY;       // affine viterbi (placed by the striped Scores kernel)
    DVec F, B;                     // linear forward-backward   (always double)
    DVec FM, FX, FY, BM, BX, BY;  // affine forward-backward   (always double)

    // Used only by the Simd kernel.  `prof` is the query profile — the
    // substitution scores of every alphabet symbol against sequence B, laid out
    // contiguously in j so the DP row loop loads them with a vector load instead
    // of a gather (Full mode).  `subbuf` is the per-row equivalent for banded
    // mode, where a full-width profile would cost more than the banded DP itself.
    TVec prof, subbuf;

    // Used only by the striped affine kernel (leveled, in kernels_impl.inl), which
    // writes VM/VX/VY directly in striped layout (no de-stripe copy).  `sopenv` is one
    // striped openv row (the VM/VX open values feeding the VY carry); `sprof` is the
    // query profile in striped order.  Both O(n), not O(m·n).
    TVec sopenv, sprof;
    TVec spad;     // score-only Local: per striped slot, 0 (real column) or -inf (padding)

    // ── TracebackMode::Pointers: predecessors instead of retained score tables ─
    //
    // DM/DX/DY hold, for every cell, which predecessor state the forward pass chose
    // — 0=M, 1=X, 2=Y — recorded with exactly the traceback's own `>=` M>X>Y
    // tie-break, which is what makes B bit-exact and not merely correct.  DM also
    // carries the sentinel 3 = "VM <= 0 here", reproducing Local's traceback stop
    // condition without keeping a single VM value around.
    //
    // With these, the score tables collapse to two rolling rows (rM/rX/rY below),
    // so the retained footprint is 3 B/cell instead of 12.  That footprint is the
    // thing measured to drive the page-fault cost that dominates the memory-bound
    // regime — see the sorted-scheduler notes.
    BVec DM, DX, DY;
    TVec rM, rX, rY, qM, qX, qY;   // rolling current/previous rows (O(n), not O(mn))
    // Rolling rows of the leveled kernels, each family's carved from ONE block by
    // spread_rows (below): srows score-only, prows striped Pointers and hb_base_striped,
    // hrows the Hirschberg sweeps and endpoint scans, lrows the linear kernels.
    TVec srows, prows, hrows, lrows;

    // ── TracebackMode::Hirschberg: no table at all above the base case ────────
    //
    // The divide-and-conquer sweeps keep only rolling rows, so the whole recursion
    // needs O(n) rather than O(m·n): six rows for the forward half, six for the
    // reverse half, and three to hold the forward half's split row while the reverse
    // half runs.  hbD is the base case's direction table -- the recursion bottoms out
    // in the Pointers fill, so a block of at most `cutoff` rows is the largest thing
    // ever materialized.  hops is the recovered path itself, one byte per step, which
    // is what the gradient and the alignment are read from instead of tables.
    TVec hfa, hfb, hfc, hfd, hfe, hff;   // forward rolling rows (M,X,Y) x (prev,cur)
    TVec hra, hrb, hrc, hrd, hre, hrf;   // reverse rolling rows (M,X,Y) x (prev,cur)
    TVec hsM, hsX, hsY;                  // forward half's split row, saved
    BVec hbD;                            // base-case direction table (3 x block)
    BVec hops;                           // the path: 0=M(diag) 1=X(gap in b) 2=Y(gap in a)
    // Striped simd sweep only: the block's query profile in striped order, and one
    // striped row of VM/VX open values feeding the VY carry.  Both O(block width).
    // The profile is rebuilt per BLOCK (its striping geometry depends on the block's
    // own width), but serves that block's forward AND reverse sweeps, so its cost
    // amortizes over H rows rather than H/2.
    TVec hprof, hov;
    // Prefix-max sweep only (TracebackMode::HirschbergPmax): the gap ramp k*ge_a for
    // every column of the block, in striped order.  Built ONCE per sweep by a scalar
    // loop and then only loaded, which is load-bearing twice over: it keeps the ramp
    // out of the row loop (it depends on the column, not the row), and it means the
    // hot loop contains no multiply for the compiler to contract into an FMA — an FMA
    // in one level TU and a mul+add in another would round differently and break the
    // scalar==simd bit-identity the pmax family is required to hold.
    TVec hramp;
    // The integer carry's ramp row where it is loaded rather than formed in registers
    // (int_row_ramp, kernels_impl.inl: int16 at 16 lanes).  Its own vector: the Hirschberg
    // sweeps hold hramp across the base cases that run this carry.
    TVec iramp;

    // Scaled soft path (SoftImpl::Scaled): rolling backward rows (current/next, per
    // state) and the per-row power-of-two exponents of the forward tables.  The forward
    // tables reuse F / FM,FX,FY; the B tables are not used at all.
    // sqp is the soft query profile, exp(score(a, b[j-1])) laid out per residue a.
    DVec sb0, sb1, sb2, sb3, sb4, sb5, sqp;
    std::vector<int> sexp;

    void clear() noexcept {
        auto clrT = [](TVec& v) noexcept { v.clear(); v.shrink_to_fit(); };
        auto clrD = [](DVec& v) noexcept { v.clear(); v.shrink_to_fit(); };
        clrT(H);
        VM.clear(); VX.clear(); VY.clear();
        VM.shrink_to_fit(); VX.shrink_to_fit(); VY.shrink_to_fit();
        clrD(F);  clrD(B);
        clrD(FM); clrD(FX); clrD(FY); clrD(BM); clrD(BX); clrD(BY);
        clrT(prof); clrT(subbuf);
        clrT(sopenv); clrT(sprof); clrT(spad);
        auto clrB = [](BVec& v) noexcept { v.clear(); v.shrink_to_fit(); };
        clrB(DM); clrB(DX); clrB(DY);
        clrT(rM); clrT(rX); clrT(rY); clrT(qM); clrT(qX); clrT(qY);
        clrT(srows); clrT(prows); clrT(hrows); clrT(lrows);
        clrT(hfa); clrT(hfb); clrT(hfc); clrT(hfd); clrT(hfe); clrT(hff);
        clrT(hra); clrT(hrb); clrT(hrc); clrT(hrd); clrT(hre); clrT(hrf);
        clrT(hsM); clrT(hsX); clrT(hsY);
        clrT(hprof); clrT(hov); clrT(hramp); clrT(iramp);
        clrB(hbD); clrB(hops);
        clrD(sb0); clrD(sb1); clrD(sb2); clrD(sb3); clrD(sb4); clrD(sb5); clrD(sqp);
        sexp.clear(); sexp.shrink_to_fit();
    }
};

// nrows (<= 13) rolling rows of len elements each, carved from one block at a stride of r
// cache lines such that r is not a multiple of 4 and every pair of rows, k*r lines apart
// (k < nrows), sits >= 4 lines from a multiple of 4096 B.  Rows allocated one by one came out
// exactly 4096 B apart at row sizes of ~4000-4080 B (and near multiples), so the rows a kernel
// walks in lockstep shared an L1 set, and a 16 KB 4-way L1D (Bulldozer / Piledriver) thrashed:
// up to 10x the time of the neighbouring lengths.  Forcing r mod 64 (skynet; AGENTS.md,
// "FIXED beside it"), ns/cell:
// - score kernel, int32 Local L1024 (1.48 clean): k*r = 0 -> 6.5 (L1 misses 6x); r = 32
//   (three rows on one 4 KB offset) -> 2.3, misses unchanged (a 12-bit address match in the
//   load/store unit); a row one line BEHIND another (r = 63; r = 21, 3r = -1) -> 2.4, 1.76;
//   -2 lines 1.55; r = 1 (+1 line, every pair) 1.60.
// - striped Pointers int32 Local L1000 (4.65-4.86 clean), linear H / byte fills int32 L1908
//   (2.38-2.45 / 2.76-2.90): r = 4 mod 8 -> 5.0-5.5 / 2.5 / 2.9, 8 mod 16 -> 4.9 / 2.8 / 3.2,
//   16 and 48 -> 5.8 / 2.9-3.0 / 3.4-3.5, 32 -> 7.7 / 3.2 / 3.7, with FEWER L1 misses: a
//   structural hazard at strides of 256 B multiples, not capacity.  Odd r and r = 2 mod 4 clean.
// At most 9 lines of padding a row.  The block keeps the allocator's 64-byte alignment.
// The stride rule alone: the row stride in elements for nrows rows of len elements.
template <class T>
inline std::size_t spread_stride(std::size_t len, int nrows) {
    constexpr std::size_t line = 64 / sizeof(T);
    std::size_t lines = (len + line - 1) / line;
    auto spread = [nrows](std::size_t r) {
        if (r % 4 == 0) return false;
        for (int k = 1; k < nrows; ++k) {
            const std::size_t d = (std::size_t(k) * r) % 64;
            if (d < 4 || d > 60) return false;
        }
        return true;
    };
    while (!spread(lines)) ++lines;
    return lines * line;
}

template <class T, class Vec>
inline T* spread_rows(Vec& block, std::size_t len, int nrows, std::size_t& stride) {
    stride = spread_stride<T>(len, nrows);
    if (block.size() < stride * nrows) block.resize(stride * nrows);
    return block.data();
}

// Whether the whole-table layouts are spread too: the striped Scores tables placed at
// table_lines and padded to spread_stride, the linear H table padded.  Only where the L1D is
// narrow (<= 4 ways: Bulldozer / Piledriver, 16 KB 4-way), which is where rows sharing a set
// thrashed: on a 12-way L1 the placement COST the Scores kernel (nighthaven AVX2: float32
// Global L2030 1.58x, int32 / double L1000 1.22-1.25x; solace AVX-512 1.06-1.13x) and the
// padded H stride moved its arms +-4 %; the rolling-row blocks (spread_rows) were neutral
// there and are not gated.  glibc reads the associativity from cpuid (skynet's VM reports
// 2-way, which the gate takes); macOS has no _SC_LEVEL1_DCACHE_ASSOC: off.
// NWGRAD_TABLE_SPREAD=0/1 overrides; table_spread_mode() is the tests' switch (-1 = auto).
inline std::atomic<int>& table_spread_mode() noexcept {
    static std::atomic<int> m{-1};
    return m;
}
inline bool table_spread() noexcept {
    const int m = table_spread_mode().load(std::memory_order_relaxed);
    if (m >= 0) return m != 0;
    static const bool narrow = [] {
        if (const char* e = std::getenv("NWGRAD_TABLE_SPREAD")) return std::atoi(e) != 0;
#if defined(_SC_LEVEL1_DCACHE_ASSOC)
        const long a = sysconf(_SC_LEVEL1_DCACHE_ASSOC);
        return a > 0 && a <= 4;
#else
        return false;
#endif
    }();
    return narrow;
}

// The lines (mod 64) at which the striped Scores tables start, VM at 0, VX at a, VY at b,
// for a row stride of R lines (mod 64; spread_stride's rule with nrows = 2).  The six rows a
// fill walks, i (0, a, b) and i-1 (R lines earlier), are kept >= 4 lines apart and, first,
// at most two to a class of line mod 4 (spread_rows' 256 B hazard).  Computed once.
struct TableLines { int a, b; };
inline TableLines table_lines(std::size_t R) {
    static const std::array<TableLines, 64> tbl = [] {
        std::array<TableLines, 64> t{};
        auto dist = [](int x) { x = ((x % 64) + 64) % 64; return x < 64 - x ? x : 64 - x; };
        for (int r = 0; r < 64; ++r) {
            TableLines best{21, 42}; int best_score = 1 << 30;
            for (int a = 1; a < 64; ++a)
                for (int b = a + 1; b < 64; ++b) {
                    const int v[6] = {0, a, b, -r, a - r, b - r};
                    int mind = 64, cls[4] = {0, 0, 0, 0};
                    for (int x = 0; x < 6; ++x) {
                        ++cls[((v[x] % 4) + 4) % 4];
                        for (int y = x + 1; y < 6; ++y) mind = std::min(mind, dist(v[x] - v[y]));
                    }
                    if (mind < 4) continue;
                    const int mc = std::max(std::max(cls[0], cls[1]), std::max(cls[2], cls[3]));
                    const int score = mc * 100 - mind;
                    if (score < best_score) { best_score = score; best = {a, b}; }
                }
            t[(std::size_t)r] = best;
        }
        return t;
    }();
    return tbl[R % 64];
}

// The default (and, for the soft path, only) buffer type.  Aliasing rather than
// renaming keeps every existing `DpBuffer` reference and the double build untouched.
using DpBuffer = DpBufferT<double>;
