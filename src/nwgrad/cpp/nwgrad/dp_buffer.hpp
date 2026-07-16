#pragma once

// DpBuffer — the DP table storage, split out of aligner.hpp so the leveled kernel
// TUs (which are compiled per ISA level) can see it without dragging in the whole
// Aligner template.  It is plain data: the only currency that crosses a level
// boundary (see simd_levels.hpp).

#include <cstddef>
#include <cstdlib>
#include <new>
#include <vector>
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

// All DP double storage uses it.  at/rat/band_fill take DVec& accordingly.
using DVec = std::vector<double, AlignedAllocator<double>>;

// ── DpBuffer ──────────────────────────────────────────────────────────────────
//
// Holds all DP table vectors for one Aligner computation.
// Lives either inside the Aligner (own_buf_) or externally (e.g. per-thread).
// Vectors grow on demand and are never implicitly freed; call clear() to release.

struct DpBuffer {
    DVec H;                        // linear viterbi
    DVec VM, VX, VY;               // affine viterbi
    DVec F, B;                     // linear forward-backward
    DVec FM, FX, FY, BM, BX, BY;  // affine forward-backward

    // Used only by the Simd kernel.  `prof` is the query profile — the
    // substitution scores of every alphabet symbol against sequence B, laid out
    // contiguously in j so the DP row loop loads them with a vector load instead
    // of a gather (Full mode).  `subbuf` is the per-row equivalent for banded
    // mode, where a full-width profile would cost more than the banded DP itself.
    DVec prof, subbuf;

    // Used only by the striped affine kernel (leveled, in kernels_impl.inl), which
    // writes VM/VX/VY directly in striped layout (no de-stripe copy).  `sopenv` is one
    // striped openv row (the VM/VX open values feeding the VY carry); `sprof` is the
    // query profile in striped order.  Both O(n), not O(m·n).
    DVec sopenv, sprof;

    void clear() noexcept {
        auto clr = [](DVec& v) noexcept { v.clear(); v.shrink_to_fit(); };
        clr(H);
        clr(VM); clr(VX); clr(VY);
        clr(F);  clr(B);
        clr(FM); clr(FX); clr(FY); clr(BM); clr(BX); clr(BY);
        clr(prof); clr(subbuf);
        clr(sopenv); clr(sprof);
    }
};
