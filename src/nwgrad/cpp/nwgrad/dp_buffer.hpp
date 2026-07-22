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

// The always-double DP storage (forward-backward / soft path) uses it.
using DVec = std::vector<double, AlignedAllocator<double>>;

// Direction-pointer storage for the variant-B Viterbi (see aligner.hpp).  One byte
// per cell per state instead of a retained score table: 3 B/cell against 12 (float32)
// or 24 (double), which is the whole point of B.
using BVec = std::vector<unsigned char, AlignedAllocator<unsigned char>>;

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
    TVec VM, VX, VY;               // affine viterbi
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

    void clear() noexcept {
        auto clrT = [](TVec& v) noexcept { v.clear(); v.shrink_to_fit(); };
        auto clrD = [](DVec& v) noexcept { v.clear(); v.shrink_to_fit(); };
        clrT(H);
        clrT(VM); clrT(VX); clrT(VY);
        clrD(F);  clrD(B);
        clrD(FM); clrD(FX); clrD(FY); clrD(BM); clrD(BX); clrD(BY);
        clrT(prof); clrT(subbuf);
        clrT(sopenv); clrT(sprof);
        auto clrB = [](BVec& v) noexcept { v.clear(); v.shrink_to_fit(); };
        clrB(DM); clrB(DX); clrB(DY);
        clrT(rM); clrT(rX); clrT(rY); clrT(qM); clrT(qX); clrT(qY);
    }
};

// The default (and, for the soft path, only) buffer type.  Aliasing rather than
// renaming keeps every existing `DpBuffer` reference and the double build untouched.
using DpBuffer = DpBufferT<double>;
