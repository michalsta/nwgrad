#pragma once

// DpBuffer — the DP table storage, split out of aligner.hpp so the leveled kernel
// TUs (which are compiled per ISA level) can see it without dragging in the whole
// Aligner template.  It is plain data: the only currency that crosses a level
// boundary (see simd_levels.hpp).

#include <cstddef>
#include <new>
#include <vector>

// ── 64-byte-aligned allocator for the DP tables ───────────────────────────────
//
// std::vector guarantees only 16-byte alignment, so on AVX2 every 256-bit (32-byte)
// W-wide load/store into VM/VX/VY is misaligned and some split cache lines — a
// measured ~15-18% loss on AVX2 (nothing on SSE2/NEON, where W=2 is 16 bytes and
// 16-byte alignment already suffices).  A 64-byte base plus per-row padding to a
// multiple of W (see rowsz in kernels_impl.inl / cell_index in aligner.hpp) keeps
// every striped row's W-wide accesses off cache-line boundaries.  Uses C++17 aligned
// operator new, so it is portable (no posix_memalign / _aligned_malloc split).
template <class T, std::size_t Align = 64>
struct AlignedAllocator {
    using value_type = T;
    static constexpr std::align_val_t kAlign{Align};

    AlignedAllocator() noexcept = default;
    template <class U> AlignedAllocator(const AlignedAllocator<U, Align>&) noexcept {}

    T* allocate(std::size_t n) {
        if (n == 0) return nullptr;
        return static_cast<T*>(::operator new(n * sizeof(T), kAlign));
    }
    void deallocate(T* p, std::size_t) noexcept { ::operator delete(p, kAlign); }

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
