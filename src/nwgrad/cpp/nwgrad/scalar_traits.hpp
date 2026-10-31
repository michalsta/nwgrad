#pragma once

// ── The scalar type of the Viterbi / hard-gradient DP ─────────────────────────
//
// Aligner<..., T> runs its DP in T: double, float, or (since the integer variants) a
// signed integer — int16_t, int32_t, int64_t.  The two families differ in exactly one
// value the fills depend on, the "unreachable" sentinel:
//
//   floating point   -inf.  Arithmetic on it stays -inf (a finite penalty subtracted
//                    from -inf is -inf), so an unreachable cell can never win a max,
//                    whatever chain of moves is applied to it.
//   integer          lowest()/2.  There is no infinity, and lowest() itself would wrap
//                    on the first `- ge`.  Halving leaves |lowest|/2 on either side of
//                    the sentinel.  set_problem() then refuses a problem whose a-priori
//                    bound (m+n+2+64) * max step exceeds nw_int_cap = |lowest|/4 - 1 (the
//                    64: a striped kernel's padding columns, see nw_pad), so every value
//                    the DP can form stays within |lowest|/4 of EITHER zero
//                    (a reachable score: a path has at most m+n+2 steps) OR the sentinel
//                    (an unreachable value drifts by at most that many steps too).  Two
//                    consequences carry the whole design: nothing wraps, and an
//                    unreachable value — however it drifted — stays below every reachable
//                    one, so it can never win a max.  The quarter rather than the half is
//                    what makes the second hold without reasoning about which cells drift
//                    which way, and what lets two in-range values be ADDED (the prefix-max
//                    ramp g[k] + k*ge_a, a Hirschberg join's forward + reverse score)
//                    without wrapping either.
//
// Everything on the hard path reads the sentinel through nw_ninf<T>; the soft path is
// double-only and keeps its own -inf.  Integer arithmetic is exact and associative,
// so the floating-point care taken elsewhere (left association, no FMA, the pmax
// carry's rounding) is moot for an integer T: every traceback mode, every ISA level
// and the prefix-max carry are then bit-identical by construction.

#include <algorithm>
#include <cstdint>
#include <limits>
#include <type_traits>

template <class T>
inline constexpr bool nw_is_int = std::is_integral_v<T>;

// v, opaque to clang's optimizer (an empty asm on a register: no instruction).  The scalar
// Local linear fills clamp an integer cell as max(max(diag, up), 0) and only then take the
// left carry, so the carry passes through one max, not a max and the clamp — gcc makes
// that rewrite itself (int64 L300 Local hard on nighthaven, 2 cmovs on the carry against
// clang's 4: clang 0.55x gcc).  Integer max being exact and associative, it is the same
// value; but clang reassociates the smax chain and puts the 0 back on the carry, unless
// the clamped value is pinned.  gcc: a plain copy.
template <class T>
[[gnu::always_inline]] inline T nw_opaque(T v) noexcept {
#if defined(__clang__)
    __asm__("" : "+r"(v));
#endif
    return v;
}

// int16 (nw_sat) has a second regime past the quarter rule: SATURATING arithmetic.  Every
// add / subtract on a DP value then clamps at the type's bottom (vector: vadd / vsub in
// the level kernels, the packed saturating instructions vpaddsw / vpsubsw, sqadd / sqsub;
// scalar: nw_add / nw_sub below), so an unreachable value can never wrap, however far it
// drifts -- it comes to rest at lowest() -- and climbs at most by the gains a path can
// collect.  The bound (nw_int_bound_check, aligner.hpp) then asks only that no REACHABLE
// value leaves the range and that a sentinel-derived value stays below every reachable
// one, which takes BLOSUM62 pairs from ~370 to ~1000 residues.  The regime is chosen per
// problem (Aligner::sat_, ViterbiJob::sat ...): a problem inside the quarter rule runs
// the plain kernels it always ran, bit for bit (the saturating instructions issue on
// fewer ports on Intel cores: 6-19 % on the int16 kernels, nighthaven 2026-10-30), a
// problem past it the saturating variant of the same kernel.  The saturating regime's
// sentinel is lowest() itself (nw_ninf_s; the aligner's NINF member follows the regime),
// self-preserving under any penalty, so the whole range below the reachable values is the
// sentinel's depth; the plain regime keeps lowest() / 2.
// The scalar fills saturate always (a clamp at the bottom: a compare and a cmov), so a
// table is the same bits at every level in either regime.  int32 / int64 have no
// saturating regime (x86 has no 32 / 64-bit saturating vector ops): the quarter rule.
template <class T>
inline constexpr bool nw_sat = std::is_integral_v<T> && sizeof(T) == 2;

template <class T>
inline constexpr T nw_ninf = std::is_floating_point_v<T>
    ? -std::numeric_limits<T>::infinity()
    : static_cast<T>(std::numeric_limits<T>::lowest() / 2);

// The sentinel of a kernel variant: lowest() in the saturating regime (Sat, an nw_sat T),
// nw_ninf<T> otherwise.
template <class T, bool Sat>
inline constexpr T nw_ninf_s = (Sat && nw_sat<T>) ? std::numeric_limits<T>::lowest() : nw_ninf<T>;

// Scalar add / subtract on DP values: for nw_sat types computed in int and clamped at the
// bottom (the top is never reached: the bound keeps every reachable value below it, and
// a sentinel-derived value below those), the plain operation otherwise.  The floating
// types pass through unchanged, so a kernel can spell every site with these.
template <class T>
[[gnu::always_inline]] inline constexpr T nw_add(T a, T b) noexcept {
    if constexpr (nw_sat<T>) return static_cast<T>(std::max(int(a) + int(b), int(std::numeric_limits<T>::lowest())));
    else return static_cast<T>(a + b);
}
template <class T>
[[gnu::always_inline]] inline constexpr T nw_sub(T a, T b) noexcept {
    if constexpr (nw_sat<T>) return static_cast<T>(std::max(int(a) - int(b), int(std::numeric_limits<T>::lowest())));
    else return static_cast<T>(a - b);
}

// The regime-selected form: Sat clamps (the saturating regime), !Sat is the plain
// operation (the quarter rule guarantees it never leaves the range, so the clamp would
// never fire: the same bits, without the compare and cmov per step -- which cost the
// scalar fills 1.4-1.5x at int16 when every site clamped, nighthaven and spot 2026-10-31).
template <bool Sat, class T>
[[gnu::always_inline]] inline constexpr T nw_add_s(T a, T b) noexcept {
    if constexpr (Sat && nw_sat<T>) return nw_add<T>(a, b);
    else return static_cast<T>(a + b);
}
template <bool Sat, class T>
[[gnu::always_inline]] inline constexpr T nw_sub_s(T a, T b) noexcept {
    if constexpr (Sat && nw_sat<T>) return nw_sub<T>(a, b);
    else return static_cast<T>(a - b);
}

// The largest magnitude the a-priori bound (m+n+2) * max step may reach for an integer
// T (see above).  Floating types have no cap (infinity).
template <class T>
inline constexpr T nw_int_cap = std::is_floating_point_v<T>
    ? std::numeric_limits<T>::infinity()
    : static_cast<T>(-(std::numeric_limits<T>::lowest() / 4) - 1);

// The substitution score of a striped kernel's PADDING column (a column past the end of
// B that fills out the last vector).  Floating T pads with -inf, which makes every
// padding cell -inf whatever is added to it.  An integer T cannot: a padding cell would
// be computed as sentinel + sentinel and wrap on the next penalty.  So an integer T pads
// with 0 — the padding columns then behave as an imaginary tail of B that scores 0,
// every one of their values an ordinary in-range DP value (which is why the bound above
// allows 64 columns of padding: nw_int_pad_cols) — and every argmax that could read a
// padding cell masks it (the Local row best, the score kernels, the endpoint scan).  The
// padding can never feed a real cell: it lies to the right of every real column, and
// the DP flows right and down.
template <class T>
inline constexpr T nw_pad = std::is_floating_point_v<T>
    ? -std::numeric_limits<T>::infinity() : T(0);
inline constexpr int nw_int_pad_cols = 64;

// The scalar type's name, for messages and the Python surface.
template <class T> constexpr const char* nw_type_name() noexcept {
    if constexpr (std::is_same_v<T, double>)       return "double";
    else if constexpr (std::is_same_v<T, float>)   return "float32";
    else if constexpr (sizeof(T) == 2)             return "int16";
    else if constexpr (sizeof(T) == 4)             return "int32";
    else                                           return "int64";
}

// The unsigned integer of T's width, for bitwise equality / bit tricks on a T.
template <class T>
using nw_bits_t = std::conditional_t<sizeof(T) == 8, std::uint64_t,
                  std::conditional_t<sizeof(T) == 4, std::uint32_t, std::uint16_t>>;
static_assert(sizeof(nw_bits_t<double>) == 8 && sizeof(nw_bits_t<float>) == 4 &&
              sizeof(nw_bits_t<std::int16_t>) == 2);
