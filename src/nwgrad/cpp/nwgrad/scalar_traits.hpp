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
//                    bound (m+n+2) * max step exceeds nw_int_cap = |lowest|/4 - 1, so
//                    every value the DP can form stays within |lowest|/4 of EITHER zero
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

#include <cstdint>
#include <limits>
#include <type_traits>

template <class T>
inline constexpr bool nw_is_int = std::is_integral_v<T>;

template <class T>
inline constexpr T nw_ninf = std::is_floating_point_v<T>
    ? -std::numeric_limits<T>::infinity()
    : static_cast<T>(std::numeric_limits<T>::lowest() / 2);

// The largest magnitude the a-priori bound (m+n+2) * max step may reach for an integer
// T (see above).  Floating types have no cap (infinity).
template <class T>
inline constexpr T nw_int_cap = std::is_floating_point_v<T>
    ? std::numeric_limits<T>::infinity()
    : static_cast<T>(-(std::numeric_limits<T>::lowest() / 4) - 1);

// The unsigned integer of T's width, for bitwise equality / bit tricks on a T.
template <class T>
using nw_bits_t = std::conditional_t<sizeof(T) == 8, std::uint64_t,
                  std::conditional_t<sizeof(T) == 4, std::uint32_t, std::uint16_t>>;
static_assert(sizeof(nw_bits_t<double>) == 8 && sizeof(nw_bits_t<float>) == 4 &&
              sizeof(nw_bits_t<std::int16_t>) == 2);
