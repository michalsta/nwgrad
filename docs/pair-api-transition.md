# Pair API transition

**Status:** done. Release N was 0.7.0; the N+1 removal is on `main` for 0.8.0
(2026-10-20; see TODO.md).

This plan covers the remaining `SeqPair` lifecycle API, for both precisions and
for standalone pairs and batch views. Release N introduces the final API and
deprecates the compatibility API. Release N+1 removes compatibility completely.
Use the actual release numbers in warnings and documentation once assigned:
the current tree still declares version 0.6.0, despite already removing APIs
previously deprecated in 0.6.0. This plan does not restore those removed batch
APIs.

## Final API introduced in release N

| Method/property | Contract |
|---|---|
| `align(keep_paths=True)` | Full computation; returns the score. Publishes the score and configured hard/soft gradient immediately. Works with `grad_mode="none"`, with no gradient. |
| `banded_align(bandwidth, keep_paths=True)` | Computation around the cached guide; returns the score and publishes the configured gradient immediately. Works with `grad_mode="none"`. Requires an established guide. |
| `score_and_grad(keep_paths=True)` | Existing convenience returning `(score, grad)` for hard/soft modes. Preserve its default and its error for `grad_mode="none"`; implement through the same full computation. |
| `drop_paths()` | Release the stored alignment path. Preserve current scores, gradients and guides. Preserve the existing scratch-memory release behavior of `drop_dp()`. |
| `stored_path_valid` | Whether `aligned()` and `coordinates()` can read a stored path. |

Keep `score`, `grad`, `score_valid`, `grad_valid`, `guide_j`, `path_valid`,
`aligned()`, `coordinates()` and `set_params()`. Document `path_valid` explicitly
as guide validity; it is distinct from stored-path availability. Python `grad`
remains `None` when no gradient is available. Keep existing C++ accessors and
return types, including the throwing gradient accessor.

The generic `align` methods let score-only users migrate without inventing a
dummy gradient or changing the existing `score_and_grad` return contract.
Pair `align()` operates on its stored problem; batch `align(seqs_a, seqs_b, ...)`
continues to be the streaming operation. Their signatures distinguish them.

New methods have identical semantics on a batch view and a standalone pair.
Computing a view updates that row only. Parameters/settings on a view continue
to belong to its batch. Path retention defaults to True on pairs, preserving
the existing convenience behavior; batches retain their False default.

## Release N: retain compatibility at the facade

| Deprecated pair API | Replacement | Compatibility behavior during N |
|---|---|---|
| `align_full()` | `align()`; or `score_and_grad()` when consuming both results | Compute and store the path, then hide the gradient until release. |
| `realign_banded(bw)` | `banded_align(bw)` | Compute and store the path, then hide the gradient until release. |
| `compute_grad()` | Read `grad` after a modern computation, or consume the `score_and_grad()` result | Release the cached deferred gradient with the existing validation order. |
| `drop_dp()` | `drop_paths()` | Delegate, preserving the old invalidation and memory-release behavior. |
| `dp_valid` | `stored_path_valid` | Read the same stored-path flag. |

Warn on public deprecated calls with Python `DeprecationWarning` attributed to
the caller and C++ `[[deprecated("...replacement and removal release...")]]`.
Warnings state both the replacement and the removal release. Internal calls
must not emit warnings or generate duplicate warnings for one public call.

Compatibility is behavior, not just an alias: `align_full()` followed by reading
`grad` must still show no gradient; `compute_grad()` publishes it. Dropping the
path before release still prevents release. Dropping it after release preserves
the gradient. Parameter changes invalidate results and retain the guide.
Failed scoring fills retain the new safety rule: cached results are invalidated.

Build the final computation path first. Remove `hold_grad` arguments and
`defer_grad_` from the engine's fill routines in release N. Put the small
hide/release operations and legacy error messages in an explicitly temporary
compatibility helper used by the old facade methods. Reuse existing validity
flags; add no held-gradient flag, duplicate gradient buffer or per-pair object.
Keep the normal modern path free of compatibility switches.

During N, batch `compute_grad()` must continue to release gradients hidden by
deprecated operations on its views before summing. Isolate that bridge as
temporary compatibility code. Batch `weighted_grad()` and `grads()` retain
their existing requirement for published gradients.

Switch all examples, tutorials, benchmarks and non-compatibility tests to the
final methods in N. Keep old names only in migration documentation, warning
tests and focused compatibility regressions. Consumer acceptance uses the
unmodified DiscrimAlign API-port PR #16 state.

## Release N+1: remove compatibility and simplify

Delete the five deprecated pair names, their bindings and warnings, the
hide/release helper, and the batch reduction bridge. Remove compatibility tests
and old lifecycle/error text; retain modern behavioral regressions. Do not keep
no-op shims, hidden aliases, constructor flags or version-dependent behavior.

Then the invariant is simple: after a successful computation, score is valid
and gradient is valid exactly when the configured mode computes one. Stored
path validity is independent. Batch `compute_grad()` validates published
gradients and sums them; it never changes pair validity. No operation computes
a gradient and deliberately hides it.

Audit the engine and bindings after deletion for unused helpers, legacy names,
forced path retention and validity scans whose sole purpose was release. Retain
real validation required for mixed current/stale rows; removing the bridge does
not justify removing correctness checks. Measure any claimed speed gain with
control arms. Do not promise a DP speedup from deleting facade code.

## Implementation order and release gates

1. Add final methods, bindings and documentation, with modern lifecycle tests.
2. Move compatibility behavior out of fills; deprecate old facade names and add
   focused warning/lifecycle tests.
3. Migrate repository callers and publish the release-specific migration table.
4. Validate release N: full C++/Python suites, ASan/UBSan, scalar and available
   SIMD levels including NEON on spot, and the 16 PR #16 acceptance fits.
5. In N+1, delete compatibility in one cohesive change, run the same validation,
   and confirm only modern paths remain.

Tests must cover both precisions, hard/soft/none modes, full and banded fills,
path retention on/off, empty sequences, both standalone pairs and views,
parameter invalidation, dropped paths, failed-fill recovery, and mixed modern
and deprecated calls during N. In particular, a modern fill replaces a deferred
result with a published one; a deprecated fill hides its new result even after
a previous modern fill. Verify reductions after deprecated operations on views.

The wheels workflow and the outstanding AVX-512 release checks in
`release-validation.md` remain release gates independent of this transition.
