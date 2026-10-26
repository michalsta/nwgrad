# gridgrad — grid-checkpoint hard gradient (prototype)

A prototype, not part of the library: the hard gradient (and the path) of affine Global
alignment at about the cost of a score-only fill, bit-identical to `traceback="pointers"`.

1. One forward pass over rolling rows (a copy of the D/X score kernel) saves every R-th row
   and a set of segment-aligned columns, all three states.
2. The saved lines cut the DP into blocks with exact borders. The walk back from (m, n)
   re-fills only the blocks the path crosses (~2/k of the cells for a k x k grid), records
   predecessor codes there and counts the gradient.
3. Version A: one grid level, blocks re-filled by a scalar pointer fill (block buffer
   ~mn/k^2). Version B: blocks re-filled by the same vector kernel, recursively, down to a
   small base block (memory O(k (m + n))).

Every re-filled cell is computed from the same inputs by the same expressions as the full
fill, so ties break as Pointers breaks them: `bench_grid test` checks score and gradient
bit for bit against the scalar Pointers fill (and a mutated tie-break fails it).

Scope: affine Global, untracked, opens >= 0, float / double, x86 levels only (no NEON TU).

```
./build.sh                      # release-wheel codegen, nwgrad's level TUs linked in
./bench_grid test 500           # exactness, every kernel variant, versions A and B
./bench_grid bench 5            # ns/cell vs nwgrad's score-only / pointers / hirschberg(_pmax)
./bench_grid scan 5             # forward kernel alone vs nwgrad's score-only, by length
./bench_grid prof L f32|f64 k reps [variant]   # one pair, for perf
NWGRAD_ISA=avx2 ./bench_grid …  # both kernels at one level (e.g. 8 lanes on an AVX-512 host)
```

Forward-kernel variants (`gk_impl.inl`, all bit-identical): v0 the plain copy, v1 max(M, X)
rows (MX), v2 the two-segment gap carry (Two), v3 Two + MX, v4 Two + MX + the lazy-F carry
`F - ge` (Sub), v5 Two + Sub. Those carry changes are what the library's score kernels and
fills took over; see AGENTS.md, "The gap carry, shortened".

Measured (version B, k = 32, v3 before Sub, 1 thread, protein, x score-only): nighthaven
(AVX2) 0.72-0.88 at L >= 1000 against Pointers 1.6-2.1 and hirschberg_pmax 1.2-2.1;
`scalar_traits`-era library kernels (main 491fb05) are not re-measured here.
