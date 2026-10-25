# Handoff: packed direction codes for the affine Pointers traceback

*A journal page left for the next investigator. Written 2026-10-25 on branch `integer`
(HEAD `7ecb09f`). Nothing below has been implemented or measured yet. Every number
marked "expected" is a back-of-envelope figure, not a finding.*

## What I came to suspect

I went down into the striped Pointers kernel expecting a lean thing: two rolling score
rows and a byte or so of memory per cell. That is what is there, but it is not lean. It
lays down **three separate byte tables** — `buf.DM`, `buf.DX`, `buf.DY` (`dp_buffer.hpp`,
`BVec DM, DX, DY;`) — one byte per cell each, **3 B/cell**. Yet the information in a cell
is only 6 bits:

| table | values | meaning | written at |
|---|---|---|---|
| `DM` | 0, 1, 2, **3** | M's predecessor M/X/Y; 3 = Local stop (`VM <= 0`) | `kernels_pointers_impl.inl`, `store_codes(km, dM + s*W)` in the first pass |
| `DX` | 0, 1, 2 | X's predecessor M/X/Y (`pick_code(ax, bx, cx)`) | same pass, `store_codes(kx, dX + s*W)` |
| `DY` | 0, 1, 2 | Y: 0 = open from M, 1 = open from X, 2 = extend | the final per-row loop, after the carry and lazy-F (`store_codes(k, dY + s*W)`) |

Correct the record before going further: in conversation I had claimed "about 4 bits a
cell, so two cells a byte". That was wrong. X has three predecessors and Y's open carries
which state it came from, so the code is **6 bits, and nibble packing does not fit.**
The honest ladder:

1. **One byte per cell, `DM | DX << 2 | DY << 4` → 1 B/cell (3× smaller).** This is the
   experiment. It trades three store streams for one, so it may well be *faster* even
   inside cache.
2. Four cells in three bytes (6 bits each) → 0.75 B/cell (4×). Awkward to index from the
   walk and in a striped layout. Do it **only** if step 1 shows the time still tracking
   bytes moved.

The linear Pointers fill (`kernels_linear_impl.inl`, `viterbi_linear_ptr`) already uses
one byte table with 2-bit codes (0 diagonal, 1 up, 2 left, 3 stop): four cells a byte
would be its 4×. **Out of scope** here; note it as a follow-up.

## Why it might matter (the hypothesis to test, not assume)

AGENTS.md records the Pointers kernel peaking and then *regressing* as threads are added
on long pairs (nighthaven: peaks at 6 threads, falls to 3168 Mcell/s by 12; skynet:
peaks at 30t). The explanation given there is contention for DRAM bandwidth. A 2000×2000
pair is 12 MB of codes at 3 B/cell, 4 MB at 1 B/cell. Where this would show up:

- **Local affine at double, which `auto` still runs as Pointers** (Local Hirschberg costs
  1.1–1.8× per thread on real hardware; see "`auto` stays Pointers for Local"). This is the
  primary target.
- **`hb_cutoff`.** It was set to 512 partly because 2048 "re-enters the wall". Smaller
  tables could move that, which would speed up double Global (exact Hirschberg). A
  *follow-up* measurement only; do not change the default in this work.
- **Mid lengths dropping a cache level** (200×200 protein: 120 KB → 40 KB).

Where it will **not** help: short miRNA pairs (in L1 already, and the inter-pair fills
keep score tables, not codes); float32 Global (already `hirschberg_pmax`); the soft path
(no codes).

A warning from the memory notes (`nt_stores_rejected.md`): a previous "the DP is
bandwidth-saturated" claim proved false (non-temporal stores bought 1.2 %). So the first
thing to establish is whether the Pointers regression at high thread counts really is
bandwidth. Measure it; do not inherit it.

## Step 0 — confirm the premise (do this before writing code)

On nighthaven (x86-64-v3, the primary target), Local affine double, Pointers, L = 500,
1000, 1500, 2000, at 1, 6 and 12 threads:

- throughput (Mcell/s): `tools/bench_local_hb.py` already runs Pointers against Hirschberg
  over proteome-style pairs; reuse it or a trimmed copy;
- `perf stat -e LLC-load-misses,LLC-store-misses,...` (or `uncore_imc` DRAM bytes if
  exposed) per cell.

If the 6 → 12 thread regression is absent, or the DRAM traffic per cell is far below
3 B, **stop and report**: the idea has no lever. If it is present, carry on.

## Step 1 — one byte per cell

All three fills and the walk must change **together**: the walk reads one format, and
`scalar_fallback` (the scalar fill) is the bit-exactness reference every level is
compared against.

Writers:
- `src/nwgrad/cpp/nwgrad/kernels_pointers_impl.inl` — `striped_affine_full_ptr` (leveled;
  `level_common.inc` registers it as `striped_full_ptr_entry<T>`). `DY` is only final
  after the carry: lazy-F revises Y codes (`stdx::where(F > v, k) = vtwo`), and for an
  integer T with W >= 4 so does `int_pmax_carry`. So the combined byte can only be stored
  in the last per-row loop. Two options; measure both if unsure:
  (a) keep `km`/`kx` in T-typed scratch rows (as `yc` is now; add rows to the
      `spread_rows(buf.prows, rowsz, 11, rs)` block — keep the stride rule; see the 4 KB
      aliasing notes in AGENTS.md) and store `km + 4*kx + 16*ky` once through
      `store_codes` (values ≤ 63: the saturating packs there stay exact);
  (b) store `km | kx<<2` as a byte in the first pass and OR `ky<<4` into it in the last
      (read-modify-write of a row that is still in L1).
  Row 0 and column 0 borders are written by hand (`d0M[...] = 3; d0X = 0; d0Y = ...`,
  and the three per-row `buf.DM/DX/DY.data()[i * rowsz] = ...` lines). Fold those too.
- `src/nwgrad/cpp/nwgrad/aligner.hpp` — `viterbi_affine_ptr_impl` (scalar, row-major), the
  reference. It also runs when a level has no ptr kernel (`run_viterbi`, ~line 2131).
- Optional, a separate commit: `hb_kernel_impl.inl::hb_base_striped` writes `buf.hbD` as
  three planes (`3 * plane`), walked back in `aligner.hpp` (~2848–2891), and the scalar
  `hb_base` (~3426–3454). This is Hirschberg's base case: same idea, smaller stakes
  (one block ≤ `hb_cutoff` rows).

Reader:
- `aligner.hpp::walk_affine` (Ptr template branch, ~1870–1915) reads
  `dcode(buf.DM, i, j)`, `dcode(buf.DX, i, j)`, `dcode(buf.DY, i, j)` — all at the
  CURRENT cell, so one byte serves all three: `(byte >> shift) & 3`. `dcode` (~1814)
  goes through `cell_index`, which divides in the striped layout. If the walk reads the
  byte once per step, that is fewer divisions, not more. `hard_path_packed` and every
  other path consumer go through `walk_affine`; check with `grep -n dcode aligner.hpp`.

Hazards I saw:
- **The buffer is shared with the linear pointer fill**, which grows `DM` alone (the
  e2cac6b DX/DY overrun; the `for (auto* d : {&buf.DM, &buf.DX, &buf.DY})` guards in both
  fills). If affine moves into `DM`, `DX`/`DY` go unused for affine, but the guards and
  `tests/cpp/test_buffer_reuse.cpp` (one buffer through every type, band and traceback,
  sizes rising and falling) must still hold. A separate `BVec` for the packed table is
  the safer choice.
- **Tie order is the whole contract.** The codes must be the very `>=` choices, M > X > Y,
  with the Y open chosen on the ROUNDED opens (`om >= ox`; the 2026-10-19 fix). Packing
  must not recompute anything; it only relocates bits.
- **clang + AVX-512 + libstdc++** — every 64-bit-lane mask goes through the `avx512d_*`
  helpers (see the `#if defined(__clang__) && defined(__AVX512F__)` blocks). Any new mask
  op on double / int64 lanes must too, or it will not compile under clang.
- **gcc outlines small helpers** for 64-byte vectors: give any new helper
  `[[gnu::always_inline]]`, as `pick_code` / `sel_ge` / `store_codes` have.
- **Tracked problems** (`Tr = true`) share the kernel. The tracked tests must pass too.

## Verification (bit-identity, not tolerance)

Results must be **bit-identical to `main`**: scores, paths, gradients, guides, at every
ISA level and both compilers.
- `ctest` per level (sse2/sse4 on skynet, avx2 nighthaven, avx512 solace, neon spot),
  ASan+UBSan and TSan builds (commands in AGENTS.md, "Build & Test Commands").
- Python suite, in particular `tests/Python/test_traceback_modes.py` (Pointers ==
  Scores, bit for bit), `test_rounded_open_ties.py`, `test_hirschberg*.py` (base case),
  the tracks tests, `test_integer.py` (all five scalar types run this kernel).
- `tools/fuzz_ab.py`: the old build against the new, 0 differences required.
- Plant a mutant (e.g. swap two fields' shifts, or drop the lazy-F code revision) and
  confirm it is caught. AGENTS.md mentions `xlevel.py` / `xold.py` cross-level hashers:
  they are **not in `tools/`** (scratch scripts from earlier sessions), so write the
  equivalent if needed.

## Measurement (how to decide)

Release-wheel codegen only (the hardening-off flags in `~/.claude/CLAUDE.md`, verified to
reach cc1plus). Check Gentoo hosts' load first; `portage-freeze` only by its rules. Never
time skynet and wloczykij at the same time. A/B in alternating processes with a control
arm (`tools/bench_ab.py` pattern, two builds in two venvs). Mind the editable-install
trap: assert `compiled_with()` / the build you think you are timing.

Arms: explicit `traceback="pointers"`, affine, Local and Global, double and float32 and
int32, L = 100, 200, 500, 1000, 1500, 2000, threads 1 / half the cores / all.
Hosts: nighthaven (decides), then solace (AVX-512), spot (NEON; M1 had no memory wall,
so expect little), skynet / wloczykij (narrow 4-way L1 — rerun the 4 KB-aliasing length
sweep's suspicious lengths, since a new row in `prows` changes the block).

Adopt if: a clear win at L ≥ 1000 with ≥ 6 threads on nighthaven, and no regression
over ~2 % inside cache (L ≤ 500, 1 thread) on any host. If step 1 wins and the time
still tracks bytes, consider step 2. Re-measuring the `hb_cutoff` / Local `auto` choice
under the smaller tables is a separate decision; report it, do not flip it.

## Record-keeping

Commit in small units (single-line messages, no co-author lines; nwgrad is an autonomous
repo: commit and push without asking; no tags, no releases). Record the result —
including a negative one — in AGENTS.md beside the Pointers / memory-wall material
("Accepted — the winners", "The memory wall is real"), with hosts, dates and the control
arm, as the other sections do. Delete this file once its content lives there.
