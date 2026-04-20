import random
import numpy as np
import nwgrad

# ── Parameters ────────────────────────────────────────────────────────────────
N_SEQS      = 10
SEQ_LEN     = 40        # characters per sequence
SEED        = 42

GAP_MODEL   = "affine"  # "linear" | "affine"
GAP_OPEN    = 11.0      # ignored when GAP_MODEL="linear"
GAP_EXTEND  = 1.0
ALIGN_MODE  = "global"  # "global" (NW) | "local" (SW)
GRAD_MODE   = "hard"    # "hard"  — Viterbi subgradient (pair counts)
                        # "soft"  — forward-backward expected counts (log Z score)
                        # "none"  — score only, no gradient

BAND_WIDTH  = 15        # half-width for banded realignment in the second pass
LR          = 0.1       # step size: new_matrix = old_matrix + LR * grad

# ── Helpers ───────────────────────────────────────────────────────────────────
AA = list("ACDEFGHIKLMNPQRSTVWY")

def random_seq(length, rng):
    return "".join(rng.choices(AA, k=length))

def make_blosum62():
    """BLOSUM62 as a 20×20 float64 array in canonical AA order (ACDEFGHIKLMNPQRSTVWY)."""
    return np.array([
        [ 4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0],
        [-1, 9,-3,-4,-2,-3,-3,-1,-3,-1,-1,-3,-3,-3,-3,-1,-1,-2,-2,-1],
        [-2,-3, 6, 2,-3,-1,-1,-3,-1,-4,-3, 1,-1, 0,-2, 0,-1,-4,-3,-3],
        [-2,-4, 2, 5,-3,-2,-2,-3,-1,-3,-2, 0,-1, 2,-1, 0,-1,-3,-2,-2],
        [ 0,-2,-3,-3, 6,-3,-1, 0,-3, 0, 0,-3,-4,-3,-3,-2,-2,-1, 3,-1],
        [-1,-3,-1,-2,-3, 6,-2,-4,-2,-4,-3, 0,-2,-2,-2, 0,-2,-2,-3,-3],
        [-1,-3,-1,-2,-1,-2, 8,-3,-1,-3,-2, 1,-2, 0, 0,-1,-2,-2, 2,-3],
        [ 0,-1,-3,-3, 0,-4,-3, 4,-3, 2, 1,-3,-3,-3,-3,-2,-1,-3,-1, 3],
        [-2,-3,-1,-1,-3,-2,-1,-3, 5,-2,-1, 0,-1, 1, 2, 0,-1,-3,-2,-2],
        [-1,-1,-4,-3, 0,-4,-3, 2,-2, 4, 2,-3,-3,-2,-2,-2,-1,-2,-1, 1],
        [-1,-1,-3,-2, 0,-3,-2, 1,-1, 2, 5,-2,-2, 0,-1,-1,-1,-1,-1, 1],
        [-1,-3, 1, 0,-3, 0, 1,-3, 0,-3,-2, 6,-2, 0, 0, 1, 0,-4,-2,-3],
        [-1,-3,-1,-1,-4,-2,-2,-3,-1,-3,-2,-2, 7,-1,-2,-1,-1,-4,-3,-2],
        [-2,-3, 0, 2,-3,-2, 0,-3, 1,-2, 0, 0,-1, 5, 2,-1,-1,-2,-1,-1],
        [-1,-3,-2,-1,-3,-2, 0,-3, 2,-2,-1, 0,-2, 2, 5,-1,-1,-3,-2,-1],
        [ 1,-1, 0, 0,-2, 0,-1,-2, 0,-2,-1, 1,-1,-1,-1, 4, 1,-3,-2,-2],
        [ 0,-1,-1,-1,-2,-2,-2,-1,-1,-1,-1, 0,-1,-1,-1, 1, 5,-2,-2, 0],
        [-3,-2,-4,-3,-1,-2,-2,-3,-3,-2,-1,-4,-4,-2,-3,-3,-2,11, 2,-3],
        [-2,-2,-3,-2, 3,-3, 2,-1,-2,-1,-1,-2,-3,-1,-2,-2,-2, 2, 7,-1],
        [ 0,-1,-3,-2,-1,-3,-3, 3,-2, 1, 1,-3,-2,-1,-1,-2, 0,-3,-1, 4],
    ], dtype=np.float64)

# ── Setup ─────────────────────────────────────────────────────────────────────
rng = random.Random(SEED)
seqs = [random_seq(SEQ_LEN, rng) for _ in range(N_SEQS)]

blosum_arr = make_blosum62()
mat = nwgrad.BlosumMatrix(blosum_arr)

score_label = "log Z" if GRAD_MODE == "soft" else "score"
print(f"Config: {GAP_MODEL} gap, {ALIGN_MODE} align, {GRAD_MODE} grad")
print(f"        gap_open={GAP_OPEN}  gap_extend={GAP_EXTEND}  band={BAND_WIDTH}  lr={LR}")

# ── Build upper-triangle pairs ────────────────────────────────────────────────
batch = nwgrad.SeqPairBatch()
for i in range(N_SEQS):
    for j in range(i + 1, N_SEQS):
        batch.add(nwgrad.SeqPair(
            seqs[i], seqs[j], mat,
            gap_open=GAP_OPEN, gap_extend=GAP_EXTEND,
            gap_model=GAP_MODEL, mode=ALIGN_MODE, grad_mode=GRAD_MODE,
        ))

print(f"\nSequence pairs: {len(batch)}  ({N_SEQS} sequences, upper triangle)")
print(f"Threads: {batch.n_threads}")

# ── Pass 1: full alignment + gradient, using thread-owned DP buffers ──────────
#
# score_and_grad() runs the full DP (and optionally a banded DP) on every pair
# in parallel, using one DpBuffer per thread.  The per-pair SeqPair objects store
# the resulting score, alignment path, and gradient, but their own O(mn) DP tables
# are never allocated — dp_valid stays False after this call.
#
total_score = batch.score_and_grad()
print(f"\nSum of {score_label} (full DP, pass 1): {total_score:.3f}")

if GRAD_MODE != "none":
    grad_sum = batch.compute_grad()   # sums the cached per-pair gradients; no DP needed
    print(f"Gradient sum — min: {grad_sum.min():.3f}  max: {grad_sum.max():.3f}  "
          f"total: {grad_sum.sum():.1f}")

    # gradient step: update the substitution matrix
    new_blosum_arr = blosum_arr + LR * grad_sum
    new_mat = nwgrad.BlosumMatrix(new_blosum_arr)
    batch.set_matrix(new_mat)
else:
    print("(grad_mode=none — skipping gradient step, reusing matrix)")
    new_mat = mat

# ── Pass 2: banded alignment + gradient under the updated matrix ───────────────
#
# score_and_grad(bandwidth=BAND_WIDTH) first runs a full viterbi on each pair to
# get its guide path, then runs a banded DP of half-width BAND_WIDTH around that
# path.  Again, all DP work uses per-thread buffers; pair DP tables stay empty.
#
total_score_banded = batch.score_and_grad(bandwidth=BAND_WIDTH)
print(f"\nSum of {score_label} (banded, new mat, pass 2): {total_score_banded:.3f}")
print(f"Delta vs pass 1: {total_score_banded - total_score:+.3f}")

if GRAD_MODE != "none":
    grad_sum2 = batch.compute_grad()
    print(f"Gradient sum — min: {grad_sum2.min():.3f}  max: {grad_sum2.max():.3f}  "
          f"total: {grad_sum2.sum():.1f}")
