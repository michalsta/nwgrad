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

BAND_WIDTH  = 15        # half-width for banded realignment in the second pass
LR          = 1e-3      # step size: new_params = params + LR * grad.  The gradient sums
                        # counts over every pair, so keep LR small: at 0.1 the gap
                        # costs step below zero, gaps become rewards, and the "best"
                        # alignment is all gaps.

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
params = nwgrad.AlignParams(
    blosum_arr,
    gap_open_a=GAP_OPEN, gap_extend_a=GAP_EXTEND,
    gap_open_b=GAP_OPEN, gap_extend_b=GAP_EXTEND,
)

score_label = "log Z" if GRAD_MODE == "soft" else "score"
print(f"Config: {GAP_MODEL} gap, {ALIGN_MODE} align, {GRAD_MODE} grad")
print(f"        gap_open={GAP_OPEN}  gap_extend={GAP_EXTEND}  band={BAND_WIDTH}  lr={LR}")

# ── Build upper-triangle pairs ────────────────────────────────────────────────
batch = nwgrad.SeqPairBatch(gap_model=GAP_MODEL, mode=ALIGN_MODE, grad_mode=GRAD_MODE)
ii, jj = np.triu_indices(N_SEQS, k=1)
batch.add_many([seqs[i] for i in ii], [seqs[j] for j in jj], params)

print(batch[7])
print(f"\nSequence pairs: {len(batch)}  ({N_SEQS} sequences, upper triangle)")
print(f"Threads: {batch.n_threads}")

# ── Pass 1: full alignment + gradient, using thread-owned DP buffers ──────────
#
# score_and_grad() runs the full DP on every pair in parallel, using one DP buffer
# per thread.  The batch keeps each pair's score, alignment path (guide) and
# gradient — a few hundred bytes per pair — but not the DP tables.
#
total_score = batch.score_and_grad()
print(f"\nSum of {score_label} (full DP, pass 1): {total_score:.3f}")

grad_sum = batch.compute_grad()   # sums the cached per-pair gradients; no DP needed
print(grad_sum)
grad_mat_arr = grad_sum.matrix.to_matrix()
print(f"Gradient sum — min: {grad_mat_arr.min():.3f}  max: {grad_mat_arr.max():.3f}  "
      f"total: {grad_mat_arr.sum():.1f}")

# gradient step: update the alignment parameters
new_params = params + LR * grad_sum
batch.set_params(new_params)

# ── Pass 2: banded alignment + gradient under the updated matrix ───────────────
#
# banded_grad(BAND_WIDTH) re-aligns every pair under the new params in a band of
# half-width BAND_WIDTH around the path pass 1 found — no full DP.
#
total_score_banded = batch.banded_grad(BAND_WIDTH)
print(f"\nSum of {score_label} (banded, new params, pass 2): {total_score_banded:.3f}")
print(f"Delta vs pass 1: {total_score_banded - total_score:+.3f}")

grad_sum2 = batch.compute_grad()
grad_mat_arr2 = grad_sum2.matrix.to_matrix()
print(f"Gradient sum — min: {grad_mat_arr2.min():.3f}  max: {grad_mat_arr2.max():.3f}  "
      f"total: {grad_mat_arr2.sum():.1f}")
