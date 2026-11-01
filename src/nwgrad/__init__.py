# Precision.  The default Viterbi/hard-gradient precision of the Python surface is
# float32 — the plain names (SeqPair, SeqPairBatch, nw_affine_grad, ...)
# run the float32 kernel: ~1.6-3x the double kernel's throughput, biggest under the
# memory wall (long sequences, many threads), and the correct precision for feeding an
# ML optimizer (which trains in float32 anyway).  For an integer-valued matrix the score
# is exact; at score ties float32 may pick a different — equally optimal — path, so its
# hard subgradient can differ from double's by a valid subgradient.  Where you need the
# double reference, use the *Double classes and _double-suffixed functions below.  The
# precision is chosen by the NAME you call, never inferred from any array's dtype.  (The
# soft-gradient path is double either way — log-sum-exp stays double internally — so the
# _double soft functions are numerically identical to the plain ones.)
from .nwgrad_ext import (
    Alphabet,
    DNA, DNA_N, RNA, RNA_N,
    PROTEIN, PROTEIN_X, PROTEIN_UO, PROTEIN_UOX,
    NCBI_PROTEIN, IUPAC_DNA,
    SubstMatrix, AlignParams,
    guide_j_from_aligned,
    simd_isa, compiled_with,
    available_isa_levels, get_isa_level, set_isa_level,
    # float32 (default)
    nw_score, sw_score, nw_score_affine, sw_score_affine,
    nw_grad, sw_grad, nw_affine_grad, sw_affine_grad,
    nw_soft_grad, sw_soft_grad, nw_affine_soft_grad, sw_affine_soft_grad,
    glocal_score, glocal_score_affine, glocal_grad, glocal_affine_grad,
    glocal_soft_grad, glocal_affine_soft_grad,
    BatchResult,
    SeqPair, SeqPairBatch,
    # double (explicit)
    nw_score_double, sw_score_double, nw_score_affine_double, sw_score_affine_double,
    nw_grad_double, sw_grad_double, nw_affine_grad_double, sw_affine_grad_double,
    nw_soft_grad_double, sw_soft_grad_double,
    nw_affine_soft_grad_double, sw_affine_soft_grad_double,
    glocal_score_double, glocal_score_affine_double, glocal_grad_double, glocal_affine_grad_double,
    glocal_soft_grad_double, glocal_affine_soft_grad_double,
    SeqPairDouble, SeqPairBatchDouble,
    # integer scalar types (explicit): integral parameters only, no soft path.  Every
    # hard-path result equals the double one on the same parameters, bit for bit; the
    # prefix-max Hirschberg carry is exact in integers, so it is their affine default.
    # int16 refuses problems whose a-priori bound (m + n + 66) * max_step exceeds 8191,
    # int32 2^30 - 1, int64 2^62 - 1 (OverflowError); a non-integral parameter raises
    # ValueError.
    nw_score_int16, sw_score_int16, nw_score_affine_int16, sw_score_affine_int16,
    nw_grad_int16, sw_grad_int16, nw_affine_grad_int16, sw_affine_grad_int16,
    nw_score_int32, sw_score_int32, nw_score_affine_int32, sw_score_affine_int32,
    nw_grad_int32, sw_grad_int32, nw_affine_grad_int32, sw_affine_grad_int32,
    nw_score_int64, sw_score_int64, nw_score_affine_int64, sw_score_affine_int64,
    nw_grad_int64, sw_grad_int64, nw_affine_grad_int64, sw_affine_grad_int64,
    glocal_score_int16, glocal_score_affine_int16, glocal_grad_int16, glocal_affine_grad_int16,
    glocal_score_int32, glocal_score_affine_int32, glocal_grad_int32, glocal_affine_grad_int32,
    glocal_score_int64, glocal_score_affine_int64, glocal_grad_int64, glocal_affine_grad_int64,
    SeqPairInt16, SeqPairBatchInt16,
    SeqPairInt32, SeqPairBatchInt32,
    SeqPairInt64, SeqPairBatchInt64,
)
from . import matrices

from importlib.metadata import version as _version
__version__ = _version("nwgrad")
del _version
