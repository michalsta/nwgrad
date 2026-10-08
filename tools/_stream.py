"""Streaming align over either API: BatchAligner (nwgrad < 0.6) or SeqPairBatch.align().

The benchmark tools time the same call on an old build and a new one (A/B), so they go
through this instead of either class directly.
"""
import nwgrad


class Stream:
    def __init__(self, params, band=0, gap_model="affine", mode="global", grad_mode="hard",
                 n_threads=1, kernel="auto", double=False):
        self.params, self.band, self.kernel = params, band, kernel
        if hasattr(nwgrad, "BatchAligner"):   # pre-0.6
            cls = nwgrad.BatchAlignerDouble if double else nwgrad.BatchAligner
            self._ba = cls(params, band=band, gap_model=gap_model, mode=mode,
                           grad_mode=grad_mode, n_threads=n_threads, kernel=kernel)
            self._b = None
        else:
            cls = nwgrad.SeqPairBatchDouble if double else nwgrad.SeqPairBatch
            self._b = cls(n_threads, gap_model=gap_model, mode=mode, grad_mode=grad_mode)
            self._ba = None

    def align(self, seqs_a, seqs_b, aligned_a=(), aligned_b=()):
        if self._ba is not None:
            return self._ba.align(seqs_a, seqs_b, list(aligned_a), list(aligned_b))
        return self._b.align(seqs_a, seqs_b, self.params, self.band, list(aligned_a),
                             list(aligned_b), self.kernel)

    @property
    def fill(self):
        return (self._ba or self._b).fill

    @fill.setter
    def fill(self, v):
        (self._ba or self._b).fill = v
