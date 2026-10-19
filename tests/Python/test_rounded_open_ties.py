"""A Y open's source (M or X) is chosen on the ROUNDED candidates, at every level.

The scalar fills pick the predecessor of a gap-in-A open with `pick(uy, vy, wy)` over
uy = (M - go_a) - ge_a and vy = (X - go_a) - ge_a: M wins ties.  The striped Pointers
kernel and the vectorized Hirschberg base case compared M >= X BEFORE subtracting, so
where M < X but both round to the same open they recorded X.  The values were
identical; the direction bytes were not, so the path (and, through Hirschberg's replayed
score, the score) depended on the ISA level.  Found by cross-level fuzzing (2026-10-19):
float32 Pointers with negative opens, Hirschberg's base case at double with ordinary
positive gaps.  Each fixture below failed before the fix.
"""
import numpy as np
import pytest

import nwgrad

SIMD = [l for l in nwgrad.available_isa_levels() if l != "scalar_fallback"]


def _params(m, g):
    return nwgrad.AlignParams(nwgrad.SubstMatrix(np.array(m), alphabet="ACGT"), *g)


def _run(cls, a, b, p, mode, tb, level, cutoff=None):
    batch = cls([a], [b], p, n_threads=1, traceback=tb, gap_model="affine", mode=mode,
                grad_mode="hard", kernel=level)
    batch.fill = "striped"
    if cutoff is not None:
        batch.hb_cutoff = cutoff
    batch.score_and_grad(keep_paths=True)
    g = batch.weighted_grad(np.ones(1))
    return (batch.scores().tobytes(), batch[0].aligned(),
            np.asarray(g.matrix.to_matrix()).tobytes(),
            (g.gap_open_a, g.gap_extend_a, g.gap_open_b, g.gap_extend_b))


POINTERS = [  # float32 striped Pointers, negative opens
    ("CGGCCAGAAAAA", "GACATTGATGAT",
     [[-0.3161569361202253, -0.16480742610450957, 0.17444972831435526, 0.05116921703563543],
      [-1.380617878675853, -0.059052629602431794, 0.9634783275565372, -0.7097293046442019],
      [0.6698021431339587, 1.0984903884157637, 0.9633022740254178, -1.2339684196911633],
      [-2.1355440117331823, 0.03848161528308918, 0.6916990968457613, -2.2868631705092786]],
     [-1.1487877095637236, 0.8908002039948537, -1.1003931301714314, 0.5195510125459394]),
]

HIRSCHBERG = [  # double, hb_cutoff 4, positive gaps
    ("AACGACATACGCCGGACCTTCTGGTTTCTC", "GCCGTGGACTTAAAGTGAGAACATTATAGGGACTTGAACGTG",
     [[22.8392637024425, 25.142541441623017, 3.637268233547735, -0.018280488047002316],
      [19.416609015378114, 18.23583165114234, 1.2082019033591094, 7.3567523308060085],
      [-16.15355228369181, -5.165597989005971, -12.233260451581081, -28.256792613144025],
      [2.7367987204017687, -0.666398812960846, 14.414459831840036, -7.781050850404522]],
     [4.310626539114407, 13.137964415125767, 4.021879177238041, 4.902996740752817], "global"),
    ("GATGTAACAACGCGCGAGTAATTAAGGCCCGA",
     "CGCCTTGTAATACTTTGTTTGCTCGAGGCCCGCCGCGGGACGAAAGATAGCCGCCTCCTCCACTCGCAGCGTC",
     [[-43.77642742621985, 146.68407679678072, -84.43296545400494, 156.6168238563572],
      [199.61938067447164, 18.588379502897574, 56.49014666542166, 4.312200317447689],
      [-72.01141557559787, -83.92946150790692, 42.29971723834037, 61.22395173999696],
      [105.68974821364363, 159.71632305662519, 123.13136284334043, -29.126763695535175]],
     [3.6915072516254357, 18.65584776545969, 15.820489848655408, 4.6439420449490605], "local"),
    ("AAGGGTCCTGGATTAATGAACTTCATCCTACCCTACGTTC", "AAGCATACTATTATTAGTAT",
     [[7.378473965512749, 2.4095850921381192, -16.480380249671718, -7.464439609354674],
      [10.056930460704816, 1.9969494613570524, 10.193863149995877, -7.743547504393669],
      [-2.5516796316185744, 9.443062030731813, -7.198782857682367, -2.943003917473803],
      [16.555811822356194, 3.644166118143446, -6.669046196533866, -2.552007253447707]],
     [0.9439939548672198, 0.6339609911885518, 0.5240825851442354, 0.3744964544627942], "local"),
]


@pytest.mark.parametrize("level", SIMD)
@pytest.mark.parametrize("case", range(len(POINTERS)))
def test_pointers_open_source_on_rounded_values(level, case):
    a, b, m, g = POINTERS[case]
    p = _params(m, g)
    ref = _run(nwgrad.SeqPairBatch, a, b, p, "local", "pointers", "scalar_fallback")
    assert _run(nwgrad.SeqPairBatch, a, b, p, "local", "pointers", level) == ref
    # the equality walk over score tables agrees with the scalar codes
    assert _run(nwgrad.SeqPairBatch, a, b, p, "local", "scores", level) == ref


@pytest.mark.parametrize("level", SIMD)
@pytest.mark.parametrize("case", range(len(HIRSCHBERG)))
def test_hirschberg_base_open_source_on_rounded_values(level, case):
    a, b, m, g, mode = HIRSCHBERG[case]
    p = _params(m, g)
    ref = _run(nwgrad.SeqPairBatchDouble, a, b, p, mode, "hirschberg", "scalar_fallback", 4)
    assert _run(nwgrad.SeqPairBatchDouble, a, b, p, mode, "hirschberg", level, 4) == ref
