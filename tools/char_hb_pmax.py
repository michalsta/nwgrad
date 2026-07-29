"""How wrong is the prefix-max Hirschberg carry, and what actually makes it wrong?

`traceback="hirschberg_pmax"` computes the VY gap carry as a closed-form prefix max
(VY[c] = prefixmax_k(open[k] + k*ge_a) - (c-1)*ge_a) instead of the exact serial chain.
The ramp k*ge_a is added and subtracted again, so the answer carries a rounding of order
eps * k * ge_a — proportional to the COLUMN INDEX, not (as is often assumed) to the
winning gap-run length.  That rounding can cost an argmax, and the returned path is then
SUBOPTIMAL: a different kind of error from plain Hirschberg's tie-break, which returns a
different optimum but never a worse one.

This measures the difference, and it attributes it.  Every arm is scored against the same
T=double Pointers reference, so the three effects separate cleanly:

    opt - ptr32    what float32 alone already costs
    opt - hb       float32 + Hirschberg's tie-break         (the current default's error)
    opt - pmax     the above + the ramp                     (what the variant costs)
    hb  - pmax     THE RAMP ALONE — the only number attributable to this variant, signed,
                   so a positive value means pmax came back WORSE than exact Hirschberg
                   on the same pair.  A large |difference| with a zero shortfall just
                   means the two picked different paths of equal score.

Run it with no arguments for the standard sweep.  Long lengths dominate the runtime and
the memory (the Pointers reference is O(m*n)); --max-len trims both.
"""
import argparse
import numpy as np
import nwgrad
from nwgrad.matrices import BLOSUM62

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--lens", type=int, nargs="+",
                default=[1000, 2000, 4000, 8000, 12000, 16000])
ap.add_argument("--ge", type=float, nargs="+", default=[1.0, 0.1],
                help="gap_extend values to sweep (the ramp's scale factor)")
ap.add_argument("--n", type=int, default=8, help="random pairs per cell")
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--cutoff", type=int, default=512)
ap.add_argument("--seed", type=int, default=20260728)
ap.add_argument("--matrix", choices=["blosum", "real"], default="blosum",
                help="'real' uses a random real-valued matrix — a stand-in for a LEARNED "
                     "one, where near-ties exist at every scale rather than on an integer "
                     "lattice.  That is this library's actual use case and the harder test.")
args = ap.parse_args()

AA = list(BLOSUM62.alphabet)
# The existing envelopes this variant has to be judged against, from CLAUDE.md: the
# deviation float32 Pointers and float32 Hirschberg ALREADY show against a double oracle.
ENVELOPE_F32 = 8.43e-3
ENVELOPE_F64 = 1.6e-12


def score_arm(pairs, p, tb, Batch, threads, cutoff):
    b = Batch(n_threads=threads, traceback=tb)
    if tb.startswith("hirschberg"):
        b.hb_cutoff = cutoff
    b.add_many([x for x, _ in pairs], [y for _, y in pairs], p,
               gap_model="affine", mode="global", grad_mode="hard", kernel="auto")
    b.score_and_grad()
    return np.array([b[i].score for i in range(len(pairs))], dtype=float)


def make_matrix(rng):
    if args.matrix == "blosum":
        return BLOSUM62
    # A learned matrix is continuous: score differences do not sit on an integer lattice,
    # so a perturbation of any size can flip an argmax.  BLOSUM's integers hide that.
    return nwgrad.SubstMatrix(rng.normal(0.0, 4.0, size=(len(AA), len(AA))),
                              BLOSUM62.alphabet)


def main():
    rng = np.random.default_rng(args.seed)
    print(f"isa={nwgrad.simd_isa()}  cutoff={args.cutoff}  n={args.n}/cell  "
          f"threads={args.threads}  matrix={args.matrix}")
    print("reference = T=double Pointers.  All columns are (reference - arm), so a "
          "positive number is a shortfall below the true optimum.\n")

    hdr = (f"{'ge_a':>6s} {'len':>6s} | {'f32 ptr':>11s} {'f32 hb':>11s} {'f32 pmax':>11s} "
           f"{'RAMP ONLY':>11s} | {'f64 hb':>10s} {'f64 pmax':>11s} {'RAMP':>10s}")
    print(hdr); print("-" * len(hdr))

    worst = {"f32_ramp": 0.0, "f64_ramp": 0.0, "f32_pmax": 0.0, "f64_pmax": 0.0}
    for ge in args.ge:
        p = nwgrad.AlignParams(make_matrix(rng), gap_open_a=11.0, gap_extend_a=ge,
                               gap_open_b=11.0, gap_extend_b=ge)
        for L in args.lens:
            pairs = [(''.join(rng.choice(AA, L)), ''.join(rng.choice(AA, L)))
                     for _ in range(args.n)]
            D, F = nwgrad.SeqPairBatchDouble, nwgrad.SeqPairBatch
            ref = score_arm(pairs, p, "pointers", D, args.threads, args.cutoff)
            p32 = score_arm(pairs, p, "pointers", F, args.threads, args.cutoff)
            h32 = score_arm(pairs, p, "hirschberg", F, args.threads, args.cutoff)
            m32 = score_arm(pairs, p, "hirschberg_pmax", F, args.threads, args.cutoff)
            h64 = score_arm(pairs, p, "hirschberg", D, args.threads, args.cutoff)
            m64 = score_arm(pairs, p, "hirschberg_pmax", D, args.threads, args.cutoff)

            # Signed and per-pair: how much WORSE pmax came back than exact Hirschberg.
            r32 = (h32 - m32).max()
            r64 = (h64 - m64).max()
            worst["f32_ramp"] = max(worst["f32_ramp"], r32)
            worst["f64_ramp"] = max(worst["f64_ramp"], r64)
            worst["f32_pmax"] = max(worst["f32_pmax"], (ref - m32).max())
            worst["f64_pmax"] = max(worst["f64_pmax"], (ref - m64).max())

            flag = ""
            if (ref - m32).max() > ENVELOPE_F32: flag += " *f32-envelope*"
            if (ref - m64).max() > ENVELOPE_F64: flag += " *f64-envelope*"
            print(f"{ge:6.3g} {L:6d} | {(ref-p32).max():11.4g} {(ref-h32).max():11.4g} "
                  f"{(ref-m32).max():11.4g} {r32:11.4g} | {(ref-h64).max():10.4g} "
                  f"{(ref-m64).max():11.4g} {r64:10.4g}{flag}")

    print("\nworst over the whole sweep:")
    print(f"  float32: pmax total {worst['f32_pmax']:.4g}  (ramp alone {worst['f32_ramp']:.4g})"
          f"   existing float32 envelope {ENVELOPE_F32:.4g}"
          f"  -> {'EXCEEDS' if worst['f32_pmax'] > ENVELOPE_F32 else 'within'}")
    print(f"  double : pmax total {worst['f64_pmax']:.4g}  (ramp alone {worst['f64_ramp']:.4g})"
          f"   existing double  envelope {ENVELOPE_F64:.4g}"
          f"  -> {'EXCEEDS' if worst['f64_pmax'] > ENVELOPE_F64 else 'within'}")


if __name__ == "__main__":
    main()
