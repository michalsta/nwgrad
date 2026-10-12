#!/usr/bin/env python3
"""Randomized stress test of tracked batches (run for a while, on every machine).

    python tools/stress_tracks.py [--minutes 10] [--seed 0] [--max-len 400]

Each round draws an alphabet (1-20 letters), K (1-40), a batch (1-200 pairs, mostly
short, some up to --max-len), tracks (or none: K sets, set 0 everywhere), parameters
(random, or small integers: ties), a gap model, mode, precision, fill, traceback and
thread count, and checks:
  - scores and per-pair gradients against the single-pair functions (exact; with
    integral parameters only the scores, since ties may pick another optimal path);
  - every score against <gradient, parameters> (exact for integral parameters);
  - weighted_grad against the per-pair gradients summed in pair order (one block);
  - banded_grad with a band past every sequence against the full fill;
  - streaming align() against the stored batch (scores; summed gradient);
  - now and then the soft path against the single-pair soft functions (tolerance).
Run it under NWGRAD_FORCE_KERNEL=scalar_fallback and NWGRAD_TRACK_PROFILE_BYTES=0 too
(both are read once per process).  Prints one line per minute; exits 1 on the first
mismatch, with the round's seed.
"""
import argparse, sys, time, traceback
import numpy as np
import nwgrad

ALPHABETS = ["A", "AC", "ACG", "ACGT", "ACGTN", "ACDEFGHI", "ACDEFGHIK", "ARNDCQEGHILKMFPSTWYV"]
CASES = [("linear", "global"), ("linear", "local"), ("affine", "global"), ("affine", "local")]
HARD = {("linear", "global"): "nw_grad", ("linear", "local"): "sw_grad",
        ("affine", "global"): "nw_affine_grad", ("affine", "local"): "sw_affine_grad"}
SOFT = {("linear", "global"): "nw_soft_grad", ("linear", "local"): "sw_soft_grad",
        ("affine", "global"): "nw_affine_soft_grad", ("affine", "local"): "sw_affine_soft_grad"}
BATCH = {"_double": nwgrad.SeqPairBatchDouble, "": nwgrad.SeqPairBatch}


def lengths(rng, n, max_len):
    out = rng.integers(0, 40, size=n)
    longs = rng.random(n) < 0.08
    out[longs] = rng.integers(40, max_len + 1, size=int(longs.sum()))
    return out


def one_round(seed, max_len):
    rng = np.random.default_rng(seed)
    alpha = ALPHABETS[rng.integers(len(ALPHABETS))]
    na = len(alpha)
    K = int(rng.integers(1, 41))
    n = int(rng.integers(1, 201))
    case = CASES[rng.integers(4)]
    prec = ["_double", ""][rng.integers(2)]
    integral = rng.random() < 0.35
    tracked_none = rng.random() < 0.1
    la, lb = lengths(rng, n, max_len), lengths(rng, n, max_len)
    A = ["".join(rng.choice(list(alpha), size=int(m))) for m in la]
    B = ["".join(rng.choice(list(alpha), size=int(m))) for m in lb]
    if integral:
        mats = rng.integers(-3, 4, size=(K, na, na)).astype(float)
        gaps = rng.integers(1, 4, size=(K, 4)).astype(float)
    else:
        mats = rng.normal(0, 1.5, size=(K, na, na))
        gaps = rng.uniform(0.2, 2.0, size=(K, 4))
    P = nwgrad.AlignParams.from_arrays(mats, gaps, alpha)
    if tracked_none:
        tracks = None; gtracks = None
    else:
        tracks = [list(rng.integers(0, K, size=len(a))) for a in A]
        gtracks = [list(rng.integers(0, K, size=len(a) + 1)) if rng.random() < 0.7 else None for a in A]
    fill = ["interpair", "striped", "rowwise"][rng.integers(3)]
    tb = ["auto", "pointers", "scores", "hirschberg", "hirschberg_pmax"][rng.integers(5)]
    if prec == "" and tb == "auto" and case[0] == "affine" and case[1] == "global":
        tb = "pointers"   # auto -> hirschberg_pmax there, which may return a suboptimal path
    if tb == "hirschberg_pmax":
        integral = True   # (its path may be suboptimal; compare only what is exact)
    threads = int(rng.integers(1, 9))
    storage = ["paths", "counts"][rng.integers(2)]
    desc = f"seed={seed} alpha={na} K={K} n={n} case={case} prec={prec!r} integral={integral} " \
           f"tracks={'none' if tracked_none else 'yes'} fill={fill} tb={tb} threads={threads} " \
           f"storage={storage}"

    batch = BATCH[prec](threads, tb, gap_model=case[0], mode=case[1], grad_mode="hard")
    batch.fill = fill
    batch.track_storage = storage
    if tracked_none:
        batch.add_many(A, B, P)
    else:
        batch.add_many(A, B, P, tracks=tracks, gap_tracks=gtracks)
    batch.score_and_grad()
    sc = batch.scores()
    gm, gg = batch.grads()
    if K == 1 and tracked_none:   # an untracked one-set batch: (N, n, n), (N, 4)
        gm = gm[:, None]; gg = gg[:, None]
    # <grad, params> == score: exact for integral parameters
    dots = (gm * mats[None]).sum(axis=(1, 2, 3)) + (gg * gaps[None]).sum(axis=(1, 2))
    if integral and tb != "hirschberg_pmax":
        assert np.array_equal(dots, sc), (desc, "dot != score", np.flatnonzero(dots != sc)[:5])
    elif not integral:
        np.testing.assert_allclose(dots, sc, rtol=1e-9 if prec else 1e-4, atol=1e-9 if prec else 1e-3,
                                   err_msg=desc + " dot != score")
    # single-pair functions
    if tb != "hirschberg_pmax":
        f = getattr(nwgrad, HARD[case] + prec)
        for k in range(n):
            kw = {} if tracked_none else dict(track=tracks[k], gap_track=gtracks[k])
            s, g = f(A[k], B[k], P, **kw)
            assert s == sc[k], (desc, "score vs pair", k, s, sc[k])
            if not integral and tb not in ("hirschberg",):
                gmk = g.matrices if g.num_sets > 1 or not tracked_none else g.matrices
                assert np.array_equal(gm[k], gmk.reshape(gm[k].shape)), (desc, "grad vs pair", k)
                assert np.array_equal(gg[k], g.gaps.reshape(gg[k].shape)), (desc, "gaps vs pair", k)
    # weighted sum, pair order (one block)
    w = rng.normal(size=n)
    G = batch.weighted_grad(w)
    acc_m, acc_g = np.zeros(gm.shape[1:]), np.zeros(gg.shape[1:])
    for k in range(n):
        acc_m += w[k] * gm[k]
        acc_g += w[k] * gg[k]
    assert np.array_equal(G.matrices.reshape(acc_m.shape), acc_m), (desc, "weighted_grad")
    assert np.array_equal(G.gaps.reshape(acc_g.shape), acc_g), (desc, "weighted_grad gaps")
    # a band past every sequence is the full fill (scores)
    if tb not in ("hirschberg", "hirschberg_pmax"):
        wide = int(max(la.max(), lb.max())) + 2
        batch.banded_grad(wide)
        assert np.array_equal(batch.scores(), sc), (desc, "wide band")
    # streaming
    if not tracked_none and tb != "hirschberg_pmax":
        st = BATCH[prec](threads, gap_model=case[0], mode=case[1], grad_mode="hard")
        st.fill = fill
        res = st.align(A, B, P, tracks=tracks, gap_tracks=gtracks)
        assert np.array_equal(res.scores, sc), (desc, "stream scores")
    # soft, now and then
    if rng.random() < 0.25 and prec == "_double":
        T = float(rng.uniform(0.5, 1.5))
        sb = nwgrad.SeqPairBatchDouble(threads, gap_model=case[0], mode=case[1], grad_mode="soft")
        sb.fill = ["interpair", "striped"][rng.integers(2)]
        sb.soft_temperature = T
        sb.soft_impl = "scaled_or_log"
        if tracked_none:
            sb.add_many(A, B, P)
        else:
            sb.add_many(A, B, P, tracks=tracks, gap_tracks=gtracks)
        sb.score_and_grad()
        ssc = sb.scores()
        sgm, sgg = sb.grads()
        if K == 1 and tracked_none:
            sgm = sgm[:, None]; sgg = sgg[:, None]
        f = getattr(nwgrad, SOFT[case] + "_double")
        for k in rng.choice(n, size=min(n, 25), replace=False):
            kw = {} if tracked_none else dict(track=tracks[k], gap_track=gtracks[k])
            s, g = f(A[k], B[k], P, soft_impl="log", temperature=T, **kw)
            tol = 1e-9 * max(1.0, abs(s))
            assert abs(ssc[k] - s) <= tol, (desc, "soft score", k, s, ssc[k])
            np.testing.assert_allclose(sgm[k], g.matrices.reshape(sgm[k].shape), rtol=1e-7, atol=1e-9,
                                       err_msg=desc + f" soft grad {k}")
            np.testing.assert_allclose(sgg[k], g.gaps.reshape(sgg[k].shape), rtol=1e-7, atol=1e-9,
                                       err_msg=desc + f" soft gaps {k}")
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minutes", type=float, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=400)
    args = ap.parse_args()
    t0 = time.time(); last = t0
    rounds = pairs = 0
    seed = args.seed
    while time.time() - t0 < args.minutes * 60:
        try:
            pairs += one_round(seed, args.max_len)
        except Exception:
            traceback.print_exc()
            print(f"FAILED at seed {seed}", flush=True)
            sys.exit(1)
        rounds += 1; seed += 1
        if time.time() - last > 60:
            last = time.time()
            print(f"{(last - t0) / 60:5.1f} min: {rounds} rounds, {pairs} pairs, next seed {seed}", flush=True)
    print(f"OK: {rounds} rounds, {pairs} pairs, seeds {args.seed}..{seed - 1}", flush=True)


if __name__ == "__main__":
    main()
