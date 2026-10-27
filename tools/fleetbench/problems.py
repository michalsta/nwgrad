"""The scoring problems of the fleet benchmark: one per (fixture kind, gap model, mode).

Every parameter is an integer, so nwgrad's float64 score is exact and serves as the
reference for every tool and dtype (the integer types must match it exactly).

Gap convention (nwgrad's): affine, a gap of length k costs go + ge*k; linear, ge*k.

  protein   BLOSUM62; affine go=11 ge=1; linear ge=4
  dna/mirna affine: match +2, mismatch -4, go=4 ge=2 (global and local)
            linear global: unit edit cost (match 0, mismatch -1, ge=1) -- edit distance,
                           so edlib and WFA2 solve the same problem
            linear local:  match +1, mismatch -1, ge=1 (a match must pay, or local is empty)
"""
import json

DNA = "ACGT"


def kind_of(fixture_name):
    return "protein" if fixture_name.startswith("protein") else "dna"


def problem(kind, gap, mode):
    """dict: alphabet, matrix (list of rows), go, ge, plus a human-readable label."""
    if kind == "protein":
        from nwgrad.matrices import BLOSUM62
        alpha = BLOSUM62.alphabet
        M = [[int(round(v)) for v in row] for row in BLOSUM62.to_matrix()]
        go, ge = (11, 1) if gap == "affine" else (0, 4)
        label = "BLOSUM62, " + ("gap 11+k" if gap == "affine" else "gap 4k")
        return dict(alphabet=alpha, matrix=M, go=go, ge=ge, label=label, named="blosum62")
    if gap == "affine":
        ma, mi, go, ge = 2, -4, 4, 2
    elif mode == "global":
        ma, mi, go, ge = 0, -1, 0, 1
    else:
        ma, mi, go, ge = 1, -1, 0, 1
    M = [[ma if i == j else mi for j in range(4)] for i in range(4)]
    label = f"match {ma:+d} mismatch {mi:+d}, " + (f"gap {go}+{ge}k" if gap == "affine" else f"gap {ge}k")
    return dict(alphabet=DNA, matrix=M, go=go, ge=ge, label=label, named=None,
                match=ma, mismatch=mi)


def nwgrad_params(nwgrad, prob):
    import numpy as np
    m = nwgrad.SubstMatrix(np.array(prob["matrix"], dtype=float), alphabet=prob["alphabet"])
    return nwgrad.AlignParams(m, prob["go"], prob["ge"], prob["go"], prob["ge"])


def write_problem(path, prob):
    with open(path, "w") as f:
        json.dump(prob, f)


def read_pairs(path):
    A, B = [], []
    with open(path) as f:
        for line in f:
            a, b = line.rstrip("\n").split("\t")
            A.append(a); B.append(b)
    return A, B


def write_problem_txt(path, prob):
    """For the C/C++ drivers: alphabet, then 'go ge', then the matrix rows."""
    with open(path, "w") as f:
        f.write(prob["alphabet"] + "\n")
        f.write(f"{prob['go']} {prob['ge']}\n")
        for row in prob["matrix"]:
            f.write(" ".join(str(v) for v in row) + "\n")
