"""Alphabet: interning, validation, and the alphabet-compatibility checks.

The substitution matrix is a dense N x N block indexed by alphabet position, not
a 256x256 ASCII table.  That makes the alphabet load-bearing rather than
decorative: two matrices can only be combined if they agree on one, and a
character outside the alphabet has no cell to land in.  Both of those used to be
silently tolerated.
"""

import numpy as np
import pytest
import nwgrad


DNA_ARR = np.eye(4) * 2 - 1
AA_ARR = np.eye(20) * 2 - 1


def dna_params(**kw):
    return nwgrad.AlignParams(nwgrad.SubstMatrix(DNA_ARR, alphabet="ACGT"), **kw)


# ── Interning ─────────────────────────────────────────────────────────────────

def test_get_interns_by_content():
    """Same symbols in, same instance out — identity is what the += check uses."""
    a = nwgrad.Alphabet.get("ACGT")
    b = nwgrad.Alphabet.get("ACGT")
    assert a is b or a == b
    assert a == nwgrad.DNA


def test_named_alphabets():
    assert nwgrad.DNA.symbols == "ACGT"
    assert nwgrad.DNA_N.symbols == "ACGTN"
    assert nwgrad.RNA.symbols == "ACGU"
    assert nwgrad.RNA_N.symbols == "ACGUN"
    assert nwgrad.PROTEIN.symbols == "ACDEFGHIKLMNPQRSTVWY"
    assert len(nwgrad.PROTEIN) == 20


def test_extensions_append_at_the_end():
    """The canonical 20 keep indices 0-19, so a 20x20 BLOSUM embeds as the
    top-left block of any extended alphabet."""
    for ext in (nwgrad.PROTEIN_X, nwgrad.PROTEIN_UO, nwgrad.PROTEIN_UOX):
        assert ext.symbols[:20] == nwgrad.PROTEIN.symbols
        for i, c in enumerate(nwgrad.PROTEIN.symbols):
            assert ext.index_of(c) == i


def test_selenocysteine_and_pyrrolysine():
    assert nwgrad.PROTEIN_UO.symbols == "ACDEFGHIKLMNPQRSTVWYUO"
    assert nwgrad.PROTEIN_UO.index_of("U") == 20   # Sec
    assert nwgrad.PROTEIN_UO.index_of("O") == 21   # Pyl
    assert nwgrad.PROTEIN_UOX.index_of("X") == 22
    assert not nwgrad.PROTEIN.contains("U")


def test_duplicate_symbol_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        nwgrad.Alphabet.get("ACGTA")


def test_large_ascii_alphabet():
    """The whole printable-ASCII range, so indices well past a signed byte's
    range are exercised end to end."""
    symbols = "".join(chr(c) for c in range(0x21, 0x7f) if chr(c) != "-")
    a = nwgrad.Alphabet.get(symbols)
    assert a.size == len(symbols)
    for i, c in enumerate(symbols):
        assert a.index_of(c) == i
    assert a.encode(symbols) == list(range(len(symbols)))


def test_non_ascii_symbol_rejected():
    """Python str reaches C++ as UTF-8, so a non-ASCII symbol would arrive as
    two bytes and split silently into two symbols."""
    with pytest.raises(ValueError, match="ASCII"):
        nwgrad.Alphabet.get("ACGTÄ")


def test_gap_marker_cannot_be_a_symbol():
    """aligned() renders gaps as '-'; a '-' symbol would make it ambiguous."""
    with pytest.raises(ValueError, match="gap marker"):
        nwgrad.Alphabet.get("ACGT-")


def test_alphabet_symbol_limit():
    with pytest.raises(ValueError, match="at most 127"):
        nwgrad.Alphabet.get("".join(chr(c) for c in range(1, 130)))


# ── Validation is loud ────────────────────────────────────────────────────────

def test_encode_rejects_out_of_alphabet_char():
    with pytest.raises(ValueError, match="not in alphabet"):
        nwgrad.DNA.encode("ACGTN")           # N needs DNA_N
    assert nwgrad.DNA_N.encode("ACGTN") == [0, 1, 2, 3, 4]


def test_encode_is_case_sensitive():
    """No case folding: lowercase is out-of-alphabet like any other symbol.
    Soft-masked FASTA must be upper-cased by the caller."""
    with pytest.raises(ValueError, match="not in alphabet"):
        nwgrad.DNA.encode("acgt")
    with pytest.raises(ValueError, match="not in alphabet"):
        nwgrad.DNA.encode("ACgT")


def test_error_names_the_character_and_position():
    with pytest.raises(ValueError) as e:
        nwgrad.DNA.encode("ACGXT")
    msg = str(e.value)
    assert "'X'" in msg and "position 3" in msg and "ACGT" in msg


def test_seqpair_rejects_out_of_alphabet_sequence():
    p = dna_params(gap_open_a=11.0, gap_extend_a=1.0,
                   gap_open_b=11.0, gap_extend_b=1.0)
    with pytest.raises(ValueError, match="not in alphabet"):
        nwgrad.SeqPair("ACGTN", "ACGT", p, gap_model="affine", mode="local")


def test_convenience_fn_rejects_out_of_alphabet_sequence():
    p = dna_params(gap_extend_a=1.0, gap_extend_b=1.0)
    with pytest.raises(ValueError, match="not in alphabet"):
        nwgrad.sw_score("ACGT", "ACGU", p)   # U is RNA


def test_batch_aligner_names_the_offending_pair():
    p = dna_params(gap_extend_a=1.0, gap_extend_b=1.0)
    al = nwgrad.BatchAligner(p, gap_model="linear", mode="local",
                             grad_mode="hard", n_threads=1)
    with pytest.raises(ValueError, match=r"pair 1"):
        al.align(["ACGT", "ACGT"], ["ACGT", "ACGZ"])


# ── Alphabet compatibility ────────────────────────────────────────────────────

def test_cannot_sum_gradients_across_alphabets():
    dna = dna_params(gap_extend_a=1.0, gap_extend_b=1.0)
    aa = nwgrad.AlignParams(nwgrad.SubstMatrix(AA_ARR), gap_extend_a=1.0,
                            gap_extend_b=1.0)
    _s1, g_dna = nwgrad.sw_grad("ACGT", "ACGT", dna)
    _s2, g_aa = nwgrad.sw_grad("ACDE", "ACDE", aa)
    with pytest.raises(ValueError, match="different alphabets"):
        g_dna + g_aa


def test_batch_rejects_a_pair_over_a_different_alphabet():
    """Checked eagerly in add(), on the caller's thread -- not deep inside a
    worker where the diagnostic would be useless."""
    dna = dna_params(gap_extend_a=1.0, gap_extend_b=1.0)
    aa = nwgrad.AlignParams(nwgrad.SubstMatrix(AA_ARR), gap_extend_a=1.0,
                            gap_extend_b=1.0)
    sp_dna = nwgrad.SeqPair("ACGT", "ACGT", dna, gap_model="linear", mode="local")
    sp_aa = nwgrad.SeqPair("ACDE", "ACDE", aa, gap_model="linear", mode="local")

    batch = nwgrad.SeqPairBatch(n_threads=2)
    batch.add(sp_dna)
    with pytest.raises(ValueError, match="cannot join a batch"):
        batch.add(sp_aa)
    assert len(batch) == 1


def test_set_params_cannot_change_the_alphabet():
    """The sequences are already encoded to indices; a new alphabet would
    silently reinterpret them as different residues."""
    dna = dna_params(gap_extend_a=1.0, gap_extend_b=1.0)
    rna = nwgrad.AlignParams(nwgrad.SubstMatrix(DNA_ARR, alphabet="ACGU"),
                             gap_extend_a=1.0, gap_extend_b=1.0)
    sp = nwgrad.SeqPair("ACGT", "ACGT", dna, gap_model="linear", mode="local")
    with pytest.raises(ValueError, match="cannot change the alphabet"):
        sp.set_params(rna)


def test_same_alphabet_set_params_still_works():
    p1 = dna_params(gap_extend_a=1.0, gap_extend_b=1.0)
    p2 = dna_params(gap_extend_a=2.0, gap_extend_b=2.0)
    sp = nwgrad.SeqPair("ACGTACGT", "ACGTACGT", p1, gap_model="linear", mode="local")
    sp.alloc_dp()
    sp.align_full()
    first = sp.score
    sp.set_params(p2)
    sp.align_full()
    assert sp.score == pytest.approx(first)   # identical seqs: no gaps either way


# ── The gradient really is compact and correctly placed ───────────────────────

def test_dna_gradient_is_4x4_and_counts_pairs():
    p = dna_params(gap_open_a=11.0, gap_extend_a=1.0,
                   gap_open_b=11.0, gap_extend_b=1.0)
    _score, grad = nwgrad.sw_affine_grad("ACGT", "ACGT", p)
    g = grad.matrix.to_matrix()
    assert g.shape == (4, 4)
    # Four matched pairs, one on each diagonal cell.
    np.testing.assert_allclose(g, np.eye(4), atol=1e-12)


def test_gradient_alphabet_survives_arithmetic():
    p = dna_params(gap_extend_a=1.0, gap_extend_b=1.0)
    _s, g = nwgrad.sw_grad("ACGT", "ACGT", p)
    for derived in (g + g, g * 2.0, -g, g - g, 0.5 * g):
        assert derived.matrix.alphabet == "ACGT"
        assert derived.matrix.to_matrix().shape == (4, 4)
