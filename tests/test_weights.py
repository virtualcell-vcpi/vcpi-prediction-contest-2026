"""Tests for the Mejia and pooled-variance per-compound weight schemes.

Goals:

- Pin the formula at small scale where the t-statistic and the
  min-max-square-renorm step are hand-computable.
- Pin properties the contest scoring relies on: every column sums to
  1, the gene that uniquely characterizes a compound gets the
  largest weight, degenerate compounds fall back to uniform.
- Compare Mejia vs pooled on a controlled noise scenario so the
  audit-script behavior is predictable in advance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vcpi_prediction_contest.weights import (
    _minmax_square_normalize,
    _replicate_level_log2_cpm,
    _t_vs_rest_matrix,
    compute_mejia_weights,
    compute_pooled_weights,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _counts(rows, sample_cols):
    return pd.DataFrame(rows, columns=["gene_id", *sample_cols])


def _metadata(samples_and_compounds):
    return pd.DataFrame(samples_and_compounds, columns=["sequenced_id", "compound"])


# ---------------------------------------------------------------------------
# minmax_square_normalize: pin the per-column behavior
# ---------------------------------------------------------------------------


def test_minmax_square_normalize_column_sums_to_one():
    abs_t = pd.DataFrame(
        np.array([[0.1, 5.0], [0.5, 2.0], [2.0, 1.0], [10.0, 0.5]]),
        index=["g0", "g1", "g2", "g3"],
        columns=["c0", "c1"],
    )
    w = _minmax_square_normalize(abs_t)
    np.testing.assert_allclose(w.sum(axis=0).to_numpy(), 1.0)


def test_minmax_square_normalize_concentrates_on_max_t():
    """A column with one dominant |t| should put most weight on that gene."""
    abs_t = pd.DataFrame(
        np.array([[1.0, 1.0, 1.0, 100.0]]).T,
        index=["g0", "g1", "g2", "g3"],
        columns=["c0"],
    )
    w = _minmax_square_normalize(abs_t)
    # min=1, max=100. Scaled: g3=1 (max), others ~0. Squared then normalized:
    # g3 dominates by orders of magnitude.
    assert w.loc["g3", "c0"] > 0.99


def test_minmax_square_normalize_degenerate_column_falls_back_to_uniform():
    """All identical |t| (max - min = 0) -> uniform weights."""
    abs_t = pd.DataFrame(
        np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]]),
        index=["g0", "g1", "g2"],
        columns=["degenerate", "normal"],
    )
    w = _minmax_square_normalize(abs_t)
    np.testing.assert_allclose(w["degenerate"].to_numpy(), 1.0 / 3.0)
    np.testing.assert_allclose(w.sum(axis=0).to_numpy(), 1.0)


# ---------------------------------------------------------------------------
# Replicate-level log2(CPM + 1) prep
# ---------------------------------------------------------------------------


def test_replicate_level_log2_cpm_basic():
    counts = _counts(
        [
            ["g1", 100, 500],
            ["g2", 900, 500],
        ],
        sample_cols=["s1", "s2"],
    )
    meta = _metadata([("s1", "A"), ("s2", "B")])
    log_cpm, sample_to_compound = _replicate_level_log2_cpm(
        counts, meta, sample_col="sequenced_id", compound_col="compound", gene_col="gene_id"
    )
    # library sizes 1000 each -> CPMs [100k, 900k] and [500k, 500k]
    assert log_cpm.loc["g1", "s1"] == pytest.approx(np.log2(100_001.0))
    assert log_cpm.loc["g2", "s2"] == pytest.approx(np.log2(500_001.0))
    assert sample_to_compound.loc["s1"] == "A"
    assert sample_to_compound.loc["s2"] == "B"


def test_replicate_level_drops_empty_library():
    counts = _counts(
        [["g1", 0, 100], ["g2", 0, 900]],
        sample_cols=["empty", "good"],
    )
    meta = _metadata([("empty", "A"), ("good", "B")])
    log_cpm, _ = _replicate_level_log2_cpm(
        counts, meta, sample_col="sequenced_id", compound_col="compound", gene_col="gene_id"
    )
    assert "empty" not in log_cpm.columns
    assert list(log_cpm.columns) == ["good"]


# ---------------------------------------------------------------------------
# _t_vs_rest_matrix: hand-compute on a tiny example to pin the algebra
# ---------------------------------------------------------------------------


def test_t_vs_rest_matrix_two_compounds_two_replicates_each():
    """Compound A has reps with log_cpm [4, 6] for g1 (mean 5, var 2);
    Compound B has reps with log_cpm [10, 12] for g1 (mean 11, var 2).
    Both have identical values [2, 2] for g2 (mean 2, var 0).

    Pooled var for g1: 4 values [4, 6, 10, 12], mean 8, var = ((-4)^2 + (-2)^2 + 2^2 + 4^2) / 3 = 40/3 ~ 13.33
    Pooled var for g2: identical 2s, pooled var = 0

    Welch t for compound A vs rest (= compound B):
      mean_A_g1 = 5; mean_rest_g1 = 11; mean shift = -6
      var_A_g1_raw = 2; var_rest_g1_raw = 2
      Floored at pooled = 13.33 each
      denom = sqrt(13.33/2 + 13.33/2) = sqrt(13.33) ~ 3.65
      t ~ -6 / 3.65 ~ -1.64
    """
    log_cpm = pd.DataFrame(
        np.array(
            [
                [4.0, 6.0, 10.0, 12.0],  # g1: A reps then B reps
                [2.0, 2.0, 2.0, 2.0],  # g2: identical
            ]
        ),
        index=["g1", "g2"],
        columns=["a1", "a2", "b1", "b2"],
    )
    sample_to_compound = pd.Series({"a1": "A", "a2": "A", "b1": "B", "b2": "B"})
    t_floored = _t_vs_rest_matrix(log_cpm, sample_to_compound, variance_mode="per_compound_floored")
    pooled_var_g1 = 40.0 / 3.0
    expected_t_a_g1 = -6.0 / np.sqrt(pooled_var_g1 / 2 + pooled_var_g1 / 2)
    expected_t_b_g1 = 6.0 / np.sqrt(pooled_var_g1 / 2 + pooled_var_g1 / 2)
    assert t_floored.loc["g1", "A"] == pytest.approx(expected_t_a_g1)
    assert t_floored.loc["g1", "B"] == pytest.approx(expected_t_b_g1)
    # g2 has zero pooled var; we floor a tiny epsilon, so |t| is enormous;
    # specifically: numerator = 0 so t = 0 exactly.
    assert t_floored.loc["g2", "A"] == pytest.approx(0.0)


def test_t_vs_rest_matrix_pooled_mode_uses_pooled_var_everywhere():
    """In pooled mode, the variance is the per-gene pooled variance for
    BOTH compound-p and rest, regardless of within-compound dispersion.
    Verify by constructing a case where per-compound and pooled var differ.
    """
    log_cpm = pd.DataFrame(
        np.array(
            [
                # g1: A reps tight (5, 5), B reps wide (0, 10). Pooled mean 5,
                # pooled var = ((0)^2 + (0)^2 + (-5)^2 + (5)^2) / 3 = 50/3
                [5.0, 5.0, 0.0, 10.0],
            ]
        ),
        index=["g1"],
        columns=["a1", "a2", "b1", "b2"],
    )
    sample_to_compound = pd.Series({"a1": "A", "a2": "A", "b1": "B", "b2": "B"})
    t_pooled = _t_vs_rest_matrix(log_cpm, sample_to_compound, variance_mode="pooled")
    pooled_var = 50.0 / 3.0
    # mean_A = 5; mean_rest = (0 + 10)/2 = 5. Mean shift = 0.
    expected_t = 0.0 / np.sqrt(pooled_var / 2 + pooled_var / 2)
    assert t_pooled.loc["g1", "A"] == pytest.approx(expected_t)


# ---------------------------------------------------------------------------
# compute_mejia_weights / compute_pooled_weights end-to-end
# ---------------------------------------------------------------------------


def _make_synthetic_release(
    *,
    n_compounds: int,
    n_replicates: int,
    n_genes: int,
    n_signal_genes_per_compound: int,
    seed: int,
    library_size: int = 100_000,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Generate replicate-level counts + metadata for a small synthetic screen.

    Each compound perturbs a distinct subset of `n_signal_genes_per_compound`
    genes (no overlap between compounds when possible). The "perturbation"
    inflates the gene's expected count rate by 5x. All other genes share a
    common baseline rate.

    Returns ``(counts_df, metadata_df, signal_mask)`` where ``signal_mask``
    is an (n_compounds, n_genes) bool array marking which gene is the
    signal for each compound.
    """
    rng = np.random.default_rng(seed)
    base_rate = rng.gamma(2.0, 2.0, size=n_genes) + 0.5
    # Choose signal genes per compound; allow overlap if we have too many compounds.
    signal_mask = np.zeros((n_compounds, n_genes), dtype=bool)
    for c in range(n_compounds):
        idx = rng.choice(n_genes, size=n_signal_genes_per_compound, replace=False)
        signal_mask[c, idx] = True

    sample_records = []
    counts_records = []
    for c in range(n_compounds):
        rates = np.where(signal_mask[c], base_rate * 5.0, base_rate)
        rates *= library_size / rates.sum()
        for r in range(n_replicates):
            sample_id = f"c{c:04d}_r{r}"
            counts = rng.poisson(rates)
            sample_records.append((sample_id, f"C{c:04d}"))
            counts_records.append(counts)

    counts_mat = np.stack(counts_records, axis=1)  # (n_genes, n_samples)
    gene_ids = [f"g{i:05d}" for i in range(n_genes)]
    sample_ids = [s for s, _ in sample_records]
    counts_df = pd.DataFrame(counts_mat, index=gene_ids, columns=sample_ids).reset_index()
    counts_df = counts_df.rename(columns={"index": "gene_id"})
    metadata_df = pd.DataFrame(sample_records, columns=["sequenced_id", "compound"])
    return counts_df, metadata_df, signal_mask


def test_mejia_weights_concentrate_on_signal_genes_at_n2():
    """For each compound, the genes flagged as 'signal' (perturbed 5x baseline)
    should receive a much larger share of the per-compound weight vector
    than the unperturbed genes — even with only 2 replicates per compound.
    """
    counts, meta, signal_mask = _make_synthetic_release(
        n_compounds=8,
        n_replicates=2,
        n_genes=200,
        n_signal_genes_per_compound=10,
        seed=42,
    )
    w = compute_mejia_weights(counts, meta)
    assert w.shape == (200, 8)
    np.testing.assert_allclose(w.sum(axis=0).to_numpy(), 1.0)
    # For every compound, the mean weight on signal genes should be at
    # least 10x the mean weight on non-signal genes.
    gene_index = w.index.tolist()
    for ci, compound in enumerate(w.columns):
        sig_genes = [g for i, g in enumerate(gene_index) if signal_mask[ci, i]]
        non_sig_genes = [g for i, g in enumerate(gene_index) if not signal_mask[ci, i]]
        sig_mean = w.loc[sig_genes, compound].mean()
        non_sig_mean = w.loc[non_sig_genes, compound].mean()
        assert sig_mean > 10 * non_sig_mean, (
            f"{compound}: signal weight mean {sig_mean:.4g} not >> non-signal {non_sig_mean:.4g}"
        )


def test_pooled_weights_also_concentrate_on_signal_genes_at_n2():
    """Same qualitative property as Mejia, with a slightly looser bound.

    The pooled-variance scheme cannot dampen genes that are naturally
    quiet in a particular compound (its denominator is the same per-gene
    pooled var regardless), so its concentration is weaker than Mejia's
    floored-per-compound scheme. We require >= 5x concentration here vs.
    >= 10x for Mejia.
    """
    counts, meta, signal_mask = _make_synthetic_release(
        n_compounds=8,
        n_replicates=2,
        n_genes=200,
        n_signal_genes_per_compound=10,
        seed=43,
    )
    w = compute_pooled_weights(counts, meta)
    np.testing.assert_allclose(w.sum(axis=0).to_numpy(), 1.0)
    gene_index = w.index.tolist()
    for ci, compound in enumerate(w.columns):
        sig_genes = [g for i, g in enumerate(gene_index) if signal_mask[ci, i]]
        non_sig_genes = [g for i, g in enumerate(gene_index) if not signal_mask[ci, i]]
        sig_mean = w.loc[sig_genes, compound].mean()
        non_sig_mean = w.loc[non_sig_genes, compound].mean()
        assert sig_mean > 5 * non_sig_mean, (
            f"{compound}: pooled signal mean {sig_mean:.4g} not >> non-signal {non_sig_mean:.4g}"
        )


def test_mejia_vs_pooled_agree_qualitatively_when_replicates_are_clean():
    """When replicates within each compound are reasonably tight, both
    weight schemes should rank the same genes near the top per compound.
    """
    counts, meta, _ = _make_synthetic_release(
        n_compounds=6,
        n_replicates=4,  # more replicates -> tighter per-compound var -> closer agreement
        n_genes=150,
        n_signal_genes_per_compound=8,
        seed=7,
    )
    w_mejia = compute_mejia_weights(counts, meta)
    w_pooled = compute_pooled_weights(counts, meta)
    # For each compound, the top-N genes by weight should overlap
    # substantially between the two schemes.
    top_n = 20
    for compound in w_mejia.columns:
        top_m = set(w_mejia[compound].nlargest(top_n).index)
        top_p = set(w_pooled[compound].nlargest(top_n).index)
        overlap = len(top_m & top_p)
        assert overlap >= 14, (
            f"{compound}: top-{top_n} overlap between mejia and pooled only {overlap}/{top_n}"
        )


def test_mejia_weights_singleton_compound_falls_back_gracefully():
    """A compound with only 1 replicate can't estimate within-compound var.
    The implementation should fall back to pooled var and emit a warning,
    not error.
    """
    counts = _counts(
        [
            ["g1", 100, 100, 500, 500],
            ["g2", 900, 900, 500, 500],
            ["g3", 1, 1, 1, 1],
        ],
        sample_cols=["a1", "a2", "b1", "solo_c"],
    )
    meta = _metadata([("a1", "A"), ("a2", "A"), ("b1", "B"), ("solo_c", "C")])
    w = compute_mejia_weights(counts, meta)
    assert set(w.columns) == {"A", "B", "C"}
    np.testing.assert_allclose(w.sum(axis=0).to_numpy(), 1.0)


def test_weights_rejects_missing_columns():
    bad_counts = pd.DataFrame({"foo": ["g1"], "s1": [10]})
    meta = _metadata([("s1", "A"), ("s2", "B")])
    with pytest.raises(ValueError, match="missing the `gene_id`"):
        compute_mejia_weights(bad_counts, meta)


def test_weights_no_sample_overlap_raises():
    counts = _counts([["g1", 10, 20]], sample_cols=["alpha", "beta"])
    meta = _metadata([("gamma", "A"), ("delta", "B")])
    with pytest.raises(ValueError, match="No samples overlap"):
        compute_mejia_weights(counts, meta)


def test_weights_polars_duck_type():
    class FakePolarsDF:
        def __init__(self, df):
            self._df = df

        def to_pandas(self):
            return self._df

    counts = FakePolarsDF(
        _counts([["g1", 100, 500], ["g2", 900, 500]], sample_cols=["s1", "s2"]),
    )
    meta = FakePolarsDF(_metadata([("s1", "A"), ("s2", "B")]))
    w = compute_mejia_weights(counts, meta)
    assert w.shape == (2, 2)
