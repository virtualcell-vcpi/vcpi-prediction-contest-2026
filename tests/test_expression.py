"""Tests for the counts -> per-(compound, gene) expression recipe.

Pins the canonical pipeline (CPM -> log2(CPM + 1) -> mean across replicates)
with a hand-computable tiny example, plus library-size invariance,
replicate aggregation, and the QC/edge-case behavior contestants will
trip over.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vcpi_prediction_contest import build_gene_filter, counts_to_expression, score_compounds
from vcpi_prediction_contest.expression import DEFAULT_MIN_MEAN_CPM, DEFAULT_SAMPLE_COL


def _counts(rows, columns):
    """Build a wide counts frame: first column is gene_id, rest are samples."""
    df = pd.DataFrame(rows, columns=["gene_id", *columns])
    return df


def _metadata(samples_and_compounds):
    """Build a per-sample metadata frame: sequenced_id + compound columns."""
    return pd.DataFrame(samples_and_compounds, columns=[DEFAULT_SAMPLE_COL, "compound"])


# ---------------------------------------------------------------------------
# Hand-computed round trip: pins the formula
# ---------------------------------------------------------------------------


def test_counts_to_expression_hand_computed():
    """Two compounds, two genes, one replicate each.

    CompoundA, sampleA: counts = [100, 900], library = 1000
        CPM = [100_000, 900_000]
        log2(CPM + 1) = [log2(100_001), log2(900_001)]

    CompoundB, sampleB: counts = [500, 500], library = 1000
        CPM = [500_000, 500_000]
        log2(CPM + 1) = [log2(500_001), log2(500_001)]
    """
    counts = _counts(
        [
            ["g1", 100, 500],
            ["g2", 900, 500],
        ],
        columns=["sampleA", "sampleB"],
    )
    meta = _metadata([("sampleA", "CompoundA"), ("sampleB", "CompoundB")])
    expr = counts_to_expression(counts, meta)
    assert set(expr.columns) == {"compound", "gene_id", "expression"}
    assert len(expr) == 4

    def val(c, g):
        return float(
            expr.loc[(expr["compound"] == c) & (expr["gene_id"] == g), "expression"].iloc[0]
        )

    assert val("CompoundA", "g1") == pytest.approx(np.log2(100_001.0))
    assert val("CompoundA", "g2") == pytest.approx(np.log2(900_001.0))
    assert val("CompoundB", "g1") == pytest.approx(np.log2(500_001.0))
    assert val("CompoundB", "g2") == pytest.approx(np.log2(500_001.0))


# ---------------------------------------------------------------------------
# Library-size invariance: scale one sample's counts; CPM compensates
# ---------------------------------------------------------------------------


def test_cpm_normalization_kills_library_size_effect():
    """Two replicates of the same compound with VERY different library
    sizes (10x apart) but the same relative composition. After CPM ->
    log2(CPM + 1) -> mean, the result should be identical to scoring either
    replicate alone.
    """
    counts = _counts(
        [
            ["g1", 100, 1000],
            ["g2", 900, 9000],
        ],
        columns=["repA", "repB"],
    )
    meta = _metadata([("repA", "X"), ("repB", "X")])
    expr = counts_to_expression(counts, meta)
    # Both replicates have the same CPM (100k for g1, 900k for g2), so
    # the per-compound mean equals each replicate's value.
    g1 = float(expr.loc[expr["gene_id"] == "g1", "expression"].iloc[0])
    g2 = float(expr.loc[expr["gene_id"] == "g2", "expression"].iloc[0])
    assert g1 == pytest.approx(np.log2(100_001.0))
    assert g2 == pytest.approx(np.log2(900_001.0))


# ---------------------------------------------------------------------------
# Replicate aggregation: mean of log2(CPM + 1)
# ---------------------------------------------------------------------------


def test_replicate_mean_is_mean_of_logs_not_log_of_mean():
    """Two replicates of compound X, same library size, but the gene's
    CPM differs by a factor of 1000 between them. The result must be
    the arithmetic mean of the two log2(CPM + 1) values, not log2 of the
    arithmetic mean (which would be biased toward the high replicate).
    """
    # 2-gene matrix so the library size is a round number per sample.
    counts = _counts(
        [
            ["g1", 1, 1000],
            ["other", 999_999, 999_000],
        ],
        columns=["lo", "hi"],
    )
    meta = _metadata([("lo", "X"), ("hi", "X")])
    expr = counts_to_expression(counts, meta)
    g1 = float(expr.loc[expr["gene_id"] == "g1", "expression"].iloc[0])
    cpm_lo = 1 / 1_000_000 * 1e6  # 1
    cpm_hi = 1000 / 1_000_000 * 1e6  # 1000
    expected = (np.log2(cpm_lo + 1.0) + np.log2(cpm_hi + 1.0)) / 2.0
    naive_wrong = np.log2((cpm_lo + cpm_hi) / 2.0 + 1.0)
    assert g1 == pytest.approx(expected)
    assert g1 != pytest.approx(naive_wrong)


# ---------------------------------------------------------------------------
# Multiple compounds with different replicate counts
# ---------------------------------------------------------------------------


def test_per_compound_independence():
    """Compound A has 3 replicates, B has 1, C has 2. Each compound's
    expression should depend only on its own replicates.
    """
    counts = _counts(
        [
            ["g1", 100, 100, 100, 500, 200, 200],
            ["g2", 900, 900, 900, 500, 800, 800],
        ],
        columns=["a1", "a2", "a3", "b1", "c1", "c2"],
    )
    meta = _metadata(
        [
            ("a1", "A"),
            ("a2", "A"),
            ("a3", "A"),
            ("b1", "B"),
            ("c1", "C"),
            ("c2", "C"),
        ]
    )
    expr = counts_to_expression(counts, meta)
    assert sorted(expr["compound"].unique()) == ["A", "B", "C"]
    expr_a_g1 = float(
        expr.loc[(expr["compound"] == "A") & (expr["gene_id"] == "g1"), "expression"].iloc[0]
    )
    expr_b_g1 = float(
        expr.loc[(expr["compound"] == "B") & (expr["gene_id"] == "g1"), "expression"].iloc[0]
    )
    expr_c_g1 = float(
        expr.loc[(expr["compound"] == "C") & (expr["gene_id"] == "g1"), "expression"].iloc[0]
    )
    # A: 3 replicates each with CPM = 100/1000 * 1e6 = 100_000
    assert expr_a_g1 == pytest.approx(np.log2(100_001.0))
    # B: 1 replicate with CPM = 500/1000 * 1e6 = 500_000
    assert expr_b_g1 == pytest.approx(np.log2(500_001.0))
    # C: 2 replicates each with CPM = 200/1000 * 1e6 = 200_000
    assert expr_c_g1 == pytest.approx(np.log2(200_001.0))


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_zero_library_sample_drops_from_mean():
    """A zero-library sample cannot be CPM-normalized; it must be
    skipped rather than poisoning the per-compound mean with NaN.
    """
    counts = _counts(
        [
            ["g1", 0, 100],
            ["g2", 0, 900],
        ],
        columns=["empty", "good"],
    )
    meta = _metadata([("empty", "X"), ("good", "X")])
    expr = counts_to_expression(counts, meta)
    g1 = float(expr.loc[expr["gene_id"] == "g1", "expression"].iloc[0])
    g2 = float(expr.loc[expr["gene_id"] == "g2", "expression"].iloc[0])
    assert g1 == pytest.approx(np.log2(100_001.0))
    assert g2 == pytest.approx(np.log2(900_001.0))


def test_metadata_only_sample_is_ignored():
    """Metadata may list samples that aren't in the counts matrix (e.g.
    pre-QC metadata); they should be silently ignored, not raise.
    """
    counts = _counts(
        [["g1", 100, 500], ["g2", 900, 500]],
        columns=["s1", "s2"],
    )
    meta = _metadata([("s1", "A"), ("s2", "A"), ("s3_dropped", "B")])
    expr = counts_to_expression(counts, meta)
    assert set(expr["compound"]) == {"A"}


def test_counts_sample_without_metadata_is_dropped():
    """The inverse: counts columns that have no matching metadata row
    (e.g. samples that failed QC and were filtered out of metadata)
    must be dropped from the aggregation.
    """
    counts = _counts(
        [["g1", 100, 500], ["g2", 900, 500]],
        columns=["s_known", "s_unknown"],
    )
    meta = _metadata([("s_known", "A")])
    expr = counts_to_expression(counts, meta)
    assert set(expr["compound"]) == {"A"}
    g1 = float(expr.loc[expr["gene_id"] == "g1", "expression"].iloc[0])
    assert g1 == pytest.approx(np.log2(100_001.0))


def test_no_sample_overlap_raises():
    counts = _counts([["g1", 100], ["g2", 900]], columns=["alpha"])
    meta = _metadata([("beta", "A")])
    with pytest.raises(ValueError, match="No samples overlap"):
        counts_to_expression(counts, meta)


def test_missing_gene_column_raises():
    bad = pd.DataFrame({"foo": ["g1"], "sample1": [10]})
    meta = _metadata([("sample1", "A")])
    with pytest.raises(ValueError, match="missing the `gene_id` column"):
        counts_to_expression(bad, meta)


def test_missing_metadata_column_raises():
    counts = _counts([["g1", 100]], columns=["s1"])
    bad = pd.DataFrame({"sample": ["s1"], "compound": ["A"]})
    with pytest.raises(ValueError, match="metadata is missing"):
        counts_to_expression(counts, bad)


def test_duplicate_sample_id_raises():
    counts = _counts([["g1", 100, 200]], columns=["s1", "s2"])
    bad = pd.DataFrame({"sequenced_id": ["s1", "s1"], "compound": ["A", "B"]})
    with pytest.raises(ValueError, match="duplicate sequenced_id"):
        counts_to_expression(counts, bad)


# ---------------------------------------------------------------------------
# Output shape contract
# ---------------------------------------------------------------------------


def test_output_is_long_with_no_nans():
    counts = _counts(
        [["g1", 100, 500], ["g2", 900, 500]],
        columns=["s1", "s2"],
    )
    meta = _metadata([("s1", "A"), ("s2", "B")])
    expr = counts_to_expression(counts, meta)
    assert list(expr.columns) == ["compound", "gene_id", "expression"]
    assert not expr["expression"].isna().any()
    assert len(expr) == 4
    # Output is sorted by compound (groupby with sort=True).
    assert list(expr["compound"].unique()) == ["A", "B"]


def test_output_feeds_into_score_compounds():
    """End-to-end smoke: the output of counts_to_expression must be
    directly consumable by score_compounds without any reshaping.
    """
    counts = _counts(
        [["g1", 100, 110, 90], ["g2", 900, 890, 910]],
        columns=["s1", "s2", "s3"],
    )
    meta = _metadata([("s1", "A"), ("s2", "B"), ("s3", "C")])
    truth = counts_to_expression(counts, meta)
    pred = truth.rename(columns={"expression": "predicted_expression"})
    pc = score_compounds(truth, pred)
    np.testing.assert_allclose(pc["wmse"].to_numpy(), 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Polars duck-typing
# ---------------------------------------------------------------------------


def test_polars_input_via_to_pandas_duck_type():
    """vcpi-client returns polars. We don't depend on polars but accept
    anything with .to_pandas(); fake the protocol with a tiny shim.
    """

    class FakePolarsDF:
        def __init__(self, df):
            self._df = df

        def to_pandas(self):
            return self._df

    counts = FakePolarsDF(_counts([["g1", 100, 500]], columns=["s1", "s2"]))
    meta = FakePolarsDF(_metadata([("s1", "A"), ("s2", "B")]))
    expr = counts_to_expression(counts, meta)
    assert set(expr["compound"]) == {"A", "B"}


def test_non_dataframe_input_raises():
    with pytest.raises(TypeError, match="must be a pandas DataFrame"):
        counts_to_expression([1, 2, 3], _metadata([("s1", "A")]))


# ---------------------------------------------------------------------------
# build_gene_filter
# ---------------------------------------------------------------------------


def test_build_gene_filter_default_threshold_is_one_cpm():
    """The contest default is mean CPM >= 1.0; pin it to catch silent drift."""
    assert DEFAULT_MIN_MEAN_CPM == 1.0


def test_build_gene_filter_hand_computed():
    """Three genes, two samples each with library size 1,000,000 to make
    CPM = count exactly.

    gene_lo:   counts [0, 1]      -> CPM [0, 1]      -> mean 0.5  -> dropped (< 1)
    gene_mid:  counts [2, 3]      -> CPM [2, 3]      -> mean 2.5  -> kept
    gene_hi:   counts [1000, 500] -> CPM [1000, 500] -> mean 750  -> kept
    """
    counts = _counts(
        [
            ["gene_lo", 0, 1],
            ["gene_mid", 2, 3],
            ["gene_hi", 1000, 500],
        ],
        columns=["s1", "s2"],
    )
    # Pad library size to 1,000,000 with a dummy gene so the math is exact.
    pad_s1 = 1_000_000 - (0 + 2 + 1000)
    pad_s2 = 1_000_000 - (1 + 3 + 500)
    counts = pd.concat(
        [counts, pd.DataFrame([["pad", pad_s1, pad_s2]], columns=counts.columns)],
        ignore_index=True,
    )
    kept = build_gene_filter(counts)
    assert "gene_lo" not in kept
    assert "gene_mid" in kept
    assert "gene_hi" in kept
    assert "pad" in kept


def test_build_gene_filter_boundary_inclusive():
    """A gene with mean CPM exactly equal to the threshold is kept."""
    counts = _counts(
        [
            ["on_threshold", 1, 1],
            ["pad", 999_999, 999_999],
        ],
        columns=["s1", "s2"],
    )
    kept = build_gene_filter(counts, min_mean_cpm=1.0)
    assert "on_threshold" in kept


def test_build_gene_filter_custom_threshold():
    """Override threshold to keep only the highest-expressed gene."""
    counts = _counts(
        [
            ["g_low", 1, 1],
            ["g_mid", 10, 10],
            ["g_hi", 100, 100],
            ["pad", 999_889, 999_889],
        ],
        columns=["s1", "s2"],
    )
    kept = build_gene_filter(counts, min_mean_cpm=50.0)
    # CPMs: g_low=1, g_mid=10, g_hi=100, pad ~999889. Only g_hi and pad
    # exceed 50.
    assert kept == ["g_hi", "pad"]


def test_build_gene_filter_empty_when_threshold_too_high():
    counts = _counts(
        [["g", 1, 1], ["pad", 999_999, 999_999]],
        columns=["s1", "s2"],
    )
    kept = build_gene_filter(counts, min_mean_cpm=1e9)
    assert kept == []


def test_build_gene_filter_zero_library_sample_excluded_from_mean():
    """A zero-library sample produces NaN CPM and must be skipna'd out
    of the mean instead of poisoning every gene with NaN.
    """
    counts = _counts(
        [
            ["g", 0, 1000],
            ["pad", 0, 999_000],
        ],
        columns=["empty", "good"],
    )
    # 'good' has library 1,000,000 so CPM[g] = 1000; mean is just 1000.
    kept = build_gene_filter(counts)
    assert "g" in kept
    assert "pad" in kept


def test_build_gene_filter_output_sorted_and_deduped():
    """Output must be sorted strings, no duplicates, regardless of input
    row order or gene_id dtype.
    """
    counts = _counts(
        [
            ["ENSG2", 500, 500],
            ["ENSG1", 500, 500],
            ["ENSG3", 500, 500],
        ],
        columns=["s1", "s2"],
    )
    # Library size = 1500 per sample, so CPM = 333,333 per gene; all kept.
    kept = build_gene_filter(counts)
    assert kept == ["ENSG1", "ENSG2", "ENSG3"]


def test_build_gene_filter_missing_gene_column_raises():
    bad = pd.DataFrame({"foo": ["g1"], "s1": [10]})
    with pytest.raises(ValueError, match="missing the `gene_id` column"):
        build_gene_filter(bad)


def test_build_gene_filter_no_samples_raises():
    only_genes = pd.DataFrame({"gene_id": ["g1", "g2"]})
    with pytest.raises(ValueError, match="at least one sample column"):
        build_gene_filter(only_genes)


def test_build_gene_filter_polars_duck_type():
    class FakePolarsDF:
        def __init__(self, df):
            self._df = df

        def to_pandas(self):
            return self._df

    counts = FakePolarsDF(
        _counts([["g1", 500, 500], ["pad", 999_500, 999_500]], columns=["s1", "s2"]),
    )
    kept = build_gene_filter(counts)
    assert "g1" in kept
    assert "pad" in kept


def test_build_gene_filter_realistic_dropout_pattern():
    """Realistic DRUG-seq-shaped synthetic: many lowly-expressed genes,
    a handful of housekeeping-style high-expression genes, total per-
    sample library ~= 100k UMIs. The filter should drop the lowly-
    expressed genes and keep the rest.
    """
    rng = np.random.default_rng(0)
    n_dropout = 1000  # mean ~0 counts/sample -> dropped
    n_lo = 200  # mean ~0.05 counts/sample -> mostly dropped at CPM>=1
    n_mid = 500  # mean ~5 counts/sample -> kept
    n_hi = 50  # mean ~100 counts/sample -> kept
    n_samples = 20
    parts = [
        np.zeros((n_dropout, n_samples), dtype=int),
        rng.poisson(0.05, size=(n_lo, n_samples)),
        rng.poisson(5, size=(n_mid, n_samples)),
        rng.poisson(100, size=(n_hi, n_samples)),
    ]
    counts_mat = np.vstack(parts)
    gene_ids = (
        [f"drop_{i:04d}" for i in range(n_dropout)]
        + [f"lo_{i:04d}" for i in range(n_lo)]
        + [f"mid_{i:04d}" for i in range(n_mid)]
        + [f"hi_{i:04d}" for i in range(n_hi)]
    )
    # Approx library size per sample = 1000*0 + 200*0.05 + 500*5 + 50*100 = 7510 UMIs
    # 1 CPM = 7510/1e6 ~= 0.0075 counts. So a gene with mean 0.05 counts has CPM ~6.6 (kept).
    # To make this realistic, scale library size up by adding a "background" gene.
    bg = np.full(
        (1, n_samples), 100_000, dtype=int
    )  # one fake housekeeping gene with ~100k counts/sample
    counts_mat = np.vstack([counts_mat, bg])
    gene_ids = [*gene_ids, "background"]
    sample_ids = [f"s{i:02d}" for i in range(n_samples)]
    counts = pd.DataFrame(counts_mat, index=gene_ids, columns=sample_ids).reset_index()
    counts = counts.rename(columns={"index": "gene_id"})
    kept = set(build_gene_filter(counts))
    # All hi-expression genes kept.
    assert all(f"hi_{i:04d}" in kept for i in range(n_hi))
    # All mid-expression genes kept (mean count ~5, CPM ~50, well above 1.0).
    assert all(f"mid_{i:04d}" in kept for i in range(n_mid))
    # All dropout genes dropped (CPM = 0).
    assert not any(f"drop_{i:04d}" in kept for i in range(n_dropout))
    # Most lo-expression genes dropped (mean count 0.05, CPM ~0.5).
    n_lo_kept = sum(1 for i in range(n_lo) if f"lo_{i:04d}" in kept)
    assert n_lo_kept < n_lo // 4, f"Expected most lo genes dropped, kept {n_lo_kept}/{n_lo}"
