"""Tests for the expression-prediction scoring metrics.

The tests build small ``(gene × compound)`` synthetic data, pin known
invariants of each metric (perfect, mean-baseline, scaling attacks),
and end-to-end-check :func:`score_compounds` /
:func:`aggregate_leaderboards`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vcpi_prediction_contest import metrics

# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

GENES = [f"ENSG{i:011d}" for i in range(8)]


def make_truth(rows):
    """Truth long-frame.

    rows: iterable of (compound, gene, expression).
    """
    return pd.DataFrame(
        rows,
        columns=[metrics.COMPOUND_COL, metrics.GENE_COL, metrics.EXPRESSION_COL],
    )


def make_pred(rows):
    """Prediction long-frame: (compound, gene, predicted_expression)."""
    return pd.DataFrame(
        rows,
        columns=[metrics.COMPOUND_COL, metrics.GENE_COL, metrics.PRED_COL],
    )


def make_synthetic_release(*, n_train, n_test, n_genes, seed):
    """Build a (truth_train_df, truth_test_df) pair with shared gene baselines.

    Per-gene baseline drawn ONCE from Gamma(2, 1.5), then per-compound
    Gaussian effects on top — clipped at zero. Sharing the baseline is
    the correct model of the real task: every gene has a cell-line
    baseline that all train AND test compounds perturb around.
    """
    rng = np.random.default_rng(seed)
    baseline = rng.gamma(2.0, 1.5, size=n_genes)
    train_effects = rng.normal(0.0, 0.5, size=(n_genes, n_train))
    test_effects = rng.normal(0.0, 0.5, size=(n_genes, n_test))
    gene_index = [f"g{i:04d}" for i in range(n_genes)]
    train_compounds = [f"train_{i:04d}" for i in range(n_train)]
    test_compounds = [f"test_{i:04d}" for i in range(n_test)]

    def to_long(mat, compounds):
        df = pd.DataFrame(
            np.maximum(baseline[:, None] + mat, 0.0), index=gene_index, columns=compounds
        )
        df.index.name = metrics.GENE_COL
        long = df.reset_index().melt(
            id_vars=metrics.GENE_COL,
            var_name=metrics.COMPOUND_COL,
            value_name=metrics.EXPRESSION_COL,
        )
        return long

    return to_long(train_effects, train_compounds), to_long(test_effects, test_compounds)


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------


def test_variance_weights_sum_to_one():
    truth_train, _ = make_synthetic_release(n_train=50, n_test=10, n_genes=200, seed=0)
    w = metrics.compute_variance_weights(truth_train)
    assert isinstance(w, pd.Series)
    np.testing.assert_allclose(float(w.sum()), 1.0, atol=1e-12)
    assert w.name == "weight"


def test_variance_weights_concentrate_on_high_variance_gene():
    rows = []
    # gene high_var: large per-compound variance
    rows.extend([("A", "high_var", 10.0), ("B", "high_var", 1.0), ("C", "high_var", 100.0)])
    # gene low_var: small variance
    rows.extend([("A", "low_var", 5.0), ("B", "low_var", 5.0001), ("C", "low_var", 4.9999)])
    truth = make_truth(rows)
    w = metrics.compute_variance_weights(truth)
    assert w["high_var"] > w["low_var"]


def test_variance_weights_zero_variance_falls_back_to_uniform():
    truth = make_truth(
        [
            ("A", "g0", 1.0),
            ("B", "g0", 1.0),
            ("C", "g0", 1.0),
            ("A", "g1", 2.0),
            ("B", "g1", 2.0),
            ("C", "g1", 2.0),
        ]
    )
    w = metrics.compute_variance_weights(truth)
    np.testing.assert_allclose(w.to_numpy(), 0.5)


# ---------------------------------------------------------------------------
# Per-compound metric: wMSE
# ---------------------------------------------------------------------------


def test_wmse_perfect_is_zero():
    truth_train, truth_test = make_synthetic_release(n_train=20, n_test=5, n_genes=40, seed=1)
    truth_mat, pred_mat = metrics.align_long_frames(
        truth_test,
        truth_test.rename(columns={metrics.EXPRESSION_COL: metrics.PRED_COL}),
    )
    w = metrics.compute_variance_weights(truth_train).reindex(truth_mat.index).fillna(0)
    np.testing.assert_allclose(metrics.wmse(truth_mat, pred_mat, w).to_numpy(), 0.0, atol=1e-12)


def test_wmse_grows_under_scaling():
    """k=2 must be strictly worse than k=1 (penalty for getting magnitude wrong)."""
    truth_train, truth_test = make_synthetic_release(n_train=20, n_test=5, n_genes=40, seed=2)
    truth_mat, _ = metrics.align_long_frames(
        truth_test,
        truth_test.rename(columns={metrics.EXPRESSION_COL: metrics.PRED_COL}),
    )
    w = metrics.compute_variance_weights(truth_train).reindex(truth_mat.index).fillna(0)
    perfect = metrics.wmse(truth_mat, truth_mat, w).mean()
    scaled = metrics.wmse(truth_mat, truth_mat * 2.0, w).mean()
    assert scaled > perfect + 1e-9


# ---------------------------------------------------------------------------
# Orchestrator: score_compounds and aggregate_leaderboards together
# ---------------------------------------------------------------------------


def test_score_compounds_columns():
    _, truth_test = make_synthetic_release(n_train=20, n_test=10, n_genes=50, seed=9)
    pred = truth_test.rename(columns={metrics.EXPRESSION_COL: metrics.PRED_COL})
    pc = metrics.score_compounds(truth_test, pred)
    assert set(pc.columns) == {"wmse"}
    assert len(pc) == 10


def test_score_compounds_perfect_with_external_weights_pins_wmse_zero():
    truth_train, truth_test = make_synthetic_release(n_train=30, n_test=10, n_genes=80, seed=10)
    pred = truth_test.rename(columns={metrics.EXPRESSION_COL: metrics.PRED_COL})
    weights = metrics.compute_variance_weights(truth_train)
    pc = metrics.score_compounds(truth_test, pred, weights=weights)
    np.testing.assert_allclose(pc["wmse"].to_numpy(), 0.0, atol=1e-12)


def test_aggregate_leaderboards_shape():
    _, truth_test = make_synthetic_release(n_train=20, n_test=12, n_genes=40, seed=12)
    pred = truth_test.rename(columns={metrics.EXPRESSION_COL: metrics.PRED_COL})
    pc = metrics.score_compounds(truth_test, pred)
    board = metrics.aggregate_leaderboards(pc)
    assert set(board.keys()) == {"n_compounds", "wmse_mean"}
    assert board["n_compounds"] == 12
    assert board["wmse_mean"] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Scaling-attack regression: confirm wMSE catches the constant-scaling hack
# ---------------------------------------------------------------------------


def test_scaling_attack_via_score_compounds_degrades_wmse():
    """Sweep k across positive values. ``score_compounds`` must report a
    strictly worse wMSE than the perfect predictor for every k != 1 — the
    metric is not constant-scaling-hackable.
    """
    truth_train, truth_test = make_synthetic_release(n_train=30, n_test=20, n_genes=60, seed=13)
    weights = metrics.compute_variance_weights(truth_train)

    perfect = metrics.score_compounds(
        truth_test,
        truth_test.rename(columns={metrics.EXPRESSION_COL: metrics.PRED_COL}),
        weights=weights,
    )

    for k in (0.1, 0.5, 2.0, 10.0):
        scaled_pred = truth_test.copy()
        scaled_pred[metrics.EXPRESSION_COL] = scaled_pred[metrics.EXPRESSION_COL] * k
        scaled_pred = scaled_pred.rename(columns={metrics.EXPRESSION_COL: metrics.PRED_COL})
        pc = metrics.score_compounds(truth_test, scaled_pred, weights=weights)
        assert pc["wmse"].mean() > perfect["wmse"].mean() + 1e-9, (
            f"k={k}: wMSE should be > perfect, got {pc['wmse'].mean():.6f}"
        )


# ---------------------------------------------------------------------------
# Validation: mis-shaped inputs
# ---------------------------------------------------------------------------


def test_align_no_overlap_raises():
    truth = make_truth([("A", "g0", 1.0)])
    pred = make_pred([("B", "g0", 1.0)])
    with pytest.raises(ValueError, match="No compounds in common"):
        metrics.align_long_frames(truth, pred)


def test_align_missing_truth_columns_raises():
    bad = pd.DataFrame({metrics.COMPOUND_COL: ["A"], metrics.GENE_COL: ["g"]})
    pred = make_pred([("A", "g", 1.0)])
    with pytest.raises(ValueError, match="missing required columns"):
        metrics.align_long_frames(bad, pred)


def test_align_missing_prediction_columns_raises():
    truth = make_truth([("A", "g", 1.0)])
    bad = pd.DataFrame({metrics.COMPOUND_COL: ["A"], metrics.GENE_COL: ["g"]})
    with pytest.raises(ValueError, match="missing required columns"):
        metrics.align_long_frames(truth, bad)


def test_wmse_weight_length_mismatch_raises():
    t = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=["g0", "g1", "g2"])
    p = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=["g0", "g1", "g2"])
    with pytest.raises(ValueError, match="weights 1-D array must be length 3"):
        metrics.wmse(t, p, np.array([0.5, 0.5]))


# ---------------------------------------------------------------------------
# Per-compound (matrix) weights — the Mejia / pooled-weights shape
# ---------------------------------------------------------------------------


def test_wmse_accepts_per_compound_weight_matrix():
    """A DataFrame of (gene x compound) weights should be applied column-wise."""
    truth = pd.DataFrame(
        np.array([[1.0, 1.0], [2.0, 2.0]]),
        index=["g0", "g1"],
        columns=["A", "B"],
    )
    pred = pd.DataFrame(
        np.array([[0.0, 0.0], [2.0, 2.0]]),
        index=["g0", "g1"],
        columns=["A", "B"],
    )
    # For compound A, weight all on g0 -> wMSE_A = 1 * (0-1)^2 = 1
    # For compound B, weight all on g1 -> wMSE_B = 1 * (2-2)^2 = 0
    weights = pd.DataFrame(
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        index=["g0", "g1"],
        columns=["A", "B"],
    )
    out = metrics.wmse(truth, pred, weights)
    assert out.loc["A"] == pytest.approx(1.0)
    assert out.loc["B"] == pytest.approx(0.0)


def test_wmse_series_and_broadcast_dataframe_agree():
    """Series weights and the same weights broadcast to a DataFrame must give equal results."""
    rng = np.random.default_rng(0)
    n_genes, n_compounds = 50, 8
    truth = pd.DataFrame(
        rng.normal(size=(n_genes, n_compounds)),
        index=[f"g{i:03d}" for i in range(n_genes)],
        columns=[f"c{i}" for i in range(n_compounds)],
    )
    pred = truth + rng.normal(scale=0.1, size=truth.shape)
    w_series = pd.Series(rng.exponential(size=n_genes), index=truth.index)
    w_series = w_series / w_series.sum()
    w_df = pd.DataFrame(
        np.broadcast_to(w_series.to_numpy()[:, None], truth.shape).copy(),
        index=truth.index,
        columns=truth.columns,
    )
    np.testing.assert_allclose(
        metrics.wmse(truth, pred, w_series).to_numpy(),
        metrics.wmse(truth, pred, w_df).to_numpy(),
    )


def test_score_compounds_accepts_dataframe_weights_and_reindexes_compounds():
    """Per-compound weights as a DataFrame should be reindexed to the
    aligned compound set inside score_compounds.
    """
    _, truth_test = make_synthetic_release(n_train=10, n_test=6, n_genes=40, seed=21)
    # Predict truth itself (perfect predictor).
    pred = truth_test.rename(columns={metrics.EXPRESSION_COL: metrics.PRED_COL})

    truth_mat, _ = metrics.align_long_frames(truth_test, pred)
    rng = np.random.default_rng(2)
    raw = rng.exponential(size=truth_mat.shape)
    w_mat = pd.DataFrame(
        raw / raw.sum(axis=0, keepdims=True),
        index=truth_mat.index,
        columns=truth_mat.columns,
    )
    pc = metrics.score_compounds(truth_test, pred, weights=w_mat)
    np.testing.assert_allclose(pc["wmse"].to_numpy(), 0.0, atol=1e-12)


def test_score_compounds_rejects_weights_missing_compound_columns():
    _, truth_test = make_synthetic_release(n_train=5, n_test=4, n_genes=10, seed=1)
    pred = truth_test.rename(columns={metrics.EXPRESSION_COL: metrics.PRED_COL})
    truth_mat, _ = metrics.align_long_frames(truth_test, pred)
    # Build a weights matrix that's missing one of the test compounds.
    incomplete = pd.DataFrame(
        np.ones((truth_mat.shape[0], 2)),
        index=truth_mat.index,
        columns=truth_mat.columns[:2],
    )
    with pytest.raises(ValueError, match="weights DataFrame is missing entries"):
        metrics.score_compounds(truth_test, pred, weights=incomplete)
