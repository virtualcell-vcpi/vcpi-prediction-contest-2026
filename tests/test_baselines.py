"""Tests for the reference baselines and adversarial probes.

We verify each predictor returns a valid long-format prediction frame
with the right schema and shape, and pin the calibration properties
the audit script depends on (perfect predictor pins wMSE at 0,
scaling predictor degrades wMSE, shuffling identities degrades wMSE
vs. the perfect predictor).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vcpi_prediction_contest import metrics
from vcpi_prediction_contest.baselines import (
    predict_constant,
    predict_mu_all_train,
    predict_mu_control,
    predict_per_gene_mean,
    predict_random_gaussian,
    predict_scaled_perfect,
    predict_shuffle_compounds,
    predict_technical_duplicate,
)


def _synthetic_truth(*, n_train, n_test, n_genes, seed):
    """Same shared-baseline synthetic builder as test_metrics."""
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
        return df.reset_index().melt(
            id_vars=metrics.GENE_COL,
            var_name=metrics.COMPOUND_COL,
            value_name=metrics.EXPRESSION_COL,
        )

    return (
        to_long(train_effects, train_compounds),
        to_long(test_effects, test_compounds),
        gene_index,
        train_compounds,
        test_compounds,
    )


def _assert_pred_schema(pred, *, test_compounds, gene_index):
    assert set(pred.columns) == {metrics.COMPOUND_COL, metrics.GENE_COL, metrics.PRED_COL}
    assert set(pred[metrics.COMPOUND_COL].unique()) == set(test_compounds)
    assert set(pred[metrics.GENE_COL].unique()) == set(gene_index)
    assert len(pred) == len(test_compounds) * len(gene_index)


# ---------------------------------------------------------------------------
# Paper baselines
# ---------------------------------------------------------------------------


def test_predict_mu_all_train_returns_same_vector_per_compound():
    truth_train, _, gene_index, _, _ = _synthetic_truth(n_train=10, n_test=4, n_genes=20, seed=1)
    test_compounds = ["x1", "x2", "x3"]
    pred = predict_mu_all_train(truth_train, test_compounds)
    _assert_pred_schema(pred, test_compounds=test_compounds, gene_index=gene_index)
    # Every compound should have the identical predicted vector (mode collapse).
    wide = pred.pivot_table(
        index=metrics.GENE_COL,
        columns=metrics.COMPOUND_COL,
        values=metrics.PRED_COL,
        aggfunc="mean",
    )
    np.testing.assert_allclose(wide["x1"].to_numpy(), wide["x2"].to_numpy())
    np.testing.assert_allclose(wide["x1"].to_numpy(), wide["x3"].to_numpy())


def test_predict_mu_control_uses_only_named_controls():
    truth_train, _, _, train_compounds, _ = _synthetic_truth(
        n_train=10, n_test=4, n_genes=20, seed=3
    )
    controls = train_compounds[:2]
    pred = predict_mu_control(truth_train, ["t1"], controls)
    # The predicted per-gene vector should equal the mean of the two control
    # compounds' truth.
    wide_train = truth_train.pivot_table(
        index=metrics.GENE_COL,
        columns=metrics.COMPOUND_COL,
        values=metrics.EXPRESSION_COL,
        aggfunc="mean",
    )
    expected = wide_train[controls].mean(axis=1).sort_index()
    got = (
        pred[pred[metrics.COMPOUND_COL] == "t1"]
        .set_index(metrics.GENE_COL)[metrics.PRED_COL]
        .sort_index()
    )
    np.testing.assert_allclose(got.to_numpy(), expected.to_numpy())


def test_predict_mu_control_raises_when_no_controls_present():
    truth_train, *_ = _synthetic_truth(n_train=4, n_test=2, n_genes=5, seed=4)
    with pytest.raises(ValueError, match="None of the requested control"):
        predict_mu_control(truth_train, ["t1"], ["not_in_train"])


def test_predict_technical_duplicate_perfect_scores_when_noiseless():
    _, truth_test, *_ = _synthetic_truth(n_train=5, n_test=6, n_genes=20, seed=5)
    pred = predict_technical_duplicate(truth_test)
    pc = metrics.score_compounds(truth_test, pred)
    np.testing.assert_allclose(pc["wmse"].to_numpy(), 0.0, atol=1e-12)


def test_predict_technical_duplicate_with_noise_degrades_metrics():
    _, truth_test, _, _, _ = _synthetic_truth(n_train=5, n_test=6, n_genes=20, seed=6)
    pred = predict_technical_duplicate(truth_test, noise_scale=0.5, seed=42)
    pc = metrics.score_compounds(truth_test, pred)
    assert (pc["wmse"].to_numpy() > 0).all()


# ---------------------------------------------------------------------------
# Contestant-side baselines / adversarial probes
# ---------------------------------------------------------------------------


def test_predict_constant_schema_and_constancy():
    test_compounds = ["a", "b", "c"]
    gene_index = [f"g{i}" for i in range(5)]
    pred = predict_constant(test_compounds, gene_index, value=3.14)
    _assert_pred_schema(pred, test_compounds=test_compounds, gene_index=gene_index)
    np.testing.assert_allclose(pred[metrics.PRED_COL].to_numpy(), 3.14)


def test_predict_per_gene_mean_alias_matches_mu_all_train():
    truth_train, _, _, _, _ = _synthetic_truth(n_train=8, n_test=3, n_genes=15, seed=7)
    test_compounds = ["t1", "t2"]
    a = predict_per_gene_mean(truth_train, test_compounds)
    b = predict_mu_all_train(truth_train, test_compounds)
    pd.testing.assert_frame_equal(
        a.sort_values([metrics.COMPOUND_COL, metrics.GENE_COL]).reset_index(drop=True),
        b.sort_values([metrics.COMPOUND_COL, metrics.GENE_COL]).reset_index(drop=True),
    )


def test_predict_scaled_perfect_attack_degrades_wmse_via_score_compounds():
    """Scaling predictions by 2x degrades wMSE on every compound."""
    _, truth_test, _, _, _ = _synthetic_truth(n_train=5, n_test=8, n_genes=30, seed=8)
    pred_perfect = predict_technical_duplicate(truth_test)
    pred_scaled = predict_scaled_perfect(truth_test, scale=2.0)
    pc_perfect = metrics.score_compounds(truth_test, pred_perfect)
    pc_scaled = metrics.score_compounds(truth_test, pred_scaled)
    assert (pc_scaled["wmse"].to_numpy() > pc_perfect["wmse"].to_numpy()).all()


def test_predict_shuffle_compounds_is_a_derangement():
    """No compound's predicted vector should equal its own truth vector."""
    _, truth_test, _, _, test_compounds = _synthetic_truth(n_train=5, n_test=8, n_genes=20, seed=9)
    pred = predict_shuffle_compounds(truth_test, seed=0)
    wide_truth = truth_test.pivot_table(
        index=metrics.GENE_COL,
        columns=metrics.COMPOUND_COL,
        values=metrics.EXPRESSION_COL,
        aggfunc="mean",
    )
    wide_pred = pred.pivot_table(
        index=metrics.GENE_COL,
        columns=metrics.COMPOUND_COL,
        values=metrics.PRED_COL,
        aggfunc="mean",
    )
    for c in test_compounds:
        assert not np.allclose(wide_pred[c].to_numpy(), wide_truth[c].to_numpy()), (
            f"shuffle predictor returned its own truth for {c}"
        )


def test_predict_shuffle_compounds_degrades_wmse_vs_perfect():
    """Shuffling identities should push wMSE strictly above the perfect predictor.

    Every shuffled prediction is mismatched to its true compound's
    expression vector, so the per-compound squared errors are bounded
    away from zero and the mean wMSE must exceed the perfect predictor's 0.
    """
    _, truth_test, _, _, _ = _synthetic_truth(n_train=5, n_test=20, n_genes=40, seed=10)
    pred_perfect = predict_technical_duplicate(truth_test)
    pred_shuffle = predict_shuffle_compounds(truth_test, seed=11)
    pc_perfect = metrics.score_compounds(truth_test, pred_perfect)
    pc_shuffle = metrics.score_compounds(truth_test, pred_shuffle)
    assert pc_shuffle["wmse"].mean() > pc_perfect["wmse"].mean() + 1e-9


def test_predict_random_gaussian_schema():
    test_compounds = ["a", "b"]
    gene_index = [f"g{i}" for i in range(10)]
    pred = predict_random_gaussian(test_compounds, gene_index, mean=2.0, sd=0.5, seed=0)
    _assert_pred_schema(pred, test_compounds=test_compounds, gene_index=gene_index)
    # Deterministic seed -> same output across runs.
    pred2 = predict_random_gaussian(test_compounds, gene_index, mean=2.0, sd=0.5, seed=0)
    pd.testing.assert_frame_equal(pred, pred2)
