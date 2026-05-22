"""Reference baselines and adversarial probes for the scoring panel.

This module ships a handful of **non-learned predictors** whose
behavior under the contest scoring panel is meant to be predictable.
They serve three purposes:

1. **Calibration anchors for the leaderboard**. The audit script in
   ``scripts/baseline_audit.py`` uses these to verify wMSE behaves as
   advertised — e.g. :func:`predict_technical_duplicate` should give
   wMSE = 0, :func:`predict_mu_all_train` should give a moderate
   non-zero wMSE that any real predictor must beat, etc.
2. **Sanity-check predictors for contestants**. If your model can't
   beat :func:`predict_mu_all_train`, it has mode-collapsed onto the
   per-gene mean and isn't picking up any per-compound signal.
3. **Adversarial probes** (:func:`predict_scaled_perfect`,
   :func:`predict_shuffle_compounds`, :func:`predict_constant`,
   :func:`predict_random_gaussian`) that intentionally break some
   property the metric should be sensitive to (magnitude, identity,
   per-compound differentiation) so the audit can verify wMSE
   catches the right adversary.

Every function returns a long-format ``predicted_expression`` frame
ready to pass to :func:`vcpi_prediction_contest.metrics.score_compounds`.
The ``test_compounds`` argument is always the list of compound IDs you
want to predict for.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vcpi_prediction_contest.metrics import (
    COMPOUND_COL,
    EXPRESSION_COL,
    GENE_COL,
    PRED_COL,
    _pivot,
)


def _wide_train_truth(
    truth_train: pd.DataFrame,
    *,
    gene_filter: list[str] | None = None,
) -> pd.DataFrame:
    """Truth train pivoted to (gene_id x compound) on the gene set."""
    mat = _pivot(truth_train, value_col=EXPRESSION_COL)
    if gene_filter is not None:
        mat = mat.reindex(index=sorted(set(gene_filter)))
    return mat


def _broadcast_to_compounds(
    per_gene: pd.Series,
    test_compounds: list[str],
) -> pd.DataFrame:
    """Repeat a per-gene vector across every test compound -> long frame."""
    wide = pd.DataFrame(
        np.broadcast_to(per_gene.to_numpy()[:, None], (len(per_gene), len(test_compounds))).copy(),
        index=per_gene.index,
        columns=test_compounds,
    )
    wide.index.name = GENE_COL
    return wide.reset_index().melt(
        id_vars=GENE_COL,
        var_name=COMPOUND_COL,
        value_name=PRED_COL,
    )


# ---------------------------------------------------------------------------
# Paper baselines (calibration anchors)
# ---------------------------------------------------------------------------


def predict_mu_all_train(
    truth_train: pd.DataFrame,
    test_compounds: list[str],
    *,
    gene_filter: list[str] | None = None,
) -> pd.DataFrame:
    """Predict the per-gene mean across all training compounds for every test compound.

    This is the **mode-collapse predictor**: the same vector is
    returned for every compound. Under wMSE it usually beats most
    naive models because it lives in the right neighborhood of the
    data; any compound-aware predictor worth submitting should beat
    it by a clear margin.
    """
    mat = _wide_train_truth(truth_train, gene_filter=gene_filter)
    per_gene = mat.mean(axis=1)
    return _broadcast_to_compounds(per_gene, test_compounds)


def predict_mu_control(
    truth_train: pd.DataFrame,
    test_compounds: list[str],
    control_compounds: list[str],
    *,
    gene_filter: list[str] | None = None,
) -> pd.DataFrame:
    """Predict the per-gene mean over the named control compounds in training.

    Same shape as :func:`predict_mu_all_train` but the reference is
    restricted to the named ``control_compounds`` (typically the
    DMSO / negative-control wells the contestants opt to keep in).
    Useful as a "perturbed-but-equivalent-to-untreated" baseline.
    """
    mat = _wide_train_truth(truth_train, gene_filter=gene_filter)
    shared = [c for c in control_compounds if c in mat.columns]
    if not shared:
        msg = (
            f"None of the requested control compounds {control_compounds[:5]} appear in truth_train"
        )
        raise ValueError(msg)
    per_gene = mat[shared].mean(axis=1)
    return _broadcast_to_compounds(per_gene, test_compounds)


def predict_technical_duplicate(
    truth_test: pd.DataFrame,
    test_compounds: list[str] | None = None,
    *,
    gene_filter: list[str] | None = None,
    noise_scale: float = 0.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Predict each test compound's truth (optionally with isotropic noise).

    This is the **upper-bound baseline** — the predictor's per-compound
    answer is the truth, optionally with a small amount of additive
    Gaussian noise to mimic a replicate measurement. With
    ``noise_scale=0`` it pins wMSE at its optimum of 0. With
    ``noise_scale > 0`` the audit can confirm wMSE degrades gracefully.
    """
    mat = _pivot(truth_test, value_col=EXPRESSION_COL)
    if gene_filter is not None:
        mat = mat.reindex(index=sorted(set(gene_filter)))
    if test_compounds is not None:
        mat = mat.reindex(columns=test_compounds)
    if noise_scale > 0:
        rng = np.random.default_rng(seed)
        mat = mat + rng.normal(scale=noise_scale, size=mat.shape)
    mat.index.name = GENE_COL
    return mat.reset_index().melt(
        id_vars=GENE_COL,
        var_name=COMPOUND_COL,
        value_name=PRED_COL,
    )


# ---------------------------------------------------------------------------
# Contestant-side baselines / adversarial probes
# ---------------------------------------------------------------------------


def predict_constant(
    test_compounds: list[str],
    gene_index: list[str],
    *,
    value: float = 0.0,
) -> pd.DataFrame:
    """Predict the same scalar for every (compound, gene).

    The trivial "predict zero" or "predict 5" baseline. Useful for
    showing how badly wMSE punishes a predictor with no signal at
    all: wMSE blows up by the size of the data.
    """
    rows = [(c, g, value) for c in test_compounds for g in gene_index]
    return pd.DataFrame(rows, columns=[COMPOUND_COL, GENE_COL, PRED_COL])


def predict_per_gene_mean(
    truth_train: pd.DataFrame,
    test_compounds: list[str],
    *,
    gene_filter: list[str] | None = None,
) -> pd.DataFrame:
    """Alias for :func:`predict_mu_all_train` under the contestant-facing name.

    Kept for naming continuity with the previous log2FoldChange-era
    baselines that contestants used.
    """
    return predict_mu_all_train(truth_train, test_compounds, gene_filter=gene_filter)


def predict_scaled_perfect(
    truth_test: pd.DataFrame,
    *,
    scale: float,
    gene_filter: list[str] | None = None,
) -> pd.DataFrame:
    """Predict ``scale * truth`` per (compound, gene).

    The constant-scaling leaderboard hack: every prediction is a fixed
    multiple of truth. wMSE penalizes the magnitude error
    quadratically (and asymmetrically — under-scaling and over-scaling
    don't degrade equally), so it catches this attack. Set ``scale !=
    1`` to confirm wMSE reports a strictly worse score than the
    perfect predictor.
    """
    mat = _pivot(truth_test, value_col=EXPRESSION_COL)
    if gene_filter is not None:
        mat = mat.reindex(index=sorted(set(gene_filter)))
    mat = mat * scale
    mat.index.name = GENE_COL
    return mat.reset_index().melt(
        id_vars=GENE_COL,
        var_name=COMPOUND_COL,
        value_name=PRED_COL,
    )


def predict_shuffle_compounds(
    truth_test: pd.DataFrame,
    *,
    seed: int = 0,
    gene_filter: list[str] | None = None,
) -> pd.DataFrame:
    """Predict the truth, but with the compound -> truth mapping shuffled.

    The "anonymous correct answers" attack: every prediction comes
    from the test distribution but the compound identities are
    scrambled. wMSE still degrades vs. the perfect predictor because
    each compound's prediction is now mismatched to its true
    expression vector, even though the predictions still live in
    roughly the right numerical neighborhood.
    """
    mat = _pivot(truth_test, value_col=EXPRESSION_COL)
    if gene_filter is not None:
        mat = mat.reindex(index=sorted(set(gene_filter)))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(mat.shape[1])
    # Make sure the permutation is a derangement (no compound maps to itself)
    # to keep the attack worst-case.
    while np.any(perm == np.arange(mat.shape[1])):
        perm = rng.permutation(mat.shape[1])
    shuffled = pd.DataFrame(
        mat.to_numpy()[:, perm],
        index=mat.index,
        columns=mat.columns,
    )
    shuffled.index.name = GENE_COL
    return shuffled.reset_index().melt(
        id_vars=GENE_COL,
        var_name=COMPOUND_COL,
        value_name=PRED_COL,
    )


def predict_random_gaussian(
    test_compounds: list[str],
    gene_index: list[str],
    *,
    mean: float = 0.0,
    sd: float = 1.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Predict iid Gaussian noise for every (compound, gene).

    A pure-noise predictor: wMSE should be bad. Useful as the
    "definitely-doesn't-work" floor in the calibration table.
    """
    rng = np.random.default_rng(seed)
    n_genes = len(gene_index)
    n_compounds = len(test_compounds)
    mat = pd.DataFrame(
        rng.normal(loc=mean, scale=sd, size=(n_genes, n_compounds)),
        index=gene_index,
        columns=test_compounds,
    )
    mat.index.name = GENE_COL
    return mat.reset_index().melt(
        id_vars=GENE_COL,
        var_name=COMPOUND_COL,
        value_name=PRED_COL,
    )
