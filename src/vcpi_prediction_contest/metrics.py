"""Scoring metric for the VCPI expression-prediction contest.

The task: for each held-out compound and each scored gene, predict the
**per-(compound, gene) mean expression** measured by VCPI's DRUG-seq
pipeline (currently log2(CPM + 1), but the metric panel is unit-agnostic —
any non-negative absolute-expression scale works).

The leaderboard reports a single per-compound metric, which the
contest scorer averages over the test compound set:

- **wMSE** — per-gene-weighted mean squared error between predicted
  and true expression. Lower is better. Bounded below by 0.

Inputs are long-format pandas DataFrames:

- ``truth``: ``compound``, ``gene_id``, ``expression`` (the
  per-compound mean of the chosen expression scale).
- ``prediction``: ``compound``, ``gene_id``, ``predicted_expression``.

The lower-level matrix-valued metric :func:`wmse` takes ``(n_genes x
n_compounds)`` dense DataFrames / ndarrays and returns a per-compound
Series. :func:`score_compounds` orchestrates: aligns the long frames,
derives weights from the truth (or accepts user-supplied weights for
the canonical contest case), and returns a per-compound metric table.

The ``weights`` argument can be either:

- a length-``n_genes`` per-gene Series / 1-D ndarray (broadcast to every
  compound; legacy "global variance weights"), or
- a ``(n_genes x n_compounds)`` DataFrame / 2-D ndarray (per-compound
  weight columns; the shape produced by
  :func:`~vcpi_prediction_contest.weights.compute_mejia_weights` and
  :func:`~vcpi_prediction_contest.weights.compute_pooled_weights`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Iterable

# ---------------------------------------------------------------------------
# Canonical column names — the long-format schema the loaders enforce.
# ---------------------------------------------------------------------------
COMPOUND_COL = "compound"
GENE_COL = "gene_id"
EXPRESSION_COL = "expression"
PRED_COL = "predicted_expression"


# ---------------------------------------------------------------------------
# Validation & alignment
# ---------------------------------------------------------------------------


def _require_columns(df: pd.DataFrame, cols: Iterable[str], *, name: str) -> None:
    missing = sorted(set(cols) - set(df.columns))
    if missing:
        msg = f"{name} is missing required columns: {missing}"
        raise ValueError(msg)


def _pivot(df: pd.DataFrame, *, value_col: str) -> pd.DataFrame:
    """Long -> dense ``(gene x compound)`` matrix.

    Duplicate ``(compound, gene_id)`` rows are averaged. In well-formed
    contest data each pair appears exactly once, so this is just a
    safety net.
    """
    return df.pivot_table(
        index=GENE_COL,
        columns=COMPOUND_COL,
        values=value_col,
        aggfunc="mean",
    )


def align_long_frames(
    truth: pd.DataFrame,
    prediction: pd.DataFrame,
    *,
    gene_filter: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align long ``truth`` and ``prediction`` onto a common matrix grid.

    Both inputs are pivoted to ``(gene x compound)`` and reindexed to
    the same gene set (``gene_filter`` if given, else the union of
    genes in ``truth``) and the same compound set (intersection of
    ``truth`` and ``prediction`` compounds).
    """
    _require_columns(truth, [COMPOUND_COL, GENE_COL, EXPRESSION_COL], name="truth")
    _require_columns(prediction, [COMPOUND_COL, GENE_COL, PRED_COL], name="prediction")

    truth_compounds = set(truth[COMPOUND_COL].unique())
    pred_compounds = set(prediction[COMPOUND_COL].unique())
    shared = sorted(truth_compounds & pred_compounds)
    if not shared:
        msg = "No compounds in common between truth and prediction."
        raise ValueError(msg)

    if gene_filter is None:
        gene_index = sorted(set(truth[GENE_COL].unique()))
    else:
        gene_index = sorted(set(gene_filter))

    truth_mat = _pivot(truth, value_col=EXPRESSION_COL).reindex(index=gene_index, columns=shared)
    pred_mat = _pivot(prediction, value_col=PRED_COL).reindex(index=gene_index, columns=shared)

    if truth_mat.isna().any().any():
        n_missing = int(truth_mat.isna().sum().sum())
        msg = f"truth has {n_missing} missing (gene, compound) entries on the aligned grid"
        raise ValueError(msg)
    if pred_mat.isna().any().any():
        n_missing = int(pred_mat.isna().sum().sum())
        msg = f"prediction has {n_missing} missing (gene, compound) entries on the aligned grid"
        raise ValueError(msg)
    return truth_mat, pred_mat


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------


def compute_variance_weights(
    truth_train: pd.DataFrame | np.ndarray,
    *,
    gene_filter: Iterable[str] | None = None,
) -> pd.Series:
    """Per-gene weights from training-set across-compound variance.

    Computes ``Var_compound(expression)`` for every gene, normalized to
    sum to 1. Genes that vary a lot between training compounds get more
    weight; constitutively-expressed housekeeping genes get less. This
    is the absolute-expression analog of ``|stat|``-derived weighting
    in the old log2FC scoring framework.

    Parameters
    ----------
    truth_train
        Either a long-format DataFrame with columns ``compound``,
        ``gene_id``, ``expression``, or a dense ``(n_genes x n_compounds)``
        matrix / DataFrame indexed by ``gene_id``.
    gene_filter
        Optional gene-id subset to restrict to.

    Returns
    -------
    pd.Series
        Indexed by ``gene_id``, sums to 1. Falls back to uniform weights
        when the across-compound variance is degenerate (all zeros).
    """
    if isinstance(truth_train, pd.DataFrame) and EXPRESSION_COL in truth_train.columns:
        mat = _pivot(truth_train, value_col=EXPRESSION_COL)
    else:
        mat = pd.DataFrame(truth_train).copy()
        if mat.index.name is None:
            mat.index.name = GENE_COL

    if gene_filter is not None:
        mat = mat.reindex(index=sorted(set(gene_filter)))

    gene_var = mat.var(axis=1, ddof=0).fillna(0.0)
    total = float(gene_var.sum())
    if total <= 0:
        n = len(gene_var)
        return pd.Series(np.full(n, 1.0 / max(n, 1)), index=gene_var.index, name="weight")
    weights = gene_var / total
    weights.name = "weight"
    return weights


# ---------------------------------------------------------------------------
# Per-compound metric
# ---------------------------------------------------------------------------


def _as_matrix(x: pd.DataFrame | np.ndarray) -> np.ndarray:
    if isinstance(x, pd.DataFrame):
        return x.to_numpy()
    return np.asarray(x)


def _align_weights_2d(
    weights: pd.Series | pd.DataFrame | np.ndarray,
    *,
    gene_index: pd.Index,
    compound_index: pd.Index,
) -> np.ndarray:
    """Return a dense ``(n_genes x n_compounds)`` weight array.

    Accepts:
    - ``pd.Series`` or 1-D ndarray indexed by ``gene_id``: broadcast to
      every compound (the legacy "global per-gene weight vector" case).
    - ``pd.DataFrame`` or 2-D ndarray: per-compound weight columns.
      DataFrames are reindexed to ``gene_index`` and ``compound_index``;
      missing entries raise.
    """
    n_genes = len(gene_index)
    n_compounds = len(compound_index)
    if isinstance(weights, pd.DataFrame):
        w = weights.reindex(index=gene_index, columns=compound_index)
        if w.isna().any().any():
            missing_g = int(w.isna().any(axis=1).sum())
            missing_c = int(w.isna().any(axis=0).sum())
            msg = (
                f"weights DataFrame is missing entries: {missing_g} genes "
                f"and {missing_c} compounds not covered by the matrix"
            )
            raise ValueError(msg)
        return w.to_numpy(dtype=float)
    if isinstance(weights, pd.Series):
        w = weights.reindex(gene_index)
        if w.isna().any():
            missing = int(w.isna().sum())
            msg = f"weights series is missing {missing} of the scored gene IDs"
            raise ValueError(msg)
        return np.broadcast_to(w.to_numpy(dtype=float)[:, None], (n_genes, n_compounds)).copy()
    arr = np.asarray(weights, dtype=float)
    if arr.ndim == 1:
        if arr.shape != (n_genes,):
            msg = f"weights 1-D array must be length {n_genes}, got {arr.shape}"
            raise ValueError(msg)
        return np.broadcast_to(arr[:, None], (n_genes, n_compounds)).copy()
    if arr.ndim == 2:  # noqa: PLR2004
        if arr.shape != (n_genes, n_compounds):
            msg = f"weights 2-D array must be shape ({n_genes}, {n_compounds}), got {arr.shape}"
            raise ValueError(msg)
        return arr.astype(float, copy=False)
    msg = f"weights must be 1-D or 2-D, got {arr.ndim}-D array"
    raise ValueError(msg)


def _shape_with_indices(
    truth: pd.DataFrame | np.ndarray, prediction: pd.DataFrame | np.ndarray
) -> tuple[np.ndarray, np.ndarray, pd.Index, pd.Index]:
    truth_arr = _as_matrix(truth)
    pred_arr = _as_matrix(prediction)
    if truth_arr.shape != pred_arr.shape:
        msg = f"shape mismatch: truth {truth_arr.shape} vs prediction {pred_arr.shape}"
        raise ValueError(msg)
    n_genes, n_compounds = truth_arr.shape
    gene_index = truth.index if isinstance(truth, pd.DataFrame) else pd.RangeIndex(n_genes)
    compound_index = (
        truth.columns if isinstance(truth, pd.DataFrame) else pd.RangeIndex(n_compounds)
    )
    return truth_arr, pred_arr, gene_index, compound_index


def wmse(
    truth: pd.DataFrame,
    prediction: pd.DataFrame,
    weights: pd.Series | pd.DataFrame | np.ndarray,
) -> pd.Series:
    """Per-compound weighted MSE.

    ``truth`` and ``prediction`` are aligned ``(n_genes x n_compounds)``
    DataFrames. ``weights`` is either a length-``n_genes`` per-gene
    weight vector (broadcast to every compound) or a
    ``(n_genes x n_compounds)`` per-compound weight matrix (the shape
    produced by the Mejia / pooled weight functions).
    """
    truth_arr, pred_arr, gene_index, compound_index = _shape_with_indices(truth, prediction)
    w_mat = _align_weights_2d(weights, gene_index=gene_index, compound_index=compound_index)
    sq = (pred_arr - truth_arr) ** 2
    out = (w_mat * sq).sum(axis=0)
    return pd.Series(out, index=compound_index, name="wmse")


# ---------------------------------------------------------------------------
# Top-level contest entry points
# ---------------------------------------------------------------------------


def _safe_mean(s: pd.Series) -> float:
    if len(s) == 0:
        return float("nan")
    return float(s.mean(skipna=True))


def score_compounds(
    truth: pd.DataFrame,
    prediction: pd.DataFrame,
    *,
    gene_filter: Iterable[str] | None = None,
    weights: pd.Series | pd.DataFrame | np.ndarray | None = None,
) -> pd.DataFrame:
    """Per-compound scoring table for an expression-prediction submission.

    Aligns ``truth`` and ``prediction`` onto a common ``(gene x compound)``
    grid restricted to ``gene_filter`` (or every gene in ``truth`` when
    omitted), then evaluates the contest metric per compound: ``wmse``.

    The leaderboard server passes the **canonical contest weights**
    (derived once from the official training counts). When called
    locally without ``weights``, the function falls back to the legacy
    global per-gene variance weights derived from ``truth`` itself —
    convenient for development, but the numbers will not match the
    leaderboard unless you pass the same weights.

    Parameters
    ----------
    truth
        Long-format DataFrame with columns ``compound``, ``gene_id``,
        ``expression``.
    prediction
        Long-format DataFrame with columns ``compound``, ``gene_id``,
        ``predicted_expression``.
    gene_filter
        Optional iterable of ``gene_id``s to score on.
    weights
        Either a length-``n_genes`` per-gene Series / 1-D ndarray
        (broadcast to every compound) or a
        ``(n_genes x n_compounds)`` DataFrame / 2-D ndarray
        (per-compound weight columns, the shape produced by the
        Mejia / pooled weight functions). When ``None`` (default),
        :func:`compute_variance_weights` is derived from ``truth``.

    Returns
    -------
    pd.DataFrame
        Indexed by ``compound``, with a single column ``wmse``.
    """
    truth_mat, pred_mat = align_long_frames(truth, prediction, gene_filter=gene_filter)
    gene_index = truth_mat.index
    compound_index = truth_mat.columns

    if weights is None:
        w: pd.Series | pd.DataFrame = compute_variance_weights(truth_mat)
    elif isinstance(weights, pd.DataFrame):
        w = weights.reindex(index=gene_index, columns=compound_index)
        if w.isna().any().any():
            missing_g = int(w.isna().any(axis=1).sum())
            missing_c = int(w.isna().any(axis=0).sum())
            msg = (
                f"weights DataFrame is missing entries: {missing_g} genes "
                f"and {missing_c} compounds not covered by the matrix"
            )
            raise ValueError(msg)
    elif isinstance(weights, pd.Series):
        w = weights.reindex(gene_index)
        if w.isna().any():
            missing = int(w.isna().sum())
            msg = f"weights series is missing {missing} of the scored gene IDs"
            raise ValueError(msg)
    else:
        arr = np.asarray(weights)
        if arr.ndim == 2:  # noqa: PLR2004
            w = pd.DataFrame(arr, index=gene_index, columns=compound_index)
        else:
            w = pd.Series(arr, index=gene_index, name="weight")

    return pd.DataFrame({"wmse": wmse(truth_mat, pred_mat, w)})


def aggregate_leaderboards(per_compound: pd.DataFrame) -> dict[str, object]:
    """Aggregate per-compound metrics into the leaderboard summary.

    Reports the arithmetic mean of the per-compound ``wmse`` across all
    test compounds, plus the compound count.

    Returns
    -------
    dict
        ``leaderboard`` with keys ``n_compounds`` and ``wmse_mean``.
    """
    return {
        "n_compounds": len(per_compound),
        "wmse_mean": _safe_mean(per_compound["wmse"]),
    }
