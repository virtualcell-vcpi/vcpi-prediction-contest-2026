"""Per-(compound, gene) weight matrices for the contest scoring panel.

Two recipes are shipped, both producing a ``(n_genes x n_compounds)``
weight matrix where every column sums to 1:

- :func:`compute_mejia_weights` — faithful port of Mejia et al. 2025
  (arXiv:2506.22641). For each compound, run a Welch t-test against
  every other compound's replicates ("vs Rest"), take ``|t|``, then
  per-column min-max -> square -> renormalize. The per-compound
  variance estimate is **floored at the per-gene pooled variance**
  to approximate scanpy's ``t-test_overestim_var`` stabilization,
  which matters for our DRUG-seq setup where most compounds have
  only ~2 replicates and the raw per-compound variance is noisy.
- :func:`compute_pooled_weights` — moderated alternative that
  replaces the per-compound variance with the per-gene pooled
  variance everywhere. Same downstream min-max -> square ->
  renormalize. Trades the paper's exact formula for a more stable
  denominator at low replicate count.

Both functions operate on **replicate-level** counts (one column per
``sequenced_id``), not pre-aggregated truth. The t-test needs
within-compound replicates to estimate variance; the pooled variant
needs them to estimate the pooled variance. The leaderboard server
runs these once on the official training counts and ships the
resulting ``weights.parquet`` for contestants to consume.

The downstream :mod:`vcpi_prediction_contest.metrics` functions
accept the resulting ``(n_genes x n_compounds)`` DataFrame as the
``weights`` argument; each compound is then scored against its own
column.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from loguru import logger

from vcpi_prediction_contest.expression import DEFAULT_SAMPLE_COL, _to_pandas
from vcpi_prediction_contest.metrics import COMPOUND_COL, GENE_COL

_MIN_REPLICATES_FOR_VAR = 2


def _replicate_level_log2_cpm(
    counts: object,
    metadata: object,
    *,
    sample_col: str,
    compound_col: str,
    gene_col: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Convert wide raw counts to wide replicate-level log2(CPM + 1).

    Returns
    -------
    log_cpm
        DataFrame indexed by ``gene_id`` with one column per replicate
        sample id (the matching sequenced_id). Empty-library samples
        are dropped entirely (rather than poisoned with NaN), so
        downstream variance estimates stay well-defined.
    sample_to_compound
        Series mapping each retained sample id to its compound id.
    """
    counts_df = _to_pandas(counts, name="counts")
    metadata_df = _to_pandas(metadata, name="metadata")

    if gene_col not in counts_df.columns:
        msg = f"counts is missing the `{gene_col}` column"
        raise ValueError(msg)
    for c in (sample_col, compound_col):
        if c not in metadata_df.columns:
            msg = f"metadata is missing the `{c}` column"
            raise ValueError(msg)

    counts_df = counts_df.set_index(gene_col)
    counts_df.columns = counts_df.columns.astype(str)
    metadata_df = metadata_df.copy()
    metadata_df[sample_col] = metadata_df[sample_col].astype(str)
    sample_to_compound = pd.Series(
        metadata_df[compound_col].to_numpy(),
        index=metadata_df[sample_col].to_numpy(),
        name=compound_col,
    )
    if sample_to_compound.index.has_duplicates:
        dupes = sample_to_compound.index[sample_to_compound.index.duplicated()].unique().tolist()
        msg = f"metadata has duplicate {sample_col} values: {dupes[:5]}"
        raise ValueError(msg)

    shared_samples = [s for s in counts_df.columns if s in sample_to_compound.index]
    if not shared_samples:
        msg = (
            f"No samples overlap between counts columns and metadata[{sample_col}]. "
            "Check that you passed the matching metadata frame for these counts."
        )
        raise ValueError(msg)
    counts_df = counts_df[shared_samples]
    sample_to_compound = sample_to_compound.loc[shared_samples]

    library_size = counts_df.sum(axis=0)
    nonempty = library_size > 0
    if (~nonempty).any():
        dropped = int((~nonempty).sum())
        logger.warning("Dropping {} empty-library sample(s) before weight derivation", dropped)
        counts_df = counts_df.loc[:, nonempty]
        sample_to_compound = sample_to_compound.loc[nonempty.index[nonempty]]

    cpm = counts_df.div(library_size[nonempty], axis=1) * 1e6
    log_cpm = np.log2(cpm + 1.0)
    return log_cpm, sample_to_compound


def _t_vs_rest_matrix(
    log_cpm: pd.DataFrame,
    sample_to_compound: pd.Series,
    *,
    variance_mode: Literal["per_compound_floored", "pooled"],
) -> pd.DataFrame:
    """Vectorized per-(compound, gene) Welch t-statistic vs. rest.

    For compound ``p`` and gene ``g``:

    .. code-block:: text

        t_{p,g} = (mu_{p,g} - mu_{rest,g}) /
                   sqrt(var_{p,g} / n_p + var_{rest,g} / n_rest)

    With ``variance_mode="per_compound_floored"`` (Mejia option 1)
    the variances are estimated within each compound and within rest,
    then floored at the per-gene pooled variance across all
    replicates (a stand-in for scanpy's ``overestim_var`` that
    prevents lucky-agreement inflation at n=2).

    With ``variance_mode="pooled"`` (option 2) the per-compound and
    rest variances are both replaced by the per-gene pooled variance,
    which is well-estimated at the full sample count.

    Compounds with fewer than two replicates fall back to using only
    the pooled variance (true Welch t with n_p=1 is undefined; this
    is the only sensible recourse), and a warning is emitted.
    """
    mat = log_cpm.to_numpy(dtype=float)
    gene_index = log_cpm.index
    n_genes, n_total = mat.shape

    compounds = pd.Index(sample_to_compound.unique(), name=COMPOUND_COL).sort_values()
    n_compounds = len(compounds)
    compound_to_idx = {c: i for i, c in enumerate(compounds)}
    sample_compound_idx = np.array(
        [compound_to_idx[c] for c in sample_to_compound.to_numpy()], dtype=np.int64
    )

    membership = np.zeros((n_total, n_compounds), dtype=float)
    membership[np.arange(n_total), sample_compound_idx] = 1.0
    n_per_compound = membership.sum(axis=0)
    if (n_per_compound == 0).any():
        msg = "Internal error: at least one compound has zero replicates after filtering"
        raise RuntimeError(msg)

    sum_x = mat.sum(axis=1, keepdims=True)
    sum_x2 = (mat * mat).sum(axis=1, keepdims=True)
    sum_x_per_p = mat @ membership
    sum_x2_per_p = (mat * mat) @ membership

    np_row = n_per_compound[np.newaxis, :]
    mean_p = sum_x_per_p / np_row
    n_rest_row = float(n_total) - np_row
    mean_rest = (sum_x - sum_x_per_p) / n_rest_row

    pooled_var_per_gene = (
        mat.var(axis=1, ddof=1) if n_total >= _MIN_REPLICATES_FOR_VAR else np.zeros(n_genes)
    )
    pooled_var_col = pooled_var_per_gene[:, np.newaxis]

    has_within_var = n_per_compound >= _MIN_REPLICATES_FOR_VAR
    safe_np = np.where(has_within_var[np.newaxis, :], np_row, 2.0)
    raw_var_p = np.where(
        has_within_var[np.newaxis, :],
        np.maximum(
            (sum_x2_per_p - safe_np * mean_p * mean_p) / np.maximum(safe_np - 1.0, 1.0),
            0.0,
        ),
        0.0,
    )

    has_rest_var = n_rest_row >= _MIN_REPLICATES_FOR_VAR
    safe_rest = np.where(has_rest_var, n_rest_row, 2.0)
    sum_x2_rest = sum_x2 - sum_x2_per_p
    raw_var_rest = np.where(
        has_rest_var,
        np.maximum(
            (sum_x2_rest - safe_rest * mean_rest * mean_rest) / np.maximum(safe_rest - 1.0, 1.0),
            0.0,
        ),
        0.0,
    )

    if variance_mode == "per_compound_floored":
        var_p = np.maximum(raw_var_p, pooled_var_col)
        var_rest = np.maximum(raw_var_rest, pooled_var_col)
    elif variance_mode == "pooled":
        var_p = np.broadcast_to(pooled_var_col, raw_var_p.shape).copy()
        var_rest = np.broadcast_to(pooled_var_col, raw_var_rest.shape).copy()
    else:
        msg = f"Unknown variance_mode: {variance_mode}"
        raise ValueError(msg)

    n_singleton = int((~has_within_var).sum())
    if n_singleton:
        logger.warning(
            "{} compound(s) have <2 replicates; using the pooled variance for those columns",
            n_singleton,
        )
        # Even in per_compound_floored mode, singletons can't estimate their own var.
        var_p[:, ~has_within_var] = pooled_var_col

    eps = 1e-12
    denom = np.sqrt(var_p / np_row + var_rest / n_rest_row)
    denom = np.where(denom < eps, eps, denom)
    t = (mean_p - mean_rest) / denom
    return pd.DataFrame(t, index=gene_index, columns=compounds)


def _minmax_square_normalize(abs_t: pd.DataFrame) -> pd.DataFrame:
    """Per-column min-max -> square -> normalize so columns sum to 1.

    Degenerate columns (max == min, e.g. compound where every gene has
    identical |t|) fall back to uniform ``1/n_genes`` weights.
    """
    mat = abs_t.to_numpy(dtype=float)
    n_genes = mat.shape[0]
    col_min = mat.min(axis=0)
    col_max = mat.max(axis=0)
    span = col_max - col_min
    degenerate = span <= 0
    safe_span = np.where(degenerate, 1.0, span)
    scaled = (mat - col_min[np.newaxis, :]) / safe_span[np.newaxis, :]
    squared = scaled * scaled
    col_sum = squared.sum(axis=0)
    safe_sum = np.where(col_sum > 0, col_sum, 1.0)
    normalized = squared / safe_sum[np.newaxis, :]
    # Both fallback paths converge on uniform.
    fallback = degenerate | (col_sum <= 0)
    if fallback.any():
        uniform = np.full(n_genes, 1.0 / n_genes)
        normalized[:, fallback] = uniform[:, np.newaxis]
        logger.info(
            "{} compound(s) had degenerate |t| or zero squared sum; using uniform weights for them",
            int(fallback.sum()),
        )
    return pd.DataFrame(normalized, index=abs_t.index, columns=abs_t.columns)


def compute_mejia_weights(
    counts: object,
    metadata: object,
    *,
    sample_col: str = DEFAULT_SAMPLE_COL,
    compound_col: str = COMPOUND_COL,
    gene_col: str = GENE_COL,
) -> pd.DataFrame:
    """Per-compound weights via Mejia 2025's t-test-vs-Rest recipe.

    Reproduces §3.3.3 of Mejia et al. 2025 (arXiv:2506.22641) on
    pseudobulk DRUG-seq counts:

    1. log2(CPM + 1) per replicate sample.
    2. Welch t-statistic for each (compound, gene) against every
       other compound's replicates ("vs Rest"). The per-compound
       variance estimate is floored at the per-gene pooled variance
       across all replicates — a robust approximation of scanpy's
       ``t-test_overestim_var`` that matters because most DRUG-seq
       compounds have only ~2 replicates.
    3. Per-column ``|t| -> min-max scale to [0, 1] -> square ->
       renormalize to sum to 1``.

    Returns
    -------
    pd.DataFrame
        ``(n_genes x n_compounds)`` matrix. Columns are compound IDs;
        rows are gene IDs (sorted alphabetically). Every column sums
        to 1. Degenerate columns (no variance in |t|) fall back to
        uniform ``1/n_genes`` weights.

    See Also
    --------
    compute_pooled_weights : moderated variant trading the paper's
        exact formula for a more stable denominator at low n.
    """
    log_cpm, sample_to_compound = _replicate_level_log2_cpm(
        counts,
        metadata,
        sample_col=sample_col,
        compound_col=compound_col,
        gene_col=gene_col,
    )
    t = _t_vs_rest_matrix(log_cpm, sample_to_compound, variance_mode="per_compound_floored")
    weights = _minmax_square_normalize(t.abs())
    logger.info(
        "Computed Mejia weights for {} compounds x {} genes",
        weights.shape[1],
        weights.shape[0],
    )
    return weights


def compute_pooled_weights(
    counts: object,
    metadata: object,
    *,
    sample_col: str = DEFAULT_SAMPLE_COL,
    compound_col: str = COMPOUND_COL,
    gene_col: str = GENE_COL,
) -> pd.DataFrame:
    """Per-compound weights via a pooled-variance moderated t-test.

    Identical to :func:`compute_mejia_weights` except every t-statistic
    uses the **per-gene pooled variance** across all replicates as both
    the within-compound and rest variance:

    .. code-block:: text

        t_{p,g} = (mu_{p,g} - mu_{rest,g}) /
                   sqrt(pooled_var_g * (1 / n_p + 1 / n_rest))

    This is similar in spirit to limma's empirical-Bayes-shrunk t. It
    sacrifices the paper's exact formula but gives a denominator that
    is well-estimated even when each compound has only two replicates.
    Same downstream min-max -> square -> renormalize.

    Returns
    -------
    pd.DataFrame
        Same shape and contract as :func:`compute_mejia_weights`.
    """
    log_cpm, sample_to_compound = _replicate_level_log2_cpm(
        counts,
        metadata,
        sample_col=sample_col,
        compound_col=compound_col,
        gene_col=gene_col,
    )
    t = _t_vs_rest_matrix(log_cpm, sample_to_compound, variance_mode="pooled")
    weights = _minmax_square_normalize(t.abs())
    logger.info(
        "Computed pooled-variance weights for {} compounds x {} genes",
        weights.shape[1],
        weights.shape[0],
    )
    return weights
