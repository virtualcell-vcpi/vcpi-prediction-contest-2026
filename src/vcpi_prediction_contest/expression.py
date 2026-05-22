"""Canonical recipe: vcpi-client raw UMI counts -> per-(compound, gene) expression.

The contest target is **expression**, defined as the per-(compound, gene)
mean of ``log2(CPM + 1)`` across replicate samples:

.. code-block:: text

    CPM_{g,s}        = 1e6 * count_{g,s} / sum_g count_{g,s}
    log2_CPM_{g,s}   = log2(CPM_{g,s} + 1)
    expression_{g,c} = mean_{s in replicates(c)} log2_CPM_{g,s}

This module ships :func:`counts_to_expression`, the one-shot helper that
takes vcpi-client's wide counts + metadata frames and produces the
long-format ``(compound, gene_id, expression)`` truth table the
:mod:`vcpi_prediction_contest.metrics` scorer consumes. **Use this exact
function** to build your training expression table from raw counts; if
you roll your own normalization you risk training in a different
numerical space than the leaderboard scores in.

The caller is responsible for filtering replicates before invoking this
function — e.g. restricting metadata to ``compound_concentration == 10
& cell_line == 'THP-1' & timepoint == '24h'`` to match the contest
condition, and dropping any samples they don't want included.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from vcpi_prediction_contest.metrics import (
    COMPOUND_COL,
    EXPRESSION_COL,
    GENE_COL,
)

DEFAULT_SAMPLE_COL = "sequenced_id"
_MAX_DUPES_IN_MESSAGE = 5


def _to_pandas(df: object, *, name: str) -> pd.DataFrame:
    """Coerce input to pandas.

    Accepts a pandas DataFrame or anything with ``.to_pandas()`` (e.g.
    a polars DataFrame, which is what ``vcpi-client`` returns).
    """
    if isinstance(df, pd.DataFrame):
        return df
    if hasattr(df, "to_pandas"):
        return df.to_pandas()
    msg = f"{name} must be a pandas DataFrame or expose .to_pandas() (got {type(df).__name__})"
    raise TypeError(msg)


def counts_to_expression(
    counts: object,
    metadata: object,
    *,
    sample_col: str = DEFAULT_SAMPLE_COL,
    compound_col: str = COMPOUND_COL,
    gene_col: str = GENE_COL,
) -> pd.DataFrame:
    """Aggregate raw UMI counts into the canonical contest expression table.

    Parameters
    ----------
    counts
        Wide-format gene-by-sample UMI counts as returned by
        :func:`vcpi.load_experiment`. Must have one column named
        ``gene_col`` (the row labels) and the remaining columns named
        after sample IDs that appear in ``metadata[sample_col]``.
        Accepts pandas or polars (anything with ``.to_pandas()``).
    metadata
        One row per sample with at least ``sample_col`` and
        ``compound_col``. The caller is responsible for filtering
        upstream — e.g. to the contest condition (10 uM, THP-1, 24h)
        and to any samples they wish to exclude.
    sample_col
        Name of the sample-id column in ``metadata`` (default
        ``"sequenced_id"``).
    compound_col
        Name of the compound-id column in ``metadata`` (default
        ``"compound"``). Output rows will be one per unique value.
    gene_col
        Name of the gene-id column in ``counts`` (default
        ``"gene_id"``).

    Returns
    -------
    pd.DataFrame
        Long-format ``(compound, gene_id, expression)`` table ready to
        pass to :func:`vcpi_prediction_contest.score_compounds` or save
        to disk as ``truth.parquet`` / ``train_expression.parquet``.
    """
    counts = _to_pandas(counts, name="counts")
    metadata = _to_pandas(metadata, name="metadata")

    if gene_col not in counts.columns:
        msg = f"counts is missing the `{gene_col}` column"
        raise ValueError(msg)
    for c in (sample_col, compound_col):
        if c not in metadata.columns:
            msg = f"metadata is missing the `{c}` column"
            raise ValueError(msg)

    # Tidy types so the joins/groupbys behave deterministically.
    counts = counts.set_index(gene_col)
    counts.columns = counts.columns.astype(str)
    metadata = metadata.copy()
    metadata[sample_col] = metadata[sample_col].astype(str)
    sample_to_compound = pd.Series(
        metadata[compound_col].to_numpy(),
        index=metadata[sample_col].to_numpy(),
        name=compound_col,
    )
    # Last write wins for duplicated sample IDs — flag if it ever happens.
    if sample_to_compound.index.has_duplicates:
        dupes = sample_to_compound.index[sample_to_compound.index.duplicated()].unique().tolist()
        head = dupes[:_MAX_DUPES_IN_MESSAGE]
        suffix = "..." if len(dupes) > _MAX_DUPES_IN_MESSAGE else ""
        msg = f"metadata has duplicate {sample_col} values: {head}{suffix}"
        raise ValueError(msg)

    shared_samples = [s for s in counts.columns if s in sample_to_compound.index]
    if not shared_samples:
        msg = (
            f"No samples overlap between counts columns and metadata[{sample_col}]. "
            "Check that you passed the matching metadata frame for these counts."
        )
        raise ValueError(msg)
    n_dropped = counts.shape[1] - len(shared_samples)
    if n_dropped:
        logger.warning(
            "Dropping {} count columns absent from metadata[{}] (keeping {})",
            n_dropped,
            sample_col,
            len(shared_samples),
        )
    counts = counts[shared_samples]
    sample_to_compound = sample_to_compound.loc[shared_samples]

    # CPM = 1e6 * count_{g,s} / sum_g count_{g,s}. Samples with zero
    # library size cannot be normalized and become NaN; we let pandas'
    # groupby skipna handling drop them from the per-compound mean.
    library_size = counts.sum(axis=0)
    n_empty = int((library_size == 0).sum())
    if n_empty:
        logger.warning(
            "{} sample(s) have zero total counts and will not contribute to any compound mean",
            n_empty,
        )
    safe_totals = library_size.replace(0, np.nan)
    cpm = counts.div(safe_totals, axis=1) * 1e6
    log_cpm = np.log2(cpm + 1.0)

    # samples x genes, label rows by compound, mean within compound,
    # then flip back to genes x compounds.
    sample_genes = log_cpm.T
    sample_genes.index = pd.Index(sample_to_compound.to_numpy(), name=compound_col)
    per_compound_wide = sample_genes.groupby(level=compound_col, sort=True).mean()

    long = (
        per_compound_wide.T.rename_axis(index=gene_col, columns=compound_col)
        .reset_index()
        .melt(
            id_vars=gene_col,
            var_name=compound_col,
            value_name=EXPRESSION_COL,
        )
    )
    long = long.dropna(subset=[EXPRESSION_COL]).reset_index(drop=True)
    logger.info(
        "Aggregated {} samples into {} compound x {} gene expression rows",
        len(shared_samples),
        long[compound_col].nunique(),
        long[gene_col].nunique(),
    )
    return long[[compound_col, gene_col, EXPRESSION_COL]]


DEFAULT_MIN_MEAN_CPM = 1.0


def build_gene_filter(
    counts: object,
    *,
    min_mean_cpm: float = DEFAULT_MIN_MEAN_CPM,
    gene_col: str = GENE_COL,
) -> list[str]:
    """Pick the scored gene set from training counts using a CPM cutoff.

    A gene is retained iff its mean CPM across the sample columns of
    ``counts`` is at least ``min_mean_cpm`` (default 1.0). CPM is
    computed per sample as ``1e6 * count_g / Σ_g count_g`` so the
    threshold is library-size-normalized.

    Why this cutoff:

    - ``mean CPM >= 1`` is the standard tag-seq gene-expression
      floor (edgeR's ``filterByExpr``, scanpy's ``filter_genes``
      defaults) — it excludes the noise floor without dropping
      signal-bearing low-expression genes.
    - The previous DESeq2-era pipeline used ``baseMean >= 10``,
      roughly ``mean CPM >= 100`` for typical DRUG-seq library
      sizes. That was more restrictive than necessary for the new
      task; wMSE and the variance-weighting already down-weight
      low-expression genes naturally.
    - If real data argues otherwise, override ``min_mean_cpm`` or
      pre-filter ``counts`` to a different sample set before calling
      this function.

    Parameters
    ----------
    counts
        Wide-format gene-by-sample UMI counts as returned by
        :func:`vcpi.load_experiment`. Must have one column named
        ``gene_col`` (the row labels) plus one column per sample.
        Accepts pandas or polars (anything with ``.to_pandas()``).
    min_mean_cpm
        CPM threshold; genes with strictly lower mean CPM are dropped.
        Default ``1.0``.
    gene_col
        Name of the gene-id column in ``counts`` (default
        ``"gene_id"``).

    Returns
    -------
    list[str]
        Sorted list of ``gene_id`` strings that pass the filter,
        deduplicated.

    Notes
    -----
    The caller is responsible for pre-filtering ``counts`` to the
    sample set the gene filter should be derived from. Typical
    contest usage is to first restrict to the contest condition
    (``compound_concentration == 10 uM``, ``cell_line == "THP-1"``,
    ``timepoint == "24h"``) so the gene filter reflects what's
    measurable in that exact assay rather than across all conditions.
    """
    counts_df = _to_pandas(counts, name="counts")
    if gene_col not in counts_df.columns:
        msg = f"counts is missing the `{gene_col}` column"
        raise ValueError(msg)
    indexed = counts_df.set_index(gene_col)
    if indexed.shape[1] == 0:
        msg = "counts must have at least one sample column besides gene_col"
        raise ValueError(msg)

    library_size = indexed.sum(axis=0)
    n_empty = int((library_size == 0).sum())
    if n_empty:
        logger.warning(
            "{} sample(s) have zero total counts and will not contribute to mean CPM",
            n_empty,
        )
    safe_totals = library_size.replace(0, np.nan)
    cpm = indexed.div(safe_totals, axis=1) * 1e6
    mean_cpm = cpm.mean(axis=1, skipna=True)

    keep = (mean_cpm >= min_mean_cpm).fillna(value=False)
    kept = sorted({str(g) for g in mean_cpm.index[keep]})
    logger.info(
        "Gene filter: {} / {} genes pass mean CPM >= {}",
        len(kept),
        indexed.shape[0],
        min_mean_cpm,
    )
    return kept
