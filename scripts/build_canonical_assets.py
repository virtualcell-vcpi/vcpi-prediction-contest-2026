r"""Build the canonical contest scoring assets from the combined release.

For the combined VCPI training release (t6391 + t6667), this script:

1. Rebuilds the canonical ``gene_filter.csv`` via ``build_gene_filter``
   (``mean CPM >= 1.0``) on the contest-condition training samples,
   excluding bundled held-out test compounds. The bundled wheel-shipped
   gene filter therefore reflects the actual combined training corpus
   rather than just T6667.
2. Computes the per-(gene, training_compound) Mejia weight matrix on
   the same filtered samples / kept genes, **keyed on
   ``user_compound_id``** (vcpi-client's canonical compound identifier).
   The combined train CSV's chemistry-name field is not unique to a
   vcpi compound — at least two names map to multiple distinct
   ``user_compound_id`` values, one of which is an outright canon
   labeling error (``Benztropine mesylate`` is registered against two
   chemically unrelated molecules). Joining
   ``(sequenced_id -> user_compound_id)`` from ``vcpi.metadata`` and
   using that as the compound axis avoids silently averaging samples
   from distinct molecules into one weight column.

Both artifacts are written to ``--out-dir`` so the maintainer can
inspect them, scp them off the design instance, and decide separately
how to distribute ``weights.parquet`` (likely too big to bundle in the
wheel — current canonical recipe is ``float16`` + brotli L6, ~365 MB).

Usage::

    TVC_TOKEN=...  uv run python scripts/build_canonical_assets.py \
        --train-counts /mnt/efs/P1016/vcpi_combined/contest_release/v1_2026-05-19/public/train/train_gene_counts.parquet \
        --train-metadata /mnt/efs/P1016/vcpi_combined/contest_release/v1_2026-05-19/public/train/train_metadata.csv \
        --out-dir ~/scratch/vcpi-bench/_artifacts/

The default paths point to the canonical combined release on the
``rkirchner-design`` instance. A working vcpi-client install + valid
``TVC_TOKEN`` are required so we can fetch the
``(sequenced_id -> user_compound_id)`` mapping from
``vcpi.metadata``.
"""

from __future__ import annotations

import argparse
import resource
import sys
import time
from pathlib import Path

import pandas as pd
import vcpi
from loguru import logger

from vcpi_prediction_contest import (
    build_gene_filter,
    compute_mejia_weights,
    load_test_compounds,
)
from vcpi_prediction_contest.metrics import GENE_COL

DEFAULT_COUNTS = Path(
    "/mnt/efs/P1016/vcpi_combined/contest_release/v1_2026-05-19/public/train/train_gene_counts.parquet"
)
DEFAULT_METADATA = Path(
    "/mnt/efs/P1016/vcpi_combined/contest_release/v1_2026-05-19/public/train/train_metadata.csv"
)
DEFAULT_OUT = Path("~/scratch/vcpi-bench/_artifacts").expanduser()
_LIBRARY_CONCENTRATION_UM = 10
DEFAULT_MIN_MEAN_CPM = 1.0
_UCID_COL = "user_compound_id"


def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _fetch_seq_to_ucid() -> pd.Series:
    """Query vcpi.metadata for the canonical ``sequenced_id -> user_compound_id`` mapping.

    Pulls every contest-condition sample across the three training
    jobs (THP-1, 24 h, library at 10 µM = 10000 nM, plus DMSO
    controls). Returns a Series indexed by ``sequenced_id`` (string)
    whose values are ``user_compound_id`` strings — vcpi-client's
    canonical compound identifier and the join key contestants get
    out of ``metadata.user_compound_id``.
    """
    logger.info("Querying vcpi.metadata for sequenced_id -> user_compound_id ...")
    df = vcpi.query(
        sql=(
            "SELECT sequenced_id, user_compound_id FROM metadata "
            "WHERE cell_line = 'THP-1' AND timepoint = '24h' "
            "AND ((compound_concentration = 10000 AND compound_concentration_unit = 'nM') "
            "OR user_compound_id = 'DMSO')"
        )
    ).to_pandas()
    df["sequenced_id"] = df["sequenced_id"].astype(str)
    df["user_compound_id"] = df["user_compound_id"].astype(str)
    if df["sequenced_id"].duplicated().any():
        n_dupes = int(df["sequenced_id"].duplicated().sum())
        msg = f"vcpi.metadata returned {n_dupes} duplicate sequenced_id rows; cannot build mapping"
        raise RuntimeError(msg)
    mapping = df.set_index("sequenced_id")["user_compound_id"]
    logger.info(
        "Fetched vcpi mapping: {} samples covering {} unique user_compound_ids",
        len(mapping),
        mapping.nunique(),
    )
    return mapping


def _filter_to_contest_condition(metadata: pd.DataFrame) -> pd.DataFrame:
    """Match the bundled-gene-filter sample scope on the combined release.

    Combined metadata is already restricted to THP-1 / 24h, so we only
    need to enforce the dose / control distinction: library at the
    contest concentration (10 uM) OR any "Ginkgo ..." control (DMSO +
    pos controls). Other doses on the library compounds (3/1/0.3/0.03
    uM) are excluded — they're useful training context but not the
    sample set the contest scores on.
    """
    library_at_10um = (
        (metadata["sample_type"] == "library")
        & (metadata["compound_concentration"] == _LIBRARY_CONCENTRATION_UM)
        & (metadata["compound_concentration_unit"] == "uM")
    )
    controls = metadata["sample_type"].str.startswith("Ginkgo")
    return metadata[library_at_10um | controls].copy().reset_index(drop=True)


def _restrict_counts_to_metadata(
    counts: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Slice the wide counts frame down to the sample columns we'll use."""
    keep_cols = [GENE_COL, *(s for s in metadata["sequenced_id"] if s in counts.columns)]
    missing = set(metadata["sequenced_id"]) - set(counts.columns)
    if missing:
        logger.warning(
            "{} contest-condition sample IDs are absent from the counts frame (first 5: {})",
            len(missing),
            list(missing)[:5],
        )
    return counts[keep_cols]


def _build_and_write_gene_filter(
    counts: pd.DataFrame,
    *,
    min_mean_cpm: float,
    out_dir: Path,
) -> list[str]:
    t_gf = time.perf_counter()
    logger.info("Running build_gene_filter(min_mean_cpm={}) ...", min_mean_cpm)
    kept_genes = build_gene_filter(counts, min_mean_cpm=min_mean_cpm)
    logger.info(
        "gene_filter: {} / {} genes kept ({:.1f}s, peak RSS={:.1f} MB)",
        len(kept_genes),
        counts.shape[0],
        time.perf_counter() - t_gf,
        _peak_rss_mb(),
    )
    gene_filter_path = out_dir / "gene_filter.csv"
    pd.DataFrame({GENE_COL: kept_genes}).to_csv(gene_filter_path, index=False)
    logger.info("Wrote {} ({} bytes)", gene_filter_path, gene_filter_path.stat().st_size)
    return kept_genes


def _write_weights_f16_brotli(weights: pd.DataFrame, out_dir: Path) -> Path:
    """Write the canonical shipping recipe: ``float16`` + brotli L6."""
    weights_f16 = weights.astype("float16")
    weights_path = out_dir / "weights.parquet"
    weights_f16.to_parquet(weights_path, compression="brotli", compression_level=6)
    logger.info(
        "Wrote {} ({:.1f} MB, dtype=float16, brotli L6)",
        weights_path,
        weights_path.stat().st_size / 1e6,
    )
    return weights_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-counts", type=Path, default=DEFAULT_COUNTS)
    parser.add_argument("--train-metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--min-mean-cpm",
        type=float,
        default=DEFAULT_MIN_MEAN_CPM,
        help="CPM cutoff for build_gene_filter (default 1.0).",
    )
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    logger.info("Loading metadata from {}", args.train_metadata)
    metadata = pd.read_csv(args.train_metadata, low_memory=False)
    metadata["sequenced_id"] = metadata["sequenced_id"].astype(str)
    logger.info("Metadata: {} rows", len(metadata))

    metadata = _filter_to_contest_condition(metadata)
    logger.info(
        "After contest-condition filter (10 uM library + Ginkgo controls): {} rows, {} unique chemistry names",
        len(metadata),
        metadata["compound"].nunique(),
    )

    # ------------------------------------------------------------------
    # Join vcpi's canonical (sequenced_id -> user_compound_id) onto the
    # combined-CSV samples. Chemistry name is NOT unique to a vcpi
    # compound; user_compound_id is. Samples without a matching vcpi
    # ucid are dropped (they're not in the released vcpi product, so
    # contestants can't pull them anyway).
    # ------------------------------------------------------------------
    seq_to_ucid = _fetch_seq_to_ucid()
    metadata[_UCID_COL] = metadata["sequenced_id"].map(seq_to_ucid)
    n_missing_ucid = int(metadata[_UCID_COL].isna().sum())
    if n_missing_ucid:
        sample_missing = metadata.loc[metadata[_UCID_COL].isna(), "compound"].unique()[:5].tolist()
        logger.warning(
            "Dropping {} samples not in vcpi.metadata at contest condition "
            "(unique compound names: {}, first 5: {})",
            n_missing_ucid,
            int(
                metadata.loc[metadata[_UCID_COL].isna(), "compound"].nunique(),
            ),
            sample_missing,
        )
        metadata = metadata.dropna(subset=[_UCID_COL]).copy()
    logger.info(
        "After vcpi ucid join: {} samples, {} unique user_compound_ids",
        len(metadata),
        metadata[_UCID_COL].nunique(),
    )

    # ``test_compounds.csv`` is keyed on user_compound_id (``compound``
    # column). Now that training metadata is also keyed on ucid, we
    # can exclude by direct intersection.
    test = load_test_compounds()
    test_ucids = set(test["compound"].astype(str))
    pre_exclude = len(metadata)
    metadata = metadata[~metadata[_UCID_COL].astype(str).isin(test_ucids)].copy()
    logger.info(
        "Excluded {} samples whose user_compound_id is in bundled test_compounds.csv "
        "({} test compounds total)",
        pre_exclude - len(metadata),
        len(test_ucids),
    )
    logger.info(
        "Final training samples: {} rows, {} unique user_compound_ids",
        len(metadata),
        metadata[_UCID_COL].nunique(),
    )

    logger.info("Loading wide counts from {}", args.train_counts)
    counts = pd.read_parquet(args.train_counts)
    logger.info(
        "Counts loaded: shape={}  memory={:.1f} MB  peak RSS so far={:.1f} MB",
        counts.shape,
        counts.memory_usage(deep=True).sum() / 1e6,
        _peak_rss_mb(),
    )

    counts = _restrict_counts_to_metadata(counts, metadata)
    logger.info(
        "Counts restricted to contest-condition samples: shape={}  peak RSS={:.1f} MB",
        counts.shape,
        _peak_rss_mb(),
    )

    # ------------------------------------------------------------------
    # Step 1: rebuild the bundled gene filter from the combined corpus
    # ------------------------------------------------------------------
    kept_genes = _build_and_write_gene_filter(
        counts, min_mean_cpm=args.min_mean_cpm, out_dir=args.out_dir
    )
    gene_filter_path = args.out_dir / "gene_filter.csv"

    kept_set = set(kept_genes)
    counts_filtered = counts[counts[GENE_COL].astype(str).isin(kept_set)].reset_index(drop=True)
    logger.info(
        "Filtered counts: shape={}  peak RSS={:.1f} MB",
        counts_filtered.shape,
        _peak_rss_mb(),
    )

    # ------------------------------------------------------------------
    # Step 2: compute Mejia weights keyed on user_compound_id directly
    # ------------------------------------------------------------------
    t_w = time.perf_counter()
    logger.info("Running compute_mejia_weights (keyed on user_compound_id) ...")
    weights = compute_mejia_weights(counts_filtered, metadata, compound_col=_UCID_COL)
    weights.columns = pd.Index([str(c) for c in weights.columns], name="compound")
    logger.info(
        "weights shape={}  columns dtype={}  ({:.1f}s, peak RSS={:.1f} MB)",
        weights.shape,
        weights.columns.dtype,
        time.perf_counter() - t_w,
        _peak_rss_mb(),
    )

    # ------------------------------------------------------------------
    # Step 3: write weights as float16 + brotli L6 (canonical recipe)
    # ------------------------------------------------------------------
    weights_path = _write_weights_f16_brotli(weights, args.out_dir)

    logger.info(
        "=== Done in {:.1f}s, peak RSS={:.1f} MB ===",
        time.perf_counter() - t_start,
        _peak_rss_mb(),
    )
    logger.info("Outputs:")
    for p in (gene_filter_path, weights_path):
        logger.info("  {}  ({} bytes)", p, p.stat().st_size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
