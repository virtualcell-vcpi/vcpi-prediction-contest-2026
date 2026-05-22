r"""Build the bundled ``gene_filter.csv`` from the T6667 release.

This is a one-shot data-prep script: it derives the canonical
gene_filter the leaderboard will score on from the T6667 raw counts +
metadata, applying the contest-condition filter and excluding any
sample whose compound appears in the held-out ``test_compounds.csv``.
The resulting gene list is written to
``src/vcpi_prediction_contest/data_files/gene_filter.csv`` so it ships
inside the wheel alongside ``test_compounds.csv``.

Usage::

    uv run python scripts/build_bundled_gene_filter.py \\
        --counts-h5ad /Users/rkirchner/Projects/dragseq-vcpi-explore/data/raw/6667.h5ad \\
        --metadata-csv /Users/rkirchner/Projects/dragseq-vcpi-explore/data/raw/metadata.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from loguru import logger

from vcpi_prediction_contest import load_test_compounds
from vcpi_prediction_contest.expression import build_gene_filter
from vcpi_prediction_contest.metrics import GENE_COL

DEFAULT_COUNTS_H5AD = "/Users/rkirchner/Projects/dragseq-vcpi-explore/data/raw/6667.h5ad"
DEFAULT_METADATA_CSV = "/Users/rkirchner/Projects/dragseq-vcpi-explore/data/raw/metadata.csv"
DEFAULT_OUT = (
    Path(__file__).resolve().parents[1] / "src/vcpi_prediction_contest/data_files/gene_filter.csv"
)

_LIBRARY_CONCENTRATION_UM = 10
DEFAULT_MIN_MEAN_CPM = 1.0


def _load_counts_and_var(h5ad_path: str) -> tuple[np.ndarray, list[str], list[str]]:
    """Pull ``layers['counts']`` and var/obs ID arrays straight from h5py."""
    with h5py.File(h5ad_path, "r") as f:
        counts = f["layers"]["counts"][...]
        var_ids = [b.decode("utf-8") for b in f["var"]["_index"][...]]
        obs_ids = [b.decode("utf-8") for b in f["obs"]["_index"][...]]
    return counts, var_ids, obs_ids


def _filter_metadata_to_contest(metadata: pd.DataFrame) -> pd.DataFrame:
    """Keep only the contest condition: 10 uM library + DMSO + named pos controls."""
    base = (metadata["cell_line"] == "THP-1") & (metadata["timepoint"].astype(str) == "24")
    library_at_10um = (
        (metadata["sample_type"] == "library")
        & (metadata["compound_concentration"] == _LIBRARY_CONCENTRATION_UM)
        & (metadata["compound_concentration_unit"] == "uM")
    )
    controls = metadata["sample_type"].str.startswith("Ginkgo")
    return metadata[base & (library_at_10um | controls)].copy().reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts-h5ad", default=DEFAULT_COUNTS_H5AD)
    parser.add_argument("--metadata-csv", default=DEFAULT_METADATA_CSV)
    parser.add_argument(
        "--min-mean-cpm",
        type=float,
        default=DEFAULT_MIN_MEAN_CPM,
        help="Mean CPM cutoff passed to build_gene_filter (default 1.0).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Path to write the gene_filter CSV (default ships into the wheel).",
    )
    args = parser.parse_args(argv)

    logger.info("Loading counts from {}", args.counts_h5ad)
    counts_arr, var_ids, obs_ids = _load_counts_and_var(args.counts_h5ad)
    logger.info("counts shape={}  n_var={}  n_obs={}", counts_arr.shape, len(var_ids), len(obs_ids))

    logger.info("Loading metadata from {}", args.metadata_csv)
    metadata = pd.read_csv(args.metadata_csv, low_memory=False)
    metadata["sequenced_id"] = metadata["sequenced_id"].astype(str)
    contest = _filter_metadata_to_contest(metadata)
    logger.info(
        "Contest-condition samples: {} / {} (unique compounds: {})",
        len(contest),
        len(metadata),
        contest["compound"].nunique(),
    )

    test_compounds = set(load_test_compounds()["compound"].astype(str))
    logger.info("Held-out test compounds (bundled): {}", len(test_compounds))

    train_mask = ~contest["compound"].astype(str).isin(test_compounds)
    train_samples = contest[train_mask].reset_index(drop=True)
    excluded_compounds = sorted(set(contest.loc[~train_mask, "compound"].astype(str)))
    logger.info(
        "Training samples (post-exclusion of test compounds): {} / {} (excluded {} compounds, {} samples)",
        len(train_samples),
        len(contest),
        len(excluded_compounds),
        int((~train_mask).sum()),
    )
    if len(excluded_compounds) < 0.5 * len(test_compounds):
        logger.warning(
            "Only {} of the {} test compounds were found in T6667 metadata; "
            "the bundled test_compounds.csv may have been drawn from a different release.",
            len(excluded_compounds),
            len(test_compounds),
        )

    obs_id_to_pos = {o: i for i, o in enumerate(obs_ids)}
    train_positions = np.array(
        [obs_id_to_pos[s] for s in train_samples["sequenced_id"] if s in obs_id_to_pos],
        dtype=np.int64,
    )
    if len(train_positions) != len(train_samples):
        missing = set(train_samples["sequenced_id"]) - set(obs_ids)
        logger.warning(
            "{} training samples missing from h5ad obs (first 5: {})",
            len(train_samples) - len(train_positions),
            list(missing)[:5],
        )

    train_counts = counts_arr[train_positions, :]
    train_sample_ids = [train_samples["sequenced_id"].iloc[i] for i in range(len(train_positions))]
    wide = pd.DataFrame(train_counts.T, index=var_ids, columns=train_sample_ids)
    wide.index.name = GENE_COL
    wide = wide.reset_index()  # build_gene_filter wants gene_id as a column

    logger.info("Computing gene_filter at min_mean_cpm={}", args.min_mean_cpm)
    kept_genes = build_gene_filter(wide, min_mean_cpm=args.min_mean_cpm)
    logger.info("Kept {} / {} genes", len(kept_genes), len(var_ids))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({GENE_COL: kept_genes}).to_csv(args.out, index=False)
    logger.info("Wrote {} genes to {}", len(kept_genes), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
