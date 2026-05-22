r"""Run the baseline audit against the real T6667 DRUG-seq dataset.

The synthetic audit (`baseline_audit.py`) verifies the metric's
mathematical calibration. This script repeats the same battery of
baselines against the **real** T6667 release so we can see:

- The actual magnitude of wMSE for real predictors on the contest
  data.
- Whether the Mejia and pooled-variance weight schemes actually
  diverge on real, biologically heterogeneous variances — the
  synthetic audit suggested they'd be nearly identical at n=2,
  but the synthetic noise was too clean to be a fair test.
- The scaling-attack sweep across orders of magnitude.

Inputs (default paths can be overridden via CLI):

- ``--counts-h5ad``: T6667 h5ad with a ``layers["counts"]`` int64 matrix
  (shape ``(n_samples, n_genes)``), obs index = ``sequenced_id``,
  var index = gene_id. The obs frame in the h5ad has a unicode
  decoding bug for some compound names, so we ignore it and load
  metadata from CSV instead.
- ``--metadata-csv``: CSV with one row per ``sequenced_id`` carrying
  the columns ``compound``, ``compound_concentration``,
  ``compound_concentration_unit``, ``cell_line``, ``timepoint``,
  ``sample_type``, ``is_neg_control``, ``is_pos_control``.

Usage::

    uv run python scripts/baseline_audit_real.py \\
        --counts-h5ad /Users/rkirchner/Projects/dragseq-vcpi-explore/data/raw/6667.h5ad \\
        --metadata-csv /Users/rkirchner/Projects/dragseq-vcpi-explore/data/raw/metadata.csv

The default paths point to the locally cached T6667 release so you
can just ``uv run python scripts/baseline_audit_real.py``.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import h5py
import numpy as np
import pandas as pd
from loguru import logger

from vcpi_prediction_contest.baselines import (
    predict_constant,
    predict_mu_all_train,
    predict_mu_control,
    predict_random_gaussian,
    predict_scaled_perfect,
    predict_shuffle_compounds,
    predict_technical_duplicate,
)
from vcpi_prediction_contest.expression import build_gene_filter, counts_to_expression
from vcpi_prediction_contest.metrics import (
    COMPOUND_COL,
    EXPRESSION_COL,
    GENE_COL,
    score_compounds,
)
from vcpi_prediction_contest.weights import (
    compute_mejia_weights,
    compute_pooled_weights,
)

DEFAULT_COUNTS_H5AD = "/Users/rkirchner/Projects/dragseq-vcpi-explore/data/raw/6667.h5ad"
DEFAULT_METADATA_CSV = "/Users/rkirchner/Projects/dragseq-vcpi-explore/data/raw/metadata.csv"

# Default contest condition: 10 uM, THP-1, 24h. Controls are kept in.
DEFAULT_TEST_FRACTION = 0.2
DEFAULT_SEED = 0
DEFAULT_MIN_MEAN_CPM = 1.0
_LIBRARY_CONCENTRATION_UM = 10


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@dataclass
class RealRelease:
    train_counts: pd.DataFrame  # (gene_id, sample columns) wide
    train_metadata: pd.DataFrame  # sequenced_id, compound
    train_truth: pd.DataFrame  # long: compound, gene_id, expression
    test_counts: pd.DataFrame
    test_metadata: pd.DataFrame
    test_truth: pd.DataFrame
    control_compounds: list[str]
    gene_index: list[str]


def _load_counts_and_var(h5ad_path: str) -> tuple[np.ndarray, list[str], list[str]]:
    """Pull ``layers['counts']`` and var/obs ID arrays straight from h5py.

    Skips anndata's higher-level reader because the obs frame in the
    T6667 release has unicode characters that the default codec
    chokes on.
    """
    with h5py.File(h5ad_path, "r") as f:
        counts = f["layers"]["counts"][...]
        var_ids = [b.decode("utf-8") for b in f["var"]["_index"][...]]
        obs_ids = [b.decode("utf-8") for b in f["obs"]["_index"][...]]
    logger.info(
        "Loaded counts layer: shape={} dtype={} ({} obs ids, {} var ids)",
        counts.shape,
        counts.dtype,
        len(obs_ids),
        len(var_ids),
    )
    return counts, var_ids, obs_ids


def _filter_metadata_to_contest(metadata: pd.DataFrame) -> pd.DataFrame:
    """Keep only the contest condition: 10 uM library + DMSO + named pos controls.

    Concretely: cell_line == THP-1, timepoint == 24, AND
    (sample_type == 'library' AND compound_concentration == 10 uM)
    OR sample_type starts with 'Ginkgo' (i.e. neg / pos controls).
    """
    base = (metadata["cell_line"] == "THP-1") & (metadata["timepoint"].astype(str) == "24")
    library_at_10um = (
        (metadata["sample_type"] == "library")
        & (metadata["compound_concentration"] == _LIBRARY_CONCENTRATION_UM)
        & (metadata["compound_concentration_unit"] == "uM")
    )
    controls = metadata["sample_type"].str.startswith("Ginkgo")
    return metadata[base & (library_at_10um | controls)].copy().reset_index(drop=True)


def load_real_release(
    counts_h5ad: str,
    metadata_csv: str,
    *,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    min_mean_cpm: float = DEFAULT_MIN_MEAN_CPM,
    seed: int = DEFAULT_SEED,
) -> RealRelease:
    """Build a RealRelease ready for the audit harness."""
    logger.info("Loading h5ad counts from {}", counts_h5ad)
    counts_arr, var_ids, obs_ids = _load_counts_and_var(counts_h5ad)

    logger.info("Loading metadata from {}", metadata_csv)
    metadata = pd.read_csv(metadata_csv, low_memory=False)
    metadata["sequenced_id"] = metadata["sequenced_id"].astype(str)
    contest = _filter_metadata_to_contest(metadata)
    logger.info(
        "Filtered metadata to contest condition: {}/{} samples kept, {} unique compounds",
        len(contest),
        len(metadata),
        contest["compound"].nunique(),
    )

    # Subset counts to the contest samples.
    obs_id_to_pos = {o: i for i, o in enumerate(obs_ids)}
    contest_positions = np.array(
        [obs_id_to_pos[s] for s in contest["sequenced_id"] if s in obs_id_to_pos],
        dtype=np.int64,
    )
    if len(contest_positions) != len(contest):
        missing = set(contest["sequenced_id"]) - set(obs_ids)
        msg = f"{len(contest) - len(contest_positions)} contest samples missing from h5ad obs (e.g. {list(missing)[:5]})"
        raise RuntimeError(msg)
    counts_subset = counts_arr[contest_positions, :]  # (n_samples, n_genes)

    # Build the wide-format counts frame the package expects:
    # rows = genes, columns = sample ids, plus a leading 'gene_id' column.
    sample_ids = contest["sequenced_id"].tolist()
    wide = pd.DataFrame(counts_subset.T, index=var_ids, columns=sample_ids)
    wide.index.name = GENE_COL

    # Identify control compounds (DMSO + named positive controls) BEFORE the split,
    # so we can keep all controls on the train side.
    control_mask = contest["sample_type"].str.startswith("Ginkgo")
    control_compounds = sorted(contest.loc[control_mask, "compound"].unique().tolist())
    library_compounds = sorted(contest.loc[~control_mask, "compound"].unique().tolist())
    logger.info(
        "Library compounds: {} | Control compounds: {} ({})",
        len(library_compounds),
        len(control_compounds),
        control_compounds,
    )

    # Random test split over library compounds only; controls always stay in train.
    rng = np.random.default_rng(seed)
    n_test = max(1, round(test_fraction * len(library_compounds)))
    test_idx = rng.choice(len(library_compounds), size=n_test, replace=False)
    test_compounds = sorted({library_compounds[i] for i in test_idx})
    train_compounds = sorted(set(library_compounds) - set(test_compounds) | set(control_compounds))
    logger.info(
        "Train compounds: {} (including {} controls) | Test compounds: {}",
        len(train_compounds),
        len(control_compounds),
        len(test_compounds),
    )

    train_samples = contest[contest["compound"].isin(train_compounds)].copy().reset_index(drop=True)
    test_samples = contest[contest["compound"].isin(test_compounds)].copy().reset_index(drop=True)
    logger.info(
        "Train samples: {} | Test samples: {}",
        len(train_samples),
        len(test_samples),
    )

    train_counts_wide = wide[train_samples["sequenced_id"].tolist()].copy()
    test_counts_wide = wide[test_samples["sequenced_id"].tolist()].copy()
    train_counts_wide.reset_index(inplace=True)  # noqa: PD002 — package expects gene_id as a column
    test_counts_wide.reset_index(inplace=True)  # noqa: PD002

    # Gene filter from train counts only (no test leakage).
    logger.info("Building gene filter at min_mean_cpm={}", min_mean_cpm)
    kept_genes = build_gene_filter(train_counts_wide, min_mean_cpm=min_mean_cpm)
    logger.info("Kept {}/{} genes after CPM filter", len(kept_genes), len(var_ids))

    train_counts_wide = train_counts_wide[train_counts_wide[GENE_COL].isin(kept_genes)]
    test_counts_wide = test_counts_wide[test_counts_wide[GENE_COL].isin(kept_genes)]

    train_metadata = train_samples[["sequenced_id", "compound"]].reset_index(drop=True)
    test_metadata = test_samples[["sequenced_id", "compound"]].reset_index(drop=True)

    logger.info("Aggregating train counts -> expression ({} samples)", len(train_metadata))
    train_truth = counts_to_expression(train_counts_wide, train_metadata)
    logger.info("Aggregating test counts -> expression ({} samples)", len(test_metadata))
    test_truth = counts_to_expression(test_counts_wide, test_metadata)

    return RealRelease(
        train_counts=train_counts_wide,
        train_metadata=train_metadata,
        train_truth=train_truth,
        test_counts=test_counts_wide,
        test_metadata=test_metadata,
        test_truth=test_truth,
        control_compounds=control_compounds,
        gene_index=sorted(kept_genes),
    )


# ---------------------------------------------------------------------------
# Audit harness (mirrors scripts/baseline_audit.py)
# ---------------------------------------------------------------------------


def _aggregate(per_compound: pd.DataFrame) -> dict[str, float]:
    return {
        "wmse_mean": float(per_compound["wmse"].mean(skipna=True)),
        "wmse_median": float(per_compound["wmse"].median(skipna=True)),
    }


def run_audit(release: RealRelease) -> pd.DataFrame:
    test_compounds = sorted(release.test_truth[COMPOUND_COL].unique())
    train_truth_wide = release.train_truth.pivot_table(
        index=GENE_COL, columns=COMPOUND_COL, values=EXPRESSION_COL, aggfunc="mean"
    )
    gene_index = train_truth_wide.index.tolist()

    logger.info(
        "Computing Mejia weights on train counts ({} compounds) ...", train_truth_wide.shape[1]
    )
    mejia_w_train = compute_mejia_weights(release.train_counts, release.train_metadata)
    logger.info("Computing pooled-variance weights on train counts ...")
    pooled_w_train = compute_pooled_weights(release.train_counts, release.train_metadata)

    logger.info("Concatenating train+test counts for per-test-compound weights ...")
    combined_counts = pd.concat(
        [
            release.train_counts.set_index(GENE_COL),
            release.test_counts.set_index(GENE_COL),
        ],
        axis=1,
    ).reset_index()
    combined_metadata = pd.concat([release.train_metadata, release.test_metadata], axis=0)
    logger.info("Computing Mejia weights on combined train+test counts ...")
    mejia_w_combined = compute_mejia_weights(combined_counts, combined_metadata)
    logger.info("Computing pooled-variance weights on combined train+test counts ...")
    pooled_w_combined = compute_pooled_weights(combined_counts, combined_metadata)

    def _broadcast_per_gene_to_test(weights_train: pd.DataFrame) -> pd.DataFrame:
        per_gene = weights_train.reindex(gene_index).mean(axis=1)
        per_gene = per_gene / per_gene.sum()
        return pd.DataFrame(
            np.broadcast_to(
                per_gene.to_numpy()[:, None], (len(gene_index), len(test_compounds))
            ).copy(),
            index=gene_index,
            columns=test_compounds,
        )

    def _restrict_to_test_compounds(weights_combined: pd.DataFrame) -> pd.DataFrame:
        sub = weights_combined.reindex(index=gene_index, columns=test_compounds)
        if sub.isna().any().any():
            msg = "combined weights missing test compounds — check metadata alignment"
            raise RuntimeError(msg)
        return sub

    weight_variants = {
        ("mejia", "train_broadcast"): _broadcast_per_gene_to_test(mejia_w_train),
        ("pooled", "train_broadcast"): _broadcast_per_gene_to_test(pooled_w_train),
        ("mejia", "per_test_compound"): _restrict_to_test_compounds(mejia_w_combined),
        ("pooled", "per_test_compound"): _restrict_to_test_compounds(pooled_w_combined),
    }

    scaling_attack = {
        f"scaled_perfect ({c}x)": predict_scaled_perfect(release.test_truth, scale=c)
        for c in (0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0, 1000.0)
    }

    baselines = {
        "perfect (technical_duplicate)": predict_technical_duplicate(release.test_truth),
        "noisy_duplicate (sd=0.5)": predict_technical_duplicate(
            release.test_truth, noise_scale=0.5, seed=1
        ),
        "mu_all_train (mode collapse)": predict_mu_all_train(release.train_truth, test_compounds),
        "mu_control (DMSO mean)": predict_mu_control(
            release.train_truth,
            test_compounds,
            [c for c in release.control_compounds if c == "DMSO"],
        ),
        **scaling_attack,
        "shuffle_compounds": predict_shuffle_compounds(release.test_truth, seed=2),
        "predict_zero": predict_constant(test_compounds, gene_index, value=0.0),
        "predict_constant_5": predict_constant(test_compounds, gene_index, value=5.0),
        "random_gaussian": predict_random_gaussian(
            test_compounds, gene_index, mean=5.0, sd=2.0, seed=3
        ),
    }

    records: list[dict[str, object]] = []
    for (scheme, scope), w_mat in weight_variants.items():
        logger.info(
            "Scoring {} baselines under {} weights ({} scope) ...",
            len(baselines),
            scheme,
            scope,
        )
        for name, pred in baselines.items():
            try:
                per_compound = score_compounds(release.test_truth, pred, weights=w_mat)
                agg = _aggregate(per_compound)
                records.append({"baseline": name, "weights": scheme, "scope": scope, **agg})
            except Exception as exc:  # noqa: BLE001 — audit script catches all and continues
                logger.error("Scoring failed for {} under {}/{}: {}", name, scheme, scope, exc)
                records.append(
                    {"baseline": name, "weights": scheme, "scope": scope, "error": str(exc)}
                )
    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


_BASELINE_ORDER = [
    "perfect (technical_duplicate)",
    "noisy_duplicate (sd=0.5)",
    "mu_all_train (mode collapse)",
    "mu_control (DMSO mean)",
    "scaled_perfect (0.001x)",
    "scaled_perfect (0.01x)",
    "scaled_perfect (0.1x)",
    "scaled_perfect (0.5x)",
    "scaled_perfect (1.0x)",
    "scaled_perfect (2.0x)",
    "scaled_perfect (10.0x)",
    "scaled_perfect (100.0x)",
    "scaled_perfect (1000.0x)",
    "shuffle_compounds",
    "predict_zero",
    "predict_constant_5",
    "random_gaussian",
]


def format_table(audit: pd.DataFrame, *, scope: str) -> str:
    sub = audit[audit["scope"] == scope]
    pivoted = sub.pivot_table(
        index="baseline",
        columns="weights",
        values=["wmse_mean", "wmse_median"],
        aggfunc="first",
    )
    metric_order = ["wmse_mean", "wmse_median"]
    weight_order = ["mejia", "pooled"]
    pivoted = pivoted.reindex(columns=pd.MultiIndex.from_product([metric_order, weight_order]))
    pivoted = pivoted.reindex(index=[b for b in _BASELINE_ORDER if b in pivoted.index])
    return pivoted.to_string(float_format=lambda x: f"{x:9.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--counts-h5ad", default=DEFAULT_COUNTS_H5AD)
    p.add_argument("--metadata-csv", default=DEFAULT_METADATA_CSV)
    p.add_argument("--test-fraction", type=float, default=DEFAULT_TEST_FRACTION)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--min-mean-cpm", type=float, default=DEFAULT_MIN_MEAN_CPM)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    release = load_real_release(
        args.counts_h5ad,
        args.metadata_csv,
        test_fraction=args.test_fraction,
        seed=args.seed,
        min_mean_cpm=args.min_mean_cpm,
    )
    audit = run_audit(release)

    for scope, label in (
        ("train_broadcast", "TRAIN-ONLY WEIGHTS, PER-GENE-MEAN BROADCAST"),
        ("per_test_compound", "PER-(TEST COMPOUND) WEIGHTS (faithful Mejia setup)"),
    ):
        print()
        print("=" * 110)
        print(f"REAL T6667 BASELINE AUDIT — {label}")
        print("=" * 110)
        print(format_table(audit, scope=scope))
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
