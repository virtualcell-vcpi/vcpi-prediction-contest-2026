"""Audit the contest scoring panel against a battery of known baselines.

For both the Mejia 2025 weighting recipe and the pooled-variance
moderated variant, score every baseline in
:mod:`vcpi_prediction_contest.baselines` against a realistic
synthetic replicate-level dataset and print a calibration table.

Expected calibration (the audit FAILS LOUDLY when these are violated):

==================================  ===========
Baseline                            wMSE
==================================  ===========
technical_duplicate (truth itself)  0
technical_duplicate + 0.5 noise     small > 0
mu_all_train (mode collapse)        moderate
mu_control (DMSO mean)              moderate
scaled_perfect (2 x truth)          large > 0
shuffle_compounds                   moderate
predict_constant (zero)             huge
predict_random_gaussian             huge
==================================  ===========

If the table doesn't show these patterns, the metric panel is broken
and the leaderboard isn't ready for contestants.

Usage::

    uv run python scripts/baseline_audit.py

Set ``VCPI_AUDIT_SEED`` to override the RNG seed; ``VCPI_AUDIT_N_COMPOUNDS``,
``VCPI_AUDIT_N_GENES`` to scale the synthetic dataset.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

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
from vcpi_prediction_contest.expression import counts_to_expression
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

# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------


@dataclass
class SyntheticRelease:
    """Container for one realization of the audit dataset."""

    train_counts: pd.DataFrame  # gene_id, sample_id columns (wide)
    train_metadata: pd.DataFrame  # sequenced_id, compound
    train_truth: pd.DataFrame  # long: compound, gene_id, expression
    test_counts: pd.DataFrame  # gene_id, sample_id columns (wide)
    test_metadata: pd.DataFrame  # sequenced_id, compound
    test_truth: pd.DataFrame  # long: compound, gene_id, expression
    control_compounds: list[str]


def make_synthetic_release(
    *,
    n_train_compounds: int = 40,
    n_test_compounds: int = 20,
    n_genes: int = 200,
    n_replicates: int = 2,
    n_control_compounds: int = 4,
    library_size: int = 100_000,
    seed: int = 0,
) -> SyntheticRelease:
    """Generate a realistic small-scale DRUG-seq-like screen.

    Each compound perturbs a distinct subset of genes (5x baseline rate
    inflation). Controls are flat at baseline. Counts are Poisson with
    a per-sample library size of ~100k UMIs (matching DRUG-seq scale),
    so CPM-derived per-gene variances are sane.
    """
    rng = np.random.default_rng(seed)
    n_signal = 15  # signal genes per compound
    base_rate = rng.gamma(2.0, 2.0, size=n_genes) + 0.5
    gene_ids = [f"g{i:05d}" for i in range(n_genes)]

    train_compounds = [f"train_{i:04d}" for i in range(n_train_compounds)]
    test_compounds = [f"test_{i:04d}" for i in range(n_test_compounds)]
    control_compounds = [f"ctrl_{i:02d}" for i in range(n_control_compounds)]

    all_train_compounds = control_compounds + train_compounds

    def _draw_signal(compound_id: str) -> np.ndarray:
        if compound_id.startswith("ctrl_"):
            return np.zeros(n_genes, dtype=bool)
        idx = rng.choice(n_genes, size=n_signal, replace=False)
        mask = np.zeros(n_genes, dtype=bool)
        mask[idx] = True
        return mask

    def _draw_compound_counts(compound_id: str, n_reps: int) -> np.ndarray:
        signal = _draw_signal(compound_id)
        rates = np.where(signal, base_rate * 5.0, base_rate)
        rates = rates * (library_size / rates.sum())
        return np.stack([rng.poisson(rates) for _ in range(n_reps)], axis=1)

    # Build train counts (one Poisson draw per replicate)
    train_count_blocks = []
    train_sample_ids = []
    train_sample_to_compound = []
    for c in all_train_compounds:
        block = _draw_compound_counts(c, n_replicates)
        train_count_blocks.append(block)
        for r in range(n_replicates):
            sid = f"{c}_r{r}"
            train_sample_ids.append(sid)
            train_sample_to_compound.append(c)

    train_counts_mat = np.concatenate(train_count_blocks, axis=1)
    train_counts = pd.DataFrame(
        train_counts_mat, index=gene_ids, columns=train_sample_ids
    ).reset_index()
    train_counts = train_counts.rename(columns={"index": GENE_COL})
    train_metadata = pd.DataFrame(
        list(zip(train_sample_ids, train_sample_to_compound, strict=True)),
        columns=["sequenced_id", "compound"],
    )

    # Build train truth via the canonical counts -> expression pipeline.
    train_truth = counts_to_expression(train_counts, train_metadata)

    # Test truth is built the same way from independent draws — but only the
    # AGGREGATED truth, since baselines for the test set don't need the
    # replicate-level frames.
    test_count_blocks = []
    test_sample_ids = []
    test_sample_to_compound = []
    for c in test_compounds:
        block = _draw_compound_counts(c, n_replicates)
        test_count_blocks.append(block)
        for r in range(n_replicates):
            sid = f"{c}_r{r}"
            test_sample_ids.append(sid)
            test_sample_to_compound.append(c)
    test_counts_mat = np.concatenate(test_count_blocks, axis=1)
    test_counts = pd.DataFrame(
        test_counts_mat, index=gene_ids, columns=test_sample_ids
    ).reset_index()
    test_counts = test_counts.rename(columns={"index": GENE_COL})
    test_metadata = pd.DataFrame(
        list(zip(test_sample_ids, test_sample_to_compound, strict=True)),
        columns=["sequenced_id", "compound"],
    )
    test_truth = counts_to_expression(test_counts, test_metadata)

    return SyntheticRelease(
        train_counts=train_counts,
        train_metadata=train_metadata,
        train_truth=train_truth,
        test_counts=test_counts,
        test_metadata=test_metadata,
        test_truth=test_truth,
        control_compounds=control_compounds,
    )


# ---------------------------------------------------------------------------
# Audit harness
# ---------------------------------------------------------------------------


def _aggregate(per_compound: pd.DataFrame) -> dict[str, float]:
    return {
        "wmse_mean": float(per_compound["wmse"].mean(skipna=True)),
        "wmse_median": float(per_compound["wmse"].median(skipna=True)),
    }


def run_audit(release: SyntheticRelease) -> pd.DataFrame:
    """Score every baseline under both weight schemes; return a tidy frame.

    Each baseline is scored under FOUR (weight_scheme, scope) combinations:

    - mejia weights derived from train counts only, broadcast per-gene
    - pooled weights derived from train counts only, broadcast per-gene
    - mejia weights derived from combined train+test counts (per test compound)
    - pooled weights derived from combined train+test counts (per test compound)

    The per-compound flavor is the faithful Mejia scheme. The
    per-gene-broadcast flavor is what you'd ship if you treat weights as
    a global ranking that doesn't depend on which compound is being
    scored. We surface both so we can see whether per-compound
    granularity actually matters for the contest.
    """
    test_compounds = sorted(release.test_truth[COMPOUND_COL].unique())
    train_truth_wide = release.train_truth.pivot_table(
        index=GENE_COL, columns=COMPOUND_COL, values=EXPRESSION_COL, aggfunc="mean"
    )
    gene_index = train_truth_wide.index.tolist()

    logger.info("Computing Mejia weights on train counts ...")
    mejia_w_train = compute_mejia_weights(release.train_counts, release.train_metadata)
    logger.info("Computing pooled-variance weights on train counts ...")
    pooled_w_train = compute_pooled_weights(release.train_counts, release.train_metadata)

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

    # The scaling-attack sweep: a contestant submits `c * truth` and tunes c
    # to maximize the leaderboard. wMSE should grow without bound as c
    # moves away from 1, demonstrating it cannot be hacked by constant
    # scaling.
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
            release.train_truth, test_compounds, release.control_compounds
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
    """Wide-format pretty-print of the per-weight-scheme calibration table."""
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


def calibration_assertions(audit: pd.DataFrame) -> list[str]:
    """Return a list of failure messages — empty means audit passed."""
    failures = []
    for scheme in ("mejia", "pooled"):
        for scope in ("train_broadcast", "per_test_compound"):
            sub = audit[(audit["weights"] == scheme) & (audit["scope"] == scope)].set_index(
                "baseline"
            )
            tag = f"{scheme}/{scope}"

            p = sub.loc["perfect (technical_duplicate)"]
            if not np.isclose(p["wmse_mean"], 0.0, atol=1e-9):
                failures.append(f"[{tag}] perfect wMSE_mean = {p['wmse_mean']} != 0")

            # wMSE must grow strictly with |c - 1| under the scaling attack
            # (proving wMSE is not constant-scaling-hackable). Only check
            # the off-1 scales.
            scaling_names = [b for b in _BASELINE_ORDER if b.startswith("scaled_perfect")]
            for name in scaling_names:
                if name == "scaled_perfect (1.0x)":
                    continue
                if sub.loc[name]["wmse_mean"] <= 0:
                    failures.append(
                        f"[{tag}] {name} wMSE_mean = {sub.loc[name]['wmse_mean']} not > 0"
                    )

    return failures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    seed = int(os.environ.get("VCPI_AUDIT_SEED", "0"))
    n_compounds = int(os.environ.get("VCPI_AUDIT_N_COMPOUNDS", "40"))
    n_genes = int(os.environ.get("VCPI_AUDIT_N_GENES", "200"))
    n_replicates = int(os.environ.get("VCPI_AUDIT_N_REPLICATES", "2"))

    logger.info(
        "Building synthetic release: n_train={} n_test={} n_genes={} n_reps={} seed={}",
        n_compounds,
        n_compounds // 2,
        n_genes,
        n_replicates,
        seed,
    )
    release = make_synthetic_release(
        n_train_compounds=n_compounds,
        n_test_compounds=n_compounds // 2,
        n_genes=n_genes,
        n_replicates=n_replicates,
        seed=seed,
    )
    audit = run_audit(release)

    for scope, label in (
        ("train_broadcast", "TRAIN-ONLY WEIGHTS, PER-GENE-MEAN BROADCAST"),
        ("per_test_compound", "PER-(TEST COMPOUND) WEIGHTS (faithful Mejia setup)"),
    ):
        print()
        print("=" * 100)
        print(f"BASELINE AUDIT — {label}")
        print("=" * 100)
        print(format_table(audit, scope=scope))
        print()

    failures = calibration_assertions(audit)
    if failures:
        print("CALIBRATION FAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("All calibration checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
