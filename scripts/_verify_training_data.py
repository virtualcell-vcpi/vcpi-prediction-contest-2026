"""Independent verification of the README training-data recipe artifacts.

Run this AFTER executing the snippet from the README's "Getting the training
data" section. It loads:

- ``train_counts.parquet``     (built by the snippet, wide gene_id x sample)
- ``train_metadata.parquet``   (built by the snippet, one row per sample)
- ``train_chemistry.parquet``  (built by the snippet, one row per compound)
- ``weights.parquet``          (saved by the snippet from ``load_weights_matrix``)
- bundled ``test_compounds.csv`` and ``gene_filter.csv`` (from the wheel)

…and runs a battery of completeness / self-consistency checks. Prints a
PASS/FAIL banner at the end and exits non-zero on any failure.

Leading underscore so pytest does not collect it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from vcpi_prediction_contest import (
    load_gene_filter,
    load_test_compounds,
    load_weights_matrix,
)

CONTROL_COMPOUNDS: tuple[str, ...] = (
    "DMSO",
    # Control compounds appear with several spelling variants in the
    # vcpi metadata (with/without trailing 'e', hyphen vs space).
    "Staurosporine",
    "Staurosporin",
    "Brefeldin A",
    "Brefeldin-A",
    "Rigosertib",
    "Trichostatin A",
    "Trichostatin-A",
)

EXPECTED_COMPOUND_COUNT = 14_031
COMPOUND_COUNT_TOLERANCE = 50

MISSING_FROM_WEIGHTS_GRACE = 7

CONTEST_LIBRARY_NM = 10_000  # 10 µM expressed in nM (vcpi storage convention)

CHECK_PREFIX = "  "


class CheckResult:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []
        self.passes: list[str] = []

    def check(self, *, cond: bool, ok: str, fail: str) -> None:
        if cond:
            self.passes.append(ok)
            print(f"{CHECK_PREFIX}PASS  {ok}", flush=True)
        else:
            self.failures.append(fail)
            print(f"{CHECK_PREFIX}FAIL  {fail}", flush=True)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        print(f"{CHECK_PREFIX}WARN  {msg}", flush=True)

    def info(self, msg: str) -> None:
        print(f"{CHECK_PREFIX}info  {msg}", flush=True)


def _load_artifacts(workdir: Path) -> dict[str, object]:
    print(f"Loading artifacts from {workdir} ...", flush=True)
    counts = pd.read_parquet(workdir / "train_counts.parquet")
    metadata = pd.read_parquet(workdir / "train_metadata.parquet")
    chemistry = pd.read_parquet(workdir / "train_chemistry.parquet")
    weights = load_weights_matrix(workdir / "weights.parquet")
    test_compounds = load_test_compounds()
    genes = load_gene_filter()
    print(f"  counts:        shape={counts.shape}", flush=True)
    print(f"  metadata:      shape={metadata.shape}", flush=True)
    print(f"  chemistry:     shape={chemistry.shape}", flush=True)
    print(f"  weights:       shape={weights.shape}", flush=True)
    print(f"  test_compounds shape={test_compounds.shape}", flush=True)
    print(f"  gene_filter:   {len(genes)} genes", flush=True)
    return {
        "counts": counts,
        "metadata": metadata,
        "chemistry": chemistry,
        "weights": weights,
        "test_compounds": test_compounds,
        "genes": genes,
    }


def _check_sample_coverage(counts: pd.DataFrame, metadata: pd.DataFrame, r: CheckResult) -> None:
    print("\n[1] Sample coverage: counts <-> metadata", flush=True)
    sample_cols = [c for c in counts.columns if c != "gene_id"]
    meta_ids = set(metadata["sequenced_id"].astype(str))
    count_ids = set(map(str, sample_cols))
    only_in_counts = count_ids - meta_ids
    only_in_meta = meta_ids - count_ids
    r.info(f"counts has {len(count_ids):,} sample columns; metadata has {len(meta_ids):,} rows")
    r.check(
        cond=not only_in_counts,
        ok="every counts column appears in metadata",
        fail=f"{len(only_in_counts)} counts columns missing from metadata "
        f"(e.g. {sorted(only_in_counts)[:5]})",
    )
    r.check(
        cond=not only_in_meta,
        ok="every metadata sequenced_id appears as a counts column",
        fail=f"{len(only_in_meta)} metadata sequenced_ids missing from counts "
        f"(e.g. {sorted(only_in_meta)[:5]})",
    )


def _check_compound_in_weights(
    metadata: pd.DataFrame, weights: pd.DataFrame, r: CheckResult
) -> None:
    print("\n[2] Compound coverage: metadata <-> weights", flush=True)
    meta_compounds = set(metadata["user_compound_id"].astype(str))
    library_compounds = meta_compounds - set(CONTROL_COMPOUNDS)
    w_cols = set(weights.columns.astype(str))
    inter = library_compounds & w_cols
    only_meta = library_compounds - w_cols
    only_w = w_cols - library_compounds
    r.info(f"|metadata library compounds|        = {len(library_compounds):,}")
    r.info(f"|weights compound columns|          = {len(w_cols):,}")
    r.info(f"|metadata ∩ weights|                 = {len(inter):,}")
    r.info(f"|metadata - weights| (missing from W) = {len(only_meta):,}")
    r.info(f"|weights - metadata| (extra in W)     = {len(only_w):,}")

    grace = MISSING_FROM_WEIGHTS_GRACE
    if len(only_meta) > grace:
        sample_missing = sorted(only_meta)[:10]
        r.failures.append(
            f"{len(only_meta)} library compounds in metadata are missing from weights "
            f"(grace={grace}); sample: {sample_missing}"
        )
        print(
            f"{CHECK_PREFIX}FAIL  {len(only_meta)} library compounds missing from weights "
            f"(grace={grace}); sample: {sample_missing}",
            flush=True,
        )
    elif only_meta:
        r.warn(
            f"{len(only_meta)} library compounds in metadata not in weights "
            f"(within grace={grace}): {sorted(only_meta)}"
        )
    else:
        print(
            f"{CHECK_PREFIX}PASS  zero library compounds missing from weights",
            flush=True,
        )
        r.passes.append("zero library compounds missing from weights")


def _check_test_train_disjoint(
    metadata: pd.DataFrame, test_compounds: pd.DataFrame, r: CheckResult
) -> None:
    print("\n[3] Test/train disjointness", flush=True)
    train_ids = set(metadata["user_compound_id"].astype(str))
    test_ids = set(test_compounds["compound"].astype(str))
    overlap = train_ids & test_ids
    r.info(f"|train compounds| = {len(train_ids):,}  |test compounds| = {len(test_ids):,}")
    r.check(
        cond=not overlap,
        ok="test_compounds disjoint from training metadata",
        fail=f"{len(overlap)} compounds appear in BOTH test and train: {sorted(overlap)[:10]}",
    )


def _check_weight_lookup(
    metadata: pd.DataFrame, weights: pd.DataFrame, genes: list[str], r: CheckResult
) -> None:
    print("\n[4] Weights lookup spot-check (5 random library compounds)", flush=True)
    library = set(metadata["user_compound_id"].astype(str)) - set(CONTROL_COMPOUNDS)
    overlap = sorted(library & set(weights.columns.astype(str)))
    rng = np.random.default_rng(0)
    sample = rng.choice(overlap, size=min(5, len(overlap)), replace=False).tolist()
    n_genes = len(genes)
    n_w_genes = weights.shape[0]
    r.info(f"|gene_filter|={n_genes:,}  |weights index|={n_w_genes:,}")
    for cid in sample:
        col = weights[cid]
        finite = np.all(np.isfinite(col.to_numpy(dtype=np.float32)))
        right_len = len(col) == n_w_genes
        r.check(
            cond=bool(finite and right_len),
            ok=f"compound {cid}: finite & len={len(col)}",
            fail=f"compound {cid}: finite={finite}, len={len(col)} (expected {n_w_genes})",
        )


def _check_smiles_roundtrip(
    metadata: pd.DataFrame,
    chemistry: pd.DataFrame,
    test_compounds: pd.DataFrame,
    r: CheckResult,
) -> None:
    print("\n[5] SMILES round-trip (5 train + 5 test compounds)", flush=True)
    rng = np.random.default_rng(0)
    train_ids = metadata["user_compound_id"].astype(str).unique().tolist()
    train_sample = rng.choice(train_ids, size=5, replace=False).tolist()
    chem_lookup = chemistry.set_index(chemistry["user_compound_id"].astype(str))
    for cid in train_sample:
        if cid not in chem_lookup.index:
            r.check(cond=False, ok="", fail=f"train compound {cid} missing from chemistry")
            continue
        row = chem_lookup.loc[cid]
        smi = row["smiles"]
        ikey = row["inchi_key"]
        r.check(
            cond=isinstance(smi, str) and len(smi) > 0,
            ok=f"train {cid}: smiles non-empty",
            fail=f"train {cid}: smiles missing (got {smi!r})",
        )
        r.check(
            cond=isinstance(ikey, str) and len(ikey) > 0,
            ok=f"train {cid}: inchi_key non-empty",
            fail=f"train {cid}: inchi_key missing (got {ikey!r})",
        )

    test_ids = test_compounds["compound"].astype(str).tolist()
    test_sample = rng.choice(test_ids, size=5, replace=False).tolist()
    test_lookup = test_compounds.set_index(test_compounds["compound"].astype(str))
    for cid in test_sample:
        row_test = test_lookup.loc[cid]
        smi_test = row_test["smiles"]
        ikey_test = row_test["inchi_key"]
        r.check(
            cond=isinstance(smi_test, str) and len(smi_test) > 0,
            ok=f"test {cid}: bundled smiles non-empty",
            fail=f"test {cid}: bundled smiles missing",
        )
        if cid in chem_lookup.index:
            ikey_chem = chem_lookup.loc[cid, "inchi_key"]
            r.check(
                cond=ikey_chem == ikey_test,
                ok=f"test {cid}: bundled inchi_key matches chemistry",
                fail=f"test {cid}: inchi_key disagrees (chem={ikey_chem}, "
                f"test_compounds={ikey_test})",
            )
        else:
            r.info(f"test {cid}: not in vcpi chemistry table (expected — test set is held out)")


def _check_compound_count(metadata: pd.DataFrame, r: CheckResult) -> None:
    print("\n[6] Distinct compound count vs vcpi.chemistry expectation", flush=True)
    n = metadata["user_compound_id"].astype(str).nunique()
    delta = abs(n - EXPECTED_COMPOUND_COUNT)
    r.info(
        f"distinct user_compound_id in metadata = {n:,} "
        f"(expected ≈ {EXPECTED_COMPOUND_COUNT:,}, tolerance ±{COMPOUND_COUNT_TOLERANCE})"
    )
    r.check(
        cond=delta <= COMPOUND_COUNT_TOLERANCE,
        ok=f"compound count within tolerance ({delta} off)",
        fail=f"compound count off by {delta} (got {n:,}, expected ≈ {EXPECTED_COMPOUND_COUNT:,})",
    )


def _check_data_corruption(
    counts: pd.DataFrame,
    chemistry: pd.DataFrame,
    metadata: pd.DataFrame,
    r: CheckResult,
) -> None:
    print("\n[7] Data corruption / sanity", flush=True)
    sample_cols = [c for c in counts.columns if c != "gene_id"]
    counts_arr = counts[sample_cols].to_numpy()
    nonneg = bool(np.all(counts_arr >= 0))
    int_like = np.issubdtype(counts_arr.dtype, np.integer)
    r.check(
        cond=bool(nonneg and int_like),
        ok=f"counts non-negative integers (dtype={counts_arr.dtype})",
        fail=f"counts not non-neg ints (dtype={counts_arr.dtype}, min={counts_arr.min()})",
    )

    n_null_smi = chemistry["smiles"].isna().sum() + (chemistry["smiles"] == "").sum()
    r.check(
        cond=n_null_smi == 0,
        ok="chemistry: zero null/empty smiles",
        fail=f"chemistry: {n_null_smi} null/empty smiles",
    )

    dmso_mask = metadata["user_compound_id"].astype(str) == "DMSO"
    library_mask = ~dmso_mask
    library_meta = metadata[library_mask]
    bad_unit = library_meta["compound_concentration_unit"] != "nM"
    bad_conc = library_meta["compound_concentration"] != CONTEST_LIBRARY_NM
    n_bad = int(bad_unit.sum() + bad_conc.sum())
    r.check(
        cond=n_bad == 0,
        ok=f"all library samples at compound_concentration={CONTEST_LIBRARY_NM} nM",
        fail=f"{n_bad} library samples have wrong concentration/unit",
    )

    cell_bad = (metadata["cell_line"] != "THP-1").sum()
    time_bad = (metadata["timepoint"] != "24h").sum()
    r.check(
        cond=cell_bad == 0 and time_bad == 0,
        ok="all samples are THP-1 / 24h",
        fail=f"{cell_bad} non-THP-1 and/or {time_bad} non-24h samples",
    )


def main() -> int:
    workdir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    art = _load_artifacts(workdir)
    r = CheckResult()
    counts = art["counts"]
    metadata = art["metadata"]
    chemistry = art["chemistry"]
    weights = art["weights"]
    test_compounds = art["test_compounds"]
    genes = art["genes"]

    metadata["user_compound_id"] = metadata["user_compound_id"].astype(str)
    chemistry["user_compound_id"] = chemistry["user_compound_id"].astype(str)
    test_compounds["compound"] = test_compounds["compound"].astype(str)

    _check_sample_coverage(counts, metadata, r)
    _check_compound_in_weights(metadata, weights, r)
    _check_test_train_disjoint(metadata, test_compounds, r)
    _check_weight_lookup(metadata, weights, genes, r)
    _check_smiles_roundtrip(metadata, chemistry, test_compounds, r)
    _check_compound_count(metadata, r)
    _check_data_corruption(counts, chemistry, metadata, r)

    print("\n" + "=" * 60, flush=True)
    print(
        f"SUMMARY: {len(r.passes)} pass | {len(r.warnings)} warn | {len(r.failures)} fail",
        flush=True,
    )
    print("=" * 60, flush=True)
    if r.failures:
        print("FAILURES:", flush=True)
        for f in r.failures:
            print(f"  - {f}", flush=True)
        return 1
    print("OK — all checks passed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
