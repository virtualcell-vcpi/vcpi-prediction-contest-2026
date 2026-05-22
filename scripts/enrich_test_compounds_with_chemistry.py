r"""Enrich ``test_compounds.csv`` with SMILES + InChIKey columns.

The bundled ``test_compounds.csv`` originally has only the assay-condition
columns (compound, concentration, cell_line, timepoint). Contestants need
the chemical structure (SMILES + InChIKey) to featurize the held-out
compounds, but VCPI doesn't publish structures for held-out compounds.
This script reads the LIMS chemistry table from the T6667 release on EFS
(joined via metadata → wells → chemistry) and writes the structure
columns back into the bundled file.

Inputs (defaults assume the files have been scp'd to ``/tmp/t6667_chem/``
from ``rkirchner-design:/mnt/efs/P1016/t6667/``):

- ``metadata.csv``                  — ``compound`` (name) + ``sequenced_id``
- ``batch{1..4}-wells.csv``         — ``compound_id`` + ``sequenced_id``
- ``t6667_compounds_canon.csv``     — ``COMPOUND_ID`` + ``smiles_canon`` + ``inchi_key_computed``

Output:

- ``src/vcpi_prediction_contest/data_files/test_compounds.csv``, augmented
  with two new columns: ``smiles`` and ``inchi_key``.

Re-run whenever ``test_compounds.csv`` is regenerated or the T6667
canonical compound table changes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

from vcpi_prediction_contest import load_test_compounds, test_compounds_path

DEFAULT_CHEM_DIR = Path("/tmp/t6667_chem")  # noqa: S108 — script-local convention, scp target
DEFAULT_OUT = Path(test_compounds_path())


def _load_compound_to_id(chem_dir: Path) -> pd.DataFrame:
    """Build a ``compound`` (name) → ``compound_id`` (LIMS int) lookup.

    Joins T6667 sample-level ``metadata.csv`` to the per-batch
    ``wells.csv`` files on ``sequenced_id``. Each compound name maps to
    exactly one LIMS ID once the join settles, so we drop duplicates and
    sanity-check the cardinality.
    """
    metadata = pd.read_csv(chem_dir / "metadata.csv", low_memory=False)
    metadata["sequenced_id"] = metadata["sequenced_id"].astype("int64")

    wells = pd.concat(
        [pd.read_csv(chem_dir / f"batch{b}-wells.csv") for b in (1, 2, 3, 4)],
        ignore_index=True,
    )
    wells["sequenced_id"] = wells["sequenced_id"].astype("int64")
    wells["compound_id"] = wells["compound_id"].astype("Int64")

    joined = metadata[["sequenced_id", "compound"]].merge(
        wells[["sequenced_id", "compound_id"]],
        on="sequenced_id",
        how="left",
    )
    bridge = (
        joined.dropna(subset=["compound_id"])
        .drop_duplicates(subset=["compound"])
        .loc[:, ["compound", "compound_id"]]
        .reset_index(drop=True)
    )
    logger.info(
        "Built compound→compound_id bridge: {} unique compounds, {} unique IDs",
        bridge["compound"].nunique(),
        bridge["compound_id"].nunique(),
    )
    return bridge


def _load_chemistry(chem_dir: Path) -> pd.DataFrame:
    """Slim down the T6667 canonical compound table to (compound_id, smiles, inchi_key).

    We use the post-canonicalization columns (``smiles_canon`` /
    ``inchi_key_computed``) because the original ``SMILES`` / ``INCHI_KEY``
    columns are populated from LIMS only for a subset of compounds —
    the canonical columns fall back to PubChem-resolved structures so
    every T6667 compound has coverage.
    """
    chem = pd.read_csv(chem_dir / "t6667_compounds_canon.csv", low_memory=False)
    chem = chem.rename(
        columns={
            "COMPOUND_ID": "compound_id",
            "smiles_canon": "smiles",
            "inchi_key_computed": "inchi_key",
        }
    )
    chem["compound_id"] = chem["compound_id"].astype("Int64")
    chem = chem[["compound_id", "smiles", "inchi_key"]].dropna(subset=["compound_id"])
    chem = chem.drop_duplicates(subset=["compound_id"])
    logger.info("Loaded chemistry for {} compound IDs", len(chem))
    return chem


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chem-dir", type=Path, default=DEFAULT_CHEM_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    test = load_test_compounds()
    logger.info("Loaded bundled test_compounds.csv: {} rows", len(test))
    test = test.drop(columns=[c for c in ("smiles", "inchi_key") if c in test.columns])

    bridge = _load_compound_to_id(args.chem_dir)
    chem = _load_chemistry(args.chem_dir)

    enriched = test.merge(bridge, on="compound", how="left").merge(
        chem, on="compound_id", how="left"
    )

    missing_id = int(enriched["compound_id"].isna().sum())
    missing_smiles = int(enriched["smiles"].isna().sum())
    missing_inchi = int(enriched["inchi_key"].isna().sum())
    logger.info(
        "Coverage: missing compound_id={}/{}  missing smiles={}/{}  missing inchi_key={}/{}",
        missing_id,
        len(enriched),
        missing_smiles,
        len(enriched),
        missing_inchi,
        len(enriched),
    )
    if missing_smiles:
        examples = enriched.loc[enriched["smiles"].isna(), "compound"].head(5).tolist()
        logger.warning("First 5 test compounds without a SMILES: {}", examples)

    enriched = enriched.drop(columns=["compound_id"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(args.out, index=False)
    logger.info("Wrote enriched test_compounds.csv to {}", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
