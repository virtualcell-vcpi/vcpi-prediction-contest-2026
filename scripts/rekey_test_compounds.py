r"""Re-key ``test_compounds.csv`` from chemistry name to ``user_compound_id``.

The bundled ``test_compounds.csv`` used to key compounds by their
chemistry name (e.g. ``"(+)-Bromocriptine methanesulfonate"``). That
worked when chemistry names were a column on every training datum, but
``vcpi-client`` only exposes ``metadata.user_compound_id`` (the numeric
LIMS ID, e.g. ``"9251300"``) — there is no chemistry name field. To let
contestants join training data straight onto the contest's canonical
compound axis, we re-key both shipped artifacts (``weights.parquet``
and ``test_compounds.csv``) on ``user_compound_id``.

This script applies the rename in place on
``src/vcpi_prediction_contest/data_files/test_compounds.csv``:

- The old ``compound`` column (chemistry name) is renamed to
  ``compound_name`` and kept for display.
- A new ``compound`` column is added that holds the
  ``user_compound_id`` (string).

The mapping is built from the LIMS canonical compound tables on EFS:

- ``/mnt/efs/P1016/t6667/data/t6667_compounds_canon.csv``
- ``/mnt/efs/P1016/t6667/data/t6391_compounds_canon.csv``

scp them down to ``--canon-dir`` (default ``/tmp/vcpi_canon``) before
running locally; on the design instance the script can just point at
the EFS paths directly.

Verified coverage: all 1,064 / 1,064 held-out test compound names
resolve to a ``user_compound_id``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

from vcpi_prediction_contest import test_compounds_path

DEFAULT_CANON_DIR = Path("/tmp/vcpi_canon")  # noqa: S108 — script-local scp target
DEFAULT_OUT = Path(test_compounds_path())
CANON_FILES = ("t6667_compounds_canon.csv", "t6391_compounds_canon.csv")


def _load_name_to_ucid(canon_dir: Path) -> pd.Series:
    """Build a ``name -> user_compound_id`` Series from the canon CSVs.

    Both canon files share ``COMPOUND_ID, NAME, ...`` columns. We
    concatenate, drop rows missing either field, cast ``COMPOUND_ID``
    to string, and keep the first occurrence per name. A small number
    of names (verified to be 2 across both files at time of writing)
    map to multiple ``COMPOUND_ID`` values; none of them are present in
    the held-out test set so the ``drop_duplicates`` policy is safe.
    """
    frames = []
    for fname in CANON_FILES:
        path = canon_dir / fname
        if not path.is_file():
            msg = f"canon file not found: {path}"
            raise FileNotFoundError(msg)
        frames.append(pd.read_csv(path, usecols=["COMPOUND_ID", "NAME"], low_memory=False))
    canon = pd.concat(frames, ignore_index=True).dropna(subset=["COMPOUND_ID", "NAME"])
    canon["COMPOUND_ID"] = canon["COMPOUND_ID"].astype("int64").astype(str)
    # Canon NAMEs occasionally have leading/trailing whitespace (e.g.
    # ' Alisertib'); strip so exact-string lookups succeed.
    canon["NAME"] = canon["NAME"].astype(str).str.strip()
    canon = canon[canon["NAME"] != ""]
    bridge = canon.drop_duplicates(subset=["NAME"]).set_index("NAME")["COMPOUND_ID"]
    logger.info(
        "Built name -> user_compound_id bridge: {} unique names from {} canon rows",
        len(bridge),
        len(canon),
    )
    return bridge


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--canon-dir",
        type=Path,
        default=DEFAULT_CANON_DIR,
        help="Directory containing t6667 / t6391 *_compounds_canon.csv files.",
    )
    parser.add_argument(
        "--in",
        dest="in_path",
        type=Path,
        default=DEFAULT_OUT,
        help="Input test_compounds.csv (defaults to the bundled wheel file).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output path (defaults to overwriting the bundled file in place).",
    )
    args = parser.parse_args(argv)

    bridge = _load_name_to_ucid(args.canon_dir)

    test = pd.read_csv(args.in_path)
    logger.info("Loaded {} ({} rows)", args.in_path, len(test))
    if "compound_name" in test.columns:
        logger.warning(
            "Input already has a `compound_name` column; assuming it has already been re-keyed."
        )
        return 0

    ucid = test["compound"].map(bridge)
    n_missing = int(ucid.isna().sum())
    if n_missing:
        missing = test.loc[ucid.isna(), "compound"].tolist()
        logger.error(
            "Cannot re-key: {} test compounds have no user_compound_id ({})",
            n_missing,
            missing[:5],
        )
        return 1

    out = test.rename(columns={"compound": "compound_name"}).copy()
    out["compound"] = ucid.astype(str).to_numpy()
    cols = [
        "compound",
        "compound_name",
        "compound_concentration",
        "compound_concentration_unit",
        "cell_line",
        "timepoint",
        "smiles",
        "inchi_key",
    ]
    out = out[cols]

    if not out["compound"].is_unique:
        dup_count = int((~out["compound"].duplicated(keep=False)).sum())
        msg = (
            f"Re-keyed test_compounds has duplicate user_compound_id values "
            f"({len(out) - dup_count} duplicates). Aborting."
        )
        raise RuntimeError(msg)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    logger.info(
        "Wrote re-keyed test_compounds.csv to {} ({} rows, {} unique user_compound_ids)",
        args.out,
        len(out),
        out["compound"].nunique(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
