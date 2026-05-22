"""Tests for the expression-task data loaders."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pandas as pd
import pytest

from vcpi_prediction_contest.data import (
    TEST_COMPOUNDS_COLS,
    WEIGHTS_SHA256,
    gene_filter_path,
    load_gene_filter,
    load_prediction,
    load_test_compounds,
    load_truth,
    load_weights,
)

# Alias on import: pytest auto-collects any module-level callable whose
# name starts with ``test_``, and the bare ``test_compounds_path``
# import would trip that pattern.
from vcpi_prediction_contest.data import test_compounds_path as _test_compounds_path


def _truth_df():
    return pd.DataFrame(
        {
            "compound": ["A", "A", "B", "B"],
            "gene_id": ["g1", "g2", "g1", "g2"],
            "expression": [3.0, 0.1, 1.5, 0.2],
        }
    )


def _pred_df():
    return pd.DataFrame(
        {
            "compound": ["A", "A", "B", "B"],
            "gene_id": ["g1", "g2", "g1", "g2"],
            "predicted_expression": [2.9, 0.05, 1.4, 0.25],
        }
    )


def test_load_truth_parquet(tmp_path):
    p = tmp_path / "truth.parquet"
    _truth_df().to_parquet(p)
    df = load_truth(p)
    assert set(df["compound"]) == {"A", "B"}
    assert "expression" in df.columns


def test_load_truth_csv(tmp_path):
    p = tmp_path / "truth.csv"
    _truth_df().to_csv(p, index=False)
    df = load_truth(p)
    assert len(df) == 4


def test_load_truth_missing_columns(tmp_path):
    p = tmp_path / "bad.parquet"
    _truth_df().drop(columns=["expression"]).to_parquet(p)
    with pytest.raises(ValueError, match="missing required columns"):
        load_truth(p)


def test_load_prediction(tmp_path):
    p = tmp_path / "pred.parquet"
    _pred_df().to_parquet(p)
    df = load_prediction(p)
    assert df.iloc[0]["predicted_expression"] == 2.9


def test_load_prediction_missing_columns(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"compound": ["A"]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_prediction(p)


def test_load_gene_filter_from_path(tmp_path):
    p = tmp_path / "genes.parquet"
    pd.DataFrame({"gene_id": ["g3", "g1", "g2"]}).to_parquet(p)
    genes = load_gene_filter(p)
    assert genes == ["g1", "g2", "g3"]


def test_load_gene_filter_defaults_to_bundled():
    bundled = load_gene_filter()
    explicit = load_gene_filter(gene_filter_path())
    assert bundled == explicit
    assert len(bundled) > 0
    assert all(isinstance(g, str) for g in bundled)
    assert bundled == sorted(set(bundled))


def test_load_weights_roundtrip(tmp_path):
    p = tmp_path / "weights.parquet"
    pd.DataFrame({"gene_id": ["g0", "g1"], "weight": [0.3, 0.7]}).to_parquet(p)
    w = load_weights(p)
    assert w.loc["g0"] == pytest.approx(0.3)
    assert w.loc["g1"] == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# Bundled test_compounds.csv (shipped inside the wheel)
# ---------------------------------------------------------------------------


def test_bundled_test_compounds_path_resolves_to_wheel_file():
    p = _test_compounds_path()
    assert p.exists()
    assert p.name == "test_compounds.csv"
    assert "vcpi_prediction_contest" in p.parts
    assert "data_files" in p.parts


def test_load_test_compounds_schema_and_contract():
    df = load_test_compounds()
    assert set(df.columns) == set(TEST_COMPOUNDS_COLS)
    assert df["compound"].is_unique
    # ``compound`` is the ``user_compound_id`` (numeric LIMS ID stored
    # as a string so it matches ``W.columns`` and
    # ``metadata.user_compound_id`` from ``vcpi-client``).
    assert df["compound"].map(type).eq(str).all(), "compound must be string-typed"
    assert df["compound"].str.fullmatch(r"\d+").all(), "compound must be a numeric user_compound_id"
    # ``compound_name`` is the human-readable chemistry name kept for
    # display.
    assert df["compound_name"].notna().all(), "every test compound must have a compound_name"
    assert df["compound_name"].is_unique, "compound_name must be unique"
    assert (df["compound_concentration"] == 10.0).all()
    assert (df["compound_concentration_unit"] == "uM").all()
    assert (df["cell_line"] == "THP-1").all()
    assert (df["timepoint"] == "24h").all()
    assert len(df) > 0
    assert df["smiles"].notna().all(), "every test compound must have a SMILES"
    assert df["inchi_key"].notna().all(), "every test compound must have an InChIKey"
    # InChIKey is always 27 chars in three hyphen-separated blocks of 14/10/1
    inchi_pattern = r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$"
    assert df["inchi_key"].str.match(inchi_pattern).all(), "malformed InChIKey"


# ---------------------------------------------------------------------------
# Bundled gene_filter.csv (shipped inside the wheel)
# ---------------------------------------------------------------------------


def test_bundled_gene_filter_path_resolves_to_wheel_file():
    p = gene_filter_path()
    assert p.exists()
    assert p.name == "gene_filter.csv"
    assert "vcpi_prediction_contest" in p.parts
    assert "data_files" in p.parts


# ---------------------------------------------------------------------------
# Cross-asset consistency (skipped unless a local weights.parquet is present)
# ---------------------------------------------------------------------------


def _resolve_local_weights_path() -> Path | None:
    """Find a locally-available weights.parquet without triggering a download.

    Resolution order, first hit wins:
    1. ``$VCPI_WEIGHTS_PATH`` (if set and the file exists).
    2. ``_artifacts/weights.parquet`` (where ``build_canonical_assets.py``
       drops the local build by default; only present on a maintainer
       machine that has just rebuilt the artifact).

    Returns ``None`` to signal "no local weights file → skip the
    cross-asset checks". CI machines without the build artifact will
    skip silently; a maintainer with `_artifacts/weights.parquet`
    (or anyone who has set ``VCPI_WEIGHTS_PATH``) gets the full
    invariant check before shipping.
    """
    env = os.environ.get("VCPI_WEIGHTS_PATH")
    if env:
        env_path = Path(env)
        if env_path.is_file():
            return env_path
    repo_default = Path(__file__).resolve().parent.parent / "_artifacts" / "weights.parquet"
    if repo_default.is_file():
        return repo_default
    return None


@pytest.fixture(scope="module")
def local_weights_path() -> Path:
    path = _resolve_local_weights_path()
    if path is None:
        pytest.skip(
            "no local weights.parquet found (set VCPI_WEIGHTS_PATH or build into _artifacts/)",
        )
    return path


@pytest.fixture(scope="module")
def local_weights_matrix(local_weights_path: Path) -> pd.DataFrame:
    return pd.read_parquet(local_weights_path)


def test_local_weights_sha_matches_data_py(local_weights_path: Path) -> None:
    """SHA-256 of the local weights file matches ``data.WEIGHTS_SHA256``.

    Catches the failure mode where the maintainer regenerates
    ``weights.parquet`` but forgets to update ``WEIGHTS_SHA256`` (or
    vice versa). A SHA drift means contestants will either fail the
    integrity check post-download or silently load the wrong matrix.
    """
    h = hashlib.sha256()
    with local_weights_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    assert h.hexdigest() == WEIGHTS_SHA256, (
        f"local weights SHA {h.hexdigest()} != data.WEIGHTS_SHA256 {WEIGHTS_SHA256}; "
        f"regenerate either the weights or the constant so they agree before shipping"
    )


def test_bundled_gene_filter_equals_weights_gene_index(
    local_weights_matrix: pd.DataFrame,
) -> None:
    """Bundled ``gene_filter.csv`` row set == weights gene-index row set.

    Catches the failure mode where ``build_canonical_assets.py``
    regenerates both ``gene_filter.csv`` and ``weights.parquet`` but
    only the latter gets copied back into the repo, leaving one gene
    on either side of the mean-CPM=1.0 threshold misaligned. A swap
    causes ``score_compounds(...).reindex(gene_filter)`` to produce a
    silent all-NaN row that turns into wMSE divide-by-zero downstream.
    """
    gf = set(load_gene_filter())
    w = set(local_weights_matrix.index.astype(str))
    only_gf = sorted(gf - w)
    only_w = sorted(w - gf)
    assert not only_gf, (
        f"bundled gene_filter has {len(only_gf)} genes missing from weights (e.g. {only_gf[:3]})"
    )
    assert not only_w, (
        f"weights index has {len(only_w)} genes missing from bundled gene_filter "
        f"(e.g. {only_w[:3]})"
    )


def test_bundled_test_compounds_disjoint_from_weights(
    local_weights_matrix: pd.DataFrame,
) -> None:
    """No test compound appears as a column in the weights matrix.

    The whole point of the held-out set is that contestants do not
    have per-compound weights for it. Any overlap is silent train/test
    leakage in the shipped artifacts.
    """
    test_ids = set(load_test_compounds()["compound"].astype(str))
    leaked = sorted(test_ids & set(local_weights_matrix.columns.astype(str)))
    assert not leaked, (
        f"{len(leaked)} test compounds appear as weight columns "
        f"(e.g. {leaked[:5]}); test/train assets are leaking"
    )
