"""Data loaders for the VCPI expression-prediction contest.

Schemas (all long-format parquet/csv; see ``docs/data_contract.md``):

- ``truth``: ``compound, gene_id, expression`` — the per-(compound,
  gene) mean expression value (currently log2(CPM + 1), but the metric
  panel is unit-agnostic).
- ``prediction`` (contestant submission): ``compound, gene_id,
  predicted_expression``.
- ``gene_filter``: a single ``gene_id`` column listing the scored
  genes.

The package also ships two canonical contest assets inside the
wheel under ``vcpi_prediction_contest.data_files``:

- ``test_compounds.csv`` (:func:`load_test_compounds` /
  :func:`test_compounds_path`) — the held-out compounds the
  leaderboard server evaluates.
- ``gene_filter.csv`` (:func:`load_gene_filter` /
  :func:`gene_filter_path`) — the gene set scored.
"""

from __future__ import annotations

import hashlib
import os
import sys
from importlib.resources import as_file, files
from pathlib import Path

import pandas as pd
import requests
from loguru import logger

from vcpi_prediction_contest.metrics import (
    COMPOUND_COL,
    EXPRESSION_COL,
    GENE_COL,
    PRED_COL,
)

REQUIRED_TRUTH_COLS: tuple[str, ...] = (COMPOUND_COL, GENE_COL, EXPRESSION_COL)
REQUIRED_PRED_COLS: tuple[str, ...] = (COMPOUND_COL, GENE_COL, PRED_COL)

# Bundled contest assets shipped inside the wheel. The leaderboard
# server reads exactly these files at scoring time; contestants get
# the same bytes so there is no ambiguity about which compounds /
# genes count.
TEST_COMPOUNDS_FILENAME = "test_compounds.csv"
# ``compound`` is the ``user_compound_id`` (numeric LIMS ID as a string),
# the canonical contest join key matching ``metadata.user_compound_id``
# from ``vcpi-client``. ``compound_name`` carries the human-readable
# chemistry name for display only.
TEST_COMPOUNDS_COLS: tuple[str, ...] = (
    COMPOUND_COL,
    "compound_name",
    "compound_concentration",
    "compound_concentration_unit",
    "cell_line",
    "timepoint",
    "smiles",
    "inchi_key",
)
GENE_FILTER_FILENAME = "gene_filter.csv"


def _read_long(path: str | Path) -> pd.DataFrame:
    """Load a long-format CSV / TSV / parquet file by extension."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def _require_columns(df: pd.DataFrame, cols: tuple[str, ...], *, path: Path) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        msg = f"{path} is missing required columns: {missing}"
        raise ValueError(msg)


def load_truth(path: str | Path) -> pd.DataFrame:
    """Load held-out test-compound expression truth (long format).

    Required columns: ``compound, gene_id, expression``.
    """
    path = Path(path)
    df = _read_long(path)
    _require_columns(df, REQUIRED_TRUTH_COLS, path=path)
    logger.info(
        "Loaded truth from {} ({} rows, {} compounds, {} genes)",
        path,
        len(df),
        df[COMPOUND_COL].nunique(),
        df[GENE_COL].nunique(),
    )
    return df


def load_prediction(path: str | Path) -> pd.DataFrame:
    """Load a submission frame and validate required columns.

    Required columns: ``compound, gene_id, predicted_expression``.
    """
    path = Path(path)
    df = _read_long(path)
    _require_columns(df, REQUIRED_PRED_COLS, path=path)
    logger.info(
        "Loaded prediction from {} ({} rows, {} compounds, {} genes)",
        path,
        len(df),
        df[COMPOUND_COL].nunique(),
        df[GENE_COL].nunique(),
    )
    return df


def gene_filter_path() -> Path:
    """Return a filesystem path to the bundled ``gene_filter.csv``.

    The file is shipped inside the installed wheel; this helper resolves
    it via :mod:`importlib.resources` so it works whether you installed
    the package or are running from source.
    """
    resource = files("vcpi_prediction_contest.data_files") / GENE_FILTER_FILENAME
    with as_file(resource) as p:
        return Path(p)


def load_gene_filter(path: str | Path | None = None) -> list[str]:
    """Load the scored gene set.

    Parameters
    ----------
    path
        Optional path to a ``gene_filter`` file with a ``gene_id``
        column (``.csv`` / ``.tsv`` / ``.parquet``). When ``None``
        (default), loads the canonical ``gene_filter.csv`` bundled
        inside the wheel — exactly the gene set the leaderboard server
        scores on.

    Returns
    -------
    list[str]
        Sorted, deduplicated ``gene_id`` strings.
    """
    resolved = gene_filter_path() if path is None else Path(path)
    df = _read_long(resolved)
    if GENE_COL not in df.columns:
        msg = f"{resolved} must contain a `{GENE_COL}` column"
        raise ValueError(msg)
    genes = sorted(df[GENE_COL].astype(str).unique().tolist())
    logger.info("Loaded gene_filter from {} ({} genes)", resolved, len(genes))
    return genes


def load_weights(path: str | Path) -> pd.Series:
    """Load a per-gene weight vector from disk.

    File must contain a ``gene_id`` column and a ``weight`` column.
    Returns a Series indexed by ``gene_id`` (string dtype).
    """
    path = Path(path)
    df = _read_long(path)
    _require_columns(df, (GENE_COL, "weight"), path=path)
    out = pd.Series(
        df["weight"].to_numpy(),
        index=df[GENE_COL].astype(str).to_numpy(),
        name="weight",
    )
    out.index.name = GENE_COL
    logger.info("Loaded {} per-gene weights from {} (sum={:.4f})", len(out), path, out.sum())
    return out


def test_compounds_path() -> Path:
    """Return a filesystem path to the bundled ``test_compounds.csv``.

    The file is shipped inside the installed wheel; this helper resolves
    it via :mod:`importlib.resources` so it works whether you installed
    the package or are running from source.
    """
    resource = files("vcpi_prediction_contest.data_files") / TEST_COMPOUNDS_FILENAME
    with as_file(resource) as p:
        return Path(p)


def load_test_compounds() -> pd.DataFrame:
    """Load the bundled ``test_compounds.csv``.

    Returns the canonical list of held-out compounds contestants must
    predict. Columns:

    - ``compound`` — ``user_compound_id`` (numeric LIMS ID as a string),
      the canonical contest join key. Matches
      ``metadata.user_compound_id`` from ``vcpi-client``,
      ``W.columns`` of :func:`load_weights_matrix`, and the ``compound``
      column of contestant submissions.
    - ``compound_name`` — human-readable chemistry name. Provided for
      display only; do not use it for joining (``vcpi-client`` does not
      expose chemistry names).
    - ``compound_concentration`` (always 10),
      ``compound_concentration_unit`` (always ``"uM"``), ``cell_line``
      (always ``"THP-1"``), ``timepoint`` (always ``"24h"``).
    - ``smiles`` (canonical SMILES from the T6667 compound table),
      ``inchi_key`` (computed InChIKey).

    The leaderboard server reads exactly this file at scoring time, so
    a submission must cover every ``compound`` row times every gene in
    the official ``gene_filter``.
    """
    path = test_compounds_path()
    # ``compound`` is a numeric user_compound_id (LIMS id) but the
    # canonical contest dtype is string — pandas would otherwise auto-
    # detect it as int64 and break comparisons against ``W.columns``
    # (string) or ``metadata.user_compound_id`` (string).
    df = pd.read_csv(path, dtype={COMPOUND_COL: str})
    _require_columns(df, TEST_COMPOUNDS_COLS, path=path)
    logger.info(
        "Loaded {} test compounds from bundled {}",
        len(df),
        TEST_COMPOUNDS_FILENAME,
    )
    return df


# ---------------------------------------------------------------------------
# Per-compound Mejia weight matrix (n_genes x n_train_compounds)
# ---------------------------------------------------------------------------
#
# This matrix is too big to ship inside the wheel (~365 MB at float16 +
# brotli L6). Instead we host it as a public GitHub Release asset and
# download it lazily on first use, then cache it under ``~/.cache``.
# ``$VCPI_WEIGHTS_PATH`` short-circuits the network round-trip so tests
# / CI / contestants who already have a copy can point at it directly.

WEIGHTS_URL = (
    "https://github.com/virtualcell-vcpi/vcpi-prediction-contest-2026/"
    "releases/download/v0.1.0/weights.parquet"
)
WEIGHTS_SHA256 = "86ca750389838eb501e4bde22bc5076df4ccd2ab54d6ca1d34cdf21c69a5034e"
WEIGHTS_FILENAME = "weights.parquet"
_WEIGHTS_ENV_VAR = "VCPI_WEIGHTS_PATH"
_DOWNLOAD_CHUNK_BYTES = 1 << 20  # 1 MiB
_DOWNLOAD_CONNECT_TIMEOUT_S = 60.0
_BYTES_PER_KIB = 1024


def _weights_cache_dir() -> Path:
    """Return ``~/.cache/vcpi-prediction-contest`` honoring ``$XDG_CACHE_HOME``."""
    base = Path(os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache"))
    return base / "vcpi-prediction-contest"


def _cached_weights_path() -> Path:
    """Path the cached canonical weights file lives at (per-sha filename)."""
    return _weights_cache_dir() / f"weights-{WEIGHTS_SHA256[:16]}.parquet"


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_DOWNLOAD_CHUNK_BYTES), b""):
            h.update(chunk)
    return h.hexdigest()


def _format_bytes(n: int) -> str:
    units = ["B", "KiB", "MiB", "GiB"]
    size = float(n)
    for u in units:
        if size < _BYTES_PER_KIB or u == units[-1]:
            return f"{size:6.1f} {u}"
        size /= _BYTES_PER_KIB
    return f"{n} B"


def _stream_download(
    url: str,
    dest: Path,
    *,
    progress: bool,
) -> None:
    """Stream ``url`` to ``dest`` in chunks, optionally drawing a progress bar."""
    show_progress = progress and sys.stderr.isatty()
    with requests.get(url, stream=True, timeout=(_DOWNLOAD_CONNECT_TIMEOUT_S, None)) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length") or 0)
        downloaded = 0
        with dest.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=_DOWNLOAD_CHUNK_BYTES):
                if not chunk:
                    continue
                fh.write(chunk)
                downloaded += len(chunk)
                if show_progress:
                    if total:
                        pct = 100.0 * downloaded / total
                        msg = (
                            f"\r  downloading weights.parquet  "
                            f"{_format_bytes(downloaded)} / {_format_bytes(total)}  "
                            f"({pct:5.1f}%)"
                        )
                    else:
                        msg = f"\r  downloading weights.parquet  {_format_bytes(downloaded)}"
                    sys.stderr.write(msg)
                    sys.stderr.flush()
        if show_progress:
            sys.stderr.write("\n")
            sys.stderr.flush()


def fetch_weights(*, force: bool = False, progress: bool = True) -> Path:
    """Download (and cache) the contest's per-compound Mejia weight matrix.

    Resolution order
    ----------------
    1. If ``$VCPI_WEIGHTS_PATH`` points at an existing file, return that
       path immediately (used in tests / CI / contestants who already
       have a local copy).
    2. If the cached file at
       ``~/.cache/vcpi-prediction-contest/weights-<sha>.parquet``
       exists and matches :data:`WEIGHTS_SHA256`, return it (unless
       ``force=True``).
    3. Otherwise stream-download from :data:`WEIGHTS_URL` with
       ``requests``, write to a temp file in the cache dir, verify
       sha256, atomic-rename into place, and return the path.

    Parameters
    ----------
    force
        If ``True``, re-download even when a valid cached copy exists.
        The env-var override in step 1 still takes precedence.
    progress
        If ``True`` and stderr is a TTY, print a one-line carriage-
        return-overwritten progress indicator while streaming. Always
        silent on non-interactive stderr (e.g. CI logs).

    Returns
    -------
    pathlib.Path
        Filesystem path to the verified parquet file.

    Raises
    ------
    RuntimeError
        On network failure or a sha256 mismatch.
    """
    env_path = os.environ.get(_WEIGHTS_ENV_VAR)
    if env_path:
        candidate = Path(env_path)
        if candidate.is_file():
            logger.info("Using weights from ${} = {}", _WEIGHTS_ENV_VAR, candidate)
            return candidate
        logger.warning(
            "${} is set to {} but that file does not exist; falling back to cache/download",
            _WEIGHTS_ENV_VAR,
            candidate,
        )

    cache_dir = _weights_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = _cached_weights_path()

    if target.is_file() and not force:
        try:
            actual = _sha256_of_file(target)
        except OSError as exc:
            msg = f"Failed to read cached weights at {target}: {exc}"
            raise RuntimeError(msg) from exc
        if actual == WEIGHTS_SHA256:
            logger.info("Using cached weights at {}", target)
            return target
        logger.warning(
            "Cached weights at {} have sha256 {} (expected {}); re-downloading",
            target,
            actual,
            WEIGHTS_SHA256,
        )
        target.unlink(missing_ok=True)

    partial = target.with_suffix(target.suffix + ".partial")
    partial.unlink(missing_ok=True)

    logger.info("Downloading weights from {} -> {}", WEIGHTS_URL, target)
    try:
        _stream_download(WEIGHTS_URL, partial, progress=progress)
    except requests.RequestException as exc:
        partial.unlink(missing_ok=True)
        msg = f"Failed to download weights from {WEIGHTS_URL}: {exc}"
        raise RuntimeError(msg) from exc

    actual = _sha256_of_file(partial)
    if actual != WEIGHTS_SHA256:
        partial.unlink(missing_ok=True)
        msg = (
            f"Downloaded weights from {WEIGHTS_URL} have sha256 {actual}, "
            f"expected {WEIGHTS_SHA256}. Discarded."
        )
        raise RuntimeError(msg)

    partial.replace(target)
    logger.info("Cached weights at {} (sha256 ok)", target)
    return target


def load_weights_matrix(path: str | Path | None = None) -> pd.DataFrame:
    """Load the per-compound Mejia weight matrix.

    Parameters
    ----------
    path
        Optional path to a ``(n_genes x n_compounds)`` parquet file
        indexed by ``gene_id``. When ``None`` (default), calls
        :func:`fetch_weights` to resolve / download the canonical
        leaderboard artifact and loads that.

    Returns
    -------
    pandas.DataFrame
        ``(n_genes x n_compounds)`` weight matrix indexed by
        ``gene_id``. Columns are training-corpus ``user_compound_id``
        strings (numeric LIMS IDs) — the same key as
        ``metadata.user_compound_id`` from ``vcpi-client``, so the
        weight matrix joins directly onto the training data with no
        name <-> id bridge required.

    Notes
    -----
    The canonical artifact is stored as ``float16`` + brotli-L6 parquet
    (~365 MB). On read most numpy / pandas operations upcast ``float16``
    to ``float32``, so the returned DataFrame is effectively ``float32``
    in memory. The maximum absolute deviation introduced by the
    ``float64 -> float16`` round-trip is ~8.5e-6, well below
    leaderboard precision.
    """
    resolved = Path(path) if path is not None else fetch_weights()
    df = pd.read_parquet(resolved)
    if df.index.name != GENE_COL:
        df.index.name = GENE_COL
    logger.info(
        "Loaded weight matrix from {} (shape={}, dtype={})",
        resolved,
        df.shape,
        df.dtypes.iloc[0] if len(df.columns) else "<empty>",
    )
    return df
