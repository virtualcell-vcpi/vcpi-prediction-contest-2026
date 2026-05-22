"""Tests for ``fetch_weights`` and ``load_weights_matrix``.

These tests never touch the real network. They exercise the four
resolution paths of :func:`fetch_weights` (env-var override, valid
cache hit, fresh download, checksum mismatch) by mocking
``requests.get`` and overriding the cache dir via ``monkeypatch``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Self

import numpy as np
import pandas as pd
import pytest
import requests

from vcpi_prediction_contest import data as data_mod
from vcpi_prediction_contest.data import fetch_weights, load_weights_matrix

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_weights_frame(n_genes: int = 4, n_compounds: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        rng.random((n_genes, n_compounds), dtype=np.float64).astype(np.float16),
        index=pd.Index([f"g{i}" for i in range(n_genes)], name="gene_id"),
        columns=[f"c{j}" for j in range(n_compounds)],
    )
    return df


def _write_tiny_weights_parquet(path: Path) -> bytes:
    """Write a small synthetic weights parquet at ``path`` and return its bytes."""
    df = _tiny_weights_frame()
    df.to_parquet(path, compression="brotli", compression_level=6)
    return path.read_bytes()


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class _MockResponse:
    """Minimal stand-in for ``requests.Response`` that supports the
    streaming context-manager + ``iter_content`` API our code uses.
    """

    def __init__(self, body: bytes, *, status: int = 200) -> None:
        self._body = body
        self.status_code = status
        self.headers = {"content-length": str(len(body))}

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            msg = f"status {self.status_code}"
            raise requests.HTTPError(msg)

    def iter_content(self, chunk_size: int = 1024) -> Any:  # noqa: ANN401
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i : i + chunk_size]


@pytest.fixture
def isolated_cache_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Redirect the on-disk cache to ``tmp_path`` so we never touch ``~/.cache``."""
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr(data_mod, "_weights_cache_dir", lambda: cache)
    monkeypatch.delenv(data_mod._WEIGHTS_ENV_VAR, raising=False)
    return cache


# ---------------------------------------------------------------------------
# fetch_weights
# ---------------------------------------------------------------------------


def test_fetch_weights_uses_env_var_path(
    monkeypatch: pytest.MonkeyPatch,
    isolated_cache_dir: Path,
    tmp_path: Path,
) -> None:
    """If ``$VCPI_WEIGHTS_PATH`` points at an existing file, use it directly."""
    target = tmp_path / "local-weights.parquet"
    _write_tiny_weights_parquet(target)
    monkeypatch.setenv(data_mod._WEIGHTS_ENV_VAR, str(target))

    def _fail(*_args: object, **_kw: object) -> None:
        msg = "network must not be touched when $VCPI_WEIGHTS_PATH is set"
        raise AssertionError(msg)

    monkeypatch.setattr(requests, "get", _fail)

    result = fetch_weights()
    assert result == target
    assert result.exists()


def test_fetch_weights_uses_cache_when_present(
    monkeypatch: pytest.MonkeyPatch,
    isolated_cache_dir: Path,
) -> None:
    """A valid cached file with the right sha256 is returned without download."""
    body = _write_tiny_weights_parquet(isolated_cache_dir / "tmp.parquet")
    sha = _sha256_bytes(body)
    monkeypatch.setattr(data_mod, "WEIGHTS_SHA256", sha)
    cached = isolated_cache_dir / f"weights-{sha[:16]}.parquet"
    cached.write_bytes(body)

    def _fail(*_args: object, **_kw: object) -> None:
        msg = "network must not be touched when cache is valid"
        raise AssertionError(msg)

    monkeypatch.setattr(requests, "get", _fail)

    result = fetch_weights()
    assert result == cached


def test_fetch_weights_downloads_when_missing(
    monkeypatch: pytest.MonkeyPatch,
    isolated_cache_dir: Path,
    tmp_path: Path,
) -> None:
    """Missing cache -> stream-download via ``requests.get`` and cache."""
    body = _write_tiny_weights_parquet(tmp_path / "synthetic.parquet")
    sha = _sha256_bytes(body)
    monkeypatch.setattr(data_mod, "WEIGHTS_SHA256", sha)

    calls: list[str] = []

    def _fake_get(url: str, *_args: object, **_kw: object) -> _MockResponse:
        calls.append(url)
        return _MockResponse(body)

    monkeypatch.setattr(requests, "get", _fake_get)

    result = fetch_weights(progress=False)
    expected = isolated_cache_dir / f"weights-{sha[:16]}.parquet"
    assert result == expected
    assert result.exists()
    assert _sha256_bytes(result.read_bytes()) == sha
    assert calls == [data_mod.WEIGHTS_URL]
    assert not (expected.with_suffix(expected.suffix + ".partial")).exists()


def test_fetch_weights_raises_on_checksum_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    isolated_cache_dir: Path,
    tmp_path: Path,
) -> None:
    """Wrong checksum -> RuntimeError, partial file cleaned up, no cache file."""
    body = _write_tiny_weights_parquet(tmp_path / "synthetic.parquet")
    bogus_sha = "0" * 64
    monkeypatch.setattr(data_mod, "WEIGHTS_SHA256", bogus_sha)

    def _fake_get(_url: str, *_args: object, **_kw: object) -> _MockResponse:
        return _MockResponse(body)

    monkeypatch.setattr(requests, "get", _fake_get)

    with pytest.raises(RuntimeError, match="sha256"):
        fetch_weights(progress=False)

    expected = isolated_cache_dir / f"weights-{bogus_sha[:16]}.parquet"
    assert not expected.exists()
    assert not expected.with_suffix(expected.suffix + ".partial").exists()


# ---------------------------------------------------------------------------
# load_weights_matrix
# ---------------------------------------------------------------------------


def test_load_weights_matrix_round_trip(tmp_path: Path) -> None:
    """A small synthetic frame round-trips through ``load_weights_matrix``."""
    src = _tiny_weights_frame(n_genes=5, n_compounds=4)
    p = tmp_path / "w.parquet"
    src.to_parquet(p, compression="brotli", compression_level=6)

    out = load_weights_matrix(p)
    assert out.shape == src.shape
    assert list(out.index) == list(src.index)
    assert list(out.columns) == list(src.columns)
    assert out.index.name == "gene_id"
    np.testing.assert_allclose(
        out.to_numpy().astype("float64"),
        src.to_numpy().astype("float64"),
        atol=1e-6,
    )


def test_load_weights_matrix_calls_fetch_when_no_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``path=None`` -> calls ``fetch_weights`` and reads what it returns."""
    p = tmp_path / "fetched.parquet"
    src = _tiny_weights_frame()
    src.to_parquet(p)

    calls = {"n": 0}

    def _fake_fetch(**_kw: object) -> Path:
        calls["n"] += 1
        return p

    monkeypatch.setattr(data_mod, "fetch_weights", _fake_fetch)

    out = load_weights_matrix()
    assert calls["n"] == 1
    assert out.shape == src.shape
    assert list(out.index) == list(src.index)
