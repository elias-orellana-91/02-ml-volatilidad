"""El modelo solo se deserializa (pickle) si su SHA-256 coincide con models/manifest.json."""
from __future__ import annotations

import shutil

import pytest

from src import manifest
from tests.conftest import load_verified_model


def test_committed_manifest_matches_artifacts():
    manifest.verify()  # no lanza
    assert load_verified_model() is not None


def test_tampered_model_fails_with_clear_message_before_loading(tmp_path, monkeypatch):
    for name in (*manifest.FILES, manifest.MANIFEST):
        shutil.copy(manifest.MODELS_DIR / name, tmp_path / name)
    (tmp_path / "model.joblib").write_bytes((tmp_path / "model.joblib").read_bytes() + b"\x00")  # archivo alterado

    import joblib

    def must_not_load(*a, **k):
        raise AssertionError("joblib.load no debe llamarse con un hash incorrecto")

    monkeypatch.setattr(joblib, "load", must_not_load)
    with pytest.raises(pytest.fail.Exception, match=r"model\.joblib.*no coincide"):
        load_verified_model(tmp_path)


def test_tampered_metadata_and_missing_manifest(tmp_path):
    for name in (*manifest.FILES, manifest.MANIFEST):
        shutil.copy(manifest.MODELS_DIR / name, tmp_path / name)
    (tmp_path / "metadata.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match=r"metadata\.json"):
        manifest.verify(tmp_path)
    (tmp_path / manifest.MANIFEST).unlink()
    with pytest.raises(ValueError, match="Falta"):
        manifest.verify(tmp_path)