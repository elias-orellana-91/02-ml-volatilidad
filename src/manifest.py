"""Manifiesto SHA-256 de los artefactos del modelo (mismo formato que ``app/ml/artifacts/manifest.json`` del proyecto 3).

``joblib`` usa pickle, que puede ejecutar código arbitrario al cargarse: antes de ``joblib.load`` se comprueba
el hash contra ``models/manifest.json``.

Tras reentrenar: ``python -m src.manifest --write`` para regenerar el manifiesto.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
FILES = ("model.joblib", "metadata.json")
MANIFEST = "manifest.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(directory: Path = MODELS_DIR) -> dict:
    manifest = {"algorithm": "sha256", "files": {name: sha256_file(directory / name) for name in FILES}}
    # newline="\n": en Windows write_text escribiría CRLF; los artefactos van con -text (.gitattributes)
    with (directory / MANIFEST).open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify(directory: Path = MODELS_DIR) -> None:
    """Lanza ``ValueError`` con un mensaje claro si falta el manifiesto o algún hash no coincide."""
    manifest_path = directory / MANIFEST
    if not manifest_path.exists():
        raise ValueError(f"Falta {manifest_path}; genera el manifiesto con: python -m src.manifest --write")
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))["files"]
    bad = [name for name in FILES if sha256_file(directory / name) != expected.get(name)]
    if bad:
        raise ValueError(
            f"El SHA-256 de {', '.join(bad)} no coincide con {MANIFEST}: el artefacto fue modificado o se reentrenó "
            "sin regenerar el manifiesto (python -m src.manifest --write). No se carga con joblib."
        )


def load_model_verified(directory: Path = MODELS_DIR):
    """Verifica el hash y SOLO entonces carga ``model.joblib``."""
    import joblib

    verify(directory)
    return joblib.load(directory / "model.joblib")


if __name__ == "__main__":
    if "--write" in sys.argv:
        print(json.dumps(write_manifest(), indent=2))
    else:
        verify()
        print("OK: los hashes coinciden con", MODELS_DIR / MANIFEST)