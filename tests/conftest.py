"""Helpers comunes: carga del modelo con verificación previa del SHA-256 (ver src/manifest.py)."""
from __future__ import annotations

import pytest

from src import manifest


def load_verified_model(directory=manifest.MODELS_DIR):
    """``joblib.load`` del modelo, pero antes comprueba el hash contra models/manifest.json.

    Si no coincide, el test FALLA con un mensaje claro (nunca se llega a deserializar el pickle).
    """
    try:
        manifest.verify(directory)
    except ValueError as exc:
        pytest.fail(str(exc), pytrace=False)
    return manifest.load_model_verified(directory)