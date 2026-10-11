"""Tests del artefacto del modelo (sin red)."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src import features as F

MODELS = Path(__file__).resolve().parents[1] / "models"


@pytest.fixture(scope="module")
def metadata():
    return json.loads((MODELS / "metadata.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def model():
    return joblib.load(MODELS / "model.joblib")


def test_metadata_fields(metadata):
    for key in ("model_name", "model_version", "trained_at", "features", "decision_threshold", "test_metrics"):
        assert key in metadata
    assert 0 < metadata["decision_threshold"] < 1
    assert len(metadata["features"]) == len(set(metadata["features"]))


def test_predict_on_synthetic_row(model, metadata):
    cols = metadata["features"]
    row = pd.DataFrame([dict.fromkeys(cols, 0.0)])[cols]
    row["vol5_vs_med252"] = 1.0
    row["sector_tecnologia"] = 1.0
    proba = model.predict_proba(row)
    assert proba.shape == (1, 2)
    assert 0.0 <= proba[0, 1] <= 1.0
    assert np.isfinite(proba).all()


def test_model_features_match_build_features(metadata):
    """Las columnas del modelo deben poder generarse con build_features (contrato con la API)."""
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=400)
    prices = pd.DataFrame(
        100 * np.cumprod(1 + rng.normal(0, 0.01, (len(idx), 2)), axis=0), index=idx, columns=["SPY", "AAPL"]
    )
    vix = pd.Series(18.0, index=idx)
    volume = pd.DataFrame(2e6, index=idx, columns=["SPY", "AAPL"])
    X = F.build_features(prices, vix, volume).dropna()
    assert set(metadata["features"]) <= set(X.columns)


def test_predict_proba_end_to_end(model, metadata):
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2020-01-01", periods=400)
    prices = pd.DataFrame(
        100 * np.cumprod(1 + rng.normal(0, 0.01, (len(idx), 2)), axis=0), index=idx, columns=["SPY", "AAPL"]
    )
    X = F.build_features(prices, pd.Series(18.0, index=idx), pd.DataFrame(2e6, index=idx, columns=["SPY", "AAPL"]))
    last = X.xs("AAPL", level="Ticker").iloc[[-1]][metadata["features"]]
    p = model.predict_proba(last)[0, 1]
    assert 0 <= p <= 1
