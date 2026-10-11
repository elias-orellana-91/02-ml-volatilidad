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


def test_required_inputs(metadata):
    ri = metadata["required_inputs"]
    assert set(ri["series"]) == {"prices", "vix", "volume"}
    assert ri["market_ticker"] == "SPY" and "SPY" in ri["tickers"]
    assert ri["min_history_days"] >= 252 + 5
    assert ri["min_history_days"] == F.min_history_days()
    assert ri["tickers"] == dict(sorted(F.TICKER_TO_SECTOR.items()))


def test_trained_on_derived_from_real_dates(metadata):
    rng = metadata["trained_on_range"]
    assert rng["start"] < rng["end"] <= "2022-12-31"
    assert rng["start"] in metadata["trained_on"] and rng["end"] in metadata["trained_on"]


def test_metadata_has_all_baselines(metadata):
    assert set(metadata["baselines_test_metrics"]) == {"Persistencia", "Logística 1 feature", "HAR-RV"}
