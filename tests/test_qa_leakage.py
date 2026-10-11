"""Tests QA independientes: leakage empírico, target a mano, métricas del modelo guardado y contrato de inferencia."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src import data as D
from src import features as F
from tests.conftest import load_verified_model

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def real():
    return D.load_prices(), D.load_vix(), D.load_volume()


def _assert_same(a: pd.DataFrame, b: pd.DataFrame):
    a, b = a.align(b, join="inner")
    np.testing.assert_allclose(a.to_numpy(float), b.to_numpy(float), rtol=1e-10, atol=1e-12, equal_nan=True)


@pytest.mark.parametrize("cut", ["2018-06-15", "2022-12-30", "2024-03-01"])
def test_perturbing_future_does_not_change_features(real, cut):
    prices, vix, volume = real
    base = F.build_features(prices, vix, volume)
    rng = np.random.default_rng(7)
    p2, v2, vol2 = prices.copy(), vix.copy(), volume.copy()
    fut = p2.index > pd.Timestamp(cut)
    p2.loc[fut] = p2.loc[fut] * rng.uniform(0.3, 3.0, (fut.sum(), p2.shape[1]))
    v2.loc[v2.index > pd.Timestamp(cut)] *= rng.uniform(0.2, 5.0, (v2.index > pd.Timestamp(cut)).sum())
    vol2.loc[vol2.index > pd.Timestamp(cut)] *= 10
    pert = F.build_features(p2, v2, vol2)
    d = base.index.get_level_values("Date") <= pd.Timestamp(cut)
    _assert_same(base[d], pert[d])


@pytest.mark.parametrize("cut", ["2019-09-30", "2023-06-01"])
def test_truncated_history_gives_same_features_at_t(real, cut):
    """Contrato de inferencia: construir con historial hasta t == construir con todo y mirar t."""
    prices, vix, volume = real
    t = pd.Timestamp(cut)
    full = F.build_features(prices, vix, volume)
    trunc = F.build_features(prices[prices.index <= t], vix[vix.index <= t], volume[volume.index <= t])
    _assert_same(full[full.index.get_level_values("Date") <= t], trunc)


def test_threshold_and_target_by_hand(real):
    prices, _, _ = real
    T = F.build_target(prices)
    r = prices.pct_change(fill_method=None)
    for tk, dt in [("AAPL", "2023-05-10"), ("XOM", "2020-03-12"), ("SPY", "2024-08-05")]:
        pos = prices.index.get_loc(pd.Timestamp(dt))
        fwd = r[tk].iloc[pos + 1 : pos + 6].std(ddof=1) * np.sqrt(252)
        # umbral: mediana de las vol 5d trailing de t-251..t
        v5 = [r[tk].iloc[j - 4 : j + 1].std(ddof=1) * np.sqrt(252) for j in range(pos - 251, pos + 1)]
        thr = np.median(v5)
        row = T.loc[(pd.Timestamp(dt), tk)]
        assert row["vol_fwd_5d"] == pytest.approx(fwd, rel=1e-9)
        assert row["threshold"] == pytest.approx(thr, rel=1e-9)
        assert row["target"] == float(fwd > thr)


def test_no_target_in_last_5_days(real):
    prices, _, _ = real
    T = F.build_target(prices)
    last5 = prices.index[-5:]
    assert T.loc[T.index.get_level_values("Date").isin(last5), "target"].isna().all()


def test_saved_model_metrics_independent(real):
    prices, vix, volume = real
    meta = json.loads((ROOT / "models/metadata.json").read_text(encoding="utf-8"))
    model = load_verified_model()
    X = F.build_features(prices, vix, volume)[meta["features"]]
    # target recalculado a mano (sin build_target)
    r = prices.pct_change(fill_method=None)
    fwd = r.rolling(5).std().shift(-5) * np.sqrt(252)
    thr = (r.rolling(5).std() * np.sqrt(252)).rolling(252).median()
    y = (fwd > thr).where(fwd.notna() & thr.notna()).stack(future_stack=True)
    y.index.names = ["Date", "Ticker"]
    df = X.join(y.rename("y")).dropna()
    te = df[df.index.get_level_values("Date") > "2022-12-31"]
    p = model.predict_proba(te[meta["features"]])[:, 1]
    yt = te["y"].astype(int).to_numpy()
    m = meta["test_metrics"]
    assert len(yt) == m["n"]
    assert roc_auc_score(yt, p) == pytest.approx(m["roc_auc"], abs=1e-6)
    assert average_precision_score(yt, p) == pytest.approx(m["pr_auc"], abs=1e-6)
    assert brier_score_loss(yt, p) == pytest.approx(m["brier"], abs=1e-6)


def test_inference_on_latest_date(real):
    prices, vix, volume = real
    meta = json.loads((ROOT / "models/metadata.json").read_text(encoding="utf-8"))
    model = load_verified_model()
    X = F.build_features(prices, vix, volume)
    last = X.xs(prices.index[-1], level="Date")[meta["features"]]
    assert not last.isna().any().any()
    p = model.predict_proba(last)[:, 1]
    assert len(p) == prices.shape[1] and ((0 <= p) & (p <= 1)).all()
