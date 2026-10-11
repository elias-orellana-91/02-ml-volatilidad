"""Tests QA ronda 2: HAR-RV sin leakage, historial mínimo, ticker desconocido, metadata autosuficiente."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src import data as D
from src import features as F
from src.baselines import HARRV, HAR_COLUMNS

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def real():
    return D.load_prices(), D.load_vix(), D.load_volume()


def test_har_inputs_no_future_and_by_hand(real):
    prices, _, _ = real
    base = F.build_har_inputs(prices)
    cut = pd.Timestamp("2021-06-30")
    p2 = prices.copy()
    p2.loc[p2.index > cut] *= np.random.default_rng(3).uniform(0.4, 2.5, (int((p2.index > cut).sum()), p2.shape[1]))
    pert = F.build_har_inputs(p2)
    d = base.index.get_level_values("Date") <= cut
    np.testing.assert_allclose(base[d].to_numpy(float), pert[d].to_numpy(float), equal_nan=True, rtol=1e-10)
    # a mano: AAPL en 2023-05-10
    r = prices["AAPL"].pct_change(fill_method=None)
    pos = prices.index.get_loc(pd.Timestamp("2023-05-10"))
    row = base.loc[(pd.Timestamp("2023-05-10"), "AAPL")]
    assert row["har_d"] == pytest.approx(abs(r.iloc[pos]) * np.sqrt(252))
    assert row["har_w"] == pytest.approx(r.iloc[pos - 4 : pos + 1].std() * np.sqrt(252))
    assert row["har_m"] == pytest.approx(r.iloc[pos - 21 : pos + 1].std() * np.sqrt(252))


def test_har_fit_by_hand_matches_ols(real):
    prices, vix, volume = real
    X, y = F.make_dataset(prices, vix, volume)
    har = F.build_har_inputs(prices).reindex(X.index)
    tgt = F.build_target(prices).reindex(X.index)
    tr = np.asarray(X.index.get_level_values("Date") <= "2021-12-20")
    m = HARRV().fit(har[tr], tgt.loc[tr, "vol_fwd_5d"])
    A = np.c_[har[tr][HAR_COLUMNS].to_numpy(), np.ones(tr.sum())]
    beta = np.linalg.lstsq(A, tgt.loc[tr, "vol_fwd_5d"].to_numpy(), rcond=None)[0]
    np.testing.assert_allclose([*m.model.coef_, m.model.intercept_], beta, rtol=1e-6)


def test_min_history_is_exact():
    n = F.min_history_days()
    assert n == 257
    rng = np.random.default_rng(0)

    def last_has_nan(rows):
        idx = pd.bdate_range("2019-01-01", periods=rows)
        px = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, .01, (rows, 2)), axis=0), index=idx, columns=["SPY", "AAPL"])
        X = F.build_features(px, pd.Series(20 + rng.random(rows), index=idx),
                             pd.DataFrame(1e6 * (1 + rng.random((rows, 2))), index=idx, columns=px.columns))
        return X.xs(idx[-1], level="Date").loc["AAPL"].isna().any()

    assert not last_has_nan(n)
    assert last_has_nan(n - 1)


def test_unknown_ticker_raises():
    idx = pd.bdate_range("2020-01-01", periods=300)
    px = pd.DataFrame(100.0, index=idx, columns=["SPY", "ZZZZ"]) + np.arange(300)[:, None]
    with pytest.raises(ValueError, match="ZZZZ"):
        F.build_features(px)


def test_metadata_self_sufficient_and_consistent():
    meta = json.loads((ROOT / "models/metadata.json").read_text(encoding="utf-8"))
    ri = meta["required_inputs"]
    assert ri["min_history_days"] == F.min_history_days()
    assert ri["market_ticker"] == "SPY"
    assert set(ri["series"]) == {"prices", "vix", "volume"}
    assert ri["tickers"] == dict(sorted(F.TICKER_TO_SECTOR.items()))
    assert len(meta["features"]) == 24
    # trained_on coincide con los datos reales
    prices, vix, volume = D.load_prices(), D.load_vix(), D.load_volume()
    X, _ = F.make_dataset(prices, vix, volume)
    d = X.index.get_level_values("Date")
    assert meta["trained_on_range"]["start"] == f"{d.min():%Y-%m-%d}"
    assert pd.Timestamp(meta["trained_on_range"]["end"]) <= pd.Timestamp("2022-12-31")


def test_inference_with_exactly_min_history(real):
    prices, vix, volume = real
    meta = json.loads((ROOT / "models/metadata.json").read_text(encoding="utf-8"))
    model = joblib.load(ROOT / "models/model.joblib")
    n = meta["required_inputs"]["min_history_days"]
    p = prices.iloc[-n:]
    X = F.build_features(p, vix.reindex(p.index), volume.reindex(p.index))
    last = X.xs(p.index[-1], level="Date")[meta["features"]]
    assert not last.isna().any().any()
    full = F.build_features(prices, vix, volume).xs(prices.index[-1], level="Date")[meta["features"]]
    np.testing.assert_allclose(last.to_numpy(), full.to_numpy(), rtol=1e-9)
    assert np.isfinite(model.predict_proba(last)).all()
