"""Tests de features y target con datos sintéticos (sin red)."""
import numpy as np
import pandas as pd
import pytest

from src import features as F
from src.train import GAP, date_splits, temporal_split

TICKERS = ["SPY", "AAPL", "JPM", "XOM"]


@pytest.fixture(scope="module")
def market():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2018-01-01", periods=700)
    rets = rng.normal(0, 0.012, (len(idx), len(TICKERS)))
    prices = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=TICKERS)
    vix = pd.Series(15 + 5 * np.abs(rng.normal(size=len(idx))), index=idx, name="VIX")
    volume = pd.DataFrame(rng.integers(1e6, 5e6, (len(idx), len(TICKERS))).astype(float), index=idx, columns=TICKERS)
    return prices, vix, volume


def test_no_leakage_future_data_does_not_change_features(market):
    prices, vix, volume = market
    t = prices.index[500]
    base = F.build_features(prices, vix, volume)

    p2, v2, vol2 = prices.copy(), vix.copy(), volume.copy()
    after = prices.index > t
    p2[after] *= np.random.default_rng(1).uniform(0.5, 2.0, (after.sum(), len(TICKERS)))
    v2[after] = 99.0
    vol2[after] = 1.0
    changed = F.build_features(p2, v2, vol2)

    cols = base.columns
    a = base.loc[base.index.get_level_values("Date") <= t, cols]
    b = changed.loc[changed.index.get_level_values("Date") <= t, cols]
    pd.testing.assert_frame_equal(a, b)


def test_features_equal_when_truncating_history(market):
    """Inferencia: calcular con historial hasta t da la misma fila en t que con todo el histórico."""
    prices, vix, volume = market
    t = prices.index[400]
    full = F.build_features(prices, vix, volume).xs(t, level="Date")
    cut = F.build_features(prices.loc[:t], vix.loc[:t], volume.loc[:t]).xs(t, level="Date")
    pd.testing.assert_frame_equal(full, cut)


def test_target_on_synthetic_data():
    # Vol constante baja y luego un salto: el target debe activarse justo antes del salto
    idx = pd.bdate_range("2019-01-01", periods=400)
    rets = np.full(len(idx), 0.001) * np.where(np.arange(len(idx)) % 2 == 0, 1, -1)
    jump_day = 320
    rets[jump_day : jump_day + 5] = [0.05, -0.05, 0.05, -0.05, 0.05]
    prices = pd.DataFrame({"SPY": 100 * np.cumprod(1 + rets), "AAA": 100 * np.cumprod(1 + rets)}, index=idx)
    T = F.build_target(prices).xs("AAA", level="Ticker")

    assert T["target"].iloc[: F.THRESHOLD_WINDOW - 1].isna().all()  # calentamiento
    assert T["target"].iloc[-F.HORIZON :].isna().all()  # sin futuro
    t = idx[jump_day - 1]  # los retornos t+1..t+5 son el salto
    assert T.loc[t, "target"] == 1
    assert T.loc[idx[290], "target"] == 0  # régimen calmo
    expected = pd.Series(rets[jump_day : jump_day + 5]).std() * np.sqrt(252)
    assert T.loc[t, "vol_fwd_5d"] == pytest.approx(expected)


def test_threshold_uses_only_past(market):
    prices, _, _ = market
    t = prices.index[450]
    thr = F.vol_threshold(prices)
    p2 = prices.copy()
    p2[p2.index > t] *= 3.0
    pd.testing.assert_frame_equal(thr.loc[:t], F.vol_threshold(p2).loc[:t])


def test_shapes_and_no_nan_after_warmup(market):
    prices, vix, volume = market
    X, y = F.make_dataset(prices, vix, volume)
    assert len(X) == len(y) > 0
    assert X.index.equals(y.index)
    assert not X.isna().any().any()
    assert set(y.unique()) <= {0, 1}
    assert X.index.names == ["Date", "Ticker"]
    for c in F.SECTOR_COLUMNS:
        assert c in X.columns
    # Calentamiento y horizonte: se pierden >= 252 días al inicio y 5 al final
    dates = X.index.get_level_values("Date").unique()
    assert dates.min() >= prices.index[F.THRESHOLD_WINDOW - 1]
    assert dates.max() <= prices.index[-F.HORIZON - 1]


def test_sector_one_hot(market):
    prices, vix, volume = market
    X = F.build_features(prices, vix, volume)
    row = X.xs("JPM", level="Ticker").iloc[-1]
    assert row["sector_bancos"] == 1 and row[F.SECTOR_COLUMNS].sum() == 1


def test_requires_market_column(market):
    prices, _, _ = market
    with pytest.raises(ValueError):
        F.build_features(prices.drop(columns="SPY"))


def test_optional_inputs_omit_columns(market):
    prices, _, _ = market
    X = F.build_features(prices)
    assert "vix" not in X.columns and "volume_ratio_5_63" not in X.columns


def test_temporal_split_has_no_overlap_and_purges(market):
    # Dataset sintético largo que cruza 2021-2023
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2016-01-01", "2024-06-30")
    prices = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.01, (len(idx), 2)), axis=0), index=idx, columns=["SPY", "AAPL"])
    X, _ = F.make_dataset(prices)
    sp = temporal_split(X)
    d = X.index.get_level_values("Date")
    assert not (sp["train"] & sp["val"]).any() and not (sp["val"] & sp["test"]).any()
    assert d[sp["train"]].max() < d[sp["val"]].min() < d[sp["test"]].min()
    # El gap entre train y val es de al menos GAP días hábiles
    gap_days = len(idx[(idx > d[sp["train"]].max()) & (idx <= d[sp["val"]].min())])
    assert gap_days >= GAP


def test_date_splits_respect_gap_and_order(market):
    prices, vix, volume = market
    X, _ = F.make_dataset(prices, vix, volume)
    d = X.index.get_level_values("Date")
    for tr, te in date_splits(d, n_splits=3, gap=GAP):
        assert d[tr].max() < d[te].min()
        between = d.unique()[(d.unique() > d[tr].max()) & (d.unique() < d[te].min())]
        assert len(between) >= GAP


def test_unknown_ticker_raises_clear_error(market):
    prices, vix, volume = market
    bad = prices.rename(columns={"XOM": "ZZZZ"})
    with pytest.raises(ValueError, match="ZZZZ"):
        F.build_features(bad, vix, volume)


def test_min_history_days_is_tight():
    n = F.min_history_days()
    assert n >= F.THRESHOLD_WINDOW + F.HORIZON
    rng = np.random.default_rng(5)

    def last_row_has_nan(rows):
        idx = pd.bdate_range("2010-01-04", periods=rows)
        px = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.01, (rows, 2)), axis=0), index=idx, columns=["SPY", "AAPL"])
        X = F.build_features(px, pd.Series(20.0, index=idx), pd.DataFrame(1e6, index=idx, columns=px.columns))
        return X.xs(idx[-1], level="Date").isna().any().any()

    assert not last_row_has_nan(n)
    assert last_row_has_nan(n - 1)


def test_har_inputs_no_leakage(market):
    prices, _, _ = market
    t = prices.index[450]
    base = F.build_har_inputs(prices)
    p2 = prices.copy()
    p2[p2.index > t] *= 4.0
    changed = F.build_har_inputs(p2)
    d = base.index.get_level_values("Date") <= t
    pd.testing.assert_frame_equal(base[d], changed[d])
