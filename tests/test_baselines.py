"""Tests de los baselines (sin red)."""
import numpy as np
import pandas as pd
import pytest

from src.baselines import HAR_COLUMNS, HARRV, single_feature_logit


def test_har_recovers_known_coefficients():
    rng = np.random.default_rng(0)
    har = pd.DataFrame(rng.uniform(0.1, 0.5, (2000, 3)), columns=HAR_COLUMNS)
    vol = 0.02 + 0.2 * har["har_d"] + 0.3 * har["har_w"] + 0.4 * har["har_m"]
    m = HARRV().fit(har, vol)
    c = m.coefficients
    assert c["har_d"] == pytest.approx(0.2) and c["har_w"] == pytest.approx(0.3)
    assert c["har_m"] == pytest.approx(0.4) and c["intercept"] == pytest.approx(0.02)


def test_har_score_is_vol_over_threshold():
    har = pd.DataFrame({"har_d": [0.2], "har_w": [0.2], "har_m": [0.2]})
    m = HARRV().fit(pd.concat([har] * 5 + [har * 2] * 5, ignore_index=True), pd.Series([0.2] * 5 + [0.4] * 5))
    thr = pd.Series([0.2])
    assert m.score(har, thr)[0] == pytest.approx(1.0)  # vol predicha == umbral


def test_single_feature_logit_uses_only_one_column():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"vol5_vs_med252": rng.lognormal(0, 0.4, 500), "otra": rng.normal(size=500)})
    y = (X["vol5_vs_med252"] + rng.normal(0, 0.3, 500) > 1).astype(int)
    m = single_feature_logit().fit(X, y)
    p1 = m.predict_proba(X)[:, 1]
    p2 = m.predict_proba(X.assign(otra=999.0))[:, 1]  # cambiar otras columnas no afecta
    np.testing.assert_allclose(p1, p2)
    assert pd.Series(p1).corr(X["vol5_vs_med252"], method="spearman") > 0.99  # monótona

