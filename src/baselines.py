"""Baselines de referencia más exigentes que la persistencia simple.

* ``HARRV``: modelo HAR-RV (Corsi, 2009). Regresión lineal de la volatilidad futura sobre la
  volatilidad realizada diaria, semanal (5d) y mensual (22d). Con solo precios diarios, la
  "vol diaria" es |retorno| anualizado. Se ajusta con OLS agrupado (todos los tickers).
  Su predicción se convierte en *score* de clasificación dividiéndola por el mismo umbral del
  target: ``score > 1`` equivale a "se predice vol futura por encima del umbral".
* ``single_feature_logit``: regresión logística con solo ``vol5_vs_med252``, el equivalente
  entrenado de la persistencia.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HAR_COLUMNS = ["har_d", "har_w", "har_m"]
SINGLE_FEATURE = "vol5_vs_med252"


class HARRV:
    """HAR-RV agrupado: ``vol_fwd_5d ~ har_d + har_w + har_m`` (OLS)."""

    def __init__(self) -> None:
        self.model = LinearRegression()

    def fit(self, har: pd.DataFrame, vol_fwd: pd.Series) -> "HARRV":
        self.model.fit(har[HAR_COLUMNS], vol_fwd)
        return self

    def predict_vol(self, har: pd.DataFrame) -> np.ndarray:
        return self.model.predict(har[HAR_COLUMNS])

    def score(self, har: pd.DataFrame, threshold: pd.Series) -> np.ndarray:
        """Vol futura predicha / umbral del ticker (> 1 => clase positiva)."""
        return self.predict_vol(har) / threshold.to_numpy()

    @property
    def coefficients(self) -> dict[str, float]:
        out = dict(zip(HAR_COLUMNS, map(float, self.model.coef_)))
        out["intercept"] = float(self.model.intercept_)
        return out


def single_feature_logit() -> Pipeline:
    """Logística con una sola feature (escalada), entrenable como el resto de modelos."""
    return Pipeline(
        [
            ("select", ColumnTransformer([("f", StandardScaler(), [SINGLE_FEATURE])])),
            ("clf", LogisticRegression(max_iter=3000)),
        ]
    )
