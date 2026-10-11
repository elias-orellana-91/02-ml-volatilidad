"""Features y target para predecir volatilidad alta a 5 días.

Garantía anti-leakage: cada feature en la fecha t usa solo información <= t
(ventanas hacia atrás, sin ``shift(-k)``). Solo el *target* mira al futuro, y
vive en ``build_target``.

``build_features`` está pensada para reutilizarse en inferencia (proyecto 3):
se le pasa el historial de precios hasta la fecha t y se toma la última fila
(``features.xs(t, level="Date")``).

Definición del target
---------------------
* ``vol_5d_trailing(t)``: desv. estándar de los retornos de t-4..t, anualizada (x sqrt(252)).
* ``threshold(t)``: mediana de ``vol_5d_trailing`` en los últimos 252 días (t-251..t) del
  mismo ticker. Solo usa datos <= t.
* ``vol_fwd_5d(t)``: desv. estándar de los retornos de t+1..t+5, anualizada.
* ``target(t) = 1`` si ``vol_fwd_5d(t) > threshold(t)``, si no 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import TICKER_TO_SECTOR

HORIZON = 5
THRESHOLD_WINDOW = 252
ANN = np.sqrt(252)

SECTOR_SLUGS = {
    "Tecnología": "tecnologia",
    "Bancos": "bancos",
    "Energía": "energia",
    "Salud": "salud",
    "Consumo": "consumo",
    "Índice": "indice",
}
SECTOR_COLUMNS = [f"sector_{s}" for s in SECTOR_SLUGS.values()]


def _roll_vol(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    return returns.rolling(window).std() * ANN


def vol_threshold(prices: pd.DataFrame) -> pd.DataFrame:
    """Umbral por ticker y fecha: mediana móvil (252d) de la vol 5d *trailing*."""
    rets = prices.pct_change(fill_method=None)
    vol5 = _roll_vol(rets, HORIZON)
    return vol5.rolling(THRESHOLD_WINDOW, min_periods=THRESHOLD_WINDOW).median()


def build_features(
    prices: pd.DataFrame,
    vix: pd.Series | None = None,
    volume: pd.DataFrame | None = None,
    market: str = "SPY",
) -> pd.DataFrame:
    """Construye las features en formato largo, índice ``(Date, Ticker)``.

    Parameters
    ----------
    prices : precios ajustados (Date x Ticker). Debe incluir la columna ``market`` (SPY).
    vix : cierre diario del VIX (opcional; añade ``vix``, ``vix_chg_5``, ``vix_vs_ma63``).
    volume : volumen diario (Date x Ticker, opcional; añade ``volume_ratio_5_63``).

    Las primeras ~252 filas de cada ticker tienen NaN (calentamiento de la mediana de 252d);
    ``dropna()`` las elimina. Tickers sin sector conocido reciben ceros en el one-hot.
    """
    if market not in prices.columns:
        raise ValueError(f"prices debe incluir la columna de mercado {market!r}")
    prices = prices.sort_index()
    rets = prices.pct_change(fill_method=None)

    vol5, vol21, vol63 = (_roll_vol(rets, w) for w in (5, 21, 63))
    thr = vol_threshold(prices)

    wide: dict[str, pd.DataFrame] = {
        "vol_5": vol5,
        "vol_21": vol21,
        "vol_63": vol63,
        "vol_ratio_5_63": vol5 / vol63,
        "vol_ratio_21_63": vol21 / vol63,
        "vol5_vs_med252": vol5 / thr,
        "ret_1": prices.pct_change(1, fill_method=None),
        "ret_5": prices.pct_change(5, fill_method=None),
        "ret_21": prices.pct_change(21, fill_method=None),
        "abs_ret_1": rets.abs(),
        "dd_63": prices / prices.rolling(63).max() - 1,
    }
    # Volatilidad del mercado (SPY) replicada para cada ticker
    for name, v in (("spy_vol_5", vol5[market]), ("spy_vol_21", vol21[market])):
        wide[name] = pd.DataFrame({c: v for c in prices.columns})
    if vix is not None:
        v = vix.reindex(prices.index).ffill()
        for name, s in (
            ("vix", v),
            ("vix_chg_5", v / v.shift(5) - 1),
            ("vix_vs_ma63", v / v.rolling(63).mean()),
        ):
            wide[name] = pd.DataFrame({c: s for c in prices.columns})
    if volume is not None:
        vol_df = volume.reindex(index=prices.index, columns=prices.columns)
        wide["volume_ratio_5_63"] = vol_df.rolling(5).mean() / vol_df.rolling(63).mean()
    wide["dow"] = pd.DataFrame(
        {c: pd.Series(prices.index.dayofweek, index=prices.index) for c in prices.columns}
    ).astype(float)

    feats = pd.concat(wide, axis=1).stack(level=1, future_stack=True)
    feats.index.names = ["Date", "Ticker"]
    feats = feats.replace([np.inf, -np.inf], np.nan).sort_index()

    sector = feats.index.get_level_values("Ticker").map(TICKER_TO_SECTOR)
    for name, slug in SECTOR_SLUGS.items():
        feats[f"sector_{slug}"] = (sector == name).astype(float)
    return feats


def build_target(prices: pd.DataFrame) -> pd.DataFrame:
    """Target y columnas auxiliares en formato largo ``(Date, Ticker)``.

    Columnas: ``vol_fwd_5d``, ``threshold``, ``target`` (NaN si no se puede calcular:
    calentamiento inicial o últimos 5 días sin futuro).
    """
    prices = prices.sort_index()
    rets = prices.pct_change(fill_method=None)
    vol_fwd = _roll_vol(rets, HORIZON).shift(-HORIZON)  # std de t+1..t+5
    thr = vol_threshold(prices)
    out = pd.concat({"vol_fwd_5d": vol_fwd, "threshold": thr}, axis=1).stack(
        level=1, future_stack=True
    )
    out.index.names = ["Date", "Ticker"]
    out = out.sort_index()
    valid = out["vol_fwd_5d"].notna() & out["threshold"].notna()
    out["target"] = np.where(valid, (out["vol_fwd_5d"] > out["threshold"]).astype(float), np.nan)
    return out


def make_dataset(
    prices: pd.DataFrame,
    vix: pd.Series | None = None,
    volume: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Devuelve ``(X, y)`` sin NaN, alineados por ``(Date, Ticker)``."""
    X = build_features(prices, vix, volume)
    T = build_target(prices)
    df = X.join(T["target"]).dropna()
    return df.drop(columns="target"), df["target"].astype(int)
