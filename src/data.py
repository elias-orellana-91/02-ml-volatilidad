"""Descarga de precios y métricas financieras básicas.

Módulo pensado para reutilizarse en otros proyectos del portafolio:
las funciones de métricas son puras (no tocan disco ni red) y operan
sobre ``pandas.Series`` / ``pandas.DataFrame`` con índice de fechas.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TRADING_DAYS = 252
_MIN_VOL = 1e-12  # por debajo se considera volatilidad nula

SECTORS: dict[str, list[str]] = {
    "Tecnología": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"],
    "Bancos": ["JPM", "BAC", "GS"],
    "Energía": ["XOM", "CVX"],
    "Salud": ["JNJ", "UNH", "PFE"],
    "Consumo": ["KO", "WMT"],
    "Índice": ["SPY"],
}
TICKERS: list[str] = [t for ts in SECTORS.values() for t in ts]
TICKER_TO_SECTOR: dict[str, str] = {t: s for s, ts in SECTORS.items() for t in ts}

START_DATE = "2015-01-01"
END_DATE = "2025-12-31"
DEFAULT_CACHE = Path(__file__).resolve().parents[1] / "data" / "prices.parquet"


# --------------------------------------------------------------------------
# Obtención de datos
# --------------------------------------------------------------------------
def download_prices(
    tickers: list[str] = TICKERS,
    start: str = START_DATE,
    end: str = END_DATE,
) -> pd.DataFrame:
    """Descarga precios de cierre ajustados diarios con yfinance."""
    import yfinance as yf  # import local: las métricas no requieren yfinance

    raw = yf.download(
        tickers, start=start, end=end, auto_adjust=True, progress=False
    )
    prices = raw["Close"].copy()
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices.index.name = "Date"
    return prices[tickers].dropna(how="all").sort_index()


def load_prices(
    path: str | Path = DEFAULT_CACHE,
    tickers: list[str] = TICKERS,
    start: str = START_DATE,
    end: str = END_DATE,
    refresh: bool = False,
) -> pd.DataFrame:
    """Lee precios desde el cache parquet; si no existe (o ``refresh``), descarga y guarda."""
    path = Path(path)
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    prices = download_prices(tickers, start, end)
    path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(path)
    return prices


# --- Añadido en el proyecto 2: volumen diario y VIX (con cache parquet) ---
VOLUME_CACHE = DEFAULT_CACHE.with_name("volume.parquet")
VIX_CACHE = DEFAULT_CACHE.with_name("vix.parquet")


def load_volume(
    path: str | Path = VOLUME_CACHE,
    tickers: list[str] = TICKERS,
    start: str = START_DATE,
    end: str = END_DATE,
    refresh: bool = False,
) -> pd.DataFrame:
    """Volumen diario por ticker (cache parquet)."""
    path = Path(path)
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    import yfinance as yf

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    vol = raw["Volume"].copy()
    vol.index = pd.to_datetime(vol.index).tz_localize(None)
    vol.index.name = "Date"
    vol = vol[tickers].dropna(how="all").sort_index()
    path.parent.mkdir(parents=True, exist_ok=True)
    vol.to_parquet(path)
    return vol


def load_vix(
    path: str | Path = VIX_CACHE,
    start: str = START_DATE,
    end: str = END_DATE,
    refresh: bool = False,
) -> pd.Series:
    """Cierre diario del índice de volatilidad ^VIX (cache parquet)."""
    path = Path(path)
    if path.exists() and not refresh:
        return pd.read_parquet(path)["VIX"]
    import yfinance as yf

    raw = yf.download("^VIX", start=start, end=end, progress=False)
    vix = raw["Close"].squeeze().rename("VIX")
    vix.index = pd.to_datetime(vix.index).tz_localize(None)
    vix.index.name = "Date"
    vix = vix.dropna().sort_index()
    path.parent.mkdir(parents=True, exist_ok=True)
    vix.to_frame().to_parquet(path)
    return vix


# --------------------------------------------------------------------------
# Métricas (funciones puras)
# --------------------------------------------------------------------------
def daily_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Retornos simples diarios; descarta la primera fila (NaN)."""
    return prices.pct_change(fill_method=None).iloc[1:]


def cumulative_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Rendimiento acumulado desde el primer precio válido (0.10 = +10 %)."""
    return prices / prices.bfill().iloc[0] - 1


def rolling_volatility(
    returns: pd.DataFrame | pd.Series, window: int = 21, periods: int = TRADING_DAYS
) -> pd.DataFrame | pd.Series:
    """Volatilidad móvil anualizada (desv. estándar * sqrt(periods))."""
    return returns.rolling(window).std() * np.sqrt(periods)


def drawdown_series(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Caída porcentual desde el máximo histórico previo (valores <= 0)."""
    return prices / prices.cummax() - 1


def max_drawdown(prices: pd.DataFrame | pd.Series) -> float | pd.Series:
    """Máximo drawdown (valor negativo)."""
    return drawdown_series(prices).min()


def annual_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Retorno por año calendario (último precio del año / último del año anterior - 1).

    El primer año se mide desde el primer precio disponible.
    """
    year_end = prices.groupby(prices.index.year).last()
    prev = year_end.shift(1)
    prev.iloc[0] = prices.bfill().iloc[0]
    out = year_end / prev - 1
    out.index.name = "Year"
    return out


def annualized_return(returns: pd.DataFrame | pd.Series, periods: int = TRADING_DAYS):
    """Retorno anualizado geométrico a partir de retornos diarios."""
    n = returns.count()
    return (1 + returns).prod() ** (periods / n) - 1


def annualized_volatility(returns: pd.DataFrame | pd.Series, periods: int = TRADING_DAYS):
    """Volatilidad anualizada de toda la muestra."""
    return returns.std() * np.sqrt(periods)


def sharpe_ratio(
    returns: pd.DataFrame | pd.Series, risk_free: float = 0.0, periods: int = TRADING_DAYS
):
    """Sharpe anualizado: (media diaria anualizada - rf) / volatilidad anualizada."""
    vol = annualized_volatility(returns, periods)
    excess = returns.mean() * periods - risk_free
    # Volatilidad 0 o NaN -> Sharpe indefinido: NaN explícito (sin RuntimeWarning)
    if np.isscalar(vol):
        return excess / vol if vol > _MIN_VOL else np.nan
    return excess / vol.where(vol > _MIN_VOL)


def summary_table(prices: pd.DataFrame, risk_free: float = 0.0) -> pd.DataFrame:
    """Tabla resumen por ticker: retorno total, anualizado, volatilidad, MDD y Sharpe."""
    rets = daily_returns(prices)
    return pd.DataFrame(
        {
            "Retorno total": cumulative_returns(prices).ffill().iloc[-1],
            "Retorno anualizado": annualized_return(rets),
            "Volatilidad": annualized_volatility(rets),
            "Max drawdown": max_drawdown(prices),
            "Sharpe": sharpe_ratio(rets, risk_free),
        }
    )
