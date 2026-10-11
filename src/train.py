"""Entrenamiento y evaluación. Uso: ``python -m src.train [--n-iter 8] [--no-shap]``.

Protocolo (split temporal, sin mezclar fechas):
  * train: 2015-2021 | validación: 2022 | test: 2023-2025 (se evalúa una sola vez, al final).
  * Se descartan (purge) los últimos 5 días hábiles de train y de validación, porque su
    target usa retornos de la ventana siguiente.
  * Tuning: ``TimeSeriesSplit`` por fechas únicas con gap de 5 días sobre train.
  * El modelo ganador (entre los de 24 features) se elige por ROC-AUC en validación (nunca por
    test). Luego todos los modelos y baselines se reentrenan con train+validación y se evalúan
    en test. Baselines: persistencia, logística de 1 feature y HAR-RV (``src/baselines.py``).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from src import data as D  # noqa: E402
from src.explain import run_shap  # noqa: E402
from src.baselines import HARRV, SINGLE_FEATURE, single_feature_logit  # noqa: E402
from src.features import (  # noqa: E402
    SECTOR_COLUMNS,
    SECTOR_SLUGS,
    TICKER_TO_SECTOR,
    build_har_inputs,
    build_target,
    make_dataset,
    min_history_days,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_VERSION = "0.1.0"
SEED = 42
GAP = 5
TRAIN_END = pd.Timestamp("2021-12-31")
VAL_END = pd.Timestamp("2022-12-31")
PERSISTENCE = "Persistencia"
LOGIT1 = "Logística 1 feature"
HAR = "HAR-RV"
BASELINES = [PERSISTENCE, LOGIT1, HAR]
# Umbral de decisión fijo: el target es ~50 % positivo por construcción. Optimizarlo sobre 2022
# (año con 65 % de positivos) sesgaría el umbral hacia abajo y degradaría el test.
DECISION_THRESHOLD = 0.5


# ---------------------------------------------------------------- datos / splits
def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    return make_dataset(D.load_prices(), D.load_vix(), D.load_volume())


def _purged(dates: pd.DatetimeIndex, mask: np.ndarray, gap: int = GAP) -> np.ndarray:
    """Quita de ``mask`` las filas de los últimos ``gap`` días hábiles del tramo."""
    last_days = np.sort(dates[mask].unique())[-gap:]
    return mask & ~dates.isin(last_days)


def load_baseline_inputs(X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Entradas HAR-RV y (vol_fwd_5d, umbral) alineadas con el índice de ``X``."""
    prices = D.load_prices()
    return build_har_inputs(prices).reindex(X.index), build_target(prices).reindex(X.index)


def temporal_split(X: pd.DataFrame) -> dict[str, np.ndarray]:
    """Máscaras booleanas: train, val, trainval (train+val purgados) y test."""
    d = X.index.get_level_values("Date")
    train = _purged(d, np.asarray(d <= TRAIN_END))
    val = _purged(d, np.asarray((d > TRAIN_END) & (d <= VAL_END)))
    test = np.asarray(d > VAL_END)
    return {"train": train, "val": val, "trainval": train | val, "test": test}


def date_splits(dates: pd.DatetimeIndex, n_splits: int = 4, gap: int = GAP):
    """TimeSeriesSplit sobre fechas únicas (todas las acciones de un día van juntas)."""
    uniq = np.sort(dates.unique())
    out = []
    for tr, te in TimeSeriesSplit(n_splits=n_splits, gap=gap).split(uniq):
        out.append((np.flatnonzero(dates.isin(uniq[tr])), np.flatnonzero(dates.isin(uniq[te]))))
    return out


# ---------------------------------------------------------------- modelos
def model_specs(n_iter: int) -> dict[str, tuple]:
    """(estimador, espacio de hiperparámetros, n_iter)."""
    return {
        LOGIT1: (single_feature_logit(), {"clf__C": [0.01, 0.1, 1.0, 10.0]}, 4),
        "Regresión logística": (
            Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=3000))]),
            {"clf__C": [0.01, 0.1, 1.0, 10.0]},
            4,
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=250, n_jobs=-1, random_state=SEED),
            {"max_depth": [4, 6, 10], "min_samples_leaf": [20, 50, 100], "max_features": ["sqrt", 0.5]},
            n_iter,
        ),
        "XGBoost": (
            XGBClassifier(tree_method="hist", eval_metric="logloss", n_jobs=-1, random_state=SEED),
            {
                "n_estimators": [150, 300, 500],
                "max_depth": [2, 3, 4, 5],
                "learning_rate": [0.02, 0.05, 0.1],
                "subsample": [0.7, 0.9, 1.0],
                "colsample_bytree": [0.6, 0.8, 1.0],
                "min_child_weight": [5, 20, 50],
            },
            n_iter,
        ),
    }


# ---------------------------------------------------------------- métricas
def evaluate(y, score, threshold: float, has_proba: bool = True) -> dict:
    """Métricas de clasificación. ``score`` es probabilidad (o un score continuo en el baseline)."""
    y = np.asarray(y)
    pred = (np.asarray(score) >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, score)),
        "pr_auc": float(average_precision_score(y, score)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "brier": float(brier_score_loss(y, score)) if has_proba else None,
        "threshold": float(threshold),
        "base_rate": float(y.mean()),
        "n": int(len(y)),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }


def _fmt(x) -> str:
    return "-" if x is None else f"{x:.3f}"


def metrics_table(rows: dict[str, dict]) -> str:
    cols = ["roc_auc", "pr_auc", "f1", "precision", "recall", "brier"]
    head = "| Modelo | ROC-AUC | PR-AUC | F1 | Precisión | Recall | Brier |\n|---|---|---|---|---|---|---|\n"
    return head + "".join(
        f"| {name} | " + " | ".join(_fmt(m[c]) for c in cols) + " |\n" for name, m in rows.items()
    )



def bootstrap_auc_diff(y, s_a, s_b, groups, n_boot: int = 300, seed: int = SEED) -> dict:
    """Diferencia de ROC-AUC (a - b) con bootstrap por bloques (meses) para respetar la autocorrelación."""
    y, s_a, s_b = np.asarray(y), np.asarray(s_a), np.asarray(s_b)
    codes, uniq = pd.factorize(groups)
    by_block = [np.flatnonzero(codes == i) for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        ix = np.concatenate([by_block[i] for i in rng.integers(0, len(uniq), len(uniq))])
        if y[ix].min() == y[ix].max():
            continue
        diffs.append(roc_auc_score(y[ix], s_a[ix]) - roc_auc_score(y[ix], s_b[ix]))
    d = np.asarray(diffs)
    return {
        "mean": float(d.mean()),
        "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
        "prob_positive": float((d > 0).mean()),
    }


def pivot_md(groups: dict[str, dict[str, dict]], models: list[str], metric: str = "roc_auc") -> str:
    """Tabla markdown modelos x grupos (sector o año) para una métrica."""
    keys = list(groups)
    out = "| Modelo | " + " | ".join(map(str, keys)) + " |\n|---|" + "---|" * len(keys) + "\n"
    for m in models:
        out += f"| {m} | " + " | ".join(_fmt(groups[k][m][metric]) for k in keys) + " |\n"
    return out


def _clean(params: dict) -> dict:
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in params.items()}


# ---------------------------------------------------------------- pipeline principal
def main(n_iter: int = 8, shap_enabled: bool = True, out_root: Path = ROOT) -> dict:
    reports, models_dir = out_root / "reports", out_root / "models"
    figs = reports / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_dataset()
    har, tgt = load_baseline_inputs(X)
    feature_names = list(X.columns)
    dates = X.index.get_level_values("Date")
    sp = temporal_split(X)
    Xtr, ytr, Xva, yva = X[sp["train"]], y[sp["train"]], X[sp["val"]], y[sp["val"]]
    Xtv, ytv, Xte, yte = X[sp["trainval"]], y[sp["trainval"]], X[sp["test"]], y[sp["test"]]
    print(f"train={len(Xtr)} val={len(Xva)} test={len(Xte)} | features={len(feature_names)}")

    cv = date_splits(Xtr.index.get_level_values("Date"))
    val_m, test_m, fitted, thresholds, best_params = {}, {}, {}, {}, {}
    scores_test: dict[str, np.ndarray] = {}

    # --- Baseline 1: persistencia (sin entrenamiento): vol 5d actual / umbral
    val_m[PERSISTENCE] = evaluate(yva, Xva[SINGLE_FEATURE].to_numpy(), 1.0, has_proba=False)
    scores_test[PERSISTENCE] = Xte[SINGLE_FEATURE].to_numpy()
    test_m[PERSISTENCE] = evaluate(yte, scores_test[PERSISTENCE], 1.0, has_proba=False)
    thresholds[PERSISTENCE] = 1.0

    # --- Baseline 2: HAR-RV (OLS agrupado; ajustado en train para validación y en train+val para test)
    har_va = HARRV().fit(har[sp["train"]], tgt.loc[sp["train"], "vol_fwd_5d"])
    val_m[HAR] = evaluate(yva, har_va.score(har[sp["val"]], tgt.loc[sp["val"], "threshold"]), 1.0, has_proba=False)
    har_te = HARRV().fit(har[sp["trainval"]], tgt.loc[sp["trainval"], "vol_fwd_5d"])
    scores_test[HAR] = har_te.score(har[sp["test"]], tgt.loc[sp["test"], "threshold"])
    test_m[HAR] = evaluate(yte, scores_test[HAR], 1.0, has_proba=False)
    thresholds[HAR] = 1.0
    print(f"[{HAR}] coeficientes: {har_te.coefficients}")

    # --- Logística de 1 feature + modelos completos (mismo protocolo de tuning)
    for name, (est, space, it) in model_specs(n_iter).items():
        print(f"[{name}] tuning...")
        search = RandomizedSearchCV(
            est, space, n_iter=it, scoring="roc_auc", cv=cv, random_state=SEED, n_jobs=1, refit=True
        )
        search.fit(Xtr, ytr)
        best_params[name] = _clean(search.best_params_)
        # Validación: modelo ajustado solo con train (selección del modelo)
        pva = search.best_estimator_.predict_proba(Xva)[:, 1]
        thresholds[name] = DECISION_THRESHOLD
        val_m[name] = evaluate(yva, pva, DECISION_THRESHOLD)
        # Reentrenar con train+val con los mismos hiperparámetros y evaluar en test (una vez)
        fitted[name] = clone(search.best_estimator_).fit(Xtv, ytv)
        scores_test[name] = fitted[name].predict_proba(Xte)[:, 1]
        test_m[name] = evaluate(yte, scores_test[name], DECISION_THRESHOLD)
        print(f"   cv_auc={search.best_score_:.3f} val_auc={val_m[name]['roc_auc']:.3f} "
              f"test_auc={test_m[name]['roc_auc']:.3f}")

    # Selección SOLO entre modelos completos y SOLO por validación (los baselines no compiten:
    # el artefacto guardado debe usar las 24 features que consumirá la API).
    candidates = [n for n in fitted if n not in BASELINES]
    best = max(candidates, key=lambda n: val_m[n]["roc_auc"])
    print(f"Mejor modelo (por validación): {best}")
    order = BASELINES + candidates  # orden de las tablas
    val_m = {n: val_m[n] for n in order}
    test_m = {n: test_m[n] for n in order}

    # Métricas por sector y por año en test (todos los modelos y baselines)
    sector_of = Xte[SECTOR_COLUMNS].idxmax(axis=1).str.removeprefix("sector_")
    inv = {v: k for k, v in SECTOR_SLUGS.items()}
    year_of = Xte.index.get_level_values("Date").year

    def by_group(labels, names) -> dict:
        out = {}
        for g in names:
            m = np.asarray(labels == g)
            out[g] = {
                n: evaluate(yte[m], scores_test[n][m], thresholds[n], has_proba=n not in (PERSISTENCE, HAR))
                for n in order
            }
        return out

    by_sector = by_group(sector_of, sorted(sector_of.unique()))
    by_sector = {inv[k]: v for k, v in by_sector.items()}
    by_year = {str(k): v for k, v in by_group(year_of, sorted(set(year_of))).items()}

    # ¿Cuánto aportan las features adicionales? AUC en test + bootstrap por meses
    auc = {n: test_m[n]["roc_auc"] for n in order}
    months = Xte.index.get_level_values("Date").to_period("M")
    pairs = {
        f"{best} - {LOGIT1}": (best, LOGIT1),
        f"{best} - {HAR}": (best, HAR),
        f"{best} - {PERSISTENCE}": (best, PERSISTENCE),
        f"{HAR} - {PERSISTENCE}": (HAR, PERSISTENCE),
        f"{LOGIT1} - {PERSISTENCE}": (LOGIT1, PERSISTENCE),
    }
    boot = {k: bootstrap_auc_diff(yte, scores_test[a], scores_test[b], months) for k, (a, b) in pairs.items()}
    total_gain = auc[best] - auc[PERSISTENCE]
    attribution = {
        "auc_test": auc,
        "gain_best_vs_persistence": total_gain,
        "gain_from_trained_single_feature": auc[LOGIT1] - auc[PERSISTENCE],
        "gain_from_har_structure": auc[HAR] - auc[PERSISTENCE],
        "gain_from_extra_features_vs_logit1": auc[best] - auc[LOGIT1],
        "gain_from_extra_features_vs_har": auc[best] - auc[HAR],
        "share_of_gain_from_extra_features_vs_har": (auc[best] - auc[HAR]) / total_gain if total_gain > 0 else None,
        "bootstrap_auc_diff_by_month_blocks": boot,
    }

    # Figuras: calibración (modelos con probabilidades) y matrices de confusión
    fig, ax = plt.subplots(figsize=(6, 5))
    for name in [LOGIT1] + candidates:
        frac, mean = calibration_curve(yte, scores_test[name], n_bins=10, strategy="quantile")
        ax.plot(mean, frac, marker="o", label=name)
    ax.plot([0, 1], [0, 1], "k--", label="Perfecta")
    ax.set(xlabel="Probabilidad predicha", ylabel="Fracción real de positivos", title="Curva de calibración (test)")
    ax.legend()
    fig.savefig(figs / "calibration.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key in zip(axes, (best, PERSISTENCE)):
        cm = np.array(test_m[key]["confusion_matrix"])
        ConfusionMatrixDisplay(cm, display_labels=["Vol normal", "Vol alta"]).plot(ax=ax, colorbar=False)
        ax.set_title(f"{key} (test)")
    fig.savefig(figs / "confusion_matrix.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Reportes
    result = {
        "best_model": best,
        "selection_criterion": "ROC-AUC en validación (2022), solo entre modelos completos (24 features)",
        "best_params": best_params,
        "har_rv_coefficients": har_te.coefficients,
        "validation": val_m,
        "test": test_m,
        "test_by_sector": by_sector,
        "test_by_year": by_year,
        "feature_contribution": attribution,
        "splits": {"train": "2015-2021", "validation": "2022", "test": "2023-2025", "purge_days": GAP},
    }
    (reports / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    md = [
        f"# Resultados\n\nMejor modelo completo (elegido en validación 2022): **{best}**. "
        "Baselines: persistencia, logística de 1 feature (`vol5_vs_med252`) y HAR-RV. "
        "Nota: la logística de 1 feature es una transformación monótona de la persistencia, por lo que "
        "tiene el mismo ROC-AUC/PR-AUC y difiere solo en umbral y calibración.\n",
        "\n## Test (2023-2025)\n\n" + metrics_table(test_m),
        "\n## Validación (2022)\n\n" + metrics_table(val_m),
        "\n## ROC-AUC en test por sector\n\n" + pivot_md(by_sector, order),
        "\n## ROC-AUC en test por año\n\n" + pivot_md(by_year, order),
        "\n## ¿Cuánto aportan las features adicionales? (diferencia de ROC-AUC en test)\n\n"
        "| Comparación | Δ AUC | IC 95 % (bootstrap por meses) | P(Δ > 0) |\n|---|---|---|---|\n",
    ]
    for k, b in boot.items():
        md.append(f"| {k} | {b['mean']:+.3f} | [{b['ci95'][0]:+.3f}, {b['ci95'][1]:+.3f}] | {b['prob_positive']:.2f} |\n")
    (reports / "metrics.md").write_text("".join(md), encoding="utf-8")

    # Modelo + metadata (consumidos por la API del proyecto 3)
    d_tv = dates[sp["trainval"]]
    metadata = {
        "model_name": best,
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "features": feature_names,
        "decision_threshold": thresholds[best],
        "target": "vol_fwd_5d > mediana móvil 252d (trailing) de la vol 5d del ticker",
        "trained_on": f"{d_tv.min():%Y-%m-%d} a {d_tv.max():%Y-%m-%d} (train+validación, {int(sp['trainval'].sum())} filas)",
        "trained_on_range": {"start": f"{d_tv.min():%Y-%m-%d}", "end": f"{d_tv.max():%Y-%m-%d}"},
        "required_inputs": {
            "series": {
                "prices": "Precios de cierre ajustados diarios (DataFrame Date x Ticker). Debe incluir el ticker objetivo y SPY.",
                "vix": "Cierre diario del ^VIX (Series indexada por fecha).",
                "volume": "Volumen diario (DataFrame Date x Ticker) del ticker objetivo y SPY.",
            },
            "market_ticker": "SPY",
            "min_history_days": min_history_days(),
            "tickers": dict(sorted(TICKER_TO_SECTOR.items())),
            "feature_builder": "src.features.build_features",
        },
        "hyperparameters": best_params[best],
        "test_metrics": test_m[best],
        "baseline_test_metrics": test_m[PERSISTENCE],
        "baselines_test_metrics": {n: test_m[n] for n in BASELINES},
        "library_versions": {
            "scikit-learn": __import__("sklearn").__version__,
            "xgboost": __import__("xgboost").__version__,
            "pandas": pd.__version__,
        },
    }
    joblib.dump(fitted[best], models_dir / "model.joblib")
    (models_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    if shap_enabled:
        imp = run_shap(fitted[best], Xtv, Xte, yte, scores_test[best], thresholds[best], figs)
        print("Top-5 SHAP:", list(imp)[:5])

    print("\n" + metrics_table(test_m))
    print(json.dumps(boot, indent=1))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Entrena y evalúa los modelos de volatilidad.")
    ap.add_argument("--n-iter", type=int, default=8, help="iteraciones de búsqueda aleatoria (RF/XGB)")
    ap.add_argument("--no-shap", action="store_true", help="omitir figuras SHAP")
    args = ap.parse_args()
    main(args.n_iter, not args.no_shap)


