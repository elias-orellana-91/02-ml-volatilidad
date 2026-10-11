"""Entrenamiento y evaluación. Uso: ``python -m src.train [--n-iter 8] [--no-shap]``.

Protocolo (split temporal, sin mezclar fechas):
  * train: 2015-2021 | validación: 2022 | test: 2023-2025 (se evalúa una sola vez, al final).
  * Se descartan (purge) los últimos 5 días hábiles de train y de validación, porque su
    target usa retornos de la ventana siguiente.
  * Tuning: ``TimeSeriesSplit`` por fechas únicas con gap de 5 días sobre train.
  * El modelo ganador se elige por ROC-AUC en validación (nunca por test). Luego todos los
    modelos se reentrenan con train+validación y se evalúan en test.
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
from src.features import SECTOR_SLUGS, make_dataset  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODEL_VERSION = "0.1.0"
SEED = 42
GAP = 5
TRAIN_END = pd.Timestamp("2021-12-31")
VAL_END = pd.Timestamp("2022-12-31")
BASELINE = "Persistencia"
# Umbral de decisión fijo: el target es ~50 % positivo por construcción. Optimizarlo sobre 2022
# (año con 64 % de positivos) sesgaría el umbral hacia abajo y degradaría el test.
DECISION_THRESHOLD = 0.5


# ---------------------------------------------------------------- datos / splits
def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    return make_dataset(D.load_prices(), D.load_vix(), D.load_volume())


def _purged(dates: pd.DatetimeIndex, mask: np.ndarray, gap: int = GAP) -> np.ndarray:
    """Quita de ``mask`` las filas de los últimos ``gap`` días hábiles del tramo."""
    last_days = np.sort(dates[mask].unique())[-gap:]
    return mask & ~dates.isin(last_days)


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


# ---------------------------------------------------------------- pipeline principal
def main(n_iter: int = 8, shap_enabled: bool = True, out_root: Path = ROOT) -> dict:
    reports, models_dir = out_root / "reports", out_root / "models"
    figs = reports / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_dataset()
    feature_names = list(X.columns)
    dates = X.index.get_level_values("Date")
    sp = temporal_split(X)
    Xtr, ytr, Xva, yva = X[sp["train"]], y[sp["train"]], X[sp["val"]], y[sp["val"]]
    Xtv, ytv, Xte, yte = X[sp["trainval"]], y[sp["trainval"]], X[sp["test"]], y[sp["test"]]
    print(f"train={len(Xtr)} val={len(Xva)} test={len(Xte)} | features={len(feature_names)}")

    cv = date_splits(Xtr.index.get_level_values("Date"))
    val_m, test_m, fitted, thresholds, best_params = {}, {}, {}, {}, {}

    # Baseline de persistencia: vol 5d actual / umbral > 1 (sin entrenamiento)
    base_val, base_test = Xva["vol5_vs_med252"].to_numpy(), Xte["vol5_vs_med252"].to_numpy()
    val_m[BASELINE] = evaluate(yva, base_val, 1.0, has_proba=False)
    test_m[BASELINE] = evaluate(yte, base_test, 1.0, has_proba=False)

    probas_test = {}
    for name, (est, space, it) in model_specs(n_iter).items():
        print(f"[{name}] tuning...")
        search = RandomizedSearchCV(
            est, space, n_iter=it, scoring="roc_auc", cv=cv, random_state=SEED, n_jobs=1, refit=True
        )
        search.fit(Xtr, ytr)
        best_params[name] = {k: (v.item() if hasattr(v, "item") else v) for k, v in search.best_params_.items()}
        # Validación: modelo ajustado solo con train (selección del modelo)
        pva = search.best_estimator_.predict_proba(Xva)[:, 1]
        thr = DECISION_THRESHOLD
        thresholds[name] = thr
        val_m[name] = evaluate(yva, pva, thr)
        # Reentrenar con train+val con los mismos hiperparámetros y evaluar en test (una vez)
        final = clone(search.best_estimator_).fit(Xtv, ytv)
        fitted[name] = final
        probas_test[name] = final.predict_proba(Xte)[:, 1]
        test_m[name] = evaluate(yte, probas_test[name], thr)
        print(f"   cv_auc={search.best_score_:.3f} val_auc={val_m[name]['roc_auc']:.3f} "
              f"test_auc={test_m[name]['roc_auc']:.3f} thr={thr:.2f}")

    best = max(fitted, key=lambda n: val_m[n]["roc_auc"])  # elegido por validación
    print(f"Mejor modelo (por validación): {best}")

    # Métricas por sector en test (mejor modelo y baseline)
    sector_of = Xte[[f"sector_{s}" for s in SECTOR_SLUGS.values()]].idxmax(axis=1).str.removeprefix("sector_")
    inv = {v: k for k, v in SECTOR_SLUGS.items()}
    by_sector = {}
    for sec in sorted(sector_of.unique()):
        m = (sector_of == sec).to_numpy()
        by_sector[inv[sec]] = {
            best: evaluate(yte[m], probas_test[best][m], thresholds[best]),
            BASELINE: evaluate(yte[m], base_test[m], 1.0, has_proba=False),
        }

    # Figuras: calibración y matrices de confusión
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, p in probas_test.items():
        frac, mean = calibration_curve(yte, p, n_bins=10, strategy="quantile")
        ax.plot(mean, frac, marker="o", label=name)
    ax.plot([0, 1], [0, 1], "k--", label="Perfecta")
    ax.set(xlabel="Probabilidad predicha", ylabel="Fracción real de positivos", title="Curva de calibración (test)")
    ax.legend()
    fig.savefig(figs / "calibration.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (title, key) in zip(axes, ((f"{best} (test)", best), (f"{BASELINE} (test)", BASELINE))):
        cm = np.array(test_m[key]["confusion_matrix"])
        ConfusionMatrixDisplay(cm, display_labels=["Vol normal", "Vol alta"]).plot(ax=ax, colorbar=False)
        ax.set_title(title)
    fig.savefig(figs / "confusion_matrix.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Reportes
    result = {
        "best_model": best,
        "selection_criterion": "ROC-AUC en validación (2022)",
        "best_params": best_params,
        "validation": val_m,
        "test": test_m,
        "test_by_sector": by_sector,
        "splits": {"train": "2015-2021", "validation": "2022", "test": "2023-2025", "purge_days": GAP},
    }
    (reports / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    md = [
        f"# Resultados\n\nMejor modelo (elegido en validación 2022): **{best}**.\n",
        "## Test (2023-2025)\n\n" + metrics_table(test_m),
        "\n## Validación (2022)\n\n" + metrics_table(val_m),
        f"\n## Test por sector — {best} vs {BASELINE}\n\n| Sector | n | ROC-AUC modelo | ROC-AUC persistencia | F1 modelo | F1 persistencia |\n|---|---|---|---|---|---|\n",
    ]
    for sec, d in by_sector.items():
        md.append(
            f"| {sec} | {d[best]['n']} | {d[best]['roc_auc']:.3f} | {d[BASELINE]['roc_auc']:.3f} "
            f"| {d[best]['f1']:.3f} | {d[BASELINE]['f1']:.3f} |\n"
        )
    (reports / "metrics.md").write_text("".join(md), encoding="utf-8")

    # Modelo + metadata (consumidos por la API del proyecto 3)
    joblib.dump(fitted[best], models_dir / "model.joblib")
    metadata = {
        "model_name": best,
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "features": feature_names,
        "decision_threshold": thresholds[best],
        "target": "vol_fwd_5d > mediana móvil 252d (trailing) de la vol 5d del ticker",
        "trained_on": "2016-01 a 2022-12 (train+validación)",
        "hyperparameters": best_params[best],
        "test_metrics": test_m[best],
        "baseline_test_metrics": test_m[BASELINE],
        "library_versions": {
            "scikit-learn": __import__("sklearn").__version__,
            "xgboost": __import__("xgboost").__version__,
            "pandas": pd.__version__,
        },
    }
    (models_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    if shap_enabled:
        imp = run_shap(fitted[best], Xtv, Xte, yte, probas_test[best], thresholds[best], figs)
        print("Top-5 SHAP:", list(imp)[:5])

    print("\n" + metrics_table(test_m))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Entrena y evalúa los modelos de volatilidad.")
    ap.add_argument("--n-iter", type=int, default=8, help="iteraciones de búsqueda aleatoria (RF/XGB)")
    ap.add_argument("--no-shap", action="store_true", help="omitir figuras SHAP")
    args = ap.parse_args()
    main(args.n_iter, not args.no_shap)

