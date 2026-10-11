"""Explicabilidad con SHAP: gráficos globales y dos explicaciones locales."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402


def compute_shap(model, X_background: pd.DataFrame, X: pd.DataFrame) -> shap.Explanation:
    """Valores SHAP (clase positiva) para un modelo de árboles o un Pipeline lineal."""
    if isinstance(model, Pipeline):  # escalado + regresión logística
        scaler, clf = model.steps[0][1], model.steps[-1][1]
        explainer = shap.LinearExplainer(clf, scaler.transform(X_background))
        values = explainer.shap_values(scaler.transform(X))
        base = np.full(len(X), float(np.ravel(explainer.expected_value)[0]))
    else:
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(X)
        ev = np.ravel(explainer.expected_value)
        if isinstance(values, list):  # versiones antiguas: lista por clase
            values, ev = values[1], ev[-1:]
        values = np.asarray(values)
        if values.ndim == 3:  # (n, features, clases)
            values, ev = values[:, :, 1], ev[-1:]
        base = np.full(len(X), float(ev[-1]))
    return shap.Explanation(values=values, base_values=base, data=X.to_numpy(), feature_names=list(X.columns))


def _save(path: Path) -> None:
    plt.gcf().savefig(path, bbox_inches="tight", dpi=130)
    plt.close("all")


def run_shap(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    proba_test: np.ndarray,
    threshold: float,
    out_dir: Path,
    max_samples: int = 2000,
    seed: int = 42,
) -> dict:
    """Genera figuras SHAP en ``out_dir`` y devuelve la importancia media |SHAP| por feature."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    pos = np.sort(rng.choice(len(X_test), size=min(max_samples, len(X_test)), replace=False))
    Xs = X_test.iloc[pos]
    bg = X_train.sample(min(500, len(X_train)), random_state=seed)
    sv = compute_shap(model, bg, Xs)

    # Global: beeswarm y barras
    shap.summary_plot(sv.values, Xs, show=False, max_display=15)
    plt.title("SHAP: impacto de cada feature (test)")
    _save(out_dir / "shap_summary.png")
    shap.summary_plot(sv.values, Xs, plot_type="bar", show=False, max_display=15)
    plt.title("Importancia media |SHAP| (test)")
    _save(out_dir / "shap_bar.png")

    # Local: acierto y error más confiados dentro de la muestra
    pred = (proba_test[pos] >= threshold).astype(int)
    truth = y_test.iloc[pos].to_numpy()
    conf = np.abs(proba_test[pos] - threshold)
    for label, mask in (("acierto", pred == truth), ("error", pred != truth)):
        if not mask.any():
            continue
        i = int(np.flatnonzero(mask)[np.argmax(conf[mask])])
        idx = Xs.index[i]
        shap.plots.waterfall(sv[i], max_display=12, show=False)
        plt.title(
            f"Caso de {label}: {idx[1]} {idx[0]:%Y-%m-%d} | p={proba_test[pos][i]:.2f}, real={truth[i]}",
            fontsize=10,
        )
        _save(out_dir / f"shap_waterfall_{label}.png")

    importance = pd.Series(np.abs(sv.values).mean(axis=0), index=Xs.columns).sort_values(ascending=False)
    result = importance.round(5).to_dict()
    (out_dir.parent / "shap_importance.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
