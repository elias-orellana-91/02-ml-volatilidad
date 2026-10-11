# Resultados

Mejor modelo completo (elegido en validación 2022): **Regresión logística**. Baselines: persistencia, logística de 1 feature (`vol5_vs_med252`) y HAR-RV. Nota: la logística de 1 feature es una transformación monótona de la persistencia, por lo que tiene el mismo ROC-AUC/PR-AUC y difiere solo en umbral y calibración.

## Test (2023-2025)

| Modelo | ROC-AUC | PR-AUC | F1 | Precisión | Recall | Brier |
|---|---|---|---|---|---|---|
| Persistencia | 0.588 | 0.546 | 0.517 | 0.515 | 0.518 | - |
| Logística 1 feature | 0.588 | 0.546 | 0.464 | 0.537 | 0.408 | 0.243 |
| HAR-RV | 0.627 | 0.577 | 0.592 | 0.518 | 0.690 | - |
| Regresión logística | 0.646 | 0.598 | 0.517 | 0.597 | 0.455 | 0.233 |
| Random Forest | 0.614 | 0.583 | 0.459 | 0.578 | 0.380 | 0.236 |
| XGBoost | 0.639 | 0.598 | 0.475 | 0.600 | 0.393 | 0.235 |

## Validación (2022)

| Modelo | ROC-AUC | PR-AUC | F1 | Precisión | Recall | Brier |
|---|---|---|---|---|---|---|
| Persistencia | 0.584 | 0.723 | 0.678 | 0.677 | 0.679 | - |
| Logística 1 feature | 0.584 | 0.723 | 0.604 | 0.707 | 0.527 | 0.244 |
| HAR-RV | 0.627 | 0.753 | 0.737 | 0.688 | 0.794 | - |
| Regresión logística | 0.713 | 0.814 | 0.729 | 0.758 | 0.701 | 0.207 |
| Random Forest | 0.644 | 0.766 | 0.756 | 0.679 | 0.852 | 0.217 |
| XGBoost | 0.666 | 0.781 | 0.734 | 0.708 | 0.762 | 0.216 |

## ROC-AUC en test por sector

| Modelo | Bancos | Consumo | Energía | Índice | Salud | Tecnología |
|---|---|---|---|---|---|---|
| Persistencia | 0.549 | 0.603 | 0.522 | 0.671 | 0.584 | 0.608 |
| Logística 1 feature | 0.549 | 0.603 | 0.522 | 0.671 | 0.584 | 0.608 |
| HAR-RV | 0.574 | 0.650 | 0.581 | 0.736 | 0.595 | 0.649 |
| Regresión logística | 0.640 | 0.657 | 0.603 | 0.774 | 0.575 | 0.659 |
| Random Forest | 0.614 | 0.600 | 0.530 | 0.730 | 0.558 | 0.643 |
| XGBoost | 0.642 | 0.620 | 0.578 | 0.754 | 0.579 | 0.660 |

## ROC-AUC en test por año

| Modelo | 2023 | 2024 | 2025 |
|---|---|---|---|
| Persistencia | 0.548 | 0.543 | 0.613 |
| Logística 1 feature | 0.548 | 0.543 | 0.613 |
| HAR-RV | 0.573 | 0.552 | 0.640 |
| Regresión logística | 0.604 | 0.576 | 0.677 |
| Random Forest | 0.576 | 0.566 | 0.664 |
| XGBoost | 0.598 | 0.589 | 0.671 |

## ¿Cuánto aportan las features adicionales? (diferencia de ROC-AUC en test)

| Comparación | Δ AUC | IC 95 % (bootstrap por meses) | P(Δ > 0) |
|---|---|---|---|
| Regresión logística - Logística 1 feature | +0.057 | [+0.033, +0.081] | 1.00 |
| Regresión logística - HAR-RV | +0.019 | [-0.001, +0.036] | 0.96 |
| Regresión logística - Persistencia | +0.057 | [+0.033, +0.081] | 1.00 |
| HAR-RV - Persistencia | +0.038 | [+0.025, +0.052] | 1.00 |
| Logística 1 feature - Persistencia | +0.000 | [+0.000, +0.000] | 0.00 |
