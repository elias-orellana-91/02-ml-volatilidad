# Resultados

Mejor modelo (elegido en validación 2022): **Regresión logística**.
## Test (2023-2025)

| Modelo | ROC-AUC | PR-AUC | F1 | Precisión | Recall | Brier |
|---|---|---|---|---|---|---|
| Persistencia | 0.588 | 0.546 | 0.517 | 0.515 | 0.518 | - |
| Regresión logística | 0.646 | 0.598 | 0.517 | 0.597 | 0.455 | 0.233 |
| Random Forest | 0.614 | 0.583 | 0.459 | 0.578 | 0.380 | 0.236 |
| XGBoost | 0.639 | 0.598 | 0.475 | 0.600 | 0.393 | 0.235 |

## Validación (2022)

| Modelo | ROC-AUC | PR-AUC | F1 | Precisión | Recall | Brier |
|---|---|---|---|---|---|---|
| Persistencia | 0.584 | 0.723 | 0.678 | 0.677 | 0.679 | - |
| Regresión logística | 0.713 | 0.814 | 0.729 | 0.758 | 0.701 | 0.207 |
| Random Forest | 0.644 | 0.766 | 0.756 | 0.679 | 0.852 | 0.217 |
| XGBoost | 0.666 | 0.781 | 0.734 | 0.708 | 0.762 | 0.216 |

## Test por sector — Regresión logística vs Persistencia

| Sector | n | ROC-AUC modelo | ROC-AUC persistencia | F1 modelo | F1 persistencia |
|---|---|---|---|---|---|
| Bancos | 2238 | 0.640 | 0.549 | 0.540 | 0.499 |
| Consumo | 1492 | 0.657 | 0.603 | 0.537 | 0.547 |
| Energía | 1492 | 0.603 | 0.522 | 0.439 | 0.439 |
| Índice | 746 | 0.774 | 0.671 | 0.536 | 0.542 |
| Salud | 2238 | 0.575 | 0.584 | 0.524 | 0.549 |
| Tecnología | 3730 | 0.659 | 0.608 | 0.513 | 0.518 |
