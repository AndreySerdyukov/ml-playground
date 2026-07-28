# Wine Quality (good-wine classification)

Physicochemical measurements of white wines with an expert quality score (0–10). The task here is
a **binary** one: will a wine be rated **Good** (quality ≥ 7) or **Standard** (< 7)? – the same
framing as the source notebook.

## Dataset

`train.csv` – 1725 labelled wines (11 features + `quality`).
`test.csv` – 576 unlabelled wines (competition-style holdout, no `quality`).

| Column | Meaning |
|---|---|
| `fixed acidity` | Tartaric acid (g/L) |
| `volatile acidity` | Acetic acid (g/L) |
| `citric acid` | Citric acid (g/L) |
| `residual sugar` | Residual sugar (g/L) |
| `chlorides` | Chlorides (g/L) |
| `free sulfur dioxide` | Free SO₂ (mg/L) |
| `total sulfur dioxide` | Total SO₂ (mg/L) |
| `density` | Density (g/cm³) |
| `pH` | pH |
| `sulphates` | Potassium sulphate (g/L) |
| `alcohol` | Alcohol (% vol) |
| `quality` | Expert score 0–10 (target; binarised to `quality ≥ 7` = Good) |

## Notes

- **~36% of wines are Good** (quality ≥ 7) – roughly balanced, so both precision and recall matter.
- The original notebook used plain logistic regression (ROC-AUC ≈ 0.82). The production model here is
  an **ExtraTrees ensemble** (ROC-AUC ≈ 0.85, PR-AUC ≈ 0.74) – it lifts the whole precision/recall
  frontier: e.g. at 90% precision recall goes from ~1.5% (logreg) to ~15%. Alcohol is by far the
  strongest quality driver, then density and residual sugar.
- Quality labels are subjective, so ROC-AUC ~0.85 is near the practical ceiling for these 11 features.

## Provenance

Source research notebooks are in `notebooks/wine/`. The production trainer that ships weights into
the app is the standalone `backend/training/wine.py`.
