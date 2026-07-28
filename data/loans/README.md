# Credit Approval Prediction

Anonymised credit-application dataset. The task is to predict whether a loan application is
**approved** or **rejected**, minimising both approving bad borrowers and rejecting good ones.

## Dataset

All attribute names and values are **anonymised for confidentiality** – the columns carry no
human-readable meaning, only a type prefix:

- **N…** – numerical feature (e.g. `N2`, `N3`, `N7`, `N10`, `N13`, `N14`).
- **C…** – categorical feature already stored as a binary flag `0/1` (e.g. `C1`, `C8`, `C9`, `C11`).
- **C…_enc** – categorical feature encoded with arbitrary numeric codes, **not ordinal**
  (e.g. `C4_enc`, `C5_enc`, `C6_enc`, `C12_enc`).
- **Target** – the label:
  - `Target = 1`: credit approved.
  - `Target = 0`: credit rejected.

`train.csv` – 586 labelled rows (15 columns incl. `Target`).
`test.csv` – 104 unlabelled rows (competition-style holdout, no `Target`).

## Notes

- Because names are anonymised, the app's form exposes the raw column names as-is (honest, no
  invented labels). The `*_enc` codes are fed to the model as numbers – the tree ensembles tolerate
  the arbitrary encoding; this matches the original research notebook.
- Class balance: 325 rejected / 261 approved (~44.5% approved) – roughly balanced, so accuracy and
  ROC-AUC are both meaningful.

## Provenance

Source research notebooks are in `notebooks/loans/`. The production trainer that ships weights into
the app is the standalone `backend/training/loans.py` (reads this folder, writes
`backend/models/loans.joblib`).
