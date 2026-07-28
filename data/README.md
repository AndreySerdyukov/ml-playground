# Data

Training and test datasets, one folder per model. This is the source of truth for training –
`backend/training/<model>.py` reads from here, so weights can be reproduced from a clean clone.

- `cars/` – used-car listings
  - `train.csv` – labelled training data (`priceUSD` target)
  - `test.csv` – unlabelled holdout (competition-style, no target)
  - `README.md` – column description
- `loans/` – anonymised credit-approval applications
  - `train.csv` – labelled training data (`Target`: 1 approved / 0 rejected)
  - `test.csv` – unlabelled holdout (competition-style, no target)
  - `README.md` – column description
- `bayesian/` – NLS wage panel (schooling, experience, union, …)
  - `train.csv` – worker-year rows, `Wage` = log hourly wage (target)
  - `README.md` – column description

Note: `backend/models/*.joblib` are gitignored (rebuilt from this data via the training scripts).
