# Notebooks

Source research notebooks behind the trained models, one folder per model.

- `cars/` – used-car price regression
  - `ML_Rergression_Cars_Selection.ipynb` – EDA, feature/model selection, hyperparameter search
  - `ML_Rergression_Cars_Model.ipynb` – final pipeline
- `loans/` – credit-approval classification
  - `ML_Classification_Credits_Selection.ipynb` – EDA, feature/model selection, hyperparameter search
  - `ML_Classification_Credits_Model.ipynb` – final stacked ensemble
- `bayesian/` – wage / union-membership Bayesian regression
  - `Bayesian_Regression.ipynb` – EDA + PyMC models vs econometric panel models
- `wine/` – wine-quality (good ≥ 7) classification
  - `ML_Classification_Wine_Selection.ipynb` – EDA, scaler/model grid search
  - `ML_Classification_Wine_Model.ipynb` – final model (notebook used logistic regression)

The **production** training that ships weights into the app is the standalone script
`backend/training/<model>.py` (reads `data/<model>/`, writes `backend/models/<model>.joblib`). The
notebooks are the exploration these scripts distil; they are not run by the app.
