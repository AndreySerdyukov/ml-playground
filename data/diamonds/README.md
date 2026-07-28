# Diamond Sales Price Prediction

Predict a diamond's `total_sales_price` (USD) from its carat, quality grades and physical
measurements. The quality metric is **Mean Absolute Percentage Error (MAPE)**.

## Files

- `train.csv` – 67,598 labelled rows (includes `total_sales_price`).
- `test.csv` – 22,453 rows, same features **without** the target (competition-style holdout).
- `test_Y_true.csv` – the true `total_sales_price` for `test.csv`, used only for final scoring.

## Features

- **size** – carat weight (0.15–2.0).
- **color** – colour grade D (best) … M.
- **clarity** – clarity grade IF (best) … I3, GIA scale.
- **cut** – cut grade (Excellent / Very Good).
- **symmetry** – symmetry grade (Excellent / Very Good).
- **polish** – polish grade (Excellent / Very Good).
- **depth_percent** – total depth as % of the average diameter.
- **table_percent** – table facet width as % of the average diameter.
- **meas_length / meas_width / meas_depth** – physical measurements in mm.
- **total_sales_price** – target, USD (242–19,996).

## Methodology note (why this dataset ships with a training script)

A common mistake on this task inflates MAPE to ~18.7: running an IQR outlier filter over **all**
numeric columns – including the target `total_sales_price` – deletes the entire expensive tail
(~11% of rows) from training, while the real `test.csv` keeps the full price range, so the model
cannot reach expensive stones. The correct approach used by `backend/training/diamonds.py`: split
first, clean/impute only inside a pipeline fit on the training data, **never filter on the target**,
and never touch the held-out test set by hand.
