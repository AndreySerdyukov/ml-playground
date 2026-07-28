# Uplift Modeling (X5 Retail Hero)

Predict how much sending a promotional **SMS** raises a customer's probability of purchase, so a
campaign can target only the customers the SMS actually persuades. From the
[X5 Retail Hero uplift competition](https://ods.ai/competitions/x5-retailhero-uplift-modeling).

## File

- `train.csv` – 200,039 rows from the randomized A/B test. Columns:
  - `treatment_flg` – 1 = SMS sent (treatment), 0 = no SMS (control); the groups are ~50/50.
  - `target` – 1 = the customer made a purchase, 0 = not (overall rate ≈ 62%).
  - 13 customer-profile features (see below).

Control conversion ≈ 60.3%, treatment ≈ 63.7% → average treatment effect ≈ **+3.3pp**. The point of
uplift modeling is to find *which* customers drive that lift (some are persuaded, some would have bought
anyway, some are put off), not just the average.

## Features

Compact, interpretable customer profile distilled from the raw loyalty/purchase history (the full X5
feature matrix has ~85 columns built from 45M purchase lines; these 13 carry most of the signal and can
be entered by hand):

`age`, `gender` (F/M/U), `issue_redeem_days_diff` (days from card issue to first redeem),
`total_purchase_count`, `total_purchase_sum` (₽), `avg_purchase_sum` (₽),
`avg_days_between_purchases`, `purchase_frequency_monthly`, `unique_products`,
`alcohol_purchase_ratio`, `private_label_purchase_ratio`, `growth_rate_last_3_months`,
`count_above_1000` (baskets over 1000₽).

## Model

`backend/training/uplift.py` trains an **S-learner** (a.k.a. Treatment Dummy): a single
`HistGradientBoostingClassifier` on `[features + treatment_flg]`. At serve time it scores each customer
twice (treatment on / off) and reports the difference as the uplift. Evaluated honestly on a held-out
slice of the A/B test with **Qini** and **uplift@k** (not against a proxy uplift file).
