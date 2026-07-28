# Wage & Union Membership

Panel data from the US National Longitudinal Survey: 545 young male workers followed 1980–1987
(≈4360 worker-year rows). The task is to estimate hourly wage from schooling, experience and a
few demographics, showcasing a **Bayesian linear regression**.

## Dataset (`train.csv`)

| Column | Meaning |
|---|---|
| `id` | Worker id (panel identifier) |
| `Years` | Calendar year of the observation (1980–1987) |
| `School` | Years of schooling |
| `Expert` | Work experience, years |
| `Union` | Union member (1 = yes, 0 = no) |
| `Married` | Married (1 = yes, 0 = no) |
| `Black` | Black (1 = yes, 0 = no) |
| `Hisp` | Hispanic (1 = yes, 0 = no) |
| `Wage` | **Log** hourly wage (the target; `exp(Wage)` = wage in $/hr) |

The model uses `School, Expert, Union, Married, Black` (the notebook's chosen `model_five`).
`Wage` is a natural log, so the trainer models it in log-space and reports the prediction as
`exp(Wage)` = dollars per hour (positive, intuitive), same log-target trick as the Cars model.

## Provenance

Source research notebook is in `notebooks/bayesian/` (PyMC / arviz study comparing several
Bayesian specifications against econometric panel models). The production trainer that ships
weights into the app is the standalone `backend/training/bayesian.py`.
