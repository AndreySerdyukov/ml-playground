"""Обучение и экспорт РЕАЛЬНОЙ модели «Uplift» (X5 Retail Hero) для ML Playground.

Данные — `data/uplift/train.csv` (компактный набор из соревновательного A/B-теста: `treatment_flg`,
`target` + 13 профильных фич). Задача — оценить, на сколько отправка рекламного SMS повышает
вероятность покупки КОНКРЕТНОГО клиента, чтобы таргетировать только тех, кого SMS реально убеждает.

**Подход — S-learner (Treatment Dummy):** одна модель классификации на `[фичи + treatment_flg]`;
uplift = `P(покупка|SMS) − P(покупка|без SMS)` (два прогона того же классификатора). Из пяти подходов
исходного проекта у него лучший uplift@k и он естественно кладётся в сервинг (serve-обёртка
`app.uplift.SoloModelUplift`, два прогона на инференсе).

**Честная оценка (в отличие от исходного ноутбука, где мерили MAE против прокси-файла `uplift_sub.csv`):**
split A/B → Qini-коэффициент и uplift@k по РЕАЛЬНЫМ `treatment_flg`/`target` на отложенной выборке.

Итог — бандл `{"model": SoloModelUplift, "meta": ModelInfo(...).model_dump()}` в
`backend/models/uplift.joblib` + `uplift.model_card.json`. Форма строится из `features` (только профиль,
флаг лечения не спрашиваем). Этот файл — ЕДИНСТВЕННЫЙ источник истины для описания модели Uplift.

Запуск (из каталога backend, в .venv):
    python training/uplift.py
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Делаем пакет `app` импортируемым при запуске файла напрямую (training/ -> backend/).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.schemas.predict import FeatureSpec, ModelInfo
from app.uplift import SoloModelUplift

TREATMENT_COL = "treatment_flg"
TARGET_COL = "target"

# Метаописание профильных фич: тип поля + лейбл/единица/пример (пример = реальная строка клиента).
FEATURE_META: dict[str, dict[str, Any]] = {
    "age": {"kind": "num", "label": "Age", "unit": "yrs", "example": 34},
    "gender": {"kind": "cat", "label": "Gender", "choices": ["F", "M", "U"], "example": "M"},
    "issue_redeem_days_diff": {"kind": "num", "label": "Days to first redeem", "unit": "days", "example": 137},
    "total_purchase_count": {"kind": "num", "label": "Total purchases", "example": 38},
    "total_purchase_sum": {"kind": "num", "label": "Total spend", "unit": "₽", "example": 7837},
    "avg_purchase_sum": {"kind": "num", "label": "Avg receipt", "unit": "₽", "example": 206},
    "avg_days_between_purchases": {"kind": "num", "label": "Days between visits", "unit": "days", "example": 2.5},
    "purchase_frequency_monthly": {"kind": "num", "label": "Purchases / month", "example": 12.3},
    "unique_products": {"kind": "num", "label": "Unique products", "example": 24},
    "alcohol_purchase_ratio": {"kind": "num", "label": "Alcohol share", "example": 0.07},
    "private_label_purchase_ratio": {"kind": "num", "label": "Private-label share", "example": 0.6},
    "growth_rate_last_3_months": {"kind": "num", "label": "3-mo spend growth", "example": 176},
    "count_above_1000": {"kind": "num", "label": "Baskets > 1000₽", "example": 2},
}
FEATURES: list[str] = list(FEATURE_META)  # порядок = порядок полей формы
NUM_FEATURES = [c for c, m in FEATURE_META.items() if m["kind"] == "num"]
CAT_FEATURES = [c for c, m in FEATURE_META.items() if m["kind"] == "cat"]

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "uplift"
DEFAULT_TRAIN_CSV = DATA_DIR / "train.csv"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "models" / "uplift.joblib"


def build_specs() -> list[FeatureSpec]:
    """FeatureSpec для формы (только профиль клиента; treatment_flg не вводится)."""
    specs: list[FeatureSpec] = []
    for name in FEATURES:
        meta = FEATURE_META[name]
        if meta["kind"] == "num":
            specs.append(
                FeatureSpec(name=name, type="number", label=meta["label"],
                            unit=meta.get("unit"), example=meta["example"])
            )
        else:
            specs.append(
                FeatureSpec(name=name, type="category", label=meta["label"],
                            choices=meta["choices"], example=meta["example"])
            )
    return specs


def build_pipeline() -> Pipeline:
    """S-learner base: препроцессинг + HistGradientBoostingClassifier на [фичи + treatment_flg]."""
    preproc = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), [*NUM_FEATURES, TREATMENT_COL]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
        ]
    )
    clf = HistGradientBoostingClassifier(
        max_iter=500, learning_rate=0.05, max_leaf_nodes=31,
        min_samples_leaf=50, l2_regularization=1.0, random_state=42,
    )
    return Pipeline(steps=[("preproc", preproc), ("clf", clf)])


def uplift_at_k(y: np.ndarray, w: np.ndarray, score: np.ndarray, k: float) -> float:
    """uplift@k: разница response-rate лечение/контроль среди top-k% по предсказанному uplift."""
    n = int(np.ceil(k * len(score)))
    idx = np.argsort(-score)[:n]
    yt, wt = y[idx], w[idx]
    rt = yt[wt == 1].mean() if (wt == 1).any() else 0.0
    rc = yt[wt == 0].mean() if (wt == 0).any() else 0.0
    return float(rt - rc)


def _qini_area(y: np.ndarray, w: np.ndarray, order: np.ndarray) -> float:
    """Площадь Qini-кривой над случайной прямой при ранжировании по `order`."""
    y_, w_ = y[order].astype(float), w[order].astype(float)
    nt, nc = np.cumsum(w_), np.cumsum(1 - w_)
    with np.errstate(divide="ignore", invalid="ignore"):
        curve = np.cumsum(y_ * w_) - np.cumsum(y_ * (1 - w_)) * np.where(nc > 0, nt / nc, 0.0)
    curve = np.concatenate([[0.0], curve])
    x = np.arange(len(curve))
    rand = x / x[-1] * curve[-1]
    return float(np.trapezoid(curve - rand, x))


def qini_coefficient(y: np.ndarray, w: np.ndarray, score: np.ndarray) -> float:
    """Нормированный Qini: 0 = случайный таргетинг, 1 = идеальное ранжирование."""
    model = _qini_area(y, w, np.argsort(-score))
    # Идеал: сначала лечёные-покупатели, потом контроль-покупатели снизу.
    perfect_order = np.argsort(-np.where(w == 1, y + 1, -(1 - y)))
    perfect = _qini_area(y, w, perfect_order)
    return model / perfect if perfect else 0.0


def evaluate(wrap: SoloModelUplift, x_val: pd.DataFrame, w_val: np.ndarray, y_val: np.ndarray) -> dict[str, float]:
    """Честные uplift-метрики по A/B на отложенной выборке."""
    score = wrap.predict(x_val)
    p_treat, _ = wrap.predict_scenarios(x_val)  # P(покупка|SMS) — база для response-AUC
    ate = float(y_val[w_val == 1].mean() - y_val[w_val == 0].mean())
    return {
        "qini": qini_coefficient(y_val, w_val, score),
        "uplift_at_10": uplift_at_k(y_val, w_val, score, 0.10),
        "uplift_at_20": uplift_at_k(y_val, w_val, score, 0.20),
        "uplift_at_30": uplift_at_k(y_val, w_val, score, 0.30),
        "ate": ate,
        "response_auc": float(roc_auc_score(y_val, p_treat)),
        "score_std": float(score.std()),
    }


def build_model_info(specs: list[FeatureSpec], m: dict[str, float], n: int) -> ModelInfo:
    """Карточка модели (category='Uplift' → «богатая» карточка с двумя сценариями)."""
    description = (
        f"Which customers a promo SMS actually persuades to buy. Qini {m['qini']:.3f}, "
        f"uplift@30% {m['uplift_at_30']:.1%} vs {m['ate']:.1%} average lift "
        f"(S-learner on {n:,} A/B records)"
    )
    return ModelInfo(
        name="uplift",
        task="regression",
        target="uplift from SMS",
        category="Uplift",
        emoji="",
        is_stub=False,
        description=description,
        features=specs,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Обучить и экспортировать модель Uplift (S-learner).")
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if not args.train_csv.exists():
        raise SystemExit(f"Не найден датасет: {args.train_csv}")

    df = pd.read_csv(args.train_csv).drop_duplicates().reset_index(drop=True)
    specs = build_specs()
    x = df[FEATURES]
    w = df[TREATMENT_COL].to_numpy()
    y = df[TARGET_COL].to_numpy()
    print(f"train.csv: строк={len(df)}, фич={len(FEATURES)}; лечение={int(w.sum())}/контроль={int((1-w).sum())}")

    # Сплит с раскладкой по лечение×исход, чтобы обе группы были в train и val.
    x_tr, x_val, w_tr, w_val, y_tr, y_val = train_test_split(
        x, w, y, test_size=0.25, random_state=42, stratify=w * 2 + y
    )

    # Оценка: base на train-части (фичи + флаг), uplift-метрики на val.
    eval_pipe = build_pipeline()
    eval_pipe.fit(x_tr.assign(**{TREATMENT_COL: w_tr}), y_tr)
    eval_wrap = SoloModelUplift(eval_pipe, FEATURES, TREATMENT_COL)
    metrics = evaluate(eval_wrap, x_val, w_val, y_val)
    print(
        f"  Qini={metrics['qini']:.4f}  uplift@10%={metrics['uplift_at_10']:.4f}  "
        f"@20%={metrics['uplift_at_20']:.4f}  @30%={metrics['uplift_at_30']:.4f}  "
        f"(ATE={metrics['ate']:.4f})  responseAUC={metrics['response_auc']:.4f}  scoreStd={metrics['score_std']:.4f}"
    )

    # Финальная модель — на всех данных; отгружаем serve-обёртку.
    final_pipe = build_pipeline()
    final_pipe.fit(x.assign(**{TREATMENT_COL: w}), y)
    final_model = SoloModelUplift(final_pipe, FEATURES, TREATMENT_COL)

    info = build_model_info(specs, metrics, len(df))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final_model, "meta": info.model_dump()}, args.out, compress=3)
    size_mb = args.out.stat().st_size / 1_048_576
    print(f"  ✓ артефакт: {args.out}  ({size_mb:.2f} МБ)")

    card = {
        "name": "uplift",
        "task": "regression",
        "target": TARGET_COL,
        "approach": "S-learner (Treatment Dummy), HistGradientBoostingClassifier",
        "features": FEATURES,
        "metrics_holdout_ab": metrics,
        "sklearn_version": sklearn.__version__,
        "dataset": str(Path(*args.train_csv.parts[-2:])),
        "n_rows": len(df),
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    card_path = args.out.with_name("uplift.model_card.json")
    card_path.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  ✓ model card: {card_path}")


if __name__ == "__main__":
    main()
