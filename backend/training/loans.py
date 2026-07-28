"""Обучение и экспорт РЕАЛЬНОЙ модели «Loans» (одобрение кредита) для ML Playground.

Исходные ноутбуки – в `notebooks/loans/`, данные (train/test) – в `data/loans/` (в самом репо).
Задача – бинарная классификация: одобрить (`Approved`) или отклонить (`Rejected`) заявку.
Данные АНОНИМИЗИРОВАНЫ: колонки безымянные (`C1, N2, C4_enc, …`), смысл скрыт для
конфиденциальности. Все признаки подаём в модель как числа (`*_enc` – произвольные категор.
коды, но древесные ансамбли это терпят) – ровно как в исходном ноутбуке.

Модель воспроизводит финальный ноутбук `ML_Classification_Credits_Model.ipynb` –
`StackingClassifier` из трёх базовых (GradientBoosting + AdaBoost + RandomForest) с
LogisticRegression-мета (`stack_method="predict_proba"`). Сверху – `SimpleImputer(median)`
на случай пропусков в живой форме. Целевая 1/0 маппится в строковые классы `Approved`/`Rejected`,
чтобы UI подписал бары вероятностей человекочитаемо.

Итог – бандл `{"model": <fitted pipeline>, "meta": ModelInfo(...).model_dump()}` в
`backend/models/loans.joblib`; его читает реестр (`app/repositories/model_registry.py`),
форма строится из `features`. Рядом – `loans.model_card.json` с метриками и провенансом.
Этот файл – ЕДИНСТВЕННЫЙ источник истины для описания модели Loans (`ModelInfo`).

Запуск (из каталога backend, в .venv):
    python training/loans.py
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
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Делаем пакет `app` импортируемым при запуске файла напрямую (training/ -> backend/).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.schemas.predict import FeatureSpec, ModelInfo, ThresholdPoint

# Колонка-таргет в датасете.
TARGET_COL = "Target"
# Целевая 1/0 -> человекочитаемые метки классов (их увидит UI).
CLASS_LABELS: dict[float, str] = {1.0: "Approved", 0.0: "Rejected"}
# Положительный класс для метрик (precision/recall/ROC-AUC).
POSITIVE_LABEL = "Approved"

# Признаки в порядке колонок датасета (= порядок полей формы). Все числовые.
# Имена АНОНИМНЫЕ – показываем как есть (без выдуманных лейблов).
FEATURE_COLUMNS: list[str] = [
    "C1", "N2", "N3", "C4_enc", "C5_enc", "C6_enc", "N7",
    "C8", "C9", "N10", "C11", "C12_enc", "N13", "N14",
]

# Значения одной реальной (одобренной) строки train – для кнопки «Try an example».
# Ключи держим в паре с FEATURE_COLUMNS (build_feature_specs читает по каждому имени).
FEATURE_EXAMPLES: dict[str, float] = {
    "C1": 1.0, "N2": 28.0, "N3": 2.0, "C4_enc": 2.0, "C5_enc": 4.0,
    "C6_enc": 8.0, "N7": 4.165, "C8": 1.0, "C9": 1.0, "N10": 2.0,
    "C11": 1.0, "C12_enc": 2.0, "N13": 181.0, "N14": 1.0,
}

# Данные лежат в самом репо: training/ -> backend/ -> repo-root(parents[2]) -> data/loans.
DEFAULT_TRAIN_CSV = Path(__file__).resolve().parents[2] / "data" / "loans" / "train.csv"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "models" / "loans.joblib"


def load_data(csv_path: Path) -> pd.DataFrame:
    """Прочитать датасет, убрать дубли и строки без таргета."""
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates()
    df = df.dropna(subset=[TARGET_COL])
    return df.reset_index(drop=True)


def build_feature_specs() -> list[FeatureSpec]:
    """FeatureSpec для формы: все 14 признаков – числовые, лейбл = сырое анонимное имя."""
    return [
        FeatureSpec(
            name=name, type="number", label=name, example=FEATURE_EXAMPLES[name]
        )
        for name in FEATURE_COLUMNS
    ]


def build_pipeline() -> Pipeline:
    """Импьютер + стек-ансамбль (воспроизводит финальный ноутбук).

    Базовые: GradientBoosting / AdaBoost(стампы) / RandomForest; мета: StandardScaler+LogReg.
    Гиперпараметры – из ноутбука; `random_state=42` добавлен для детерминированного артефакта.
    """
    grad = GradientBoostingClassifier(
        learning_rate=0.11,
        max_features=None,
        min_samples_leaf=1,
        min_samples_split=4,
        n_estimators=100,
        subsample=0.9,
        random_state=42,
    )
    forest = RandomForestClassifier(
        criterion="gini",
        max_depth=4,
        max_features=None,
        min_samples_leaf=2,
        min_samples_split=2,
        n_estimators=1000,
        random_state=42,
        n_jobs=-1,
    )
    # sklearn 1.9: параметр называется `estimator` (`base_estimator` устарел в 1.2, удалён в 1.4).
    ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
        n_estimators=200,
        learning_rate=0.1,
        random_state=42,
    )
    # L1-регуляризация: в sklearn 1.9 `penalty="l1"` → `l1_ratio=1` (даёт те же коэффициенты).
    final_estimator = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1, l1_ratio=1, solver="liblinear", random_state=42),
    )
    stacking = StackingClassifier(
        estimators=[("grad", grad), ("ada", ada), ("forest", forest)],
        final_estimator=final_estimator,
        stack_method="predict_proba",
    )
    # Импьютер сверху – чтобы пропуск в живой форме не ронял инференс (в обучении данные полные).
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", stacking),
        ]
    )


def evaluate(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    """Метрики на holdout: accuracy, ROC-AUC, precision/recall/F1 (класс Approved), матрица ошибок."""
    pred = model.predict(x_test)
    classes = list(model.classes_)
    pos_idx = classes.index(POSITIVE_LABEL)
    proba_pos = model.predict_proba(x_test)[:, pos_idx]
    y_true_bin = (y_test.to_numpy() == POSITIVE_LABEL).astype(int)
    # Порядок меток для матрицы ошибок – единый источник истины CLASS_LABELS.
    labels = [CLASS_LABELS[0.0], CLASS_LABELS[1.0]]
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_true_bin, proba_pos)),
        "precision": float(precision_score(y_test, pred, pos_label=POSITIVE_LABEL)),
        "recall": float(recall_score(y_test, pred, pos_label=POSITIVE_LABEL)),
        "f1": float(f1_score(y_test, pred, pos_label=POSITIVE_LABEL)),
        # Матрица ошибок в порядке labels=[Rejected, Approved].
        "confusion_matrix": confusion_matrix(y_test, pred, labels=labels).tolist(),
        "labels": labels,
    }


def threshold_curve(y_bin: np.ndarray, proba_pos: np.ndarray) -> list[ThresholdPoint]:
    """Кривая порог→(precision, recall) класса Approved по OOF-вероятностям – для слайдера в UI."""
    pts: list[ThresholdPoint] = []
    for t in np.round(np.arange(0.05, 0.96, 0.05), 2):
        y_pred = (proba_pos >= t).astype(int)
        pts.append(
            ThresholdPoint(
                threshold=float(t),
                precision=float(precision_score(y_bin, y_pred, zero_division=0)),
                recall=float(recall_score(y_bin, y_pred, zero_division=0)),
            )
        )
    return pts


def build_model_info(
    specs: list[FeatureSpec], metrics: dict[str, Any], n_rows: int, curve: list[ThresholdPoint]
) -> ModelInfo:
    """Карточка модели для реестра/формы (is_stub=False – бейдж «demo» исчезнет)."""
    auc = metrics["roc_auc"]
    acc = metrics["accuracy"]
    # Метрики – с holdout; n_rows – размер датасета, на котором обучена финальная модель.
    description = (
        f"Credit approval classifier – ROC-AUC {auc:.2f}, accuracy {acc:.0%} "
        f"(stacked ensemble on {n_rows:,} anonymised applications)"
    )
    return ModelInfo(
        name="loans",
        task="classification",
        target="credit decision",
        category="Classification",
        emoji="",
        is_stub=False,
        description=description,
        features=specs,
        # Интерактивный порог в UI: класс-positive = Approved, слайдер стартует с argmax-порога 0.5.
        positive_class=POSITIVE_LABEL,
        default_threshold=0.5,
        threshold_curve=curve,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Обучить и экспортировать модель Loans.")
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if not args.train_csv.exists():
        raise SystemExit(f"Не найден датасет: {args.train_csv}")

    df = load_data(args.train_csv)
    specs = build_feature_specs()

    x = df[FEATURE_COLUMNS]
    y = df[TARGET_COL].astype(float).map(CLASS_LABELS)
    if y.isna().any():
        raise SystemExit("В таргете есть значения вне {0, 1} – проверьте датасет.")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    print(
        f"Обучение: {len(FEATURE_COLUMNS)} фич, стек-ансамбль, "
        f"строк={len(df)} (train={len(x_train)}, test={len(x_test)})"
    )

    # Оценка на holdout.
    eval_model = build_pipeline()
    eval_model.fit(x_train, y_train)
    metrics = evaluate(eval_model, x_test, y_test)
    metrics["n_train"] = len(x_train)
    metrics["n_test"] = len(x_test)
    print(
        f"  accuracy={metrics['accuracy']:.3f}  ROC-AUC={metrics['roc_auc']:.3f}  "
        f"precision={metrics['precision']:.3f}  recall={metrics['recall']:.3f}  "
        f"F1={metrics['f1']:.3f}"
    )
    print(f"  confusion[Rejected/Approved]={metrics['confusion_matrix']}")

    # Финальная модель – на всех данных.
    final_model = build_pipeline()
    final_model.fit(x, y)

    # Кривая порог→(precision, recall) по 5-fold OOF – для интерактивного слайдера в UI.
    y_bin = (y == POSITIVE_LABEL).astype(int).to_numpy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    base_proba = cross_val_predict(
        build_pipeline(), x, y_bin, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    curve = threshold_curve(y_bin, base_proba)

    info = build_model_info(specs, metrics, len(df), curve)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final_model, "meta": info.model_dump()}, args.out, compress=3)
    size_mb = args.out.stat().st_size / 1_048_576
    print(f"  ✓ артефакт: {args.out}  ({size_mb:.2f} МБ)")

    # Model card (провенанс + метрики) рядом с артефактом; реестр читает только *.joblib.
    card = {
        "name": "loans",
        "task": "classification",
        "target": TARGET_COL,
        "classes": [CLASS_LABELS[0.0], CLASS_LABELS[1.0]],
        "estimator": "stacking(grad+ada+forest -> logreg)",
        "features": FEATURE_COLUMNS,
        "metrics": metrics,
        "sklearn_version": sklearn.__version__,
        # Только хвост пути – без абсолютного пути с юзернеймом (не тащим PII в репо).
        "dataset": str(Path(*args.train_csv.parts[-2:])),
        "n_rows": len(df),
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    card_path = args.out.with_name("loans.model_card.json")
    card_path.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  ✓ model card: {card_path}")


if __name__ == "__main__":
    main()
