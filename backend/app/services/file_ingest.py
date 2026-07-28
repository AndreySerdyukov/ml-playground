"""Парсинг загруженного табличного файла (CSV/Excel) в список записей.

Чистый слой без FastAPI: транспорт (api) читает байты и зовёт сюда, а результат
уходит в бизнес-логику инференса. Валидацию набора колонок делает уже сервис.
"""
from __future__ import annotations

import io

import pandas as pd


def parse_table(filename: str, raw: bytes, max_rows: int = 2000) -> list[dict[str, object]]:
    """Прочитать CSV/Excel из байтов в список записей (формат — по расширению файла)."""
    name = (filename or "").lower()
    try:
        if name.endswith((".xlsx", ".xls")):
            frame = pd.read_excel(io.BytesIO(raw))
        else:
            frame = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise ValueError(f"Не удалось прочитать файл: {exc}") from exc

    if frame.empty:
        raise ValueError("Файл пустой или без строк данных")
    if len(frame) > max_rows:
        raise ValueError(f"Слишком много строк: {len(frame)} (максимум {max_rows})")
    records: list[dict[str, object]] = frame.to_dict("records")
    return records
