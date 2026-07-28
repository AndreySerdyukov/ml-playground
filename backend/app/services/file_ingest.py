"""Parsing an uploaded tabular file (CSV/Excel) into a list of records.

A pure layer without FastAPI: the transport (api) reads the bytes and calls in here, and the result
goes to the inference business logic. Validation of the column set is done by the service.
"""
from __future__ import annotations

import io

import pandas as pd


def parse_table(filename: str, raw: bytes, max_rows: int = 2000) -> list[dict[str, object]]:
    """Read CSV/Excel from bytes into a list of records (format is determined by the file extension)."""
    name = (filename or "").lower()
    try:
        if name.endswith((".xlsx", ".xls")):
            frame = pd.read_excel(io.BytesIO(raw))
        else:
            frame = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise ValueError(f"Could not read the file: {exc}") from exc

    if frame.empty:
        raise ValueError("The file is empty or has no data rows")
    if len(frame) > max_rows:
        raise ValueError(f"Too many rows: {len(frame)} (maximum {max_rows})")
    records: list[dict[str, object]] = frame.to_dict("records")
    return records
