from collections.abc import Mapping
from typing import Any

import polars as pl

from chain.data_types import Entry


def entry_from_mapping(row: Mapping[str, Any]) -> Entry:
    raw_cat = row.get("category")
    category = None if raw_cat is None else str(raw_cat)
    return Entry(
        id=str(row["id"]).strip(),
        prompt=str(row["prompt"]),
        answer=str(row["answer"]),
        category=category,
    )


def convert_to_entry(row: pl.Series | pl.DataFrame) -> Entry:
    if isinstance(row, pl.DataFrame):
        if row.height != 1:
            raise ValueError("convert_to_entry expects a single-row DataFrame")
        return entry_from_mapping(row.row(0, named=True))
    if "category" in row:
        category = row["category"].item()
    else:
        category = None
    return Entry(
        id=str(row["id"].item()).strip(),
        prompt=str(row["prompt"].item()),
        answer=str(row["answer"].item()),
        category=None if category is None else str(category),
    )