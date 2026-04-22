from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from registry.utils.datasets_utils import (
    holdout_split,
    category_leave_one_out_split,
    get_data_dir,
)

STRATEGIES = {
    "holdout": {
        "function": holdout_split,
    },
    "holdout-80-20": {
        "function": holdout_split,
        "args": {
            "ratio": 0.8,
        },
    },
    "leave-one-out-category": {
        "function": category_leave_one_out_split
    },
    "none": {
        "function": lambda train: (train, None),
    }
}
def _get_train_csv() -> Path:
    """Path to raw ``train.csv`` from the Kaggle competition download."""
    return get_data_dir() / "train.csv"

def get_train_df() -> pl.DataFrame:
    return pl.read_csv(_get_train_csv())

# TODO: Add support for other datasets (e.g. synthetic.csv)
def get_train_val_split(
    strategy: str = "holdout", 
    *, 
    holdout_category: str | None = None
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Retrieve a train/val split from the Kaggle train CSV.
    Args:
        strategy: Name of the split strategy to use.
        holdout_category: Category to hold out for validation. Only used for
            the leave-one-out strategy.
    Returns:
        (train, val): Tuple of train and validation DataFrames.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    split_fn = STRATEGIES[strategy]["function"]
    args = STRATEGIES[strategy]["args"]

    # Args specified in params override default strategy configs
    if holdout_category and split_fn is category_leave_one_out_split:
        args["holdout_category"] = holdout_category

    train = get_train_df()
    return split_fn(train, **args)