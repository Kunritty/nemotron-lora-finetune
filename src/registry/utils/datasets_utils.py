from __future__ import annotations

from pathlib import Path
import math

import kagglehub
import polars as pl

COMPETITION_SLUG = "nvidia-nemotron-model-reasoning-challenge"

# Filters for different categories of puzzles in the dataset. Splits based on 
# keywords present in specific prompt formats.

TRANSFORMATION_RULE = r"(?i)transformation rule"

PUZZLE_CATEGORIES: list[tuple[str, str] | tuple[str, str, str]] = [
    ("numerals", r"(?i)numeral system"),
    ("unit conversion", r"(?i)unit conversion"),
    ("bit manipulation", r"(?i)bit manipulation"),
    ("gravity", r"(?i)gravitational constant"),
    ("encryption", r"(?i)encryption"),
    ("equation_symbols", TRANSFORMATION_RULE, "no_digits"),
    ("equation_numeric", TRANSFORMATION_RULE, "has_digits"),
]

CATEGORY_NAMES: frozenset[str] = frozenset(spec[0] for spec in PUZZLE_CATEGORIES)
DEFAULT_HOLDOUT_CATEGORY: str = PUZZLE_CATEGORIES[0][0]

def get_data_dir() -> Path:
    return Path(kagglehub.competition_download(COMPETITION_SLUG))


def add_puzzle_categories(df: pl.DataFrame) -> pl.DataFrame:
    """Attach ``category`` from the first matching rule in :data:`PUZZLE_CATEGORIES`."""
    prompt = pl.col("prompt")
    expr: pl.Expr | None = None
    for spec in PUZZLE_CATEGORIES:
        if len(spec) == 2:
            name, pattern = spec
            cond = prompt.str.contains(pattern)
        else:
            name, pattern, mode = spec
            base = prompt.str.contains(pattern)
            if mode == "no_digits":
                cond = base & ~prompt.str.contains(r"\d")
            elif mode == "has_digits":
                cond = base & prompt.str.contains(r"\d")
            else:
                raise ValueError(f"unknown category mode: {mode!r}")
        lit = pl.lit(name)
        expr = pl.when(cond).then(lit) if expr is None else expr.when(cond).then(lit)
    cat = pl.lit(None).cast(pl.Utf8) if expr is None else expr.otherwise(None)
    return df.with_columns(category=cat)


def get_subsets(train: pl.DataFrame) -> dict[str, pl.DataFrame]:
    subsets: dict[str, pl.DataFrame] = {}
    for spec in PUZZLE_CATEGORIES:
        if len(spec) == 2:
            name, pattern = spec
            subsets[name] = train.filter(pl.col("prompt").str.contains(pattern))
        else:
            name, pattern, mode = spec
            base = train.filter(pl.col("prompt").str.contains(pattern))
            if mode == "no_digits":
                subsets[name] = base.filter(~pl.col("prompt").str.contains(r"\d"))
            elif mode == "has_digits":
                subsets[name] = base.filter(pl.col("prompt").str.contains(r"\d"))
            else:
                raise ValueError(f"unknown category mode: {mode!r}")
    return subsets

def _prepare_tagged_subsets(subsets: dict[str, pl.DataFrame]) -> list[tuple[str, pl.DataFrame]]:
    """Filters empty subsets and attaches the category name as a column."""
    return [
        (name, df.with_columns(pl.lit(name).alias("category")))
        for name, df in subsets.items()
        if not df.is_empty()
    ]

def _create_empty_with_category(schema_source: pl.DataFrame) -> pl.DataFrame:
    """Returns an empty DataFrame with the original schema plus a 'category' column."""
    return (
        schema_source
        .head(0)
        .with_columns(pl.lit(None).cast(pl.Utf8).alias("category"))
    )

def _shuffle_split(
    df: pl.DataFrame, ratio: float, seed: int | None
) -> tuple[pl.DataFrame, pl.DataFrame]:
    n = len(df)
    if n == 0:
        return df, df
    if seed is not None:
        shuffled = df.sample(fraction=1.0, shuffle=True, seed=seed)
    else:
        shuffled = df.sample(fraction=1.0, shuffle=True)
    k = int(math.floor(n * ratio))
    if ratio > 0 and k == 0:
        k = 1
    if ratio < 1.0 and k >= n > 1:
        k = n - 1
    return shuffled.head(k), shuffled.tail(n - k)




def holdout_split(
    train: pl.DataFrame,
    ratio: float = 0.8,
    *,
    seed: int | None = 0,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Stratified split with even distribution across categories. Also adds
    'category' column to splits for later reference.
    
    Args:
    train: The input DataFrame to split.
    ratio: The ratio of the dataset to allocate to the validation set.
    seed: The random seed to use for shuffling.

    Returns:
    train: Training DF with additional "category" column.
    val: Validation DF with additional "category" column.
    """
    tagged_subsets = _prepare_tagged_subsets(get_subsets(train))
    
    train_parts, val_parts = [], []
    for _, tagged_df in tagged_subsets:
        tr, va = _shuffle_split(tagged_df, ratio, seed)
        train_parts.append(tr)
        val_parts.append(va)

    if not train_parts:
        empty = _create_empty_with_category(train)
        return empty, empty
        
    return pl.concat(train_parts), pl.concat(val_parts)


def category_leave_one_out_split(
    train: pl.DataFrame,
    *,
    holdout_category: str = DEFAULT_HOLDOUT_CATEGORY,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Leave-one-out strategy split. Holdout category should be defined when called
    and run multiple times for full validation coverage.

    Args:
    train: The input DataFrame to split.
    holdout_category: The category to hold out for validation.

    Returns:
    train: Training DF with additional "category" column.
    val: Validation DF with additional "category" column.
    """
    if holdout_category not in CATEGORY_NAMES:
        raise ValueError(f"holdout_category {holdout_category!r} must be one of {sorted(CATEGORY_NAMES)}")

    tagged_subsets = _prepare_tagged_subsets(get_subsets(train))
    
    train_parts, val_parts = [], []
    for name, tagged_df in tagged_subsets:
        if name == holdout_category:
            val_parts.append(tagged_df)
        else:
            train_parts.append(tagged_df)

    empty = _create_empty_with_category(train)
    return (
        pl.concat(train_parts) if train_parts else empty,
        pl.concat(val_parts) if val_parts else empty
    )