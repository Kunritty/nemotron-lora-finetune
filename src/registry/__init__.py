"""
Registry of datasets and models used for the project. Reference
this module for ground truth information on dataset and model usage.
"""

from . import datasets as datasets
from .datasets import get_train_csv, get_train_val_split
from .utils import datasets_utils as datasets_utils
from .utils.datasets_utils import (
    CATEGORY_NAMES,
    COMPETITION_SLUG,
    DEFAULT_HOLDOUT_CATEGORY,
    PUZZLE_CATEGORIES,
    add_puzzle_categories,
    category_leave_one_out_split,
    get_data_dir,
    get_subsets,
    holdout_split,
)
from .models import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL_ID,
    MODELS,
    OllamaModel,
    default_model_id,
    get_model,
    list_models,
)

__all__ = [
    "CATEGORY_NAMES",
    "COMPETITION_SLUG",
    "DEFAULT_BASE_URL",
    "DEFAULT_HOLDOUT_CATEGORY",
    "DEFAULT_MODEL_ID",
    "MODELS",
    "OllamaModel",
    "PUZZLE_CATEGORIES",
    "STRATEGIES",
    "add_puzzle_categories",
    "category_leave_one_out_split",
    "default_model_id",
    "datasets",
    "datasets_utils",
    "get_data_dir",
    "get_model",
    "get_subsets",
    "get_train_csv",
    "get_train_val_split",
    "holdout_split",
    "list_models",
]

# convenience alias
STRATEGIES = datasets.STRATEGIES
