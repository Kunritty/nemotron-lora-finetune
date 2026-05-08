"""
Registry of datasets and models used for the project. Reference
this module for ground truth information on dataset and model usage.
"""

from . import datasets as datasets
from .datasets import get_train_csv_path, get_train_df, get_train_val_split
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
from .transformers_chat import TransformersChatLLM
from .models import get_model, get_transformers_chat_model, list_models, default_model_id
from .ollama import DEFAULT_BASE_URL, DEFAULT_MODEL_ID, MODELS, OllamaModel

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
    "get_transformers_chat_model",
    "get_subsets",
    "get_train_csv_path",
    "get_train_df",
    "get_train_val_split",
    "holdout_split",
    "list_models",
    "TransformersChatLLM",
]

# convenience alias
STRATEGIES = datasets.STRATEGIES
