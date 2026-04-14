"""
Registry of datasets and models used for the project. Reference
this module for ground truth information on dataset and model usage.
"""

from .datasets import (
    PUZZLE_CATEGORIES,
    STRATEGIES,
    add_puzzle_categories,
    get_train_csv,
    get_train_val_split,
    load_train_labeled,
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
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL_ID",
    "MODELS",
    "OllamaModel",
    "PUZZLE_CATEGORIES",
    "STRATEGIES",
    "add_puzzle_categories",
    "default_model_id",
    "get_model",
    "get_train_csv",
    "get_train_val_split",
    "list_models",
    "load_train_labeled",
]
