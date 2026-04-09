"""
Registry of datasets and models used for the project. Reference 
this module for ground truth information on dataset and model usage.
"""

from .datasets import get_train_csv
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
    "default_model_id",
    "get_model",
    "get_train_csv",
    "list_models",
]