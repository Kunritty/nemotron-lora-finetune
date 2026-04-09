import os
from pathlib import Path
import kagglehub

COMPETITION_SLUG = "nvidia-nemotron-model-reasoning-challenge"
STRATEGIES = {
    "holdout-80-20"
}


def _get_data_dir() -> Path:
    """Download or fetch data directory"""
    return Path(kagglehub.competition_download(COMPETITION_SLUG))

def get_train_csv() -> Path:    
    """Get raw training data from Kaggle"""
    return _get_data_dir() / "train.csv"