
from pathlib import Path

import joblib

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models" / "house_price_model.pkl"


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run: python train.py"
        )

    return joblib.load(MODEL_PATH)
