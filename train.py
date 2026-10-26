
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "homeprices.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "house_price_model.pkl"


def train_model():
    data = pd.read_csv(DATA_PATH)

    X = data[["area"]]
    y = data["price"]

    model = LinearRegression()
    model.fit(X, y)

    predictions = model.predict(X)

    print("Slope:", model.coef_[0])
    print("Intercept:", model.intercept_)
    print("MAE:", mean_absolute_error(y, predictions))
    print("R2:", r2_score(y, predictions))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
