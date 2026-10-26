# House Price Prediction — Linear Regression

A machine learning project that predicts house prices from house area using Scikit-learn and serves predictions through FastAPI.

## Technologies

- Python
- Pandas and NumPy
- Scikit-learn
- FastAPI and Uvicorn
- Pytest

## Project Features

- Loads training data from CSV
- Trains a Linear Regression model
- Evaluates predictions using MAE and R²
- Saves the trained model using Joblib
- Provides REST API prediction and health endpoints
- Validates incoming requests
- Includes automated API tests

## Setup

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Train the Model

```powershell
python train.py
```

## Start the API

```powershell
python -m uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/docs to test the API.

## Example Request

`POST /api/predict`

```json
{
  "area": 3200
}
```

## Run Tests

```powershell
python -m pytest
```

## Dataset Disclaimer

The included dataset is synthetic and intended for learning. Predictions are not suitable for real estate valuation.


# Linear Regression

## 1. What is Linear Regression?

Linear Regression is a supervised machine learning algorithm used to predict a numerical value.

**Examples:**
- Predicting house prices
- Predicting sales
- Predicting salary

## 2. Formula

`y = mx + b`

- `y` = predicted value
- `x` = input feature
- `m` = coefficient or slope
- `b` = intercept

## 3. Project Objective

Predict house prices based on the area of a house.

- Feature: `area`
- Target: `price`
- Algorithm: Linear Regression
- Library: Scikit-learn

## 4. How the Model Works

1. Load the dataset using Pandas.
2. Separate features and target values.
3. Create a Linear Regression model.
4. Train the model using `fit()`.
5. Predict prices using `predict()`.
6. Visualize the regression line using Matplotlib.

## 5. Important Methods

| Method | Purpose |
|---|---|
| `pd.read_csv()` | Reads a CSV file |
| `model.fit()` | Trains the model |
| `model.predict()` | Predicts values |
| `model.coef_` | Returns the coefficient |
| `model.intercept_` | Returns the intercept |

## 6. How to Run

Install the libraries:

```bash
python -m pip install pandas matplotlib scikit-learn
```

Run:

```bash
python house_price_prediction.py
```

## 7. Interview Questions

**Q1. What is Linear Regression?**

It is a supervised learning algorithm used to predict continuous numerical values.

**Q2. What is a feature?**

A feature is an input variable used by a model to make predictions.

**Q3. What is a target?**

A target is the value the model is trained to predict.

**Q4. What does `fit()` do?**

It trains the model using the provided training data.

**Q5. What does `predict()` do?**

It generates predictions for new input data.
