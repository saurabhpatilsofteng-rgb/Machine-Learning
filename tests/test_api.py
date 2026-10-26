
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_prediction_endpoint():
    response = client.post(
        "/api/predict",
        json={"area": 3200},
    )

    assert response.status_code == 200
    assert response.json()["area"] == 3200
    assert abs(
        response.json()["predicted_price"] - 6400000
    ) < 0.01


def test_invalid_area():
    response = client.post(
        "/api/predict",
        json={"area": -100},
    )

    assert response.status_code == 422
