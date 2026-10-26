
from fastapi import APIRouter, HTTPException

from app.api.schemas import HouseRequest, PredictionResponse
from app.services.prediction_service import get_prediction_service

router = APIRouter()


@router.get("/health")
def health_check():
    return {"status": "healthy"}


@router.post("/predict", response_model=PredictionResponse)
def predict_price(house: HouseRequest):
    try:
        service = get_prediction_service()
        price = service.predict(house.area)

        return PredictionResponse(
            area=house.area,
            predicted_price=price,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
