
from pydantic import BaseModel, Field


class HouseRequest(BaseModel):
    area: float = Field(gt=0, description="House area in square feet")


class PredictionResponse(BaseModel):
    area: float
    predicted_price: float
