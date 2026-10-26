
from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(
    title="House Price Prediction API",
    description="Linear Regression model served through FastAPI",
    version="1.0.0",
)

app.include_router(router, prefix="/api", tags=["Prediction"])
