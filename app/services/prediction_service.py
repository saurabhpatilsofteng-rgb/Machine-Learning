
from app.ml.model import load_model


class PredictionService:
    def __init__(self):
        self.model = load_model()

    def predict(self, area: float) -> float:
        prediction = self.model.predict([[area]])
        return float(prediction[0])


prediction_service = None


def get_prediction_service():
    global prediction_service

    if prediction_service is None:
        prediction_service = PredictionService()

    return prediction_service
