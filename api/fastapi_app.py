import logging
import os
import time
from pathlib import Path

import mlflow
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest


logging.basicConfig(level=logging.INFO)
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.getenv("INFERENCE_MODEL_NAME", "titanic-survival-predictor")
MODEL_STAGE = os.getenv("INFERENCE_MODEL_STAGE", "Production")
FEATURE_COLUMNS = [
    "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "IsAlone"
]

REQUESTS = Counter("inference_requests_total", "Total inference API requests", ["endpoint", "status"])
PREDICTIONS = Counter("model_predictions_total", "Total model predictions", ["model"])
PREDICTION_DURATION = Histogram("prediction_duration_seconds", "Prediction duration in seconds")

app = FastAPI(title="Titanic MLOps Inference API", version="1.0.0")


class PredictionRequest(BaseModel):
    data: list[list[float]] = Field(..., min_length=1)
    feature_names: list[str] = Field(..., min_length=1)


class PredictionResponse(BaseModel):
    predictions: list
    probabilities: list[list[float]]
    model_name: str
    model_stage: str


class ProductionModel:
    def __init__(self):
        self.model = None
        self.model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        self.loaded_at = None
        self.last_error = None

    def load(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        try:
            self.model = mlflow.pyfunc.load_model(self.model_uri)
            self.loaded_at = pd.Timestamp.utcnow().isoformat()
            self.last_error = None
            logging.info("Loaded production model from %s", self.model_uri)
            return True
        except Exception as error:
            self.last_error = str(error)
            logging.exception("Could not load production model")
            return False

    def predict(self, frame):
        if self.model is None and not self.load():
            raise RuntimeError(self.last_error or "Production model is not loaded")
        with PREDICTION_DURATION.time():
            predictions = self.model.predict(frame)
        PREDICTIONS.labels(model=MODEL_NAME).inc(len(predictions))
        probabilities = []
        for prediction in predictions:
            if int(prediction) == 1:
                probabilities.append([0.0, 1.0])
            else:
                probabilities.append([1.0, 0.0])
        return predictions.tolist(), probabilities


production_model = ProductionModel()


@app.on_event("startup")
def startup():
    production_model.load()


@app.get("/health")
def health():
    healthy = production_model.model is not None
    REQUESTS.labels(endpoint="health", status="200" if healthy else "503").inc()
    return {
        "status": "healthy" if healthy else "unhealthy",
        "model_loaded": healthy,
        "model_name": MODEL_NAME,
        "model_stage": MODEL_STAGE,
        "model_uri": production_model.model_uri,
        "loaded_at": production_model.loaded_at,
        "error": production_model.last_error,
    }


@app.get("/model-info")
def model_info():
    return {
        "model_name": MODEL_NAME,
        "stage": MODEL_STAGE,
        "tracking_uri": MLFLOW_URI,
        "features": FEATURE_COLUMNS,
        "loaded_at": production_model.loaded_at,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    started = time.perf_counter()
    try:
        frame = pd.DataFrame(request.data, columns=request.feature_names)
        missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing model features: {missing}")
        frame = frame[FEATURE_COLUMNS]
        predictions, probabilities = production_model.predict(frame)
        REQUESTS.labels(endpoint="predict", status="200").inc()
        logging.info("Predicted %s rows in %.4fs", len(frame), time.perf_counter() - started)
        return PredictionResponse(
            predictions=predictions,
            probabilities=probabilities,
            model_name=MODEL_NAME,
            model_stage=MODEL_STAGE,
        )
    except Exception as error:
        REQUESTS.labels(endpoint="predict", status="500").inc()
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/reload-model")
def reload_model():
    if not production_model.load():
        raise HTTPException(status_code=503, detail=production_model.last_error)
    return {"status": "model reloaded", "model_uri": production_model.model_uri}


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

