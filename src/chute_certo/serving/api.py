"""FastAPI app for Brasileirão match result prediction."""

import os
from contextlib import asynccontextmanager

import mlflow.sklearn
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from chute_certo.predictions.store import load_predictions


class PredictRequest(BaseModel):
    home_form_points: float
    home_form_scored: float
    home_form_conceded: float
    away_form_points: float
    away_form_scored: float
    away_form_conceded: float


class PredictResponse(BaseModel):
    home_win: float
    draw: float
    away_win: float


_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_uri = os.environ["MODEL_URI"]
    _state["model"] = mlflow.sklearn.load_model(model_uri)
    yield
    _state.clear()


app = FastAPI(title="Chute Certo API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/predictions")
def get_predictions():
    return load_predictions()


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    model = _state["model"]
    X = np.array(
        [
            [
                req.home_form_points,
                req.home_form_scored,
                req.home_form_conceded,
                req.away_form_points,
                req.away_form_scored,
                req.away_form_conceded,
            ]
        ]
    )
    proba = model.predict_proba(X)[0]
    proba_map = dict(zip(model.classes_, proba, strict=True))
    return PredictResponse(
        home_win=proba_map.get("H", 0.0),
        draw=proba_map.get("D", 0.0),
        away_win=proba_map.get("A", 0.0),
    )
