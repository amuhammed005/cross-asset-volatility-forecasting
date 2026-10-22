"""FastAPI application serving GARCH-family volatility forecasts."""

import sqlite3

from fastapi import FastAPI
from pydantic import BaseModel

from config import settings
from data.repository import SQLRepository
from model.garch import GarchModel


class FitIn(BaseModel):
    ticker: str
    use_new_data: bool
    n_observations: int
    p: int
    q: int
    vol_model: str = "GARCH"


class FitOut(FitIn):
    success: bool
    message: str


class PredictIn(BaseModel):
    ticker: str
    n_days: int


class PredictOut(PredictIn):
    success: bool
    forecast: dict
    message: str


def build_model(ticker: str, use_new_data: bool) -> GarchModel:
    """Instantiate a GarchModel with a fresh database connection and repository."""
    connection = sqlite3.connect(settings.db_name, check_same_thread=False)
    repo = SQLRepository(connection=connection)
    return GarchModel(ticker=ticker, repo=repo, use_new_data=use_new_data)


app = FastAPI(title="Cross-Asset Volatility Forecasting API")


@app.get("/hello", status_code=200)
def hello():
    """Health check endpoint."""
    return {"message": "Hello world"}


@app.post("/fit", status_code=200, response_model=FitOut)
def fit_model(request: FitIn):
    """Fit a GARCH-family model and save it to disk."""
    response = request.dict()
    try:
        model = build_model(ticker=request.ticker, use_new_data=request.use_new_data)
        model.wrangle_data(n_observations=request.n_observations)
        model.fit(p=request.p, q=request.q, vol_model=request.vol_model)
        filename = model.dump()
        response["success"] = True
        response["message"] = f"Trained and saved `{filename}`."
    except Exception as e:
        response["success"] = False
        response["message"] = str(e)
    return response


@app.post("/predict", status_code=200, response_model=PredictOut)
def get_prediction(request: PredictIn):
    """Load the latest saved model and generate a volatility forecast."""
    response = request.dict()
    try:
        model = build_model(ticker=request.ticker, use_new_data=False)
        model.load()
        prediction = model.predict_volatility(horizon=request.n_days)
        response["success"] = True
        response["forecast"] = prediction
        response["message"] = ""
    except Exception as e:
        response["success"] = False
        response["forecast"] = {}
        response["message"] = str(e)
    return response