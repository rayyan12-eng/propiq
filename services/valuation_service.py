"""
PropIQ Valuation Service
-------------------------
Serves the trained TensorFlow price model behind a small FastAPI app.
This is the "tool" the agent (agent.py) calls to get a price estimate.

Run standalone:
    uvicorn services.valuation_service:app --reload --port 8001
"""
import os

import joblib
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(os.path.dirname(HERE), "ml", "model")

app = FastAPI(title="PropIQ Valuation Service", version="1.0.0")

_model = None
_preprocessor = None
_feature_spec = None


def _load_artifacts():
    global _model, _preprocessor, _feature_spec
    if _model is None:
        _model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "price_model.keras"))
        _preprocessor = joblib.load(os.path.join(MODEL_DIR, "preprocessor.joblib"))
        _feature_spec = joblib.load(os.path.join(MODEL_DIR, "feature_spec.joblib"))
    return _model, _preprocessor, _feature_spec


VALID_AREAS = [
    "Downtown Dubai", "Dubai Marina", "Business Bay", "JVC",
    "Jumeirah Village Triangle", "Dubai Hills Estate", "Arjan",
    "Al Furjan", "Dubai Sports City", "Palm Jumeirah",
]
VALID_TYPES = ["Apartment", "Townhouse", "Villa"]


class PropertyFeatures(BaseModel):
    area: str = Field(..., description=f"One of: {VALID_AREAS}")
    property_type: str = Field(..., description=f"One of: {VALID_TYPES}")
    bedrooms: int = Field(..., ge=0, le=10)
    size_sqft: float = Field(..., gt=100)
    building_age_years: int = Field(0, ge=0, le=60)
    floor: int = Field(0, ge=0, le=120)
    has_pool: bool = False
    has_gym_building: bool = False
    near_metro: bool = False
    parking_spaces: int = Field(1, ge=0, le=10)
    service_charge_psf: float = Field(14.0, ge=0)


class PredictionResponse(BaseModel):
    predicted_price_aed: float
    predicted_price_range_aed: list[float]
    estimated_annual_rent_aed: float
    estimated_gross_yield_pct: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: PropertyFeatures):
    if features.area not in VALID_AREAS:
        raise HTTPException(400, f"Unknown area '{features.area}'. Valid: {VALID_AREAS}")
    if features.property_type not in VALID_TYPES:
        raise HTTPException(400, f"Unknown property_type. Valid: {VALID_TYPES}")

    model, preprocessor, spec = _load_artifacts()

    import pandas as pd

    row = pd.DataFrame(
        [
            {
                "bedrooms": features.bedrooms,
                "size_sqft": features.size_sqft,
                "building_age_years": features.building_age_years,
                "floor": features.floor,
                "has_pool": int(features.has_pool),
                "has_gym_building": int(features.has_gym_building),
                "near_metro": int(features.near_metro),
                "parking_spaces": features.parking_spaces,
                "service_charge_psf": features.service_charge_psf,
                "area": features.area,
                "property_type": features.property_type,
            }
        ]
    )
    X = preprocessor.transform(row)
    pred_log = model.predict(X, verbose=0).flatten()[0]
    price = float(np.expm1(pred_log))

    # simple uncertainty band based on held-out MAPE (~13%)
    low, high = price * 0.87, price * 1.13
    annual_rent = price * 0.065  # midpoint of Dubai's typical 5.5-7.5% gross yield band
    gross_yield = (annual_rent / price) * 100

    return PredictionResponse(
        predicted_price_aed=round(price, -3),
        predicted_price_range_aed=[round(low, -3), round(high, -3)],
        estimated_annual_rent_aed=round(annual_rent, -2),
        estimated_gross_yield_pct=round(gross_yield, 2),
    )
