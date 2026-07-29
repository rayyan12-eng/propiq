from fastapi.testclient import TestClient

from services.valuation_service import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_predict_valid_payload():
    payload = {
        "area": "JVC",
        "property_type": "Apartment",
        "bedrooms": 2,
        "size_sqft": 1100,
        "building_age_years": 3,
        "near_metro": True,
        "has_pool": False,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["predicted_price_aed"] > 0
    assert len(body["predicted_price_range_aed"]) == 2
    assert body["predicted_price_range_aed"][0] < body["predicted_price_aed"] < body["predicted_price_range_aed"][1]
    assert body["estimated_annual_rent_aed"] > 0
    assert 0 < body["estimated_gross_yield_pct"] < 20


def test_predict_unknown_area_rejected():
    payload = {
        "area": "Atlantis",
        "property_type": "Apartment",
        "bedrooms": 2,
        "size_sqft": 1000,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 400


def test_predict_invalid_bedrooms_rejected():
    payload = {
        "area": "JVC",
        "property_type": "Apartment",
        "bedrooms": -1,
        "size_sqft": 1000,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422  # pydantic validation error (bedrooms ge=0 constraint)
