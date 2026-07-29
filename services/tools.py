"""
Non-ML tools available to the PropIQ agent. Kept as plain, testable
Python functions (no framework dependency) so they're trivial to unit
test and to swap for real data sources later (e.g. a live listings DB
instead of the static COMPARABLES table).
"""
from __future__ import annotations

import pandas as pd

_DATA_PATH = None


def _load_dataset():
    """Lazily load the same dataset the model trained on, used here as
    a stand-in 'listings database' for comparables/neighborhood stats."""
    import os

    global _DATA_PATH
    if _DATA_PATH is None:
        _DATA_PATH = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ml", "dubai_properties.csv"
        )
    return pd.read_csv(_DATA_PATH)


def estimate_mortgage(
    price_aed: float,
    down_payment_pct: float = 20.0,
    annual_rate_pct: float = 4.0,
    term_years: int = 25,
) -> dict:
    """Standard amortizing mortgage calculation."""
    if not (0 <= down_payment_pct <= 100):
        raise ValueError("down_payment_pct must be between 0 and 100")

    principal = price_aed * (1 - down_payment_pct / 100)
    monthly_rate = (annual_rate_pct / 100) / 12
    n_payments = term_years * 12

    if monthly_rate == 0:
        monthly_payment = principal / n_payments
    else:
        monthly_payment = (
            principal * monthly_rate * (1 + monthly_rate) ** n_payments
        ) / ((1 + monthly_rate) ** n_payments - 1)

    total_paid = monthly_payment * n_payments
    total_interest = total_paid - principal

    return {
        "loan_amount_aed": round(principal, 2),
        "down_payment_aed": round(price_aed - principal, 2),
        "monthly_payment_aed": round(monthly_payment, 2),
        "total_interest_aed": round(total_interest, 2),
        "term_years": term_years,
        "annual_rate_pct": annual_rate_pct,
    }


def get_comparable_listings(area: str, bedrooms: int, property_type: str | None = None, limit: int = 5) -> list[dict]:
    """Returns similar listings from the dataset for the given area/bedroom count."""
    df = _load_dataset()
    subset = df[(df["area"] == area) & (df["bedrooms"] == bedrooms)]
    if property_type:
        subset = subset[subset["property_type"] == property_type]

    if subset.empty:
        return []

    sample = subset.sample(min(limit, len(subset)), random_state=1)
    return sample[
        ["area", "property_type", "bedrooms", "size_sqft", "price_aed", "annual_rent_aed"]
    ].to_dict(orient="records")


def get_neighborhood_stats(area: str) -> dict:
    """Aggregate stats for an area: median price, price/sqft, typical yield."""
    df = _load_dataset()
    subset = df[df["area"] == area]
    if subset.empty:
        return {"area": area, "error": "No data for this area"}

    subset = subset.copy()
    subset["price_per_sqft"] = subset["price_aed"] / subset["size_sqft"]
    subset["gross_yield_pct"] = (subset["annual_rent_aed"] / subset["price_aed"]) * 100

    return {
        "area": area,
        "sample_size": int(len(subset)),
        "median_price_aed": round(float(subset["price_aed"].median()), -3),
        "median_price_per_sqft_aed": round(float(subset["price_per_sqft"].median()), 1),
        "median_gross_yield_pct": round(float(subset["gross_yield_pct"].median()), 2),
        "median_service_charge_psf": round(float(subset["service_charge_psf"].median()), 1),
    }
