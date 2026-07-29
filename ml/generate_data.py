"""
Generates a synthetic but realistic Dubai residential sales dataset.

In a real deployment, swap this out for actual data (e.g. Dubai Land
Department transaction exports, or a Kaggle Dubai real estate dataset).
The synthetic generator exists so the rest of the pipeline (training,
serving, CI) is fully runnable and testable without a data dependency.
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

# (area, base AED/sqft, desirability multiplier)
AREAS = [
    ("Downtown Dubai", 2200, 1.35),
    ("Dubai Marina", 1850, 1.25),
    ("Business Bay", 1500, 1.10),
    ("JVC", 950, 0.85),
    ("Jumeirah Village Triangle", 1000, 0.87),
    ("Dubai Hills Estate", 1650, 1.15),
    ("Arjan", 900, 0.80),
    ("Al Furjan", 1050, 0.90),
    ("Dubai Sports City", 850, 0.78),
    ("Palm Jumeirah", 3200, 1.60),
]

PROPERTY_TYPES = ["Apartment", "Townhouse", "Villa"]


def generate(n_rows: int = 6000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_rows):
        area_name, base_psf, desirability = AREAS[rng.integers(0, len(AREAS))]
        ptype = rng.choice(PROPERTY_TYPES, p=[0.72, 0.18, 0.10])

        bedrooms = int(rng.choice([0, 1, 2, 3, 4, 5], p=[0.05, 0.30, 0.32, 0.20, 0.10, 0.03]))
        if ptype == "Villa":
            bedrooms = max(bedrooms, 3)
        if ptype == "Townhouse":
            bedrooms = max(bedrooms, 2)

        base_size = {0: 450, 1: 750, 2: 1100, 3: 1600, 4: 2400, 5: 3200}[bedrooms]
        size_sqft = max(350, rng.normal(base_size, base_size * 0.12))

        building_age = max(0, rng.integers(0, 20))
        floor = rng.integers(1, 45) if ptype == "Apartment" else 0

        has_pool = int(rng.random() < (0.35 if ptype != "Apartment" else 0.10))
        has_gym_building = int(rng.random() < (0.85 if ptype == "Apartment" else 0.20))
        near_metro = int(rng.random() < 0.4)
        parking_spaces = int(rng.choice([0, 1, 2, 3], p=[0.05, 0.45, 0.40, 0.10]))
        service_charge_psf = round(rng.uniform(8, 22), 1)

        age_decay = max(0.55, 1 - 0.02 * building_age)
        type_premium = {"Apartment": 1.0, "Townhouse": 1.08, "Villa": 1.18}[ptype]

        psf = (
            base_psf
            * desirability
            * age_decay
            * type_premium
            * (1.06 if near_metro else 1.0)
            * (1.04 if has_pool else 1.0)
            * (1 + rng.normal(0, 0.05))
        )

        price = psf * size_sqft
        price = max(300_000, price)

        annual_rent = price * rng.uniform(0.055, 0.075)  # rough Dubai gross yield range

        rows.append(
            {
                "area": area_name,
                "property_type": ptype,
                "bedrooms": bedrooms,
                "size_sqft": round(size_sqft),
                "building_age_years": building_age,
                "floor": floor,
                "has_pool": has_pool,
                "has_gym_building": has_gym_building,
                "near_metro": near_metro,
                "parking_spaces": parking_spaces,
                "service_charge_psf": service_charge_psf,
                "price_aed": round(price),
                "annual_rent_aed": round(annual_rent),
            }
        )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate()
    df.to_csv("ml/dubai_properties.csv", index=False)
    print(f"Wrote {len(df)} rows to ml/dubai_properties.csv")
    print(df.head())
