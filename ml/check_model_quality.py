"""
CI quality gate for the trained model. Fails (non-zero exit) if the
model's held-out MAPE exceeds MAX_ACCEPTABLE_MAPE, so a bad retrain
never gets built into the Docker image.

Run:
    python ml/check_model_quality.py
"""
import json
import os
import sys

MAX_ACCEPTABLE_MAPE = 30.0  # percent

HERE = os.path.dirname(os.path.abspath(__file__))
METRICS_PATH = os.path.join(HERE, "model", "metrics.json")


def main():
    if not os.path.exists(METRICS_PATH):
        print(f"No metrics file found at {METRICS_PATH} - did training run?")
        sys.exit(1)

    with open(METRICS_PATH) as f:
        metrics = json.load(f)

    mape = metrics.get("test_mape_pct")
    mae = metrics.get("test_mae_aed")
    print(f"Model quality check: MAE={mae:,.0f} AED, MAPE={mape:.2f}%")

    if mape is None or mape > MAX_ACCEPTABLE_MAPE:
        print(f"FAIL: MAPE {mape}% exceeds threshold of {MAX_ACCEPTABLE_MAPE}%")
        sys.exit(1)

    print("PASS: model meets quality threshold")


if __name__ == "__main__":
    main()
