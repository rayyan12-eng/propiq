"""
Trains a TensorFlow regression model to predict Dubai property prices
(AED) from structured features, and saves it together with the
preprocessing pipeline so the serving layer can reproduce it exactly.

Run:
    python ml/train_model.py
Outputs:
    ml/model/price_model.keras
    ml/model/preprocessor.joblib
    ml/model/metrics.json   <- read by CI to gate regressions
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "bedrooms",
    "size_sqft",
    "building_age_years",
    "floor",
    "has_pool",
    "has_gym_building",
    "near_metro",
    "parking_spaces",
    "service_charge_psf",
]
CATEGORICAL_FEATURES = ["area", "property_type"]
TARGET = "price_aed"

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "dubai_properties.csv")
MODEL_DIR = os.path.join(HERE, "model")


def build_model(input_dim: int) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.15),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mae", metrics=["mae", "mape"])
    return model


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    # log-transform the target: prices are right-skewed, this stabilizes training
    y = np.log1p(df[TARGET].values)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    model = build_model(X_train_t.shape[1])

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True
    )
    model.fit(
        X_train_t,
        y_train,
        validation_split=0.15,
        epochs=100,
        batch_size=64,
        callbacks=[early_stop],
        verbose=2,
    )

    # Evaluate on real AED scale (invert the log1p transform)
    preds_log = model.predict(X_test_t, verbose=0).flatten()
    preds_aed = np.expm1(preds_log)
    actual_aed = np.expm1(y_test)

    mae_aed = float(np.mean(np.abs(preds_aed - actual_aed)))
    mape_pct = float(np.mean(np.abs((preds_aed - actual_aed) / actual_aed)) * 100)

    print(f"Test MAE: {mae_aed:,.0f} AED")
    print(f"Test MAPE: {mape_pct:.2f}%")

    model.save(os.path.join(MODEL_DIR, "price_model.keras"))
    joblib.dump(preprocessor, os.path.join(MODEL_DIR, "preprocessor.joblib"))
    joblib.dump(
        {"numeric": NUMERIC_FEATURES, "categorical": CATEGORICAL_FEATURES},
        os.path.join(MODEL_DIR, "feature_spec.joblib"),
    )

    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump({"test_mae_aed": mae_aed, "test_mape_pct": mape_pct}, f, indent=2)


if __name__ == "__main__":
    main()
