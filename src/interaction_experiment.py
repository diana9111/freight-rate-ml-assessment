from pathlib import Path
import time

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

df = pd.read_csv(DATA / "train-test.csv", parse_dates=["date"])


# =========================================================
# CHRONOLOGICAL SPLIT
# =========================================================

train = df[df["date"] < "2025-09-01"].copy()
valid = df[df["date"] >= "2025-09-01"].copy()


TARGET = "posted_rate"

BASE_FEATURES = [
    "pickup",
    "delivery",
    "equipment",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "weight",
    "market_index",
    "quote_signal",
    "year",
    "month",
    "dayofweek",
]

CAT_FEATURES = [
    "pickup",
    "delivery",
    "equipment",
]


# =========================================================
# DATE FEATURES
# =========================================================

def add_date_features(data):
    data = data.copy()

    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month
    data["dayofweek"] = data["date"].dt.dayofweek

    return data


train = add_date_features(train)
valid = add_date_features(valid)


# =========================================================
# FEATURE SETS
# =========================================================

feature_sets = {
    "BASELINE": [],

    "DISTANCE_SQUARED": [
        "distance_squared",
    ],

    "DISTANCE_LOG": [
        "distance_log",
    ],

    "DISTANCE_X_MARKET": [
        "distance_x_market",
    ],

    "DISTANCE_X_QUOTE": [
        "distance_x_quote",
    ],

    "ALL_INTERACTIONS": [
        "distance_squared",
        "distance_log",
        "distance_x_market",
        "distance_x_quote",
    ],
}


# =========================================================
# CREATE ENGINEERED FEATURES
# =========================================================

for data in [train, valid]:

    # Distance curvature
    data["distance_squared"] = data["distance"] ** 2

    # Log distance reduces the scale of very large distances
    data["distance_log"] = np.log1p(data["distance"].clip(lower=0))

    # Interactions
    data["distance_x_market"] = (
        data["distance"] * data["market_index"]
    )

    data["distance_x_quote"] = (
        data["distance"] * data["quote_signal"]
    )


# =========================================================
# MODEL EVALUATION
# =========================================================

def evaluate_model(name, extra_features):

    features = BASE_FEATURES + extra_features

    model = CatBoostRegressor(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=12,
        random_strength=1,
        loss_function="RMSE",
        verbose=False,
        random_seed=42,
    )

    start = time.perf_counter()

    model.fit(
        train[features],
        train[TARGET],
        cat_features=CAT_FEATURES,
    )

    fit_time = time.perf_counter() - start

    predictions = model.predict(valid[features])

    predictions = np.maximum(predictions, 0)

    mae = mean_absolute_error(
        valid[TARGET],
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            valid[TARGET],
            predictions,
        )
    )

    r2 = r2_score(
        valid[TARGET],
        predictions,
    )

    print(
        f"{name:<25} | "
        f"MAE={mae:.4f} | "
        f"RMSE={rmse:.4f} | "
        f"R2={r2:.4f} | "
        f"time={fit_time:.2f}s"
    )

    return {
        "name": name,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "fit_time": fit_time,
    }


# =========================================================
# RUN EXPERIMENT
# =========================================================

print("=" * 90)
print("TARGETED FEATURE INTERACTION EXPERIMENT")
print("=" * 90)

results = []

for name, extra_features in feature_sets.items():
    results.append(
        evaluate_model(
            name,
            extra_features,
        )
    )


# =========================================================
# RESULTS
# =========================================================

results_df = (
    pd.DataFrame(results)
    .sort_values("mae")
)

print("\n" + "=" * 90)
print("RESULTS SORTED BY MAE")
print("=" * 90)

print(
    results_df.to_string(index=False)
)


# =========================================================
# BEST
# =========================================================

print("\n" + "=" * 90)
print("CURRENT BEST")
print("=" * 90)

best = results_df.iloc[0]

print(f"Configuration: {best['name']}")
print(f"MAE:           {best['mae']:.4f}")
print(f"RMSE:          {best['rmse']:.4f}")
print(f"R2:            {best['r2']:.4f}")


print("\n" + "=" * 90)
print("INTERACTION EXPERIMENT COMPLETE")
print("=" * 90)