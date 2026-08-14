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

FEATURES = [
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
# EXPERIMENT
# =========================================================

# We train on clipped targets, but ALWAYS evaluate against
# the original validation target.

CLIP_VALUES = [
    None,   # baseline
    5000,
    7500,
    10000,
    15000,
]


def evaluate_model(clip_value):

    if clip_value is None:
        name = "BASELINE"
        y_train = train[TARGET].copy()
    else:
        name = f"CLIP_{clip_value}"
        y_train = train[TARGET].clip(upper=clip_value)

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
        train[FEATURES],
        y_train,
        cat_features=CAT_FEATURES,
    )

    fit_time = time.perf_counter() - start

    predictions = model.predict(valid[FEATURES])

    # DO NOT clip predictions.
    # We want to evaluate honestly against the original target.

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
        f"{name:<15} | "
        f"MAE={mae:.4f} | "
        f"RMSE={rmse:.4f} | "
        f"R2={r2:.4f} | "
        f"time={fit_time:.2f}s"
    )

    return {
        "name": name,
        "clip_value": clip_value,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "fit_time": fit_time,
    }


# =========================================================
# RUN
# =========================================================

print("=" * 90)
print("TARGET OUTLIER / CLIPPING EXPERIMENT")
print("=" * 90)

print("\nImportant:")
print("Training target may be clipped.")
print("Validation target is NEVER clipped.")
print("Predictions are NEVER clipped.")

results = []

for clip_value in CLIP_VALUES:
    results.append(
        evaluate_model(clip_value)
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
print("OUTLIER EXPERIMENT COMPLETE")
print("=" * 90)