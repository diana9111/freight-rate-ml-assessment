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
# CLIPPING VALUES
# =========================================================

CLIP_VALUES = [
    12000,
    13000,
    14000,
    15000,
    16000,
    17500,
    20000,
]


# =========================================================
# EXPERIMENT
# =========================================================

def evaluate_clip(clip_value):

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

    # IMPORTANT:
    # No clipping of predictions.
    # Validation target is completely untouched.

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
        f"CLIP={clip_value:>5} | "
        f"MAE={mae:.4f} | "
        f"RMSE={rmse:.4f} | "
        f"R2={r2:.4f} | "
        f"time={fit_time:.2f}s"
    )

    return {
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
print("CLIPPING THRESHOLD FINE-TUNING")
print("=" * 90)

results = []

for clip_value in CLIP_VALUES:
    results.append(
        evaluate_clip(clip_value)
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
print("BEST CLIPPING THRESHOLD")
print("=" * 90)

best = results_df.iloc[0]

print(f"Clip value: {best['clip_value']}")
print(f"MAE:        {best['mae']:.4f}")
print(f"RMSE:       {best['rmse']:.4f}")
print(f"R2:         {best['r2']:.4f}")

print("\n" + "=" * 90)
print("CLIPPING FINE-TUNING COMPLETE")
print("=" * 90)