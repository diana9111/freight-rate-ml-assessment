from pathlib import Path
import time

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

df = pd.read_csv(DATA / "train-test.csv", parse_dates=["date"])

# Same chronological split
train = df[df["date"] < "2025-09-01"].copy()
valid = df[df["date"] >= "2025-09-01"].copy()

TARGET = "posted_rate"

CAT_FEATURES = [
    "pickup",
    "delivery",
    "equipment",
]


def add_features(data, engineered=False):
    data = data.copy()

    # Existing date features
    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month
    data["dayofweek"] = data["date"].dt.dayofweek

    if engineered:
        # -------------------------------------------------
        # Geographic differences
        # -------------------------------------------------
        data["lat_diff"] = (
            data["delivery_lat"] - data["pickup_lat"]
        )

        data["lon_diff"] = (
            data["delivery_lon"] - data["pickup_lon"]
        )

        data["abs_lat_diff"] = data["lat_diff"].abs()
        data["abs_lon_diff"] = data["lon_diff"].abs()

        # -------------------------------------------------
        # Distance transformations
        # -------------------------------------------------
        data["log_distance"] = np.log1p(
            data["distance"].clip(lower=0)
        )

        data["sqrt_distance"] = np.sqrt(
            data["distance"].clip(lower=0)
        )

        data["distance_squared"] = (
            data["distance"] ** 2
        )

        # -------------------------------------------------
        # Simple rate-related interactions
        # -------------------------------------------------
        data["distance_market"] = (
            data["distance"] * data["market_index"]
        )

        data["distance_quote"] = (
            data["distance"] * data["quote_signal"]
        )

    return data


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

ENGINEERED_FEATURES = BASE_FEATURES + [
    "lat_diff",
    "lon_diff",
    "abs_lat_diff",
    "abs_lon_diff",
    "log_distance",
    "sqrt_distance",
    "distance_squared",
    "distance_market",
    "distance_quote",
]


def evaluate(name, train_df, valid_df, features):
    X_train = train_df[features]
    y_train = train_df[TARGET]

    X_valid = valid_df[features]
    y_valid = valid_df[TARGET]

    model = CatBoostRegressor(
        iterations=500,
        depth=8,
        learning_rate=0.05,
        loss_function="RMSE",
        verbose=False,
        random_seed=42,
    )

    start = time.perf_counter()

    model.fit(
        X_train,
        y_train,
        cat_features=CAT_FEATURES,
    )

    fit_time = time.perf_counter() - start

    predictions = model.predict(X_valid)

    mae = mean_absolute_error(y_valid, predictions)
    rmse = np.sqrt(mean_squared_error(y_valid, predictions))
    r2 = r2_score(y_valid, predictions)

    print(f"\n{name}")
    print("-" * 60)
    print(f"MAE:       {mae:.4f}")
    print(f"RMSE:      {rmse:.4f}")
    print(f"R2:        {r2:.4f}")
    print(f"Fit time:  {fit_time:.2f}s")

    return mae, rmse, r2


# ---------------------------------------------------------
# Prepare both versions
# ---------------------------------------------------------

train_base = add_features(train, engineered=False)
valid_base = add_features(valid, engineered=False)

train_engineered = add_features(train, engineered=True)
valid_engineered = add_features(valid, engineered=True)


# ---------------------------------------------------------
# Experiment A — current benchmark
# ---------------------------------------------------------

evaluate(
    "A. CURRENT BASELINE FEATURES",
    train_base,
    valid_base,
    BASE_FEATURES,
)


# ---------------------------------------------------------
# Experiment B — engineered features
# ---------------------------------------------------------

evaluate(
    "B. ENGINEERED FEATURES",
    train_engineered,
    valid_engineered,
    ENGINEERED_FEATURES,
)


print("\n" + "=" * 70)
print("FEATURE ENGINEERING EXPERIMENT COMPLETE")
print("=" * 70)