from pathlib import Path
import time

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

df = pd.read_csv(DATA / "train-test.csv", parse_dates=["date"])

# Same chronological split as every previous experiment
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


def add_date_features(data):
    data = data.copy()

    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month
    data["dayofweek"] = data["date"].dt.dayofweek

    return data


train = add_date_features(train)
valid = add_date_features(valid)


def evaluate_model(depth, iterations):
    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_valid = valid[FEATURES]
    y_valid = valid[TARGET]

    model = CatBoostRegressor(
        iterations=iterations,
        depth=depth,
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

    print(
        f"depth={depth}, iterations={iterations} | "
        f"MAE={mae:.4f} | "
        f"RMSE={rmse:.4f} | "
        f"R2={r2:.4f} | "
        f"time={fit_time:.2f}s"
    )

    return {
        "depth": depth,
        "iterations": iterations,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "fit_time": fit_time,
    }


print("=" * 70)
print("CATBOOST HYPERPARAMETER EXPERIMENT")
print("=" * 70)

results = []

for depth in [6, 8, 10]:
    for iterations in [500, 1000]:
        results.append(
            evaluate_model(
                depth=depth,
                iterations=iterations,
            )
        )


results_df = pd.DataFrame(results).sort_values("mae")

print("\n" + "=" * 70)
print("RESULTS SORTED BY MAE")
print("=" * 70)

print(results_df.to_string(index=False))

print("\n" + "=" * 70)
print("CURRENT BEST")
print("=" * 70)

best = results_df.iloc[0]

print(f"Depth:      {int(best['depth'])}")
print(f"Iterations: {int(best['iterations'])}")
print(f"MAE:        {best['mae']:.4f}")
print(f"RMSE:       {best['rmse']:.4f}")
print(f"R2:         {best['r2']:.4f}")