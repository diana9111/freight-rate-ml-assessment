from pathlib import Path
import time

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

df = pd.read_csv(DATA / "train-test.csv", parse_dates=["date"])

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


def evaluate_l2(l2_value):
    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_valid = valid[FEATURES]
    y_valid = valid[TARGET]

    model = CatBoostRegressor(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=l2_value,
        random_strength=1,
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
        f"L2={l2_value:>2} | "
        f"MAE={mae:.4f} | "
        f"RMSE={rmse:.4f} | "
        f"R2={r2:.4f} | "
        f"time={fit_time:.2f}s"
    )

    return {
        "l2": l2_value,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "fit_time": fit_time,
    }


print("=" * 70)
print("L2 REGULARIZATION FINE-TUNING")
print("=" * 70)

results = []

for l2 in [5, 7, 8, 9, 10, 12, 15]:
    results.append(evaluate_l2(l2))


results_df = pd.DataFrame(results).sort_values("mae")

print("\n" + "=" * 70)
print("RESULTS SORTED BY MAE")
print("=" * 70)

print(results_df.to_string(index=False))

print("\n" + "=" * 70)
print("BEST L2")
print("=" * 70)

best = results_df.iloc[0]

print(f"L2:   {int(best['l2'])}")
print(f"MAE:  {best['mae']:.4f}")
print(f"RMSE: {best['rmse']:.4f}")
print(f"R2:   {best['r2']:.4f}")