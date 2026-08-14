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


def add_date_features(data):
    data = data.copy()

    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month
    data["dayofweek"] = data["date"].dt.dayofweek

    return data


train = add_date_features(train)
valid = add_date_features(valid)


X_train = train[FEATURES]
X_valid = valid[FEATURES]

y_train = train[TARGET]
y_valid = valid[TARGET]


# =========================================================
# EVALUATION FUNCTION
# =========================================================

def evaluate_model(name, loss_function, log_target=False):

    if log_target:
        target_train = np.log1p(y_train)
    else:
        target_train = y_train

    model = CatBoostRegressor(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=12,
        random_strength=1,
        loss_function=loss_function,
        verbose=False,
        random_seed=42,
    )

    start = time.perf_counter()

    model.fit(
        X_train,
        target_train,
        cat_features=CAT_FEATURES,
    )

    fit_time = time.perf_counter() - start

    predictions = model.predict(X_valid)

    if log_target:
        predictions = np.expm1(predictions)

    predictions = np.maximum(predictions, 0)

    mae = mean_absolute_error(y_valid, predictions)
    rmse = np.sqrt(mean_squared_error(y_valid, predictions))
    r2 = r2_score(y_valid, predictions)

    print(
        f"{name:<15} | "
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
# EXPERIMENT
# =========================================================

print("=" * 80)
print("LOSS FUNCTION / TARGET TRANSFORMATION EXPERIMENT")
print("=" * 80)

results = []

results.append(
    evaluate_model(
        "RMSE baseline",
        loss_function="RMSE",
        log_target=False,
    )
)

results.append(
    evaluate_model(
        "MAE loss",
        loss_function="MAE",
        log_target=False,
    )
)

results.append(
    evaluate_model(
        "Log target",
        loss_function="RMSE",
        log_target=True,
    )
)


# =========================================================
# RESULTS
# =========================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

print(results_df.to_string(index=False))

print("\n" + "=" * 80)
print("BEST BY MAE")
print("=" * 80)

best = results_df.sort_values("mae").iloc[0]

print(f"Model: {best['name']}")
print(f"MAE:   {best['mae']:.4f}")
print(f"RMSE:  {best['rmse']:.4f}")
print(f"R2:    {best['r2']:.4f}")

print("\n" + "=" * 80)
print("LOSS EXPERIMENT COMPLETE")
print("=" * 80)