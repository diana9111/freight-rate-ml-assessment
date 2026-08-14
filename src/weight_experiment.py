from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

df = pd.read_csv(DATA / "train-test.csv", parse_dates=["date"])

# Same chronological split as our original baseline
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


FEATURES_WITH_DATE = FEATURES + [
    "year",
    "month",
    "dayofweek",
]


def evaluate_model(name, train_df, valid_df):
    X_train = train_df[FEATURES_WITH_DATE].copy()
    y_train = train_df[TARGET]

    X_valid = valid_df[FEATURES_WITH_DATE].copy()
    y_valid = valid_df[TARGET]

    model = CatBoostRegressor(
        iterations=500,
        depth=8,
        learning_rate=0.05,
        loss_function="RMSE",
        verbose=False,
        random_seed=42,
    )

    model.fit(
        X_train,
        y_train,
        cat_features=CAT_FEATURES,
    )

    predictions = model.predict(X_valid)

    mae = mean_absolute_error(y_valid, predictions)
    rmse = np.sqrt(mean_squared_error(y_valid, predictions))
    r2 = r2_score(y_valid, predictions)

    print(f"\n{name}")
    print("-" * 50)
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.4f}")

    return mae, rmse, r2


# ---------------------------------------------------------
# Prepare date features
# ---------------------------------------------------------

train = add_date_features(train)
valid = add_date_features(valid)


# ---------------------------------------------------------
# Experiment 1 — RAW WEIGHT
# ---------------------------------------------------------

train_raw = train.copy()
valid_raw = valid.copy()

evaluate_model(
    "1. RAW WEIGHT",
    train_raw,
    valid_raw,
)


# ---------------------------------------------------------
# Experiment 2 — NEGATIVE WEIGHT -> MISSING
# ---------------------------------------------------------

train_missing = train.copy()
valid_missing = valid.copy()

train_missing.loc[train_missing["weight"] < 0, "weight"] = np.nan
valid_missing.loc[valid_missing["weight"] < 0, "weight"] = np.nan

evaluate_model(
    "2. NEGATIVE WEIGHT -> MISSING",
    train_missing,
    valid_missing,
)


# ---------------------------------------------------------
# Experiment 3 — NEGATIVE WEIGHT -> ABSOLUTE VALUE
# ---------------------------------------------------------

train_abs = train.copy()
valid_abs = valid.copy()

train_abs["weight"] = train_abs["weight"].abs()
valid_abs["weight"] = valid_abs["weight"].abs()

evaluate_model(
    "3. ABSOLUTE VALUE OF WEIGHT",
    train_abs,
    valid_abs,
)


print("\n" + "=" * 70)
print("EXPERIMENT COMPLETE")
print("=" * 70)