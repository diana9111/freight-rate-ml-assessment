from pathlib import Path
import time

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "train-test.csv"

FEATURES = [
    "pickup",
    "delivery",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "equipment",
    "weight",
    "market_index",
    "quote_signal",
    "year",
    "month",
    "dayofweek",
]
CATEGORICAL = ["pickup", "delivery", "equipment"]


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["year"] = out["date"].dt.year
    out["month"] = out["date"].dt.month
    out["dayofweek"] = out["date"].dt.dayofweek
    return out


def metrics(y_true, predictions):
    return {
        "MAE": mean_absolute_error(y_true, predictions),
        "RMSE": mean_squared_error(y_true, predictions) ** 0.5,
        "R2": r2_score(y_true, predictions),
    }


def main():
    df = pd.read_csv(TRAIN_PATH, parse_dates=["date"]).sort_values("date")

    # First realistic baseline: train on earlier dates and validate on later dates.
    train_df = df[df["date"] < "2025-09-01"].copy()
    valid_df = df[df["date"] >= "2025-09-01"].copy()

    train_df = add_date_features(train_df)
    valid_df = add_date_features(valid_df)

    X_train = train_df[FEATURES].copy()
    X_valid = valid_df[FEATURES].copy()
    y_train = train_df["posted_rate"]
    y_valid = valid_df["posted_rate"]

    for col in CATEGORICAL:
        X_train[col] = X_train[col].fillna("MISSING")
        X_valid[col] = X_valid[col].fillna("MISSING")

    print(f"Train rows:      {len(train_df):,}")
    print(f"Validation rows: {len(valid_df):,}")
    print(f"Train dates:     {train_df.date.min().date()} -> {train_df.date.max().date()}")
    print(f"Valid dates:     {valid_df.date.min().date()} -> {valid_df.date.max().date()}")
    print()

    # Trivial benchmark.
    dummy = DummyRegressor(strategy="mean")
    start = time.perf_counter()
    dummy.fit(np.zeros((len(y_train), 1)), y_train)
    dummy_pred = dummy.predict(np.zeros((len(y_valid), 1)))
    dummy_time = time.perf_counter() - start
    print("Mean predictor")
    print(f"  fit+predict time: {dummy_time:.4f}s")
    print(f"  {metrics(y_valid, dummy_pred)}")
    print()

    # First ML baseline.
    model = CatBoostRegressor(
        iterations=500,
        depth=7,
        learning_rate=0.08,
        loss_function="RMSE",
        random_seed=42,
        verbose=False,
        thread_count=-1,
    )

    start = time.perf_counter()
    model.fit(X_train, y_train, cat_features=CATEGORICAL)
    fit_time = time.perf_counter() - start

    start = time.perf_counter()
    pred = model.predict(X_valid)
    predict_time = time.perf_counter() - start

    print("CatBoost baseline")
    print(f"  fit time:     {fit_time:.3f}s")
    print(f"  predict time: {predict_time:.3f}s")
    print(f"  {metrics(y_valid, pred)}")


if __name__ == "__main__":
    main()
