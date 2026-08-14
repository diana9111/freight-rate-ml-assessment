from pathlib import Path

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


# ---------------------------------------------------------
# Train our current best model
# ---------------------------------------------------------

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

model.fit(
    train[FEATURES],
    train[TARGET],
    cat_features=CAT_FEATURES,
)

predictions = model.predict(valid[FEATURES])


# ---------------------------------------------------------
# Create error table
# ---------------------------------------------------------

results = valid.copy()

results["prediction"] = predictions
results["error"] = results["prediction"] - results[TARGET]
results["absolute_error"] = results["error"].abs()
results["squared_error"] = results["error"] ** 2

print("=" * 80)
print("OVERALL MODEL PERFORMANCE")
print("=" * 80)

mae = mean_absolute_error(results[TARGET], results["prediction"])
rmse = np.sqrt(mean_squared_error(results[TARGET], results["prediction"]))
r2 = r2_score(results[TARGET], results["prediction"])

print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2:   {r2:.4f}")


# ---------------------------------------------------------
# Largest individual errors
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("20 LARGEST ABSOLUTE ERRORS")
print("=" * 80)

columns = [
    "load_id",
    "date",
    "pickup",
    "delivery",
    "equipment",
    "distance",
    "weight",
    "market_index",
    "quote_signal",
    "posted_rate",
    "prediction",
    "error",
    "absolute_error",
]

print(
    results[columns]
    .sort_values("absolute_error", ascending=False)
    .head(20)
    .to_string(index=False)
)


# ---------------------------------------------------------
# Error by distance band
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("ERROR BY DISTANCE BAND")
print("=" * 80)

results["distance_band"] = pd.cut(
    results["distance"],
    bins=[-np.inf, 100, 250, 500, 750, 1000, 1500, np.inf],
    labels=[
        "<=100",
        "101-250",
        "251-500",
        "501-750",
        "751-1000",
        "1001-1500",
        ">1500",
    ],
)

distance_errors = (
    results.groupby("distance_band", observed=False)
    .agg(
        rows=("posted_rate", "size"),
        mae=("absolute_error", "mean"),
        mean_error=("error", "mean"),
        median_error=("error", "median"),
        target_mean=("posted_rate", "mean"),
    )
)

print(distance_errors.to_string())


# ---------------------------------------------------------
# Error by equipment
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("ERROR BY EQUIPMENT")
print("=" * 80)

equipment_errors = (
    results.groupby("equipment")
    .agg(
        rows=("posted_rate", "size"),
        mae=("absolute_error", "mean"),
        mean_error=("error", "mean"),
        median_error=("error", "median"),
        target_mean=("posted_rate", "mean"),
    )
    .sort_values("mae", ascending=False)
)

print(equipment_errors.to_string())


# ---------------------------------------------------------
# Error by negative weight
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("ERROR BY WEIGHT SIGN")
print("=" * 80)

results["negative_weight"] = results["weight"] < 0
results["missing_weight"] = results["weight"].isna()

weight_errors = (
    results.groupby("negative_weight")
    .agg(
        rows=("posted_rate", "size"),
        mae=("absolute_error", "mean"),
        mean_error=("error", "mean"),
        target_mean=("posted_rate", "mean"),
    )
)

print(weight_errors.to_string())


print("\nMissing weight:")
print(
    results[results["missing_weight"]]
    .agg(
        rows=("posted_rate", "size"),
        mae=("absolute_error", "mean"),
        mean_error=("error", "mean"),
        target_mean=("posted_rate", "mean"),
    )
    .to_string()
)


# ---------------------------------------------------------
# Error by month / date
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("ERROR BY DATE")
print("=" * 80)

date_errors = (
    results.groupby("date")
    .agg(
        rows=("posted_rate", "size"),
        mae=("absolute_error", "mean"),
        mean_error=("error", "mean"),
    )
)

print(date_errors.to_string())


# ---------------------------------------------------------
# Systematic bias
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("SYSTEMATIC ERROR")
print("=" * 80)

print(f"Mean error:   {results['error'].mean():.4f}")
print(f"Median error: {results['error'].median():.4f}")

under = (results["prediction"] < results[TARGET]).mean()
over = (results["prediction"] > results[TARGET]).mean()

print(f"Predictions below actual: {under:.2%}")
print(f"Predictions above actual: {over:.2%}")


# ---------------------------------------------------------
# Worst routes
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("ROUTES WITH HIGHEST MAE")
print("=" * 80)

route_errors = (
    results.groupby(["pickup", "delivery"])
    .agg(
        rows=("posted_rate", "size"),
        mae=("absolute_error", "mean"),
        mean_error=("error", "mean"),
        target_mean=("posted_rate", "mean"),
    )
    .query("rows >= 10")
    .sort_values("mae", ascending=False)
)

print(route_errors.head(20).to_string())


# ---------------------------------------------------------
# Prediction vs target summary
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("PREDICTION VS ACTUAL")
print("=" * 80)

print(
    results[
        ["posted_rate", "prediction", "error", "absolute_error"]
    ].describe().to_string()
)


print("\n" + "=" * 80)
print("ERROR ANALYSIS COMPLETE")
print("=" * 80)