from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data" / "train-test.csv"
DECEMBER_PATH = ROOT / "data" / "december-chart-inputs.csv"


# =========================================================
# FEATURES AVAILABLE IN THE DECEMBER FILE
# =========================================================

FEATURES = [
    "pickup",
    "delivery",
    "distance",
    "equipment",
    "weight",
    "year",
    "month",
    "dayofweek",
]

CATEGORICAL = [
    "pickup",
    "delivery",
    "equipment",
]


def add_date_features(df):
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["dayofweek"] = df["date"].dt.dayofweek

    return df


def make_model():
    return CatBoostRegressor(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=12,
        random_strength=1,
        loss_function="RMSE",
        random_seed=42,
        verbose=False,
        thread_count=-1,
    )


# =========================================================
# LOAD DATA
# =========================================================

train = pd.read_csv(
    TRAIN_PATH,
    parse_dates=["date"],
)

december = pd.read_csv(
    DECEMBER_PATH,
    parse_dates=["date"],
)

train = add_date_features(train)
december = add_date_features(december)


# =========================================================
# CHRONOLOGICAL VALIDATION
# =========================================================

train_part = train[train["date"] < "2025-09-01"].copy()
valid_part = train[train["date"] >= "2025-09-01"].copy()

X_train = train_part[FEATURES].copy()
X_valid = valid_part[FEATURES].copy()

y_train = train_part["posted_rate"]
y_valid = valid_part["posted_rate"]


for col in CATEGORICAL:
    X_train[col] = X_train[col].fillna("MISSING")
    X_valid[col] = X_valid[col].fillna("MISSING")


print("=" * 80)
print("DECEMBER MODEL - REDUCED FEATURE SET")
print("=" * 80)

print(f"Training rows:   {len(X_train):,}")
print(f"Validation rows: {len(X_valid):,}")

print("\nFeatures:")
for feature in FEATURES:
    print(f"  - {feature}")


# =========================================================
# TRAIN
# =========================================================

model = make_model()

print("\nTraining...")

model.fit(
    X_train,
    y_train,
    cat_features=CATEGORICAL,
)

pred = model.predict(X_valid)


# =========================================================
# VALIDATION RESULTS
# =========================================================

mae = mean_absolute_error(y_valid, pred)
rmse = np.sqrt(mean_squared_error(y_valid, pred))
r2 = r2_score(y_valid, pred)

print("\n" + "=" * 80)
print("VALIDATION RESULTS")
print("=" * 80)

print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2:   {r2:.4f}")


# =========================================================
# TRAIN FINAL MODEL ON ALL TRAINING DATA
# =========================================================

print("\n" + "=" * 80)
print("TRAINING FINAL DECEMBER MODEL")
print("=" * 80)

X_full = train[FEATURES].copy()
y_full = train["posted_rate"]

X_december = december[FEATURES].copy()

for col in CATEGORICAL:
    X_full[col] = X_full[col].fillna("MISSING")
    X_december[col] = X_december[col].fillna("MISSING")


final_model = make_model()

final_model.fit(
    X_full,
    y_full,
    cat_features=CATEGORICAL,
)


# =========================================================
# DECEMBER PREDICTIONS
# =========================================================

december_predictions = final_model.predict(X_december)

december_output = december[
    [
        "pickup",
        "delivery",
        "distance",
        "equipment",
        "weight",
        "date",
    ]
].copy()

december_output["predicted_rate"] = december_predictions

print("\n" + "=" * 80)
print("DECEMBER 2025 PREDICTIONS")
print("=" * 80)

print(
    december_output[
        ["date", "predicted_rate"]
    ].to_string(index=False)
)

print("\n" + "=" * 80)
print("DECEMBER SUMMARY")
print("=" * 80)

print(
    f"Minimum predicted rate: "
    f"${december_predictions.min():,.2f}"
)

print(
    f"Maximum predicted rate: "
    f"${december_predictions.max():,.2f}"
)

print(
    f"Mean predicted rate:    "
    f"${december_predictions.mean():,.2f}"
)

print(
    f"Std predicted rate:     "
    f"${december_predictions.std():,.2f}"
)

print("\nDECEMBER MODEL COMPLETE")