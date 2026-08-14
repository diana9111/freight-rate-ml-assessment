from pathlib import Path
import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------
# 1. Load datasets
# ---------------------------------------------------------
train = pd.read_csv(DATA / "train-test.csv", parse_dates=["date"])
validation = pd.read_csv(DATA / "validation.csv", parse_dates=["date"])
template = pd.read_csv(DATA / "validation-predictions-template.csv")
december = pd.read_csv(DATA / "december-chart-inputs.csv", parse_dates=["date"])


# ---------------------------------------------------------
# 2. Dataset sizes
# ---------------------------------------------------------
section("DATASET SIZES")

print(f"Training:   {train.shape[0]:,} rows x {train.shape[1]} columns")
print(f"Validation: {validation.shape[0]:,} rows x {validation.shape[1]} columns")
print(f"Template:   {template.shape[0]:,} rows x {template.shape[1]} columns")
print(f"December:   {december.shape[0]:,} rows x {december.shape[1]} columns")


# ---------------------------------------------------------
# 3. Column information
# ---------------------------------------------------------
section("TRAINING COLUMNS AND DATA TYPES")

print(train.dtypes)


# ---------------------------------------------------------
# 4. First few rows
# ---------------------------------------------------------
section("FIRST 5 TRAINING ROWS")

print(train.head().to_string())


# ---------------------------------------------------------
# 5. Missing values
# ---------------------------------------------------------
section("MISSING VALUES — TRAINING")

missing_train = train.isna().sum()
missing_train_pct = (missing_train / len(train) * 100).round(3)

missing_table = pd.DataFrame({
    "missing_count": missing_train,
    "missing_percent": missing_train_pct
})

print(missing_table[missing_table["missing_count"] > 0])


section("MISSING VALUES — VALIDATION")

missing_valid = validation.isna().sum()
missing_valid_pct = (missing_valid / len(validation) * 100).round(3)

missing_valid_table = pd.DataFrame({
    "missing_count": missing_valid,
    "missing_percent": missing_valid_pct
})

print(missing_valid_table[missing_valid_table["missing_count"] > 0])


# ---------------------------------------------------------
# 6. Duplicate IDs / rows
# ---------------------------------------------------------
section("DUPLICATES")

print(f"Duplicate training rows: {train.duplicated().sum():,}")
print(f"Duplicate validation rows: {validation.duplicated().sum():,}")

if "load_id" in train.columns:
    print(f"Duplicate training load_id: {train['load_id'].duplicated().sum():,}")

if "load_id" in validation.columns:
    print(f"Duplicate validation load_id: {validation['load_id'].duplicated().sum():,}")


# ---------------------------------------------------------
# 7. Date ranges
# ---------------------------------------------------------
section("DATE RANGES")

print(f"Training:   {train['date'].min()} -> {train['date'].max()}")
print(f"Validation: {validation['date'].min()} -> {validation['date'].max()}")
print(f"December:   {december['date'].min()} -> {december['date'].max()}")


# ---------------------------------------------------------
# 8. Categorical variables
# ---------------------------------------------------------
section("CATEGORICAL FEATURES")

for col in ["pickup", "delivery", "equipment"]:
    print(f"\n{col}")
    print(f"  Training unique:   {train[col].nunique(dropna=False):,}")
    print(f"  Validation unique: {validation[col].nunique(dropna=False):,}")

    train_values = set(train[col].dropna().unique())
    valid_values = set(validation[col].dropna().unique())
    unseen = sorted(valid_values - train_values)

    print(f"  Validation categories unseen in training: {len(unseen):,}")

    if unseen:
        print(f"  Examples: {unseen[:20]}")


# ---------------------------------------------------------
# 9. Equipment distribution
# ---------------------------------------------------------
section("EQUIPMENT DISTRIBUTION")

print(train["equipment"].value_counts(dropna=False).to_string())


# ---------------------------------------------------------
# 10. Numerical summary
# ---------------------------------------------------------
section("NUMERICAL FEATURE SUMMARY")

numeric_columns = train.select_dtypes(include=np.number).columns

print(
    train[numeric_columns]
    .describe()
    .T
    .to_string()
)


# ---------------------------------------------------------
# 11. Target distribution
# ---------------------------------------------------------
section("TARGET — POSTED RATE")

target = train["posted_rate"]

print(target.describe().to_string())

print("\nSelected percentiles:")
print(
    target.quantile(
        [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    ).to_string()
)


# ---------------------------------------------------------
# 12. Suspicious values
# ---------------------------------------------------------
section("NON-POSITIVE / SUSPICIOUS VALUES")

for col in ["distance", "weight", "market_index", "quote_signal", "posted_rate"]:
    if col in train.columns:
        non_positive = (train[col] <= 0).sum()
        print(f"{col}: <= 0 → {non_positive:,}")


# ---------------------------------------------------------
# 13. Negative values specifically
# ---------------------------------------------------------
section("NEGATIVE VALUES")

for col in train.select_dtypes(include=np.number).columns:
    count = (train[col] < 0).sum()
    if count:
        print(f"{col}: {count:,} negative values")


# ---------------------------------------------------------
# 14. Extreme weight observations
# ---------------------------------------------------------
section("WEIGHT EXTREMES")

print("Smallest weights:")
print(train[["load_id", "weight"]].sort_values("weight").head(10).to_string(index=False))

print("\nLargest weights:")
print(train[["load_id", "weight"]].sort_values("weight", ascending=False).head(10).to_string(index=False))


# ---------------------------------------------------------
# 15. Extreme target observations
# ---------------------------------------------------------
section("HIGHEST POSTED RATES")

print(
    train[
        ["load_id", "pickup", "delivery", "distance", "equipment", "weight", "posted_rate"]
    ]
    .sort_values("posted_rate", ascending=False)
    .head(15)
    .to_string(index=False)
)


# ---------------------------------------------------------
# 16. Correlations
# ---------------------------------------------------------
section("NUMERICAL CORRELATIONS WITH POSTED RATE")

correlations = (
    train[numeric_columns]
    .corr()["posted_rate"]
    .sort_values(ascending=False)
)

print(correlations.to_string())


# ---------------------------------------------------------
# 17. Average rate by equipment
# ---------------------------------------------------------
section("RATE BY EQUIPMENT")

equipment_summary = (
    train.groupby("equipment")["posted_rate"]
    .agg(["count", "mean", "median", "std"])
    .sort_values("mean", ascending=False)
)

print(equipment_summary.to_string())


# ---------------------------------------------------------
# 18. Average rate by month
# ---------------------------------------------------------
section("RATE BY MONTH")

monthly = (
    train.assign(month=train["date"].dt.month)
    .groupby("month")["posted_rate"]
    .agg(["count", "mean", "median", "std"])
)

print(monthly.to_string())


# ---------------------------------------------------------
# 19. Distance vs rate by broad distance bands
# ---------------------------------------------------------
section("RATE BY DISTANCE BAND")

train["distance_band"] = pd.cut(
    train["distance"],
    bins=[-np.inf, 100, 250, 500, 750, 1000, 1500, np.inf],
    labels=[
        "<=100",
        "101-250",
        "251-500",
        "501-750",
        "751-1000",
        "1001-1500",
        ">1500",
    ]
)

distance_summary = (
    train.groupby("distance_band", observed=False)["posted_rate"]
    .agg(["count", "mean", "median", "std"])
)

print(distance_summary.to_string())


# ---------------------------------------------------------
# 20. Train vs final validation numerical distributions
# ---------------------------------------------------------
section("TRAIN VS FINAL VALIDATION — NUMERICAL MEDIANS")

compare_columns = [
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "weight",
    "market_index",
    "quote_signal",
]

comparison = pd.DataFrame({
    "train_median": train[compare_columns].median(),
    "validation_median": validation[compare_columns].median(),
})

comparison["difference"] = (
    comparison["validation_median"] - comparison["train_median"]
)

print(comparison.to_string())


# ---------------------------------------------------------
# 21. Final checks
# ---------------------------------------------------------
section("FINAL CHECKS")

print(f"Training target missing: {train['posted_rate'].isna().sum():,}")
print(f"Validation target column exists: {'posted_rate' in validation.columns}")
print(f"Template columns: {list(template.columns)}")
print(f"December columns: {list(december.columns)}")

print("\nEDA COMPLETE.")