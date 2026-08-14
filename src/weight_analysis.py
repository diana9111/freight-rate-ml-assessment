from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

train = pd.read_csv(DATA / "train-test.csv")


print("=" * 70)
print("NEGATIVE WEIGHT ANALYSIS")
print("=" * 70)

negative = train[train["weight"] < 0].copy()

print(f"\nNegative-weight rows: {len(negative):,}")

print("\nWeight values:")
print(negative["weight"].value_counts().sort_index().to_string())


print("\n" + "=" * 70)
print("NEGATIVE WEIGHT SUMMARY")
print("=" * 70)

print(
    negative[
        [
            "distance",
            "weight",
            "market_index",
            "quote_signal",
            "posted_rate",
        ]
    ].describe().to_string()
)


print("\n" + "=" * 70)
print("NEGATIVE WEIGHTS BY EQUIPMENT")
print("=" * 70)

print(
    negative["equipment"]
    .value_counts(dropna=False)
    .to_string()
)


print("\n" + "=" * 70)
print("NEGATIVE WEIGHTS — RATE COMPARISON")
print("=" * 70)

comparison = train.assign(
    negative_weight=train["weight"] < 0
).groupby("negative_weight")["posted_rate"].agg(
    ["count", "mean", "median", "std", "min", "max"]
)

print(comparison.to_string())


print("\n" + "=" * 70)
print("NEGATIVE WEIGHTS — DISTANCE COMPARISON")
print("=" * 70)

distance_comparison = train.assign(
    negative_weight=train["weight"] < 0
).groupby("negative_weight")["distance"].agg(
    ["count", "mean", "median", "std", "min", "max"]
)

print(distance_comparison.to_string())


print("\n" + "=" * 70)
print("NEGATIVE WEIGHTS — SAMPLE ROWS")
print("=" * 70)

columns = [
    "load_id",
    "pickup",
    "delivery",
    "distance",
    "equipment",
    "weight",
    "date",
    "market_index",
    "quote_signal",
    "posted_rate",
]

print(
    negative[columns]
    .sort_values("weight")
    .head(30)
    .to_string(index=False)
)


print("\n" + "=" * 70)
print("MISSING WEIGHTS")
print("=" * 70)

missing_weight = train[train["weight"].isna()]

print(f"Missing-weight rows: {len(missing_weight):,}")

print(
    missing_weight[
        [
            "distance",
            "equipment",
            "market_index",
            "quote_signal",
            "posted_rate",
        ]
    ].describe().to_string()
)


print("\nAnalysis complete.")