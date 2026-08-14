from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

df = pd.read_csv(DATA / "train-test.csv", parse_dates=["date"])


# =========================================================
# BASIC TARGET ANALYSIS
# =========================================================

print("=" * 80)
print("TARGET / RATE STRUCTURE ANALYSIS")
print("=" * 80)

print(f"Rows: {len(df):,}")

print("\nTarget summary:")
print(df["posted_rate"].describe().to_string())


# =========================================================
# RATE PER MILE
# =========================================================

df["rate_per_mile"] = (
    df["posted_rate"] / df["distance"].clip(lower=1)
)

print("\n" + "=" * 80)
print("RATE PER MILE")
print("=" * 80)

print(
    df["rate_per_mile"]
    .describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    .to_string()
)


# =========================================================
# DISTANCE BANDS
# =========================================================

df["distance_band"] = pd.cut(
    df["distance"],
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

print("\n" + "=" * 80)
print("RATE STRUCTURE BY DISTANCE")
print("=" * 80)

distance_summary = (
    df.groupby("distance_band", observed=False)
    .agg(
        rows=("posted_rate", "size"),
        rate_mean=("posted_rate", "mean"),
        rate_median=("posted_rate", "median"),
        rate_per_mile_mean=("rate_per_mile", "mean"),
        rate_per_mile_median=("rate_per_mile", "median"),
        distance_mean=("distance", "mean"),
    )
)

print(distance_summary.to_string())


# =========================================================
# HIGH-RATE LOADS
# =========================================================

print("\n" + "=" * 80)
print("HIGH-RATE LOADS")
print("=" * 80)

for threshold in [5000, 7500, 10000, 15000]:
    subset = df[df["posted_rate"] >= threshold]

    print(
        f"Rate >= ${threshold:,}: "
        f"{len(subset):,} rows "
        f"({len(subset) / len(df):.2%})"
    )


# =========================================================
# HIGH-RATE LOAD CHARACTERISTICS
# =========================================================

high_rate = df[df["posted_rate"] >= 10000].copy()

print("\n" + "=" * 80)
print("LOADS WITH POSTED RATE >= $10,000")
print("=" * 80)

if len(high_rate) > 0:
    print(
        high_rate[
            [
                "pickup",
                "delivery",
                "equipment",
                "distance",
                "weight",
                "market_index",
                "quote_signal",
                "posted_rate",
                "rate_per_mile",
            ]
        ]
        .sort_values("posted_rate", ascending=False)
        .head(30)
        .to_string(index=False)
    )
else:
    print("No loads found.")


# =========================================================
# RATE BY EQUIPMENT
# =========================================================

print("\n" + "=" * 80)
print("TARGET BY EQUIPMENT")
print("=" * 80)

equipment_summary = (
    df.groupby("equipment")
    .agg(
        rows=("posted_rate", "size"),
        rate_mean=("posted_rate", "mean"),
        rate_median=("posted_rate", "median"),
        rate_per_mile_mean=("rate_per_mile", "mean"),
        distance_mean=("distance", "mean"),
    )
    .sort_values("rate_mean", ascending=False)
)

print(equipment_summary.to_string())


# =========================================================
# RATE BY MARKET INDEX
# =========================================================

print("\n" + "=" * 80)
print("TARGET BY MARKET INDEX")
print("=" * 80)

df["market_band"] = pd.qcut(
    df["market_index"],
    q=5,
    duplicates="drop",
)

market_summary = (
    df.groupby("market_band", observed=False)
    .agg(
        rows=("posted_rate", "size"),
        market_mean=("market_index", "mean"),
        rate_mean=("posted_rate", "mean"),
        rate_median=("posted_rate", "median"),
        rate_per_mile_mean=("rate_per_mile", "mean"),
    )
)

print(market_summary.to_string())


# =========================================================
# RATE BY QUOTE SIGNAL
# =========================================================

print("\n" + "=" * 80)
print("TARGET BY QUOTE SIGNAL")
print("=" * 80)

df["quote_band"] = pd.qcut(
    df["quote_signal"],
    q=5,
    duplicates="drop",
)

quote_summary = (
    df.groupby("quote_band", observed=False)
    .agg(
        rows=("posted_rate", "size"),
        quote_mean=("quote_signal", "mean"),
        rate_mean=("posted_rate", "mean"),
        rate_median=("posted_rate", "median"),
        rate_per_mile_mean=("rate_per_mile", "mean"),
    )
)

print(quote_summary.to_string())


# =========================================================
# CORRELATIONS
# =========================================================

print("\n" + "=" * 80)
print("NUMERIC CORRELATIONS WITH POSTED RATE")
print("=" * 80)

numeric_columns = [
    "posted_rate",
    "distance",
    "weight",
    "market_index",
    "quote_signal",
    "rate_per_mile",
]

print(
    df[numeric_columns]
    .corr()["posted_rate"]
    .sort_values(ascending=False)
    .to_string()
)


# =========================================================
# RATE PER MILE CORRELATIONS
# =========================================================

print("\n" + "=" * 80)
print("CORRELATIONS WITH RATE PER MILE")
print("=" * 80)

print(
    df[
        [
            "rate_per_mile",
            "distance",
            "weight",
            "market_index",
            "quote_signal",
        ]
    ]
    .corr()["rate_per_mile"]
    .sort_values(ascending=False)
    .to_string()
)


# =========================================================
# EXTREME RATE PER MILE
# =========================================================

print("\n" + "=" * 80)
print("EXTREME RATE-PER-MILE LOADS")
print("=" * 80)

print(
    df[
        [
            "pickup",
            "delivery",
            "equipment",
            "distance",
            "weight",
            "market_index",
            "quote_signal",
            "posted_rate",
            "rate_per_mile",
        ]
    ]
    .sort_values("rate_per_mile", ascending=False)
    .head(20)
    .to_string(index=False)
)


# =========================================================
# ROUTE ANALYSIS FOR HIGH-RATE LOADS
# =========================================================

print("\n" + "=" * 80)
print("ROUTES WITH HIGHEST AVERAGE POSTED RATE")
print("=" * 80)

route_summary = (
    df.groupby(["pickup", "delivery"])
    .agg(
        rows=("posted_rate", "size"),
        rate_mean=("posted_rate", "mean"),
        rate_median=("posted_rate", "median"),
        distance_mean=("distance", "mean"),
        rate_per_mile_mean=("rate_per_mile", "mean"),
    )
    .query("rows >= 10")
    .sort_values("rate_mean", ascending=False)
)

print(route_summary.head(25).to_string())


# =========================================================
# FINAL
# =========================================================

print("\n" + "=" * 80)
print("TARGET ANALYSIS COMPLETE")
print("=" * 80)