# Freight Rate Prediction — ML Assessment Report

## 1. Executive Summary

This project develops a machine-learning model for predicting freight load rates from historical shipment data.

The final solution uses **CatBoostRegressor** and a chronological validation strategy designed to reflect the forward-looking nature of the assessment.

Two prediction outputs were produced:

1. **Validation predictions** for 12,000 unseen loads.
2. **December 2025 predictions** for a fixed Lexington → Fort Wayne shipment scenario.

The final model achieved a validation **MAE of 106.9832**.

---

## 2. Dataset

The complete training dataset contains **48,000 rows**.

The target variable is:

```text
posted_rate
```

The data was sorted chronologically before splitting.

### Chronological split

| Dataset | Date range | Rows |
|---|---|---:|
| Training | Before 2025-09-01 | 38,477 |
| Validation | 2025-09-01 onward | 9,523 |
| Total | — | 48,000 |

A chronological split was selected instead of a random split because the December task requires forecasting into a future period. This makes the validation setup more representative of the actual prediction problem.

---

## 3. Feature Engineering

The final full model uses the following features:

| Feature | Type / Purpose |
|---|---|
| `pickup` | Categorical origin |
| `delivery` | Categorical destination |
| `pickup_lat` | Pickup latitude |
| `pickup_lon` | Pickup longitude |
| `delivery_lat` | Delivery latitude |
| `delivery_lon` | Delivery longitude |
| `distance` | Shipment distance |
| `equipment` | Equipment type |
| `weight` | Shipment weight |
| `market_index` | Market condition indicator |
| `quote_signal` | Quote-related signal |
| `year` | Extracted from date |
| `month` | Extracted from date |
| `dayofweek` | Extracted from date |

The categorical variables are:

```text
pickup
delivery
equipment
```

CatBoost handles these categorical variables directly.

---

## 4. Data Quality Handling

A small number of observations contained negative shipment weights whose magnitudes were consistent with normal shipment weights.

The final pipeline therefore applies:

```python
weight = abs(weight)
```

Missing categorical values are replaced with:

```text
MISSING
```

Numeric missing values are left for CatBoost to handle natively.

### Target clipping

**No target clipping is used in the final model.**

The original target values are retained, including extreme observations.

The maximum observed target is approximately:

```text
$25,533
```

---

## 5. Model Selection

The final model is a `CatBoostRegressor`.

### Final configuration

| Parameter | Value |
|---|---:|
| Model | CatBoostRegressor |
| Iterations | 450 |
| Depth | 6 |
| Learning rate | 0.03 |
| L2 leaf regularization | 12 |
| Random strength | 1 |
| Loss function | RMSE |
| Random seed | 42 |
| Threads | All available |

The final configuration was selected through experimentation with model complexity and regularization.

An L2 regularization experiment also tested values from 5 through 15. The earlier 500-iteration configuration with `l2_leaf_reg=12` performed strongly, while further tuning of the learning rate and iteration count produced the final model used here.

---

## 6. Validation Results

The final configuration was evaluated on the chronological holdout.

### Overall validation performance

| Metric | Result |
|---|---:|
| **MAE** | **106.9832** |
| RMSE | 632.3776 |
| R² | 0.8283 |

The most important practical metric is MAE because a small number of extreme target values have a disproportionate effect on squared-error metrics.

### Validation performance excluding extreme outliers

Rows with:

```text
posted_rate > $8,000
```

were examined separately.

There are:

```text
142 / 48,000 rows
```

above this threshold, representing approximately **0.30%** of the full dataset.

On the September+ validation set, 35 observations exceeded $8,000.

| Metric | Overall | Excluding >$8,000 |
|---|---:|---:|
| MAE | 106.98 | **74.27** |
| RMSE | 632.38 | 270.54 |
| R² | 0.8283 | **0.9619** |

This indicates substantially stronger performance on the normal, predictable portion of the freight-rate distribution.

---

## 7. Extreme Target Analysis

The extreme target observations appear difficult to predict from the available features.

For the complete dataset:

| Relationship | Correlation |
|---|---:|
| `posted_rate` vs `quote_signal` | -0.040 |
| `posted_rate` vs `market_index` | 0.034 |

Both correlations are very weak.

The extreme observations also occur across different dates, routes, and equipment types rather than being concentrated in one obvious segment.

This provides evidence that a substantial part of the extreme-rate error may represent unpredictable noise or anomalous observations rather than a systematic pattern that the available features can reliably learn.

Importantly, these rows were **not removed from final training** and the target was **not clipped**.

---

## 8. Final Validation Predictions

The final model was retrained using the complete training dataset and used to generate predictions for the provided validation set.

Output:

```text
data/validation_predictions.csv
```

The file contains exactly two columns:

```text
load_id
predicted_rate
```

Validation checks confirmed:

| Check | Result |
|---|---:|
| Rows | **12,000** |
| Unique IDs | **12,000** |
| Missing values | **0** |
| Non-positive predictions | **0** |
| Expected IDs present | **Yes** |

The resulting prediction range was approximately:

```text
Minimum: $270.02
Maximum: $6,654.34
```

---

## 9. December 2025 Forecast

The December input file does not contain the full set of features available in the training data.

It provides:

- `pickup`
- `delivery`
- `distance`
- `equipment`
- `weight`
- `date`

The following training features are therefore unavailable for the December task:

- `pickup_lat`
- `pickup_lon`
- `delivery_lat`
- `delivery_lon`
- `market_index`
- `quote_signal`

To avoid inventing unavailable information, a compatible reduced-feature CatBoost model was trained for the December predictions.

### December model features

```text
pickup
delivery
distance
equipment
weight
year
month
dayofweek
```

---

## 10. December Scenario

The December prediction task uses fixed shipment characteristics:

| Input | Value |
|---|---|
| Pickup | Lexington |
| Delivery | Fort Wayne |
| Distance | 360 miles |
| Equipment | Dry Van |
| Weight | 32,000 lb |
| Prediction period | December 1–31, 2025 |

Only the date changes from one prediction to another.

---

## 11. December Results

The December model generated one prediction for every day from December 1 through December 31, 2025.

Output:

```text
data/december_predictions.csv
```

### Summary

| Statistic | Value |
|---|---:|
| Minimum predicted rate | **$856.53** |
| Maximum predicted rate | **$879.58** |
| Mean predicted rate | **$868.05** |
| Standard deviation | **$8.98** |
| Number of predictions | **31** |

The predictions remain within a relatively narrow range because the shipment characteristics are fixed and only calendar features vary.

---

## 12. December Visualization

A chart was generated from the final December predictions:

```text
scorer_results/candidate_december.png
```

The chart visualizes the predicted rate throughout December 2025 for the fixed Lexington → Fort Wayne shipment.

The visualization confirms that the model produces a smooth daily forecast while responding to day-of-week effects.

---

## 13. Output Validation

A dedicated validation script was created:

```text
src/score.py
```

It verifies the final prediction files before submission.

### Validation checks

For the 12,000-row validation file, it checks:

- exact column names and order
- exact row count
- missing IDs
- duplicate IDs
- expected validation IDs
- invalid or non-finite predictions
- non-positive predictions

For the December file, it checks:

- required seven columns
- exactly 31 rows
- every date from December 1 through December 31, 2025
- no duplicate dates
- fixed pickup
- fixed delivery
- fixed distance
- fixed equipment
- fixed weight
- positive predictions
- valid numeric values

The final files successfully passed these checks.

---

## 14. Reproducibility

The main model can be executed with:

```bat
python src\final_model.py
```

The output validation can be executed with:

```bat
python src\score.py --predictions data\validation_predictions.csv --december-predictions data\december_predictions.csv
```

A successful validation reports:

```text
Validated 12,000 final predictions.
Validated 31 fixed December predictions.
Created chart: scorer_results\candidate_december.png
Final validation metrics are calculated by Spotter after submission.
```

---

## 15. Project Outputs

The repository contains the following important deliverables:

```text
README.md
REPORT.md

src/
├── final_model.py
└── score.py

data/
├── validation_predictions.csv
└── december_predictions.csv

scorer_results/
└── candidate_december.png
```

Additional experiment scripts are retained in `src/` to document the model-development process.

---

## 16. Conclusion

The final CatBoost solution achieves:

**Validation MAE: 106.9832**

The model uses a chronological validation strategy, categorical handling through CatBoost, date-derived features, basic data-quality corrections, and no target clipping.

The model performs particularly well on the normal freight-rate distribution, with a validation MAE of **74.27** after excluding extreme observations above $8,000.

The final pipeline also successfully produces and validates the required 12,000 validation predictions and 31 December 2025 predictions.

The official assessment score is determined by Spotter after submission.