# Freight Rate Prediction — ML Assessment

## Overview

This project develops a machine-learning model to predict freight load rates from historical shipment data.

The assessment includes two prediction tasks:

1. **Validation predictions** for 12,000 unseen loads.
2. **December 2025 rate predictions** for a fixed Lexington → Fort Wayne Dry Van load, where only the date changes.

The final solution uses **CatBoostRegressor** and a chronological validation split to better represent the forward-looking nature of the December forecasting task.

---

## Final Model Performance

The final model was evaluated using a chronological holdout:

* **Training:** dates before September 1, 2025
* **Validation:** dates from September 1, 2025 onward
* **Training rows:** 38,477
* **Validation rows:** 9,523

| Metric |       Result |
| ------ | -----------: |
| MAE    | **106.9832** |
| RMSE   |     632.3776 |
| R²     |       0.8283 |

The MAE is the primary practical indicator of prediction accuracy because the dataset contains a small number of extreme rate outliers that disproportionately affect RMSE.

### Performance excluding extreme outliers

There are 142 observations above $8,000 in the complete 48,000-row dataset, representing approximately 0.30% of the data.

On the chronological validation set, excluding the 35 observations above $8,000:

| Metric |    Result |
| ------ | --------: |
| MAE    | **74.27** |
| RMSE   |    270.54 |
| R²     |    0.9619 |

The extreme observations have very weak relationships with the available market variables:

* Correlation with `quote_signal`: **-0.040**
* Correlation with `market_index`: **0.034**

This suggests that these extreme values behave more like unpredictable data anomalies than systematic market movements.

---

## Model

The final model is a `CatBoostRegressor` with the following configuration:

| Parameter              |             Value |
| ---------------------- | ----------------: |
| Model                  | CatBoostRegressor |
| Iterations             |               450 |
| Depth                  |                 6 |
| Learning rate          |              0.03 |
| L2 leaf regularization |                12 |
| Random strength        |                 1 |
| Loss function          |              RMSE |
| Random seed            |                42 |
| Threads                |     All available |

No target clipping is used in the final model.

The model was selected after experimenting with regularization and training configuration. The final configuration achieved a validation MAE of **106.9832**.

---

## Features

### Full model

The full model uses:

* `pickup`
* `delivery`
* `pickup_lat`
* `pickup_lon`
* `delivery_lat`
* `delivery_lon`
* `distance`
* `equipment`
* `weight`
* `market_index`
* `quote_signal`
* `year`
* `month`
* `dayofweek`

Categorical features:

* `pickup`
* `delivery`
* `equipment`

### Date features

The original `date` field is converted into:

* `year`
* `month`
* `dayofweek`

### Data-quality handling

The model applies two simple data-quality treatments:

1. Negative `weight` values are converted to their absolute values because a small number of observations contained sign-flipped weights.
2. Missing categorical values are replaced with `"MISSING"`.

Numeric missing values are left for CatBoost to handle natively.

---

## December 2025 Forecast

The December task provides only:

* pickup
* delivery
* distance
* equipment
* weight
* date

It does **not** contain:

* pickup coordinates
* delivery coordinates
* `market_index`
* `quote_signal`

Therefore, a compatible reduced-feature CatBoost model is trained using:

* `pickup`
* `delivery`
* `distance`
* `equipment`
* `weight`
* `year`
* `month`
* `dayofweek`

The December scenario is fixed at:

| Input     | Value               |
| --------- | ------------------- |
| Pickup    | Lexington           |
| Delivery  | Fort Wayne          |
| Distance  | 360 miles           |
| Equipment | Dry Van             |
| Weight    | 32,000 lb           |
| Date      | December 1–31, 2025 |

### December prediction summary

| Statistic          |  Prediction |
| ------------------ | ----------: |
| Minimum            | **$856.53** |
| Maximum            | **$879.58** |
| Mean               | **$868.05** |
| Standard deviation |   **$8.98** |

The resulting prediction file contains all **31 days of December 2025**.

---

## December Forecast Visualization

The generated December chart is available at:

`scorer_results/candidate_december.png`

It shows the predicted freight rate across December while keeping the route, distance, equipment, and weight fixed.

---

## Output Files

The final prediction files are:

### Validation predictions

`data/validation_predictions.csv`

Contains:

* `load_id`
* `predicted_rate`

Validation checks confirmed:

* **12,000 rows**
* **12,000 unique load IDs**
* no missing values
* positive predicted rates
* all expected validation IDs are present

### December predictions

`data/december_predictions.csv`

Contains:

* `pickup`
* `delivery`
* `distance`
* `equipment`
* `weight`
* `date`
* `predicted_rate`

Validation checks confirmed:

* **31 rows**
* one row for every day from December 1–31, 2025
* no missing values
* correct fixed shipment inputs
* positive predicted rates

---

## Project Structure

```text
freight-rate-ml-assessment-starter/
│
├── data/
│   ├── train-test.csv
│   ├── validation.csv
│   ├── december-chart-inputs.csv
│   ├── validation_predictions.csv
│   └── december_predictions.csv
│
├── src/
│   ├── final_model.py
│   └── score.py
│
├── scorer_results/
│   └── candidate_december.png
│
├── REPORT.md
└── README.md
```

---

## Running the Model

Activate the project virtual environment and run:

```bat
python src\final_model.py
```

The script trains the final model, generates validation predictions, trains the compatible December model, and writes both prediction files to `data/`.

---

## Validating the Outputs

After generating the prediction files, run:

```bat
python src\score.py --predictions data\validation_predictions.csv --december-predictions data\december_predictions.csv
```

A successful run should report:

```text
Validated 12,000 final predictions.
Validated 31 fixed December predictions.
Created chart: scorer_results\candidate_december.png
Final validation metrics are calculated by Spotter after submission.
```

The scorer validates the required file structure, row counts, IDs, dates, fixed December inputs, and prediction values.

---

## Validation Strategy

A chronological split was used rather than a random train/validation split:

```text
Training:   date < 2025-09-01
Validation: date >= 2025-09-01
```

This better reflects the actual forecasting problem because the December prediction task requires applying information from the past to a future period.

---

## Key Findings

### 1. CatBoost handles the categorical shipment features naturally

`pickup`, `delivery`, and `equipment` are passed directly as categorical features rather than manually encoded.

### 2. The model performs strongly on the predictable portion of the dataset

The validation MAE is **106.98**, while excluding extreme rate outliers reduces MAE to **74.27** and increases R² to **0.9619**.

### 3. Extreme rate observations dominate squared error

Only approximately 0.30% of all observations have `posted_rate > $8,000`, but these observations account for a disproportionate amount of RMSE.

The model therefore has substantially stronger performance on the normal freight-rate distribution than the overall RMSE alone suggests.

### 4. No target clipping is used

The final model trains on the original target values, including the extreme observations. The final target maximum is approximately **$25,533**.

---

## Final Submission Status

The final pipeline produces:

* a validated **12,000-row validation prediction file**
* a validated **31-day December prediction file**
* a December prediction visualization
* a reproducible model-training script
* an output validation/scoring script
* documentation describing the approach and results

The final chronological validation result is:

> **MAE: 106.9832**

The final validation metrics shown here are local evaluation results. The official assessment score is determined by Spotter after submission.
