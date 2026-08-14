# Freight Rate ML Assessment

Machine learning solution for the Freight Rate Prediction assessment.

## Objective

Train a regression model on `data/train-test.csv`, validate the approach, predict rates for every load in `data/validation.csv`, and produce the required December 2025 predictions.

## Project structure

```text
freight-rate-ml-assessment/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   └── baseline.py
├── data/                 # local assessment data; ignored by Git
└── reports/              # analysis/report artifacts
```

## Local setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

python -m pip install -r requirements.txt
```

Place the assessment CSV files inside `data/`:

- `train-test.csv`
- `validation.csv`
- `validation-predictions-template.csv`
- `december-chart-inputs.csv`

## Baseline

Run:

```bash
python src/baseline.py
```

The baseline currently uses a chronological holdout: training data before 2025-09-01 and validation data from 2025-09-01 onward. It reports MAE, RMSE, and R² for a mean predictor and a first CatBoost model.

This is intentionally a baseline, not the final submission model.

## Next steps

1. Complete exploratory data analysis.
2. Investigate data-quality issues and unusual values.
3. Compare time-based and alternative validation strategies.
4. Engineer useful date, route, and interaction features.
5. Compare several regression models.
6. Tune the strongest candidates.
7. Train the selected model and generate `validation_predictions.csv`.
8. Fill `data/december-chart-inputs.csv` predictions.
9. Run the provided `score.py`.
10. Write the final report and record the 2–3 minute Loom walkthrough.
