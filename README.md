# Printing Quotation Prediction & Revenue Forecasting (Machine Learning)

A reproducible Python ML project for **estimating printing-job quotation amounts** and **forecasting weekly order revenue** from an SQLite order database. Adapted from an existing Google Colab `quotation.py` prototype. Built for local execution, academic demonstration, and subsequent ERP integration.

> **Status:** Research / academic prototype, **not** production pricing software. Predictions are statistical estimates, not validated cost sheets. Real-world scores cannot be claimed until evaluated against your own database. Synthetic examples are not evidence of business accuracy.

## Features

- **Enhanced quotation model:** `GradientBoostingRegressor` using quantity, GSM, binding, paper specifications, lamination, CTP, sheet size, printing type and supply source.
- **Simplified quotation model:** regression using sheet count, sheet size, printing side and inferred job type.
- **Revenue forecast:** Holt-Winters additive trend, weekly aggregation and a future four-week forecast.
- **Evaluation:** same seeded 80/20 **held-out** quotation split for both models and median baseline; MAE, RMSE and R². Time-ordered forecast holdout (last 20%).
- **Demo:** local Streamlit UI, JSON command-line predictor, unit tests and explicitly synthetic demo data.

## Project structure

```text
printing-quotation-ml/
├── app.py                       # Optional Streamlit web interface
├── requirements.txt
├── requirements-dev.txt
├── src/quotation_ml/
│   ├── features.py              # SQLite + nested JSON → job features
│   ├── modeling.py              # training/evaluation and prediction
│   └── forecasting.py           # weekly revenue backtest + 4-week projection
├── scripts/
│   ├── train.py                 # train models, optionally forecast
│   ├── predict.py               # JSON CLI inference
│   └── make_demo_data.py        # SYNTHETIC data, not client data
├── examples/job_spec.json
├── tests/
├── docs/
│   ├── Printing_Quotation_ML_Project_Report.pdf
│   ├── Printing_Quotation_ML_Project_Report.docx
│   ├── DATA_DICTIONARY.md
│   ├── METHODOLOGY.md
│   └── PROJECT_REPORT.md
├── legacy/quotation_original.py # archival Colab export (not runnable standalone)
└── artifacts/                  # training outputs; excluded from Git
```

## Getting started

Tested with Python 3.11+ compatible code; exact dependencies may need a compatible supported Python environment.

```bash
# At the project root
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest -q
```

### A. Run with your real printing ERP data

Place your **private** SQLite file somewhere outside version control (for example `data/my_database.db`) and execute:

```bash
python scripts/train.py --db data/my_database.db --output artifacts --forecast
python scripts/predict.py --spec examples/job_spec.json --output artifacts --model enhanced --currency INR
streamlit run app.py
```

Required `orders` table columns: `total_amount`, `details` (JSON string). `date_created` is also required for forecasting. Historical currency is **not confirmed** by the source file; change the displayed currency label to match your stored values. Database dates must be parseable. Models are persisted as `.joblib` files in the excluded `artifacts/` folder.

### B. Run without private data: synthetic smoke test

```bash
python scripts/make_demo_data.py --output data/demo_synthetic.db
python scripts/train.py --db data/demo_synthetic.db --output artifacts --forecast
python scripts/predict.py --spec examples/job_spec.json
streamlit run app.py
```

The generated database is **synthetic and unsuitable for claiming real performance**. Use it only to demonstrate the pipeline.

### What the training output means

- `artifacts/metrics.json`: valid/dropped record counts, median baseline, simple/enhanced held-out metrics.
- `artifacts/quotation_actual_vs_predicted.png`: enhanced model test predictions, *not* training fit.
- `artifacts/enhanced_holdout_predictions.csv`: held-out predicted vs actual amounts. **Contains potentially confidential business values: do not publish.**
- `artifacts/forecast.json` and `artifacts/weekly_revenue_forecast.png`: chronological forecast evaluation and future projection.

The reported quotation evaluation uses **models fitted only on training data**. After reporting metrics, a new deployment model is fitted on all valid records. Therefore, *deployed model results are not the held-out scores*. The forecasting cap is learned from training history for backtesting; errors are compared to uncapped test revenue.

## Data provenance and safety

This repository intentionally **does not contain real client records**, database files, exported business predictions, trained models or passwords. `.gitignore` excludes them. Check commit history and generated files before publishing. `joblib.load` can execute code: load only models trained locally in a trusted environment.

The original Colab source is preserved under `legacy/` for traceability. Its original Colab document URL was redacted for safer public sharing. It references `/content/DB/my_database.db`, calls notebook-only `display`, repeats parsing code, uses `enhanced_preds` before creation in one cell, and computes some errors on training data. It should not be used as the primary executable implementation.

## Known limitations

1. Most features are extracted from **only the first** `paperEntries` object; multiple paper stocks are not represented fully.
2. `job_type` is heuristically inferred from the presence of binding and may be wrong.
3. `total_quantity` uses the maximum nested printing quantity, falling back to paper sheets; verify this against the ERP schema and business definitions.
4. Unknown categorical values are ignored by the encoder rather than invented; actual costs/prices can change across time, clients and suppliers.
5. An 80/20 **random** quotation split does not establish forward-time performance. A date-based backtest and independent production pilot are necessary.
6. `total_amount` is the **historical charged/order amount**, not necessarily direct manufacturing cost or profit. Discounts, taxes, freight, substrate wastage and margins are not fully modelled.
7. Forecasts have no uncertainty bands and should not be treated as financial commitments.

## GitHub upload

```bash
git init
git add .
git status        # verify that NO .db, .env, joblib or client results are staged
git commit -m "Initial printing quotation ML project"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

Create an **empty** repository on GitHub and replace `YOUR_USERNAME/YOUR_REPOSITORY` with your details. See the **PDF and editable DOCX report** under `docs/`, and `docs/PROJECT_REPORT.md` for the academic report source and `docs/METHODOLOGY.md` for evaluation details.

## Academic integrity

The provided source did **not** include the original SQLite database or a live output log. This project contains no invented real metrics, no verified accuracy claim, and no deployment claim. Re-run training with authorised, properly de-identified data before submitting empirical results.
# quotation_ml_calculator
