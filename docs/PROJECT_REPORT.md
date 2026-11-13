# ML-Based Printing Quotation Estimation and Business Revenue Forecasting

**Project type:** Applied Machine Learning / Printing ERP Analytics  
**Author:** [Enter student name and roll number]  
**Institute:** [Enter institute and department]  
**Academic year:** 2026-27  
**Source:** User-supplied `quotation.py` (Colab export)  
**Study status:** Implementation reconstructed; real-dataset evaluation pending.

## Abstract

Printing businesses regularly estimate quotations based on paper quantity and specifications, printing format, lamination, binding and production operations. This project evaluates machine-learning approaches to predict the historical order amount from those specifications and to forecast weekly order revenue. Historical order records are read from an SQLite database, with nested JSON order details converted into tabular features. Two Gradient Boosting regressors are trained for quotation prediction: a simple model for a small set of user-facing inputs and an enhanced model for detailed production specifications. A Holt-Winters Exponential Smoothing model forecasts weekly revenue. Reproducible holdout evaluation, a median baseline, local inference, an optional Streamlit UI, and tests make the system easier to demonstrate and extend. Since the original database was not attached, numerical results remain to be measured on authorized historical data.

**Keywords:** machine learning, printing quotation, Gradient Boosting Regression, feature engineering, SQLite, revenue forecasting, Holt-Winters.

## 1. Introduction and problem

Quotation generation in offset printing depends on material and job specifications. Manual estimates can be inconsistent, and planning can be difficult when historic orders are not analyzed systematically. The original notebook explores an ERP SQLite database and develops several iterations of regression-based estimates. This report describes a reproducible project based on those original ideas. It predicts past *order amounts*, rather than asserting that it calculates the actual manufacturing cost or ensures profitability.

## 2. Objectives

- Transform historical printing order JSON into machine-learning features.
- Train and compare simplified and enhanced quotation models with a held-out test set.
- Forecast short-term weekly revenue from dated order records.
- Provide testable local code, a command-line predictor, a visual demo, and safe GitHub submission files.
- Report assumptions and protect confidential client and order data.

## 3. Dataset and input

The source uses SQLite `orders` records with `details`, `total_amount`, and `date_created`. The project does not require nonessential tables. Features include production run quantity or sheets count, GSM, binding, lamination indicators, paper type, sheet size, purchase and supply sources, CTP specifications, and printing mode. The detailed original model reads the first paper entry when an order contains several entries. The number of original records, date span and real monetary denomination have **not been verified** because no SQLite database was supplied.

## 4. Models and algorithm

**Simplified quotation:** GradientBoostingRegressor (`n_estimators=100`) on sheets count, sheet size, printing side and inferred job type. The simpler feature set supports quick interactive estimates but necessarily omits cost-relevant details.

**Enhanced quotation:** GradientBoostingRegressor (`n_estimators=200`, `learning_rate=0.08`, `max_depth=5`) on production quantity, GSM, lamination, binding, paper and CTP properties, and printing mode. Categorical variables are one-hot encoded by a fitted pipeline. This version corresponds to the most detailed experimental direction in the uploaded notebook, while the parser and evaluation are cleaned up.

**Weekly revenue:** Holt-Winters Exponential Smoothing with additive trend and no seasonality. Revenue is aggregated weekly; an outlier cap computed from the training section is used only as a modelling choice, with actual holdout revenue retained for evaluation.

## 5. System architecture

`SQLite orders -> validation and JSON parsing -> feature engineering -> train/test split -> model pipeline -> MAE/RMSE/R² + plots -> locally saved model -> CLI / Streamlit quotation interface.`

Separately, `dated orders -> weekly aggregation -> time-based split -> Holt-Winters model -> 4-week revenue projection`.

## 6. Implementation environment

Python, SQLite, pandas, NumPy, scikit-learn, statsmodels, matplotlib, joblib, and optional Streamlit. The source is organized into extraction, modelling, forecasting, scripts, tests and documentation. The repository includes synthetic test-data generation but deliberately excludes production databases and trained models.

## 7. Experimental methodology and metrics

Quotations use a reproducible 80% training / 20% testing split with seed 42 and an identical split for both models. A median prediction baseline supplies context. Model quality is reported using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² score. Forecasting uses the first 80% of time-ordered weeks for training and final 20% for backtesting, with a 4-week horizon generated after refitting on the complete history.

**Results table (fill from `artifacts/metrics.json` and `artifacts/forecast.json` after running on real data):**

| Experiment | Test MAE | Test RMSE | Test R² | Status |
|---|---|---|---|---|
| Median quotation baseline | Pending | Pending | Pending | Real database not supplied |
| Simplified Gradient Boosting | Pending | Pending | Pending | Real database not supplied |
| Enhanced Gradient Boosting | Pending | Pending | Pending | Real database not supplied |
| Weekly revenue Holt-Winters | Pending | Pending | Pending | Real database not supplied |

The original notebook prints some evaluations and contains hard-coded reference values for previous model performance, but the complete execution evidence and database are not available. Those values must not be presented as verified results. Synthetic demonstration metrics, if generated, are labelled as synthetic.

## 8. Observations and limitations

The feature engineering assumes that the first paper entry adequately describes the job, that binding indicates page count, and that the maximum nested printing quantity approximates production quantity. Each assumption may fail for some real jobs. The model also predicts historical charged amounts rather than a validated paper/press/labour cost calculation. The quotation holdout is random, so repeated order patterns and changing market rates can cause optimistic results. Weekly revenue may be volatile and outliers may be genuine. No uncertainty intervals or pilot deployment evidence have been supplied.

## 9. Future improvements

Create an anonymized schema dictionary and validate all nested JSON variants. Model every paper entry and labour, machine, overhead, wastage, taxes and profit margins separately. Compare against an explicit cost-engineering formula and time-based holdout. Evaluate category-specific errors and underquoting risk. Log predictions with human approval and monitor performance by period. Integrate an authorized, isolated pricing service with the ERP only after appropriate reliability and privacy checks.

## 10. Conclusion

This project organizes an existing printing quotation ML experiment into a testable, locally runnable implementation. It includes two quotation regressors and a weekly revenue forecast, with an improved evaluation protocol and an optional user interface. Actual business accuracy is an **open empirical question** until tested on the missing historical database and subsequently verified on unseen future orders.

## References

1. User-provided source code: `quotation.py`, Google Colab export (primary project input).
2. scikit-learn official documentation: GradientBoostingRegressor, Pipeline, OneHotEncoder, model evaluation.
3. statsmodels official documentation: Holt-Winters ExponentialSmoothing.
4. SQLite documentation: SQLite SQL language and Python sqlite3 interface.

**Reproducibility note:** Report metrics and images should be regenerated from authorized data. Do not submit synthetic demo scores as real experimental results.
