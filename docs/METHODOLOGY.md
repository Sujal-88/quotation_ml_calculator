# Reproducible methodology

## Quotation task

**Input:** historical `orders.details` describing printing specifications. **Target:** `orders.total_amount` (an observed order amount, not a guaranteed manufacturing cost).

1. Validate the SQLite order table, read it read-only, retain only positive numeric amounts and parseable JSON.
2. Extract simple and enhanced engineered features. Preserve raw categorical names, make numeric parsing explicit, count skipped records.
3. Use the **same 80/20 seeded (42) random holdout** for both models. The simple baseline is the training-set median.
4. Fit `ColumnTransformer(OneHotEncoder(handle_unknown='ignore'))` inside an sklearn `Pipeline`; encoding learns categories from training only.
5. Fit Gradient Boosting Regressors: simplified (`n_estimators=100`), enhanced (`n_estimators=200`, `learning_rate=0.08`, `max_depth=5`). Compare MAE/RMSE/R² on the untouched test partition.
6. Write an actual-vs-predicted chart and prediction values for the held-out enhanced model. **After evaluation**, fit deployable predictors on all available valid records.
7. Store models locally; do not commit business models or predictions.

A random split can give optimistic performance when repeating clients, identical jobs, or changing paper prices appear in both partitions. To estimate real deployment accuracy, validate on orders created **after** the training cutoff and compare to a cost-engineering baseline.

## Forecast task

1. Parse `date_created` timestamps, sum nonnegative `total_amount` by week ending Sunday; fill zero-activity weeks with zero.
2. Partition history chronologically (first 80% train, final 20% test).
3. Calculate the 95th percentile weekly revenue cap using the training set **only**. Fit additive-trend Holt-Winters (`seasonal=None`) on capped training history.
4. Backtest on **uncapped** true weekly revenue (MAE/RMSE/R²).
5. Refit using all history with its own full-history cap for the next four projected weeks.

Outlier capping may suppress genuinely large legitimate order revenue. Use uncapped comparisons and investigate why high weeks occur. The method assumes a reasonably regular weekly series and does not model holidays, seasonality, capacity or changing market conditions.

## Evaluation reference

**MAE** = mean absolute difference between actual and predicted amounts.

**RMSE** = square root of the mean squared prediction errors; penalizes larger misses more.

**R²** = 1 minus residual sum of squares divided by total sum of squares; may be negative, particularly on small or volatile holdouts.

No real MAE, RMSE, R² or forecast values can be authenticated from the source Python file alone, because the supporting SQLite database and execution logs were not provided. Hard-coded reference numbers in the Colab source are not validated experimental results.
