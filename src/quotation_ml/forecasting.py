"""Weekly revenue forecasting with a chronological holdout."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from .features import load_orders
from .modeling import metrics


def weekly_revenue(orders: pd.DataFrame) -> pd.Series:
    if "date_created" not in orders:
        raise ValueError("Revenue forecasting requires orders.date_created")
    dates = pd.to_datetime(orders["date_created"], errors="coerce", utc=True).dt.tz_convert(None)
    amounts = pd.to_numeric(orders["total_amount"], errors="coerce")
    valid = pd.DataFrame({"date": dates, "amount": amounts}).dropna()
    valid = valid[valid["amount"] >= 0]
    if valid.empty:
        raise ValueError("No dated orders with numeric revenue")
    return valid.set_index("date")["amount"].resample("W-SUN").sum().asfreq("W-SUN", fill_value=0)


def forecast_revenue(db_path: str | Path, output_dir: str | Path = "artifacts", weeks: int = 4) -> dict:
    if not 1 <= weeks <= 52:
        raise ValueError("weeks must be between 1 and 52")
    weekly = weekly_revenue(load_orders(db_path))
    if len(weekly) < 12:
        raise ValueError(f"Need at least 12 calendar weeks; found {len(weekly)}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cut = int(len(weekly) * .8)
    train, test = weekly.iloc[:cut], weekly.iloc[cut:]
    # Cap determined from training data only; do not peek at the holdout.
    cap = float(train.quantile(.95))
    capped_train = train.clip(upper=cap)
    fitted = ExponentialSmoothing(capped_train, trend="add", seasonal=None,
                                  initialization_method="estimated").fit(optimized=True)
    backtest = np.maximum(0, np.asarray(fitted.forecast(len(test)), dtype=float))
    backtest_metrics = metrics(test.to_numpy(), backtest)
    # Refit with all history for the actual future forecast (evaluation already finished).
    full_cap = float(weekly.quantile(.95))
    full_fit = ExponentialSmoothing(weekly.clip(upper=full_cap), trend="add",
                                    seasonal=None, initialization_method="estimated").fit(optimized=True)
    future_values = np.maximum(0, np.asarray(full_fit.forecast(weeks), dtype=float))
    future_dates = pd.date_range(weekly.index[-1] + pd.Timedelta(days=7), periods=weeks, freq="W-SUN")
    forecast = [{"week_ending": str(date.date()), "forecast_amount": round(float(value), 2)}
                for date, value in zip(future_dates, future_values)]
    result = {
        "history_weeks": len(weekly), "training_weeks": len(train), "test_weeks": len(test),
        "capping_threshold_train": round(cap, 2),
        "evaluation": backtest_metrics,
        "evaluation_note": "Errors compare holdout predictions with UNcapped actual weekly revenue; cap learned from training only.",
        "future_weeks": forecast,
    }
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(weekly.index, weekly, label="Observed weekly revenue")
    ax.plot(test.index, backtest, linestyle="--", label="Held-out predictions")
    ax.plot(future_dates, future_values, linestyle=":", marker="o", label="Future forecast")
    ax.set(xlabel="Week ending", ylabel="Revenue amount (currency per source database)", title="Weekly order revenue: observed, backtest and future")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "weekly_revenue_forecast.png", dpi=170)
    plt.close(fig)
    with (output_dir / "forecast.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result
