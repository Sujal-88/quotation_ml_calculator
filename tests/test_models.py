from datetime import datetime, timedelta
import json
import sqlite3

from quotation_ml.features import extract_features
from quotation_ml.forecasting import forecast_revenue
from quotation_ml.modeling import predict_quote, train_quotes


def create_fixture(path):
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE orders (date_created TEXT, total_amount REAL, details TEXT)")
        for i in range(80):
            qty = 300 + 55 * i
            payload = {"bindingDetails": "No Binding" if i % 3 else "Perfect Binding",
                "lamination": ["Gloss"] if i % 2 else [], "paperEntries": [{
                "gSM": 90 + 10 * (i % 5), "paperQuantity": str(qty),
                "sheetSize": "23x36" if i % 2 else "A4", "paperType": "Art Paper",
                "printingSpecs": {"printingType": ["Front & Back"] if i % 2 else ["Single Side"],
                                  "printingDetails": {"one": {"qty": qty}}}}]}
            date = (datetime(2024, 1, 1) + timedelta(days=i * 7)).isoformat()
            db.execute("INSERT INTO orders VALUES (?, ?, ?)", (date, qty * 2 + 400, json.dumps(payload)))


def test_end_to_end_train_predict_forecast(tmp_path):
    db = tmp_path / "test.db"
    out = tmp_path / "models"
    create_fixture(db)
    result = train_quotes(db, out)
    assert result["dataset"]["usable_records"] == 80
    assert "mae" in result["models"]["enhanced"]
    assert (out / "enhanced_holdout_predictions.csv").exists()
    with sqlite3.connect(db) as con:
        payload = json.loads(con.execute("SELECT details FROM orders LIMIT 1").fetchone()[0])
    amount = predict_quote(extract_features(payload), out)
    assert amount >= 0
    weekly = forecast_revenue(db, out, weeks=4)
    assert len(weekly["future_weeks"]) == 4
    assert (out / "weekly_revenue_forecast.png").exists()
