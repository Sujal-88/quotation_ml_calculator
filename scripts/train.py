"""Train quotation models and optionally a revenue forecaster from a local SQLite DB."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from quotation_ml.forecasting import forecast_revenue
from quotation_ml.modeling import train_quotes

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--db", required=True, help="Path to the PRIVATE SQLite database")
parser.add_argument("--output", default="artifacts", help="Directory for models and evaluation figures")
parser.add_argument("--forecast", action="store_true", help="Additionally build weekly revenue forecast")
args = parser.parse_args()
print("Quotation evaluation:\n" + json.dumps(train_quotes(args.db, args.output), indent=2))
if args.forecast:
    print("Revenue forecast:\n" + json.dumps(forecast_revenue(args.db, args.output), indent=2))
