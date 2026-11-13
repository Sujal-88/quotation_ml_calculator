"""Price a printing job described by a JSON file with the trained model."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from quotation_ml.modeling import predict_quote

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--spec", required=True, help="JSON file with job specifications")
parser.add_argument("--output", default="artifacts", help="Directory holding locally-trained model")
parser.add_argument("--model", choices=["enhanced", "simplified"], default="enhanced")
parser.add_argument("--currency", default="INR", help="Label only: confirm actual DB currency")
args = parser.parse_args()
spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
price = predict_quote(spec, args.output, args.model)
print(json.dumps({"model": args.model, "predicted_amount": round(price, 2), "currency_label": args.currency}, indent=2))
