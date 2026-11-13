"""Generate clearly SYNTHETIC examples for smoke tests; not a real-world validation."""
import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path
import random
import sqlite3

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", default="data/demo_synthetic.db")
parser.add_argument("--rows", type=int, default=180)
args = parser.parse_args()
if args.rows < 20:
    parser.error("Please generate at least 20 rows")
randomizer = random.Random(42)
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
if output.exists():
    output.unlink()
with sqlite3.connect(output) as db:
    db.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, date_created TEXT, total_amount REAL, details TEXT)")
    for n in range(args.rows):
        sheets = randomizer.choice([250, 500, 1000, 1500, 3200, 5000])
        gsm = randomizer.choice([80, 100, 130, 170, 250, 300])
        binding = randomizer.choice(["No Binding", "Perfect Binding", "Staple Binding"])
        paper_type = randomizer.choice(["Art Paper", "Maplitho", "Cardstock"])
        size = randomizer.choice(["23x36", "20x30", "A4"])
        lamination = randomizer.choice([[], ["Gloss"], ["Matte"]])
        printing = randomizer.choice(["Single Side", "Front & Back"])
        details = {
            "bindingDetails": binding, "lamination": lamination,
            "paperEntries": [{"gSM": gsm, "paperQuantity": f"{sheets:,} Sheets",
                 "paperType": paper_type, "sheetSize": size,
                 "paperSuppliedBy": "Company", "paperPurchaseFrom": "Vendor",
                 "printingSpecs": {"ctpSuppliedBy": "Company", "totalSetsofCtp": 2,
                   "printingType": [printing], "printingDetails": {"ink": {"qty": sheets}}}}],
        }
        amount = (500 + 2.1 * sheets + 7 * gsm + (800 if binding != "No Binding" else 0)
                  + (400 if lamination else 0) + (250 if printing != "Single Side" else 0)
                  + randomizer.gauss(0, 350))
        when = datetime(2025, 2, 1) + timedelta(days=randomizer.randrange(500))
        db.execute("INSERT INTO orders (date_created, total_amount, details) VALUES (?, ?, ?)",
                   (when.strftime("%Y-%m-%d"), round(max(100, amount), 2), json.dumps(details)))
print(f"Created {args.rows} explicitly synthetic orders at {output}")
