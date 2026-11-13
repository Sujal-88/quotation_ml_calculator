"""Parse the JSON structure found in the original 1Click printing ERP order data."""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

NUMERIC_ENHANCED = [
    "has_lamination", "num_lamination_types", "has_gloss", "has_matte",
    "gsm", "total_quantity", "num_paper_entries", "ctp_sets",
]
CATEGORICAL_ENHANCED = [
    "binding", "paper_type", "paper_supplied_by", "paper_purchase_from",
    "sheet_size", "ctp_supplied_by", "printing_type",
]
NUMERIC_SIMPLE = ["sheets_qty"]
CATEGORICAL_SIMPLE = ["job_size", "print_side", "job_type"]
FEATURE_COLUMNS = list(dict.fromkeys(NUMERIC_ENHANCED + CATEGORICAL_ENHANCED + NUMERIC_SIMPLE + CATEGORICAL_SIMPLE))


def number(value: Any, default: float = 0.0) -> float:
    """Read a numeric value, including '3,200 Sheets', without joining unrelated digits."""
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        v = float(value)
        return v if pd.notna(v) and abs(v) != float("inf") else default
    match = re.search(r"-?\d[\d,]*(?:\.\d+)?", str(value or ""))
    if not match:
        return default
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return default


def text_value(value: Any, default: str = "Unknown") -> str:
    """Normalize potentially missing categories without changing observed values."""
    if value is None or isinstance(value, (dict, list)):
        return default
    return str(value).strip() or default


def json_dict(payload: Any) -> dict:
    """Support dicts, JSON, and double-encoded JSON as seen in the source notebook."""
    current = payload
    for _ in range(3):
        if isinstance(current, dict):
            return current
        if not isinstance(current, str):
            break
        current = json.loads(current)
    if not isinstance(current, dict):
        raise ValueError("details must decode to a JSON object")
    return current


def as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def extract_features(payload: Any) -> dict[str, Any]:
    """Extract compatible simplified/enhanced features from a printing job.

    Only the first paper entry supplies detailed specs, matching the uploaded
    prototype. `num_paper_entries` counts all entries. Multi-paper jobs need a
    richer schema before this can be considered production-ready.
    """
    details = json_dict(payload)
    paper_entries = [p for p in as_list(details.get("paperEntries")) if isinstance(p, dict)]
    paper = paper_entries[0] if paper_entries else {}
    specs = paper.get("printingSpecs") or {}
    if not isinstance(specs, dict):
        specs = {}

    lamination = [str(item) for item in as_list(details.get("lamination")) if item]
    binding = text_value(details.get("bindingDetails"), "No Binding")
    printing_types = [str(x) for x in as_list(specs.get("printingType")) if x]
    print_side = ("FB" if any(any(term in x.lower() for term in
        ("back", "fb", "double", "both", "duplex")) for x in printing_types)
        else "Single Sided")

    qty_sheets = max(0, number(paper.get("paperQuantity")))
    qty_run = 0.0
    printing_details = specs.get("printingDetails") or {}
    records = printing_details.values() if isinstance(printing_details, dict) else as_list(printing_details)
    for record in records:
        if isinstance(record, dict):
            qty_run = max(qty_run, number(record.get("qty")))
    if qty_run <= 0:
        qty_run = qty_sheets
    if qty_run <= 0:
        qty_run = max(0, number(details.get("totalQuantity")))

    return {
        "has_lamination": int(bool(lamination)),
        "num_lamination_types": len(lamination),
        "has_gloss": int(any("gloss" in x.lower() for x in lamination)),
        "has_matte": int(any("matte" in x.lower() or "matt" in x.lower() for x in lamination)),
        "gsm": max(0, number(paper.get("gSM"))),
        "total_quantity": qty_run,
        "num_paper_entries": len(paper_entries),
        "ctp_sets": max(0, number(specs.get("totalSetsofCtp"))),
        "binding": binding,
        "paper_type": text_value(paper.get("paperType")),
        "paper_supplied_by": text_value(paper.get("paperSuppliedBy")),
        "paper_purchase_from": text_value(paper.get("paperPurchaseFrom")),
        "sheet_size": text_value(paper.get("sheetSize")),
        "ctp_supplied_by": text_value(specs.get("ctpSuppliedBy")),
        "printing_type": text_value(printing_types[0] if printing_types else "None", "None"),
        "sheets_qty": qty_sheets,
        "job_size": text_value(paper.get("sheetSize")),
        "print_side": print_side,
        # Heuristic from the uploaded prototype, not a verified definition of page count:
        "job_type": "Multi Page" if binding.lower() != "no binding" else "Single Page",
    }


def load_orders(db_path: str | Path) -> pd.DataFrame:
    path = Path(db_path)
    if not path.is_file():
        raise FileNotFoundError(f"SQLite file not found: {path}")
    with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as connection:
        cols = {row[1] for row in connection.execute("PRAGMA table_info(orders)")}
        needed = {"total_amount", "details"}
        if not needed.issubset(cols):
            raise ValueError(f"orders table must contain columns {sorted(needed)}")
        select = "SELECT total_amount, details" + (", date_created" if "date_created" in cols else "") + " FROM orders"
        return pd.read_sql_query(select, connection)


def build_quotation_table(orders: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return valid records and transparent processing counts; never expose raw details."""
    clean_rows = []
    excluded = {"missing_or_invalid_amount": 0, "invalid_json": 0}
    for record in orders.to_dict("records"):
        value = pd.to_numeric(record.get("total_amount"), errors="coerce")
        if pd.isna(value) or float(value) <= 0:
            excluded["missing_or_invalid_amount"] += 1
            continue
        try:
            features = extract_features(record.get("details"))
        except (ValueError, TypeError, json.JSONDecodeError):
            excluded["invalid_json"] += 1
            continue
        features["total_amount"] = float(value)
        clean_rows.append(features)
    return pd.DataFrame(clean_rows, columns=FEATURE_COLUMNS + ["total_amount"]), {
        "raw_records": len(orders), "usable_records": len(clean_rows), **excluded
    }


def features_from_form(spec: dict[str, Any]) -> dict[str, Any]:
    """Accept the same flattened feature structure used by the model and web UI."""
    missing = set(FEATURE_COLUMNS) - set(spec)
    # Optional fields are given safe defaults; callers must still supply critical specs.
    for field in ["total_quantity", "gsm", "sheet_size", "printing_type"]:
        if field not in spec:
            raise ValueError(f"Missing required input: {field}")
    result = {}
    for col in FEATURE_COLUMNS:
        if col in NUMERIC_ENHANCED + NUMERIC_SIMPLE:
            result[col] = max(0, number(spec.get(col, 0)))
        else:
            result[col] = text_value(spec.get(col), "No Binding" if col == "binding" else "Unknown")
    result["job_size"] = text_value(spec.get("job_size", spec.get("sheet_size")))
    result["sheets_qty"] = max(0, number(spec.get("sheets_qty", spec.get("total_quantity"))))
    result["job_type"] = text_value(spec.get("job_type", "Multi Page" if result["binding"] != "No Binding" else "Single Page"))
    result["print_side"] = text_value(spec.get("print_side", "FB" if any(t in result["printing_type"].lower() for t in ["fb", "back", "both", "double"]) else "Single Sided"))
    return result
