import json
import pandas as pd
import pytest
from quotation_ml.features import build_quotation_table, extract_features, features_from_form, number


def sample():
    return {"bindingDetails": "Perfect Binding", "lamination": ["Matte"],
            "paperEntries": [{"gSM": "250", "paperQuantity": "3,200 Sheets", "sheetSize": "23x36",
                              "paperType": "Art Paper", "printingSpecs": {"printingType": ["Front & Back"],
                                                                    "printingDetails": {"a": {"qty": "1,800"}}}}]}


def test_parse_actual_schema_and_quantity():
    row = extract_features(json.dumps(json.dumps(sample())))
    assert row["gsm"] == 250
    assert row["sheets_qty"] == 3200
    assert row["total_quantity"] == 1800
    assert row["has_matte"] == 1
    assert row["print_side"] == "FB"
    assert row["job_type"] == "Multi Page"


def test_number_handling():
    assert number("1,200 Sheets") == 1200
    assert number("3.5 Sheets") == 3.5
    assert number(None) == 0


def test_invalid_json_reported_not_silent():
    df, stats = build_quotation_table(pd.DataFrame([
        {"total_amount": 100, "details": json.dumps(sample())},
        {"total_amount": 500, "details": "not-json"},
        {"total_amount": None, "details": json.dumps(sample())},
    ]))
    assert len(df) == 1
    assert stats["invalid_json"] == 1
    assert stats["missing_or_invalid_amount"] == 1


def test_form_validation():
    with pytest.raises(ValueError, match="total_quantity"):
        features_from_form({"gsm": 100, "sheet_size": "A4", "printing_type": "Single Side"})
