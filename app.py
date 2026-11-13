"""Optional local Streamlit interface for quotation estimates, not an invoicing system."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import streamlit as st
from quotation_ml.modeling import predict_quote

st.set_page_config(page_title="Printing Quotation ML", page_icon="🖨️", layout="centered")
st.title("Printing Quotation Predictor")
st.caption("Research prototype: predictions must be reviewed against actual material, labour and machine costs.")

with st.form("quote_input"):
    model = st.selectbox("Prediction model", ["enhanced", "simplified"])
    currency = st.text_input("Currency label (verify against your database)", "INR")
    left, right = st.columns(2)
    with left:
        quantity = st.number_input("Production quantity", min_value=1, value=1000, step=100)
        sheets = st.number_input("Paper sheets quantity", min_value=0, value=1000, step=100)
        gsm = st.number_input("Paper GSM", min_value=0, value=100, step=10)
        size = st.text_input("Sheet size (must match training categories)", "23x36")
        paper = st.text_input("Paper type", "Art Paper")
        binding = st.selectbox("Binding", ["No Binding", "Perfect Binding", "Staple Binding"])
    with right:
        print_type = st.selectbox("Printing type", ["Single Side", "Front & Back"])
        lamination = st.selectbox("Lamination", ["None", "Gloss", "Matte"])
        supplied = st.text_input("Paper supplied by", "Company")
        purchase = st.text_input("Paper purchased from", "Vendor")
        ctp = st.text_input("CTP supplied by", "Company")
        sets = st.number_input("CTP sets", min_value=0, value=2)
    submitted = st.form_submit_button("Estimate quotation")

if submitted:
    spec = {
        "total_quantity": quantity, "sheets_qty": sheets, "gsm": gsm,
        "sheet_size": size, "job_size": size, "printing_type": print_type,
        "print_side": "FB" if print_type == "Front & Back" else "Single Sided",
        "binding": binding, "job_type": "Single Page" if binding == "No Binding" else "Multi Page",
        "has_lamination": int(lamination != "None"), "num_lamination_types": int(lamination != "None"),
        "has_gloss": int(lamination == "Gloss"), "has_matte": int(lamination == "Matte"),
        "paper_type": paper, "paper_supplied_by": supplied, "paper_purchase_from": purchase,
        "num_paper_entries": 1, "ctp_supplied_by": ctp, "ctp_sets": sets,
    }
    try:
        amount = predict_quote(spec, output_dir=Path(__file__).parent / "artifacts", mode=model)
        st.metric("Estimated quotation (not a final price)", f"{currency} {amount:,.2f}")
        st.warning("Estimate only. Unknown categories may be ignored by the encoder; confirm against the training data and current costs.")
    except (OSError, ValueError, RuntimeError) as exc:
        st.error(str(exc))
        st.info("Train locally first: python scripts/train.py --db path/to/my_database.db")
