"""
Stakeholder-facing dashboard.
Run with: streamlit run dashboard/app.py

Shows:
- List of machines at risk of failure, ranked by risk score
- Model comparison chart
- Feature importance (which sensors drive predictions)
"""

import streamlit as st
import pandas as pd
import joblib
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")

st.title("🔧 Predictive Maintenance Dashboard")
st.caption("Which machines need maintenance soon — and why.")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


@st.cache_data
def load_data():
    features = pd.read_csv(os.path.join(DATA_DIR, "features.csv"))
    bundle = joblib.load(os.path.join(DATA_DIR, "best_model.pkl"))
    comparison = pd.read_csv(os.path.join(DATA_DIR, "model_comparison.csv"))
    return features, bundle, comparison


try:
    features, bundle, comparison = load_data()
except FileNotFoundError:
    st.error(
        "No processed data found. Run the pipeline first:\n\n"
        "```\npython src/generate_data.py\npython src/data_pipeline.py\n"
        "python src/feature_engineering.py\npython src/train_model.py\n```"
    )
    st.stop()

model = bundle["model"]
feature_cols = bundle["feature_cols"]
model_name = bundle["model_name"]

# latest reading per unit = current status
latest = features.sort_values("cycle").groupby("unit_id").tail(1).copy()

X_latest = latest[feature_cols]
if model_name == "Logistic Regression":
    X_latest = bundle["scaler"].transform(X_latest)

latest["risk_score"] = model.predict_proba(X_latest)[:, 1]
latest = latest.sort_values("risk_score", ascending=False)

col1, col2, col3 = st.columns(3)
col1.metric("Machines Monitored", len(latest))
col2.metric("High Risk (>70%)", int((latest["risk_score"] > 0.7).sum()))
col3.metric("Model in Use", model_name)

st.subheader("⚠️ Machines at Risk — Ranked")
display_df = latest[["unit_id", "cycle", "risk_score"]].copy()
display_df["risk_score"] = (display_df["risk_score"] * 100).round(1).astype(str) + "%"
display_df.columns = ["Machine ID", "Current Cycle", "Failure Risk (next 30 cycles)"]
st.dataframe(display_df.head(20), use_container_width=True)

st.subheader("📊 Model Comparison")
st.dataframe(comparison, use_container_width=True)
st.bar_chart(comparison.set_index("model")[["precision", "recall", "f1", "roc_auc"]])

st.subheader("🔍 Feature Importance")
img_path = os.path.join(DATA_DIR, "feature_importance.png")
if os.path.exists(img_path):
    st.image(img_path, caption="Which sensor trends drive failure predictions")
else:
    st.info("Run `python src/evaluate.py` to generate this chart.")

st.subheader("📈 Individual Machine Detail")
selected_unit = st.selectbox("Select a machine to inspect", latest["unit_id"].unique())
unit_history = features[features["unit_id"] == selected_unit]
sensor_to_plot = st.selectbox(
    "Sensor", ["temperature", "vibration", "pressure", "rotation_speed", "oil_quality"]
)
st.line_chart(unit_history.set_index("cycle")[sensor_to_plot])
