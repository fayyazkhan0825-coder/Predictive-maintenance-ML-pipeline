# Predictive Maintenance for Industrial Equipment
 LINK : https://fayyazkhan0825-coder-predictive-maintenance-dashboardapp-0zatv5.streamlit.app/

End-to-end ML pipeline that predicts equipment failure before it happens,
using time-series sensor data (temperature, vibration, pressure, etc.).

Built to mirror the kind of solution IBM Consulting delivers via IBM Maximo
for manufacturing / industrial clients.

---

## Project Structure

```
predictive_maintenance/
├── data/
│   ├── raw/            <- original sensor data goes here
│   └── processed/      <- cleaned + feature-engineered data
├── src/
│   ├── generate_data.py      <- creates realistic synthetic sensor data
│   ├── data_pipeline.py      <- ingestion + cleaning
│   ├── feature_engineering.py<- rolling stats, degradation trends, labels
│   ├── train_model.py        <- trains & compares 3 models
│   └── evaluate.py           <- metrics, confusion matrix, feature importance
├── dashboard/
│   └── app.py           <- Streamlit dashboard for stakeholders
└── README.md
```

## The Business Problem (use this framing on your resume/interview)

> "A factory has multiple machines, each with sensors streaming readings
> (temperature, vibration, pressure, rotation speed). Machines fail
> unpredictably, causing costly unplanned downtime. Goal: predict — using
> only sensor data — whether a machine will fail within the next N cycles,
> so maintenance can be scheduled proactively instead of reactively."

## Step-by-Step Build Plan

### Step 1 — Get data
Two options:
- **Option A (fastest, works offline):** Run `src/generate_data.py` — it
  creates realistic synthetic multi-sensor time-series data with actual
  degradation patterns before failure (same structure as NASA's C-MAPSS
  turbofan dataset).
- **Option B (more impressive on resume):** Download the real
  [NASA C-MAPSS Turbofan Engine Degradation dataset](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps)
  from Kaggle and drop the files into `data/raw/`. The pipeline code works
  with either — same column structure.

### Step 2 — Data pipeline (`data_pipeline.py`)
Ingests raw sensor logs, handles missing values, resamples time series,
and outputs a clean dataframe. This is your "Data Engineer" evidence.

### Step 3 — Feature engineering (`feature_engineering.py`)
- Rolling mean/std over sensor windows (captures trend, not just snapshot)
- Rate-of-change features (is the sensor value drifting?)
- Remaining Useful Life (RUL) label creation
- Binary failure label: "will this unit fail within next 30 cycles?"

### Step 4 — Model training (`train_model.py`)
Trains and compares:
- Logistic Regression (baseline, interpretable)
- Random Forest (handles non-linearity, gives feature importance)
- XGBoost (usually best performer, industry standard)

### Step 5 — Evaluation (`evaluate.py`)
- Precision / Recall / F1 / ROC-AUC (NOT just accuracy — false negatives
  = missed failures = expensive, so recall matters most here)
- Confusion matrix
- Feature importance chart (which sensors matter most)

### Step 6 — Dashboard (`dashboard/app.py`)
Streamlit app showing:
- Machine health score per unit
- Alert list: "these 5 machines need maintenance in the next 2 weeks"
- Model comparison chart

This is the piece you demo in an interview — a recruiter can see it
running, not just read code.

---

## Resume Bullet (once built)

> "Built end-to-end predictive maintenance pipeline on multi-sensor
> time-series data — engineered degradation-trend features, trained and
> compared 3 ML models (Logistic Regression, Random Forest, XGBoost)
> optimizing for recall to minimize missed failures, and built a Streamlit
> dashboard surfacing at-risk equipment to stakeholders."

## Next Steps After Building
1. Push to GitHub with this README
2. Record a 60-second Loom/screen recording of the dashboard for your
   portfolio/LinkedIn
3. Be ready to explain: why recall > accuracy here, and what features
   mattered most
