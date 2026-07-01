"""
Trains and compares 3 models for predicting equipment failure:
- Logistic Regression (interpretable baseline)
- Random Forest (handles non-linearity, gives feature importance)
- XGBoost (typically strongest performer)

Saves the best model (by recall, since missed failures are costly) to
data/processed/best_model.pkl
"""

import pandas as pd
import numpy as np
import joblib
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("xgboost not installed - run: pip install xgboost --break-system-packages")


DROP_COLS = ["unit_id", "cycle", "max_cycle", "RUL", "failure_within_horizon"]
TARGET = "failure_within_horizon"


def load_features(path: str = "data/processed/features.csv"):
    df = pd.read_csv(path)
    # split by unit_id so no data leakage between train/test (same machine
    # shouldn't appear in both sets)
    unit_ids = df["unit_id"].unique()
    train_ids, test_ids = train_test_split(unit_ids, test_size=0.25, random_state=42)

    train_df = df[df["unit_id"].isin(train_ids)]
    test_df = df[df["unit_id"].isin(test_ids)]

    feature_cols = [c for c in df.columns if c not in DROP_COLS]

    X_train, y_train = train_df[feature_cols], train_df[TARGET]
    X_test, y_test = test_df[feature_cols], test_df[TARGET]

    return X_train, X_test, y_train, y_test, feature_cols


def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "precision": round(precision_score(y_test, preds), 3),
        "recall": round(recall_score(y_test, preds), 3),
        "f1": round(f1_score(y_test, preds), 3),
        "roc_auc": round(roc_auc_score(y_test, probs), 3),
    }
    print(f"\n{name}")
    print(json.dumps(metrics, indent=2))
    print(confusion_matrix(y_test, preds))
    return metrics


def main():
    X_train, X_test, y_train, y_test, feature_cols = load_features()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []

    # 1. Logistic Regression
    log_reg = LogisticRegression(max_iter=1000, class_weight="balanced")
    log_reg.fit(X_train_scaled, y_train)
    results.append(evaluate("Logistic Regression", log_reg, X_test_scaled, y_test))

    # 2. Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=10,
                                 class_weight="balanced", random_state=42)
    rf.fit(X_train, y_train)
    results.append(evaluate("Random Forest", rf, X_test, y_test))

    # 3. XGBoost
    best_model, best_name = rf, "Random Forest"
    if HAS_XGB:
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        xgb = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            scale_pos_weight=scale_pos_weight, eval_metric="logloss",
            random_state=42
        )
        xgb.fit(X_train, y_train)
        results.append(evaluate("XGBoost", xgb, X_test, y_test))

    # pick best model by recall (missed failures are the costly error)
    results_df = pd.DataFrame(results)
    best_row = results_df.sort_values("recall", ascending=False).iloc[0]
    print(f"\nBest model by recall: {best_row['model']}")

    model_map = {"Logistic Regression": log_reg, "Random Forest": rf}
    if HAS_XGB:
        model_map["XGBoost"] = xgb
    best_model = model_map[best_row["model"]]

    os.makedirs("data/processed", exist_ok=True)
    joblib.dump({"model": best_model, "scaler": scaler,
                 "feature_cols": feature_cols, "model_name": best_row["model"]},
                "data/processed/best_model.pkl")
    results_df.to_csv("data/processed/model_comparison.csv", index=False)
    print("\nSaved best model -> data/processed/best_model.pkl")
    print("Saved comparison table -> data/processed/model_comparison.csv")


if __name__ == "__main__":
    main()
