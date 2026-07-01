"""
Loads the saved best model and produces:
- Feature importance chart (which sensors matter most for failure prediction)
- Confusion matrix plot
Saves both as PNGs for use in your resume portfolio / README / dashboard.
"""

import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def load_model(path: str = "data/processed/best_model.pkl"):
    return joblib.load(path)


def plot_feature_importance(bundle, top_n: int = 10,
                             out_path: str = "data/processed/feature_importance.png"):
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    if not hasattr(model, "feature_importances_"):
        print(f"{bundle['model_name']} has no feature_importances_ attribute "
              "(e.g. Logistic Regression) - skipping chart.")
        return

    importances = pd.Series(model.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=True).tail(top_n)

    plt.figure(figsize=(8, 6))
    importances.plot(kind="barh", color="#2E86AB")
    plt.title(f"Top {top_n} Feature Importances — {bundle['model_name']}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved -> {out_path}")


def plot_confusion_matrix(bundle, features_path: str = "data/processed/features.csv",
                           out_path: str = "data/processed/confusion_matrix.png"):
    from train_model import load_features  # reuse same split logic
    _, X_test, _, y_test, _ = load_features(features_path)

    model = bundle["model"]
    model_name = bundle["model_name"]

    X_input = X_test
    if model_name == "Logistic Regression":
        X_input = bundle["scaler"].transform(X_test)

    preds = model.predict(X_input)
    cm = confusion_matrix(y_test, preds)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=["No Failure Soon", "Failure Soon"])
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    bundle = load_model()
    plot_feature_importance(bundle)
    plot_confusion_matrix(bundle)
