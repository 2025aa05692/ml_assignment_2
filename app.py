"""
ML Assignment 2 - Streamlit app for classification model demo.
Features: CSV upload, model selection, evaluation metrics, confusion matrix.
"""
import os
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVED_DIR = os.path.join(APP_DIR, "model", "saved")
TARGET_COL = "quality"

# Display name -> saved filename (without .joblib)
MODEL_FILE_MAP = {
    "Logistic Regression": "Logistic_Regression",
    "Decision Tree": "Decision_Tree",
    "kNN": "kNN",
    "Naive Bayes": "Naive_Bayes",
    "Random Forest (Ensemble)": "Random_Forest_Ensemble",
    "XGBoost (Ensemble)": "XGBoost_Ensemble",
}


@st.cache_resource
def load_models_and_metrics():
    """Load scaler, feature names, available models, and saved metrics."""
    scaler_path = os.path.join(MODEL_SAVED_DIR, "scaler.joblib")
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    fn_path = os.path.join(MODEL_SAVED_DIR, "feature_names.joblib")
    feature_names = joblib.load(fn_path) if os.path.exists(fn_path) else None

    models = {}
    for display_name, file_stem in MODEL_FILE_MAP.items():
        path = os.path.join(MODEL_SAVED_DIR, f"{file_stem}.joblib")
        if os.path.exists(path):
            models[display_name] = joblib.load(path)

    metrics_path = os.path.join(MODEL_SAVED_DIR, "metrics.json")
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)

    le_path = os.path.join(MODEL_SAVED_DIR, "label_encoder_xgb.joblib")
    label_encoder_xgb = joblib.load(le_path) if os.path.exists(le_path) else None

    return scaler, feature_names, models, metrics, label_encoder_xgb


def main():
    st.set_page_config(page_title="ML Assignment 2 - Classification Demo", layout="wide")
    st.title("Machine Learning Assignment 2 - Classification Models Demo")
    st.markdown("Upload a **test** CSV (same schema as training: 12 features + `quality` target).")

    scaler, feature_names, models, saved_metrics, label_encoder_xgb = load_models_and_metrics()
    if not models:
        st.error("No saved models found. Run model/train_models.py first.")
        return

    # a) Dataset upload (CSV)
    uploaded = st.file_uploader("Upload test data (CSV)", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV file to run predictions and see metrics.")
        # Still show saved metrics and model comparison
        st.subheader("Saved evaluation metrics (on train/test split)")
        if saved_metrics:
            st.dataframe(pd.DataFrame(saved_metrics).T)
        return

    df = pd.read_csv(uploaded)
    if TARGET_COL not in df.columns:
        st.error(f"CSV must contain target column: '{TARGET_COL}'.")
        return

    X = df.drop(columns=[TARGET_COL], errors="ignore")
    y_true = df[TARGET_COL]

    # Align columns to training feature order
    if feature_names:
        missing = set(feature_names) - set(X.columns)
        if missing:
            st.warning(f"Missing columns: {list(missing)}. Using 0 for missing.")
        X = X.reindex(columns=feature_names, fill_value=0)
    X_scaled = scaler.transform(X) if scaler is not None else X.values

    # b) Model selection dropdown
    model_name = st.selectbox("Select model", options=list(models.keys()))
    model = models[model_name]

    # Predict (XGBoost returns 0..n-1; decode to original labels)
    y_pred = model.predict(X_scaled)
    if model_name == "XGBoost (Ensemble)" and label_encoder_xgb is not None:
        y_pred = label_encoder_xgb.inverse_transform(y_pred.astype(int))

    # c) Display evaluation metrics
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        matthews_corrcoef, roc_auc_score, confusion_matrix, classification_report
    )

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    auc_val = 0.0
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_scaled)
            auc_val = roc_auc_score(y_true, proba, multi_class="ovr", average="weighted")
        except Exception:
            pass

    st.subheader("Evaluation metrics (on uploaded test data)")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Accuracy", f"{acc:.4f}")
    col2.metric("AUC", f"{auc_val:.4f}")
    col3.metric("Precision", f"{precision:.4f}")
    col4.metric("Recall", f"{recall:.4f}")
    col5.metric("F1", f"{f1:.4f}")
    col6.metric("MCC", f"{mcc:.4f}")

    # d) Confusion matrix or classification report
    st.subheader("Confusion matrix")
    cm = confusion_matrix(y_true, y_pred)
    labels = sorted(np.unique(np.concatenate([y_true.values, y_pred])))
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    st.pyplot(fig)
    plt.close()

    st.subheader("Classification report")
    report = classification_report(y_true, y_pred, zero_division=0)
    st.text(report)

    # Show saved metrics table for comparison
    st.subheader("Saved metrics (train/test split) for all models")
    if saved_metrics:
        st.dataframe(pd.DataFrame(saved_metrics).T)
# --- Download sample test CSV ---
test_csv_path = os.path.join(model, "test_data_ml.csv")
if os.path.exists(test_csv_path):
    with open(test_csv_path, "rb") as f:
        st.download_button(
            label="Download sample test CSV",
            data=f,
            file_name="test_data_ml.csv",
            mime="text/csv"
        )
else:
    st.warning("Sample test CSV not found in model folder.")


if __name__ == "__main__":
    main()
