import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)

st.set_page_config(page_title="Adult Income Classification", layout="centered")

st.title("Adult Income Classification App")

# -------------------------------
# Model Selection
# -------------------------------
model_name = st.selectbox(
    "Select Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

# -------------------------------
# Download Sample Test Dataset
# -------------------------------
st.subheader("📥 Download Sample Test Dataset")

test_data_path = os.path.join("model", "test_data_ml.csv")

if os.path.exists(test_data_path):
    sample_df = pd.read_csv(test_data_path)
    csv_sample = sample_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download test_data_ml.csv",
        data=csv_sample,
        file_name="test_data_ml.csv",
        mime="text/csv"
    )
else:
    st.error("test_data_ml.csv not found inside model folder.")

# -------------------------------
# Upload Test File
# -------------------------------
uploaded_file = st.file_uploader("Upload Test CSV File", type=["csv"])

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    if "income" not in data.columns:
        st.error("Uploaded file must contain 'income' column.")
        st.stop()

    X = data.drop("income", axis=1)
    y = data["income"]

    # -------------------------------
    # Model Files Mapping
    # -------------------------------
    model_files = {
        "Logistic Regression": "Logistic Regression.pkl",
        "Decision Tree": "Decision Tree.pkl",
        "KNN": "KNN.pkl",
        "Naive Bayes": "Naive Bayes.pkl",
        "Random Forest": "Random Forest.pkl",
        "XGBoost": "XGBoost.pkl"
    }

    model_path = os.path.join("model", model_files[model_name])

    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.stop()

    model = joblib.load(model_path)

    # -------------------------------
    # Predictions
    # -------------------------------
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    # -------------------------------
    # Evaluation Metrics
    # -------------------------------
    st.subheader("📊 Evaluation Metrics")

    st.write("Accuracy:", round(accuracy_score(y, y_pred), 4))
    st.write("AUC:", round(roc_auc_score(y, y_prob), 4))
    st.write("Precision:", round(precision_score(y, y_pred), 4))
    st.write("Recall:", round(recall_score(y, y_pred), 4))
    st.write("F1 Score:", round(f1_score(y, y_pred), 4))
    st.write("MCC:", round(matthews_corrcoef(y, y_pred), 4))

    # -------------------------------
    # Confusion Matrix
    # -------------------------------
    st.subheader("🔢 Confusion Matrix")

    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=["<=50K", ">50K"],
        yticklabels=["<=50K", ">50K"],
        ax=ax
    )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_title("Confusion Matrix")

    st.pyplot(fig)

    # -------------------------------
    # Download Predictions
    # -------------------------------
    result_df = data.copy()
    result_df["Predicted_Income"] = y_pred
    result_df["Prediction_Probability"] = y_prob

    csv_result = result_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Predictions CSV",
        data=csv_result,
        file_name=f"{model_name}_predictions.csv",
        mime="text/csv"
    )

else:
    st.info("Please select a model and upload a test dataset to view results.")
