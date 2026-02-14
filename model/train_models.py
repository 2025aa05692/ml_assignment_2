"""
ML Assignment 2 - Train all 6 classification models and save them.
Dataset: Wine Quality (12 features, 1599 instances).
Metrics: Accuracy, AUC, Precision, Recall, F1, MCC.
"""
import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

# XGBoost (optional import for environments without it)
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(PROJECT_DIR, "data", "wine_quality_12features.csv")
MODEL_DIR = os.path.join(SCRIPT_DIR, "saved")
os.makedirs(MODEL_DIR, exist_ok=True)

TARGET_COL = "quality"
RANDOM_STATE = 42


def load_data():
    """Load and split data."""
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    # Multi-class: use stratify
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler


def multiclass_auc(y_true, y_proba, average="weighted"):
    """Compute AUC for multi-class using OneVsRest or weighted."""
    n_classes = len(np.unique(y_true))
    if n_classes == 2:
        return roc_auc_score(y_true, y_proba[:, 1])
    return roc_auc_score(y_true, y_proba, multi_class="ovr", average=average)


def evaluate(y_true, y_pred, y_proba=None):
    """Compute all required metrics."""
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    auc = 0.0
    if y_proba is not None:
        try:
            auc = multiclass_auc(y_true, y_proba)
        except Exception:
            pass
    return {
        "Accuracy": round(acc, 4),
        "AUC": round(auc, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1": round(f1, 4),
        "MCC": round(mcc, 4),
    }


def main():
    # Prepare data (create data file if missing)
    if not os.path.exists(DATA_PATH):
        import sys
        sys.path.insert(0, PROJECT_DIR)
        from prepare_data import feature_cols
        # Run prepare_data
        exec(open(os.path.join(PROJECT_DIR, "prepare_data.py")).read())

    X_train, X_test, y_train, y_test, scaler = load_data()
    df = pd.read_csv(DATA_PATH)
    feature_names = list(df.drop(columns=[TARGET_COL]).columns)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    joblib.dump(feature_names, os.path.join(MODEL_DIR, "feature_names.joblib"))

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "kNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "Random Forest (Ensemble)": RandomForestClassifier(random_state=RANDOM_STATE),
    }
    label_encoder = None
    if HAS_XGB:
        models["XGBoost (Ensemble)"] = xgb.XGBClassifier(
            eval_metric="mlogloss", random_state=RANDOM_STATE
        )
        label_encoder = LabelEncoder()
        label_encoder.fit(y_train)

    def safe_name(n):
        return n.replace(" ", "_").replace("(", "").replace(")", "")

    results = {}
    for name, model in models.items():
        y_tr, y_te = y_train, y_test
        if name == "XGBoost (Ensemble)" and label_encoder is not None:
            y_tr = label_encoder.transform(y_train)
            y_te = label_encoder.transform(y_test)
        model.fit(X_train, y_tr)
        y_pred = model.predict(X_test)
        if name == "XGBoost (Ensemble)" and label_encoder is not None:
            y_pred = label_encoder.inverse_transform(y_pred.astype(int))
        y_proba = getattr(model, "predict_proba", lambda X: None)(X_test)
        metrics = evaluate(y_test, y_pred, y_proba)
        results[name] = metrics
        joblib.dump(model, os.path.join(MODEL_DIR, f"{safe_name(name)}.joblib"))
        print(f"{name}: Accuracy={metrics['Accuracy']}, F1={metrics['F1']}")
    if label_encoder is not None:
        joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder_xgb.joblib"))

    # Save metrics for README and app
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Also save confusion matrix info from last model for app
    cm = confusion_matrix(y_test, models["Logistic Regression"].predict(X_test))
    np.save(os.path.join(MODEL_DIR, "confusion_matrix_example.npy"), cm)
    print("Models and metrics saved to", MODEL_DIR)


if __name__ == "__main__":
    main()
