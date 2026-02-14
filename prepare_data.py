"""
Prepare Wine Quality dataset with 12+ features and 500+ instances.
Adds one engineered feature to meet assignment minimum (12 features).
Source: UCI Wine Quality (red wine).
"""
import pandas as pd
import os

# Load wine quality (semicolon-separated)
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

wine_path = os.path.join(os.path.dirname(__file__), "..", "winequality-red.csv")
df = pd.read_csv(wine_path, sep=";")

# Add 12th feature: total sulfur (free + total sulfur dioxide)
df["total_sulfur"] = df["free sulfur dioxide"] + df["total sulfur dioxide"]

# Target: quality (multi-class 3-8)
# Ensure we have feature columns only (12) + target
feature_cols = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol", "total_sulfur"
]
target_col = "quality"

out_path = os.path.join(DATA_DIR, "wine_quality_12features.csv")
df[feature_cols + [target_col]].to_csv(out_path, index=False)
print(f"Saved {len(df)} rows, {len(feature_cols)} features to {out_path}")
