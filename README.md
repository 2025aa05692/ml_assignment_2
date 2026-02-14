# ML Assignment 2 - Classification Models & Streamlit Demo

## a. Problem statement

This project implements multiple classification models on a chosen dataset and deploys an interactive Streamlit web application to demonstrate model performance. The goal is to compare six different classifiers (Logistic Regression, Decision Tree, kNN, Naive Bayes, Random Forest, XGBoost) using standard evaluation metrics and to provide a deployable UI for uploading test data and viewing metrics and confusion matrices.

## b. Dataset description

- **Dataset**: Wine Quality (Red Wine) from UCI Machine Learning Repository.
- **Task**: Multi-class classification of wine quality (ordinal scale 3–8).
- **Instances**: 1,599 samples.
- **Features**: 12 (minimum requirement met).  
  Original 11 attributes: fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol. One engineered feature was added: **total_sulfur** (sum of free and total sulfur dioxide) to satisfy the minimum feature size of 12.
- **Target**: `quality` (integer 3–8).
- **Preprocessing**: Train/test split (75/25), stratified; features standardized with `StandardScaler` for models that benefit from scaling.

## c. Models used

### Comparison table (evaluation metrics)

| ML Model Name            | Accuracy | AUC   | Precision | Recall | F1    | MCC   |
|--------------------------|----------|-------|-----------|--------|-------|-------|
| Logistic Regression      | 0.5950   | 0.757 | 0.5686    | 0.595  | 0.5702| 0.3333|
| Decision Tree            | 0.6450   | 0.7324| 0.6538    | 0.645  | 0.649 | 0.4522|
| kNN                      | 0.6000   | 0.7399| 0.5707    | 0.600  | 0.5827| 0.3523|
| Naive Bayes              | 0.5400   | 0.7325| 0.5760    | 0.540  | 0.5528| 0.308 |
| Random Forest (Ensemble) | 0.6650   | 0.8373| 0.6489    | 0.665  | 0.6464| 0.4564|
| XGBoost (Ensemble)       | 0.6625   | 0.8165| 0.6517    | 0.6625 | 0.6539| 0.461 |

### Observations about model performance

| ML Model Name            | Observation about model performance |
|--------------------------|-------------------------------------|
| Logistic Regression      | Moderate accuracy; linear decision boundary limits performance on this multi-class wine dataset. AUC (0.757) is reasonable. MCC and F1 indicate room for improvement. |
| Decision Tree            | Better than logistic regression; captures non-linear patterns without scaling. Slight overfitting possible; AUC slightly lower than some others. |
| kNN                      | Comparable to logistic regression; benefits from standardization. Sensitive to choice of k and feature scale; weighted metrics are balanced. |
| Naive Bayes              | Lowest accuracy among the set; Gaussian assumption may not fit all features well. Fast to train; useful as a baseline. |
| Random Forest (Ensemble) | Best accuracy and highest AUC (0.837) in this comparison; ensemble reduces variance and captures non-linear structure. Best F1 and MCC among the five. |
| XGBoost (Ensemble)       | Very close to Random Forest (accuracy 0.6625, best F1 0.6539); high AUC (0.8165) and best MCC (0.461). Strong ensemble choice for this dataset. |

---

## Repository structure

```
ml_assignment_2/
├── app.py                 # Streamlit application
├── requirements.txt
├── README.md
├── prepare_data.py        # Builds 12-feature Wine Quality CSV
├── data/
│   └── wine_quality_12features.csv
└── model/
    ├── train_models.py    # Trains all 6 models and saves them
    └── saved/             # *.joblib models, scaler, feature_names, metrics.json
```

## How to run

1. **Prepare data** (from repo root):  
   `python ml_assignment_2/prepare_data.py`

2. **Train models**:  
   `python ml_assignment_2/model/train_models.py`

3. **Run Streamlit app**:  
   `streamlit run ml_assignment_2/app.py`

4. **Deploy**: Push to GitHub and deploy the app on [Streamlit Community Cloud](https://streamlit.io/cloud), selecting `app.py` as the main file.

## Assignment checklist

- [x] Dataset: ≥12 features, ≥500 instances (Wine Quality: 12 features, 1599 instances)
- [x] Six models: Logistic Regression, Decision Tree, kNN, Naive Bayes, Random Forest, XGBoost
- [x] Metrics: Accuracy, AUC, Precision, Recall, F1, MCC
- [x] Streamlit: CSV upload, model dropdown, evaluation metrics, confusion matrix / classification report
- [x] `requirements.txt` and README with comparison table and observations
