# Diabetes Prediction — Classification Project

A supervised classification project comparing five machine learning models to predict diabetes risk from patient health data, with a focus on handling class imbalance and choosing a decision threshold appropriate for a medical screening context.

## Dataset

- **Source:** Kaggle — Diabetes Prediction Dataset
- **Size:** ~100,000 rows
- **Target:** `diabetes` (binary — 0 = no diabetes, 1 = diabetes)
- **Features:** `gender`, `age`, `hypertension`, `heart_disease`, `smoking_history`, `bmi`, `HbA1c_level`, `blood_glucose_level`
- **Class balance:** Imbalanced — roughly 91% negative (no diabetes) vs. 9% positive (diabetes) in the test set (27,453 vs. 2,547)

## Preprocessing

Built with `ColumnTransformer` + `Pipeline` to avoid data leakage (fit only on training data, never on the full dataset before splitting):

| Column type | Columns | Steps |
|---|---|---|
| Numeric | `age`, `bmi`, `HbA1c_level`, `blood_glucose_level` | Median imputation → `StandardScaler` |
| Categorical | `gender`, `smoking_history` | Most-frequent imputation → `OneHotEncoder(drop='first')` |

Train/test split: 80/20, stratified on the target (`stratify=y`) to preserve class balance in both sets.

## Models Compared

Each model was wrapped in the same preprocessing pipeline and evaluated identically.

| Model | Precision (1) | Recall (1) | F1 (1) | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.87 | 0.60 | 0.71 | 0.960 |
| KNN | 0.91 | 0.61 | 0.73 | 0.905 |
| SVM | 0.99 | 0.58 | 0.73 | 0.934 |
| Random Forest | 0.95 | 0.68 | 0.79 | 0.959 |
| **XGBoost** | **0.96** | **0.68** | **0.80** | **0.977** |

**Winner: XGBoost** — best or tied-best across precision, F1, and ROC-AUC. Tree-based models (Random Forest, XGBoost) outperformed distance/linear-based models, likely due to their ability to capture non-linear feature interactions common in medical risk factors (e.g. age × BMI × glucose level).

## Decision Threshold Tuning

All models above use the default 0.5 classification threshold, which under-catches positive cases (recall of only ~0.68 even for the best model) — a risk in a screening context where missing a real case is costlier than a false alarm.

A precision-recall curve was plotted across all thresholds for the XGBoost model to find a better balance:

| Threshold | Precision (1) | Recall (1) | False Negatives |
|---|---|---|---|
| 0.10 | 0.50 | 0.89 | 286 |
| 0.20 | 0.70 | 0.79 | 532 |
| **0.23** | **0.74** | **0.77** | **581** |
| 0.30 | 0.84 | 0.73 | 677 |
| 0.40 | 0.92 | 0.70 | 764 |
| 0.50 (default) | 0.96 | 0.68 | 810 |

**Final threshold chosen: 0.23** — the point where precision and recall cross and balance (~0.74–0.77 each), rather than accepting the default 0.5. This threshold catches 229 more real diabetic cases than the default, at the cost of 604 more false alarms — a trade-off favoring recall, appropriate for a medical screening use case where a false alarm (follow-up test) is less costly than a missed diagnosis.

## Final Model

- **Algorithm:** XGBoost Classifier (full pipeline: preprocessing + model)
- **Decision threshold:** 0.23 (applied manually on `predict_proba` output, not the pipeline's default `.predict()`)
- **Saved as:** `diabetes_xgb_model.joblib`

```python
import joblib

# Load
xgb_pipe = joblib.load('diabetes_xgb_model.joblib')

# Predict using the custom threshold
probabilities = xgb_pipe.predict_proba(X_new)[:, 1]
predictions = (probabilities >= 0.23).astype(int)
```

## Key Takeaways

- Accuracy is a misleading metric on imbalanced data (96%+ accuracy was achievable even with poor recall) — precision, recall, F1, and ROC-AUC give a truer picture.
- The default 0.5 classification threshold is an arbitrary sklearn convention, not a rule — it can and should be tuned to match the real-world cost of false positives vs. false negatives.
- Tree-based ensemble models (Random Forest, XGBoost) outperformed linear/distance-based models (Logistic Regression, KNN, SVM) on this tabular dataset.
