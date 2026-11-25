
import sqlite3
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
import joblib

# ==========================================
# 1. LOAD DATA
# ==========================================

with sqlite3.connect("data/nyc_311.db") as conn:
    df = pd.read_sql_query("""
        SELECT
            borough,
            problem_formerly_complaint_type AS complaint,
            day_of_week,
            hour,
            created_month,
            resolution_hours
        FROM service_requests
        WHERE LOWER(TRIM(status)) = 'closed'
          AND resolution_hours IS NOT NULL
          AND resolution_hours >= 0
          AND borough IN (
              'BRONX', 'BROOKLYN',
              'MANHATTAN', 'QUEENS',
              'STATEN ISLAND'
          )
    """, conn)

# Define target
df["delayed"] = (df["resolution_hours"] > 168).astype(int)

# Use request characteristics, not closure information
features = [
    "borough",
    "complaint",
    "day_of_week",
    "hour"
]

X = df[features].fillna("Unknown").astype(str)
y = df["delayed"]

# ==========================================
# 2. CHRONOLOGICAL TRAIN / TEST SPLIT
# ==========================================

train_mask = df["created_month"].isin([
    "2025-01", "2025-04", "2025-07"
])

test_mask = df["created_month"].eq("2025-10")

X_train = X.loc[train_mask]
X_test = X.loc[test_mask]

y_train = y.loc[train_mask]
y_test = y.loc[test_mask]

assert len(X_train) > 0
assert len(X_test) > 0

print("\nTRAINING DATA:", len(X_train))
print("TEST DATA:", len(X_test))

print(
    "Training delay rate:",
    round(y_train.mean() * 100, 2), "%"
)

print(
    "Test delay rate:",
    round(y_test.mean() * 100, 2), "%"
)

# ==========================================
# 3. BASELINE MODEL
# ==========================================

baseline = DummyClassifier(strategy="prior")
baseline.fit(X_train, y_train)

baseline_probs = baseline.predict_proba(X_test)[:, 1]
baseline_pred = baseline.predict(X_test)

# ==========================================
# 4. LOGISTIC REGRESSION
# ==========================================

model = Pipeline([
    (
        "encoder",
        OneHotEncoder(handle_unknown="ignore")
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            solver="liblinear"
        )
    )
])

print("\nTraining logistic regression...")

model.fit(X_train, y_train)

# Predict on unseen October data
probabilities = model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= 0.5).astype(int)

# ==========================================
# 5. MODEL EVALUATION
# ==========================================

def evaluate(name, actual, pred, probs):
    return {
        "model": name,
        "accuracy": accuracy_score(actual, pred),
        "balanced_accuracy": balanced_accuracy_score(
            actual, pred
        ),
        "precision": precision_score(
            actual, pred, zero_division=0
        ),
        "recall": recall_score(
            actual, pred, zero_division=0
        ),
        "f1": f1_score(
            actual, pred, zero_division=0
        ),
        "roc_auc": roc_auc_score(actual, probs),
        "pr_auc": average_precision_score(
            actual, probs
        )
    }

results = pd.DataFrame([
    evaluate(
        "Dummy baseline",
        y_test,
        baseline_pred,
        baseline_probs
    ),
    evaluate(
        "Logistic regression",
        y_test,
        predictions,
        probabilities
    )
])

print("\nMODEL COMPARISON")
print(results.round(4).to_string(index=False))

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test, predictions))

# ==========================================
# 6. SAVE RESULTS
# ==========================================

Path("models").mkdir(exist_ok=True)
Path("reports/tables").mkdir(parents=True, exist_ok=True)

joblib.dump(
    model,
    "models/nyc_311_delay_logistic.joblib"
)

results.to_csv(
    "reports/tables/ml_model_comparison.csv",
    index=False
)

print("\nMODEL TRAINING COMPLETE!")
print("Model saved to models/")
