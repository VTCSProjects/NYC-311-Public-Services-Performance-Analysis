
import sqlite3
from pathlib import Path

import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score
)
from sklearn.base import clone

# Load database
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
              'BRONX', 'BROOKLYN', 'MANHATTAN',
              'QUEENS', 'STATEN ISLAND'
          )
    """, conn)

df["delayed"] = (df["resolution_hours"] > 168).astype(int)

features = ["borough", "complaint", "day_of_week", "hour"]
X = df[features].fillna("Unknown").astype(str)
y = df["delayed"]

# Initial training: January and April
train = df["created_month"].isin(["2025-01", "2025-04"])

# Model selection: July
val = df["created_month"].eq("2025-07")

# Final evaluation: October
test = df["created_month"].eq("2025-10")

def pipeline(classifier):
    return Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ("model", classifier)
    ])

models = {
    "Complaint only - Logistic": (
        ["complaint"],
        pipeline(LogisticRegression(
            max_iter=1000, solver="liblinear"
        ))
    ),
    "All features - Logistic": (
        features,
        pipeline(LogisticRegression(
            max_iter=1000, solver="liblinear"
        ))
    ),
    "All features - Random Forest": (
        features,
        pipeline(RandomForestClassifier(
            n_estimators=120,
            max_depth=20,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        ))
    )
}

def evaluate(name, model, columns, mask):
    probs = model.predict_proba(X.loc[mask, columns])[:, 1]
    pred = (probs >= 0.5).astype(int)
    actual = y.loc[mask]

    return {
        "model": name,
        "accuracy": accuracy_score(actual, pred),
        "precision": precision_score(
            actual, pred, zero_division=0
        ),
        "recall": recall_score(
            actual, pred, zero_division=0
        ),
        "f1": f1_score(actual, pred, zero_division=0),
        "roc_auc": roc_auc_score(actual, probs),
        "pr_auc": average_precision_score(actual, probs)
    }

# Compare models on July
validation_results = []

for name, (columns, model) in models.items():
    print(f"\nTraining: {name}")

    model.fit(X.loc[train, columns], y.loc[train])

    validation_results.append(
        evaluate(name, model, columns, val)
    )

validation = pd.DataFrame(validation_results)
validation = validation.sort_values("pr_auc", ascending=False)

print("\nJULY VALIDATION RESULTS")
print(validation.round(4).to_string(index=False))

# Select best model using July PR-AUC
winner = validation.iloc[0]["model"]
columns, chosen_model = models[winner]

print("\nSelected model:", winner)

# Refit on January, April and July
final_train = train | val
final_model = clone(chosen_model)

final_model.fit(
    X.loc[final_train, columns],
    y.loc[final_train]
)

# Evaluate on October
final_results = pd.DataFrame([
    evaluate(winner, final_model, columns, test)
])

print("\nOCTOBER FINAL EVALUATION")
print(final_results.round(4).to_string(index=False))

# Save artifacts
Path("models").mkdir(exist_ok=True)
Path("reports/tables").mkdir(parents=True, exist_ok=True)

joblib.dump(
    {
        "model": final_model,
        "features": columns
    },
    "models/nyc_311_delay_best.joblib"
)

validation.to_csv(
    "reports/tables/ml_validation_comparison.csv",
    index=False
)

final_results.to_csv(
    "reports/tables/ml_holdout_evaluation.csv",
    index=False
)

print("\nMODEL COMPARISON COMPLETE!")
