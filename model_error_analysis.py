
import sqlite3
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix

# Load selected model
saved = joblib.load(
    "models/nyc_311_delay_best.joblib"
)

model = saved["model"]
features = saved["features"]

# Load October evaluation records
with sqlite3.connect("data/nyc_311.db") as conn:
    df = pd.read_sql_query("""
        SELECT
            unique_key,
            borough,
            agency,
            problem_formerly_complaint_type AS complaint,
            day_of_week,
            hour,
            created_month,
            resolution_hours
        FROM service_requests
        WHERE LOWER(TRIM(status)) = 'closed'
          AND resolution_hours IS NOT NULL
          AND resolution_hours >= 0
          AND created_month = '2025-10'
          AND borough IN (
              'BRONX', 'BROOKLYN', 'MANHATTAN',
              'QUEENS', 'STATEN ISLAND'
          )
    """, conn)

X = df[features].fillna("Unknown").astype(str)

df["actual_delay"] = (
    df["resolution_hours"] > 168
).astype(int)

df["predicted_probability"] = (
    model.predict_proba(X)[:, 1]
)

df["predicted_delay"] = (
    df["predicted_probability"] >= 0.5
).astype(int)

# Evaluate errors
tn, fp, fn, tp = confusion_matrix(
    df["actual_delay"],
    df["predicted_delay"]
).ravel()

print("\nMODEL ERROR SUMMARY")
print("True negatives:", tn)
print("False positives:", fp)
print("False negatives:", fn)
print("True positives:", tp)

# Identify missed delays
missed = df[
    (df["actual_delay"] == 1)
    & (df["predicted_delay"] == 0)
]

print("\nTOP 10 COMPLAINTS WITH MISSED DELAYS")
print(
    missed["complaint"]
    .value_counts()
    .head(10)
    .to_string()
)

# Identify false alarms
false_alarms = df[
    (df["actual_delay"] == 0)
    & (df["predicted_delay"] == 1)
]

print("\nTOP 10 COMPLAINTS WITH FALSE ALARMS")
print(
    false_alarms["complaint"]
    .value_counts()
    .head(10)
    .to_string()
)

# Compare predicted and actual rates
print("\nPREDICTION CALIBRATION")

df["risk_group"] = pd.cut(
    df["predicted_probability"],
    bins=[0, .2, .4, .6, .8, 1],
    include_lowest=True
)

calibration = df.groupby(
    "risk_group",
    observed=True
).agg(
    requests=("actual_delay", "size"),
    actual_delay_pct=("actual_delay", "mean"),
    predicted_delay_pct=("predicted_probability", "mean")
)

calibration[
    ["actual_delay_pct", "predicted_delay_pct"]
] *= 100

print(calibration.round(2).to_string())

# Save results
output = Path("reports/tables")
output.mkdir(parents=True, exist_ok=True)

calibration.to_csv(
    output / "model_calibration.csv"
)

missed.to_csv(
    output / "missed_delays.csv",
    index=False
)

false_alarms.to_csv(
    output / "false_alarms.csv",
    index=False
)

print("\nERROR ANALYSIS COMPLETE!")
