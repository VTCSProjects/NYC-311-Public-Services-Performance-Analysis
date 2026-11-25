
import sqlite3
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Load data
with sqlite3.connect("data/nyc_311.db") as conn:
    df = pd.read_sql_query("""
        SELECT
            borough,
            agency,
            problem_formerly_complaint_type AS complaint,
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

# Target: closure took over 7 days
df["delayed_7d"] = (
    df["resolution_hours"] > 168
).astype(int)

features = [
    "borough",
    "agency",
    "complaint",
    "created_month"
]

X = df[features].fillna("Unknown").astype(str)
y = df["delayed_7d"]

# Regularized logistic regression
model = Pipeline([
    (
        "encoder",
        OneHotEncoder(handle_unknown="ignore")
    ),
    (
        "classifier",
        LogisticRegression(max_iter=500)
    )
])

print("Fitting adjusted logistic regression...")
model.fit(X, y)

# Observed delay rates
results = (
    df.groupby("borough")["delayed_7d"]
    .agg(["size", "mean"])
    .reset_index()
)

results.columns = [
    "borough",
    "requests",
    "observed_rate"
]

# Model-based standardization:
# Assign each borough to the same case mix
adjusted_rates = {}

for borough in sorted(df["borough"].unique()):
    scenario = X.copy()
    scenario["borough"] = borough

    probabilities = model.predict_proba(
        scenario
    )[:, 1]

    adjusted_rates[borough] = probabilities.mean()

results["adjusted_rate"] = (
    results["borough"].map(adjusted_rates)
)

results["observed_pct"] = (
    results["observed_rate"] * 100
)

results["adjusted_pct"] = (
    results["adjusted_rate"] * 100
)

results["difference_pp"] = (
    results["adjusted_pct"] -
    results["observed_pct"]
)

print("\nOBSERVED VS ADJUSTED DELAY RATES")
print(
    results[
        [
            "borough",
            "requests",
            "observed_pct",
            "adjusted_pct",
            "difference_pp"
        ]
    ].round(2).to_string(index=False)
)

# Save results
output = Path("reports/tables")
output.mkdir(parents=True, exist_ok=True)

results.to_csv(
    output / "adjusted_borough_delays.csv",
    index=False
)

print("\nADJUSTED ANALYSIS COMPLETE!")
