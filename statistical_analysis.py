
import sqlite3
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from statsmodels.stats.proportion import proportion_confint

# Connect to database
conn = sqlite3.connect("data/nyc_311.db")

df = pd.read_sql_query("""
    SELECT
        borough,
        agency,
        created_month,
        problem_formerly_complaint_type AS complaint,
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

conn.close()

# Define seven-day delay
df["delayed"] = (df["resolution_hours"] > 168).astype(int)

print("TOTAL VALID CLOSED REQUESTS:", len(df))

# ==========================================
# TEST 1: DELAY RATES AND CONFIDENCE INTERVALS
# ==========================================

results = []

for borough, group in df.groupby("borough"):
    n = len(group)
    delays = group["delayed"].sum()

    lower, upper = proportion_confint(
        delays, n, alpha=0.05, method="wilson"
    )

    results.append({
        "borough": borough,
        "requests": n,
        "delayed": delays,
        "delay_pct": round(delays / n * 100, 2),
        "ci_lower": round(lower * 100, 2),
        "ci_upper": round(upper * 100, 2)
    })

summary = pd.DataFrame(results)
summary = summary.sort_values(
    "delay_pct", ascending=False
)

print("\nBOROUGH DELAY RATES WITH 95% CONFIDENCE INTERVALS")
print(summary.to_string(index=False))

# ==========================================
# TEST 2: CHI-SQUARE INDEPENDENCE TEST
# ==========================================

table = pd.crosstab(df["borough"], df["delayed"])

chi2, p, dof, expected = chi2_contingency(table)

n = table.to_numpy().sum()
cramers_v = np.sqrt(
    chi2 / (n * min(table.shape[0] - 1,
                    table.shape[1] - 1))
)

print("\nCHI-SQUARE TEST")
print("Chi-square statistic:", round(chi2, 3))
print("Degrees of freedom:", dof)
print("P-value:", p)
print("Cramer's V:", round(cramers_v, 4))

if p < 0.05:
    print("Result: Statistically significant association")
else:
    print("Result: No statistically significant association")

# ==========================================
# TEST 3: WITHIN-COMPLAINT COMPARISON
# ==========================================

print("\nTOP COMPLAINT CATEGORIES: DELAY COMPARISON")

top_categories = df["complaint"].value_counts().head(10).index

comparison = (
    df[df["complaint"].isin(top_categories)]
    .groupby(["complaint", "borough"])
    .agg(
        requests=("delayed", "size"),
        delay_pct=("delayed", "mean")
    )
    .reset_index()
)

comparison["delay_pct"] *= 100
comparison = comparison[comparison["requests"] >= 100]

print(comparison.round(2).to_string(index=False))

# Save results
output = Path("reports/tables")
output.mkdir(parents=True, exist_ok=True)

summary.to_csv(
    output / "statistical_delay_summary.csv",
    index=False
)

comparison.to_csv(
    output / "complaint_statistical_comparison.csv",
    index=False
)

print("\nSTATISTICAL ANALYSIS COMPLETE!")
