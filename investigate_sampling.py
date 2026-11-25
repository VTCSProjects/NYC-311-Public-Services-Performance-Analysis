
import sqlite3
import pandas as pd

conn = sqlite3.connect("data/nyc_311.db")

# ANALYSIS 1: How much of each month was sampled?
query1 = """
SELECT
    created_month,
    COUNT(*) AS total_records,
    COUNT(DISTINCT SUBSTR(created_date, 1, 10)) AS days_covered,
    MIN(created_date) AS earliest_record,
    MAX(created_date) AS latest_record,
    SUM(CASE WHEN borough = 'MANHATTAN'
        THEN 1 ELSE 0 END) AS manhattan_requests
FROM service_requests
GROUP BY created_month
ORDER BY created_month;
"""

monthly = pd.read_sql_query(query1, conn)

monthly["manhattan_pct"] = (
    monthly["manhattan_requests"] /
    monthly["total_records"] * 100
).round(2)

print("\nMONTHLY SAMPLING AUDIT")
print(monthly.to_string(index=False))


# ANALYSIS 2: Compare the same agency across boroughs.
query2 = """
SELECT
    borough,
    agency,
    resolution_hours
FROM service_requests
WHERE status = 'Closed'
  AND closed_date IS NOT NULL
  AND resolution_hours >= 0
  AND borough IN ('MANHATTAN', 'BROOKLYN');
"""

df = pd.read_sql_query(query2, conn)

agency_stats = df.groupby(
    ["agency", "borough"]
).agg(
    requests=("resolution_hours", "size"),
    median_hours=("resolution_hours", "median")
).reset_index()

comparison = agency_stats.pivot(
    index="agency",
    columns="borough",
    values=["requests", "median_hours"]
)

# Only compare agencies with at least 30
# observations in both boroughs.
comparison = comparison.dropna()

comparison = comparison[
    (comparison[("requests", "MANHATTAN")] >= 30) &
    (comparison[("requests", "BROOKLYN")] >= 30)
]

print("\nSAME AGENCY: MANHATTAN VS BROOKLYN")
print(comparison.round(2).to_string())

conn.close()
