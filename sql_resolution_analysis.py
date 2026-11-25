
import sqlite3
from pathlib import Path
import pandas as pd

# Connect to your existing database
conn = sqlite3.connect("data/nyc_311.db")

# Save analytical results
output_dir = Path("reports/tables")
output_dir.mkdir(parents=True, exist_ok=True)

# Only compare closed requests with valid durations
valid_cases = """
    LOWER(TRIM(status)) = 'closed'
    AND resolution_hours IS NOT NULL
    AND resolution_hours >= 0
"""

def run_analysis(title, query, filename):
    result = pd.read_sql_query(query, conn)
    print(f"\n{title}")
    print(result.to_string(index=False))
    result.to_csv(output_dir / filename, index=False)
    return result


# ANALYSIS 1: DELAY RATE BY BOROUGH

query1 = f"""
SELECT
    borough,
    COUNT(*) AS closed_requests,

    ROUND(AVG(resolution_hours), 2) AS avg_hours,

    ROUND(
        100.0 * SUM(
            CASE WHEN resolution_hours > 168
            THEN 1 ELSE 0 END
        ) / COUNT(*), 2
    ) AS pct_over_7_days,

    ROUND(
        100.0 * SUM(
            CASE WHEN resolution_hours > 720
            THEN 1 ELSE 0 END
        ) / COUNT(*), 2
    ) AS pct_over_30_days

FROM service_requests

WHERE {valid_cases}
AND borough IN (
    'BRONX', 'BROOKLYN', 'MANHATTAN',
    'QUEENS', 'STATEN ISLAND'
)

GROUP BY borough
ORDER BY pct_over_7_days DESC;
"""

run_analysis(
    "BOROUGH DELAY RATES",
    query1,
    "borough_delay_rates.csv"
)


# ANALYSIS 2: MEDIAN RESOLUTION BY BOROUGH
# Window functions find the middle observation(s)

query2 = f"""
WITH ranked AS (
    SELECT
        borough,
        resolution_hours,

        ROW_NUMBER() OVER (
            PARTITION BY borough
            ORDER BY resolution_hours
        ) AS rn,

        COUNT(*) OVER (
            PARTITION BY borough
        ) AS n

    FROM service_requests
    WHERE {valid_cases}
    AND borough IN (
        'BRONX', 'BROOKLYN', 'MANHATTAN',
        'QUEENS', 'STATEN ISLAND'
    )
)

SELECT
    borough,
    ROUND(AVG(resolution_hours), 2) AS median_hours
FROM ranked

WHERE rn IN ((n + 1) / 2, (n + 2) / 2)

GROUP BY borough
ORDER BY median_hours DESC;
"""

run_analysis(
    "MEDIAN RESOLUTION TIMES BY BOROUGH",
    query2,
    "borough_medians.csv"
)


# ANALYSIS 3: COMPARE SAME COMPLAINTS ACROSS BOROUGHS

query3 = f"""
SELECT
    borough,
    problem_formerly_complaint_type AS complaint,
    COUNT(*) AS closed_requests,

    ROUND(AVG(resolution_hours), 2) AS avg_hours,

    ROUND(
        100.0 * SUM(
            CASE WHEN resolution_hours > 168
            THEN 1 ELSE 0 END
        ) / COUNT(*), 2
    ) AS pct_over_7_days

FROM service_requests

WHERE {valid_cases}
AND problem_formerly_complaint_type IN (
    'Illegal Parking',
    'HEAT/HOT WATER',
    'Noise - Residential'
)
AND borough IN (
    'BRONX', 'BROOKLYN', 'MANHATTAN',
    'QUEENS', 'STATEN ISLAND'
)

GROUP BY borough, problem_formerly_complaint_type

HAVING COUNT(*) >= 100

ORDER BY complaint, pct_over_7_days DESC;
"""

run_analysis(
    "SAME COMPLAINTS ACROSS BOROUGHS",
    query3,
    "complaint_borough_comparison.csv"
)


# ANALYSIS 4: HEATING COMPLAINTS BY SAMPLED MONTH

query4 = """
SELECT
    created_month,
    COUNT(*) AS total_requests,

    SUM(
        CASE
            WHEN problem_formerly_complaint_type =
                 'HEAT/HOT WATER'
            THEN 1 ELSE 0
        END
    ) AS heating_requests,

    ROUND(
        100.0 * SUM(
            CASE
                WHEN problem_formerly_complaint_type =
                     'HEAT/HOT WATER'
                THEN 1 ELSE 0
            END
        ) / COUNT(*), 2
    ) AS heating_pct

FROM service_requests
GROUP BY created_month
ORDER BY created_month;
"""

run_analysis(
    "HEATING COMPLAINTS BY MONTH",
    query4,
    "heating_by_month.csv"
)

conn.close()

print("\nADVANCED SQL ANALYSIS COMPLETE!")
print("Results saved to:", output_dir)
