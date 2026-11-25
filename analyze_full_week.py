
import sqlite3
from pathlib import Path

import pandas as pd

# Load our cleaned dataset
df = pd.read_csv(
    "data/processed/nyc_311_week1_clean.csv",
    low_memory=False
)

df["resolution_hours"] = pd.to_numeric(
    df["resolution_hours"], errors="coerce"
)

# Create a separate database for the larger dataset
Path("data").mkdir(exist_ok=True)

conn = sqlite3.connect("data/nyc_311_week1.db")

df.to_sql(
    "service_requests",
    conn,
    if_exists="replace",
    index=False,
    chunksize=5000
)

conn.executescript("""
CREATE INDEX idx_week_borough
ON service_requests(borough);

CREATE INDEX idx_week_complaint
ON service_requests(complaint_type);

CREATE INDEX idx_week_agency
ON service_requests(agency);
""")

print("DATABASE CREATED")
print("Records:", len(df))

# QUERY 1: Borough complaint volumes
query1 = """
SELECT
    borough,
    COUNT(*) AS total_requests,
    ROUND(
        100.0 * COUNT(*) /
        (SELECT COUNT(*) FROM service_requests),
        2
    ) AS percent_of_total
FROM service_requests
GROUP BY borough
ORDER BY total_requests DESC;
"""

print("\n1. REQUESTS BY BOROUGH")
print(pd.read_sql_query(query1, conn).to_string(index=False))


# QUERY 2: Most common complaints in each borough
query2 = """
WITH complaint_counts AS (
    SELECT
        borough,
        complaint_type,
        COUNT(*) AS requests
    FROM service_requests
    GROUP BY borough, complaint_type
),
ranked AS (
    SELECT *,
        ROW_NUMBER() OVER (
            PARTITION BY borough
            ORDER BY requests DESC, complaint_type
        ) AS complaint_rank
    FROM complaint_counts
)
SELECT borough, complaint_type, requests
FROM ranked
WHERE complaint_rank <= 3
ORDER BY borough, complaint_rank;
"""

print("\n2. TOP 3 COMPLAINTS PER BOROUGH")
print(pd.read_sql_query(query2, conn).to_string(index=False))


# QUERY 3: Fairer comparison of resolution times
# Compare the same complaint type AND agency.
# Calculate median using SQL window functions.
query3 = """
WITH ranked AS (
    SELECT
        borough,
        agency,
        complaint_type,
        resolution_hours,
        ROW_NUMBER() OVER (
            PARTITION BY borough, agency, complaint_type
            ORDER BY resolution_hours
        ) AS rn,
        COUNT(*) OVER (
            PARTITION BY borough, agency, complaint_type
        ) AS n
    FROM service_requests
    WHERE status = 'Closed'
      AND resolution_hours IS NOT NULL
      AND resolution_hours >= 0
      AND borough IN (
          'MANHATTAN', 'BROOKLYN',
          'BRONX', 'QUEENS', 'STATEN ISLAND'
      )
      AND complaint_type IN (
          'Noise - Residential',
          'HEAT/HOT WATER',
          'Illegal Parking',
          'UNSANITARY CONDITION'
      )
),
medians AS (
    SELECT
        borough,
        agency,
        complaint_type,
        MAX(n) AS requests,
        ROUND(AVG(resolution_hours), 2) AS median_hours
    FROM ranked
    WHERE rn IN ((n + 1) / 2, (n + 2) / 2)
    GROUP BY borough, agency, complaint_type
)
SELECT *
FROM medians
WHERE requests >= 30
ORDER BY complaint_type, agency, borough;
"""

print("\n3. RESOLUTION TIME BY AGENCY, COMPLAINT AND BOROUGH")
print(pd.read_sql_query(query3, conn).to_string(index=False))

conn.close()

print("\nANALYSIS COMPLETE!")
