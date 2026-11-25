
import sqlite3
from pathlib import Path

import pandas as pd

# =====================================================
# STEP 1: LOAD CLEANED DATASET
# =====================================================

csv_path = Path("data/processed/nyc_311_2025_clean.csv")
db_path = Path("data/nyc_311.db")

df = pd.read_csv(
    csv_path,
    dtype={
        "unique_key": "string",
        "incident_zip": "string"
    },
    low_memory=False
)

# Parse mixed date formats
df["created_date"] = pd.to_datetime(
    df["created_date"],
    format="mixed",
    errors="coerce"
)

df["closed_date"] = pd.to_datetime(
    df["closed_date"],
    format="mixed",
    errors="coerce"
)

# Validate data
assert len(df) == 296434
assert df["unique_key"].notna().all()
assert not df["unique_key"].duplicated().any()
assert df["created_date"].notna().all()

# Create month column
df["created_month"] = (
    df["created_date"].dt.strftime("%Y-%m")
)

print("DATASET VALIDATION SUCCESSFUL")
print("Total records:", len(df))
print("Total columns:", len(df.columns))

# =====================================================
# STEP 2: CREATE SQLITE DATABASE
# =====================================================

db_path.parent.mkdir(parents=True, exist_ok=True)

conn = sqlite3.connect(db_path)

df.to_sql(
    "service_requests",
    conn,
    if_exists="replace",
    index=False
)

# =====================================================
# STEP 3: CREATE DATABASE INDEXES
# =====================================================

conn.executescript("""
CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_key
ON service_requests(unique_key);

CREATE INDEX IF NOT EXISTS idx_borough
ON service_requests(borough);

CREATE INDEX IF NOT EXISTS idx_agency
ON service_requests(agency);

CREATE INDEX IF NOT EXISTS idx_complaint
ON service_requests(problem_formerly_complaint_type);

CREATE INDEX IF NOT EXISTS idx_month
ON service_requests(created_month);

CREATE INDEX IF NOT EXISTS idx_resolution
ON service_requests(resolution_hours);
""")

conn.commit()

# Verify database
db_count = pd.read_sql_query(
    "SELECT COUNT(*) AS total FROM service_requests",
    conn
)["total"].iloc[0]

assert db_count == len(df)

print("\nSQLITE DATABASE CREATED SUCCESSFULLY")
print("Records loaded:", db_count)
print("Database location:", db_path)

# =====================================================
# SQL ANALYSIS 1: MONTHLY BOROUGH DISTRIBUTION
# =====================================================

query1 = """
SELECT
    created_month,
    borough,
    COUNT(*) AS requests
FROM service_requests
GROUP BY created_month, borough
ORDER BY created_month, borough;
"""

monthly = pd.read_sql_query(query1, conn)

print("\nMONTHLY REQUESTS BY BOROUGH")

print(
    monthly.pivot(
        index="created_month",
        columns="borough",
        values="requests"
    ).fillna(0).astype(int).to_string()
)

# =====================================================
# SQL ANALYSIS 2: TOP 3 AGENCIES PER BOROUGH
# =====================================================

query2 = """
WITH agency_counts AS (
    SELECT
        borough,
        agency,
        COUNT(*) AS requests
    FROM service_requests
    WHERE borough != 'Unspecified'
      AND borough IS NOT NULL
    GROUP BY borough, agency
),
ranked AS (
    SELECT
        borough,
        agency,
        requests,
        ROW_NUMBER() OVER (
            PARTITION BY borough
            ORDER BY requests DESC, agency
        ) AS agency_rank
    FROM agency_counts
)
SELECT borough, agency, requests
FROM ranked
WHERE agency_rank <= 3
ORDER BY borough, agency_rank;
"""

print("\nTOP 3 AGENCIES PER BOROUGH")

print(
    pd.read_sql_query(query2, conn).to_string(index=False)
)

# =====================================================
# SQL ANALYSIS 3: MANHATTAN VS BROOKLYN COMPLAINTS
# =====================================================

query3 = """
WITH complaint_counts AS (
    SELECT
        borough,
        problem_formerly_complaint_type AS complaint_type,
        COUNT(*) AS requests
    FROM service_requests
    WHERE borough IN ('MANHATTAN', 'BROOKLYN')
    GROUP BY borough, problem_formerly_complaint_type
),
ranked AS (
    SELECT
        borough,
        complaint_type,
        requests,
        ROW_NUMBER() OVER (
            PARTITION BY borough
            ORDER BY requests DESC, complaint_type
        ) AS complaint_rank
    FROM complaint_counts
)
SELECT borough, complaint_type, requests
FROM ranked
WHERE complaint_rank <= 5
ORDER BY borough, complaint_rank;
"""

print("\nTOP COMPLAINTS: MANHATTAN VS BROOKLYN")

print(
    pd.read_sql_query(query3, conn).to_string(index=False)
)

conn.close()

print("\nALL SQL ANALYSES COMPLETE!")
