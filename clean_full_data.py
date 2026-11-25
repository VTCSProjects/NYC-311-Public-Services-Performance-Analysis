
import pandas as pd
from pathlib import Path

# Load the larger dataset
df = pd.read_csv(
    "data/raw/nyc_311_jan_2025_week1.csv",
    dtype={
        "Unique Key": "string",
        "Incident Zip": "string"
    },
    low_memory=False
)

print("ORIGINAL DATASET")
print("Rows:", len(df))
print("Columns:", len(df.columns))

# Standardize important column names
df.rename(columns={
    "Unique Key": "unique_key",
    "Created Date": "created_date",
    "Closed Date": "closed_date",
    "Agency": "agency",
    "Agency Name": "agency_name",
    "Problem (formerly Complaint Type)": "complaint_type",
    "Problem Detail (formerly Descriptor)": "descriptor",
    "Borough": "borough",
    "Status": "status"
}, inplace=True)

# Parse dates with an explicit format
date_format = "%m/%d/%Y %I:%M:%S %p"

for col in ["created_date", "closed_date"]:
    df[col] = pd.to_datetime(
        df[col],
        format=date_format,
        errors="coerce"
    )

# Check for missing and duplicate values
print("\nDUPLICATE REQUEST IDS")
print(df["unique_key"].duplicated().sum())

print("\nMISSING VALUES (%)")
missing = df.isna().mean().mul(100)
print(missing.sort_values(ascending=False).head(15).round(2))

# Check date coverage
print("\nDATE COVERAGE")
print("Earliest:", df["created_date"].min())
print("Latest:", df["created_date"].max())

print("\nREQUESTS BY CREATION DATE")
print(df["created_date"].dt.date.value_counts().sort_index())

# Calculate closure duration
duration = (
    df["closed_date"] - df["created_date"]
).dt.total_seconds() / 3600

closed = (
    df["status"].astype("string")
    .str.strip().str.lower().eq("closed")
)

print("\nSTATUS / DATE INCONSISTENCIES")
print("Closed without date:", (closed & df["closed_date"].isna()).sum())
print("Non-closed with date:", (~closed & df["closed_date"].notna()).sum())
print("Negative durations:", duration.lt(0).sum())

# Use only valid closed requests for resolution analysis
valid = (
    closed &
    df["created_date"].notna() &
    df["closed_date"].notna() &
    duration.ge(0)
)

df["resolution_hours"] = duration.where(valid)

print("\nVALID CLOSURE DURATION STATISTICS")
print(df["resolution_hours"].describe())

print("\nTOP 10 COMPLAINT TYPES")
print(df["complaint_type"].value_counts().head(10))

print("\nREQUESTS BY BOROUGH")
print(df["borough"].value_counts(dropna=False))

# Save without changing the original CSV
Path("data/processed").mkdir(parents=True, exist_ok=True)

df.to_csv(
    "data/processed/nyc_311_week1_clean.csv",
    index=False
)

print("\nCLEANED DATA SAVED SUCCESSFULLY!")
