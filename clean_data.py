
import pandas as pd
from pathlib import Path

# 1. Load downloaded data
df = pd.read_csv(
    "data/raw/nyc_311_2025_sample.csv",
    dtype={"unique_key": "string", "incident_zip": "string"}
)

print("DATASET SHAPE")
print(df.shape)

# 2. Check missing values
print("\nMISSING VALUES (%)")
missing = df.isnull().mean() * 100
print(missing.sort_values(ascending=False).round(2))

# 3. Check duplicate complaint IDs
duplicates = df["unique_key"].duplicated().sum()
print("\nDuplicate IDs:", duplicates)

# 4. Convert dates
df["created_date"] = pd.to_datetime(
    df["created_date"], errors="coerce"
)
df["closed_date"] = pd.to_datetime(
    df["closed_date"], errors="coerce"
)

# 5. Calculate resolution time
df["resolution_hours"] = (
    df["closed_date"] - df["created_date"]
).dt.total_seconds() / 3600

# Check negative durations
negative = (df["resolution_hours"] < 0).sum()
print("\nNegative durations:", negative)

# Treat invalid negative durations as missing
df.loc[
    df["resolution_hours"] < 0, "resolution_hours"
] = float("nan")

# 6. Review closure information
print("\nCLOSED DATE AVAILABLE")
print(df["closed_date"].notna().sum())

print("\nREQUEST STATUS")
print(df["status"].value_counts(dropna=False))

# 7. Analyze resolution times
print("\nRESOLUTION HOURS")
print(df["resolution_hours"].describe())

# 8. Count requests by borough
print("\nREQUESTS BY BOROUGH")
print(df["borough"].value_counts(dropna=False))

# 9. Save cleaned dataset
Path("data/processed").mkdir(parents=True, exist_ok=True)

df.to_csv(
    "data/processed/nyc_311_clean.csv",
    index=False
)

print("\nCleaned dataset saved!")
