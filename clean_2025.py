
import pandas as pd
from pathlib import Path

folder = Path("data/processed")

# Load all four months from the combined dataset
input_file = folder / "nyc_311_2025_combined.csv"
data = pd.read_csv(input_file, low_memory=False)

# Standardize column names
data.columns = (
    data.columns.str.strip().str.lower()
    .str.replace(r"[^a-z0-9]+", "_", regex=True)
    .str.strip("_")
)

# Validate combined dataset
assert len(data) == 296434, "Unexpected record count"
assert data["unique_key"].notna().all(), "Missing IDs found"
assert not data["unique_key"].duplicated().any(), "Duplicate IDs found"

# Parse dates
created = pd.to_datetime(
    data["created_date"],
    format="mixed",
    errors="coerce"
)

closed = pd.to_datetime(
    data["closed_date"],
    format="mixed",
    errors="coerce"
)

assert created.notna().all(), "Invalid creation dates found"

# Identify data quality problems
data["missing_closure_date"] = closed.isna()

data["negative_resolution_time"] = closed < created

data["closed_status_missing_date"] = (
    data["status"].astype("string").str.strip().eq("Closed")
    & closed.isna()
)

# Calculate resolution time in hours
data["resolution_hours"] = (
    (closed - created).dt.total_seconds() / 3600
)

# Remove invalid durations without deleting records
data.loc[
    data["resolution_hours"] < 0,
    "resolution_hours"
] = float("nan")

# Create analysis features
data["month"] = created.dt.month
data["day_of_week"] = created.dt.day_name()
data["hour"] = created.dt.hour

# Validate month coverage
expected_months = {1, 4, 7, 10}
actual_months = set(data["month"].unique())

assert actual_months == expected_months, (
    f"Unexpected months: {actual_months}"
)

# Print data quality summary
print("\nDATA QUALITY SUMMARY")
print("Total records:", len(data))
print("Negative resolution times:",
      data["negative_resolution_time"].sum())
print("Missing closure dates:",
      data["missing_closure_date"].sum())
print("Closed status without closure date:",
      data["closed_status_missing_date"].sum())
print("Valid resolution times:",
      data["resolution_hours"].notna().sum())

print("\nRECORDS BY MONTH")
print(data["month"].value_counts().sort_index())

# Save anomalous records
anomalies = data[data["negative_resolution_time"]]
anomalies.to_csv(
    folder / "nyc_311_negative_times.csv",
    index=False
)

# Save cleaned dataset
output = folder / "nyc_311_2025_clean.csv"
data.to_csv(output, index=False)

print("\nSUCCESS!")
print("Clean dataset:", output)
print("Records preserved:", len(data))
print("Columns:", len(data.columns))
