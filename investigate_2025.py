
import pandas as pd

df = pd.read_csv(
    "data/processed/nyc_311_2025_clean.csv",
    low_memory=False
)

created = pd.to_datetime(
    df["created_date"], format="mixed", errors="coerce"
)
closed_date = pd.to_datetime(
    df["closed_date"], format="mixed", errors="coerce"
)

df["resolution_hours"] = pd.to_numeric(
    df["resolution_hours"], errors="coerce"
)

valid = df[
    df["status"].eq("Closed") &
    df["resolution_hours"].notna()
].copy()

category = "problem_formerly_complaint_type"

# 1. Investigate negative durations
negative = closed_date < created

print("\nNEGATIVE DURATION RECORDS BY STATUS")
print(df.loc[negative, "status"].value_counts().to_string())

# 2. Investigate zero-hour closures
zero = valid[valid["resolution_hours"] == 0]

print("\nZERO-HOUR CLOSURES BY AGENCY")
print(zero["agency"].value_counts().head(10).to_string())

print("\nZERO-HOUR CLOSURES BY COMPLAINT")
print(zero[category].value_counts().head(10).to_string())

# 3. Long-running requests
long = valid[valid["resolution_hours"] > 4320]

print("\nREQUESTS TAKING OVER 180 DAYS BY AGENCY")
print(long["agency"].value_counts().head(15).to_string())

print("\nREQUESTS TAKING OVER 180 DAYS BY COMPLAINT")
print(long[category].value_counts().head(15).to_string())

# 4. Compare the same complaint across boroughs
complaints = [
    "Illegal Parking",
    "Noise - Residential",
    "HEAT/HOT WATER"
]

for complaint in complaints:
    subset = valid[valid[category] == complaint]

    results = subset.groupby("borough").agg(
        requests=("unique_key", "count"),
        median_hours=("resolution_hours", "median")
    )

    print(f"\n{complaint.upper()} BY BOROUGH")
    print(results.round(2).to_string())

print("\nINVESTIGATION COMPLETE")
