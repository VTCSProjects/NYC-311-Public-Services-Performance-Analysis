
import pandas as pd

df = pd.read_csv(
    "data/processed/nyc_311_2025_clean.csv",
    low_memory=False
)

df["resolution_hours"] = pd.to_numeric(
    df["resolution_hours"], errors="coerce"
)

closed = df[
    (df["status"] == "Closed") &
    (df["resolution_hours"].notna())
].copy()

print("\nCLOSED REQUESTS WITH VALID DURATIONS:")
print(len(closed))

# 1. Investigate unusual resolution times
print("\nRESOLUTION TIME OUTLIERS")
print("Zero-hour closures:",
      (closed["resolution_hours"] == 0).sum())
print("Over 7 days:",
      (closed["resolution_hours"] > 168).sum())
print("Over 30 days:",
      (closed["resolution_hours"] > 720).sum())
print("Over 180 days:",
      (closed["resolution_hours"] > 4320).sum())

# 2. Examine the 10 longest durations
columns = [
    "unique_key",
    "problem_formerly_complaint_type",
    "agency",
    "borough",
    "created_date",
    "closed_date",
    "resolution_hours"
]

print("\nTEN LONGEST RESOLUTION TIMES")
print(
    closed.nlargest(10, "resolution_hours")[columns]
    .to_string(index=False)
)

# 3. Calculate complaint category statistics
category_stats = closed.groupby(
    "problem_formerly_complaint_type"
).agg(
    requests=("unique_key", "size"),
    median_hours=("resolution_hours", "median"),
    p90_hours=(
        "resolution_hours",
        lambda x: x.quantile(0.90)
    )
)

# Require at least 200 observations
reliable = category_stats[
    category_stats["requests"] >= 200
]

print("\n10 SLOWEST COMPLAINT CATEGORIES")
print(
    reliable.sort_values(
        "median_hours", ascending=False
    ).head(10).round(2)
)

print("\n10 FASTEST COMPLAINT CATEGORIES")
print(
    reliable.sort_values(
        "median_hours"
    ).head(10).round(2)
)

# 4. Look for status/date inconsistencies
print("\nDATA QUALITY FOLLOW-UP")
print("Closed without closure date:",
      df["closed_status_missing_date"].sum())
print("Non-closed with closure date:",
      (
          (df["status"] != "Closed") &
          df["closed_date"].notna()
      ).sum())

print("\nDEEP EDA COMPLETE")
