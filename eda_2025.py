
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load cleaned dataset
df = pd.read_csv(
    "data/processed/nyc_311_2025_clean.csv",
    low_memory=False
)

output = Path("reports/figures")
output.mkdir(parents=True, exist_ok=True)

print("TOTAL REQUESTS:", len(df))

# 1. Top complaint categories
complaints = (
    df["problem_formerly_complaint_type"]
    .value_counts()
    .head(10)
)

print("\nTOP 10 COMPLAINTS")
print(complaints)

plt.figure(figsize=(11, 6))
complaints.sort_values().plot(kind="barh")
plt.title("Top 10 NYC 311 Complaint Categories")
plt.xlabel("Number of Requests")
plt.tight_layout()
plt.savefig(output / "top_complaints.png", dpi=200)
plt.close()

# 2. Requests by borough
boroughs = df[
    df["borough"] != "Unspecified"
]["borough"].value_counts()

print("\nREQUESTS BY BOROUGH")
print(boroughs)

plt.figure(figsize=(10, 6))
boroughs.plot(kind="bar")
plt.title("NYC 311 Requests by Borough")
plt.ylabel("Requests")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(output / "borough_requests.png", dpi=200)
plt.close()

# 3. Monthly comparison
monthly = df.groupby("month").size()

print("\nREQUESTS BY MONTH")
print(monthly)

plt.figure(figsize=(8, 5))
monthly.plot(kind="bar")
plt.title("NYC 311 Requests by Sampled Week")
plt.xlabel("Month (2025)")
plt.ylabel("Requests")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(output / "monthly_requests.png", dpi=200)
plt.close()

# 4. Resolution time analysis
# Only use requests marked Closed with valid durations
closed = df[
    (df["status"] == "Closed") &
    (df["resolution_hours"].notna())
].copy()

closed["resolution_hours"] = pd.to_numeric(
    closed["resolution_hours"], errors="coerce"
)

print("\nRESOLUTION TIME STATISTICS (HOURS)")
print(closed["resolution_hours"].describe(
    percentiles=[0.5, 0.9, 0.99]
))

print("\nMEDIAN RESOLUTION HOURS BY BOROUGH")
print(
    closed.groupby("borough")["resolution_hours"]
    .median()
    .sort_values()
)

print("\nEDA COMPLETE!")
print("Charts saved to:", output)
