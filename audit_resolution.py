
import pandas as pd

df = pd.read_csv("data/processed/nyc_311_clean.csv")

df["created_date"] = pd.to_datetime(
    df["created_date"], errors="coerce"
)
df["closed_date"] = pd.to_datetime(
    df["closed_date"], errors="coerce"
)

# 1. Check closure date consistency with status
print("\nSTATUS VS CLOSURE DATE")
print(pd.crosstab(
    df["status"],
    df["closed_date"].notna(),
    rownames=["Status"],
    colnames=["Has closure date"]
))

# 2. Study resolution time percentiles
valid = df[
    df["closed_date"].notna() &
    df["resolution_hours"].notna() &
    (df["resolution_hours"] >= 0)
].copy()

print("\nRESOLUTION PERCENTILES")
print(
    valid["resolution_hours"].quantile(
        [0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    )
)

# 3. How many requests take over 30 days?
valid["over_30_days"] = valid["resolution_hours"] > 720

print("\nPERCENT OVER 30 DAYS")
print(round(valid["over_30_days"].mean() * 100, 2), "%")

# 4. Which complaint types take longest?
print("\nRESOLUTION TIME BY COMPLAINT TYPE")

complaints = valid.groupby("complaint_type").agg(
    requests=("unique_key", "count"),
    mean_hours=("resolution_hours", "mean"),
    median_hours=("resolution_hours", "median"),
    percent_over_30_days=("over_30_days", "mean")
)

complaints = complaints[complaints["requests"] >= 30]

complaints["percent_over_30_days"] *= 100

print(
    complaints.sort_values(
        "median_hours", ascending=False
    ).head(15).round(2)
)

# 5. Identify the longest recorded cases
print("\n10 LONGEST RESOLUTION TIMES")

print(
    valid.nlargest(10, "resolution_hours")[
        [
            "complaint_type",
            "agency",
            "created_date",
            "closed_date",
            "resolution_hours"
        ]
    ].to_string(index=False)
)

# 6. Resolution time by borough
print("\nMEDIAN RESOLUTION BY BOROUGH")

print(
    valid.groupby("borough")["resolution_hours"]
    .median()
    .sort_values(ascending=False)
    .round(2)
)
