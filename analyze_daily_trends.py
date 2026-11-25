
import pandas as pd
from pathlib import Path

df = pd.read_csv(
    "data/processed/nyc_311_week1_flagged.csv",
    parse_dates=["created_date"],
    low_memory=False
)

df["hotspot_flag"] = (
    df["hotspot_flag"].astype("string")
    .str.lower().eq("true").fillna(False)
)

df["day"] = df["created_date"].dt.date

# Calculate overall daily requests.
daily = df.groupby("day").agg(
    total_requests=("unique_key", "count"),
    hotspot_requests=("hotspot_flag", "sum")
)

daily["without_hotspot"] = (
    daily["total_requests"] - daily["hotspot_requests"]
)

daily["hotspot_pct"] = (
    daily["hotspot_requests"]
    / daily["total_requests"] * 100
).round(2)

# Calculate daily Bronx noise requests.
bronx_noise = df[
    (df["borough"] == "BRONX") &
    (df["complaint_type"] == "Noise - Residential")
]

noise_counts = bronx_noise.groupby("day").size()

daily["bronx_noise_requests"] = (
    noise_counts.reindex(daily.index, fill_value=0)
)

daily["bronx_noise_without_hotspot"] = (
    daily["bronx_noise_requests"]
    - daily["hotspot_requests"]
)

# Identify partial boundary days.
first_day = df["created_date"].min().date()
last_day = df["created_date"].max().date()

daily["partial_boundary_day"] = (
    pd.Series(daily.index.isin([first_day, last_day]),
              index=daily.index)
)

print("\nDAILY NYC 311 TRENDS")
print(daily.to_string())

# Save aggregated results for visualization.
Path("data/processed").mkdir(exist_ok=True)

daily.to_csv(
    "data/processed/nyc_311_daily_trends.csv"
)

print("\nDaily trends saved successfully!")
