
import pandas as pd
from pathlib import Path

df = pd.read_csv(
    "data/processed/nyc_311_week1_clean.csv",
    low_memory=False
)

# Normalize addresses.
addresses = (
    df["Incident Address"]
    .astype("string")
    .str.upper()
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
)

# Find the leading Bronx residential noise hotspot.
bronx_noise = (
    (df["borough"] == "BRONX") &
    (df["complaint_type"] == "Noise - Residential")
)

top_address = addresses[bronx_noise].value_counts().index[0]

# Flag these records, but keep them in the dataset.
df["hotspot_flag"] = (
    bronx_noise & addresses.eq(top_address)
).fillna(False)

print("HOTSPOT RECORDS:", df["hotspot_flag"].sum())

# Compare the original and adjusted datasets.
scenarios = {
    "All requests": df,
    "Excluding hotspot": df.loc[~df["hotspot_flag"]]
}

for name, data in scenarios.items():
    print(f"\n{name.upper()}")
    print("Total requests:", len(data))

    bronx_share = (
        data["borough"].eq("BRONX").mean() * 100
    )
    print("Bronx share:", round(bronx_share, 2), "%")

    noise = data[
        data["complaint_type"] == "Noise - Residential"
    ]

    bronx_noise_share = (
        noise["borough"].eq("BRONX").mean() * 100
    )
    print(
        "Bronx share of residential noise:",
        round(bronx_noise_share, 2), "%"
    )

    print(
        "Median closure hours:",
        round(data["resolution_hours"].median(), 2)
    )

# Examine whether closure outcomes differ.
print("\nHOTSPOT VS OTHER BRONX NOISE")

for name, subset in {
    "Hotspot": df[df["hotspot_flag"]],
    "Other Bronx noise": df[
        bronx_noise & ~df["hotspot_flag"]
    ]
}.items():
    print(f"\n{name}")
    print("Requests:", len(subset))
    print("Median closure hours:",
          round(subset["resolution_hours"].median(), 2))
    print("Statuses:")
    print(subset["status"].value_counts())

# Save locally for future SQL and Power BI analysis.
Path("data/processed").mkdir(parents=True, exist_ok=True)

output = "data/processed/nyc_311_week1_flagged.csv"
df.to_csv(output, index=False)

print("\nSaved:", output)
